#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tfidf_router.py — 语义路由桥接层（TF-IDF + BGE-zh 双引擎）
==========================================================
嵌入到 intent_classifier 降级链中:
  关键词 L0/L1/L2 匹配 → 未命中 → TF-IDF/BGE-zh Top-5 召回 → 返回候选 skill

用法（独立运行）:
  python tfidf_router.py "检查路由有没有毛病"          # 默认 BGE-zh
  python tfidf_router.py --tfidf "检查路由有没有毛病"   # TF-IDF 引擎
  python tfidf_router.py --bge "检查路由有没有毛病"     # BGE-zh 引擎
  → 返回 Top-5 skill 名 + 相似度分数

集成方式:
  在 intent_classifier_canonical.md 降级链 Step 4 后增加:
  "若 L0-L2 关键词均未命中 → 调用 tfidf_router.py <query> → 取 Top-3 注入 L3 LLM 精选"
"""
import argparse, contextlib, json, os, sys, pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

_SKILLS_DIR = None
_INDEX_DIR = "output"


def configure(skills_dir=None, index_dir=None):
    """设置技能库目录与 TF-IDF 索引目录（CLI / 集成调用前调用）"""
    global _SKILLS_DIR, _INDEX_DIR
    if skills_dir:
        _SKILLS_DIR = skills_dir
    if index_dir:
        _INDEX_DIR = index_dir

# 单例缓存
_vec = None
_mat = None
_idx = None

def _ensure_loaded():
    global _vec, _mat, _idx
    if _vec is None:
        mpath = os.path.join(_INDEX_DIR, "tfidf_matrix.npz")
        vpath = os.path.join(_INDEX_DIR, "tfidf_vectorizer.pkl")
        ipath = os.path.join(_INDEX_DIR, "skill_ids.json")
        if not all(os.path.exists(p) for p in [mpath, vpath, ipath]):
            raise FileNotFoundError("TF-IDF 索引未构建! 先运行 build_skill_vectors.py")
        _mat = sparse.load_npz(mpath)
        with open(vpath, "rb") as f:
            _vec = pickle.load(f)
        _idx = json.load(open(ipath, encoding="utf-8"))

def route(query, top_k=5, threshold=0.08):
    """
    TF-IDF 语义路由入口
    
    参数:
      query: 用户输入文本
      top_k: 返回前 K 个候选
      threshold: 最低相似度阈值（0.08≈宽松, 0.15≈保守）
    
    返回:
      [{"name": "skill-name", "score": 0.xx, "domain": "xxx"}, ...]
    """
    _ensure_loaded()
    q_vec = _vec.transform([query])
    sims = cosine_similarity(q_vec, _mat)[0]
    top = np.argsort(sims)[::-1][:top_k * 2]  # 多取些用于去重
    
    results = []
    seen = set()
    for i in top:
        if sims[i] < threshold:
            continue
        name = _idx[i]["name"]
        if name in seen:
            continue
        seen.add(name)
        results.append({
            "name": name,
            "score": round(float(sims[i]), 4),
            "domain": _idx[i]["domain"]
        })
        if len(results) >= top_k:
            break
    return results

def hybrid_route(query, keyword_matches=None, top_k=5):
    """
    混合路由: 关键词优先 + TF-IDF 补充
    
    参数:
      query: 用户输入
      keyword_matches: 关键词匹配结果 [skill_name, ...]（可选）
      top_k: 最多返回 K 个
    
    返回:
      合并去重后的候选列表
    """
    tfidf = route(query, top_k=top_k)
    kw = keyword_matches or []
    
    result = []
    seen = set()
    
    # 关键词结果优先
    for name in kw:
        if name not in seen:
            result.append({"name": name, "score": 1.0, "source": "keyword"})
            seen.add(name)
    
    # TF-IDF 补充（去重，降权为 0.5 标记来源）
    for r in tfidf:
        if r["name"] not in seen:
            r["source"] = "tfidf"
            r["score"] = r["score"] * 0.5  # 标记为非精确匹配
            result.append(r)
            seen.add(r["name"])
    
    return result[:top_k]

# ====== BGE-zh Full-Body Router ======
_bge_emb = None
_bge_skills = None
_bge_model = None

def _ensure_bge_loaded():
    """加载 Full-Body BGE-zh 索引（懒加载，首次调用时初始化）"""
    global _bge_emb, _bge_skills, _bge_model
    if _bge_emb is None:
        eval_dir = os.path.dirname(os.path.abspath(__file__))
        emb_path = os.path.join(eval_dir, 'bge_fullbody_embeddings.npy')
        skills_path = os.path.join(eval_dir, 'bge_fullbody_skills.json')
        
        if not os.path.exists(emb_path):
            raise FileNotFoundError(
                f"BGE full-body index not found: {emb_path}\n"
                "Run: python eval/save_bge_fullbody.py first"
            )
        
        import os as _os
        _os.environ['HF_HUB_OFFLINE'] = '1'
        _os.environ['TRANSFORMERS_OFFLINE'] = '1'
        
        from sentence_transformers import SentenceTransformer
        cache_dir = _os.path.expanduser(r'~/.cache/huggingface/hub/models--BAAI--bge-small-zh-v1.5')
        snap = sorted(_os.listdir(_os.path.join(cache_dir, 'snapshots')))[-1]
        model_path = _os.path.join(cache_dir, 'snapshots', snap)
        
        _bge_model = SentenceTransformer(model_path, device='cpu')
        _bge_emb = np.load(emb_path)
        with open(skills_path, 'r', encoding='utf-8') as f:
            _bge_skills = json.load(f)

def bge_route(query, top_k=5, domain=None):
    """
    BGE-zh Full-Body 语义路由 (推荐)
    
    参数:
      query: 用户输入
      top_k: 返回前K个
      domain: 限定领域（None=全局搜索）
    
    返回: [{"name","score","domain"},...]
    """
    _ensure_bge_loaded()
    
    q_emb = _bge_model.encode([query])
    sims = cosine_similarity(q_emb, _bge_emb)[0]
    
    # Filter by domain if specified
    if domain:
        candidates = [(i, sims[i]) for i, s in enumerate(_bge_skills) if s['domain'] == domain]
    else:
        candidates = [(i, sims[i]) for i in range(len(sims))]
    
    candidates.sort(key=lambda x: -x[1])
    
    results = []
    seen = set()
    for i, score in candidates[:top_k * 2]:
        name = _bge_skills[i]['name']
        if name in seen:
            continue
        seen.add(name)
        results.append({
            'name': name,
            'score': round(float(score), 4),
            'domain': _bge_skills[i]['domain'],
            'source': 'bge-fullbody'
        })
        if len(results) >= top_k:
            break
    
    return results

def smart_route(query, top_k=5):
    """
    智能路由: BGE-zh (优先) + TF-IDF 回退
    """
    try:
        return bge_route(query, top_k=top_k)
    except (FileNotFoundError, ImportError):
        return route(query, top_k=top_k)


# ====== 三级路由管道 (Three-Stage Pipeline) ======
# Bilibili方法论: L0规则→L1语义→L2 LLM决策
# Stage 1: 域分类 (classify_domain)
# Stage 2: BGE-zh 语义召回 (bge_route)
# Stage 3: LLM 精选 (build_llm_prompt → external LLM)

def build_skill_profile(name, max_desc=300, skills_dir=None):
    """
    为 LLM 决策构建 skill 结构化描述
    返回: {name, domain, desc, triggers, not_for, url}
    """
    skills_dir = skills_dir or _SKILLS_DIR or os.path.join("examples", "skills")
    sc = skills_dir
    
    profile = {
        'name': name,
        'domain': 'unknown',
        'desc': '',
        'triggers': [],
        'not_for': [],
    }
    
    # 从 skill_content JSON 取元数据
    for df in _os.listdir(sc):
        if not df.endswith('.json') or df == 'skill_ids.json':
            continue
        with open(_os.path.join(sc, df), 'r', encoding='utf-8') as f:
            data = json.load(f)
            for s in data.get('skills', []):
                if s.get('name') == name:
                    profile['domain'] = df.replace('.json', '')
                    profile['triggers'] = s.get('triggers', [])
                    break
        if profile['domain'] != 'unknown':
            break
    
    # 从 SKILL.md 取描述
    md_path = _os.path.join(skills_dir, name, 'SKILL.md')
    if _os.path.exists(md_path):
        with open(md_path, 'r', encoding='utf-8') as f:
            body = f.read()
        
        # 跳过 YAML frontmatter
        if body.startswith('---'):
            end = body.find('---', 3)
            if end > 0:
                body = body[end + 3:].strip()
        
        # 提取描述（跳过标题行）
        lines = body.split('\n')
        desc_parts = []
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
            desc_parts.append(stripped)
            if len(' '.join(desc_parts)) > max_desc:
                break
        profile['desc'] = ' '.join(desc_parts)
        
        # 提取别用我
        for line in body.split('\n'):
            if '【别用我】' in line:
                profile['not_for'].append(line.strip())
    
    return profile


def three_stage_route(query, top_k=5, domain=None, skills_dir=None):
    """
    三级路由管道（完整版）
    
    Stage 1: L0 域分类 (classify_domain)
    Stage 2: BGE-zh 语义召回 Top-K
    Stage 3: 构建 LLM 决策提示词
    
    返回: {
        'stage1_domain': str,
        'stage2_candidates': [...],
        'stage3_llm_prompt': str,  # 可直接喂给LLM的决策提示词
        'selected': str,           # BGE Top-1（LLM未介入时的默认选择）
    }
    """
    # Import domain classifier (avoid circular)
    try:
        from domain_classifier import classify_domain
    except ImportError:
        classify_domain = lambda q: None
    
    # Stage 1: 域分类
    detected_domain = domain or classify_domain(query)
    
    # Stage 2: BGE-zh 语义召回
    candidates = bge_route(query, top_k=top_k, domain=detected_domain)
    if not candidates:
        candidates = bge_route(query, top_k=top_k)  # fallback: no domain filter
    
    # Stage 3: 构建 LLM 提示词
    profiles = []
    for c in candidates[:top_k]:
        p = build_skill_profile(c['name'], skills_dir=skills_dir)
        profiles.append((c, p))
    
    # 构建 LLM 决策提示词
    candidate_lines = []
    for i, (c, p) in enumerate(profiles, 1):
        triggers_str = ', '.join(p['triggers'][:5]) if p['triggers'] else '(无)'
        not_for_str = '; '.join(p['not_for'][:2]) if p['not_for'] else '(无)'
        desc_str = p['desc'][:200] if p['desc'] else '(无描述)'
        
        candidate_lines.append(
            f"{i}. **{p['name']}** (领域:{p['domain']}, 相似度:{c['score']:.3f})\n"
            f"   描述: {desc_str}\n"
            f"   触发词: {triggers_str}\n"
            f"   不适用: {not_for_str}"
        )
    
    system_prompt = """你是 Skill 路由决策器。根据用户查询，从候选 Skill 中选择最匹配的一个。

决策规则（按优先级）:
1. 触发词优先 — 查询中包含某skill的触发词，权重最高
2. 意图理解 — "报错怎么修"→debugging, "项目交接"→handoff, "咋装"→install
3. 注意不适用场景 — 候选标注了"NOT USE WHEN X"且查询匹配X → 排除
4. 口语化理解 — "拆小"=拆分, "上线"=部署, "画个图"=图表
5. 输出格式: **只输出 skill 名称，不要解释。**"""

    user_prompt = f"""用户查询: "{query}"

候选 Skill (按语义相似度排序):
{chr(10).join(candidate_lines)}

请选择最匹配用户意图的 Skill。只输出名称。"""

    return {
        'stage1_domain': detected_domain,
        'stage2_candidates': candidates,
        'stage3_llm_prompt': f"[SYSTEM]\n{system_prompt}\n\n[USER]\n{user_prompt}",
        'selected': candidates[0]['name'] if candidates else None,
    }


# ---- CLI ----
def _run_cli(args, query):
    if args.three_stage:
        result = three_stage_route(query, skills_dir=args.skills_dir)
        print(f"查询: {query}")
        print(f"Stage 1 — 域分类: {result['stage1_domain']}")
        print(f"Stage 2 — BGE 召回 (Top-5):")
        for i, c in enumerate(result['stage2_candidates'], 1):
            print(f"  {i}. {c['name']:<40} {c['score']:.4f}  ({c['domain']})")
        print(f"Stage 2 — 默认选择: {result['selected']}")
        print(f"\n{'='*70}")
        print("Stage 3 — LLM 决策提示词 (复制以下内容发给 LLM):")
        print(f"{'='*70}")
        print(result['stage3_llm_prompt'])
    elif not args.tfidf:
        try:
            results = bge_route(query)
        except Exception as e:
            print(f"BGE Error: {e}")
            print("Falling back to TF-IDF...")
            results = route(query)
        print(f"查询: {query}  (引擎: bge)")
        print(f"{'排名':<6} {'Skill':<40} {'相似度':>6} {'领域':<15}")
        print("-" * 72)
        for i, r in enumerate(results, 1):
            print(f"{i:<6} {r['name']:<40} {r['score']:>6.4f} {r['domain']:<15}")
        if not results:
            print("  (无匹配)")
    else:
        results = route(query)
        print(f"查询: {query}  (引擎: tfidf)")
        print(f"{'排名':<6} {'Skill':<40} {'相似度':>6} {'领域':<15}")
        print("-" * 72)
        for i, r in enumerate(results, 1):
            print(f"{i:<6} {r['name']:<40} {r['score']:>6.4f} {r['domain']:<15}")
        if not results:
            print("  (无匹配)")


if __name__ == "__main__":
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = argparse.ArgumentParser(description="语义路由 CLI（TF-IDF / BGE / 三级路由）")
    p.add_argument("query", help="用户查询文本")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--tfidf", action="store_true", help="TF-IDF 引擎")
    g.add_argument("--bge", action="store_true", help="BGE-zh Full-Body 引擎（默认）")
    g.add_argument("--three-stage", "--llm", dest="three_stage", action="store_true",
                   help="三级路由: 域分类+BGE召回+LLM决策提示词")
    p.add_argument("--skills-dir", default=os.path.join("examples", "skills"),
                   help="技能库目录（含 *.json 与 SKILL.md 目录）")
    p.add_argument("--index-dir", default="output",
                   help="TF-IDF 索引目录（build_skill_vectors.py 的输出）")
    p.add_argument("--output", default="-",
                   help="输出文件路径（默认 - 输出到 stdout）")
    args = p.parse_args()
    configure(skills_dir=args.skills_dir, index_dir=args.index_dir)
    out = sys.stdout if args.output == "-" else open(args.output, "w", encoding="utf-8")
    try:
        with contextlib.redirect_stdout(out):
            _run_cli(args, args.query)
    finally:
        if out is not sys.stdout:
            out.close()
