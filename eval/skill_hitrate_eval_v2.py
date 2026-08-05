#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
skill_hitrate_eval_v2.py — 语义路由评估（TF-IDF 轻量版）
========================================================
对比: 关键词匹配 vs TF-IDF 语义 vs 混合路由
零额外依赖（numpy + sklearn + scipy，系统 Python 自带）
"""
import argparse, contextlib, json, os, sys, pickle, glob as _glob
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

# ---- 加载 ----

def load_skills(skills_dir):
    skills = []
    for fpath in sorted(_glob.glob(os.path.join(skills_dir, "*.json"))):
        bn = os.path.basename(fpath)
        if bn in ("embeddings.npy", "skill_ids.json", "tfidf_matrix.npz", "tfidf_vectorizer.pkl"):
            continue
        data = json.load(open(fpath, encoding="utf-8"))
        for s in data.get("skills", []):
            skills.append({
                "name": s["name"],
                "domain": os.path.splitext(bn)[0],
                "triggers": s.get("triggers", []),
                "desc": s.get("desc", "")[:500]
            })
    return skills

def load_tfidf(index_dir):
    """加载 TF-IDF 向量索引"""
    matrix = sparse.load_npz(os.path.join(index_dir, "tfidf_matrix.npz"))
    with open(os.path.join(index_dir, "tfidf_vectorizer.pkl"), "rb") as f:
        vectorizer = pickle.load(f)
    skill_index = json.load(open(os.path.join(index_dir, "skill_ids.json"), encoding="utf-8"))
    return vectorizer, matrix, skill_index

# ---- 匹配方法 ----

def keyword_match(query, skills):
    """关键词子串匹配"""
    q_lower = query.lower()
    scores = []
    for s in skills:
        score = 0
        for tg in s["triggers"]:
            tg_lower = tg.lower()
            if tg_lower in q_lower or (len(tg_lower) >= 2 and q_lower in tg_lower):
                score += 1
        if score > 0:
            scores.append((s["name"], score, s["domain"]))
    scores.sort(key=lambda x: -x[1])
    return [s[0] for s in scores]

def tfidf_match(query, vectorizer, matrix, skill_index, top_k=10, threshold=0.05):
    """TF-IDF 余弦相似度 Top-K 召回"""
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, matrix)[0]
    top_indices = np.argsort(sims)[::-1][:top_k]
    results = []
    for idx in top_indices:
        if sims[idx] > threshold:
            results.append((skill_index[idx]["name"], float(sims[idx])))
    return [r[0] for r in results]

def hybrid_match(query, skills, vectorizer, matrix, skill_index):
    """混合路由: keyword 优先 + tfidf 补充"""
    kw = keyword_match(query, skills)
    tf = tfidf_match(query, vectorizer, matrix, skill_index)
    seen = set()
    result = []
    for name in kw:
        if name not in seen:
            result.append(name); seen.add(name)
    for name in tf:
        if name not in seen:
            result.append(name); seen.add(name)
    return result

# ---- 评估 ----

def eval_method(queries_all, match_fn, **kwargs):
    results = []
    for tier_data in queries_all:
        for q in tier_data["queries"]:
            query = q["query"]
            expected = q.get("expected_skill")
            if isinstance(expected, str):
                expected_set = {expected}
            elif isinstance(expected, list):
                expected_set = set(expected)
            else:
                expected_set = set()
            
            matched = match_fn(query, **kwargs) if kwargs else match_fn(query)
            
            top1 = matched[0] if matched else None
            top1_correct = top1 in expected_set if expected_set else (top1 is None)
            top3_correct = any(n in expected_set for n in matched[:3]) if expected_set else True
            any_hit = bool(matched)
            
            results.append({
                "query": query, "expected": list(expected_set),
                "matched": matched[:5], "top1": top1,
                "top1_correct": top1_correct, "top3_correct": top3_correct,
                "any_hit": any_hit
            })
    
    exp_r = [r for r in results if r["expected"]]
    n_exp = len(exp_r)
    tp = sum(1 for r in exp_r if r["top1_correct"])
    fp = sum(1 for r in exp_r if r["top1"] and not r["top1_correct"])
    fn = sum(1 for r in exp_r if not r["any_hit"])
    p = tp/(tp+fp)*100 if (tp+fp) else 0
    r_ = tp/(tp+fn)*100 if (tp+fn) else 0
    f1 = 2*p*r_/(p+r_) if (p+r_) else 0
    
    neg_r = [r for r in results if not r["expected"]]
    fp_neg = sum(1 for r in neg_r if r["any_hit"])
    
    return {
        "top1_acc": round(sum(1 for r in exp_r if r["top1_correct"])/n_exp*100,1) if n_exp else 0,
        "top3_acc": round(sum(1 for r in exp_r if r["top3_correct"])/n_exp*100,1) if n_exp else 0,
        "hit_rate": round(sum(1 for r in exp_r if r["any_hit"])/n_exp*100,1) if n_exp else 0,
        "precision": round(p,1), "recall": round(r_,1), "f1": round(f1,1),
        "fp_rate": round(fp_neg/len(neg_r)*100,1) if neg_r else 0,
        "results": results, "tp": tp, "fp": fp, "fn": fn
    }

def print_failures(results, label):
    failures = [r for r in results if r["expected"] and not r["top1_correct"]]
    if failures:
        print(f"\n  [{label}] Top-1 未命中 ({len(failures)}):")
        for r in failures[:10]:  # 最多显示10条
            exp = " | ".join(r["expected"])
            got = r["top1"] or "(无匹配)"
            print(f"    ❌ '{r['query']}' → 期望[{exp}] 命中[{got}]")
        if len(failures) > 10:
            print(f"    ... 还有 {len(failures)-10} 条")

def run(skills_dir, index_dir, test_queries):
    skills = load_skills(skills_dir)
    data = json.load(open(test_queries, encoding="utf-8"))
    
    # ---- 方法1: 关键词 ----
    print("=" * 65)
    print("方法1: 纯关键词匹配")
    r1 = eval_method(data, lambda q: keyword_match(q, skills))
    
    # ---- 方法2: TF-IDF ----
    print("\n加载 TF-IDF 索引...")
    vectorizer, matrix, skill_index = load_tfidf(index_dir)
    
    print("\n" + "=" * 65)
    print("方法2: TF-IDF 语义匹配")
    r2 = eval_method(data, 
                     lambda q: tfidf_match(q, vectorizer, matrix, skill_index))
    
    print("\n" + "=" * 65)
    print("方法3: 混合路由 (keyword + TF-IDF)")
    r3 = eval_method(data,
                     lambda q: hybrid_match(q, skills, vectorizer, matrix, skill_index))
    
    # ---- 逐条对比 ----
    print(f"\n{'='*65}")
    print("📋 逐条对比 (仅显示有差异的)")
    print(f"{'='*65}")
    for tier_data in data:
        tier = tier_data["tier"]
        print(f"\n--- {tier.upper()} ---")
        for q in tier_data["queries"]:
            query = q["query"]
            kw = keyword_match(query, skills)[:3]
            tf = tfidf_match(query, vectorizer, matrix, skill_index)[:3]
            hy = hybrid_match(query, skills, vectorizer, matrix, skill_index)[:3]
            if kw != tf:
                exp = q.get("expected_skill", "")
                exp_str = exp if isinstance(exp, str) else " | ".join(exp) if exp else "(无)"
                print(f"  '{query}' → 期望[{exp_str}]")
                print(f"    KW: {kw or '(无)'}")
                print(f"    TF: {tf or '(无)'}")
                print(f"    HY: {hy or '(无)'}")
    
    # ---- 对比表 ----
    print(f"\n{'='*65}")
    print("📊 三种方法对比")
    print(f"{'='*65}")
    print(f"{'方法':<20} {'Top-1':>7} {'Top-3':>7} {'命中率':>7} {'P':>7} {'R':>7} {'F1':>7} {'误触发':>7}")
    print(f"{'-'*20} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    for r, label in [(r1,"关键词"), (r2,"TF-IDF"), (r3,"混合路由")]:
        print(f"{label:<20} {r['top1_acc']:>6.1f}% {r['top3_acc']:>6.1f}% {r['hit_rate']:>6.1f}% "
              f"{r['precision']:>6.1f}% {r['recall']:>6.1f}% {r['f1']:>6.1f}% {r['fp_rate']:>6.1f}%")
    
    # 提升幅度
    print(f"\n📈 提升幅度:")
    for metric, name in [("top1_acc","Top-1"), ("top3_acc","Top-3"), ("f1","F1")]:
        kw_v = r1[metric]
        tf_v = r2[metric]
        hy_v = r3[metric]
        print(f"  {name}:  {kw_v}% (KW) → {tf_v}% (TF-IDF, +{tf_v-kw_v:.1f}pp) → {hy_v}% (混合, +{hy_v-kw_v:.1f}pp)")
    
    # 失败详情
    print_failures(r1["results"], "关键词")
    print_failures(r2["results"], "TF-IDF")


def main(argv=None):
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = argparse.ArgumentParser(description="语义路由评估：关键词 vs TF-IDF vs 混合")
    p.add_argument("--skills-dir", default=os.path.join("examples", "skills"),
                   help="技能库目录: 内含 *.json")
    p.add_argument("--index-dir", default="output",
                   help="TF-IDF 索引目录（build_skill_vectors.py 的输出）")
    p.add_argument("--test-queries", default=os.path.join(here, "examples", "test_queries.json"),
                   help="测试集 JSON（tier 分层结构）")
    p.add_argument("--output", default="-",
                   help="报告输出文件路径（默认 - 输出到 stdout）")
    args = p.parse_args(argv)
    out = sys.stdout if args.output == "-" else open(args.output, "w", encoding="utf-8")
    try:
        with contextlib.redirect_stdout(out):
            run(args.skills_dir, args.index_dir, args.test_queries)
    finally:
        if out is not sys.stdout:
            out.close()


if __name__ == "__main__":
    main()
