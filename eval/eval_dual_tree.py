#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_dual_tree.py — 双树协同（Memory Tree + Skill Tree）概念验证
================================================================
模拟: 当 Memory 上下文（用户画像/历史教训）注入 Skill 路由时，
     命中率能提升多少。

Memory 信号来源（模拟, 从 --memory-map JSON 加载）:
  - user_favorites:    用户常用 skill → 权重
  - context_skill_map: 上下文标签 → 关联 skill 列表
"""
import argparse, contextlib, json, os, pickle, sys
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def load_index(index_dir):
    matrix = sparse.load_npz(os.path.join(index_dir, "tfidf_matrix.npz"))
    with open(os.path.join(index_dir, "tfidf_vectorizer.pkl"), "rb") as f:
        vectorizer = pickle.load(f)
    skill_index = json.load(open(os.path.join(index_dir, "skill_ids.json"),
                                 encoding="utf-8"))
    return vectorizer, matrix, skill_index


def load_memory_map(path):
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("user_favorites", {}), data.get("context_skill_map", {})
    return {}, {}


def tfidf_match(query, vectorizer, matrix, skill_index, threshold=0.10):
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, matrix)[0]
    idxs = np.argsort(sims)[::-1][:15]
    return [(skill_index[i]["name"], float(sims[i]))
            for i in idxs if sims[i] > threshold]


def extract_context_tags(query, context_skill_map):
    """从 query 中提取上下文标签（模拟 Memory Tree 结构化输出）"""
    return [tag for tag in context_skill_map if tag in query]


def memory_boost(query, tfidf_results, context_skill_map, user_favorites):
    """
    双树协同核心:
    1. 从 query 提取上下文标签
    2. 根据标签找到关联 skill
    3. 对这些 skill 的 TF-IDF 分数做加权提升
    """
    tags = extract_context_tags(query, context_skill_map)
    if not tags:
        return tfidf_results  # 无上下文，不增强

    boost_skills = {}
    for tag in tags:
        for sk in context_skill_map.get(tag, []):
            boost_skills[sk] = max(boost_skills.get(sk, 0), 0.3)  # 上下文标签 boost

    # 用户偏好 boost
    result_names = [r[0] for r in tfidf_results]
    for sk, weight in user_favorites.items():
        if sk in result_names:
            boost_skills[sk] = max(boost_skills.get(sk, 0), weight * 0.15)

    boosted = []
    for name, score in tfidf_results:
        b = boost_skills.get(name, 0)
        boosted.append((name, min(score + b, 1.0)))
    boosted.sort(key=lambda x: -x[1])
    return boosted


def run_eval(all_q, match_fn):
    tp = fp = fn = 0
    fp_neg = 0
    for q in all_q:
        matched = match_fn(q["query"])
        names = [m[0] for m in matched]
        exp = q.get("expected_skill")
        if exp is None:
            if names:
                fp_neg += 1
            continue
        exp_set = {exp} if isinstance(exp, str) else set(exp)
        top1 = names[0] if names else None
        if top1 and top1 in exp_set:
            tp += 1
        elif top1:
            fp += 1
        if not names:
            fn += 1
    n_exp = sum(1 for q in all_q if q.get("expected_skill") is not None)
    n_neg = sum(1 for q in all_q if q.get("expected_skill") is None)
    p = tp / (tp + fp) * 100 if (tp + fp) else 0
    r = tp / (tp + fn) * 100 if (tp + fn) else 0
    f1 = 2 * p * r / (p + r) if (p + r) else 0
    return {"label": "", "top1": tp / n_exp * 100 if n_exp else 0, "f1": f1,
            "fp_rate": fp_neg / n_neg * 100 if n_neg else 0,
            "tp": tp, "fp": fp, "fn": fn}


def run(index_dir, test_queries, memory_map):
    vectorizer, matrix, skill_index = load_index(index_dir)
    user_favorites, context_skill_map = load_memory_map(memory_map)

    data = json.load(open(test_queries, encoding="utf-8"))
    all_q = []
    for tier in data:
        all_q.extend(tier["queries"])

    print("=" * 65)
    print("双树协同对比: TF-IDF 基线 vs Memory 增强")
    print("=" * 65)

    base_fn = lambda q: tfidf_match(q, vectorizer, matrix, skill_index, 0.10)
    mem_fn = lambda q: memory_boost(q, base_fn(q), context_skill_map, user_favorites)
    r_base = run_eval(all_q, base_fn)
    r_mem = run_eval(all_q, mem_fn)
    r_base["label"], r_mem["label"] = "TF-IDF 基线", "TF-IDF + Memory"

    print(f"{'方法':<25} {'Top-1':>7} {'F1':>7} {'误触发':>7} {'TP':>5} {'FP':>5} {'FN':>3}")
    print(f"{'-'*25} {'-'*7} {'-'*7} {'-'*7} {'-'*5} {'-'*5} {'-'*3}")
    for r in [r_base, r_mem]:
        print(f"{r['label']:<25} {r['top1']:>6.1f}% {r['f1']:>6.1f}% {r['fp_rate']:>6.1f}% "
              f"{r['tp']:>5} {r['fp']:>5} {r['fn']:>3}")

    print(f"\nMemory 增强提升: Top-1 +{r_mem['top1']-r_base['top1']:.1f}pp, "
          f"F1 +{r_mem['f1']-r_base['f1']:.1f}pp")

    print(f"\nMemory 增强生效的 query:")
    for q in all_q:
        query = q["query"]
        tags = extract_context_tags(query, context_skill_map)
        if not tags:
            continue
        base = [m[0] for m in base_fn(query)][:3]
        boosted = [m[0] for m in mem_fn(query)][:3]
        exp = q.get("expected_skill")
        exp_str = exp if isinstance(exp, str) else " | ".join(exp) if exp else "(无)"
        if base != boosted:
            print(f"  '{query}' 标签:{tags}")
            print(f"    期望: {exp_str}")
            print(f"    基线: {base}")
            print(f"    增强: {boosted}")


def main(argv=None):
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = argparse.ArgumentParser(description="双树协同（Memory + Skill）路由增强评估")
    p.add_argument("--index-dir", default="output",
                   help="TF-IDF 索引目录（build_skill_vectors.py 的输出）")
    p.add_argument("--test-queries", default=os.path.join(here, "examples", "test_queries.json"),
                   help="测试集 JSON（tier 分层结构）")
    p.add_argument("--memory-map", default=os.path.join(here, "examples", "memory_context.json"),
                   help="Memory 模拟配置 JSON: {user_favorites, context_skill_map}")
    p.add_argument("--output", default="-",
                   help="报告输出文件路径（默认 - 输出到 stdout）")
    args = p.parse_args(argv)
    out = sys.stdout if args.output == "-" else open(args.output, "w", encoding="utf-8")
    try:
        with contextlib.redirect_stdout(out):
            run(args.index_dir, args.test_queries, args.memory_map)
    finally:
        if out is not sys.stdout:
            out.close()


if __name__ == "__main__":
    main()
