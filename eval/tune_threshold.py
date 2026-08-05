#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TF-IDF 阈值扫描 — 找 Top-1/F1 最优 & 误触发 <=5% 的阈值"""
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


def tfidf_match(query, vectorizer, matrix, skill_index, threshold):
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, matrix)[0]
    idxs = np.argsort(sims)[::-1][:10]
    return [(skill_index[i]["name"], float(sims[i]))
            for i in idxs if sims[i] > threshold]


def eval_threshold(vectorizer, matrix, skill_index, all_queries, threshold):
    tp = fp = fn = 0
    fp_neg = 0
    for q in all_queries:
        matched = tfidf_match(q["query"], vectorizer, matrix, skill_index, threshold)
        names = [m[0] for m in matched]
        expected = q.get("expected_skill")
        if expected is None:
            if names:
                fp_neg += 1
            continue
        exp_set = {expected} if isinstance(expected, str) else set(expected)
        top1 = names[0] if names else None
        if top1 and top1 in exp_set:
            tp += 1
        elif top1 and top1 not in exp_set:
            fp += 1
        if not names:
            fn += 1
    n_exp = sum(1 for q in all_queries if q.get("expected_skill") is not None)
    n_neg = sum(1 for q in all_queries if q.get("expected_skill") is None)
    p = tp / (tp + fp) * 100 if (tp + fp) else 0
    r = tp / (tp + fn) * 100 if (tp + fn) else 0
    f1 = 2 * p * r / (p + r) if (p + r) else 0
    top1_acc = tp / n_exp * 100 if n_exp else 0
    fp_rate = fp_neg / n_neg * 100 if n_neg else 0
    return top1_acc, f1, fp_rate


def run(index_dir, test_queries):
    vectorizer, matrix, skill_index = load_index(index_dir)
    data = json.load(open(test_queries, encoding="utf-8"))
    all_queries = []
    for tier in data:
        for q in tier["queries"]:
            all_queries.append(q)

    print(f"{'阈值':>8} {'Top-1':>7} {'F1':>7} {'误触发':>7} {'判定'}")
    print(f"{'-'*8} {'-'*7} {'-'*7} {'-'*7} {'-'*8}")
    best = (0, 0, 0, 0)
    for t in [0.03, 0.05, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.15, 0.18, 0.20]:
        top1, f1, fp = eval_threshold(vectorizer, matrix, skill_index, all_queries, t)
        judge = "OK" if fp <= 5 and top1 > best[1] else ("MISFIRE" if fp > 5 else "")
        print(f"{t:>8.2f} {top1:>6.1f}% {f1:>6.1f}% {fp:>6.1f}% {judge}")
        if fp <= 5 and top1 > best[1]:
            best = (t, top1, f1, fp)

    print(f"\n推荐阈值: {best[0]:.2f} (Top-1={best[1]:.1f}%, F1={best[2]:.1f}%, 误触发={best[3]:.1f}%)")
    return best[0]


def main(argv=None):
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = argparse.ArgumentParser(description="TF-IDF 相似度阈值扫描")
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
            run(args.index_dir, args.test_queries)
    finally:
        if out is not sys.stdout:
            out.close()


if __name__ == "__main__":
    main()
