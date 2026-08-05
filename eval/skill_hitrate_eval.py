#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
skill_hitrate_eval.py — Skill 路由效果量化评估
==============================================
模拟当前 L1/L2 关键词匹配路由，对测试集逐条跑，输出 Precision/Recall/F1/Top-N。
不依赖 LLM — 纯模拟关键词路由逻辑，度量触发词覆盖质量。
"""
import argparse, contextlib, json, glob, os, sys, re
from collections import defaultdict

def load_skills(skills_dir):
    """加载所有 skill 的 name + triggers + domain + desc"""
    skills = []
    for fpath in sorted(glob.glob(os.path.join(skills_dir, "*.json"))):
        domain = os.path.splitext(os.path.basename(fpath))[0]
        data = json.load(open(fpath, encoding="utf-8"))
        for s in data.get("skills", []):
            skills.append({
                "name": s["name"],
                "domain": domain,
                "triggers": s.get("triggers", []),
                "desc": s.get("desc", "")[:500]
            })
    return skills

def match_query(query, skills):
    """
    模拟关键词路由:
    1. 对每个 skill 的 triggers 做子串匹配（忽略大小写）
    2. 返回按匹配数排序的 skill 列表
    """
    q_lower = query.lower()
    scores = []
    for s in skills:
        score = 0
        matched = []
        for tg in s["triggers"]:
            tg_lower = tg.lower()
            # 触发词作为子串出现在 query 中（或反过来）
            if tg_lower in q_lower or (len(tg_lower) >= 2 and q_lower in tg_lower):
                score += 1
                matched.append(tg)
        if score > 0:
            scores.append((s["name"], score, matched, s["domain"]))
    # 按匹配分降序
    scores.sort(key=lambda x: -x[1])
    return scores

def eval_tier(queries, skills, tier_name):
    """对一层测试集跑评估"""
    results = []
    for q in queries:
        query = q["query"]
        expected = q.get("expected_skill")
        # expected 可能是字符串或多个候选
        if isinstance(expected, str):
            expected_set = {expected}
        elif isinstance(expected, list):
            expected_set = set(expected)
        else:
            expected_set = set()
        
        matches = match_query(query, skills)
        matched_names = [m[0] for m in matches]
        
        # Top-1 是否正确
        top1 = matched_names[0] if matched_names else None
        top1_correct = top1 in expected_set if expected_set else (top1 is None)
        
        # Top-3 是否含正确
        top3 = matched_names[:3]
        top3_correct = any(n in expected_set for n in top3) if expected_set else True
        
        # 是否至少命中一个
        any_hit = bool(matched_names)
        
        results.append({
            "query": query,
            "expected": list(expected_set),
            "matched": matched_names[:5],
            "top1": top1,
            "top1_correct": top1_correct,
            "top3_correct": top3_correct,
            "any_hit": any_hit,
            "match_details": [(m[0], m[1], m[2]) for m in matches[:5]]
        })
    
    # 汇总
    n = len(queries)
    top1_acc = sum(1 for r in results if r["top1_correct"]) / n * 100 if n else 0
    top3_acc = sum(1 for r in results if r["top3_correct"]) / n * 100 if n else 0
    any_hit_rate = sum(1 for r in results if r["any_hit"]) / n * 100 if n else 0
    
    # Precision/Recall（有预期的 query）
    expected_queries = [r for r in results if r["expected"]]
    tp = sum(1 for r in expected_queries if r["top1_correct"])
    fp = sum(1 for r in expected_queries if r["top1"] and not r["top1_correct"])
    fn = sum(1 for r in expected_queries if not r["any_hit"])
    precision = tp / (tp + fp) * 100 if (tp + fp) else 0
    recall = tp / (tp + fn) * 100 if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    
    return {
        "tier": tier_name,
        "n": n,
        "top1_acc": round(top1_acc, 1),
        "top3_acc": round(top3_acc, 1),
        "any_hit_rate": round(any_hit_rate, 1),
        "precision": round(precision, 1),
        "recall": round(recall, 1),
        "f1": round(f1, 1),
        "results": results
    }

def run(skills_dir, test_queries):
    skills = load_skills(skills_dir)
    print(f"已加载 {len(skills)} 个 skill\n")
    
    data = json.load(open(test_queries, encoding="utf-8"))
    
    tier_results = []
    for tier_data in data:
        tier_name = tier_data["tier"]
        queries = tier_data["queries"]
        print(f"\n{'='*60}")
        print(f"【{tier_name.upper()}】{tier_data.get('description','')} ({len(queries)} 条)")
        print(f"{'='*60}")
        
        result = eval_tier(queries, skills, tier_name)
        tier_results.append(result)
        
        for r in result["results"]:
            q = r["query"]
            ok = "✅" if r["top1_correct"] else ("⚠️" if r["top3_correct"] else "❌")
            matched_str = " → ".join(r["matched"][:3]) if r["matched"] else "(无匹配)"
            exp_str = " | ".join(r["expected"]) if r["expected"] else "(不应触发)"
            print(f"  {ok} '{q}'")
            print(f"     期望: {exp_str}")
            print(f"     命中: {matched_str}")
    
    # ---- 全局汇总 ----
    print(f"\n{'='*60}")
    print(f"📊 全局汇总")
    print(f"{'='*60}")
    print(f"{'层级':<10} {'数量':>4} {'Top-1':>7} {'Top-3':>7} {'命中率':>7} {'Precision':>10} {'Recall':>8} {'F1':>7}")
    print(f"{'-'*10} {'-'*4} {'-'*7} {'-'*7} {'-'*7} {'-'*10} {'-'*8} {'-'*7}")
    
    # 综合计算
    all_easy = [r for t in tier_results if t["tier"] == "easy" for r in t["results"]]
    all_medium = [r for t in tier_results if t["tier"] == "medium" for r in t["results"]]
    all_hard = [r for t in tier_results if t["tier"] == "hard" for r in t["results"]]
    all_negative = [r for t in tier_results if t["tier"] == "negative" for r in t["results"]]
    all_expected = all_easy + all_medium + all_hard  # negative 不参与 P/R/F1
    
    for tr in tier_results:
        print(f"{tr['tier']:<10} {tr['n']:>4} {tr['top1_acc']:>6.1f}% {tr['top3_acc']:>6.1f}% {tr['any_hit_rate']:>6.1f}% {tr['precision']:>9.1f}% {tr['recall']:>7.1f}% {tr['f1']:>6.1f}%")
    
    # 综合行
    n_all = sum(tr["n"] for tr in tier_results)
    # 综合 P/R/F1
    tp_all = sum(1 for r in all_expected if r["top1_correct"])
    fp_all = sum(1 for r in all_expected if r["top1"] and not r["top1_correct"])
    fn_all = sum(1 for r in all_expected if not r["any_hit"])
    p_all = tp_all / (tp_all + fp_all) * 100 if (tp_all + fp_all) else 0
    r_all = tp_all / (tp_all + fn_all) * 100 if (tp_all + fn_all) else 0
    f1_all = 2 * p_all * r_all / (p_all + r_all) if (p_all + r_all) else 0
    
    top1_all = sum(1 for r in all_expected if r["top1_correct"]) / len(all_expected) * 100
    top3_all = sum(1 for r in all_expected if r["top3_correct"]) / len(all_expected) * 100
    hit_all = sum(1 for r in all_expected if r["any_hit"]) / len(all_expected) * 100
    
    # 误触发检查（negative）
    fp_neg = sum(1 for r in all_negative if r["any_hit"])
    fp_rate = fp_neg / len(all_negative) * 100 if all_negative else 0
    
    print(f"{'综合':<10} {n_all:>4} {top1_all:>6.1f}% {top3_all:>6.1f}% {hit_all:>6.1f}% {p_all:>9.1f}% {r_all:>7.1f}% {f1_all:>6.1f}%")
    print(f"\n误触发率 (negative): {fp_neg}/{len(all_negative)} = {fp_rate:.1f}%")
    
    # 失败明细
    failures = [r for r in all_expected if not r["top1_correct"]]
    if failures:
        print(f"\n⚠️ Top-1 未命中 ({len(failures)} 条):")
        for r in failures:
            exp = " | ".join(r["expected"])
            got = r["top1"] or "(无)"
            query_text = r["query"] if isinstance(r["query"], str) else r["query"].get("query", str(r["query"]))
            print(f"  ❌ '{query_text}' → 期望[{exp}] 实际命中[{got}]")


def main(argv=None):
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = argparse.ArgumentParser(description="Skill 路由效果量化评估（关键词模拟）")
    p.add_argument("--skills-dir", default=os.path.join("examples", "skills"),
                   help="技能库目录: 内含 *.json, 结构 {\"skills\":[{name,triggers,...}]}")
    p.add_argument("--test-queries", default=os.path.join(here, "examples", "test_queries.json"),
                   help="测试集 JSON（tier 分层结构）")
    p.add_argument("--output", default="-",
                   help="报告输出文件路径（默认 - 输出到 stdout）")
    args = p.parse_args(argv)
    out = sys.stdout if args.output == "-" else open(args.output, "w", encoding="utf-8")
    try:
        with contextlib.redirect_stdout(out):
            run(args.skills_dir, args.test_queries)
    finally:
        if out is not sys.stdout:
            out.close()


if __name__ == "__main__":
    main()
