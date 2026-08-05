#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collision_check.py — Skill 触发词冲突检测（V2 域冲突 + V3 Skill 冲突）
================================================================
V2: 检测两个不同域共享相同触发词（跨域误抢）
V3: 检测同域内两个不同 skill 共享相同触发词（域内冲突）
"""
import argparse, contextlib, json, glob, os, sys
from collections import defaultdict

def load_all(skills_dir):
    trig_map = defaultdict(list)  # trigger → [(skill, domain)]
    for fpath in sorted(glob.glob(os.path.join(skills_dir, "*.json"))):
        bn = os.path.basename(fpath)
        if bn in ("embeddings.npy", "skill_ids.json", "tfidf_matrix.npz", "tfidf_vectorizer.pkl"):
            continue
        domain = os.path.splitext(bn)[0]
        data = json.load(open(fpath, encoding="utf-8"))
        for s in data.get("skills", []):
            for tg in s.get("triggers", []):
                trig_map[tg.lower()].append((s["name"], domain))
    return trig_map

def run(skills_dir):
    trig_map = load_all(skills_dir)
    
    # V2: 跨域冲突
    cross_domain = []
    for tg, skills in trig_map.items():
        domains = set(d for _, d in skills)
        if len(domains) > 1:
            cross_domain.append((tg, skills, domains))
    cross_domain.sort(key=lambda x: -len(x[2]))
    
    # V3: 同域内冲突
    intra_domain = []
    for tg, skills in trig_map.items():
        if len(skills) <= 1:
            continue
        by_domain = defaultdict(list)
        for sn, sd in skills:
            by_domain[sd].append(sn)
        for domain, snames in by_domain.items():
            if len(snames) > 1:
                intra_domain.append((tg, domain, snames))
    
    print("=" * 60)
    print(f"V2 跨域冲突: {len(cross_domain)} 个触发词被多个域共享")
    print("=" * 60)
    for tg, skills, domains in cross_domain[:20]:
        domain_str = ", ".join(sorted(domains))
        skill_str = ", ".join(f"{s}({d})" for s, d in skills[:5])
        if len(skills) > 5:
            skill_str += f" ... +{len(skills)-5}"
        print(f"  🔴 '{tg}' → {domain_str}")
        print(f"     涉及: {skill_str}")
    if len(cross_domain) > 20:
        print(f"  ... 还有 {len(cross_domain)-20} 个")
    
    print(f"\n{'='*60}")
    print(f"V3 域内冲突: {len(intra_domain)} 处同域 skill 共享触发词")
    print(f"{'='*60}")
    for tg, domain, snames in intra_domain:
        print(f"  🟡 [{domain}] '{tg}' → {', '.join(snames)}")
    
    # 风险评估
    high_risk = [c for c in cross_domain if len(c[2]) >= 3]
    if high_risk:
        print(f"\n⚠️ 高风险跨域冲突 (≥3域共享): {len(high_risk)} 个")
        for tg, skills, domains in high_risk:
            print(f"  ❌ '{tg}' → {sorted(domains)}")
    
    print(f"\n总结: V2={len(cross_domain)}, V3={len(intra_domain)}, 高风险={len(high_risk)}")


def main(argv=None):
    p = argparse.ArgumentParser(description="Skill 触发词冲突检测（V2 跨域 + V3 域内）")
    p.add_argument("--skills-dir", default=os.path.join("examples", "skills"),
                   help="技能库目录: 内含 *.json, 结构 {\"skills\":[{name,triggers,...}]}")
    p.add_argument("--output", default="-",
                   help="报告输出文件路径（默认 - 输出到 stdout）")
    args = p.parse_args(argv)
    out = sys.stdout if args.output == "-" else open(args.output, "w", encoding="utf-8")
    try:
        with contextlib.redirect_stdout(out):
            run(args.skills_dir)
    finally:
        if out is not sys.stdout:
            out.close()


if __name__ == "__main__":
    main()
