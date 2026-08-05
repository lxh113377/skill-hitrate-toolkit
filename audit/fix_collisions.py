#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fix_collisions.py — 修复高风险跨域触发词
=======================================
为泛化词添加领域特化触发词（保留原词 + 增加消歧词）。

默认 dry-run: 只输出建议不修改输入；加 --apply 写回。
修复映射从 --fix-map JSON 加载（默认 examples/collision_fixes.json）。
"""
import argparse, contextlib, json, glob, os, sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def load_fix_map(path):
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def run(skills_dir, fixes, apply_changes=False):
    fixed = 0
    for fpath in sorted(glob.glob(os.path.join(skills_dir, "*.json"))):
        bn = os.path.basename(fpath)
        if bn in ("embeddings.npy", "skill_ids.json", "tfidf_matrix.npz",
                  "tfidf_vectorizer.pkl"):
            continue
        data = json.load(open(fpath, encoding="utf-8"))
        changed = False
        for s in data.get("skills", []):
            name = s.get("name")
            if name in fixes:
                existing = set(s.get("triggers", []))
                new_triggers = [t for t in fixes[name] if t not in existing]
                if new_triggers:
                    s["triggers"] = list(existing) + new_triggers
                    changed = True
        if changed:
            if apply_changes:
                json.dump(data, open(fpath, "w", encoding="utf-8"),
                          ensure_ascii=False, separators=(",", ":"))
                action = "已写回"
            else:
                action = "dry-run, 加 --apply 写回"
            count = sum(1 for s in data["skills"] if s.get("name") in fixes)
            print(f"[FIXED] {bn}: {count} skills +领域消歧触发词 ({action})")
            fixed += count
    print(f"\n总计: {fixed} 个 skill 添加了领域消歧触发词")
    return fixed


def main(argv=None):
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = argparse.ArgumentParser(description="修复高风险跨域触发词（默认 dry-run）")
    p.add_argument("--skills-dir", default=os.path.join("examples", "skills"),
                   help="技能库目录: 内含 *.json, 结构 {\"skills\":[{name,triggers,...}]}")
    p.add_argument("--fix-map", default=os.path.join(here, "examples", "collision_fixes.json"),
                   help="修复映射 JSON: {skill名: [新增触发词, ...]}")
    p.add_argument("--apply", action="store_true",
                   help="写回修改（默认仅 dry-run 输出建议）")
    p.add_argument("--output", default="-",
                   help="输出文件路径（默认 - 输出到 stdout）")
    args = p.parse_args(argv)
    fixes = load_fix_map(args.fix_map)
    out = sys.stdout if args.output == "-" else open(args.output, "w", encoding="utf-8")
    try:
        with contextlib.redirect_stdout(out):
            run(args.skills_dir, fixes, args.apply)
    finally:
        if out is not sys.stdout:
            out.close()


if __name__ == "__main__":
    main()
