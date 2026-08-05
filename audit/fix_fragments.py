#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fix_fragments.py — 批量修复触发词碎片
======================================
策略: 只修「明确截断」类碎片（从 SKILL.md 提取时被截断的），
      不碰「极短泛化词」（同域共享词在 L2 域内匹配中是正常行为，
      需要 embedding 层来治本，不在此脚本处理范围）。

默认 dry-run: 只输出修复建议，不修改输入文件；加 --apply 才写回。
修复映射从 --fix-map JSON 加载（默认 examples/fix_map.json）。
"""
import argparse, contextlib, json, glob, os, sys


def load_fix_map(path):
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def fix_skill(skill, fix_map):
    """替换截断碎片为修复后的语义短语"""
    trigs = skill.get("triggers", [])
    fixed = []
    changed = False
    for tg in trigs:
        if tg in fix_map:
            fixed.append(fix_map[tg])
            changed = True
        else:
            fixed.append(tg)
    if changed:
        skill["triggers"] = fixed
    return changed


def run(skills_dir, fix_map, apply_changes=False):
    files = sorted(glob.glob(os.path.join(skills_dir, "*.json")))
    total_fixed = 0
    for fpath in files:
        data = json.load(open(fpath, encoding="utf-8"))
        domain_fixed = 0
        for s in data.get("skills", []):
            if fix_skill(s, fix_map):
                domain_fixed += 1
        if domain_fixed:
            if apply_changes:
                json.dump(data, open(fpath, "w", encoding="utf-8"),
                          ensure_ascii=False, separators=(",", ":"))
                print(f"[FIXED] {os.path.basename(fpath)}: {domain_fixed} skills (已写回)")
            else:
                print(f"[WOULD-FIX] {os.path.basename(fpath)}: {domain_fixed} skills "
                      "(dry-run, 加 --apply 写回)")
            total_fixed += domain_fixed
        else:
            print(f"[SKIP]  {os.path.basename(fpath)}: no fragments to fix")
    print(f"\n总计修复: {total_fixed} 个 skill 的触发词碎片")
    return total_fixed


def main(argv=None):
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = argparse.ArgumentParser(description="批量修复触发词碎片（默认 dry-run）")
    p.add_argument("--skills-dir", default=os.path.join("examples", "skills"),
                   help="技能库目录: 内含 *.json, 结构 {\"skills\":[{name,triggers,...}]}")
    p.add_argument("--fix-map", default=os.path.join(here, "examples", "fix_map.json"),
                   help="修复映射 JSON: {碎片触发词: 正确语义短语}")
    p.add_argument("--apply", action="store_true",
                   help="写回修改（默认仅 dry-run 输出建议）")
    p.add_argument("--output", default="-",
                   help="输出文件路径（默认 - 输出到 stdout）")
    args = p.parse_args(argv)
    fix_map = load_fix_map(args.fix_map)
    out = sys.stdout if args.output == "-" else open(args.output, "w", encoding="utf-8")
    try:
        with contextlib.redirect_stdout(out):
            run(args.skills_dir, fix_map, args.apply)
    finally:
        if out is not sys.stdout:
            out.close()


if __name__ == "__main__":
    main()
