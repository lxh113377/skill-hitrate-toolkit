#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_skill_vectors.py — 构建 Skill TF-IDF 语义向量索引（轻量，零依赖安装）
==========================================================================
使用 sklearn TfidfVectorizer 为每个 skill 生成稀疏向量，
支持中文分词 + 英文 tokenization，存储为 numpy/csr 矩阵。

输出 (--output 目录):
  tfidf_matrix.npz       (N x M 稀疏矩阵)
  tfidf_vectorizer.pkl   (vectorizer 参数)
  skill_ids.json         (索引→skill 名映射)
"""
import argparse, json, glob, os, sys, pickle, re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def build_text(skill):
    """拼接 skill 文本: name + desc + triggers（用于 TF-IDF）"""
    name = skill.get("name", "")
    desc = skill.get("desc", "")[:600]
    triggers = " ".join(skill.get("triggers", []))
    # 中英混合，保留原始文本让 TfidfVectorizer 处理
    return f"{name} {triggers} {desc}"

def run(skills_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    # 加载所有 skill
    skills_meta = []
    texts = []
    for fpath in sorted(glob.glob(os.path.join(skills_dir, "*.json"))):
        bn = os.path.basename(fpath)
        if bn in ("skill_ids.json",):
            continue
        domain = os.path.splitext(bn)[0]
        data = json.load(open(fpath, encoding="utf-8"))
        for s in data.get("skills", []):
            text = build_text(s)
            skills_meta.append({"name": s["name"], "domain": domain})
            texts.append(text)
    
    print(f"编码 {len(skills_meta)} 个 skill ...")
    
    # TF-IDF: char-level + word-level 混合（中英兼容）
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",        # char-level with word boundaries
        ngram_range=(2, 4),        # 2-4 gram（覆盖中文词 + 英文短词）
        max_features=5000,         # 词汇上限
        sublinear_tf=True,         # 1+log(tf) 平滑
        max_df=0.85,               # 忽略过于通用的词
        min_df=1,                  # 保留所有词
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    print(f"  矩阵形状: {tfidf_matrix.shape}")
    print(f"  词汇量: {len(vectorizer.get_feature_names_out())}")
    
    # 保存矩阵（稀疏格式）
    from scipy import sparse
    sparse.save_npz(os.path.join(out_dir, "tfidf_matrix.npz"), tfidf_matrix)
    
    # 保存 vectorizer
    with open(os.path.join(out_dir, "tfidf_vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
    
    # 保存 skill 索引
    json.dump(skills_meta, 
              open(os.path.join(out_dir, "skill_ids.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    
    print(f"完成: tfidf_matrix.npz + tfidf_vectorizer.pkl + skill_ids.json")


def main(argv=None):
    p = argparse.ArgumentParser(description="构建 Skill TF-IDF 语义向量索引")
    p.add_argument("--skills-dir", default=os.path.join("examples", "skills"),
                   help="技能库目录: 内含 *.json, 结构 {\"skills\":[{name,triggers,desc,...}]}")
    p.add_argument("--output", default="output",
                   help="索引输出目录（tfidf_matrix.npz / tfidf_vectorizer.pkl / skill_ids.json）")
    args = p.parse_args(argv)
    run(args.skills_dir, args.output)


if __name__ == "__main__":
    main()
