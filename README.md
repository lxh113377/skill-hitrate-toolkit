# skill-hitrate-toolkit

Skill 触发词质量审计与命中率评估工具链 —— 给「AI Agent 技能库」（Skill 目录 + 触发词）做量化体检：触发词碎片检测、跨域冲突检测、关键词 / TF-IDF 路由命中率评估、阈值调优与路由 CLI。

[![CI](https://github.com/lxh113377/skill-hitrate-toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/lxh113377/skill-hitrate-toolkit/actions)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/lxh113377/skill-hitrate-toolkit?style=social)](https://github.com/lxh113377/skill-hitrate-toolkit)

作者在真实技能库（135 skills）上的实测收益：

| 指标 | 纯关键词 | TF-IDF | 提升 |
|------|:---:|:---:|:---:|
| Top-1 | 6.7% | 28.9% | +4.3x |
| Top-3 | 22.2% | 51.1% | +2.3x |
| F1 | 12.5% | 44.8% | +3.6x |

## 工具清单

| 工具 | 路径 | 功能 |
|------|------|------|
| 触发词审计 | `audit/skill_hitrate_audit.py` | 触发词质量 PASS/WEAK/FAIL，反掩盖安全聚合 |
| 碎片检测 | `audit/fragment_detector.py` | 识别截断碎片 / 泛化短词 |
| 冲突检测 | `audit/collision_check.py` | V2 跨域 + V3 域内触发词冲突 |
| 碎片修复 | `audit/fix_fragments.py` | 截断碎片 → 语义短语（默认 dry-run） |
| 冲突修复 | `audit/fix_collisions.py` | 泛化词 → 添加领域消歧词（默认 dry-run） |
| 向量构建 | `eval/build_skill_vectors.py` | TF-IDF 语义向量索引 |
| 关键词评估 | `eval/skill_hitrate_eval.py` | 纯关键词命中率（P/R/F1/Top-N） |
| 语义评估 | `eval/skill_hitrate_eval_v2.py` | 关键词 vs TF-IDF vs 混合对比 |
| 双树评估 | `eval/eval_dual_tree.py` | Memory 上下文增强路由评估 |
| 阈值调优 | `eval/tune_threshold.py` | 相似度阈值扫描（误触发 ≤5%） |
| 路由 CLI | `eval/tfidf_router.py` | TF-IDF / BGE / 三级路由 CLI |

## 快速开始

```bash
pip install -r requirements.txt
```

**工作流 1：触发词质量诊断（5 分钟）**

```bash
python audit/skill_hitrate_audit.py --skills-dir examples/skills
python audit/fragment_detector.py --skills-dir examples/skills
python audit/collision_check.py --skills-dir examples/skills
```

**工作流 2：命中率完整评估（10 分钟）**

```bash
python eval/build_skill_vectors.py --skills-dir examples/skills --output output
python eval/skill_hitrate_eval.py --skills-dir examples/skills
python eval/skill_hitrate_eval_v2.py --skills-dir examples/skills --index-dir output
python eval/tune_threshold.py --index-dir output
```

**工作流 3：修复 + 验证闭环（默认 dry-run，不修改输入）**

```bash
python audit/fix_fragments.py --skills-dir examples/skills
python audit/fix_collisions.py --skills-dir examples/skills
python eval/build_skill_vectors.py --skills-dir examples/skills --output output
python eval/skill_hitrate_eval_v2.py --skills-dir examples/skills --index-dir output
```

确认修复建议无误后，给修复脚本加 `--apply` 才写回文件。

## 技能库数据格式

`--skills-dir` 指向的目录内放若干 `{domain}.json`：

```json
{
  "skills": [
    {
      "name": "pdf",
      "domain": "doc",
      "desc": "PDF 读取、合并、表单填写",
      "triggers": ["PDF 合并", "表单填写"]
    }
  ]
}
```

## CLI 约定

- 所有脚本支持 `--output`（默认 `-` 输出到 stdout；`build_skill_vectors.py` 的 `--output` 是索引输出目录）。
- 消费技能库的脚本支持 `--skills-dir`（默认 `examples/skills`）；消费向量索引的脚本支持 `--index-dir`（默认 `output`）。
- 测试集统一为 tier 分层 JSON（`easy` / `medium` / `negative`），默认 `examples/test_queries.json`。
- 修复映射与 Memory 模拟配置全部外置：`examples/fix_map.json`、`examples/collision_fixes.json`、`examples/memory_context.json`。

## 测试

```bash
python -m pytest tests/ -q
```

CI（GitHub Actions）在 Ubuntu + Windows 上跑单测与工作流冒烟。

## 许可

MIT
