# skill-hitrate-toolkit

Quality audit and hit-rate evaluation toolchain for AI-agent skill libraries. It measures trigger-word quality, detects fragments and collisions, evaluates keyword / TF-IDF routing, tunes similarity thresholds, and provides a routing CLI.

Measured on the author's real 135-skill library:

| Metric | Keyword | TF-IDF | Gain |
|--------|:---:|:---:|:---:|
| Top-1 | 6.7% | 28.9% | +4.3x |
| Top-3 | 22.2% | 51.1% | +2.3x |
| F1 | 12.5% | 44.8% | +3.6x |

## Tools

| Tool | Path | Purpose |
|------|------|---------|
| Trigger audit | `audit/skill_hitrate_audit.py` | PASS/WEAK/FAIL per skill with anti-mask aggregation |
| Fragment detector | `audit/fragment_detector.py` | Find truncated / overly generic triggers |
| Collision check | `audit/collision_check.py` | Cross-domain (V2) and intra-domain (V3) collisions |
| Fragment fixer | `audit/fix_fragments.py` | Replace truncated triggers (dry-run by default) |
| Collision fixer | `audit/fix_collisions.py` | Add domain-specific disambiguation triggers (dry-run by default) |
| Vector builder | `eval/build_skill_vectors.py` | Build TF-IDF index |
| Keyword eval | `eval/skill_hitrate_eval.py` | Precision / Recall / F1 / Top-N |
| Semantic eval | `eval/skill_hitrate_eval_v2.py` | Keyword vs TF-IDF vs hybrid comparison |
| Dual-tree eval | `eval/eval_dual_tree.py` | Memory-context boosted routing |
| Threshold tuning | `eval/tune_threshold.py` | Scan thresholds with <=5% misfire |
| Routing CLI | `eval/tfidf_router.py` | TF-IDF / BGE / three-stage routing |

## Quickstart

```bash
pip install -r requirements.txt
```

**Workflow 1: trigger quality diagnosis (5 min)**

```bash
python audit/skill_hitrate_audit.py --skills-dir examples/skills
python audit/fragment_detector.py --skills-dir examples/skills
python audit/collision_check.py --skills-dir examples/skills
```

**Workflow 2: hit-rate evaluation (10 min)**

```bash
python eval/build_skill_vectors.py --skills-dir examples/skills --output output
python eval/skill_hitrate_eval.py --skills-dir examples/skills
python eval/skill_hitrate_eval_v2.py --skills-dir examples/skills --index-dir output
python eval/tune_threshold.py --index-dir output
```

**Workflow 3: fix + verify loop (dry-run by default)**

```bash
python audit/fix_fragments.py --skills-dir examples/skills
python audit/fix_collisions.py --skills-dir examples/skills
python eval/build_skill_vectors.py --skills-dir examples/skills --output output
python eval/skill_hitrate_eval_v2.py --skills-dir examples/skills --index-dir output
```

Add `--apply` to the fix scripts only after reviewing the suggested changes.

## Skill library data format

`--skills-dir` contains `{domain}.json` files:

```json
{
  "skills": [
    {
      "name": "pdf",
      "domain": "doc",
      "desc": "PDF reading, merging, form filling",
      "triggers": ["PDF 合并", "表单填写"]
    }
  ]
}
```

## CLI conventions

- Every script supports `--output` (default `-` = stdout; `build_skill_vectors.py` uses it as the index output directory).
- Skill-consuming scripts support `--skills-dir` (default `examples/skills`); index-consuming scripts support `--index-dir` (default `output`).
- Test sets use a tiered JSON layout (`easy` / `medium` / `negative`), default `examples/test_queries.json`.
- Fix maps and memory simulation configs are external: `examples/fix_map.json`, `examples/collision_fixes.json`, `examples/memory_context.json`.

## Tests

```bash
python -m pytest tests/ -q
```

CI (GitHub Actions) runs unit tests plus workflow smoke tests on Ubuntu and Windows.

## License

MIT
