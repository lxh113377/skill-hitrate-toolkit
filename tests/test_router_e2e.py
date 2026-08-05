import subprocess
import sys
import json

import pytest

def _has_sklearn():
    try:
        import sklearn  # noqa: F401
        return True
    except ImportError:
        return False


from conftest import ROOT  # noqa: E402

pytestmark = pytest.mark.skipif(
    not _has_sklearn(),
    reason="需要 numpy/scipy/scikit-learn 才能跑语义路由端到端",
)


def _run_py(args, cwd=ROOT):
    return subprocess.run(
        [sys.executable, *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=180,
    )


def test_workflow_build_and_route(tmp_path):
    skills = ROOT / "examples" / "skills"
    out_dir = tmp_path / "output"
    # 1. 构建索引
    r = _run_py(["eval/build_skill_vectors.py", "--skills-dir", str(skills),
                 "--output", str(out_dir)])
    assert r.returncode == 0, r.stderr
    assert (out_dir / "tfidf_matrix.npz").exists()
    # 2. TF-IDF 路由
    r = _run_py(["eval/tfidf_router.py", "--tfidf", "--skills-dir", str(skills),
                 "--index-dir", str(out_dir), "PDF 合并"])
    assert r.returncode == 0, r.stderr
    assert "pdf" in r.stdout
    # 3. 阈值扫描
    r = _run_py(["eval/tune_threshold.py", "--index-dir", str(out_dir),
                 "--test-queries", str(ROOT / "examples" / "test_queries.json")])
    assert r.returncode == 0, r.stderr
    assert "推荐阈值" in r.stdout
    # 4. 双树协同
    r = _run_py(["eval/eval_dual_tree.py", "--index-dir", str(out_dir),
                 "--test-queries", str(ROOT / "examples" / "test_queries.json"),
                 "--memory-map", str(ROOT / "examples" / "memory_context.json")])
    assert r.returncode == 0, r.stderr
    assert "双树协同对比" in r.stdout


def test_audit_and_fix_dry_run(tmp_path):
    skills = ROOT / "examples" / "skills"
    r = _run_py(["audit/skill_hitrate_audit.py", "--skills-dir", str(skills)])
    assert r.returncode == 0
    assert "[WEAK]" in r.stdout
    # dry-run: 有可修碎片时只出建议、不改输入文件
    fix_dir = tmp_path / "fixable"
    fix_dir.mkdir()
    (fix_dir / "web.json").write_text(
        json.dumps({"skills": [{"name": "demo", "triggers": ["只加载所需内容减"]}]},
                   ensure_ascii=False),
        encoding="utf-8",
    )
    r = _run_py(["audit/fix_fragments.py", "--skills-dir", str(skills)])
    assert r.returncode == 0
    r = _run_py(["audit/fix_fragments.py", "--skills-dir", str(fix_dir)])
    assert r.returncode == 0
    assert "dry-run" in r.stdout
    assert "只加载所需内容减" in (fix_dir / "web.json").read_text(encoding="utf-8")
    # --apply 才写回
    r = _run_py(["audit/fix_fragments.py", "--skills-dir", str(fix_dir), "--apply"])
    assert r.returncode == 0
    assert "按需加载记忆文件" in (fix_dir / "web.json").read_text(encoding="utf-8")
