import pytest

from conftest import load_module, write_skills_dir

audit = load_module("audit", "skill_hitrate_audit.py")


def test_example_skills_weak_not_fatal(tmp_path, capsys):
    """示例库含 1 个 WEAK 触发词 → 聚合 WEAK 且退出码 0"""
    skills_dir = write_skills_dir(tmp_path, {
        "web": [
            {"name": "good", "triggers": ["正常触发词", "另一个好词"]},
            {"name": "web-deploy", "triggers": ["部署上线", "的"]},
        ],
    })
    code = audit.run(str(skills_dir))
    out = capsys.readouterr().out
    assert "[WEAK]" in out
    assert code == 0


def test_all_garbage_is_fatal(tmp_path, capsys):
    skills_dir = write_skills_dir(tmp_path, {
        "web": [{"name": "bad", "triggers": ["的", "了"]}],
    })
    code = audit.run(str(skills_dir))
    out = capsys.readouterr().out
    assert "[FAIL]" in out
    assert code == 1


def test_cli_exit_code(tmp_path):
    skills_dir = write_skills_dir(tmp_path, {
        "web": [{"name": "bad", "triggers": ["的"]}],
    })
    with pytest.raises(SystemExit) as exc:
        audit.main(["--skills-dir", str(skills_dir)])
    assert exc.value.code == 1
