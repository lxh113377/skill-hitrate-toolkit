from conftest import load_module, write_skills_dir

collision = load_module("audit", "collision_check.py")


def test_cross_domain_collision_detected(tmp_path, capsys):
    skills_dir = write_skills_dir(tmp_path, {
        "web": [{"name": "deploy", "triggers": ["部署上线"]}],
        "system": [{"name": "ci", "triggers": ["部署上线"]}],
    })
    collision.run(str(skills_dir))
    out = capsys.readouterr().out
    assert "V2 跨域冲突: 1" in out


def test_intra_domain_collision_detected(tmp_path, capsys):
    skills_dir = write_skills_dir(tmp_path, {
        "doc": [
            {"name": "pdf", "triggers": ["生成报告"]},
            {"name": "docx", "triggers": ["生成报告"]},
        ],
    })
    collision.run(str(skills_dir))
    out = capsys.readouterr().out
    assert "V3 域内冲突: 1" in out


def test_no_collision_when_unique(tmp_path, capsys):
    skills_dir = write_skills_dir(tmp_path, {
        "web": [{"name": "deploy", "triggers": ["部署上线"]}],
        "system": [{"name": "ci", "triggers": ["构建流水线"]}],
    })
    collision.run(str(skills_dir))
    out = capsys.readouterr().out
    assert "V2 跨域冲突: 0" in out
