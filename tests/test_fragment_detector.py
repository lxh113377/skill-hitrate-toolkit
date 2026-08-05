from conftest import load_module, write_skills_dir

frag = load_module("audit", "fragment_detector.py")


def test_suspicious_short_chinese():
    reasons = frag.is_suspicious_fragment("的", "any-skill")
    assert reasons is not None


def test_normal_trigger_not_suspicious():
    assert frag.is_suspicious_fragment("代码审查", "github-code-review") is None
    assert frag.is_suspicious_fragment("生成图片", "imagegen") is None
    assert frag.is_suspicious_fragment("debug", "debugging-fixing") is None


def test_truncation_ending_detected():
    assert frag.is_suspicious_fragment("只加载所需内容减", "memory-loader") is not None


def test_run_reports_fragments(tmp_path, capsys):
    skills_dir = write_skills_dir(tmp_path, {
        "web": [{"name": "demo", "triggers": ["正常触发词", "只加载所需内容减"]}],
    })
    frag.run(str(skills_dir))
    out = capsys.readouterr().out
    assert "可疑触发词" in out
    assert "只加载所需内容减" in out
