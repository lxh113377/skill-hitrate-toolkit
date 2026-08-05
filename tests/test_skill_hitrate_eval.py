from conftest import load_module, ROOT

eval_mod = load_module("eval", "skill_hitrate_eval.py")


def test_keyword_eval_on_examples(capsys):
    skills_dir = str(ROOT / "examples" / "skills")
    test_queries = str(ROOT / "examples" / "test_queries.json")
    eval_mod.run(skills_dir, test_queries)
    out = capsys.readouterr().out
    assert "全局汇总" in out
    assert "Top-1" in out
    # 示例库 easy 层应全部命中
    assert "已加载 10 个 skill" in out
