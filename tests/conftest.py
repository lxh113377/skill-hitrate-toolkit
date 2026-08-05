import importlib.util
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_module(pkg, name):
    """从 audit/ 或 eval/ 加载脚本模块（避免包结构依赖）"""
    path = ROOT / pkg / name
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_skills_dir(tmp_path, domains):
    """domains: {domain_name: [skill_dict, ...]} → 写入临时技能库目录"""
    d = tmp_path / "skills"
    d.mkdir()
    for domain, skills in domains.items():
        (d / f"{domain}.json").write_text(
            json.dumps({"skills": skills}, ensure_ascii=False),
            encoding="utf-8",
        )
    return d
