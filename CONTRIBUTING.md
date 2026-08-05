# 贡献指南

欢迎 issue 与 PR。请遵守以下约定：

1. **先开 issue 再提大改动**：功能类改动建议先说明动机与方案，避免返工。
2. **保持 CLI 约定**：新脚本或改动需遵循 `--skills-dir` / `--output` / `--index-dir` 参数约定（见 README）。
3. **必须带测试**：`tests/` 下为改动补测试，运行 `python -m pytest tests/ -q` 全绿。
4. **不要提交个人数据**：技能库示例只放通用示例；不要把个人路径、密钥、私人项目名提交进仓库。
5. **中文注释优先**：本项目默认中文文档与注释。

## 本地开发

```bash
pip install -r requirements.txt
python -m pytest tests/ -q
```
