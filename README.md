# TenCap

一个轻量级的 tensor / ndarray 捕获（dump）工具，面向“单进程视角”记录调试数据到磁盘。

## API

- Setup: `tc.setup(root_dir, ...)` 全局初始化/更新配置
- Context: `with tc.scope(tag, ...):` 定义捕获作用域（可嵌套）
- Action: `tc.dump_np(obj, name)` / `tc.dump_torch(obj, name)`
- Tick: `tc.step()` 结束当前 step，step 计数 +1

说明：
- `dump_*` 在未满足捕获条件（未 `setup` / 不在 `scope` / interval 未命中 / 超出 max_captures）时返回 `None`，否则返回实际写入的文件路径。

## 保存路径布局（默认）

`{root_dir}/{tag_path}/pid_{pid}/cap_{capture_count}/{name}.{ext}`

其中：
- `tag_path` 为嵌套 `scope(tag)` 的 tag 拼接路径（例如 `train/attn`）
- `capture_count` 为该 `(root_dir, tag_path, pid)` 下的捕获次数，从 0 开始；每个 step 内首次 dump 才会触发并在 `step()` 后自增

## 示例

```python
import numpy as np
import tencap as tc

tc.setup("/tmp/tencap", interval=10, max_captures=3)

for _ in range(100):
    x = np.arange(6).reshape(2, 3)
    with tc.scope("train"):
        tc.dump_np(x, name="x")
    tc.step()
```

## 开发环境（.venv）

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip

# 仅 numpy
pip install -e .

# 需要 torch 时（会安装 torch）
pip install -e ".[torch]"

python -m unittest discover -s tests -p "test_*.py"
```
