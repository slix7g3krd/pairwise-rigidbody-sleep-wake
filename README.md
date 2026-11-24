# pairwise-rigidbody-sleep-wake

## 领域

游戏、图形与多媒体

## 主题

刚体休眠与唤醒传播

## 说明

滞回休眠、连通唤醒与清理登记。

初始状态下 `gameplay/rigidbody_sleep_wake.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
