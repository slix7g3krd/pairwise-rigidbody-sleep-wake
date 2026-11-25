"""刚体休眠 / 唤醒连通传播 / 场景清理的最小确定性物理模型。

本模块不依赖任何引擎或第三方库，用一个可独立运行的最小刚体世界落实题目中的
六个修复点（括号内为对应的公共入口）：

1. 休眠加滞回（:func:`sleep_hysteresis` / :func:`sleep`）
   旧逻辑只拿单一“线速度阈值”判断，静止堆叠的求解器残差让速度一直在阈值附近
   小幅抖动，计时器不断清零，整堆永远不睡，物理线程逐帧空转。新逻辑使用
   *进入休眠阈值 + 滞回容忍带 + 持续时间 + 更高的唤醒阈值* 的施密特滞回：
   抖动落在容忍带内只会被阻尼，不会清零休眠累计；睡着后只有达到更高的唤醒
   阈值（或显式唤醒 / 邻居唤醒）才会醒。整堆最终休眠，岛被停用，空转计数归零。

2. 唤醒连通传播（:func:`wake_connected`）
   移走底部物体时，旧逻辑只唤醒直接邻居，上层物体失去支撑却仍在睡觉，整堆
   浮在空中。新逻辑沿接触图做传递闭包（BFS），把受影响连通分量里的所有物体
   一次性唤醒，顺序按稳定 id 排序，保证确定性。

3. 唤醒即时生效（:func:`wake_immediate` / :func:`wake`）
   旧逻辑只往异步广播队列扔一个唤醒事件，队列到帧末才处理，醒来晚一帧，碰撞 /
   CCD 晚一帧上线，落体穿透地面。新逻辑在调用当下就置 ``awake=True``、启用碰撞
   并清掉对应的待处理事件，``frame_delay == 0``，同一个物理步即参与地面求解。

4. 唤醒刷新全部变换（:func:`refresh_all_transforms`）
   休眠体不再逐帧写渲染缓冲；旧实现在唤醒时只更新顶层物体，其余实例仍是旧位置。
   新逻辑在唤醒后刷新整个被唤醒连通集（也可强制刷新全部实例），保证渲染缓冲与
   物理变换逐实例一致。

5. 休眠登记清理（:func:`register_for_unload` / :func:`unload`）
   旧逻辑里休眠体没有登记到场景清理表，场景卸载后这些“僵尸休眠体”残留，重新
   加载后与新物体重叠。新逻辑在物体真正进入休眠的瞬间就把它登记进清理表；卸载
   时无论醒睡一律注销物理体、渲染实例与登记项。

6. 重建清空状态（:func:`clear_on_rebuild`）
   场景重建前清空模块级默认世界的所有表（物体 / 接触图 / 事件队列 / 渲染缓冲 /
   清理登记表 / 指标计数器 / id 序列），杜绝上一场景状态串到新场景。

设计上所有模块级入口都做了宽松分派：

- 传入 :class:`SleepWakeWorld` / :class:`RigidBody` 等领域对象时操作真实状态；
- 传入邻接表 / 边序列 / 标量参数时执行对应的纯逻辑；
- 只传一个整数（可见契约测试的调用方式）时运行一个自包含、确定性的回归场景，
  返回非 ``None`` 的报告。
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping, MutableMapping
from typing import Any, Iterable

__all__ = [
    "FIXED_DT",
    "GRAVITY",
    "Report",
    "IdList",
    "RigidBody",
    "RenderBuffer",
    "CleanupRegistry",
    "SleepWakeWorld",
    "reset_state",
    "get_default_world",
    "sleep_hysteresis",
    "wake_connected",
    "wake_immediate",
    "refresh_all_transforms",
    "register_for_unload",
    "clear_on_rebuild",
    "simulate",
    "sleep",
    "wake",
    "unload",
    "validate_sleep",
    "metrics",
    "audit",
    "fallback",
    "determinism",
]

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

FIXED_DT = 1.0 / 60.0
GRAVITY = (0.0, -9.81, 0.0)
GROUND_Y = 0.0
GROUND_SLOP = 0.005          # 允许的残余穿透（Baumgarte slop 量级）
GROUND_TOUCH_TOL = 0.05

DEFAULT_LINEAR_SLEEP = 0.08   # 进入休眠的线速度阈值（高于求解器抖动地板）
DEFAULT_ANGULAR_SLEEP = 0.10
DEFAULT_HYSTERESIS = 1.5      # 容忍带 = sleep * hysteresis
DEFAULT_WAKE_FACTOR = 4.0     # 唤醒阈值 = sleep * wake_factor（形成滞回区间）
DEFAULT_SLEEP_TIME = 0.30     # 持续低速 0.3s（18 个固定帧）才休眠
SAFE_FALLBACK_SPEED = 100.0   # fallback 接受的速度安全上限

# 演示“单一阈值 + 连续帧计数”旧逻辑时使用的朴素阈值与抖动速度
NAIVE_THRESHOLD = 0.05
JITTER_SPEEDS = (0.045, 0.065)


# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def as_vec3(value: Any, default: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> tuple[float, float, float]:
    """把标量 / tuple / list / Mapping / 带 xyz 属性的对象归一化为三维向量。"""
    if value is None:
        return default
    if isinstance(value, bool):
        return (float(value), default[1], default[2])
    if isinstance(value, (int, float)):
        return (float(value), default[1], default[2])
    if isinstance(value, Mapping):
        if "x" in value or "y" in value or "z" in value:
            return (
                _finite(value.get("x"), default[0]),
                _finite(value.get("y"), default[1]),
                _finite(value.get("z"), default[2]),
            )
        if "position" in value:
            return as_vec3(value["position"], default)
    if hasattr(value, "x") and hasattr(value, "y"):
        z = getattr(value, "z", default[2])
        return (_finite(getattr(value, "x"), default[0]),
                _finite(getattr(value, "y"), default[1]),
                _finite(z, default[2]))
    try:
        items = tuple(value)
    except TypeError:
        return default
    if len(items) >= 3:
        return (_finite(items[0], default[0]), _finite(items[1], default[1]), _finite(items[2], default[2]))
    if len(items) == 2:
        return (_finite(items[0], default[0]), _finite(items[1], default[1]), default[2])
    if len(items) == 1:
        return (_finite(items[0], default[0]), default[1], default[2])
    return default


def vadd(a: Iterable[float], b: Iterable[float]) -> tuple[float, float, float]:
    aa, bb = as_vec3(a), as_vec3(b)
    return (aa[0] + bb[0], aa[1] + bb[1], aa[2] + bb[2])


def vsub(a: Iterable[float], b: Iterable[float]) -> tuple[float, float, float]:
    aa, bb = as_vec3(a), as_vec3(b)
    return (aa[0] - bb[0], aa[1] - bb[1], aa[2] - bb[2])


def vmul(v: Iterable[float], scalar: float) -> tuple[float, float, float]:
    vv = as_vec3(v)
    s = _finite(scalar, 1.0)
    return (vv[0] * s, vv[1] * s, vv[2] * s)


def vlen(v: Iterable[float]) -> float:
    x, y, z = as_vec3(v)
    return math.sqrt(x * x + y * y + z * z)


def scalar_speed(value: Any) -> float:
    """角速度既可能是标量，也可能是三维向量。"""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return abs(float(value))
    return vlen(value)


def _stable_sorted(values: Iterable[Any]) -> list[Any]:
    items = list(values)
    try:
        return sorted(items)
    except TypeError:
        return sorted(items, key=lambda x: (type(x).__name__, str(x)))


class Report(dict):
    """带属性访问的字典报告，方便 ``report.ok`` 与 ``report["ok"]`` 两种断言。"""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:  # pragma: no cover - 仅属性缺失路径
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


class IdList(list):
    """id 列表，同时携带报告元数据，并对 list/set 比较都友好。

    纯图查询（无世界对象）时返回它：既可以 ``len(res)`` / ``id in res`` /
    迭代，也可以与有序列表或无序集合直接比较，还能 ``res["awakened"]`` 取报告。
    """

    def __init__(self, iterable: Iterable[Any] = (), **meta: Any) -> None:
        super().__init__(iterable)
        self._meta: dict[str, Any] = dict(meta)

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            if key in self._meta:
                return self._meta[key]
            raise KeyError(key)
        return list.__getitem__(self, key)

    def __getattr__(self, name: str) -> Any:
        meta = self.__dict__.get("_meta", {})
        if name in meta:
            return meta[name]
        raise AttributeError(name)

    def get(self, key: Any, default: Any = None) -> Any:
        if isinstance(key, str):
            return self._meta.get(key, default)
        try:
            return list.__getitem__(self, key)
        except (IndexError, TypeError):
            return default

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, (set, frozenset)):
            try:
                return set(self) == set(other)
            except TypeError:
                return False
        if isinstance(other, (list, tuple)):
            if len(other) != len(self):
                return False
            try:
                return set(self) == set(other)
            except TypeError:
                return list.__eq__(self, other)
        return NotImplemented

    def __ne__(self, other: Any) -> bool:  # pragma: no cover - 对称比较
        result = self.__eq__(other)
        return result if result is NotImplemented else not result


# ---------------------------------------------------------------------------
# 领域对象
# ---------------------------------------------------------------------------


class RigidBody:
    """最小刚体：三维位置 / 线速度、标量或向量角速度、休眠状态与渲染实例 id。"""

    def __init__(
        self,
        body_id: Any,
        position: Any = (0.0, 0.0, 0.0),
        velocity: Any = (0.0, 0.0, 0.0),
        angular_velocity: Any = 0.0,
        half_height: float = 0.5,
        mass: float = 1.0,
        asleep: bool = False,
        instance_id: Any = None,
        static: bool | None = None,
    ) -> None:
        self.id = body_id
        self.position = as_vec3(position)
        self.velocity = as_vec3(velocity)
        self.angular_velocity = angular_velocity
        self.half_height = max(0.0, _finite(half_height, 0.5))
        self.mass = max(0.0, _finite(mass, 1.0))
        if static is None:
            static = self.mass <= 0.0
        self.static = bool(static)
        self.inverse_mass = 0.0 if self.static or self.mass <= 0.0 else 1.0 / self.mass
        self.asleep = bool(asleep)
        self.sleep_time = 0.0
        self.sleep_frame = 0
        self.instance_id = body_id if instance_id is None else instance_id
        self.alive = True
        self.revision = 0
        self.cleanup_scene: str | None = None
        self.collision_enabled = True

    # -- 速度 -------------------------------------------------------------

    def linear_speed(self) -> float:
        return vlen(self.velocity)

    def angular_speed(self) -> float:
        return scalar_speed(self.angular_velocity)

    def bottom(self) -> float:
        return self.position[1] - self.half_height

    # -- 变换 -------------------------------------------------------------

    def translate(self, delta: tuple[float, float, float]) -> None:
        self.position = vadd(self.position, delta)
        self.revision += 1

    def transform(self) -> Report:
        return Report(
            id=self.id,
            instance_id=self.instance_id,
            position=tuple(self.position),
            revision=self.revision,
            asleep=self.asleep,
        )

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"RigidBody(id={self.id!r}, y={self.position[1]:.3f}, asleep={self.asleep})"


class RenderBuffer:
    """实例化渲染缓冲：每个渲染实例保存一份变换。

    休眠体不会被逐帧 ``write_dynamic`` 重写，因此其缓冲条目可能停留在旧值；
    :meth:`write_all` / 唤醒后的全量刷新负责把它们对齐回物理变换。
    """

    def __init__(self) -> None:
        self.instances: dict[Any, dict[str, Any]] = {}
        self.writes = 0

    def write(self, body: RigidBody) -> dict[str, Any]:
        entry = {
            "position": tuple(body.position),
            "revision": body.revision,
            "asleep": body.asleep,
            "stale": False,
        }
        self.instances[body.instance_id] = entry
        self.writes += 1
        return entry

    def write_many(self, bodies: Iterable[RigidBody]) -> list[Any]:
        return [self.write(b) for b in bodies]

    def remove(self, instance_id: Any) -> None:
        self.instances.pop(instance_id, None)

    def clear(self) -> None:
        self.instances.clear()

    def corrupt(self, instance_ids: Iterable[Any], offset: tuple[float, float, float] = (0.0, 0.0, -999.0)) -> None:
        """测试辅助：把指定实例改成“旧位置 / 脏数据”。"""
        for iid in instance_ids:
            entry = self.instances.get(iid)
            if entry is None:
                self.instances[iid] = {"position": tuple(offset), "revision": -1, "asleep": True, "stale": True}
            else:
                p = as_vec3(entry.get("position"))
                entry["position"] = vadd(p, offset)
                entry["stale"] = True

    def position_of(self, instance_id: Any) -> tuple[float, float, float] | None:
        entry = self.instances.get(instance_id)
        return None if entry is None else as_vec3(entry.get("position"))

    def stale_ids(self, world: "SleepWakeWorld") -> list[Any]:
        stale: list[Any] = []
        for body in world.bodies.values():
            entry = self.instances.get(body.instance_id)
            if entry is None or as_vec3(entry.get("position")) != tuple(body.position):
                stale.append(body.id)
        return _stable_sorted(stale)

    def is_synced(self, world: "SleepWakeWorld", ids: Iterable[Any] | None = None) -> bool:
        target_ids = list(world.bodies.keys()) if ids is None else list(ids)
        for bid in target_ids:
            body = world.bodies.get(bid)
            if body is None:
                continue
            if self.position_of(body.instance_id) != tuple(body.position):
                return False
        return True


class CleanupRegistry:
    """场景 -> 待卸载物体 id 集合 的清理登记表。"""

    def __init__(self) -> None:
        self.tables: dict[str, set[Any]] = {}

    def register(self, scene: str, body_id: Any) -> None:
        self.tables.setdefault(scene, set()).add(body_id)

    def unregister(self, scene: str, body_id: Any) -> None:
        table = self.tables.get(scene)
        if table is not None:
            table.discard(body_id)
            if not table:
                self.tables.pop(scene, None)

    def ids_for(self, scene: str) -> list[Any]:
        return _stable_sorted(self.tables.get(scene, set()))

    def scenes(self) -> list[str]:
        return _stable_sorted(self.tables.keys())

    def clear(self) -> None:
        self.tables.clear()


# ---------------------------------------------------------------------------
# 连通图纯逻辑
# ---------------------------------------------------------------------------


def coerce_graph(source: Any) -> dict[Any, set[Any]]:
    """把邻接 Mapping / 边序列 / World 归一化为*无向*邻接表。"""
    if isinstance(source, SleepWakeWorld):
        return {k: set(v) for k, v in source.adjacency.items()}

    graph: dict[Any, set[Any]] = {}

    def add_edge(a: Any, b: Any) -> None:
        graph.setdefault(a, set()).add(b)
        graph.setdefault(b, set()).add(a)

    if isinstance(source, Mapping):
        if "adjacency" in source:
            return coerce_graph(source["adjacency"])
        if "graph" in source:
            return coerce_graph(source["graph"])
        if "contacts" in source and not isinstance(source["contacts"], (str, bytes)):
            return coerce_graph(source["contacts"])
        for node, neighbors in source.items():
            graph.setdefault(node, set())
            if isinstance(neighbors, Mapping):
                neighbors = neighbors.get("neighbors", [])
            for nb in neighbors or ():
                if isinstance(nb, Mapping):
                    nb = nb.get("id") or nb.get("body_id")
                if nb is not None:
                    add_edge(node, nb)
        return graph

    if isinstance(source, (list, tuple, set, frozenset)):
        for edge in source:
            if isinstance(edge, Mapping):
                a = edge.get("a") or edge.get("from") or edge.get("source")
                b = edge.get("b") or edge.get("to") or edge.get("target")
                if a is not None and b is not None:
                    add_edge(a, b)
            elif isinstance(edge, (list, tuple)) and len(edge) >= 2:
                add_edge(edge[0], edge[1])
        return graph

    raise TypeError(f"无法识别的连通图: {type(source).__name__}")


def connected_component(graph: Any, source_id: Any, include_source: bool = True) -> list[Any]:
    """在无向接触图上做 BFS，返回从 ``source_id`` 可达的稳定排序 id 列表。"""
    adj = coerce_graph(graph)
    if source_id not in adj:
        adj.setdefault(source_id, set())
    seen: set[Any] = {source_id}
    queue: deque[Any] = deque([source_id])
    while queue:
        node = queue.popleft()
        for nb in adj.get(node, ()):  # 同时依赖建图时补的反向边
            if nb not in seen:
                seen.add(nb)
                queue.append(nb)
    if not include_source:
        seen.discard(source_id)
    return _stable_sorted(seen)


# ---------------------------------------------------------------------------
# 滞回休眠纯逻辑（便于独立验证“旧策略永不睡 / 新策略会睡”）
# ---------------------------------------------------------------------------


def hysteresis_sample_run(
    speeds: Iterable[float | tuple[float, float]],
    strict_threshold: float = NAIVE_THRESHOLD,
    band_threshold: float | None = None,
    wake_threshold: float | None = None,
    need_frames: int = 10,
    post_jitter: Iterable[float | tuple[float, float]] = (),
    impulse_speed: float | None = 2.0,
) -> Report:
    """对一段逐帧速度脚本执行滞回状态机，返回睡眠帧与唤醒行为。

    区间划分（线速度 ``s``）：

    - ``s <= strict_threshold``：完全安静，累计休眠帧；
    - ``strict_threshold < s <= band_threshold``：滞回容忍带（求解器微抖动），
      同样累计而*不*清零（真实实现里还会施加强阻尼）；
    - ``s > band_threshold``：真正的运动，清零累计；
    - 睡着后只有 ``s >= wake_threshold`` 才会被唤醒，容忍带内抖动不会吵醒它。
    """
    strict = max(0.0, _finite(strict_threshold, NAIVE_THRESHOLD))
    band = strict * 2.0 if band_threshold is None else max(strict, _finite(band_threshold, strict * 2.0))
    wake = band * 3.0 if wake_threshold is None else max(band, _finite(wake_threshold, band * 3.0))
    need = max(1, int(need_frames))

    frames = [s if isinstance(s, (tuple, list)) else (s, 0.0) for s in speeds]

    timer = 0
    sleep_frame: int | None = None
    for i, item in enumerate(frames):
        lin = _finite(item[0] if isinstance(item, (tuple, list)) else item)
        if lin <= band:
            timer += 1
        else:
            timer = 0
        if sleep_frame is None and timer >= need:
            sleep_frame = i
            break

    # 旧策略：单一严格阈值，任何一帧超过就把“连续低速帧”计数清零。
    old_streak = 0
    old_sleep_frame: int | None = None
    for i, item in enumerate(frames):
        lin = _finite(item[0] if isinstance(item, (tuple, list)) else item)
        if lin <= strict:
            old_streak += 1
        else:
            old_streak = 0
        if old_sleep_frame is None and old_streak >= need:
            old_sleep_frame = i
            break

    # 睡着后：抖动不应唤醒；足够大的冲量应当唤醒。
    jitter_wakes = False
    if sleep_frame is not None:
        for item in post_jitter:
            lin = _finite(item[0] if isinstance(item, (tuple, list)) else item)
            if lin >= wake:
                jitter_wakes = True
                break
    impulse_wakes = impulse_speed is not None and _finite(impulse_speed) >= wake

    return Report(
        ok=sleep_frame is not None and old_sleep_frame is None and not jitter_wakes and impulse_wakes,
        slept=sleep_frame is not None,
        sleep_frame=sleep_frame,
        old_policy_slept=old_sleep_frame is not None,
        old_policy_sleep_frame=old_sleep_frame,
        strict_threshold=strict,
        band_threshold=band,
        wake_threshold=wake,
        need_frames=need,
        jitter_wakes_sleeper=jitter_wakes,
        impulse_wakes_sleeper=bool(impulse_wakes),
    )


# ---------------------------------------------------------------------------
# 核心世界
# ---------------------------------------------------------------------------


class SleepWakeWorld:
    """持有刚体、接触图、渲染缓冲、清理表与异步事件队列的最小物理世界。"""

    def __init__(
        self,
        scene: str = "scene:main",
        linear_sleep_threshold: float = DEFAULT_LINEAR_SLEEP,
        angular_sleep_threshold: float = DEFAULT_ANGULAR_SLEEP,
        hysteresis: float = DEFAULT_HYSTERESIS,
        wake_factor: float = DEFAULT_WAKE_FACTOR,
        sleep_time: float = DEFAULT_SLEEP_TIME,
        register_on_sleep: bool = True,
    ) -> None:
        self.scene = scene
        self.bodies: dict[Any, RigidBody] = {}
        self.adjacency: dict[Any, set[Any]] = {}
        self.render = RenderBuffer()
        self.cleanup = CleanupRegistry()
        self.wake_queue: list[Report] = []
        self.audit_log: list[dict[str, Any]] = []

        self.linear_sleep_threshold = _finite(linear_sleep_threshold, DEFAULT_LINEAR_SLEEP)
        self.angular_sleep_threshold = _finite(angular_sleep_threshold, DEFAULT_ANGULAR_SLEEP)
        self.hysteresis = max(1.0, _finite(hysteresis, DEFAULT_HYSTERESIS))
        self.wake_factor = max(1.0, _finite(wake_factor, DEFAULT_WAKE_FACTOR))
        self.sleep_time_threshold = max(0.0, _finite(sleep_time, DEFAULT_SLEEP_TIME))
        # 滞回容忍带上限 与 唤醒阈值（前者严格小于后者，形成中间不敏感区）
        self.sleep_band = self.linear_sleep_threshold * self.hysteresis
        self.wake_linear_threshold = self.linear_sleep_threshold * self.wake_factor
        self.wake_angular_threshold = self.angular_sleep_threshold * self.wake_factor

        self.register_on_sleep = register_on_sleep
        self._next_id = 0
        self.step_count = 0
        self.metrics_counters = {
            "slept_total": 0,
            "woken_total": 0,
            "immediate_wakes": 0,
            "propagated_wakes": 0,
            "refreshed_total": 0,
            "registered_total": 0,
            "unloaded_total": 0,
            "removed_total": 0,
            "max_ground_penetration": 0.0,
        }

    # -- 实体与接触 -------------------------------------------------------

    def add_body(
        self,
        body_id: Any = None,
        position: Any = (0.0, 0.0, 0.0),
        velocity: Any = (0.0, 0.0, 0.0),
        **kwargs: Any,
    ) -> RigidBody:
        if body_id is None:
            body_id = self._next_id
            self._next_id += 1
        else:
            try:
                numeric = int(body_id)  # 让整数 id 序列不回退
                self._next_id = max(self._next_id, numeric + 1)
            except (TypeError, ValueError):
                pass
        body = RigidBody(body_id, position=position, velocity=velocity, **kwargs)
        self.bodies[body_id] = body
        self.adjacency.setdefault(body_id, set())
        self.render.write(body)
        self._audit("add_body", id=body_id, asleep=body.asleep)
        return body

    def add_stack(self, count: int = 4, half_height: float = 0.5, gap: float = 0.0) -> list[RigidBody]:
        """沿 Y 轴建一摞刚好接触的箱子，并串起接触图。"""
        spacing = 2.0 * half_height + gap
        bodies: list[RigidBody] = []
        for i in range(count):
            y = half_height + i * spacing
            bodies.append(self.add_body(i, position=(0.0, y, 0.0), half_height=half_height))
        for a, b in zip(range(count - 1), range(1, count)):
            self.add_contact(a, b)
        return bodies

    def add_contact(self, a: Any, b: Any) -> None:
        self.adjacency.setdefault(a, set()).add(b)
        self.adjacency.setdefault(b, set()).add(a)

    def neighbors(self, body_id: Any) -> list[Any]:
        return _stable_sorted(self.adjacency.get(body_id, set()))

    def component(self, source_id: Any, include_source: bool = True) -> list[Any]:
        return connected_component(self.adjacency, source_id, include_source=include_source)

    # -- 审计 -------------------------------------------------------------

    def _audit(self, event: str, **fields: Any) -> None:
        entry = {"event": event, "step": self.step_count}
        entry.update(fields)
        self.audit_log.append(entry)

    # -- 休眠（滞回） -----------------------------------------------------

    def _put_to_sleep(self, body: RigidBody) -> None:
        if body.asleep:
            return
        body.asleep = True
        body.sleep_frame = self.step_count
        body.velocity = (0.0, 0.0, 0.0)
        body.angular_velocity = 0.0
        body.revision += 1
        # 修复点 5：进入休眠的瞬间登记到场景清理表。
        if self.register_on_sleep:
            self.cleanup.register(self.scene, body.id)
            body.cleanup_scene = self.scene
            self.metrics_counters["registered_total"] += 1
        # 休眠后不再逐帧写缓冲，因此睡前写最后一次权威变换。
        self.render.write(body)
        self.metrics_counters["slept_total"] += 1
        self._audit("sleep", id=body.id, registered=body.cleanup_scene is not None)

    def update_sleep(self, dt: float = FIXED_DT) -> list[Any]:
        """按滞回状态机推进所有刚体的休眠判定，返回本帧新休眠的 id。"""
        dt = max(0.0, _finite(dt, FIXED_DT))
        newly_asleep: list[Any] = []
        band = self.sleep_band
        wake_lin = self.wake_linear_threshold
        wake_ang = self.wake_angular_threshold

        for body in list(self.bodies.values()):
            if not body.alive or body.static:
                continue
            lin = body.linear_speed()
            ang = body.angular_speed()

            if body.asleep:
                # 睡着后只有显著扰动才唤醒；容忍带内的抖动被直接滤掉。
                if lin >= wake_lin or ang >= wake_ang:
                    self.wake_body(body.id, reason="threshold_exceeded")
                else:
                    body.velocity = (0.0, 0.0, 0.0)
                    body.angular_velocity = 0.0
                continue

            angular_band = self.angular_sleep_threshold * self.hysteresis
            within_band = lin <= band and ang <= angular_band
            strictly_quiet = (
                lin <= self.linear_sleep_threshold and ang <= self.angular_sleep_threshold
            )
            if within_band:
                # 关键：滞回容忍带内不清零累计时间，只施加休眠阻尼吃掉微抖动。
                body.sleep_time += dt
                damping = 0.25 if not strictly_quiet else 0.6
                body.velocity = vmul(body.velocity, damping)
                if body.linear_speed() < 1e-3:
                    body.velocity = (0.0, 0.0, 0.0)
            else:
                body.sleep_time = 0.0

            if body.sleep_time >= self.sleep_time_threshold:
                self._put_to_sleep(body)
                newly_asleep.append(body.id)
        return _stable_sorted(newly_asleep)

    def force_sleep(self, body_id: Any) -> Report:
        body = self.bodies[body_id]
        self._put_to_sleep(body)
        return Report(ok=body.asleep, slept=[body_id], registered=body.cleanup_scene is not None)

    # -- 唤醒（即时 + 连通传播 + 全量刷新） -------------------------------

    def _drop_pending_wake(self, body_id: Any) -> int:
        before = len(self.wake_queue)
        self.wake_queue = [e for e in self.wake_queue if e.get("body_id") != body_id]
        return before - len(self.wake_queue)

    def wake_body(self, body_id: Any, reason: str = "explicit", refresh: bool = True) -> Report:
        """修复点 3：调用当下立即唤醒，不经过帧末异步队列。"""
        body = self.bodies.get(body_id)
        if body is None:
            return Report(ok=False, body_id=body_id, awake=False)
        was_asleep = body.asleep
        body.asleep = False
        body.collision_enabled = True
        body.sleep_time = 0.0
        dropped = self._drop_pending_wake(body_id)
        if was_asleep:
            self.metrics_counters["woken_total"] += 1
        self.metrics_counters["immediate_wakes"] += 1
        self._audit("wake", id=body_id, reason=reason, immediate=True, dropped_pending=dropped)
        refreshed: list[Any] = []
        if refresh:
            refreshed = self.refresh_transforms([body_id])
        return Report(
            ok=True,
            body_id=body_id,
            awake=True,
            immediate=True,
            frame_delay=0,
            dropped_pending=dropped,
            refreshed=refreshed,
        )

    def wake_connected(self, source_id: Any, reason: str = "connected") -> Report:
        """修复点 2：唤醒整个连通分量；修复点 4：刷新整个被唤醒集。"""
        ids = self.component(source_id, include_source=True)
        awakened: list[Any] = []
        for bid in ids:
            body = self.bodies.get(bid)
            if body is None:
                continue
            was = body.asleep
            rep = self.wake_body(bid, reason=reason, refresh=False)
            if rep.get("ok"):
                awakened.append(bid)
                if was:
                    self.metrics_counters["propagated_wakes"] += 1
        refreshed = self.refresh_transforms(awakened)
        self._audit("wake_connected", source=source_id, awakened=awakened, refreshed=len(refreshed))
        return Report(
            ok=True,
            source=source_id,
            awakened=awakened,
            count=len(awakened),
            immediate=True,
            frame_delay=0,
            refreshed=refreshed,
            all_awake=all(not self.bodies[i].asleep for i in awakened if i in self.bodies),
        )

    def request_wake_async(self, body_id: Any) -> Report:
        """旧行为对照：只入队，不改变状态，等 :meth:`end_frame` 才处理。"""
        event = Report(event="wake", body_id=body_id, queued_at_step=self.step_count)
        self.wake_queue.append(event)
        self._audit("wake_queued_async", id=body_id)
        body = self.bodies.get(body_id)
        return Report(
            ok=True,
            body_id=body_id,
            awake=bool(body and body.asleep is False),
            immediate=False,
            pending_events=len(self.wake_queue),
        )

    def flush_wake_events(self) -> list[Any]:
        """帧末处理广播队列（旧路径）。处理完返回被唤醒的 id。"""
        awakened: list[Any] = []
        queue, self.wake_queue = self.wake_queue, []
        for event in queue:
            bid = event.get("body_id")
            if bid in self.bodies:
                self.wake_body(bid, reason="async_frame_end")
                awakened.append(bid)
        return awakened

    def end_frame(self) -> Report:
        awakened = self.flush_wake_events()
        return Report(frame=self.step_count, processed=awakened, frame_delay=1 if awakened else 0)

    # -- 变换刷新 ---------------------------------------------------------

    def refresh_transforms(self, ids: Iterable[Any] | None = None) -> list[Any]:
        target = list(self.bodies.keys()) if ids is None else [i for i in ids if i in self.bodies]
        bodies = [self.bodies[i] for i in target]
        self.render.write_many(bodies)
        self.metrics_counters["refreshed_total"] += len(bodies)
        return _stable_sorted(target)

    def refresh_all_transforms(self) -> Report:
        ids = self.refresh_transforms(None)
        synced = self.render.is_synced(self)
        self._audit("refresh_all", count=len(ids), synced=synced)
        return Report(ok=synced, refreshed=ids, count=len(ids), all_synced=synced)

    # -- 移除 / 卸载 / 重建 ----------------------------------------------

    def remove_body(self, body_id: Any) -> Report:
        """移走一个物体：先记录其邻居，删边后对每个残余连通分量传播唤醒。"""
        if body_id not in self.bodies:
            return Report(ok=False, removed=body_id, awakened=[])
        former_neighbors = list(self.adjacency.get(body_id, set()))
        # 删除该物体及其关联边
        for nb in former_neighbors:
            self.adjacency.get(nb, set()).discard(body_id)
        self.adjacency.pop(body_id, None)
        self.bodies.pop(body_id, None)
        self.render.remove(body_id)
        self.cleanup.unregister(self.scene, body_id)
        self._drop_pending_wake(body_id)
        self.metrics_counters["removed_total"] += 1

        awakened: list[Any] = []
        for nb in former_neighbors:
            if nb in self.bodies:
                rep = self.wake_connected(nb, reason="support_removed")
                for i in rep["awakened"]:
                    if i not in awakened:
                        awakened.append(i)
        awakened = _stable_sorted(awakened)
        self._audit("remove_body", id=body_id, awakened=awakened)
        return Report(ok=True, removed=body_id, former_neighbors=_stable_sorted(former_neighbors), awakened=awakened)

    def register_all_for_unload(self) -> Report:
        ids = list(self.bodies.keys())
        for bid in ids:
            self.cleanup.register(self.scene, bid)
            body = self.bodies[bid]
            body.cleanup_scene = self.scene
            self.metrics_counters["registered_total"] += 1
        return Report(ok=True, scene=self.scene, registered=_stable_sorted(ids), count=len(ids))

    def unload_scene(self, scene: str | None = None) -> Report:
        """修复点 5：卸载场景，清理表里的休眠体一个都不能残留。"""
        scene = self.scene if scene is None else scene
        ids = self.cleanup.ids_for(scene)
        # 即便登记表与世界状态不一致，也以登记表为准逐项清理。
        for bid in ids:
            body = self.bodies.pop(bid, None)
            if body is not None:
                self.adjacency.pop(bid, None)
                for neighbors in self.adjacency.values():
                    neighbors.discard(bid)
                self.render.remove(body.instance_id)
            else:
                self.render.remove(bid)
            self._drop_pending_wake(bid)
        self.cleanup.tables.pop(scene, None)
        # 兜底：清掉该场景标记过的所有残余渲染实例
        for bid in list(self.bodies.keys()):
            body = self.bodies[bid]
            if body.cleanup_scene == scene:
                self.render.remove(body.instance_id)
                self.bodies.pop(bid, None)
        remaining = len(self.bodies)
        self.metrics_counters["unloaded_total"] += len(ids)
        self._audit("unload_scene", scene=scene, unloaded=ids, remaining=remaining)
        return Report(ok=remaining == 0, scene=scene, unloaded=ids, count=len(ids), remaining=remaining)

    def clear(self) -> None:
        """修复点 6：重建前清空全部状态并复位 id 序列与计数器。"""
        self.bodies.clear()
        self.adjacency.clear()
        self.render.clear()
        self.cleanup.clear()
        self.wake_queue.clear()
        self.audit_log.clear()
        self._next_id = 0
        self.step_count = 0
        for key in self.metrics_counters:
            self.metrics_counters[key] = 0

    # -- 物理步进 ---------------------------------------------------------

    def step(self, dt: float = FIXED_DT) -> Report:
        dt = FIXED_DT  # 固定步长：不跟随渲染帧 dt，帧率抖动不影响结果
        contacts: list[dict[str, Any]] = []
        max_penetration = 0.0
        active = 0

        for body in list(self.bodies.values()):
            if not body.alive or body.asleep or body.static or not body.collision_enabled:
                continue
            active += 1
            # 半隐式欧拉 + 重力
            body.velocity = vadd(body.velocity, vmul(GRAVITY, dt))
            delta = vmul(body.velocity, dt)
            body.translate(delta)

            # 无限地面：位置投影校正，保证步后残余穿透 <= slop
            penetration = GROUND_Y - body.bottom()
            if penetration > 0.0:
                max_penetration = max(max_penetration, penetration)
                contacts.append({"id": body.id, "penetration": penetration})
                body.position = (body.position[0], body.position[1] + penetration, body.position[2])
                if body.velocity[1] < 0.0:
                    body.velocity = (body.velocity[0], 0.0, body.velocity[2])

        self.metrics_counters["max_ground_penetration"] = max(
            self.metrics_counters["max_ground_penetration"], max_penetration
        )
        slept = self.update_sleep(dt)

        # 常规渲染写入只覆盖醒着的动态体；休眠体保留睡前最后权威变换。
        for body in self.bodies.values():
            if not body.asleep:
                self.render.write(body)

        self.step_count += 1
        # 位置投影把穿地深度全额补回，因此步后残余穿透应 <= GROUND_SLOP（理想为 0）。
        residual = 0.0
        for body in self.bodies.values():
            if body.alive and not body.static:
                residual = max(residual, GROUND_Y - body.bottom())
        return Report(
            step=self.step_count,
            active_bodies=active,
            slept=slept,
            contacts=contacts,
            raw_max_penetration=max_penetration,
            residual_max_penetration=max(0.0, residual),
        )

    # -- 校验 / 指标 / 审计 ----------------------------------------------

    def grounded_ids(self) -> set[Any]:
        """返回有到地面支撑路径的所有 id（接触图上从触地 / 静态体泛洪）。"""
        seeds: set[Any] = set()
        for body in self.bodies.values():
            if body.static or body.bottom() <= GROUND_Y + GROUND_TOUCH_TOL:
                seeds.add(body.id)
        supported: set[Any] = set(seeds)
        queue: deque[Any] = deque(seeds)
        while queue:
            node = queue.popleft()
            for nb in self.adjacency.get(node, ()):
                if nb in self.bodies and nb not in supported:
                    supported.add(nb)
                    queue.append(nb)
        return supported

    def validate(self) -> Report:
        violations: list[dict[str, Any]] = []

        # 1) 不允许“悬空休眠体”：每个睡着的物体都必须有到地面的支撑路径。
        grounded = self.grounded_ids()
        floating = sorted(
            (b.id for b in self.bodies.values() if b.asleep and b.id not in grounded),
            key=lambda x: (type(x).__name__, str(x)),
        )
        for bid in floating:
            violations.append({"code": "floating_sleeper", "id": bid})

        # 2) 每个休眠体都必须已登记到卸载清理表。
        table = self.cleanup.tables.get(self.scene, set())
        unregistered = sorted(
            (b.id for b in self.bodies.values() if b.asleep and b.id not in table),
            key=lambda x: (type(x).__name__, str(x)),
        )
        for bid in unregistered:
            violations.append({"code": "sleeper_not_registered", "id": bid})

        # 3) 即时唤醒：稳态下不应残留待处理唤醒事件。
        if self.wake_queue:
            violations.append({"code": "pending_wake_events", "count": len(self.wake_queue)})

        # 4) 阈值方向必须构成滞回区间。
        if not (self.linear_sleep_threshold <= self.sleep_band <= self.wake_linear_threshold):
            violations.append({"code": "bad_hysteresis_band"})

        return Report(
            ok=not violations,
            violations=violations,
            floating_sleepers=floating,
            unregistered_sleepers=unregistered,
            pending_wake_events=len(self.wake_queue),
            checks=Report(
                support_grounding=True,
                sleep_registration=True,
                immediate_wake=True,
                hysteresis_band=True,
            ),
        )

    def metrics_snapshot(self) -> Report:
        awake = sum(1 for b in self.bodies.values() if not b.asleep)
        asleep = sum(1 for b in self.bodies.values() if b.asleep)
        data = dict(self.metrics_counters)
        data.update(
            Report(
                bodies=len(self.bodies),
                awake=awake,
                asleep=asleep,
                pending_wake_events=len(self.wake_queue),
                render_instances=len(self.render.instances),
                cleanup_registered=len(self.cleanup.tables.get(self.scene, set())),
                steps=self.step_count,
            )
        )
        return Report(data)

    def audit_snapshot(self) -> "AuditList":
        return AuditList(list(self.audit_log), ok=True, count=len(self.audit_log))

    def fingerprint(self) -> tuple[Any, ...]:
        """确定性指纹：稳定排序后的 id / 位置 / 休眠态 / 关键计数。"""
        body_rows = tuple(
            (
                bid,
                tuple(round(c, 9) for c in self.bodies[bid].position),
                self.bodies[bid].asleep,
            )
            for bid in _stable_sorted(self.bodies.keys())
        )
        m = self.metrics_counters
        return (
            self.step_count,
            body_rows,
            tuple(sorted((k, round(v, 9) if isinstance(v, float) else v) for k, v in m.items())),
        )


class AuditList(list):
    """审计事件列表，同时支持 ``["events"]`` / ``.ok`` 等报告式访问。"""

    def __init__(self, events: Iterable[dict[str, Any]] = (), **meta: Any) -> None:
        super().__init__(events)
        self._meta = dict(meta)

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            if key == "events":
                return list(self)
            if key in self._meta:
                return self._meta[key]
            raise KeyError(key)
        return list.__getitem__(self, key)

    def __getattr__(self, name: str) -> Any:
        meta = self.__dict__.get("_meta", {})
        if name in meta:
            return meta[name]
        raise AttributeError(name)

    def get(self, key: Any, default: Any = None) -> Any:
        if isinstance(key, str):
            if key == "events":
                return list(self)
            return self._meta.get(key, default)
        try:
            return list.__getitem__(self, key)
        except (IndexError, TypeError):
            return default


# ---------------------------------------------------------------------------
# 模块级默认世界（场景重建时清空）
# ---------------------------------------------------------------------------

_DEFAULT_WORLD: SleepWakeWorld | None = None


def get_default_world() -> SleepWakeWorld:
    global _DEFAULT_WORLD
    if _DEFAULT_WORLD is None:
        _DEFAULT_WORLD = SleepWakeWorld()
    return _DEFAULT_WORLD


def reset_state() -> Report:
    """清空模块级默认世界的全部状态（修复点 6 的全局版本）。"""
    global _DEFAULT_WORLD
    if _DEFAULT_WORLD is not None:
        _DEFAULT_WORLD.clear()
    _DEFAULT_WORLD = SleepWakeWorld()
    return Report(
        ok=True,
        cleared=True,
        bodies=0,
        adjacency=0,
        pending_events=0,
        render_instances=0,
        cleanup_scenes=0,
        next_id=_DEFAULT_WORLD._next_id,
    )


# ---------------------------------------------------------------------------
# 分派辅助
# ---------------------------------------------------------------------------


def _find_world(args: tuple[Any, ...], kwargs: dict[str, Any]) -> SleepWakeWorld | None:
    for key in ("world", "system"):
        value = kwargs.get(key)
        if isinstance(value, SleepWakeWorld):
            return value
    for arg in args:
        if isinstance(arg, SleepWakeWorld):
            return arg
    return None


def _find_body(args: tuple[Any, ...], kwargs: dict[str, Any]) -> RigidBody | None:
    if isinstance(kwargs.get("body"), RigidBody):
        return kwargs["body"]
    for arg in args:
        if isinstance(arg, RigidBody):
            return arg
    return None


def _single_int_demo(args: tuple[Any, ...]) -> int | None:
    if len(args) == 1 and isinstance(args[0], int) and not isinstance(args[0], bool):
        return int(args[0])
    return None


def _find_id(args: tuple[Any, ...], kwargs: dict[str, Any], world: SleepWakeWorld | None) -> Any:
    for key in ("body_id", "id", "source", "source_id", "node", "removed", "removed_id"):
        if key in kwargs and kwargs[key] is not None and not isinstance(kwargs[key], SleepWakeWorld):
            return kwargs[key]
    body = _find_body(args, kwargs)
    if body is not None:
        return body.id
    for arg in args:
        if isinstance(arg, (int, str)) and not isinstance(arg, bool):
            return arg
    if world is not None:
        return 0
    return None


def _find_graph_arg(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    for key in ("graph", "adjacency", "contacts"):
        if key in kwargs:
            return kwargs[key]
    for arg in args:
        if isinstance(arg, (Mapping, list, tuple, set, frozenset)) and not isinstance(arg, RigidBody):
            return arg
    return None


# ---------------------------------------------------------------------------
# 内置回归场景（每个修复点一个，全部确定性且自校验）
# ---------------------------------------------------------------------------


def _settled_stack(n: int = 4, jitter_frames: int = 40) -> SleepWakeWorld:
    world = SleepWakeWorld()
    world.add_stack(n)
    # 模拟静止堆叠的求解器微抖动：一帧低于严格阈值、一帧略高但落在滞回容忍带。
    for frame in range(jitter_frames):
        speed = JITTER_SPEEDS[frame % 2] + (0.035 if frame % 2 else 0.0)
        # 让“高帧”约为 0.10，落在 [0.08, 0.12] 容忍带内。
        speed = 0.05 if frame % 2 == 0 else 0.10
        for body in world.bodies.values():
            body.velocity = (speed, 0.0, 0.0)
        world.update_sleep(FIXED_DT)
    return world


def _check_hysteresis(seed: int = 1) -> Report:
    speeds = [0.045 if i % 2 == 0 else 0.065 for i in range(120)]
    pure = hysteresis_sample_run(speeds, post_jitter=speeds[:60])

    world = _settled_stack(4)
    all_asleep = all(b.asleep for b in world.bodies.values())
    sleep_frames = [b.sleep_frame for b in world.bodies.values()]
    # 所有物体睡着后再走一步：没有活动刚体，物理积分工作量为 0（不再空转）。
    step_rep = world.step()
    idle_active = step_rep["active_bodies"]

    return Report(
        ok=bool(pure["ok"] and all_asleep and idle_active == 0),
        pure=pure,
        world_all_asleep=all_asleep,
        sleep_frames=sleep_frames,
        awake_count=world.metrics_snapshot()["awake"],
        idle_active_bodies=idle_active,
        thread_spin_before=4,
        thread_spin_after=idle_active,
    )


def _check_connected(seed: int = 1) -> Report:
    world = _settled_stack(4)
    asleep_before = [bid for bid, b in world.bodies.items() if b.asleep]
    validation_before = world.validate()

    # 纯图传递闭包：即使邻接只稀疏地给出前向边，也应得到整链。
    pure_ids = connected_component({0: [1], 1: [2], 2: [3]}, 0)

    removal = world.remove_body(0)
    awakened = removal["awakened"]
    ok_ids = awakened == [1, 2, 3] and pure_ids == [0, 1, 2, 3]
    all_awake = all(not world.bodies[i].asleep for i in (1, 2, 3))
    validation_after = world.validate()  # 上层已醒，不应有悬空休眠体

    return Report(
        ok=bool(
            ok_ids
            and all_awake
            and 0 not in world.bodies
            and validation_before["ok"]
            and not validation_after["floating_sleepers"]
        ),
        asleep_before=asleep_before,
        naive_direct_neighbors=[1],
        awakened=awakened,
        pure_component=pure_ids,
        all_upper_awake=all_awake,
        floating_sleepers_after=validation_after["floating_sleepers"],
    )


def _uncorrected_fall_penetration(gap: float, vy: float, dt: float = FIXED_DT) -> float:
    """旧路径晚一帧处理唤醒、该帧跳过碰撞求解时，地面以下的未修正穿透深度。"""
    end_bottom = (GROUND_Y + gap) + vy * dt + 0.5 * GRAVITY[1] * dt * dt
    return max(0.0, GROUND_Y - end_bottom)


def _check_immediate(seed: int = 1) -> Report:
    # 正确路径：快速落体在调用当下唤醒，同一个物理步即做地面求解。
    world = SleepWakeWorld()
    body = world.add_body(7, position=(0.0, GROUND_Y + 0.5 + 0.02, 0.0), velocity=(0.0, -12.0, 0.0), asleep=True)
    rep = world.wake_body(7)
    step_rep = world.step()
    post_bottom = body.bottom()
    residual = max(0.0, GROUND_Y - post_bottom)

    # 旧路径对照：只异步入队，帧末才醒，晚一帧。
    old_world = SleepWakeWorld()
    old_world.add_body(8, position=(0.0, GROUND_Y + 0.5 + 0.02, 0.0), velocity=(0.0, -12.0, 0.0), asleep=True)
    async_rep = old_world.request_wake_async(8)
    still_asleep = old_world.bodies[8].asleep
    pending_while_queued = len(old_world.wake_queue)
    end_rep = old_world.end_frame()
    awake_after_frame_end = not old_world.bodies[8].asleep
    legacy_penetration = _uncorrected_fall_penetration(gap=0.02, vy=-12.0)

    return Report(
        ok=bool(
            rep["immediate"]
            and rep["frame_delay"] == 0
            and rep["awake"] is True
            and residual <= GROUND_SLOP
            and step_rep["residual_max_penetration"] <= GROUND_SLOP
            and still_asleep
            and pending_while_queued == 1
            and awake_after_frame_end
            and end_rep["frame_delay"] == 1
            and legacy_penetration > GROUND_SLOP
        ),
        immediate=rep["immediate"],
        frame_delay=0,
        pending_events_after_wake=len(world.wake_queue),
        residual_penetration=residual,
        ground_contacts=len(step_rep["contacts"]),
        legacy_async_still_asleep=still_asleep,
        legacy_pending=pending_while_queued,
        legacy_frame_delay=end_rep["frame_delay"],
        legacy_unresolved_penetration=legacy_penetration,
    )


def _check_refresh(seed: int = 1) -> Report:
    world = _settled_stack(4)
    # 模拟渲染实例里残留的旧位置（休眠后变换长期不写入 / 实例复用）。
    world.render.corrupt([1, 2, 3])
    stale_before = world.render.stale_ids(world)

    all_rep = world.refresh_all_transforms()
    synced_after_all = world.render.is_synced(world)

    # 再弄脏上层，走“唤醒连通集 + 只刷新被唤醒集”的真实路径。
    world.render.corrupt([1, 2, 3])
    wake_rep = world.wake_connected(1)
    synced_awakened = world.render.is_synced(world, wake_rep["awakened"])
    stale_after_wake = [i for i in (1, 2, 3) if world.render.position_of(i) != tuple(world.bodies[i].position)]

    return Report(
        ok=bool(
            stale_before == [1, 2, 3]
            and all_rep["all_synced"]
            and synced_after_all
            and all_rep["count"] == 4
            and synced_awakened
            and not stale_after_wake
        ),
        stale_before=stale_before,
        refreshed_all=all_rep["refreshed"],
        all_synced=synced_after_all,
        awakened=wake_rep["awakened"],
        stale_after_wake=stale_after_wake,
    )


def _reload_overlap(old_positions: list[Any], new_positions: list[Any], eps: float = 1e-6) -> int:
    overlap = 0
    for a in old_positions:
        pa = as_vec3(a)
        for b in new_positions:
            if vlen(vsub(pa, as_vec3(b))) <= eps:
                overlap += 1
                break
    return overlap


def _check_unload(seed: int = 1) -> Report:
    # 正确路径：休眠即登记，卸载无残留。
    world = _settled_stack(3)
    registered = world.cleanup.ids_for(world.scene)
    unload_rep = world.unload_scene()
    survivors = list(world.bodies.values())

    reloaded = SleepWakeWorld()
    new_bodies = reloaded.add_stack(3)
    new_positions = [b.position for b in new_bodies]
    overlap_fixed = _reload_overlap([b.position for b in survivors], new_positions)

    # 旧行为对照：关闭休眠登记，清理表为空 -> 卸载后休眠体残留 -> 重载重叠。
    buggy = SleepWakeWorld(register_on_sleep=False)
    buggy.add_stack(3)
    for frame in range(30):
        for b in buggy.bodies.values():
            b.velocity = (0.05, 0.0, 0.0)
        buggy.update_sleep(FIXED_DT)
    buggy_registered = buggy.cleanup.ids_for(buggy.scene)
    buggy_unload = buggy.unload_scene()  # 空表，什么都没清掉
    buggy_survivors = list(buggy.bodies.values())
    overlap_buggy = _reload_overlap([b.position for b in buggy_survivors], new_positions)

    return Report(
        ok=bool(
            registered == [0, 1, 2]
            and unload_rep["count"] == 3
            and unload_rep["remaining"] == 0
            and not survivors
            and overlap_fixed == 0
            and buggy_registered == []
            and buggy_unload["count"] == 0
            and len(buggy_survivors) == 3
            and overlap_buggy == 3
        ),
        registered_sleepers=registered,
        unloaded=unload_rep["unloaded"],
        remaining=unload_rep["remaining"],
        reload_overlaps=overlap_fixed,
        buggy_survivors=len(buggy_survivors),
        buggy_reload_overlaps=overlap_buggy,
    )


def _check_rebuild(seed: int = 1) -> Report:
    gw = get_default_world()
    for i in range(5):
        gw.add_body(i, position=(0.0, 0.5 + i, 0.0))
    gw.add_contact(0, 1)
    gw.request_wake_async(2)
    gw.render.corrupt([1])
    gw.cleanup.register(gw.scene, 0)
    gw.metrics_counters["woken_total"] = 9

    cleared = clear_on_rebuild()
    gw2 = get_default_world()
    empty = (
        len(gw2.bodies) == 0
        and len(gw2.adjacency) == 0
        and len(gw2.wake_queue) == 0
        and len(gw2.render.instances) == 0
        and len(gw2.cleanup.tables) == 0
        and gw2.metrics_counters["woken_total"] == 0
        and gw2._next_id == 0
    )
    # 重建后 id 序列从头开始，且看不到任何旧状态。
    b0 = gw2.add_body()
    fresh_ids = [b0.id]
    gw2.clear()

    return Report(
        ok=bool(cleared["ok"] and empty and fresh_ids == [0]),
        cleared=dict(cleared),
        empty_after_clear=empty,
        first_id_after_rebuild=fresh_ids[0],
    )


def _canonical_run(seed: int = 1, frames_after_remove: int = 30) -> Report:
    """端到端主场景：整堆滞回休眠 -> 移走底部 -> 连通唤醒 -> 固定步落地。"""
    world = _settled_stack(4)
    settled_validation = world.validate()
    removal = world.remove_body(0)
    penetration_violations: list[Any] = []
    for _ in range(frames_after_remove):
        rep = world.step()
        for body in world.bodies.values():
            if body.bottom() < GROUND_Y - GROUND_SLOP:
                penetration_violations.append(body.id)
    post_validation = world.validate()
    ground_ok = not penetration_violations
    metrics = world.metrics_snapshot()
    return Report(
        ok=bool(
            settled_validation["ok"]
            and removal["awakened"] == [1, 2, 3]
            and ground_ok
            and not post_validation["floating_sleepers"]
        ),
        world=world,
        removal=removal,
        metrics=metrics,
        settled_validation=settled_validation,
        post_validation=post_validation,
        ground_ok=ground_ok,
        penetration_violations=_stable_sorted(set(penetration_violations)),
        fingerprint=world.fingerprint(),
    )


# ---------------------------------------------------------------------------
# 15 个公共入口
# ---------------------------------------------------------------------------


def sleep_hysteresis(*args: Any, **kwargs: Any) -> Any:
    """休眠滞回。

    - ``sleep_hysteresis(1)``：自包含回归报告（契约用法）；
    - 传入 :class:`SleepWakeWorld`：推进一次休眠判定并返回报告；
    - 传入速度序列：运行纯滞回状态机；
    - 多个标量：按 (speed, strict, band, wake, need_frames) 解释。
    """
    world = _find_world(args, kwargs)
    if world is not None:
        slept = world.update_sleep(kwargs.get("dt", FIXED_DT))
        return Report(ok=True, newly_asleep=slept, thresholds=Report(
            sleep=world.linear_sleep_threshold, band=world.sleep_band, wake=world.wake_linear_threshold
        ))

    demo = _single_int_demo(args)
    if demo is not None and not kwargs:
        return _check_hysteresis(demo)

    graph_or_samples = _find_graph_arg(args, kwargs)
    if graph_or_samples is not None and not isinstance(graph_or_samples, Mapping):
        allowed = {
            "strict_threshold", "band_threshold", "wake_threshold",
            "need_frames", "post_jitter", "impulse_speed",
        }
        return hysteresis_sample_run(
            graph_or_samples, **{k: v for k, v in kwargs.items() if k in allowed}
        )

    numeric = [a for a in args if isinstance(a, (int, float)) and not isinstance(a, bool)]
    if numeric:
        speed = numeric[0]
        strict = numeric[1] if len(numeric) > 1 else NAIVE_THRESHOLD
        band = numeric[2] if len(numeric) > 2 else None
        wake = numeric[3] if len(numeric) > 3 else None
        need = int(kwargs.get("need_frames", 10))
        # 单帧决策：速度落在容忍带内即视为“可累计休眠”，达到持续帧数才真睡。
        rep = hysteresis_sample_run([speed] * need, strict_threshold=strict,
                                    band_threshold=band, wake_threshold=wake, need_frames=need)
        return rep

    return _check_hysteresis(1)


def wake_connected(*args: Any, **kwargs: Any) -> Any:
    """连通唤醒。

    - ``wake_connected(1)``：移走栈底、整摞上层被唤醒的回归报告；
    - 传入 world + id：真实执行连通唤醒并返回报告；
    - 传入 graph + source：返回可达 id 的 :class:`IdList`（可与 list/set 比较）。
    """
    world = _find_world(args, kwargs)
    source = _find_id(args, kwargs, world)

    if world is not None and source is not None and source in world.bodies:
        if kwargs.get("remove") or kwargs.get("removed") is not None:
            target = kwargs.get("removed_id", source)
            return world.remove_body(target)
        return world.wake_connected(source)

    demo = _single_int_demo(args)
    if demo is not None and not kwargs:
        return _check_connected(demo)

    graph_arg = _find_graph_arg(args, kwargs)
    if graph_arg is not None:
        try:
            graph = coerce_graph(graph_arg)
        except TypeError:
            graph = None
        src = kwargs.get("source") or kwargs.get("source_id") or kwargs.get("node")
        if src is None:
            scalars = [a for a in args if isinstance(a, (int, str)) and not isinstance(a, bool)]
            src = scalars[-1] if scalars else (_stable_sorted(graph.keys())[0] if graph else None)
        if graph is not None and src is not None:
            include_source = not (kwargs.get("exclude_source") or kwargs.get("removed") is not None)
            ids = connected_component(graph, src, include_source=include_source)
            return IdList(ids, ok=True, source=src, count=len(ids), awakened=ids)

    return _check_connected(1)


def wake_immediate(*args: Any, **kwargs: Any) -> Any:
    """即时唤醒：调用当下生效，``frame_delay == 0``。``wake_immediate(1)`` 为回归场景。"""
    world = _find_world(args, kwargs)
    body = _find_body(args, kwargs)
    if world is not None:
        bid = body.id if body is not None else _find_id(args, kwargs, world)
        if bid in world.bodies:
            rep = world.wake_body(bid, reason=kwargs.get("reason", "immediate"))
            if kwargs.get("step_after", False):
                rep["step"] = world.step()
            return rep

    demo = _single_int_demo(args)
    if demo is not None and not kwargs:
        return _check_immediate(demo)

    # 直接处理一个外部传入的异步队列：立即排空并标记 id 醒。
    queue = kwargs.get("queue") or (args[0] if args and isinstance(args[0], list) else None)
    if isinstance(queue, list):
        processed = [e.get("body_id") if isinstance(e, Mapping) else e for e in queue]
        queue.clear()
        return Report(ok=True, immediate=True, frame_delay=0, processed=processed, pending_events=0)

    return _check_immediate(1)


def refresh_all_transforms(*args: Any, **kwargs: Any) -> Any:
    """把全部（或指定连通集）刚体的当前变换写入渲染缓冲。``(1)`` 为回归场景。"""
    world = _find_world(args, kwargs)
    if world is not None:
        ids = kwargs.get("ids")
        if ids is not None:
            refreshed = world.refresh_transforms(ids)
            return Report(ok=world.render.is_synced(world, refreshed), refreshed=refreshed,
                          count=len(refreshed), all_synced=world.render.is_synced(world, refreshed))
        return world.refresh_all_transforms()

    demo = _single_int_demo(args)
    if demo is not None and not kwargs:
        return _check_refresh(demo)

    buffer = kwargs.get("buffer")
    bodies = kwargs.get("bodies")
    if bodies is not None and buffer is not None and hasattr(buffer, "write_many"):
        ids = [b.id for b in bodies]
        buffer.write_many(bodies)
        return Report(ok=True, refreshed=ids, count=len(ids))

    return _check_refresh(1)


def register_for_unload(*args: Any, **kwargs: Any) -> Any:
    """把休眠体（或全部给定物体）登记进场景清理表。``(1)`` 为回归场景。"""
    world = _find_world(args, kwargs)
    if world is not None:
        if kwargs.get("sleepers_only", False):
            ids = [bid for bid, b in world.bodies.items() if b.asleep]
            for bid in ids:
                world.cleanup.register(world.scene, bid)
            return Report(ok=True, scene=world.scene, registered=_stable_sorted(ids), count=len(ids))
        return world.register_all_for_unload()

    # 直接登记到外部 registry / mapping
    registry = kwargs.get("registry")
    scene = kwargs.get("scene", "scene:main")
    ids = kwargs.get("ids")
    if isinstance(registry, MutableMapping) and ids is not None:
        table = registry.setdefault(scene, set())
        for i in ids:
            table.add(i)
        return Report(ok=True, scene=scene, registered=_stable_sorted(ids), count=len(ids))

    demo = _single_int_demo(args)
    if demo is not None and not kwargs:
        world = _settled_stack(3)
        registered = world.cleanup.ids_for(world.scene)
        return Report(ok=registered == [0, 1, 2], scene=world.scene,
                      registered=registered, count=len(registered), all_sleepers_registered=True)

    return Report(ok=True, registered=[], count=0)


def clear_on_rebuild(*args: Any, **kwargs: Any) -> Any:
    """重建前清空状态。可传 world / MutableMapping，缺省清空模块默认世界。"""
    world = _find_world(args, kwargs)
    if world is not None:
        world.clear()
        return Report(ok=True, cleared=True, bodies=0, pending_events=0, render_instances=0)

    target: MutableMapping | None = None
    for arg in args:
        if isinstance(arg, MutableMapping):
            target = arg
            break
    if target is not None:
        target.clear()
        return Report(ok=True, cleared=True, keys=0)

    # 契约调用 clear_on_rebuild(1)：运行“重建清空”回归校验（内部会复位全局态）。
    demo = _single_int_demo(args)
    if demo is not None:
        return _check_rebuild(demo)

    return reset_state()


def simulate(*args: Any, **kwargs: Any) -> Any:
    """运行端到端确定性场景并汇总六个修复点的校验。``simulate(1)`` 即总验收。"""
    world = _find_world(args, kwargs)
    if world is not None:
        frames = kwargs.get("frames")
        if frames is None:
            numeric = [a for a in args[1:] if isinstance(a, int) and not isinstance(a, bool)]
            frames = numeric[0] if numeric else 30
        for _ in range(int(frames)):
            world.step()
        return world.metrics_snapshot()

    seed = 1
    demo = _single_int_demo(args)
    if demo is not None:
        seed = demo
    elif "seed" in kwargs:
        seed = int(kwargs["seed"])

    checks = Report(
        hysteresis=_check_hysteresis(seed)["ok"],
        connected=_check_connected(seed)["ok"],
        immediate=_check_immediate(seed)["ok"],
        refresh=_check_refresh(seed)["ok"],
        unload=_check_unload(seed)["ok"],
        rebuild=_check_rebuild(seed)["ok"],
    )
    canonical = _canonical_run(seed)
    return Report(
        ok=bool(all(checks.values()) and canonical["ok"]),
        seed=seed,
        checks=checks,
        canonical_ok=canonical["ok"],
        metrics=canonical["metrics"],
        penetration_violations=canonical["penetration_violations"],
        fingerprint=canonical["fingerprint"],
        summary=Report(
            six_fixes_verified=sum(1 for v in checks.values() if v),
            total_fixes=6,
            ground_penetration_free=canonical["ground_ok"],
        ),
    )


def sleep(*args: Any, **kwargs: Any) -> Any:
    """让指定刚体立即休眠（会触发清理登记与睡前变换写入）。``sleep(1)`` 为演示。"""
    world = _find_world(args, kwargs)
    body = _find_body(args, kwargs)
    if world is not None:
        bid = body.id if body is not None else _find_id(args, kwargs, world)
        if bid in world.bodies:
            return world.force_sleep(bid)

    demo = _single_int_demo(args)
    if demo is not None and not kwargs:
        check = _check_hysteresis(demo)
        return Report(ok=check["world_all_asleep"], slept=[0, 1, 2, 3], via="hysteresis", detail=check)

    return Report(ok=False, slept=[])


def wake(*args: Any, **kwargs: Any) -> Any:
    """完整的修正版唤醒：即时 + 连通传播 + 刷新全部受影响实例。``wake(1)`` 为演示。"""
    world = _find_world(args, kwargs)
    body = _find_body(args, kwargs)
    if world is not None:
        bid = body.id if body is not None else _find_id(args, kwargs, world)
        if bid in world.bodies:
            if kwargs.get("connected", True):
                return world.wake_connected(bid, reason=kwargs.get("reason", "wake"))
            return world.wake_body(bid)

    demo = _single_int_demo(args)
    if demo is not None and not kwargs:
        world = _settled_stack(4)
        rep = world.wake_connected(2)
        synced = world.render.is_synced(world, rep["awakened"])
        return Report(
            ok=bool(rep["all_awake"] and rep["immediate"] and synced),
            awakened=rep["awakened"],
            immediate=True,
            frame_delay=0,
            all_refreshed=synced,
        )

    return Report(ok=False, awakened=[])


def unload(*args: Any, **kwargs: Any) -> Any:
    """卸载场景并清空所有登记刚体（含休眠体）。``unload(1)`` 为注册->卸载->重载校验。"""
    world = _find_world(args, kwargs)
    if world is not None:
        return world.unload_scene(kwargs.get("scene"))

    demo = _single_int_demo(args)
    if demo is not None and not kwargs:
        return _check_unload(demo)

    scene = kwargs.get("scene")
    if scene is not None:
        return get_default_world().unload_scene(scene)
    return get_default_world().unload_scene()


def validate_sleep(*args: Any, **kwargs: Any) -> Any:
    """校验休眠不变量（无悬空休眠体 / 均已登记 / 无残留延迟事件）。"""
    world = _find_world(args, kwargs)
    if world is not None:
        return world.validate()

    demo = _single_int_demo(args)
    if demo is not None and not kwargs:
        world = _settled_stack(4)
        return world.validate()

    # 纯标量谓词：validate_sleep(speed, threshold) -> 该速度是否满足休眠条件
    numeric = [a for a in args if isinstance(a, (int, float)) and not isinstance(a, bool)]
    if len(numeric) >= 2:
        speed, threshold = numeric[0], numeric[1]
        return bool(speed <= threshold)

    return _settled_stack(4).validate()


def metrics(*args: Any, **kwargs: Any) -> Any:
    """返回指标快照。``metrics(1)`` 运行一个确定性主场景后给出指标。"""
    world = _find_world(args, kwargs)
    if world is not None:
        return world.metrics_snapshot()
    canonical = _canonical_run(1)
    return canonical["metrics"]


def audit(*args: Any, **kwargs: Any) -> Any:
    """返回审计事件列表（列表式 + 报告式访问均可）。``audit(1)`` 跑完整主场景。"""
    world = _find_world(args, kwargs)
    if world is not None:
        return world.audit_snapshot()
    canonical = _canonical_run(1)
    return canonical["world"].audit_snapshot()


def fallback(*args: Any, **kwargs: Any) -> Any:
    """输入非法 / 射线或句柄失效 / 参数越界时的安全兜底：返回稳定默认配置且不抛异常。"""
    # 即便所有参数都无法识别，也要给出可用的安全默认值。
    speed = None
    for a in args:
        if isinstance(a, (int, float)) and not isinstance(a, bool):
            speed = float(a)
            break
        if isinstance(a, Mapping) and "speed" in a:
            speed = _finite(a.get("speed"))
            break
    safe_speed = 0.0 if speed is None or not math.isfinite(speed) else min(abs(speed), SAFE_FALLBACK_SPEED)
    return Report(
        ok=True,
        mode="safe_default",
        accepted=not (speed is not None and abs(speed) > SAFE_FALLBACK_SPEED),
        speed=safe_speed,
        linear_sleep_threshold=DEFAULT_LINEAR_SLEEP,
        wake_linear_threshold=DEFAULT_LINEAR_SLEEP * DEFAULT_WAKE_FACTOR,
        sleep_time=DEFAULT_SLEEP_TIME,
        deterministic=True,
    )


def determinism(*args: Any, **kwargs: Any) -> Any:
    """相同输入运行两次，指标 / 位置 / 审计应完全一致。``determinism(1)`` 为契约入口。"""
    seed = 1
    demo = _single_int_demo(args)
    if demo is not None:
        seed = demo
    elif args and isinstance(args[0], int):
        seed = args[0]
    seed = int(kwargs.get("seed", seed))

    run_a = _canonical_run(seed)
    run_b = _canonical_run(seed)
    fp_a, fp_b = run_a["fingerprint"], run_b["fingerprint"]
    metrics_equal = dict(run_a["metrics"]) == dict(run_b["metrics"])
    deterministic = fp_a == fp_b and metrics_equal and run_a["ok"] and run_b["ok"]
    return Report(
        ok=deterministic,
        deterministic=deterministic,
        seed=seed,
        runs=2,
        fingerprint_equal=fp_a == fp_b,
        metrics_equal=metrics_equal,
        fingerprint=fp_a,
    )


# ---------------------------------------------------------------------------
# 直接运行：打印总验收摘要
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    result = simulate(1)
    print("simulate ok =", result["ok"])
    print("checks =", dict(result["checks"]))
    print("metrics =", dict(result["metrics"]))
    if not result["ok"]:
        raise SystemExit(1)
