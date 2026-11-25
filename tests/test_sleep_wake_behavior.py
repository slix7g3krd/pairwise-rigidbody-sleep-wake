"""刚体休眠 / 唤醒传播六个修复点的行为回归测试。

这些测试针对题目描述的真实故障：滞回休眠、连通唤醒、即时生效、全量刷新、
休眠登记清理与重建清空。全部只依赖标准库与本模块，确定性、可重复。
"""

from __future__ import annotations

import pytest

from gameplay import rigidbody_sleep_wake as rsw
from gameplay.rigidbody_sleep_wake import (
    FIXED_DT,
    SleepWakeWorld,
    connected_component,
    hysteresis_sample_run,
)


# ---------------------------------------------------------------------------
# 1. 休眠加滞回
# ---------------------------------------------------------------------------


def test_jitter_stack_eventually_sleeps_but_naive_policy_never_does():
    # 速度在单一严格阈值 0.05 上下抖动：旧的“连续低速帧”策略永远被清零。
    speeds = [0.045 if i % 2 == 0 else 0.065 for i in range(120)]
    rep = hysteresis_sample_run(speeds, strict_threshold=0.05, need_frames=10)
    assert rep["old_policy_slept"] is False
    assert rep["slept"] is True
    assert rep["sleep_frame"] is not None


def test_sleeping_body_ignores_band_jitter_but_wakes_on_real_impulse():
    world = rsw._settled_stack(2)
    assert all(b.asleep for b in world.bodies.values())

    body = world.bodies[0]
    body.velocity = (0.10, 0.0, 0.0)  # 落在滞回容忍带内，远低于唤醒阈值
    world.update_sleep(FIXED_DT)
    assert body.asleep is True

    body.velocity = (2.0, 0.0, 0.0)   # 达到唤醒阈值 -> 立即醒
    world.update_sleep(FIXED_DT)
    assert body.asleep is False


def test_settled_stack_sleeps_and_physics_idle_work_drops_to_zero():
    rep = rsw.sleep_hysteresis(1)
    assert rep["ok"] is True
    assert rep["world_all_asleep"] is True
    assert rep["idle_active_bodies"] == 0


# ---------------------------------------------------------------------------
# 2. 唤醒连通传播
# ---------------------------------------------------------------------------


def test_connected_component_is_transitive():
    graph = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2]}
    ids = connected_component(graph, 0)
    assert ids == [0, 1, 2, 3]
    # 稀疏前向边也应被当作无向链处理
    assert connected_component({0: [1], 1: [2], 2: [3]}, 2) == [0, 1, 2, 3]


def test_wake_connected_pure_result_compares_to_list_and_set():
    res = rsw.wake_connected({0: [1], 1: [2], 2: [3]}, 0)
    assert list(res) == [0, 1, 2, 3]
    assert res == {3, 2, 1, 0}
    assert 2 in res
    assert len(res) == 4


def test_removing_bottom_wakes_entire_upper_stack_not_only_neighbor():
    world = rsw._settled_stack(4)
    removal = world.remove_body(0)
    assert removal["awakened"] == [1, 2, 3]  # 而不是只有直接邻居 [1]
    assert all(not world.bodies[i].asleep for i in (1, 2, 3))
    validation = world.validate()
    assert validation["floating_sleepers"] == []
    assert rsw.wake_connected(1)["ok"] is True


# ---------------------------------------------------------------------------
# 3. 唤醒即时生效
# ---------------------------------------------------------------------------


def test_wake_takes_effect_immediately_without_pending_event():
    world = SleepWakeWorld()
    world.add_body(1, position=(0, 2, 0), asleep=True)
    rep = world.wake_body(1)
    body = world.bodies[1]
    assert rep["immediate"] is True
    assert rep["frame_delay"] == 0
    assert body.asleep is False
    assert body.collision_enabled is True
    assert world.wake_queue == []


def test_async_queue_path_is_one_frame_late_by_contrast():
    world = SleepWakeWorld()
    world.add_body(1, position=(0, 2, 0), velocity=(0, -12, 0), asleep=True)
    world.request_wake_async(1)
    assert world.bodies[1].asleep is True
    assert len(world.wake_queue) == 1
    end = world.end_frame()
    assert end["frame_delay"] == 1
    assert world.bodies[1].asleep is False


def test_immediate_wake_fast_body_does_not_penetrate_ground():
    rep = rsw.wake_immediate(1)
    assert rep["ok"] is True
    assert rep["frame_delay"] == 0
    assert rep["residual_penetration"] <= rsw.GROUND_SLOP
    # 晚一帧的旧路径确实产生 > slop 的未修正穿透
    assert rep["legacy_unresolved_penetration"] > rsw.GROUND_SLOP


# ---------------------------------------------------------------------------
# 4. 唤醒刷新全部变换
# ---------------------------------------------------------------------------


def test_refresh_all_repairs_every_stale_instance():
    world = rsw._settled_stack(4)
    world.render.corrupt([1, 2, 3])
    assert world.render.stale_ids(world) == [1, 2, 3]
    rep = world.refresh_all_transforms()
    assert rep["count"] == 4
    assert rep["all_synced"] is True
    assert world.render.stale_ids(world) == []


def test_wake_refreshes_the_whole_connected_set_not_only_top():
    rep = rsw.refresh_all_transforms(1)
    assert rep["ok"] is True
    assert rep["all_synced"] is True

    world = rsw._settled_stack(4)
    world.render.corrupt([0, 1, 2, 3])
    wake = world.wake_connected(1)
    # 从中间节点沿接触链传递，整摞（含底部 0）都被唤醒并刷新。
    assert wake["awakened"] == [0, 1, 2, 3]
    assert world.render.is_synced(world, [0, 1, 2, 3]) is True
    assert world.render.stale_ids(world) == []


# ---------------------------------------------------------------------------
# 5. 休眠登记清理
# ---------------------------------------------------------------------------


def test_sleep_registers_body_for_unload():
    world = rsw._settled_stack(3)
    assert world.cleanup.ids_for(world.scene) == [0, 1, 2]
    validation = world.validate()
    assert validation["unregistered_sleepers"] == []


def test_unload_removes_sleepers_and_reload_has_no_overlap():
    rep = rsw.unload(1)
    assert rep["ok"] is True
    assert rep["remaining"] == 0
    assert rep["reload_overlaps"] == 0
    # 反证：缺少休眠登记的旧实现会残留并在重载时重叠
    assert rep["buggy_survivors"] == 3
    assert rep["buggy_reload_overlaps"] == 3


# ---------------------------------------------------------------------------
# 6. 重建清空状态
# ---------------------------------------------------------------------------


def test_clear_on_rebuild_resets_global_world_and_id_sequence():
    rep = rsw.clear_on_rebuild(1)
    assert rep["ok"] is True
    assert rep["empty_after_clear"] is True
    assert rep["first_id_after_rebuild"] == 0
    gw = rsw.get_default_world()
    assert gw.bodies == {} and gw.wake_queue == [] and gw.render.instances == {}


# ---------------------------------------------------------------------------
# 端到端 / 确定性 / 观测入口
# ---------------------------------------------------------------------------


def test_simulate_verifies_all_six_fixes():
    result = rsw.simulate(1)
    assert result["ok"] is True
    assert all(result["checks"].values())
    assert result["summary"]["six_fixes_verified"] == 6
    assert result["penetration_violations"] == []


def test_determinism_is_stable_across_runs():
    rep = rsw.determinism(1)
    assert rep["deterministic"] is True
    assert rep["fingerprint_equal"] is True
    assert rep["metrics_equal"] is True
    # 任意次数重复 simulate 指纹一致
    fps = {rsw.simulate(7)["fingerprint"] for _ in range(3)}
    assert len(fps) == 1


def test_observation_and_fallback_entries_are_well_formed():
    assert rsw.metrics(1)["bodies"] >= 1
    audit = rsw.audit(1)
    assert len(audit) > 0
    assert audit["count"] == len(audit)
    assert isinstance(audit["events"], list)
    fb = rsw.fallback("unexpected", object())
    assert fb["ok"] is True and fb["mode"] == "safe_default"
    assert rsw.validate_sleep(1)["ok"] is True


@pytest.mark.parametrize("fn_name", [
    "sleep_hysteresis", "wake_connected", "wake_immediate",
    "refresh_all_transforms", "register_for_unload", "clear_on_rebuild",
    "simulate", "sleep", "wake", "unload", "validate_sleep",
    "metrics", "audit", "fallback", "determinism",
])
def test_every_entry_supports_integer_demo(fn_name):
    fn = getattr(rsw, fn_name)
    result = fn(1)
    assert result is not None
