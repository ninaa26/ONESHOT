"""Actuator: first-order servo for the rudder and the sheet winch."""

import math

import pytest

from sailbench.sim.actuator import Actuator


class TestFirstOrderLag:
    """The lag itself."""

    def test_instant_when_no_time_constant(self) -> None:
        """tau 0 means the surface tracks its command exactly."""
        assert Actuator(tau_s=0.0).advance(20.0, 0.02) == pytest.approx(20.0)

    def test_reaches_one_time_constant_in_one_tau(self) -> None:
        """After tau seconds a first-order lag has covered 1 - 1/e of the gap."""
        a = Actuator(tau_s=0.5)
        assert a.advance(10.0, 0.5) == pytest.approx(10.0 * (1 - math.exp(-1.0)))

    def test_is_exact_regardless_of_step_size(self) -> None:
        """One big step and many small ones must agree to machine precision.

        This is the property that keeps the actuator from setting the accuracy of
        the whole simulation: the lag is solved in closed form, not stepped.
        """
        one = Actuator(tau_s=0.3)
        one.advance(15.0, 0.6)
        many = Actuator(tau_s=0.3)
        for _ in range(600):
            many.advance(15.0, 0.001)
        assert many.position == pytest.approx(one.position, rel=1e-12)

    def test_approaches_but_never_overshoots(self) -> None:
        """A first-order lag is monotonic towards its command."""
        a = Actuator(tau_s=0.2)
        last = 0.0
        for _ in range(200):
            now = a.advance(10.0, 0.01)
            assert last <= now <= 10.0
            last = now

    def test_settles_on_the_command(self) -> None:
        """Given long enough it arrives."""
        a = Actuator(tau_s=0.1)
        for _ in range(500):
            a.advance(-12.0, 0.02)
        assert a.position == pytest.approx(-12.0)


class TestRateLimit:
    """Slew saturation on top of the lag."""

    def test_caps_movement_per_step(self) -> None:
        """A rate limit bounds how far the surface can move in one step."""
        a = Actuator(tau_s=0.0, max_rate=100.0)
        assert a.advance(90.0, 0.02) == pytest.approx(2.0)

    def test_unlimited_when_zero(self) -> None:
        """max_rate 0 means no limit."""
        assert Actuator(tau_s=0.0, max_rate=0.0).advance(90.0, 0.02) == pytest.approx(90.0)

    def test_limits_both_directions(self) -> None:
        """It bounds retreat as well as advance."""
        a = Actuator(tau_s=0.0, max_rate=100.0, position=50.0)
        assert a.advance(0.0, 0.02) == pytest.approx(48.0)


class TestManualHelmAids:
    """Deadband and auto-centre, which should be off for RL."""

    def test_deadband_swallows_small_commands(self) -> None:
        """Inside the deadband the command reads as zero."""
        a = Actuator(tau_s=0.0, deadband=1.5, position=0.0)
        assert a.advance(1.0, 0.02) == pytest.approx(0.0)

    def test_deadband_passes_larger_commands(self) -> None:
        """Outside it, the command is honoured in full."""
        assert Actuator(tau_s=0.0, deadband=1.5).advance(5.0, 0.02) == pytest.approx(5.0)

    def test_auto_centre_returns_towards_zero(self) -> None:
        """With no command the surface drifts back to centre."""
        a = Actuator(tau_s=0.0, center_tau_s=0.5, position=10.0)
        moved = a.advance(0.0, 0.1)
        assert 0.0 < moved < 10.0

    def test_auto_centre_uses_its_own_time_constant(self) -> None:
        """Centring is slower than the servo's own response, by design."""
        a = Actuator(tau_s=0.05, center_tau_s=0.5, position=10.0)
        assert a.advance(0.0, 0.1) == pytest.approx(10.0 * math.exp(-0.1 / 0.5))

    def test_both_off_by_default(self) -> None:
        """A plain actuator has no helm aids, which is what RL needs."""
        a = Actuator()
        assert a.deadband == 0.0
        assert a.center_tau_s == 0.0


class TestSampling:
    """position_at lets an integrator see the surface mid-step."""

    def test_does_not_move_the_actuator(self) -> None:
        """Sampling is read-only; only advance() commits."""
        a = Actuator(tau_s=0.1, position=3.0)
        a.position_at(20.0, 0.02)
        assert a.position == 3.0

    def test_agrees_with_advance_at_the_full_step(self) -> None:
        """Sampling at dt must equal advancing by dt."""
        sampled = Actuator(tau_s=0.1, position=3.0).position_at(20.0, 0.02)
        moved = Actuator(tau_s=0.1, position=3.0).advance(20.0, 0.02)
        assert sampled == pytest.approx(moved)

    def test_zero_offset_is_the_current_position(self) -> None:
        """At the start of the step the surface has not moved yet."""
        assert Actuator(tau_s=0.1, position=3.0).position_at(20.0, 0.0) == pytest.approx(3.0)

    def test_is_monotonic_across_the_step(self) -> None:
        """Mid-step samples lie between the endpoints."""
        a = Actuator(tau_s=0.1, position=0.0)
        samples = [a.position_at(10.0, t) for t in (0.0, 0.005, 0.01, 0.015, 0.02)]
        assert samples == sorted(samples)


class TestObservationExposure:
    """The RL observation must show where the surfaces are, not just the command."""

    def test_off_by_default_keeps_the_observation_shape(self) -> None:
        """Existing checkpoints depend on a 13-element observation."""
        from sailbench.rl.envs.waypoint_env import WaypointEnv, WaypointEnvConfig

        env = WaypointEnv(WaypointEnvConfig(simulator_config="flingo_floty.yaml"))
        obs, _ = env.reset(seed=0)
        assert obs.shape == (13,)

    def test_on_adds_the_two_actuator_positions(self) -> None:
        """Opting in appends rudder and sheet position."""
        from sailbench.rl.envs.waypoint_env import WaypointEnv, WaypointEnvConfig

        env = WaypointEnv(WaypointEnvConfig(
            simulator_config="flingo_floty.yaml", include_actuator_state=True))
        obs, _ = env.reset(seed=0)
        assert obs.shape == (15,)
        assert env.observation_space.contains(obs)

    def test_reported_position_lags_the_command(self) -> None:
        """The whole point: with a slow winch, command and position differ.

        The sheet has a 3 s time constant, so one step after a full-scale sheet
        command the surface has barely moved and the observation must say so.
        """
        import numpy as np_

        from sailbench.rl.envs.waypoint_env import WaypointEnv, WaypointEnvConfig

        env = WaypointEnv(WaypointEnvConfig(
            simulator_config="flingo_floty.yaml", include_actuator_state=True))
        env.reset(seed=0)
        obs, *_ = env.step(np_.array([0.0, 1.0]))  # sheet hard out
        commanded, actual = float(obs[12]), float(obs[14])
        assert commanded == pytest.approx(1.0)
        assert actual < commanded, "observation reports the command, not the surface"
