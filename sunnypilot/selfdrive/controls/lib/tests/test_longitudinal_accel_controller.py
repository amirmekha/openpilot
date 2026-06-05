"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from cereal import log
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import get_T_FOLLOW
from openpilot.sunnypilot.selfdrive.controls.lib.acceleration_controller import (
  ACCELERATION_PROFILE_ENABLED_PARAM,
  ACCELERATION_PROFILE_PARAM,
  AccelerationProfile,
  AccelerationProfileController,
  PROFILE_CONFIGS,
)
from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_accel_controller import (
  LongitudinalAccelerationController,
  LongitudinalComfortController,
)

T_FOLLOW = get_T_FOLLOW(log.LongitudinalPersonality.standard)


class ParamsStub:
  def __init__(self, values):
    self.values = values

  def get_bool(self, key):
    return bool(self.values[key])

  def get(self, key, return_default=False):
    return self.values[key]

  def put(self, key, value, block=False):
    self.values[key] = value


class FakeMpc:
  def __init__(self):
    self.cruise_max_accel = None

  def set_cruise_max_accel(self, cruise_max_accel):
    self.cruise_max_accel = cruise_max_accel


def make_lead(status=True, dRel=0.0, vLead=0.0, aLeadK=0.0):
  return SimpleNamespace(status=status, dRel=dRel, vLead=vLead, aLeadK=aLeadK)


def make_sm(lead_one=None, lead_two=None, force_decel=False, personality=log.LongitudinalPersonality.standard):
  return {
    'radarState': SimpleNamespace(leadOne=lead_one or make_lead(status=False), leadTwo=lead_two or make_lead(status=False)),
    'controlsState': SimpleNamespace(forceDecel=force_decel),
    'selfdriveState': SimpleNamespace(personality=personality),
  }


def make_comfort(profile=AccelerationProfile.normal):
  apc = AccelerationProfileController(ParamsStub({
    ACCELERATION_PROFILE_ENABLED_PARAM: True,
    ACCELERATION_PROFILE_PARAM: int(profile),
  }))
  return LongitudinalComfortController(apc)


def make_lac():
  return LongitudinalAccelerationController(FakeMpc())


# ---------------- _lead_brake_cap ----------------

def test_lead_brake_cap_none_when_no_lead():
  assert LongitudinalComfortController._lead_brake_cap(None, 10.0, T_FOLLOW) == (None, False)
  assert LongitudinalComfortController._lead_brake_cap(make_lead(status=False), 10.0, T_FOLLOW) == (None, False)


def test_lead_brake_cap_no_cap_when_steady_and_far():
  cap, emergency = LongitudinalComfortController._lead_brake_cap(make_lead(dRel=60.0, vLead=20.0, aLeadK=0.0), 20.0, T_FOLLOW)
  assert cap is None
  assert emergency is False


def test_lead_brake_cap_far_slow_closing_comfortable():
  cap, emergency = LongitudinalComfortController._lead_brake_cap(make_lead(dRel=40.0, vLead=10.0, aLeadK=0.0), 15.0, T_FOLLOW)
  assert cap == pytest.approx(-1.275)
  assert emergency is False


def test_lead_brake_cap_gap_error_only():
  cap, emergency = LongitudinalComfortController._lead_brake_cap(make_lead(dRel=10.0, vLead=5.0, aLeadK=0.0), 5.2, T_FOLLOW)
  assert cap == pytest.approx(-0.3948)
  assert emergency is False


def test_lead_brake_cap_deep_cap_is_emergency():
  cap, emergency = LongitudinalComfortController._lead_brake_cap(make_lead(dRel=8.0, vLead=0.0, aLeadK=0.0), 15.0, T_FOLLOW)
  assert cap == pytest.approx(-3.5)  # clipped to ACCEL_MIN
  assert emergency is True


def test_lead_brake_cap_inside_stop_distance_stays_finite():
  # available_distance floors at 1.0 so the kinematic cap cannot blow up; result clips to ACCEL_MIN.
  cap, emergency = LongitudinalComfortController._lead_brake_cap(make_lead(dRel=4.0, vLead=0.0, aLeadK=-1.0), 10.0, T_FOLLOW)
  assert cap == pytest.approx(-3.5)
  assert emergency is True


# ---------------- fix #2: matched-speed hard-braking lead ----------------

def test_matched_speed_hard_brake_is_emergency():
  # closing_speed ~ 0 (matched speed) but lead brakes hard within desired gap -> must be emergency.
  cap, emergency = LongitudinalComfortController._lead_brake_cap(make_lead(dRel=20.0, vLead=15.0, aLeadK=-3.0), 15.0, T_FOLLOW)
  assert cap == pytest.approx(-1.65)
  assert emergency is True


def test_matched_speed_soft_brake_not_emergency():
  # a mild lead deceleration at matched speed must NOT trip the emergency floor.
  cap, emergency = LongitudinalComfortController._lead_brake_cap(make_lead(dRel=20.0, vLead=15.0, aLeadK=-1.0), 15.0, T_FOLLOW)
  assert cap == pytest.approx(-0.775)
  assert emergency is False


# ---------------- accel_cap aggregation ----------------

def test_accel_cap_takes_most_braking_lead():
  sm = make_sm(lead_one=make_lead(dRel=40.0, vLead=10.0, aLeadK=0.0),   # mild cap
               lead_two=make_lead(dRel=8.0, vLead=0.0, aLeadK=0.0))     # deep cap
  cap, emergency = make_comfort().accel_cap(sm, 15.0)
  assert cap == pytest.approx(-3.5)
  assert emergency is True


def test_accel_cap_force_decel_no_lead():
  cap, emergency = make_comfort().accel_cap(make_sm(force_decel=True), 15.0)
  assert cap == pytest.approx(-0.05)
  assert emergency is True


def test_accel_cap_force_decel_keeps_deeper_lead_cap():
  sm = make_sm(lead_one=make_lead(dRel=8.0, vLead=0.0, aLeadK=0.0), force_decel=True)
  cap, emergency = make_comfort().accel_cap(sm, 15.0)
  assert cap == pytest.approx(-3.5)  # min(-0.05, -3.5)
  assert emergency is True


def test_accel_cap_none_without_lead_or_force_decel():
  cap, emergency = make_comfort().accel_cap(make_sm(), 15.0)
  assert cap is None
  assert emergency is False


# ---------------- apply: floor + fix #1 ----------------

def test_apply_cap_wins_over_positive_target():
  comfort = make_comfort()
  comfort.accel_cap = lambda sm, v_ego: (-1.0, False)
  assert comfort.apply(0.5, False, make_sm(), 15.0) == pytest.approx(-1.0)


def test_apply_passes_through_mild_target():
  # a comfortable target above the comfort limit is neither weakened nor strengthened
  comfort = make_comfort()
  comfort.accel_cap = lambda sm, v_ego: (None, False)
  assert comfort.apply(-2.0, False, make_sm(), 15.0) == pytest.approx(-2.0)


def test_apply_emergency_floor_allows_full_brake():
  comfort = make_comfort()
  comfort.accel_cap = lambda sm, v_ego: (None, True)
  assert comfort.apply(-5.0, False, make_sm(), 15.0) == pytest.approx(-3.5)


def test_apply_does_not_weaken_planner_hard_brake():
  # FIX #1: a genuine plan/e2e hard brake below -2.5 with NO radar emergency must be preserved.
  comfort = make_comfort()
  comfort.accel_cap = lambda sm, v_ego: (None, False)
  assert comfort.apply(-3.2, False, make_sm(), 15.0) == pytest.approx(-3.2)


def test_apply_low_speed_stop_softening():
  comfort = make_comfort()
  comfort.accel_cap = lambda sm, v_ego: (None, False)
  assert comfort.apply(-2.0, True, make_sm(), 0.0) == pytest.approx(-0.45)
  assert comfort.apply(-2.0, True, make_sm(), 2.5) == pytest.approx(-0.825)
  assert comfort.apply(-2.0, True, make_sm(), 4.0) == pytest.approx(-1.05)
  # softening only applies below 5 m/s; at/above 5 the comfort limit governs
  assert comfort.apply(-2.0, True, make_sm(), 5.0) == pytest.approx(-2.0)


def test_apply_stop_softening_disabled_when_emergency():
  comfort = make_comfort()
  comfort.accel_cap = lambda sm, v_ego: (None, True)
  assert comfort.apply(-3.0, True, make_sm(), 0.0) == pytest.approx(-3.0)


# ---------------- lead_start_accel (SnG) + fix #4 ----------------

def test_lead_start_accel_none_above_creep_speed():
  sm = make_sm(lead_one=make_lead(dRel=8.0, vLead=1.0, aLeadK=0.0))
  assert make_comfort().lead_start_accel(sm, 1.0) is None


def test_lead_start_accel_none_on_force_decel():
  sm = make_sm(lead_one=make_lead(dRel=8.0, vLead=1.0, aLeadK=0.0), force_decel=True)
  assert make_comfort().lead_start_accel(sm, 0.0) is None


def test_lead_start_accel_ignores_stationary_lead():
  # a stopped lead (vLead at/below the gate) must not trigger creep
  sm = make_sm(lead_one=make_lead(dRel=8.0, vLead=0.0, aLeadK=0.0))
  assert make_comfort().lead_start_accel(sm, 0.0) is None


def test_lead_start_accel_ignores_too_close_lead():
  sm = make_sm(lead_one=make_lead(dRel=5.0, vLead=1.0, aLeadK=0.0))
  assert make_comfort().lead_start_accel(sm, 0.0) is None


def test_lead_start_accel_value_for_moving_lead():
  sm = make_sm(lead_one=make_lead(dRel=8.0, vLead=1.0, aLeadK=0.0))
  cfg = PROFILE_CONFIGS[AccelerationProfile.normal]
  expected = float(np.interp(1.0, [0.0, 0.5, 2.0], cfg.sng_start_accel_v))
  assert make_comfort().lead_start_accel(sm, 0.0) == pytest.approx(expected)


def test_lead_start_accel_takes_max_across_leads():
  sm = make_sm(lead_one=make_lead(dRel=8.0, vLead=0.9, aLeadK=0.0),
               lead_two=make_lead(dRel=8.0, vLead=2.0, aLeadK=0.0))
  cfg = PROFILE_CONFIGS[AccelerationProfile.normal]
  expected = max(float(np.interp(0.9, [0.0, 0.5, 2.0], cfg.sng_start_accel_v)),
                 float(np.interp(2.0, [0.0, 0.5, 2.0], cfg.sng_start_accel_v)))
  assert make_comfort().lead_start_accel(sm, 0.0) == pytest.approx(expected)


# ---------------- update_accel_target: fix #5 ----------------

def test_update_accel_target_launches_when_clear():
  lac = make_lac()
  lac.comfort_controller.accel_cap = lambda sm, v_ego: (None, False)
  lac.comfort_controller.lead_start_accel = lambda sm, v_ego: 0.3
  out_a, should_stop = lac.update_accel_target(make_sm(), 0.0, -0.2, True)
  assert should_stop is False
  assert out_a == pytest.approx(0.3)


def test_update_accel_target_skips_launch_when_braking():
  # FIX #5: don't clear should_stop and then re-clamp negative in the same cycle.
  lac = make_lac()
  lac.comfort_controller.accel_cap = lambda sm, v_ego: (-1.5, True)
  lac.comfort_controller.lead_start_accel = lambda sm, v_ego: 0.3
  out_a, should_stop = lac.update_accel_target(make_sm(), 0.0, -0.2, True)
  assert should_stop is True
  assert out_a <= 0.0


# ---------------- update_mpc_targets ----------------

def test_update_mpc_targets_sets_cruise_max_accel():
  lac = make_lac()
  lac.comfort_controller.accel_cap = lambda sm, v_ego: (None, False)
  v_cruise, a_desired = lac.update_mpc_targets(make_sm(), 10.0, 25.0, 0.5)
  assert lac.mpc.cruise_max_accel == pytest.approx(PROFILE_CONFIGS[AccelerationProfile.normal].mpc_cruise_max_accel)
  assert v_cruise == pytest.approx(25.0)
  assert a_desired == pytest.approx(0.5)


def test_update_mpc_targets_caps_a_desired_when_lead_braking():
  lac = make_lac()
  lac.comfort_controller.accel_cap = lambda sm, v_ego: (-0.8, False)
  _, a_desired = lac.update_mpc_targets(make_sm(), 10.0, 25.0, 0.5)
  assert a_desired == pytest.approx(-0.8)
