"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

import pytest

from cereal import custom
from openpilot.common.params import UnknownKeyName
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import CRUISE_MAX_ACCEL
from openpilot.selfdrive.controls.lib.longitudinal_planner import A_CRUISE_MAX_BP, A_CRUISE_MAX_VALS
from openpilot.sunnypilot.selfdrive.controls.lib.acceleration_controller import (
  ACCEL_MAX_BP,
  ACCELERATION_PROFILE_ENABLED_PARAM,
  ACCELERATION_PROFILE_PARAM,
  AccelerationProfile,
  AccelerationProfileController,
  PROFILE_CONFIGS,
)
from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_accel_controller import SNG_START_ACCEL_BP

ALL_PROFILES = [AccelerationProfile.eco, AccelerationProfile.normal, AccelerationProfile.sport]


def make_controller(profile, enabled=True):
  return AccelerationProfileController(ParamsStub({
    ACCELERATION_PROFILE_ENABLED_PARAM: enabled,
    ACCELERATION_PROFILE_PARAM: int(profile),
  }))


class ParamsStub:
  def __init__(self, values):
    self.values = values

  def get_bool(self, key):
    return bool(self.values[key])

  def get(self, key, return_default=False):
    return self.values[key]

  def put(self, key, value, block=False):
    self.values[key] = value


@pytest.mark.parametrize("profile", [AccelerationProfile.sport, AccelerationProfile.normal, AccelerationProfile.eco])
def test_reads_acceleration_profile_values(profile):
  params = ParamsStub({
    ACCELERATION_PROFILE_ENABLED_PARAM: True,
    ACCELERATION_PROFILE_PARAM: int(profile),
  })

  controller = AccelerationProfileController(params)

  assert controller.profile == profile


def test_disabled_acceleration_profile_uses_normal():
  params = ParamsStub({
    ACCELERATION_PROFILE_ENABLED_PARAM: False,
    ACCELERATION_PROFILE_PARAM: int(AccelerationProfile.sport),
  })

  controller = AccelerationProfileController(params)

  assert controller.profile == AccelerationProfile.normal


def test_invalid_acceleration_profile_resets_to_normal():
  params = ParamsStub({
    ACCELERATION_PROFILE_ENABLED_PARAM: True,
    ACCELERATION_PROFILE_PARAM: 9,
  })

  controller = AccelerationProfileController(params)

  # out-of-range resets to the documented default rather than clipping to an enum endpoint
  assert controller.profile == AccelerationProfile.normal
  assert params.values[ACCELERATION_PROFILE_PARAM] == int(AccelerationProfile.normal)


def test_acceleration_profile_values_match_cereal():
  accel_personality = custom.LongitudinalPlanSP.AccelerationPersonality

  assert AccelerationProfile is accel_personality
  assert int(accel_personality.sport) == int(AccelerationProfile.sport)
  assert int(accel_personality.normal) == int(AccelerationProfile.normal)
  assert int(accel_personality.eco) == int(AccelerationProfile.eco)


def test_longitudinal_plan_sp_accepts_acceleration_personality():
  accel_personality = custom.LongitudinalPlanSP.AccelerationPersonality
  msg = custom.LongitudinalPlanSP.new_message()

  msg.accelPersonality = accel_personality.eco

  assert str(msg.accelPersonality) == "eco"


def test_controller_returns_cereal_acceleration_personality():
  params = ParamsStub({
    ACCELERATION_PROFILE_ENABLED_PARAM: True,
    ACCELERATION_PROFILE_PARAM: int(AccelerationProfile.sport),
  })

  controller = AccelerationProfileController(params)

  assert controller.cereal_accel_personality() == custom.LongitudinalPlanSP.AccelerationPersonality.sport


# ---------------- numeric profile values ----------------

def test_normal_profile_matches_stock():
  # The NORMAL profile must be a true no-op vs upstream so the (default-on) feature does not
  # change baseline behavior. Guards against a silent PROFILE_CONFIGS edit.
  cfg = PROFILE_CONFIGS[AccelerationProfile.normal]
  assert cfg.accel_max_v == A_CRUISE_MAX_VALS
  assert ACCEL_MAX_BP == A_CRUISE_MAX_BP
  assert cfg.mpc_cruise_max_accel == CRUISE_MAX_ACCEL


def test_max_accel_interp_normal():
  controller = make_controller(AccelerationProfile.normal)
  assert controller.max_accel(0.0) == pytest.approx(1.6)
  assert controller.max_accel(10.0) == pytest.approx(1.2)
  assert controller.max_accel(25.0) == pytest.approx(0.8)
  assert controller.max_accel(40.0) == pytest.approx(0.6)
  assert controller.max_accel(17.5) == pytest.approx(1.0)


def test_max_accel_clamps_outside_breakpoints():
  controller = make_controller(AccelerationProfile.normal)
  assert controller.max_accel(-5.0) == pytest.approx(1.6)
  assert controller.max_accel(100.0) == pytest.approx(0.6)


@pytest.mark.parametrize("v_ego", [0.0, 10.0, 25.0, 40.0])
def test_max_accel_monotonic_across_profiles(v_ego):
  eco = make_controller(AccelerationProfile.eco).max_accel(v_ego)
  normal = make_controller(AccelerationProfile.normal).max_accel(v_ego)
  sport = make_controller(AccelerationProfile.sport).max_accel(v_ego)
  assert eco < normal < sport


def test_scalar_knobs_monotonic_across_profiles():
  eco = PROFILE_CONFIGS[AccelerationProfile.eco]
  normal = PROFILE_CONFIGS[AccelerationProfile.normal]
  sport = PROFILE_CONFIGS[AccelerationProfile.sport]
  assert eco.mpc_cruise_max_accel < normal.mpc_cruise_max_accel < sport.mpc_cruise_max_accel
  assert eco.positive_ramp_rate < normal.positive_ramp_rate < sport.positive_ramp_rate
  assert eco.start_accel_min < normal.start_accel_min < sport.start_accel_min


@pytest.mark.parametrize("profile", ALL_PROFILES)
def test_sng_start_accel_endpoints(profile):
  controller = make_controller(profile)
  cfg = PROFILE_CONFIGS[profile]
  for v_lead, expected in zip(SNG_START_ACCEL_BP, cfg.sng_start_accel_v, strict=True):
    assert controller.sng_start_accel(v_lead, SNG_START_ACCEL_BP) == pytest.approx(expected)


# ---------------- param refresh cadence + transitions ----------------

def test_update_params_refreshes_every_50_frames():
  params = ParamsStub({
    ACCELERATION_PROFILE_ENABLED_PARAM: True,
    ACCELERATION_PROFILE_PARAM: int(AccelerationProfile.sport),
  })
  controller = AccelerationProfileController(params)
  controller.update_params()  # consumes the frame-0 refresh
  params.values[ACCELERATION_PROFILE_PARAM] = int(AccelerationProfile.eco)

  for _ in range(49):
    controller.update_params()
  assert controller.profile == AccelerationProfile.sport  # not yet refreshed

  controller.update_params()  # 50th frame -> refresh
  assert controller.profile == AccelerationProfile.eco


def test_disabled_after_enabled_snaps_to_normal():
  controller = make_controller(AccelerationProfile.sport)
  assert controller.profile == AccelerationProfile.sport

  controller.params.values[ACCELERATION_PROFILE_ENABLED_PARAM] = False
  controller.read_params()
  assert controller.profile == AccelerationProfile.normal


def test_unknown_key_falls_back_to_normal():
  class BadParams:
    def get_bool(self, key):
      raise UnknownKeyName(key)

    def get(self, key, return_default=False):
      raise UnknownKeyName(key)

    def put(self, key, value, block=False):
      pass

  controller = AccelerationProfileController(BadParams())
  assert controller.profile == AccelerationProfile.normal
