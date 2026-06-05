"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

import pytest

from cereal import custom
from openpilot.sunnypilot.selfdrive.controls.lib.acceleration_controller import (
  ACCELERATION_PROFILE_ENABLED_PARAM,
  ACCELERATION_PROFILE_PARAM,
  AccelerationProfile,
  AccelerationProfileController,
)


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


def test_invalid_acceleration_profile_is_clipped():
  params = ParamsStub({
    ACCELERATION_PROFILE_ENABLED_PARAM: True,
    ACCELERATION_PROFILE_PARAM: 9,
  })

  controller = AccelerationProfileController(params)

  assert controller.profile == AccelerationProfile.eco
  assert params.values[ACCELERATION_PROFILE_PARAM] == AccelerationProfile.eco


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
