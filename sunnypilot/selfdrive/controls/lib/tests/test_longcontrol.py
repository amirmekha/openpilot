"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

from cereal import car, custom
import pytest

from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.longcontrol import LongControl, LongCtrlState
from openpilot.sunnypilot.selfdrive.controls.lib.acceleration_controller import AccelerationProfile, PROFILE_CONFIGS


def make_longitudinal_car_params(**kwargs):
  CP = car.CarParams.new_message(**kwargs)
  CP.longitudinalTuning.kpBP = [0.]
  CP.longitudinalTuning.kpV = [0.]
  CP.longitudinalTuning.kiBP = [0.]
  CP.longitudinalTuning.kiV = [0.]
  return CP


def make_car_state(v_ego=0.0, a_ego=0.0):
  CS = car.CarState.new_message()
  CS.vEgo = v_ego
  CS.aEgo = a_ego
  CS.brakePressed = False
  CS.cruiseState.standstill = False
  return CS


def set_accel_profile(controller, profile):
  controller.sp_controller.accel_profile_controller.profile = profile
  controller.sp_controller.accel_profile_controller.frame = 1


def test_negative_target_prevents_positive_output():
  CP = make_longitudinal_car_params(vEgoStarting=0.5, startingState=False)
  CP_SP = custom.CarParamsSP.new_message()
  controller = LongControl(CP, CP_SP)
  controller.long_control_state = LongCtrlState.pid
  controller.last_output_accel = 0.2
  controller.pid.i = 1.0

  output = controller.update(True, make_car_state(v_ego=12.0, a_ego=0.3), a_target=-0.5, should_stop=False,
                             accel_limits=(-3.5, 2.0))

  assert output <= 0.05


def test_inactive_output_resets_without_ramp():
  CP = make_longitudinal_car_params(vEgoStarting=0.5, startingState=False)
  CP_SP = custom.CarParamsSP.new_message()
  controller = LongControl(CP, CP_SP)
  controller.last_output_accel = -0.3

  output = controller.update(False, make_car_state(v_ego=12.0, a_ego=0.0), a_target=0.5, should_stop=False,
                             accel_limits=(-3.5, 2.0))

  assert output == 0.0


@pytest.mark.parametrize("profile", [AccelerationProfile.eco, AccelerationProfile.normal, AccelerationProfile.sport])
def test_starting_output_ramps_from_previous_accel(profile):
  CP = make_longitudinal_car_params(startingState=True, vEgoStarting=0.5, startAccel=1.0)
  CP_SP = custom.CarParamsSP.new_message()
  controller = LongControl(CP, CP_SP)
  set_accel_profile(controller, profile)
  controller.long_control_state = LongCtrlState.starting
  controller.last_output_accel = -0.3

  output = controller.update(True, make_car_state(v_ego=0.0), a_target=0.8, should_stop=False,
                             accel_limits=(-3.5, 2.0))

  assert output == pytest.approx(-0.3 + PROFILE_CONFIGS[profile].positive_ramp_rate * DT_CTRL)
