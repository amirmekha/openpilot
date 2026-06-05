"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

from types import SimpleNamespace

from cereal import car, custom
import pytest

from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.longcontrol import LongControl, LongCtrlState
from openpilot.sunnypilot.selfdrive.controls.lib.acceleration_controller import AccelerationProfile, PROFILE_CONFIGS
from openpilot.sunnypilot.selfdrive.controls.lib.longcontrol_controller import LongControlController

ALL_PROFILES = [AccelerationProfile.eco, AccelerationProfile.normal, AccelerationProfile.sport]


def make_long_control_controller(profile=AccelerationProfile.normal):
  controller = LongControlController()
  controller.accel_profile_controller.profile = profile
  return controller


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

  # positive PID output is capped to the speed/target-dependent value, not just below the global ceiling
  assert output == pytest.approx(0.01)


def test_inactive_output_resets_without_ramp():
  CP = make_longitudinal_car_params(vEgoStarting=0.5, startingState=False)
  CP_SP = custom.CarParamsSP.new_message()
  controller = LongControl(CP, CP_SP)
  controller.last_output_accel = -0.3

  output = controller.update(False, make_car_state(v_ego=12.0, a_ego=0.0), a_target=0.5, should_stop=False,
                             accel_limits=(-3.5, 2.0))

  assert output == 0.0


@pytest.mark.parametrize("profile", ALL_PROFILES)
def test_starting_output_launches_from_zero_not_stopped_accel(profile):
  CP = make_longitudinal_car_params(startingState=True, vEgoStarting=0.5, startAccel=1.0)
  CP_SP = custom.CarParamsSP.new_message()
  controller = LongControl(CP, CP_SP)
  set_accel_profile(controller, profile)
  controller.long_control_state = LongCtrlState.starting
  controller.last_output_accel = -0.3

  output = controller.update(True, make_car_state(v_ego=0.0), a_target=0.8, should_stop=False,
                             accel_limits=(-3.5, 2.0))

  # start-and-go ramps the launch up from ~0, not from the stopped-state negative accel
  assert output == pytest.approx(PROFILE_CONFIGS[profile].positive_ramp_rate * DT_CTRL)


# ---------------- LongControlController.limit_pid_output guards ----------------

def make_cs(v_ego=12.0, a_ego=0.3):
  return SimpleNamespace(vEgo=v_ego, aEgo=a_ego)


def test_limit_pid_output_passes_through_negative_output():
  c = make_long_control_controller()
  assert c.limit_pid_output(-0.5, a_target=-0.5, error=-0.8, CS=make_cs()) == -0.5


def test_limit_pid_output_passes_through_when_target_not_negative():
  c = make_long_control_controller()
  assert c.limit_pid_output(0.5, a_target=0.0, error=-0.8, CS=make_cs()) == 0.5


def test_limit_pid_output_passes_through_when_error_not_negative():
  c = make_long_control_controller()
  assert c.limit_pid_output(0.5, a_target=-0.5, error=-0.1, CS=make_cs()) == 0.5


def test_limit_pid_output_passes_through_at_low_speed():
  c = make_long_control_controller()
  assert c.limit_pid_output(0.5, a_target=-0.5, error=-0.8, CS=make_cs(v_ego=2.0)) == 0.5


def test_limit_pid_output_passes_through_when_not_accelerating():
  c = make_long_control_controller()
  assert c.limit_pid_output(0.5, a_target=-0.5, error=-0.8, CS=make_cs(a_ego=0.0)) == 0.5


def test_limit_pid_output_caps_gas_while_braking():
  c = make_long_control_controller()
  assert c.limit_pid_output(0.5, a_target=-0.5, error=-0.8, CS=make_cs()) == pytest.approx(0.01)


# ---------------- LongControlController.starting_output floor ----------------

@pytest.mark.parametrize("profile", ALL_PROFILES)
def test_starting_output_floors_at_start_accel_min(profile):
  c = make_long_control_controller(profile)
  start_accel_min = PROFILE_CONFIGS[profile].start_accel_min
  # both inputs below the floor -> floored up to start_accel_min
  assert c.starting_output(start_accel=0.0, a_target=0.0) == pytest.approx(start_accel_min)
  # a_target below floor -> raised to floor
  assert c.starting_output(start_accel=0.5, a_target=0.05) == pytest.approx(start_accel_min)
  # start_accel below floor -> floor wins
  assert c.starting_output(start_accel=0.05, a_target=0.8) == pytest.approx(start_accel_min)
  # both above floor -> a_target wins, capped by start_accel
  assert c.starting_output(start_accel=1.0, a_target=0.8) == pytest.approx(0.8)


# ---------------- LongControlController.limit_output ramp ----------------

@pytest.mark.parametrize("profile", ALL_PROFILES)
def test_limit_output_ramps_up_at_profile_rate(profile):
  c = make_long_control_controller(profile)
  rate = PROFILE_CONFIGS[profile].positive_ramp_rate
  assert c.limit_output(1.0, last_output_accel=0.0, should_stop=False) == pytest.approx(rate * DT_CTRL)


def test_limit_output_should_stop_bypasses_ramp():
  c = make_long_control_controller()
  assert c.limit_output(1.0, last_output_accel=0.0, should_stop=True) == 1.0


def test_limit_output_decel_passes_through():
  c = make_long_control_controller()
  assert c.limit_output(-0.5, last_output_accel=0.0, should_stop=False) == -0.5


def test_limit_output_within_rate_not_clamped():
  c = make_long_control_controller(AccelerationProfile.normal)
  rate = PROFILE_CONFIGS[AccelerationProfile.normal].positive_ramp_rate
  small_step = rate * DT_CTRL * 0.5
  assert c.limit_output(small_step, last_output_accel=0.0, should_stop=False) == pytest.approx(small_step)


def test_limit_output_starting_launches_from_zero():
  c = make_long_control_controller(AccelerationProfile.normal)
  rate = PROFILE_CONFIGS[AccelerationProfile.normal].positive_ramp_rate
  # is_starting floors the negative stopped-state accel to 0 so the ramp launches from ~0
  assert c.limit_output(1.0, last_output_accel=-1.0, should_stop=False, is_starting=True) == pytest.approx(rate * DT_CTRL)
