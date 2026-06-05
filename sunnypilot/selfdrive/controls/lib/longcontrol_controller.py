"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

import numpy as np

from openpilot.common.realtime import DT_CTRL
from openpilot.sunnypilot.selfdrive.controls.lib.acceleration_controller import AccelerationProfileController

NEGATIVE_TARGET_POSITIVE_CAP_BP = [-0.6, -0.1]
NEGATIVE_TARGET_POSITIVE_CAP_V = [0.0, 0.05]


class LongControlController:
  def __init__(self):
    self.accel_profile_controller = AccelerationProfileController()

  def update_params(self) -> None:
    self.accel_profile_controller.update_params()

  def starting_output(self, start_accel: float, a_target: float) -> float:
    start_accel_min = self.accel_profile_controller.start_accel_min()
    return min(max(a_target, start_accel_min), max(start_accel, start_accel_min))

  @staticmethod
  def limit_pid_output(output_accel: float, a_target: float, error: float, CS) -> float:
    if output_accel <= 0.0:
      return output_accel
    if a_target >= -0.10 or error >= -0.35:
      return output_accel
    if CS.vEgo <= 2.5 or CS.aEgo <= 0.05:
      return output_accel

    positive_cap = np.interp(a_target, NEGATIVE_TARGET_POSITIVE_CAP_BP, NEGATIVE_TARGET_POSITIVE_CAP_V)
    return min(output_accel, float(positive_cap))

  def limit_output(self, output_accel: float, last_output_accel: float, should_stop: bool) -> float:
    if should_stop or output_accel <= last_output_accel:
      return output_accel

    return min(output_accel, last_output_accel + self.accel_profile_controller.positive_ramp_rate() * DT_CTRL)
