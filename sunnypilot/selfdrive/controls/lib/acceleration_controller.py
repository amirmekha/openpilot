"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

from dataclasses import dataclass

import numpy as np

from cereal import custom
from openpilot.common.params import Params, UnknownKeyName

ACCELERATION_PROFILE_PARAM = "AccelPersonality"
ACCELERATION_PROFILE_ENABLED_PARAM = "AccelPersonalityEnabled"
ACCEL_MAX_BP = [0.0, 10.0, 25.0, 40.0]

AccelerationProfile = custom.LongitudinalPlanSP.AccelerationPersonality
ACCELERATION_PROFILE_VALUES = tuple(AccelerationProfile.schema.enumerants.values())


@dataclass(frozen=True)
class AccelerationProfileConfig:
  accel_max_v: list[float]
  mpc_cruise_max_accel: float
  positive_ramp_rate: float
  start_accel_min: float
  sng_start_accel_v: list[float]


PROFILE_CONFIGS = {
  AccelerationProfile.eco: AccelerationProfileConfig(
    accel_max_v=[1.0, 0.85, 0.60, 0.45],
    mpc_cruise_max_accel=0.9,
    positive_ramp_rate=1.2,
    start_accel_min=0.08,
    sng_start_accel_v=[0.015, 0.08, 0.30],
  ),
  AccelerationProfile.normal: AccelerationProfileConfig(
    accel_max_v=[1.6, 1.2, 0.80, 0.60],
    mpc_cruise_max_accel=1.6,
    positive_ramp_rate=2.0,
    start_accel_min=0.15,
    sng_start_accel_v=[0.02, 0.12, 0.45],
  ),
  AccelerationProfile.sport: AccelerationProfileConfig(
    accel_max_v=[1.8, 1.4, 1.00, 0.75],
    mpc_cruise_max_accel=1.8,
    positive_ramp_rate=3.0,
    start_accel_min=0.20,
    sng_start_accel_v=[0.03, 0.18, 0.60],
  ),
}


class AccelerationProfileController:
  def __init__(self, params: Params | None = None):
    self.params = params or Params()
    self.frame = 0
    self.profile = AccelerationProfile.normal
    self.read_params()

  @property
  def config(self) -> AccelerationProfileConfig:
    return PROFILE_CONFIGS[self.profile]

  def read_params(self) -> None:
    try:
      if not self.params.get_bool(ACCELERATION_PROFILE_ENABLED_PARAM):
        self.profile = AccelerationProfile.normal
        return

      profile = self.params.get(ACCELERATION_PROFILE_PARAM, return_default=True)
      if profile not in ACCELERATION_PROFILE_VALUES:
        # Corrupted/out-of-range value: reset to the documented default rather than clipping to an
        # arbitrary enum endpoint (which would depend on enum ordering).
        profile = AccelerationProfile.normal
        self.params.put(ACCELERATION_PROFILE_PARAM, int(profile), block=True)
    except UnknownKeyName:
      profile = AccelerationProfile.normal

    self.profile = profile

  def update_params(self) -> None:
    if self.frame % 50 == 0:
      self.read_params()
    self.frame += 1

  def max_accel(self, v_ego: float) -> float:
    return float(np.interp(v_ego, ACCEL_MAX_BP, self.config.accel_max_v))

  def mpc_cruise_max_accel(self) -> float:
    return self.config.mpc_cruise_max_accel

  def positive_ramp_rate(self) -> float:
    return self.config.positive_ramp_rate

  def start_accel_min(self) -> float:
    return self.config.start_accel_min

  def sng_start_accel(self, v_lead: float, bp: list[float]) -> float:
    return float(np.interp(v_lead, bp, self.config.sng_start_accel_v))

  def cereal_accel_personality(self):
    return self.profile
