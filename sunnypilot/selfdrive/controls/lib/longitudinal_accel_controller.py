"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

import numpy as np

from opendbc.car.interfaces import ACCEL_MIN
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
  STOP_DISTANCE, get_T_FOLLOW, get_safe_obstacle_distance, get_stopped_equivalence_factor,
)
from openpilot.sunnypilot.selfdrive.controls.lib.acceleration_controller import AccelerationProfileController

COMFORT_ACCEL_MIN = -2.5
LOW_SPEED_STOP_ACCEL_MIN = -1.2
LEAD_BRAKE_LOOKAHEAD = 1.5
LEAD_BRAKE_GAIN = 0.55
LEAD_GAP_GAIN = 0.10
LEAD_SOFT_BRAKE_BUFFER = 1.5
EMERGENCY_TTC = 1.8
EMERGENCY_GAP_RATIO = 0.55
FORCE_DECEL_ACCEL_CAP = -0.05
SNG_START_ACCEL_BP = [0.0, 0.5, 2.0]


class LongitudinalComfortController:
  def __init__(self, accel_profile_controller: AccelerationProfileController):
    self.accel_profile_controller = accel_profile_controller

  @staticmethod
  def _lead_brake_cap(lead, v_ego, t_follow):
    if lead is None or not lead.status:
      return None, False

    d_rel = max(float(lead.dRel), 0.0)
    v_lead = max(float(lead.vLead), 0.0)
    closing_speed = max(float(v_ego) - v_lead, 0.0)
    lead_brake = max(-float(lead.aLeadK), 0.0)
    effective_closing_speed = closing_speed + lead_brake * LEAD_BRAKE_LOOKAHEAD

    desired_gap = get_safe_obstacle_distance(v_ego, t_follow) - get_stopped_equivalence_factor(v_lead)
    desired_gap = max(STOP_DISTANCE, float(desired_gap))
    gap_error = max(desired_gap - d_rel, 0.0)
    available_distance = max(d_rel - STOP_DISTANCE, 1.0)

    caps = []
    if effective_closing_speed > 0.1:
      caps.append(-(effective_closing_speed ** 2) / (2.0 * available_distance))
    if gap_error > 0.0 and (closing_speed > 0.1 or lead_brake > 0.2):
      caps.append(-LEAD_GAP_GAIN * gap_error)
    if lead_brake > 0.2 and d_rel < desired_gap + max(10.0, v_ego * LEAD_SOFT_BRAKE_BUFFER):
      caps.append(-LEAD_BRAKE_GAIN * lead_brake)

    if not caps:
      return None, False

    cap = float(np.clip(min(caps), ACCEL_MIN, -0.05))
    ttc = d_rel / max(closing_speed, 0.1) if closing_speed > 0.1 else float('inf')
    emergency = cap < COMFORT_ACCEL_MIN or d_rel < desired_gap * EMERGENCY_GAP_RATIO or ttc < EMERGENCY_TTC
    return cap, emergency

  def lead_start_accel(self, sm, v_ego):
    if v_ego > 0.5 or sm['controlsState'].forceDecel:
      return None

    start_accel = None
    for lead in (sm['radarState'].leadOne, sm['radarState'].leadTwo):
      if not lead.status or lead.dRel < STOP_DISTANCE - 0.1 or lead.vLead <= 0.01:
        continue
      lead_start_accel = self.accel_profile_controller.sng_start_accel(lead.vLead, SNG_START_ACCEL_BP)
      start_accel = lead_start_accel if start_accel is None else max(start_accel, lead_start_accel)

    return start_accel

  def accel_cap(self, sm, v_ego):
    t_follow = get_T_FOLLOW(sm['selfdriveState'].personality)
    force_decel = sm['controlsState'].forceDecel
    emergency = force_decel
    accel_cap = None

    for lead in (sm['radarState'].leadOne, sm['radarState'].leadTwo):
      lead_cap, lead_emergency = self._lead_brake_cap(lead, v_ego, t_follow)
      if lead_cap is not None:
        accel_cap = lead_cap if accel_cap is None else min(accel_cap, lead_cap)
      emergency |= lead_emergency

    if force_decel:
      accel_cap = FORCE_DECEL_ACCEL_CAP if accel_cap is None else min(accel_cap, FORCE_DECEL_ACCEL_CAP)

    return accel_cap, emergency

  def apply(self, output_a_target, output_should_stop, sm, v_ego):
    accel_cap, emergency = self.accel_cap(sm, v_ego)
    if accel_cap is not None:
      output_a_target = min(output_a_target, accel_cap)

    min_accel = ACCEL_MIN if emergency else COMFORT_ACCEL_MIN
    if output_should_stop and v_ego < 5.0 and not emergency:
      min_accel = max(min_accel, float(np.interp(v_ego, [0.0, 5.0], [-0.45, LOW_SPEED_STOP_ACCEL_MIN])))

    return max(output_a_target, min_accel)


class LongitudinalAccelerationController:
  def __init__(self, mpc):
    self.mpc = mpc
    self.accel_profile_controller = AccelerationProfileController()
    self.comfort_controller = LongitudinalComfortController(self.accel_profile_controller)

  def update_params(self) -> None:
    self.accel_profile_controller.update_params()

  def max_accel(self, v_ego: float) -> float:
    return self.accel_profile_controller.max_accel(v_ego)

  def update_mpc_targets(self, sm, v_ego: float, v_cruise: float, a_desired: float) -> tuple[float, float]:
    self.mpc.set_cruise_max_accel(self.accel_profile_controller.mpc_cruise_max_accel())
    accel_cap, _ = self.comfort_controller.accel_cap(sm, v_ego)
    if accel_cap is not None:
      a_desired = min(a_desired, accel_cap)

    return v_cruise, a_desired

  def update_accel_target(self, sm, v_ego: float, output_a_target: float,
                          output_should_stop: bool) -> tuple[float, bool]:
    start_accel = self.comfort_controller.lead_start_accel(sm, v_ego)
    if start_accel is not None:
      output_should_stop = False
      output_a_target = max(output_a_target, start_accel)

    output_a_target = self.comfort_controller.apply(output_a_target, output_should_stop, sm, v_ego)
    return output_a_target, output_should_stop

  def cereal_accel_personality(self):
    return self.accel_profile_controller.cereal_accel_personality()
