"""Enter standing only after commanded and measured motion have settled."""
import math


class LowSpeedHold:
    def __init__(self, cfg):
        self.cfg = cfg
        self.requested = False
        self.reason = 'disabled'
        self.settled_s=0.

    def select(self, mode, reference, actual, ready, pending=False,dt=.02,motion_requested=False):
        self.requested = False
        if not self.cfg.get('enabled', False):
            self.settled_s=0.
            self.reason = 'disabled'
            return None
        if mode not in ('standing', 'locomotion', 'standing_hold') or pending:
            self.settled_s=0.
            self.reason = 'mode_transition_or_ineligible'
            return None
        ref = math.hypot(*reference[:2])
        if not math.isfinite(ref) or not math.isfinite(reference[2]):
            self.settled_s=0.
            self.reason = 'nonfinite_velocity'
            return None
        enter = self.cfg.get('reference_enter_speed_m_s', .1)
        actual_enter = self.cfg.get('actual_enter_speed_m_s', .4)
        leave = self.cfg.get('reference_exit_speed_m_s', .1)
        if min(enter,actual_enter,leave)<=0:
            raise ValueError('Speed thresholds must be positive')
        if mode == 'standing_hold':
            self.settled_s=0.
            if motion_requested or ref > leave or abs(reference[2]) > self.cfg.get('reference_yaw_exit_rad_s',.1):
                self.reason = 'resume_locomotion'
                return 'locomotion'
            self.reason = 'holding'
            return None
        if motion_requested:
            self.settled_s=0.;self.reason='movement_key_held'
            return None
        if actual is None or len(actual)<3:
            self.settled_s=0.
            self.reason='actual_velocity_unavailable'
            return None
        speed=math.hypot(*actual[:2])
        if not math.isfinite(speed) or not math.isfinite(actual[2]):
            self.settled_s=0.
            self.reason='nonfinite_velocity'
            return None
        eligible = (ref < enter and speed < actual_enter
            and abs(reference[2]) < self.cfg.get('reference_yaw_enter_rad_s',.1)
            and abs(actual[2]) < self.cfg.get('actual_enter_yaw_rad_s',.15))
        self.settled_s=self.settled_s+dt if eligible else 0.
        self.requested=eligible and self.settled_s+1.e-9>=self.cfg.get('settle_s',0.)
        if self.requested:
            self.reason = 'enter_standing_hold' if ready else 'standing_hold_adapter_missing'
            return 'standing_hold' if ready else None
        self.reason = 'settling' if eligible else 'outside_deadzone'
        return None

    def status(self, ready):
        return dict(enabled=self.cfg.get('enabled', False), adapter='standing_hold',
                    ready=ready, requested=self.requested, reason=self.reason,
                    reference_enter_speed_m_s=self.cfg.get('reference_enter_speed_m_s', .1),
                    actual_enter_speed_m_s=self.cfg.get('actual_enter_speed_m_s', .4),
                    reference_exit_speed_m_s=self.cfg.get('reference_exit_speed_m_s', .1),
                    actual_enter_yaw_rad_s=self.cfg.get('actual_enter_yaw_rad_s',.15),
                    settle_s=self.cfg.get('settle_s',0.),settled_s=round(self.settled_s,3),
                    speed_definition='horizontal Euclidean norm; m/s')
