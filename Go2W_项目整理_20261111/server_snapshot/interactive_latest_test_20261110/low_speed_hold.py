"""Policy-independent low-speed standing gate, staged off until a hold adapter is ready."""
import math


class LowSpeedHold:
    def __init__(self, cfg):
        self.cfg = cfg
        self.requested = False
        self.reason = 'disabled'

    def select(self, mode, reference, actual, ready, pending=False):
        self.requested = False
        if not self.cfg.get('enabled', False):
            self.reason = 'disabled'
            return None
        if mode not in ('standing', 'locomotion', 'standing_hold') or pending:
            self.reason = 'mode_transition_or_ineligible'
            return None
        ref = math.hypot(*reference[:2])
        if not math.isfinite(ref):
            self.reason = 'nonfinite_velocity'
            return None
        enter = self.cfg.get('reference_enter_speed_m_s', .1)
        actual_enter = self.cfg.get('actual_enter_speed_m_s', .4)
        leave = self.cfg.get('reference_exit_speed_m_s', .1)
        if min(enter,actual_enter,leave)<=0:
            raise ValueError('Speed thresholds must be positive')
        if mode == 'standing_hold':
            if ref > leave or abs(reference[2]) > self.cfg.get('reference_yaw_exit_rad_s',.1):
                self.reason = 'resume_locomotion'
                return 'locomotion'
            self.reason = 'holding'
            return None
        if actual is None or len(actual)<2:
            self.reason='actual_velocity_unavailable'
            return None
        speed=math.hypot(*actual[:2])
        if not math.isfinite(speed):
            self.reason='nonfinite_velocity'
            return None
        self.requested = (ref < enter and speed < actual_enter
            and abs(reference[2]) < self.cfg.get('reference_yaw_enter_rad_s',.1)
            ) # Angular stability is checked by the dwell gate, not preparation entry.
        if self.requested:
            self.reason = 'enter_standing_hold' if ready else 'standing_hold_adapter_missing'
            return 'standing_hold' if ready else None
        self.reason = 'outside_deadzone'
        return None

    def status(self, ready):
        return dict(enabled=self.cfg.get('enabled', False), adapter='standing_hold',
                    ready=ready, requested=self.requested, reason=self.reason,
                    reference_enter_speed_m_s=self.cfg.get('reference_enter_speed_m_s', .1),
                    actual_enter_speed_m_s=self.cfg.get('actual_enter_speed_m_s', .4),
                    reference_exit_speed_m_s=self.cfg.get('reference_exit_speed_m_s', .1),
                    speed_definition='horizontal Euclidean norm; m/s')
