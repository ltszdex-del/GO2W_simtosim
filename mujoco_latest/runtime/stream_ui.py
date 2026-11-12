"""Stream-visible controls. No dependency on omni.appwindow/native keyboard."""
import time


class StreamPanel:
    def __init__(self, controller):
        import omni.ui as ui
        self.controller=controller
        self.deadline=0.
        self.notice='Ready. Click a key below to test input.'
        self.window=ui.Window('Go2W | INPUT AND STATE',width=425,height=850,position_x=1008,position_y=65)
        with self.window.frame:
            with ui.VStack(spacing=7):
                ui.Label('WASDQE | M: step/wheel | K: high/low',height=25)
                ui.Label('Vertical at COM: +UP=CCW / -DOWN=CW\nGreen dashed=ref, blue=actual; 0.30m per rad/s',height=38)
                self.policy_label=ui.Label('Policy availability: checking...',height=25)
                self.limits_label=ui.Label('Checking active limits...',word_wrap=True,height=62)
                self.label=ui.Label('Waiting for simulation...',word_wrap=True,height=314)
                ui.Label('Virtual keys: 0.3 second acceleration pulse',height=25)
                with ui.HStack(height=36):
                    for key,name in [('W','Forward'),('S','Back'),('A','Strafe L'),('D','Strafe R')]:
                        ui.Button(f'{key} {name}',clicked_fn=lambda k=key:self.press(k))
                with ui.HStack(height=36):
                    for key,name in [('Q','Turn left'),('E','Turn right')]:
                        ui.Button(f'{key} {name}',clicked_fn=lambda k=key:self.press(k))
                with ui.HStack(height=36):
                    for key,name in [('H','Handstand'),('P','Lie down'),('U','Get up')]:
                        ui.Button(f'{key} {name}',clicked_fn=lambda k=key:self.press(k))
                ui.Button('M SWITCH STEPPING / WHEEL DRIVE',height=36,clicked_fn=lambda:self.press('M'))
                ui.Button('K SWITCH HIGH / LOW SPEED',height=32,clicked_fn=lambda:self.press('K'))
                ui.Button('RESUME SIMULATION',height=32,clicked_fn=self.controller.request_resume)
                ui.Button('R RESET / U STANDING POSE',height=36,clicked_fn=self.controller.request_reset)
                ui.Button('RELEASE ALL / ZERO COMMAND',height=40,clicked_fn=self.release)
                self.feedback=ui.Label(self.notice,word_wrap=True,height=55)
    def press(self,key):
        if key=='K':
            held=self.controller.keys & set('WASDQE') if self.controller.source=='stream-virtual-keyboard' else set()
            accepted=self.controller.feed('stream-virtual-keyboard',held|{'K'})
            if accepted:self.controller.feed('stream-virtual-keyboard',held)
            self.notice='K: speed mode toggle received' if accepted else 'BUSY: another source holds input'
            return
        self.controller.release('stream-virtual-keyboard')
        if self.controller.feed('stream-virtual-keyboard',[key]):
            self.deadline=time.monotonic()+.3
            self.notice=f'RECEIVED {key} (virtual keyboard)'
            target={'H':'handstand','P':'lying_down','U':'getting_up'}.get(key)
            if target and target not in self.controller.adapters:self.notice+=' - policy NOT LOADED'
            elif key=='U' and self.controller.mode!='prone':self.notice+=' - requires PRONE start'
        else:
            self.notice='BUSY: another input source holds the lease. Release it first.'
    def reopen(self):
        self.window.visible=True
        self.notice='F1: panel reopened'
    def release(self):
        # Explicit stop button clears whichever source currently owns input.
        self.controller.release(self.controller.source)
        self.deadline=0.
        self.notice='RELEASED - command zero'
    def update(self):
        if self.deadline and time.monotonic()>=self.deadline:
            self.controller.feed('stream-virtual-keyboard',[]);self.deadline=0.
        s=self.controller.status();p=s['physics'];vx,vy,w=s['command_m_s_rad_s']
        ax,ay,az=s['acceleration_m_s2_rad_s2']
        actual=s['actual_velocity_yaw_m_s_rad_s']
        actual_text=' / '.join(f'{v:+.3f}' for v in actual) if actual else 'waiting'
        actual_yaw_text=f'{actual[2]:+.3f} rad/s' if actual is not None else 'waiting'
        error=s['heading_error_rad'];error_text=f'{error:+.3f}' if error is not None else '-'
        event=s.get('last_trigger') or '-'
        recovery=s.get('wheel_idle_recovery')
        recovery_text=f"{100*recovery['blend']:.0f}% (goal {100*recovery['goal']:.0f}%)" if recovery else 'step mode'
        loaded=', '.join(self.controller.adapters) or 'none (standing PD only)'
        self.policy_label.text=f'Loaded policies: {loaded}'
        limits=s['command_limits'];profiles=s['configured_profiles']
        profile_name='WHEEL' if s['mobility_profile']=='wheel' else 'STEPPING'
        self.limits_label.text=(f"{s['speed_mode']} {profile_name}: vx {limits['max_forward_m_s']:.2f}, yaw +/-{limits['current_yaw_rad_s']:.2f} (zero {limits['max_yaw_rad_s']:.2f})\n"
            f"HIGH STEP: {profiles['stepping']['max_forward_m_s']:.2f} / {profiles['stepping']['max_yaw_rad_s']:.2f} | WHEEL: {profiles['wheel']['max_forward_m_s']:.2f} / {profiles['wheel']['max_yaw_rad_s']:.2f}\nM: step/wheel | K: high/low (same acceleration)")

        timeline=s.get('simulation',{})
        sim_state='PAUSED - press robot key / RESUME' if timeline.get('paused') else 'RUNNING'
        self.label.text=(f"MODE: {s['mode']} | {s['mobility_profile']}     SIM: {p.get('sim_s',0):.1f} s\n"
            f"SIMULATION: {sim_state}\n"
            f"INPUT: {s['source'] or '-'}   KEYS: {' '.join(s['keys']) or '-'}\n"
            f"REF vx/vy: {vx:+.3f} / {vy:+.3f} m/s; wz: {w:+.3f} rad/s\n"
            f"ACCEL ax/ay/alpha: {ax:+.2f} / {ay:+.2f} / {az:+.2f}\n"
            f"ACTUAL vx/vy/wz (sim): {actual_text}\n"
            f"ACTUAL YAW RATE: {actual_yaw_text}\n"
            f"WHEEL STAND RECOVERY: {recovery_text}\n"
            f"HEADING ERROR: {error_text} rad (yaw-only frame)\n"
            f"INPUT STATE: {s.get('input_state','-')} | timeouts: {s['stats'].get('timeouts',0)}\n"
            f"RX packets: {s['stats'].get('input_packets',0)}   Last action: {event}\n"
            f"HEIGHT: {p.get('xyz',[0,0,0])[2]:.4f} m   TORQUE: {p.get('peak_torque_nm',0):.2f} Nm\n"
            f"POLICY: {s['policy_path']}\nPENDING: {s['pending'] or '-'}; U requires prone start")
        self.feedback.text=self.notice
