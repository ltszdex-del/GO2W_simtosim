"""Three-channel input, safe mode transitions and exclusive adapter execution."""
import collections
import threading
import time
from typing import Protocol
from commands import CommandIntegrator,ALL_KEYS,MOTION_KEYS,wrap,PROFILE_KEYS
from low_speed_hold import LowSpeedHold

class PolicyAdapter(Protocol):
    manifest:dict
    path:str
    def load(self):...
    def reset(self,state):...
    def observe(self,state):...
    def infer(self,observation):...
    def apply(self,action,dt):...
    def is_complete(self,state):...

class Controller:
    def __init__(self,cfg,clock=time.monotonic):
        self.cfg,self.clock=cfg,clock
        self.lock=threading.RLock()
        self.keys,self.events=set(),collections.deque()
        self.source,self.last_input=None,-1e9
        self.mode='standing';self.message='运动、倒立、趴下、起立策略未加载；当前为站立PD'
        self.integrator=CommandIntegrator(cfg);self.command=self.integrator.velocity
        self.adapters={};self.active=None;self.pending=None;self.stable_time=0.
        self.stats=collections.Counter();self.physics={};self.last_trigger=None;self.input_state='IDLE'
        self.contract=None;self.state={}
        self.policy_contracts={}
        self.mobility_profile='stepping';self.pending_mobility=None;self.step_adapter=None
        self.step_control={k:cfg[k] for k in PROFILE_KEYS}
        self.policy_errors={}
        self.reset_requested=False
        self.resume_requested=False
        self.simulation_state={}
        self.low_speed_hold=LowSpeedHold(cfg.get('low_speed_hold',{}))
    def request_reset(self):
        with self.lock:
            self.reset_requested=True
            self.resume_requested=True
    def request_resume(self):
        with self.lock:
            self.resume_requested=True
    def reset_runtime(self,state):
        with self.lock:
            self.keys.clear();self.events.clear();self.pending=None;self.active=None
            self.source=None;self.last_input=-1e9;self.stable_time=0.
            self.state=state;self.mode='standing';self.input_state='IDLE'
            self.integrator.stop(state.get('yaw_rad'));self.reset_requested=False
            self.pending_mobility=None
            self.last_trigger='R';self.stats['resets']+=1
            self.low_speed_hold=LowSpeedHold(self.cfg.get('low_speed_hold',{}))
            self.message='R: reset to U standing pose'
    def _set_mobility(self,profile,state):
        if self.step_adapter is None:self.step_adapter=self.adapters['locomotion']
        self.adapters['locomotion']=self.adapters['wheel_locomotion'] if profile=='wheel' else self.step_adapter
        self.mobility_profile=profile
        self.cfg.update(self.step_control)
        if profile=='wheel':self.cfg.update(self.adapters['wheel_locomotion'].command_profile)
        self.integrator.stop(state.get('yaw_rad'));self.keys.difference_update(MOTION_KEYS)
        if self.cfg.get('low_speed_hold',{}).get('enabled',False):
            self._enter_standing_hold(state)
        else:
            self._activate('locomotion',state)
        self.message='M: '+profile
    def register(self,mode,adapter):
        required={'observation_order','normalization','scales','history','phase','previous_action',
                  'target','task_id','joint_order','output','pd','limits','timing','asset_sha256'}
        if not required<=adapter.manifest.keys():raise ValueError('Incomplete policy manifest')
        if self.contract is None:raise ValueError('Bind actual environment contract before registering policy')
        for key in ('joint_order','output','pd','limits','timing','asset_sha256'):
            if adapter.manifest[key]!=self.policy_contracts.get(mode,self.contract)[key]:raise ValueError(f'Policy/environment mismatch: {key}')
        adapter.load();self.adapters[mode]=adapter
    def feed(self,source,keys,confirmed_held=False):
        keys={k.upper() for k in keys}&ALL_KEYS
        if 'R' in keys:
            self.request_reset()
            return True
        with self.lock:
            now=self.clock()
            if self.source!=source and now-self.last_input<self.cfg['input_timeout_s']:return False
            if keys:self.resume_requested=True
            live_native=confirmed_held and source=='webrtc-keyboard'
            if self.source!=source or (now-self.last_input>=self.cfg['input_timeout_s'] and not live_native):
                self.integrator.stop(self.state.get('yaw_rad'))
                self.keys.clear()
            for key in sorted(keys-self.keys):
                if key in 'HPUMK':self.events.append((key,now,source))
            self.keys=keys;self.source=source;self.last_input=now
            self.input_state='ACTIVE' if keys else 'BRAKING'
            self.stats['input_packets']+=1
            return True
    def release(self,source):
        """Emergency/focus release; ordinary keyup uses feed with fewer keys."""
        with self.lock:
            if source==self.source:
                self.keys.clear();self.events.clear();self.pending=None
                self.integrator.stop(self.state.get('yaw_rad'))
                self.source=None;self.last_input=-1e9
                self.input_state='EMERGENCY RELEASE';self.stats['releases']+=1
    def _activate(self,target,state,preserve_command=False):
        adapter=self.adapters[target]
        adapter.reset(state)
        self.active=adapter;self.mode=target;self.pending=None
        if not preserve_command:
            self.integrator.stop(state.get('yaw_rad'))
            self.keys.difference_update(MOTION_KEYS)
        self.message=f'执行 {target}'
    def _enter_standing_hold(self,state):
        """Shared endpoint for U completion and the low-speed gate; no get-up replay."""
        if 'standing_hold' not in self.adapters:return False
        self.adapters['standing_hold'].local_pu_scope=True
        self._activate('standing_hold',state,preserve_command=True)
        return True
    def tick(self,dt,state=None):
        with self.lock:
            if state is not None:self.state=state
            state=self.state;now=self.clock();yaw=state.get('yaw_rad')
            if self.keys and now-self.last_input>self.cfg['input_timeout_s']:
                self.keys.clear();self.events.clear();self.pending=None
                self.integrator.stop(yaw);self.source=None
                self.input_state='TIMED OUT - safe stop';self.stats['timeouts']+=1
            enabled=self.mode in ('standing','locomotion','standing_hold')
            effective=self.keys if not self.pending and not self.pending_mobility else set()
            if self.mobility_profile=='wheel':effective=effective & set('WSQE')
            self.integrator.step(effective,dt,yaw,enabled)
            while self.events:
                key,stamp,source=self.events.popleft()
                if now-stamp>self.cfg['input_timeout_s'] or source!=self.source:continue
                self.last_trigger=key;self.stats['trigger_'+key]+=1
                if key=='K' and self.cfg.get('fixed_low_limits'):
                    self.message='LOW limits fixed';continue
                if key=='K':
                    self.integrator.speed_scale=1. if self.cfg.get('fixed_low_limits') else (.5 if self.integrator.speed_scale==1. else 1.)
                    self.message='K: '+('LOW / half speed limits' if self.integrator.speed_scale==.5 else 'HIGH / full speed limits')
                    continue
                if key=='M':
                    if self.mode not in ('standing','locomotion','standing_hold') or self.pending:self.message='M: 先返回站立再切换'
                    elif 'wheel_locomotion' not in self.adapters:self.message='M: 轮驱权重尚未可用'
                    else:self.pending_mobility='wheel' if self.mobility_profile=='stepping' else 'stepping'
                    continue
                target,start={'H':('handstand','standing'),'P':('lying_down','standing'),'U':('getting_up','prone')}[key]
                if key=='H' and self.mode=='handstand':target,start='handstand_return','handstand'
                supported=self.mode==start or (start=='standing' and self.mode in ('locomotion','standing_hold'))
                if key=='U' and self.mode=='lying_down' and self.active and hasattr(self.active,'can_start_getup'):
                    supported=self.active.can_start_getup(state)
                if not supported:self.message=f'{key}: 当前 {self.mode} 不支持此转换（需要 {start}）'
                elif self.pending:self.message='减速/等待稳定中，不重复触发'
                elif target not in self.adapters:self.message=f'{key}: 未加载 {target}；保持当前控制'
                elif start=='standing':
                    self.pending=target;self.stable_time=0.;self.keys.difference_update(MOTION_KEYS)
                    self.message='先制动，等待真实速度稳定后切换'
                else:self._activate(target,state)
            if self.pending_mobility:
                actual=state.get('actual_velocity_yaw_m_s_rad_s',[float('inf')]*3)
                if max(abs(v) for v in self.command)<.01 and sum(v*v for v in actual[:2])<.15**2 and abs(actual[2])<.15:
                    profile=self.pending_mobility;self.pending_mobility=None;self._set_mobility(profile,state)
            preparing_p=False
            if self.pending:
                actual=state.get('actual_velocity_yaw_m_s_rad_s',[float('inf')]*3)
                prepare=max(abs(v) for v in self.command)<.01 and sum(v*v for v in actual[:2])<.15**2
                if (self.pending=='lying_down' and self.active is not self.adapters.get('standing_hold')
                        and self.cfg.get('low_speed_hold',{}).get('enabled',False)):
                    preparing_p=True
                    prepare=self.low_speed_hold.select(self.mode,self.command,actual,
                        'standing_hold' in self.adapters,dt=dt)=='standing_hold'
                if prepare:
                    if 'standing_hold' in self.adapters and self.active is not self.adapters['standing_hold']:
                        pending=self.pending
                        self._enter_standing_hold(state)
                        self.pending=pending
                    elif self.pending=='handstand' and 'standing_hold' not in self.adapters:self.active=None
                stable=max(abs(v) for v in self.command)<.01 and max(abs(v) for v in actual)<.05
                if self.active is self.adapters.get('standing_hold'):
                    stable=max(abs(v) for v in self.command)<.01 and sum(v*v for v in actual[:2])<.10**2 and abs(actual[2])<.35
                candidate=self.adapters[self.pending]
                if (self.pending=='lying_down' and self.active is self.adapters.get('standing_hold')
                        and hasattr(self.active,'ready_for_handoff')):
                    stable=stable and self.active.ready_for_handoff(state)
                if hasattr(candidate,'ready_to_start'):stable=stable and candidate.ready_to_start(state)
                self.stable_time=self.stable_time+dt if stable else 0.
                if self.stable_time>=.3:self._activate(self.pending,state)
            selection=None if preparing_p else self.low_speed_hold.select(self.mode,self.command,
                state.get('actual_velocity_yaw_m_s_rad_s'),'standing_hold' in self.adapters,
                bool(self.pending or self.pending_mobility),dt=dt,motion_requested=bool(effective&MOTION_KEYS))
            # Explicit U/H-return hold must still yield to a new movement command,
            # even when automatic parking-to-U transitions are disabled.
            if (self.mode=='standing_hold' and not self.pending and not self.pending_mobility
                    and max(abs(v) for v in self.command)>.1):
                selection='locomotion'
            if self.mobility_profile=='wheel' and selection=='standing_hold':
                actual=state.get('actual_velocity_yaw_m_s_rad_s',[0.,0.,float('inf')])
                if abs(actual[2])>.12:selection=None
            # The wheel policy owns zero-command braking and recovery.
            if (self.mobility_profile=='wheel' and self.cfg.get('wheel_policy_handles_stopping',False)
                    and not self.cfg.get('low_speed_hold',{}).get('enabled',False) and not self.pending):
                explicit_pu_hold=(self.mode=='standing_hold' and
                    getattr(getattr(self.active,'ctx',None),'local_pu_isolation',False))
                if not explicit_pu_hold:selection='locomotion' if self.mode=='standing_hold' else None
            if selection=='standing_hold':self._enter_standing_hold(state)
            elif selection=='locomotion' and 'locomotion' in self.adapters:
                self._activate('locomotion',state,preserve_command=True)
            if self.mode in ('standing','locomotion') and not self.pending:
                if 'locomotion' in self.adapters and self.active is None:self._activate('locomotion',state)
            if self.input_state=='BRAKING' and not any(self.command):self.input_state='STOPPED'
    def execute(self,state,dt):
        """One adapter owns all motor targets; called once per policy step."""
        with self.lock:
            if self.active is None:return None
            obs=self.active.observe(state)
            return self.active.apply(self.active.infer(obs),dt)
    def after_step(self,state):
        with self.lock:
            if self.active and hasattr(self.active,'after_physics') and self.active.after_physics(state):
                self.message=getattr(self.active,'failure','Policy guard failed')
                self.active=None;self.mode='error';self.integrator.stop(state.get('yaw_rad'));self.keys.clear()
                return 'error'
            if self.active and self.mode in ('lying_down','getting_up','handstand','handstand_return') and self.active.is_complete(state):
                previous=self.mode
                self.active=None
                self.mode={'lying_down':'prone','getting_up':'standing'}.get(previous,previous)
                self.integrator.stop(state.get('yaw_rad'));self.keys.clear()
                if previous in ('getting_up','handstand_return') and self._enter_standing_hold(state):return 'standing_hold'
                self.message=f'{previous} 完成；当前实际电机目标保持'
                return self.mode
            return None
    def status(self):
        with self.lock:
            yaw=self.state.get('yaw_rad')
            heading=self.integrator.yaw_target
            return dict(schema_version=2,command_frame='body_yaw_horizontal',mode=self.mode,message=self.message,
                mobility_profile=self.mobility_profile,pending_mobility=self.pending_mobility,source=self.source,keys=sorted(self.keys),command_m_s_rad_s=self.command[:],
                command_axes=['vx','vy','wz'],acceleration_m_s2_rad_s2=self.integrator.acceleration[:],
                command_limits=self.integrator.limits(),speed_mode='LOW' if self.cfg.get('fixed_low_limits') or self.integrator.speed_scale==.5 else 'HIGH',
                configured_profiles={'stepping':dict(self.step_control),'wheel':dict(self.cfg['wheel_command_profile'])},
                actual_velocity_yaw_m_s_rad_s=self.state.get('actual_velocity_yaw_m_s_rad_s'),
                wheel_idle_recovery=dict(enabled=self.cfg.get('wheel_idle_assist',False),
                    blend=getattr(self.active,'idle_blend',0.),goal=getattr(self.active,'idle_goal',0.),
                    requested=getattr(self.active,'idle_requested',False),
                    actual_speed=getattr(self.active,'idle_actual_speed',None),actual_yaw=getattr(self.active,'idle_actual_yaw',None)) if self.mobility_profile=='wheel' else None,
                actual_velocity_source='simulation_root_truth; actor must explicitly declare measurement interface',
                yaw_target_rad=heading,yaw_actual_rad=yaw,
                heading_error_rad=wrap(heading-yaw) if heading is not None and yaw is not None else None,
                input_age_s=round(min(999.,self.clock()-self.last_input),3),
                policy_path=self.active.path if self.active else 'builtin://pose-pd-hold',
                policies=self.cfg['policies'],policy_errors=dict(self.policy_errors),stats=dict(self.stats),physics=self.physics,
                simulation=dict(self.simulation_state),
                low_speed_hold=self.low_speed_hold.status('standing_hold' in self.adapters),
                last_trigger=self.last_trigger,input_state=self.input_state,pending=self.pending)
