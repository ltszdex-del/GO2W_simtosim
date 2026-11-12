"""Editable test runtime, preserving the archived deployment snapshot."""
import bootstrap
import shutil,json
from pathlib import Path
source=bootstrap.SNAPSHOT/'interactive_latest_test_20261110'
dest=bootstrap.ROOT/'runtime'
if not dest.exists():
    dest.mkdir()
    for p in source.iterdir():
        if p.is_file():shutil.copy2(p,dest/p.name)
        elif p.name.endswith('_runtime'):shutil.copytree(p,dest/p.name)
c=json.loads((source/'config.json').read_text())
limit_keys=('min_forward_m_s','max_forward_m_s','max_lateral_m_s','max_yaw_rad_s','yaw_cruise_limit_rad_s')
for profile in (c,c['wheel_command_profile']):
    for key in limit_keys:profile[key]*=.5
c['low_speed_hold'].update(enabled=True,reference_enter_speed_m_s=.001,
    actual_enter_speed_m_s=.15,reference_exit_speed_m_s=.12,
    reference_yaw_enter_rad_s=.001,reference_yaw_exit_rad_s=.12,
    actual_enter_yaw_rad_s=.15,settle_s=.1)
c['fixed_low_limits']=True
for key in ('locomotion','wheel_locomotion'):c['policies'][key]['command_limit_scale']=.5
from local_policy_deployment import apply_deployment
apply_deployment(c)
(dest/'config.json').write_text(json.dumps(c,indent=2),encoding='utf-8')
p=dest/'control.py';s=(source/p.name).read_text(encoding='utf-8');s=s.replace("self.integrator.speed_scale=.5 if self.integrator.speed_scale==1. else 1.","self.integrator.speed_scale=1. if self.cfg.get('fixed_low_limits') else (.5 if self.integrator.speed_scale==1. else 1.)")
p.write_text(s,encoding='utf-8')
p=dest/'control.py';s=p.read_text(encoding='utf-8')
s=s.replace("        self._activate('standing_hold',state,preserve_command=True)",
    "        self.adapters['standing_hold'].local_pu_scope=(self.pending!='handstand' and self.mode!='handstand_return')\n        self._activate('standing_hold',state,preserve_command=True)")
s=s.replace("                selection='locomotion' if self.mode=='standing_hold' else None",
    "                explicit_pu_hold=(self.mode=='standing_hold' and\n                    getattr(getattr(self.active,'ctx',None),'local_pu_isolation',False))\n                if not explicit_pu_hold:selection='locomotion' if self.mode=='standing_hold' else None")
p.write_text(s,encoding='utf-8')
p=dest/'control.py';s=p.read_text(encoding='utf-8').replace("speed_mode='LOW' if self.integrator.speed_scale==.5 else 'HIGH'", "speed_mode='LOW' if self.cfg.get('fixed_low_limits') or self.integrator.speed_scale==.5 else 'HIGH'")
s=s.replace("if key=='K':", "if key=='K' and self.cfg.get('fixed_low_limits'):\n                    self.message='LOW limits fixed';continue\n                if key=='K':")
s=s.replace("self._enter_standing_hold(state);self.message='M: '+profile", """if self.cfg.get('low_speed_hold',{}).get('enabled',False):
            self._enter_standing_hold(state)
        else:
            self._activate('locomotion',state)
        self.message='M: '+profile""")
s=s.replace("if self.mobility_profile=='wheel' and selection=='standing_hold':", """# Explicit U/H-return hold must still yield to a new movement command,
            # even when automatic parking-to-U transitions are disabled.
            if (self.mode=='standing_hold' and not self.pending and not self.pending_mobility
                    and max(abs(v) for v in self.command)>.1):
                selection='locomotion'
            if self.mobility_profile=='wheel' and selection=='standing_hold':""")
p.write_text(s,encoding='utf-8')
p=dest/'step_followup_adapter.py';s=(source/p.name).read_text(encoding='utf-8');s=s.replace('c=env.control;m=self.command_contract', '''c=env.control;m=dict(self.command_contract)
        scale=self.spec.get('command_limit_scale',1.)
        m['speed_limits']=[v*scale for v in m['speed_limits']]
        for key in ('min_forward_speed','yaw_speed_limit'):m[key]*=scale''');p.write_text(s,encoding='utf-8')
p=dest/'step_followup_adapter.py';s=p.read_text(encoding='utf-8')
s=s.replace("((2.3,1.3),(9.2,5.2)))", "((2.3,1.3),(9.2,5.2)),((1.15,.65),(9.2,5.2)))")
s=s.replace("in (-limits[0],-1.5)", "in (-limits[0],-1.5,-.75)")
p.write_text(s,encoding='utf-8')
p=dest/'wheel_mirror_brake_adapter.py';s=(source/p.name).read_text(encoding='utf-8');s=s.replace("self.command_profile.update(wheel_envelope_lever=0.,wheel_reference_speed_m_s=0.)", "self.command_profile.update(wheel_envelope_lever=0.,wheel_reference_speed_m_s=0.)\n        for key in "+repr(limit_keys)+":self.command_profile[key]*=spec.get('command_limit_scale',1.)")
p.write_text(s,encoding='utf-8')
# Reproducible local P/U adapter changes, checked against the archived source.
import hashlib
patch=json.loads((bootstrap.ROOT/'pu_adapter_edits.json').read_text(encoding='utf-8'))
p=source/'pu_adapter.py'
assert hashlib.sha256(p.read_bytes()).hexdigest()==patch['source_sha256']
s=p.read_text(encoding='utf-8')
for edit in patch['edits']:
    assert s.count(edit['before'])==1,edit['before']
    s=s.replace(edit['before'],edit['after'],1)
(dest/'pu_adapter.py').write_text(s,encoding='utf-8')
# Preserve automatic standing and local H integration fixes on regeneration.
for patch_file in ('standing_hold_edits.json','handstand_edits.json'):
    patches=json.loads((bootstrap.ROOT/patch_file).read_text(encoding='utf-8'))
    for name,patch in patches.items():
        p=dest/name;s=p.read_text(encoding='utf-8')
        if hashlib.sha256(s.encode()).hexdigest()==patch['target_sha256']:continue
        assert hashlib.sha256(s.encode()).hexdigest()==patch['source_sha256'],name
        for edit in patch['edits']:
            assert s.count(edit['before'])==1,(name,edit['before'])
            s=s.replace(edit['before'],edit['after'],1)
        assert hashlib.sha256(s.encode()).hexdigest()==patch['target_sha256'],name
        p.write_text(s,encoding='utf-8')
student_adapter=bootstrap.ROOT/'handstand_distill_20261112/handstand_student_adapter.py'
if student_adapter.exists():shutil.copy2(student_adapter,dest/student_adapter.name)
print(dest)
