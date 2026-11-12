from pathlib import Path
import datetime,json,shutil,sys,torch,os,signal,time,math,importlib
from types import SimpleNamespace as NS
live=Path('/home/ubuntu/Go2W/interactive_latest_test_20261110');sys.path.insert(0,str(live))
from commands import profile_from_contract
cfg=json.loads((live/'config.json').read_text())
for mode in ('locomotion','wheel_locomotion'):
    spec=cfg['policies'][mode];ck=torch.load(spec['checkpoint'],map_location='cpu',weights_only=False)
    profile=profile_from_contract(ck['manifest']['command'],ck['manifest']['control'])
    profile.update(wheel_envelope_lever=0.,wheel_reference_speed_m_s=0.)
    if mode=='locomotion':cfg.update(profile)
    else:cfg['wheel_command_profile']=profile
assert cfg['max_forward_m_s']==2.3 and cfg['max_yaw_rad_s']==1.6
assert cfg['wheel_command_profile']['max_forward_m_s']==3. and cfg['wheel_command_profile']['max_yaw_rad_s']==2.
control=(live/'control.py').read_text()
needle="                command_axes=['vx','vy','wz'],acceleration_m_s2_rad_s2=self.integrator.acceleration[:],"
assert needle in control
control=control.replace(needle,needle+"\n                command_limits={k:self.integrator.cfg[k] for k in PROFILE_KEYS},\n                configured_profiles={'stepping':dict(self.step_control),'wheel':dict(self.cfg['wheel_command_profile'])},")
panel=(live/'stream_ui.py').read_text()
panel=panel.replace("width=425,height=700","width=425,height=780")
needle="                self.policy_label=ui.Label('Policy availability: checking...',height=25)"
assert needle in panel
panel=panel.replace(needle,needle+"\n                self.limits_label=ui.Label('Checking active limits...',word_wrap=True,height=62)")
needle="        self.policy_label.text=f'Loaded policies: {loaded}'"
assert needle in panel
panel=panel.replace(needle,needle+'''
        limits=s['command_limits'];profiles=s['configured_profiles']
        profile_name='WHEEL' if s['mobility_profile']=='wheel' else 'STEPPING'
        self.limits_label.text=(f"ACTIVE {profile_name}: forward {limits['max_forward_m_s']:.1f} m/s, yaw +/-{limits['max_yaw_rad_s']:.1f} rad/s\\n"
            f"STEP: {profiles['stepping']['max_forward_m_s']:.1f} / {profiles['stepping']['max_yaw_rad_s']:.1f} | WHEEL: {profiles['wheel']['max_forward_m_s']:.1f} / {profiles['wheel']['max_yaw_rad_s']:.1f}\\nM: switch stepping / wheel")
''')
old='            f"LIMIT vx/vy/wz: {self.controller.cfg[\'max_forward_m_s\']} / {self.controller.cfg[\'max_lateral_m_s\']} / {self.controller.cfg[\'max_yaw_rad_s\']}\\n"\n'
assert old in panel;panel=panel.replace(old,'')
html=(live/'controls_panel.html').read_text()
needle='`模式：${s.mode} ｜ 按键：'
assert needle in html
html=html.replace(needle,'`当前模式：${s.mobility_profile} ｜ 前进上限 ${s.command_limits.max_forward_m_s} m/s ｜ 转向上限 ±${s.command_limits.max_yaw_rad_s} rad/s\\n状态：${s.mode} ｜ 按键：')
compile(panel,'stream_ui.py','exec');compile(control,'control.py','exec')
namespace={'__name__':'limits_test'};exec(compile(control,'control.py','exec'),namespace)
c=namespace['Controller'](json.loads(json.dumps(cfg)));c.adapters={'locomotion':NS(), 'wheel_locomotion':NS(command_profile=cfg['wheel_command_profile'])}
c._enter_standing_hold=lambda state:None
results=[]
for mode,forward,yaw in [('stepping',2.3,1.6),('wheel',3.,2.),('stepping',2.3,1.6)]:
    c._set_mobility(mode,{'yaw_rad':0.})
    for _ in range(400):c.integrator.step({'W','Q'},.02)
    assert math.isclose(c.command[0],forward,abs_tol=1e-8) and math.isclose(c.command[2],yaw,abs_tol=1e-8),c.command
    assert c.status()['command_limits']['max_yaw_rad_s']==yaw
    results.append(dict(mode=mode,reference_vx=c.command[0],reference_yaw=c.command[2]))
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
backup=live/'backups'/('explicit_limits_'+stamp);backup.mkdir(parents=True)
for name in ('config.json','control.py','stream_ui.py','controls_panel.html'):shutil.copy2(live/name,backup/name)
stopped=[]
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():continue
    try:args=(proc/'cmdline').read_bytes().split(b'\0');cwd=(proc/'cwd').resolve()
    except OSError:continue
    if cwd==live and any(x==b'main.py' or x.endswith(b'/main.py') for x in args):os.kill(int(proc.name),signal.SIGTERM);stopped.append(int(proc.name))
deadline=time.monotonic()+30
for pid in stopped:
    while Path(f'/proc/{pid}/stat').exists() and Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1][0]!='Z':
        assert time.monotonic()<deadline;time.sleep(.2)
(live/'config.json').write_text(json.dumps(cfg,indent=2));(live/'control.py').write_text(control)
(live/'stream_ui.py').write_text(panel);(live/'controls_panel.html').write_text(html)
receipt=dict(backup=str(backup),actual_integrator_limits_verified=results)
(Path(__file__).parent/'test_limits_verified.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
