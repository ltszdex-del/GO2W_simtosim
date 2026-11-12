from pathlib import Path
r=Path(__file__).parent
def replace(kind,name,old,new):
    p=r/kind/name;s=p.read_text();assert old in s,(kind,name,old);p.write_text(s.replace(old,new))
with (r/'wheel/config.py').open('a') as f:f.write("\nCOMMAND.update(yaw_speed_limit=2.,max_speed=3.,speed_limits=[3.,1.3],axis_acceleration=[4.,1.2],backward_acceleration=6.,axis_brake=[4.8,3.12],brake_acceleration=4.8,yaw_acceleration=2.4,yaw_brake=9.6)\n")
replace('wheel','learning.py','env.drive_speed_cap=2.5','env.drive_speed_cap=3.')
replace('wheel','learning.py','env.turn_cap=3.','env.turn_cap=2.')
replace('wheel','learning.py','yaw_course_cap=.6+2.4*env.turn_course_progress','yaw_course_cap=2.')
with (r/'step/config.py').open('a') as f:f.write("\nGAIT.update(period=.28,period_range_s=[.23,.33])\n")
replace('step','period_math.py','.25+.10*torch.rand','.23+.10*torch.rand')
replace('step','period_math.py','evaluation_period=.30','evaluation_period=.28')
replace('step','env.py',"getattr(self,'evaluation_period',.30)","getattr(self,'evaluation_period',.28)")
replace('step','learning.py','(.25,.30,.35)','(.23,.28,.33)')
p=r/'run_one.py';s=p.read_text().replace('refine_geometry500_20261110','refine_followup_20261111').replace('go2w_{kind}_geometry500_20261110','go2w_{kind}_followup_20261111').replace("goal=1999 if kind=='wheel' else 9399","updates=400 if kind=='wheel' else 200\ngoal=2399 if kind=='wheel' else 9599").replace("('train500',500,2048)","('train',updates,2048)").replace('updates=500','updates=updates');p.write_text(s)
p=r/'migrate.py';s=p.read_text().replace('iterations=500,goal_iteration=ck[\'iteration\']+500',"iterations=int(sys.argv[3]),goal_iteration=ck['iteration']+int(sys.argv[3])");p.write_text(s)
