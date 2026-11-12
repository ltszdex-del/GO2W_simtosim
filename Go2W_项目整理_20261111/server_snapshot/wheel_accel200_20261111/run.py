from pathlib import Path
import datetime,json,os,shutil,subprocess,sys,torch,math
h=Path(__file__).resolve().parent;b=h.parent;r=b/'go2w_wheel_accel200_20261111'
def status(**data):
    data.update(time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    p=h/'status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(p)
try:
    old=b/'go2w_wheel_rearfix600_20261111'
    parent=Path((old/'logs/train/latest.txt').read_text().strip())
    ck=torch.load(parent,map_location='cpu',weights_only=False);goal=ck['iteration']+200
    assert not r.exists(),r
    shutil.copytree(h/'wheel',r,ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(old/'assets',r/'assets',symlinks=True);shutil.copy2(old/'U_final_standing.json',r/'U_final_standing.json')
    subprocess.run([sys.executable,str(h/'migrate.py'),str(r),str(parent),'200'],check=True)
    env={**os.environ,'OMNI_KIT_ACCEPT_EULA':'YES','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'}
    for phase,n,updates in [('preflight',64,2),('train',4096,200)]:
        log=r/'logs'/phase;log.mkdir(parents=True)
        cmd=[sys.executable,'-u','main.py','--mode','train','--iterations',str(updates),'--num_envs',str(n),'--checkpoint',str(r/'resume/parent.pt'),'--livestream','0','--device','cuda:0','--log_dir',str(log)]
        with (log/'console.log').open('w') as output:
            p=subprocess.Popen(cmd,cwd=r,env=env,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT)
            status(state='running',phase=phase,pid=p.pid,parent=str(parent),goal_iteration=goal,num_envs=n,log=str(log));code=p.wait()
        assert code==0 and (log/'exit_code').read_text().strip()=='0'
        rows=[json.loads(x) for x in (log/'progress.jsonl').read_text().splitlines()]
        assert len(rows)==updates and all(all(math.isfinite(v) for v in row['loss']) for row in rows)
    status(state='complete',checkpoint=(log/'latest.txt').read_text().strip(),goal_iteration=goal,num_envs=4096)
except BaseException as error:status(state='failed',error=repr(error));raise
