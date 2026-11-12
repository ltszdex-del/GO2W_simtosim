from pathlib import Path
import sys,subprocess,os,json,math,datetime
kind=sys.argv[1];b=Path('/home/ubuntu/Go2W');h=b/'refine_yaw600_20261111';r=b/f'go2w_{kind}_yaw600_20261111'
env={**os.environ,'OMNI_KIT_ACCEPT_EULA':'YES','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'}
updates=600
goal=3299 if kind=='wheel' else 10499
def record(**data):
    data.update(goal_iteration=goal,num_envs=2048,time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    path=h/f'{kind}_status.json';temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2));temp.replace(path)
try:
    for phase,iterations,n in [('preflight',2,64),('train',updates,2048)]:
        log=r/'logs'/phase;log.mkdir(parents=True,exist_ok=False)
        cmd=[sys.executable,'-u','main.py','--mode','train','--iterations',str(iterations),'--num_envs',str(n),
             '--checkpoint',str(r/'resume/parent.pt'),'--livestream','0','--device','cuda:0','--log_dir',str(log)]
        with (log/'console.log').open('w') as output:
            p=subprocess.Popen(cmd,cwd=r,env=env,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT)
            (log/'launch.json').write_text(json.dumps(dict(pid=p.pid,command=cmd),indent=2))
            record(state='running',phase=phase,pid=p.pid,log=str(log));code=p.wait()
        assert code==0 and (log/'exit_code').read_text().strip()=='0',(phase,code)
        rows=[json.loads(x) for x in (log/'progress.jsonl').read_text().splitlines()]
        assert len(rows)==iterations and all(all(math.isfinite(v) for v in row['loss']) for row in rows)
        if kind=='wheel':assert all(all(math.isfinite(v) for key,v in row.items() if key.startswith('geometry/')) for row in rows)
    record(state='complete',updates=updates,checkpoint=(log/'latest.txt').read_text().strip(),log=str(log))
except BaseException as error:record(state='failed',error=repr(error));raise
