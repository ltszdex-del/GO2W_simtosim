from pathlib import Path
import subprocess,sys,os,json,datetime
live=Path('/home/ubuntu/Go2W/interactive_latest_test_20261110');p=live/'verify_import.py';s=p.read_text()
start=s.index("    env.reset();c.reset_runtime(env.snapshot());c._set_mobility('wheel',env.snapshot())")
end=s.index("    result=dict(status='passed'",start)
s=s[:start]+"    with torch.inference_mode():\n"+''.join('    '+line+'\n' for line in s[start:end].splitlines())+s[end:]
compile(s,str(p),'exec');p.write_text(s)
folder=live/'logs'/('import_idle_retry_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'));folder.mkdir()
with (folder/'console.log').open('w') as output:
    proc=subprocess.Popen([sys.executable,'-u','verify_import.py','--livestream','0','--device','cuda:0','--log_dir',str(folder)],cwd=live,
        env={**os.environ,'OMNI_KIT_ACCEPT_EULA':'YES','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'},stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
(Path(__file__).parent/'runtime_check.json').write_text(json.dumps(dict(pid=proc.pid,folder=str(folder))));print(str(folder))
