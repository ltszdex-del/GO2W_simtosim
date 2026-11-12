"""Full P -> U -> real standing_hold -> repeated movement, without reset shortcuts."""
import bootstrap,json,torch,numpy as np
from run import create_runtime,advance
rows=[]
for key in 'WSADQE':
    env,c=create_runtime();hold_frame=None;start_frame=None;blocked=0;events=[];previous=None;seen_getup=False
    min_separation=10.;peak_speed=0.;last=None
    with torch.inference_mode():
        for frame in range(1500):
            keys=['P'] if frame==100 else (['U'] if frame==450 else [])
            seen_getup|=c.mode=='getting_up'
            if c.mode=='standing_hold' and seen_getup and hold_frame is None:hold_frame=frame
            # Request movement immediately upon U completion; the handoff gate
            # must wait for real settling, then accept the still-held key.
            if hold_frame is not None:
                age=frame-hold_frame
                keys=[key] if age<100 or 180<=age<240 or 320<=age<380 else []
            c.feed('pu_handoff_test',keys);s=advance(env,c)
            if c.mode!=previous:
                events.append(dict(frame=frame,mode=c.mode,motor_mode=env.motor_mode));previous=c.mode
            assert c.mode!='error',(key,frame,s['message'])
            if c.mode=='standing_hold':
                assert env.motor_mode=='standing_hold',(c.mode,env.motor_mode)
                if keys and not env.standing_ready():
                    assert max(abs(v) for v in c.command)==0.
                    blocked+=1
            if hold_frame is not None and c.mode=='locomotion':
                if start_frame is None:start_frame=frame
                feet,_=env.robot.find_bodies(['FR_foot','FL_foot','RR_foot','RL_foot'])
                positions=env.robot.data.body_link_pos_w.torch[0,feet].numpy()
                distances=np.linalg.norm(positions[:,None]-positions[None,:],axis=-1)+np.eye(4)*100
                min_separation=min(min_separation,float(distances.min()))
                peak_speed=max(peak_speed,float(np.linalg.norm(s['actual_velocity_yaw_m_s_rad_s'])))
                assert s['physics']['gravity_b'][2]<-.85,(key,frame,s['physics'])
                assert min_separation>.07,(key,frame,min_separation)
            if hold_frame is not None and frame-hold_frame>=470:break
        assert hold_frame is not None and start_frame is not None,(key,events)
        assert peak_speed>.1,(key,peak_speed)
    row=dict(key=key,passed=True,events=events,settling_delay_s=(start_frame-hold_frame)*.02,
             blocked_frames=blocked,min_foot_separation_m=min_separation,peak_motion_speed=peak_speed)
    rows.append(row);print(json.dumps(row),flush=True)
(bootstrap.ROOT/'logs/pu_handoff_checks.json').write_text(json.dumps(rows,indent=2))
