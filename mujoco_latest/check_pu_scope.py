"""Check numerical isolation across mobility, PU and H preparation endpoints."""
import bootstrap,json,torch
from run import create_runtime,advance
env,c=create_runtime()
with torch.inference_mode():
    c.feed('scope',[]);advance(env,c)
    assert env.pd_updates==4
    c._enter_standing_hold(env.snapshot());advance(env,c)
    assert env.pd_updates==20
    # H preparation and returned standing use the same stable U integration.
    c.pending='handstand';c._enter_standing_hold(env.snapshot());advance(env,c)
    assert env.handstand_context.local_pu_isolation and env.pd_updates==20
    c.pending=None;c.mode='handstand_return';c._enter_standing_hold(env.snapshot());advance(env,c)
    assert env.handstand_context.local_pu_isolation and env.pd_updates==20
    c._set_mobility('wheel',env.snapshot());c.feed('scope',['W']);advance(env,c)
    assert env.pd_updates==4
    assert env.model.opt.timestep==.005
print('PASS: step/wheel retain four updates; PU and H standing endpoints use 20')
