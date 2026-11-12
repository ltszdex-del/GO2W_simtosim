#!/usr/bin/env python3
"""End-to-end bridge -> controller -> command ramp test (three-axis status and bounded normal release)."""
import http.client
import json
import time


def post(path, payload):
    conn = http.client.HTTPConnection('127.0.0.1', 8765, timeout=5)
    body = json.dumps(payload)
    conn.request('POST', path, body=body,
                 headers={'Content-Type': 'application/json', 'Origin': 'http://127.0.0.1:8765'})
    r = conn.getresponse(); data = r.read().decode(); conn.close()
    return r.status, data


def get(path):
    conn = http.client.HTTPConnection('127.0.0.1', 8765, timeout=5)
    conn.request('GET', path)
    r = conn.getresponse(); data = r.read().decode(); conn.close()
    return r.status, json.loads(data) if r.status == 200 else data


print('1) release first (clean state)')
print('   POST /input [] ->', post('/input', {'session': 'e2e', 'keys': []})[0])
time.sleep(0.5)

def status():
    return get('/status')[1]


print('2) hold W for 0.3s (within 0.4s timeout)')
print('   POST /input [W] ->', post('/input', {'session': 'e2e', 'keys': ['W']})[0])
time.sleep(0.3)
st = status()
print('   after 0.3s: keys=', st['keys'], 'command=', st['command_m_s_rad_s'],
      'source=', st['source'], 'input_age_s=', round(st['input_age_s'], 3))

print('3) hold W+D+Q for another 0.25s')
print('   POST /input [W,D,Q] ->', post('/input', {'session': 'e2e', 'keys': ['W', 'D', 'Q']})[0])
time.sleep(0.25)
st = status()
print('   after 0.25s: keys=', sorted(st['keys']), 'command=', st['command_m_s_rad_s'])

print('4) normal release, wait for bounded braking')
print('   POST /input [] ->', post('/input', {'session': 'e2e', 'keys': []})[0])
time.sleep(1.0)
st = status()
print('   after release+1.0s: keys=', st['keys'], 'command=', st['command_m_s_rad_s'])

print('5) H/P/U edge triggers (expect 未加载 messages, mode stays standing)')
for k in ('H', 'P', 'U'):
    post('/input', {'session': 'e2e', 'keys': [k]})
    time.sleep(0.05)
    st = status()
    print(f'   {k}: mode={st["mode"]} msg={st["message"]}')
    post('/input', {'session': 'e2e', 'keys': []})
    time.sleep(0.05)

assert st['command_axes']==['vx','vy','wz']
assert len(st['command_m_s_rad_s'])==3
post('/release', {'session':'e2e'})
