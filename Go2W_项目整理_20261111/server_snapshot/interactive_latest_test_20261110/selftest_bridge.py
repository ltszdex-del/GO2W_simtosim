#!/usr/bin/env python3
"""Self-test the running interactive stage1: ports, bridge input, status."""
import http.client
import json
import socket
import sys


def ports():
    out = {}
    for proto in ('tcp', 'tcp6', 'udp', 'udp6'):
        try:
            with open(f'/proc/net/{proto}') as f:
                rows = f.read().splitlines()[1:]
        except OSError:
            continue
        for row in rows:
            parts = row.split()
            if len(parts) < 4:
                continue
            local = parts[1].split(':')[1]
            port = int(local, 16)
            state = parts[3]
            out.setdefault((proto, port), state)
    return out


def post(path, payload, origin='http://127.0.0.1:8765'):
    conn = http.client.HTTPConnection('127.0.0.1', 8765, timeout=5)
    body = json.dumps(payload)
    conn.request('POST', path, body=body,
                 headers={'Content-Type': 'application/json', 'Origin': origin})
    resp = conn.getresponse()
    data = resp.read().decode()
    conn.close()
    return resp.status, data


def get(path):
    conn = http.client.HTTPConnection('127.0.0.1', 8765, timeout=5)
    conn.request('GET', path)
    resp = conn.getresponse()
    data = resp.read().decode()
    conn.close()
    return resp.status, data


def main():
    p = ports()
    interesting = {49100: 'tcp', 49101: 'tcp', 47998: 'udp', 48000: 'udp', 8765: 'tcp', 8766: 'tcp'}
    print('=== interesting ports ===')
    for port, proto in sorted(interesting.items(), key=lambda x: x[0]):
        key = (proto, port)
        print(proto, port, 'state', p.get(key, 'NOT-LISTENING'))
    print('=== all udp ===')
    for (proto, port), state in sorted(p.items()):
        if proto.startswith('udp'):
            print(proto, port, state)

    print('=== bridge POST /input W ===')
    try:
        s, d = post('/input', {'session': 'selftest', 'keys': ['W']})
        print('POST /input ->', s, repr(d))
    except Exception as e:
        print('POST /input error:', e)
        sys.exit(1)

    import time
    time.sleep(0.5)
    print('=== bridge GET /status ===')
    s, d = get('/status')
    print('GET /status ->', s)
    if s == 200:
        st = json.loads(d)
        print('mode=', st['mode'])
        print('keys=', st['keys'])
        print('command_m_s_rad_s=', st['command_m_s_rad_s'])
        print('source=', st['source'])
        print('input_age_s=', st['input_age_s'])

    print('=== bridge POST /input {} (release) ===')
    s, d = post('/input', {'session': 'selftest', 'keys': []})
    print('POST /input [] ->', s, repr(d))
    time.sleep(0.5)
    s, d = get('/status')
    if s == 200:
        st = json.loads(d)
        print('after release: keys=', st['keys'], 'command=', st['command_m_s_rad_s'])


if __name__ == '__main__':
    main()
