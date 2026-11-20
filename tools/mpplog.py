"""Parses an mppshim.so log into per-fd lists of tasks: register writes, address offsets, read-back registers."""
import gzip, re, sys
from collections import deque
def parse(path):
    tasks, queue, last, info = {}, {}, None, {}
    for ln in (gzip.open(path, 'rt') if path.endswith('.gz') else open(path)).read().split('\n'):
        m = re.match(r'REQ fd=(\d+) i=(\d+) cmd=(\w+) flag=(\w+) size=(\d+) off=(\w+)', ln)
        if m:
            fd, idx, cmd = int(m[1]), int(m[2]), int(m[3], 16)
            if cmd == 0x403: last = ('info', fd); continue
            if cmd in (0x300, 0x301): last = None; continue
            if idx == 0 and cmd < 0x300 and cmd >= 0x200:
                t = {'w': {}, 'off': [], 'rd': {}, 'info': info.pop(fd, None)}; queue.setdefault(fd, deque()).append(t)
            if cmd >= 0x200 and cmd < 0x300: last = ('w', fd, cmd, queue[fd][-1])
            else: last = None
            continue
        m = re.match(r'POLLRET fd=(\d+)', ln)
        if m:
            fd = int(m[1]); t = queue[fd].popleft(); tasks.setdefault(fd, []).append(t); last = ('poll', t); continue
        m = re.match(r'RD fd=(\d+)', ln)
        if m: last = ('rd', last[1] if last and last[0] in ('poll', 'rd') else None); continue
        m = re.match(r'\s+([0-9a-f]+):((?: [0-9a-f]{8})+)', ln)
        if m and last:
            base = int(m[1], 16); words = [int(w, 16) for w in m[2].split()]
            if last[0] == 'rd' and last[1] is not None:
                for k, w in enumerate(words): last[1]['rd'][base + 4*k] = w
            elif last[0] == 'info': info[last[1]] = info.get(last[1], []) + words
            elif last[0] == 'w':
                _, fd, cmd, t = last
                if cmd == 0x200:
                    for k, w in enumerate(words): t['w'][base + 4*k] = w
                elif cmd == 0x202: t['off'] += words
    return tasks
