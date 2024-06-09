g2n = {
    "cpu-time": 0,
    "cpu-related": 1,
    "memory": 2,
    "socket": 3,
    "tcp": 4,
    "network": 5,
    "udp": 6
}


def group(mname):
    if 'mem' in mname:
        return "memory"
    if 'ss' in mname:
        return "socket"
    if 'tcp' in mname:
        return "tcp"
    if 'iface' in mname:
        return "network"
    if 'udp' in mname:
        return "udp"
    if mname in ['busy', 'iowait', 'load1', 'system', 'user']:
        return "cpu-time"
    return "cpu-related"


def group_mlist(mlist: list) -> list:
    return [x for x, _ in sorted([(mname, g2n[group(mname)]) for mname in mlist], key=lambda x: x[1])]


g2n_IBM = {
    'dashboard': 0,
    'dash-info': 1,
    'OPTIONS': 2,
    'socket': 3,
    'abstract': 4,
    'events': 6,
    'others': 5
}


def group_IBM(mname):
    if 'abstraction' in mname:
        return 'abstract'
    if 'dashboard' in mname and 'info' in mname:
        return 'dash-info'
    if 'dashboard' in mname:
        return 'dashboard'
    if 'events' in mname:
        return 'events'
    if 'socket' in mname:
        return 'socket'
    if 'OPTIONS' in mname:
        return 'OPTIONS'
    return 'others'

def group_mlist_IBM(mlist: list) -> list:
    return [x for x, _ in sorted([(mname, g2n_IBM[group_IBM(mname)]) for mname in mlist], key=lambda x: x[1])]