import csv
import os
from scapy.all import sniff, IP, TCP, UDP, get_if_list
from collections import defaultdict, deque
import time
import socket
from scapy.error import Scapy_Exception

csv_file_path = "real_time_nids_features.csv"
csv_header = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login",
    "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate"
]

# Write header only once
if not os.path.exists(csv_file_path):
    with open(csv_file_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)

# ======= Flow tracking setup =======
flows = defaultdict(lambda: {
    'start_time': None,
    'end_time': None,
    'src_bytes': 0,
    'dst_bytes': 0,
    'count': 0,
    'services': set(),
    'dst_hosts': set(),
    'ports': set(),
    'tcp_flags': [],
    'logged_in': 0,
})

window = deque(maxlen=1000)
window_duration = 2  # seconds

def map_port_to_service(port):
    common_ports = {
        20: 'ftp_data',
        21: 'ftp',
        22: 'ssh',
        23: 'telnet',
        25: 'smtp',
        53: 'domain',
        69: 'tftp_u',
        80: 'http',
        110: 'pop_3',
        111: 'sunrpc',
        119: 'nntp',
        123: 'ntp_u',
        135: 'epmap',
        137: 'netbios_ns',
        138: 'netbios_dgm',
        139: 'netbios_ssn',
        143: 'imap4',
        161: 'snmp',
        162: 'snmptrap',
        179: 'bgp',
        443: 'https',
        445: 'microsoft_ds',
        513: 'login',
        514: 'shell',
        515: 'printer',
        520: 'route',
        540: 'uucp',
        635: 'mountd',
        8080: 'http',
    }
    
    if port in common_ports:
        return common_ports[port]
    elif port >= 1024:
        return 'private'
    else:
        return 'other'

def get_service(port):
    try:
        return socket.getservbyport(port)
    except OSError:
        return str(port)

def map_flags_to_dataset(flag):
    if flag in ['PA', 'A', 'P', 'FA', 'F']:
        return 'SF'
    elif flag == 'S':
        return 'S0'
    elif flag in ['R', 'RA']:
        return 'REJ'
    else:
        return 'OTH'
        
def extract_features(pkt):
    if not IP in pkt:
        return

    ip = pkt[IP]
    proto = pkt.proto
    now = time.time()

    if TCP in pkt or UDP in pkt:
        sport = pkt.sport
        dport = pkt.dport
    else:
        sport = dport = 0

    key = (ip.src, ip.dst, sport, dport, proto)

    flow = flows[key]
    if flow['start_time'] is None:
        flow['start_time'] = now
    flow['end_time'] = now

    flow['count'] += 1
    size = len(pkt)

    if ip.src == key[0]:
        flow['src_bytes'] += size
    else:
        flow['dst_bytes'] += size

    flow['services'].add(map_port_to_service(dport))
    flow['dst_hosts'].add(ip.dst)
    flow['ports'].add(dport)

    if TCP in pkt:
        tcp = pkt[TCP]
        flow['tcp_flags'].append(tcp.flags)
        if tcp.flags == "S":
            flow['logged_in'] = 0
        if tcp.flags == "PA":
            flow['logged_in'] = 1

    window.append({
        'src': ip.src,
        'dst': ip.dst,
        'proto': proto,
        'service': map_port_to_service(dport),
        'timestamp': now
    })

    duration = flow['end_time'] - flow['start_time']
    protocol_type = {6: 'tcp', 17: 'udp', 1: 'icmp'}.get(proto, 'other')
    service = list(flow['services'])[0] if flow['services'] else '0'
    raw_flag = flow['tcp_flags'][-1] if flow['tcp_flags'] else '0'
    flag = map_flags_to_dataset(str(raw_flag))  # Normalize it
    src_bytes = flow['src_bytes']
    dst_bytes = flow['dst_bytes']
    land = 1 if ip.src == ip.dst and sport == dport else 0

    wrong_fragment = 0
    urgent = 0
    hot = 0
    num_failed_logins = 0
    logged_in = flow['logged_in']
    num_compromised = 0
    root_shell = 0
    su_attempted = 0
    num_root = 0
    num_file_creations = 0
    num_shells = 0
    num_access_files = 0
    num_outbound_cmds = 0
    is_host_login = 0
    is_guest_login = 0

    recent = [w for w in window if now - w['timestamp'] <= window_duration]
    count = len([w for w in recent if w['dst'] == ip.dst])
    srv_count = len([w for w in recent if w['dst'] == ip.dst and w['service'] == service])

    serror_rate = 0
    srv_serror_rate = 0
    rerror_rate = 0
    srv_rerror_rate = 0
    same_srv_rate = srv_count / count if count else 0
    diff_srv_rate = 1 - same_srv_rate
    srv_diff_host_rate = 0

    dst_host_count = count
    dst_host_srv_count = srv_count
    dst_host_same_srv_rate = same_srv_rate
    dst_host_diff_srv_rate = diff_srv_rate
    dst_host_same_src_port_rate = 0
    dst_host_srv_diff_host_rate = 0
    dst_host_serror_rate = 0
    dst_host_srv_serror_rate = 0
    dst_host_rerror_rate = 0
    dst_host_srv_rerror_rate = 0

    features = [
        duration, protocol_type, service, flag, src_bytes, dst_bytes, land,
        wrong_fragment, urgent, hot, num_failed_logins, logged_in, num_compromised,
        root_shell, su_attempted, num_root, num_file_creations, num_shells,
        num_access_files, num_outbound_cmds, is_host_login, is_guest_login,
        count, srv_count, serror_rate, srv_serror_rate, rerror_rate,
        srv_rerror_rate, same_srv_rate, diff_srv_rate, srv_diff_host_rate,
        dst_host_count, dst_host_srv_count, dst_host_same_srv_rate,
        dst_host_diff_srv_rate, dst_host_same_src_port_rate,
        dst_host_srv_diff_host_rate, dst_host_serror_rate,
        dst_host_srv_serror_rate, dst_host_rerror_rate,
        dst_host_srv_rerror_rate
    ]

    # Save row to CSV
    with open(csv_file_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(features)

    print("Captured:", features[:5], "...")  # Print a summary

def start_sniffing():
    interfaces = get_if_list()
    print("[*] Capturing packets & writing to:", csv_file_path)
    
    for iface in interfaces:
        try:
            print(f"[*] Testing interface: {iface}")
            packets = sniff(iface=iface, count=1, timeout=3, store=1)
            if packets:
                print(f"[+] Using interface: {iface}")
                # Start actual sniffing now
                sniff(iface=iface, prn=extract_features, store=0, timeout=20)
                break
        except Scapy_Exception as e:
            print(f"[!] Skipping interface {iface}: {e}")
        except Exception as e:
            print(f"[!] Unexpected error on {iface}: {e}")
    else:
        print("[-] No usable interface found.")

if __name__ == "__main__":
    start_sniffing()  # Or "wlan0" or your actual Colab interface