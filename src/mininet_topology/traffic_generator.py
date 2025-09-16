#!/usr/bin/env python3
"""
Traffic generator for network testing and anomaly detection
Generates various traffic patterns including normal and anomalous traffic
"""

import time
import random
import threading
import json
import os
from datetime import datetime
from scapy.all import *
import subprocess


class TrafficGenerator:
    """Generate various types of network traffic for testing"""
    
    def __init__(self, hosts=['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8']):
        self.hosts = hosts
        self.running = False
        self.traffic_data = []
        self.data_file = "data/traffic_log.json"
        os.makedirs("data", exist_ok=True)
    
    def generate_normal_traffic(self, duration=300):
        """Generate normal background traffic between hosts"""
        info("Starting normal traffic generation...")
        start_time = time.time()
        
        while time.time() - start_time < duration and self.running:
            # Random source and destination
            src = random.choice(self.hosts)
            dst = random.choice([h for h in self.hosts if h != src])
            
            # Generate different types of normal traffic
            traffic_type = random.choice(['http', 'ssh', 'ftp', 'ping'])
            
            if traffic_type == 'http':
                self._generate_http_traffic(src, dst)
            elif traffic_type == 'ssh':
                self._generate_ssh_traffic(src, dst)
            elif traffic_type == 'ftp':
                self._generate_ftp_traffic(src, dst)
            else:
                self._generate_ping_traffic(src, dst)
            
            # Random interval between traffic bursts
            time.sleep(random.uniform(0.1, 2.0))
    
    def generate_anomalous_traffic(self, duration=60):
        """Generate various types of anomalous traffic patterns"""
        info("Starting anomalous traffic generation...")
        start_time = time.time()
        
        anomaly_types = ['ddos', 'port_scan', 'bandwidth_flood', 'ping_flood']
        
        while time.time() - start_time < duration and self.running:
            anomaly = random.choice(anomaly_types)
            
            if anomaly == 'ddos':
                self._generate_ddos_attack()
            elif anomaly == 'port_scan':
                self._generate_port_scan()
            elif anomaly == 'bandwidth_flood':
                self._generate_bandwidth_flood()
            else:
                self._generate_ping_flood()
            
            time.sleep(random.uniform(5, 15))
    
    def _generate_http_traffic(self, src, dst):
        """Generate HTTP-like traffic"""
        try:
            cmd = f"mininet> {src} wget -q -O /dev/null --timeout=5 {self._get_host_ip(dst)}:8000 &"
            self._log_traffic(src, dst, 'HTTP', 'normal', random.randint(100, 1500))
        except:
            pass
    
    def _generate_ssh_traffic(self, src, dst):
        """Generate SSH-like traffic"""
        try:
            cmd = f"mininet> {src} nc -z {self._get_host_ip(dst)} 22 &"
            self._log_traffic(src, dst, 'SSH', 'normal', random.randint(50, 200))
        except:
            pass
    
    def _generate_ftp_traffic(self, src, dst):
        """Generate FTP-like traffic"""
        try:
            self._log_traffic(src, dst, 'FTP', 'normal', random.randint(1000, 5000))
        except:
            pass
    
    def _generate_ping_traffic(self, src, dst):
        """Generate ping traffic"""
        try:
            cmd = f"mininet> {src} ping -c 1 {self._get_host_ip(dst)} &"
            self._log_traffic(src, dst, 'ICMP', 'normal', 64)
        except:
            pass
    
    def _generate_ddos_attack(self):
        """Simulate DDoS attack"""
        target = random.choice(self.hosts)
        attackers = random.sample([h for h in self.hosts if h != target], 3)
        
        for attacker in attackers:
            try:
                # High frequency requests
                for _ in range(100):
                    self._log_traffic(attacker, target, 'HTTP', 'anomaly', random.randint(500, 1500))
                    time.sleep(0.01)
            except:
                pass
    
    def _generate_port_scan(self):
        """Simulate port scanning"""
        scanner = random.choice(self.hosts)
        target = random.choice([h for h in self.hosts if h != scanner])
        
        # Scan multiple ports rapidly
        for port in range(20, 100):
            try:
                self._log_traffic(scanner, target, f'TCP:{port}', 'anomaly', 64)
                time.sleep(0.05)
            except:
                pass
    
    def _generate_bandwidth_flood(self):
        """Generate high bandwidth traffic"""
        src = random.choice(self.hosts)
        dst = random.choice([h for h in self.hosts if h != src])
        
        # Generate large packets rapidly
        for _ in range(50):
            try:
                self._log_traffic(src, dst, 'UDP', 'anomaly', 1500)
                time.sleep(0.02)
            except:
                pass
    
    def _generate_ping_flood(self):
        """Generate ping flood"""
        src = random.choice(self.hosts)
        dst = random.choice([h for h in self.hosts if h != src])
        
        for _ in range(100):
            try:
                self._log_traffic(src, dst, 'ICMP', 'anomaly', 64)
                time.sleep(0.01)
            except:
                pass
    
    def _get_host_ip(self, host):
        """Get IP address for host"""
        host_num = int(host[1:])
        return f"10.0.0.{host_num}"
    
    def _log_traffic(self, src, dst, protocol, traffic_type, packet_size):
        """Log traffic information"""
        traffic_entry = {
            'timestamp': datetime.now().isoformat(),
            'src': src,
            'dst': dst,
            'src_ip': self._get_host_ip(src),
            'dst_ip': self._get_host_ip(dst),
            'protocol': protocol,
            'type': traffic_type,
            'packet_size': packet_size,
            'bandwidth': random.uniform(1, 100) if traffic_type == 'normal' else random.uniform(100, 1000)
        }
        
        self.traffic_data.append(traffic_entry)
        
        # Write to file periodically
        if len(self.traffic_data) % 100 == 0:
            self._save_traffic_data()
    
    def _save_traffic_data(self):
        """Save traffic data to file"""
        try:
            with open(self.data_file, 'w') as f:
                json.dump(self.traffic_data, f, indent=2)
        except Exception as e:
            print(f"Error saving traffic data: {e}")
    
    def start_traffic_generation(self, duration=600):
        """Start traffic generation with both normal and anomalous patterns"""
        self.running = True
        
        # Start normal traffic in background
        normal_thread = threading.Thread(
            target=self.generate_normal_traffic, 
            args=(duration,)
        )
        normal_thread.daemon = True
        normal_thread.start()
        
        # Periodically inject anomalies
        time.sleep(30)  # Let normal traffic establish
        
        for _ in range(5):  # 5 anomaly periods
            anomaly_thread = threading.Thread(
                target=self.generate_anomalous_traffic,
                args=(60,)
            )
            anomaly_thread.daemon = True
            anomaly_thread.start()
            time.sleep(120)  # Wait between anomaly periods
        
        # Wait for completion
        time.sleep(duration - 30)
        self.running = False
        self._save_traffic_data()
        
        print(f"Traffic generation complete. {len(self.traffic_data)} entries logged.")


def main():
    """Main function to run traffic generator"""
    generator = TrafficGenerator()
    
    print("Starting traffic generation...")
    print("Make sure Mininet network is running!")
    
    try:
        generator.start_traffic_generation(duration=600)  # 10 minutes
    except KeyboardInterrupt:
        generator.running = False
        generator._save_traffic_data()
        print("\nTraffic generation stopped by user")


if __name__ == '__main__':
    main()