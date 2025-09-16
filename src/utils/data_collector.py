#!/usr/bin/env python3
"""
Network data collection utilities
Collects real-time network statistics and flow information
"""

import json
import time
import psutil
import subprocess
import threading
from datetime import datetime
import os


class NetworkDataCollector:
    """Collect network statistics and flow data"""
    
    def __init__(self, output_dir="data"):
        self.output_dir = output_dir
        self.collecting = False
        self.network_stats = []
        self.flow_stats = []
        os.makedirs(output_dir, exist_ok=True)
    
    def collect_system_stats(self):
        """Collect system-level network statistics"""
        try:
            net_io = psutil.net_io_counters(pernic=True)
            net_connections = psutil.net_connections()
            
            stats = {
                'timestamp': datetime.now().isoformat(),
                'interfaces': {},
                'active_connections': len(net_connections),
                'total_connections': len([conn for conn in net_connections if conn.status == 'ESTABLISHED'])
            }
            
            for interface, counters in net_io.items():
                stats['interfaces'][interface] = {
                    'bytes_sent': counters.bytes_sent,
                    'bytes_recv': counters.bytes_recv,
                    'packets_sent': counters.packets_sent,
                    'packets_recv': counters.packets_recv,
                    'errin': counters.errin,
                    'errout': counters.errout,
                    'dropin': counters.dropin,
                    'dropout': counters.dropout
                }
            
            return stats
        except Exception as e:
            print(f"Error collecting system stats: {e}")
            return None
    
    def collect_ovs_flow_stats(self):
        """Collect OpenFlow statistics from OVS switches"""
        try:
            # Get list of bridges
            result = subprocess.run(['ovs-vsctl', 'list-br'], 
                                  capture_output=True, text=True)
            bridges = result.stdout.strip().split('\n') if result.stdout.strip() else []
            
            flow_data = {
                'timestamp': datetime.now().isoformat(),
                'switches': {}
            }
            
            for bridge in bridges:
                if bridge:
                    # Get flow stats for each bridge
                    result = subprocess.run(['ovs-ofctl', 'dump-flows', bridge],
                                          capture_output=True, text=True)
                    
                    flows = []
                    if result.stdout:
                        for line in result.stdout.strip().split('\n'):
                            if 'cookie=' in line:
                                flows.append(self._parse_flow_entry(line))
                    
                    # Get port stats
                    result = subprocess.run(['ovs-ofctl', 'dump-ports', bridge],
                                          capture_output=True, text=True)
                    
                    ports = {}
                    if result.stdout:
                        for line in result.stdout.strip().split('\n'):
                            if 'port' in line and 'rx' in line:
                                port_info = self._parse_port_stats(line)
                                if port_info:
                                    ports[port_info['port']] = port_info
                    
                    flow_data['switches'][bridge] = {
                        'flows': flows,
                        'ports': ports,
                        'flow_count': len(flows)
                    }
            
            return flow_data
        except Exception as e:
            print(f"Error collecting OVS stats: {e}")
            return None
    
    def _parse_flow_entry(self, flow_line):
        """Parse OVS flow entry"""
        try:
            parts = flow_line.split(',')
            flow_info = {
                'duration': None,
                'priority': None,
                'n_packets': 0,
                'n_bytes': 0,
                'actions': '',
                'match': ''
            }
            
            for part in parts:
                part = part.strip()
                if 'duration=' in part:
                    flow_info['duration'] = part.split('=')[1]
                elif 'priority=' in part:
                    flow_info['priority'] = int(part.split('=')[1])
                elif 'n_packets=' in part:
                    flow_info['n_packets'] = int(part.split('=')[1])
                elif 'n_bytes=' in part:
                    flow_info['n_bytes'] = int(part.split('=')[1])
                elif 'actions=' in part:
                    flow_info['actions'] = part.split('=')[1]
            
            return flow_info
        except Exception as e:
            print(f"Error parsing flow entry: {e}")
            return {}
    
    def _parse_port_stats(self, port_line):
        """Parse OVS port statistics"""
        try:
            if 'port' not in port_line:
                return None
            
            parts = port_line.split()
            port_info = {
                'port': parts[1].rstrip(':'),
                'rx_packets': 0,
                'tx_packets': 0,
                'rx_bytes': 0,
                'tx_bytes': 0,
                'rx_dropped': 0,
                'tx_dropped': 0
            }
            
            for i, part in enumerate(parts):
                if part == 'rx':
                    if i + 1 < len(parts):
                        rx_stats = parts[i + 1].split(',')
                        if len(rx_stats) >= 4:
                            port_info['rx_packets'] = int(rx_stats[1].split('=')[1])
                            port_info['rx_bytes'] = int(rx_stats[2].split('=')[1])
                            port_info['rx_dropped'] = int(rx_stats[3].split('=')[1])
                elif part == 'tx':
                    if i + 1 < len(parts):
                        tx_stats = parts[i + 1].split(',')
                        if len(tx_stats) >= 4:
                            port_info['tx_packets'] = int(tx_stats[1].split('=')[1])
                            port_info['tx_bytes'] = int(tx_stats[2].split('=')[1])
                            port_info['tx_dropped'] = int(tx_stats[3].split('=')[1])
            
            return port_info
        except Exception as e:
            print(f"Error parsing port stats: {e}")
            return None
    
    def start_continuous_collection(self, interval=5, duration=600):
        """Start continuous data collection"""
        self.collecting = True
        start_time = time.time()
        
        print(f"Starting data collection for {duration} seconds...")
        
        while self.collecting and (time.time() - start_time < duration):
            # Collect system stats
            sys_stats = self.collect_system_stats()
            if sys_stats:
                self.network_stats.append(sys_stats)
            
            # Collect OVS flow stats
            flow_stats = self.collect_ovs_flow_stats()
            if flow_stats:
                self.flow_stats.append(flow_stats)
            
            # Save data periodically
            if len(self.network_stats) % 10 == 0:
                self.save_collected_data()
            
            time.sleep(interval)
        
        self.collecting = False
        self.save_collected_data()
        print(f"Data collection complete. Collected {len(self.network_stats)} network stat entries and {len(self.flow_stats)} flow stat entries.")
    
    def save_collected_data(self):
        """Save collected data to files"""
        try:
            # Save network stats
            with open(f"{self.output_dir}/network_stats.json", 'w') as f:
                json.dump(self.network_stats, f, indent=2)
            
            # Save flow stats
            with open(f"{self.output_dir}/flow_stats.json", 'w') as f:
                json.dump(self.flow_stats, f, indent=2)
            
        except Exception as e:
            print(f"Error saving data: {e}")
    
    def stop_collection(self):
        """Stop data collection"""
        self.collecting = False


def main():
    """Main function for standalone data collection"""
    collector = NetworkDataCollector()
    
    try:
        collector.start_continuous_collection(interval=5, duration=300)  # 5 minutes
    except KeyboardInterrupt:
        collector.stop_collection()
        print("\nData collection stopped by user")


if __name__ == '__main__':
    main()