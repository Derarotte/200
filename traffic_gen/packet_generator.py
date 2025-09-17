#!/usr/bin/env python3
"""
网络流量生成器
生成正常流量和各种攻击模式，用于测试SDN网络和AI检测系统
"""

import time
import random
import threading
from dataclasses import dataclass
from typing import List, Dict, Callable, Optional
from collections import defaultdict
import json

# 导入网络模拟模块
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from network_sim.sdn_network import NetworkPacket


@dataclass
class TrafficPattern:
    """流量模式定义"""
    name: str
    description: str
    packet_rate: float  # 每秒包数
    duration: int  # 持续时间(秒)
    packet_size_range: tuple  # 包大小范围
    src_ips: List[str]
    dst_ips: List[str]
    protocols: List[str]
    port_range: tuple


class TrafficGenerator:
    """流量生成器"""

    def __init__(self):
        self.running = False
        self.threads = []
        self.generated_packets = []
        self.packet_callback: Optional[Callable] = None
        self.stats = defaultdict(int)

        # 预定义的主机列表
        self.hosts = ["10.0.0.1", "10.0.0.2", "10.0.0.3", "10.0.0.4", "10.0.0.5", "10.0.0.6"]

        # 流量模式定义
        self.patterns = {
            "normal_web": TrafficPattern(
                name="正常Web流量",
                description="模拟正常的HTTP/HTTPS访问",
                packet_rate=5.0,
                duration=60,
                packet_size_range=(500, 1500),
                src_ips=self.hosts[:4],
                dst_ips=self.hosts[2:],
                protocols=["TCP"],
                port_range=(80, 443)
            ),
            "normal_dns": TrafficPattern(
                name="正常DNS查询",
                description="模拟DNS查询流量",
                packet_rate=2.0,
                duration=60,
                packet_size_range=(50, 200),
                src_ips=self.hosts,
                dst_ips=["10.0.0.3"],  # DNS服务器
                protocols=["UDP"],
                port_range=(53, 53)
            ),
            "ddos_attack": TrafficPattern(
                name="DDoS攻击",
                description="大量并发请求攻击",
                packet_rate=100.0,
                duration=30,
                packet_size_range=(64, 512),
                src_ips=self.hosts,
                dst_ips=["10.0.0.3"],  # 目标服务器
                protocols=["TCP", "UDP"],
                port_range=(80, 8080)
            ),
            "port_scan": TrafficPattern(
                name="端口扫描",
                description="系统性端口探测",
                packet_rate=10.0,
                duration=20,
                packet_size_range=(40, 100),
                src_ips=["10.0.0.1"],
                dst_ips=["10.0.0.4"],
                protocols=["TCP"],
                port_range=(1, 65535)
            ),
            "large_transfer": TrafficPattern(
                name="大文件传输",
                description="异常大包传输",
                packet_rate=20.0,
                duration=15,
                packet_size_range=(8000, 9000),
                src_ips=["10.0.0.2"],
                dst_ips=["10.0.0.5"],
                protocols=["TCP"],
                port_range=(21, 21)  # FTP
            )
        }

    def set_packet_callback(self, callback: Callable):
        """设置数据包处理回调函数"""
        self.packet_callback = callback

    def start_traffic_generation(self, patterns: List[str], sequential=True):
        """开始生成流量"""
        print(f"开始生成流量模式: {patterns}")
        self.running = True
        self.stats.clear()

        if sequential:
            # 顺序执行模式
            thread = threading.Thread(target=self._sequential_generation, args=(patterns,))
        else:
            # 并行执行模式
            thread = threading.Thread(target=self._parallel_generation, args=(patterns,))

        thread.daemon = True
        thread.start()
        self.threads.append(thread)

    def stop_traffic_generation(self):
        """停止流量生成"""
        print("停止生成流量...")
        self.running = False

        for thread in self.threads:
            try:
                thread.join(timeout=5)
            except:
                pass

        self.threads.clear()
        print("流量生成已停止")

    def _sequential_generation(self, patterns: List[str]):
        """顺序生成流量模式"""
        for pattern_name in patterns:
            if not self.running:
                break

            if pattern_name in self.patterns:
                pattern = self.patterns[pattern_name]
                print(f"Generating traffic pattern: {pattern.name}")
                self._generate_pattern_traffic(pattern)
                print(f"Completed pattern: {pattern.name}")

                # 模式间休息
                if self.running:
                    time.sleep(5)

    def _parallel_generation(self, patterns: List[str]):
        """并行生成流量模式"""
        pattern_threads = []

        for pattern_name in patterns:
            if pattern_name in self.patterns:
                pattern = self.patterns[pattern_name]
                thread = threading.Thread(
                    target=self._generate_pattern_traffic,
                    args=(pattern,)
                )
                thread.daemon = True
                thread.start()
                pattern_threads.append(thread)

        # 等待所有模式完成
        for thread in pattern_threads:
            thread.join()

    def _generate_pattern_traffic(self, pattern: TrafficPattern):
        """生成特定模式的流量"""
        start_time = time.time()
        packet_interval = 1.0 / pattern.packet_rate
        packets_generated = 0

        while self.running and (time.time() - start_time) < pattern.duration:
            # 生成数据包
            packet = self._create_packet(pattern)

            if packet:
                self.generated_packets.append(packet)
                self.stats[f"{pattern.name}_packets"] += 1
                self.stats[f"{pattern.name}_bytes"] += packet.size
                packets_generated += 1

                # 发送到回调函数
                if self.packet_callback:
                    self.packet_callback(packet)

            # 等待下个包生成时间
            time.sleep(packet_interval)

        print(f"模式 {pattern.name}: 已生成 {packets_generated} 个数据包")

    def _create_packet(self, pattern: TrafficPattern) -> Optional[NetworkPacket]:
        """根据模式创建数据包"""
        try:
            # 随机选择源和目标
            src_ip = random.choice(pattern.src_ips)
            dst_ip = random.choice(pattern.dst_ips)

            # 防止自己给自己发包
            if src_ip == dst_ip and len(pattern.dst_ips) > 1:
                dst_ip = random.choice([ip for ip in pattern.dst_ips if ip != src_ip])

            protocol = random.choice(pattern.protocols)

            # 端口选择逻辑
            if pattern.name == "端口扫描":
                # 端口扫描：顺序扫描端口
                port_start = pattern.port_range[0]
                current_time = time.time()
                port_offset = int(current_time * pattern.packet_rate) % 1000
                dst_port = port_start + port_offset
                src_port = random.randint(30000, 65000)
            elif pattern.port_range[0] == pattern.port_range[1]:
                # 固定端口
                dst_port = pattern.port_range[0]
                src_port = random.randint(30000, 65000)
            else:
                # 端口范围
                dst_port = random.randint(pattern.port_range[0], pattern.port_range[1])
                src_port = random.randint(30000, 65000)

            # 包大小
            size = random.randint(pattern.packet_size_range[0], pattern.packet_size_range[1])

            packet = NetworkPacket(
                src_ip=src_ip,
                dst_ip=dst_ip,
                src_port=src_port,
                dst_port=dst_port,
                protocol=protocol,
                size=size,
                timestamp=time.time()
            )

            return packet

        except Exception as e:
            print(f"Error creating packet: {e}")
            return None

    def generate_baseline_traffic(self, duration: int = 300):
        """生成基线正常流量"""
        print(f"Generating baseline traffic for {duration} seconds...")

        baseline_patterns = ["normal_web", "normal_dns"]
        self.start_traffic_generation(baseline_patterns, sequential=False)

        # 等待基线流量完成
        time.sleep(duration)
        self.stop_traffic_generation()

        print("Baseline traffic generation completed")

    def generate_attack_scenarios(self):
        """生成攻击场景流量"""
        print("Starting attack scenario generation...")

        # 场景1：正常流量 + DDoS攻击
        print("\n=== Scenario 1: DDoS Attack ===")
        self.start_traffic_generation(["normal_web", "ddos_attack"], sequential=False)
        time.sleep(45)
        self.stop_traffic_generation()

        time.sleep(10)  # 场景间隔

        # 场景2：端口扫描
        print("\n=== Scenario 2: Port Scanning ===")
        self.start_traffic_generation(["normal_dns", "port_scan"], sequential=False)
        time.sleep(30)
        self.stop_traffic_generation()

        time.sleep(10)

        # 场景3：异常大包传输
        print("\n=== Scenario 3: Large File Transfer ===")
        self.start_traffic_generation(["normal_web", "large_transfer"], sequential=False)
        time.sleep(25)
        self.stop_traffic_generation()

        print("Attack scenarios completed")

    def get_traffic_stats(self) -> Dict:
        """获取流量生成统计"""
        total_packets = sum(v for k, v in self.stats.items() if k.endswith('_packets'))
        total_bytes = sum(v for k, v in self.stats.items() if k.endswith('_bytes'))

        return {
            'timestamp': time.time(),
            'total_packets_generated': total_packets,
            'total_bytes_generated': total_bytes,
            'pattern_stats': dict(self.stats),
            'active_patterns': len([k for k in self.stats.keys() if k.endswith('_packets') and self.stats[k] > 0])
        }

    def save_traffic_data(self, filename: str = None):
        """保存生成的流量数据"""
        if filename is None:
            timestamp = int(time.time())
            filename = f"traffic_data_{timestamp}.json"

        data = {
            'generation_stats': self.get_traffic_stats(),
            'generated_packets': [
                {
                    'src_ip': p.src_ip,
                    'dst_ip': p.dst_ip,
                    'src_port': p.src_port,
                    'dst_port': p.dst_port,
                    'protocol': p.protocol,
                    'size': p.size,
                    'timestamp': p.timestamp,
                    'flow_id': p.flow_id
                }
                for p in self.generated_packets[-1000:]  # 保存最近1000个包
            ]
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Traffic data saved to {filename}")


# 测试函数
def test_traffic_generator():
    """测试流量生成器"""
    generator = TrafficGenerator()

    # 设置简单的包处理回调
    def packet_handler(packet):
        print(f"Generated: {packet.flow_id} ({packet.size} bytes)")

    generator.set_packet_callback(packet_handler)

    # 测试正常流量生成
    print("Testing normal traffic generation...")
    generator.start_traffic_generation(["normal_web"], sequential=True)
    time.sleep(10)
    generator.stop_traffic_generation()

    # 打印统计
    stats = generator.get_traffic_stats()
    print("\nTraffic Generation Stats:")
    print(json.dumps(stats, indent=2))

    # 保存数据
    generator.save_traffic_data("test_traffic.json")


if __name__ == "__main__":
    test_traffic_generator()