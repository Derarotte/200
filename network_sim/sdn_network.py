#!/usr/bin/env python3
"""
SDN网络拓扑模拟器
实现软件定义网络的核心概念：集中控制、流表管理、可编程数据平面
"""

import time
import json
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import networkx as nx


@dataclass
class FlowRule:
    """OpenFlow流表规则"""
    match_fields: Dict[str, str]  # 匹配字段
    actions: List[str]  # 动作列表
    priority: int = 100
    timeout: int = 300
    packet_count: int = 0
    byte_count: int = 0
    timestamp: float = 0

    def __post_init__(self):
        if self.timestamp == 0:
            self.timestamp = time.time()


@dataclass
class NetworkPacket:
    """网络数据包"""
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    size: int
    timestamp: float
    flow_id: str = ""

    def __post_init__(self):
        if not self.flow_id:
            self.flow_id = f"{self.src_ip}:{self.src_port}->{self.dst_ip}:{self.dst_port}:{self.protocol}"


class SDNSwitch:
    """SDN交换机模拟"""

    def __init__(self, switch_id: str):
        self.switch_id = switch_id
        self.flow_table: List[FlowRule] = []
        self.ports: Dict[int, str] = {}  # port_id -> neighbor
        self.packet_buffer: deque = deque(maxlen=1000)
        self.statistics = {
            'packets_processed': 0,
            'bytes_processed': 0,
            'flows_installed': 0,
            'table_misses': 0
        }
        self.connected_hosts: List[str] = []

    def add_port(self, port_id: int, neighbor: str):
        """添加端口连接"""
        self.ports[port_id] = neighbor

    def install_flow_rule(self, rule: FlowRule):
        """安装流表规则"""
        # 检查是否有相同匹配条件的规则
        for i, existing_rule in enumerate(self.flow_table):
            if existing_rule.match_fields == rule.match_fields:
                self.flow_table[i] = rule  # 替换
                return

        # 按优先级插入
        inserted = False
        for i, existing_rule in enumerate(self.flow_table):
            if rule.priority > existing_rule.priority:
                self.flow_table.insert(i, rule)
                inserted = True
                break

        if not inserted:
            self.flow_table.append(rule)

        self.statistics['flows_installed'] += 1

    def process_packet(self, packet: NetworkPacket) -> Tuple[bool, List[str]]:
        """处理数据包，返回是否匹配和转发动作"""
        self.packet_buffer.append(packet)
        self.statistics['packets_processed'] += 1
        self.statistics['bytes_processed'] += packet.size

        # 查找匹配的流表规则
        for rule in self.flow_table:
            if self._match_packet(packet, rule.match_fields):
                rule.packet_count += 1
                rule.byte_count += packet.size
                return True, rule.actions

        # 没有匹配的规则，触发table miss
        self.statistics['table_misses'] += 1
        return False, ["CONTROLLER"]  # 发送到控制器

    def _match_packet(self, packet: NetworkPacket, match_fields: Dict[str, str]) -> bool:
        """检查数据包是否匹配流表规则"""
        for field, value in match_fields.items():
            if field == "src_ip" and packet.src_ip != value:
                return False
            elif field == "dst_ip" and packet.dst_ip != value:
                return False
            elif field == "protocol" and packet.protocol != value:
                return False
            elif field == "src_port" and packet.src_port != int(value):
                return False
            elif field == "dst_port" and packet.dst_port != int(value):
                return False
        return True

    def get_statistics(self) -> Dict:
        """获取交换机统计信息"""
        return {
            'switch_id': self.switch_id,
            **self.statistics,
            'flow_table_size': len(self.flow_table),
            'active_flows': len([r for r in self.flow_table if r.packet_count > 0])
        }


class SDNController:
    """SDN控制器模拟 - 体现集中控制和智能路由"""

    def __init__(self):
        self.switches: Dict[str, SDNSwitch] = {}
        self.topology = nx.Graph()
        self.flow_stats = defaultdict(dict)
        self.routing_table = {}
        self.security_policies = []
        self.qos_policies = {}
        self.blocked_flows = set()

    def add_switch(self, switch: SDNSwitch):
        """添加交换机到控制器"""
        self.switches[switch.switch_id] = switch
        self.topology.add_node(switch.switch_id)

    def add_link(self, switch1_id: str, switch2_id: str, bandwidth: int = 100):
        """添加交换机间链路"""
        self.topology.add_edge(switch1_id, switch2_id, bandwidth=bandwidth)

    def install_proactive_flows(self):
        """安装主动流表规则 - SDN的预配置能力"""
        print("控制器: 正在安装主动流规则...")

        # 为每个交换机安装基础转发规则
        for switch_id, switch in self.switches.items():
            # ARP处理规则
            arp_rule = FlowRule(
                match_fields={"protocol": "ARP"},
                actions=["FLOOD"],
                priority=200
            )
            switch.install_flow_rule(arp_rule)

            # 默认转发规则
            default_rule = FlowRule(
                match_fields={},
                actions=["CONTROLLER"],
                priority=1
            )
            switch.install_flow_rule(default_rule)

    def handle_packet_in(self, switch_id: str, packet: NetworkPacket):
        """处理PacketIn消息 - 控制器的智能决策"""
        print(f"控制器: 来自{switch_id}的数据包输入, 流: {packet.flow_id}")

        # 安全检查
        if self._is_malicious_flow(packet):
            self._install_block_rule(switch_id, packet)
            return

        # 计算最佳路径
        path = self._calculate_path(packet.src_ip, packet.dst_ip)

        if path:
            # 安装端到端流表规则
            self._install_path_rules(packet, path)

        # 更新流量统计
        self._update_flow_stats(packet)

    def _is_malicious_flow(self, packet: NetworkPacket) -> bool:
        """安全策略检查"""
        flow_key = packet.flow_id

        # 检查是否在黑名单中
        if flow_key in self.blocked_flows:
            return True

        # 简单的DDoS检测（高频率同源请求）
        current_time = time.time()
        src_flows = [p for p in self.flow_stats.get(packet.src_ip, {}).values()
                    if current_time - p.get('last_seen', 0) < 60]

        if len(src_flows) > 100:  # 1分钟内超过100个流
            print(f"Controller: DDoS detected from {packet.src_ip}")
            return True

        return False

    def _install_block_rule(self, switch_id: str, packet: NetworkPacket):
        """安装阻断规则"""
        block_rule = FlowRule(
            match_fields={"src_ip": packet.src_ip},
            actions=["DROP"],
            priority=1000,
            timeout=600  # 10分钟阻断
        )

        if switch_id in self.switches:
            self.switches[switch_id].install_flow_rule(block_rule)
            self.blocked_flows.add(packet.flow_id)
            print(f"Controller: Installed block rule for {packet.src_ip}")

    def _calculate_path(self, src_ip: str, dst_ip: str) -> List[str]:
        """计算最佳路径 - 体现SDN的全局视图和智能路由"""
        # 简化的路径计算：假设IP地址映射到交换机
        src_switch = self._ip_to_switch(src_ip)
        dst_switch = self._ip_to_switch(dst_ip)

        if src_switch == dst_switch:
            return [src_switch]

        try:
            # 使用网络拓扑计算最短路径
            path = nx.shortest_path(self.topology, src_switch, dst_switch)
            print(f"控制器: 计算路径 {src_switch} -> {dst_switch}: {' -> '.join(path)}")
            return path
        except nx.NetworkXNoPath:
            print(f"Controller: No path found between {src_switch} and {dst_switch}")
            return []

    def _ip_to_switch(self, ip: str) -> str:
        """IP地址到交换机的映射（简化）"""
        # 根据IP地址的最后一位决定连接的交换机
        last_octet = int(ip.split('.')[-1])
        if last_octet <= 2:
            return "s1"
        elif last_octet <= 4:
            return "s2"
        elif last_octet <= 6:
            return "s3"
        else:
            return "s4"

    def _install_path_rules(self, packet: NetworkPacket, path: List[str]):
        """在路径上安装流表规则"""
        for i, switch_id in enumerate(path):
            if switch_id not in self.switches:
                continue

            # 确定输出端口
            if i == len(path) - 1:
                # 最后一个交换机，转发到主机
                output_action = f"OUTPUT:{packet.dst_ip}"
            else:
                # 中间交换机，转发到下一跳
                next_switch = path[i + 1]
                output_action = f"OUTPUT:{next_switch}"

            # 安装双向流表规则
            forward_rule = FlowRule(
                match_fields={
                    "src_ip": packet.src_ip,
                    "dst_ip": packet.dst_ip,
                    "protocol": packet.protocol
                },
                actions=[output_action],
                priority=300
            )

            self.switches[switch_id].install_flow_rule(forward_rule)

    def _update_flow_stats(self, packet: NetworkPacket):
        """更新流量统计"""
        if packet.src_ip not in self.flow_stats:
            self.flow_stats[packet.src_ip] = {}

        flow_key = packet.flow_id
        if flow_key not in self.flow_stats[packet.src_ip]:
            self.flow_stats[packet.src_ip][flow_key] = {
                'packet_count': 0,
                'byte_count': 0,
                'first_seen': time.time(),
                'last_seen': time.time()
            }

        stats = self.flow_stats[packet.src_ip][flow_key]
        stats['packet_count'] += 1
        stats['byte_count'] += packet.size
        stats['last_seen'] = time.time()

    def get_network_stats(self) -> Dict:
        """获取全网统计信息"""
        total_switches = len(self.switches)
        total_flows = sum(len(s.flow_table) for s in self.switches.values())
        total_packets = sum(s.statistics['packets_processed'] for s in self.switches.values())
        total_bytes = sum(s.statistics['bytes_processed'] for s in self.switches.values())

        return {
            'timestamp': time.time(),
            'topology': {
                'switches': total_switches,
                'links': len(self.topology.edges),
                'hosts': sum(len(s.connected_hosts) for s in self.switches.values())
            },
            'traffic': {
                'total_flows': total_flows,
                'total_packets': total_packets,
                'total_bytes': total_bytes,
                'blocked_flows': len(self.blocked_flows)
            },
            'switches': {s_id: s.get_statistics() for s_id, s in self.switches.items()}
        }


class SDNNetwork:
    """完整的SDN网络模拟系统"""

    def __init__(self):
        self.controller = SDNController()
        self.running = False
        self.packet_queue = deque()
        self.network_thread = None

    def setup_topology(self):
        """创建4交换机6主机的网络拓扑"""
        print("正在设置SDN网络拓扑...")

        # 创建4个交换机
        switches = []
        for i in range(1, 5):
            switch = SDNSwitch(f"s{i}")
            switches.append(switch)
            self.controller.add_switch(switch)

        # 创建交换机间连接（形成冗余拓扑）
        connections = [
            ("s1", "s2"), ("s2", "s3"), ("s3", "s4"), ("s4", "s1"),  # 环形
            ("s1", "s3"), ("s2", "s4")  # 对角线
        ]

        for s1, s2 in connections:
            self.controller.add_link(s1, s2, bandwidth=100)

        # 连接主机到交换机
        host_mappings = {
            "s1": ["10.0.0.1", "10.0.0.2"],
            "s2": ["10.0.0.3", "10.0.0.4"],
            "s3": ["10.0.0.5", "10.0.0.6"],
            "s4": []
        }

        for switch_id, hosts in host_mappings.items():
            if switch_id in self.controller.switches:
                self.controller.switches[switch_id].connected_hosts = hosts

        # 安装基础流表规则
        self.controller.install_proactive_flows()

        print("SDN拓扑设置完成")
        print(f"- 交换机: {len(self.controller.switches)}")
        print(f"- 链路: {len(self.controller.topology.edges)}")
        print(f"- 主机: {sum(len(hosts) for hosts in host_mappings.values())}")

    def inject_packet(self, packet: NetworkPacket):
        """向网络注入数据包"""
        self.packet_queue.append(packet)

    def start_network(self):
        """启动网络处理"""
        self.running = True
        self.network_thread = threading.Thread(target=self._network_loop)
        self.network_thread.daemon = True
        self.network_thread.start()
        print("SDN网络已启动")

    def stop_network(self):
        """停止网络处理"""
        self.running = False
        if self.network_thread:
            self.network_thread.join()
        print("SDN network stopped")

    def _network_loop(self):
        """网络处理主循环"""
        while self.running:
            try:
                if self.packet_queue:
                    packet = self.packet_queue.popleft()
                    self._process_packet(packet)
                else:
                    time.sleep(0.01)  # 避免忙等待
            except Exception as e:
                print(f"Network processing error: {e}")

    def _process_packet(self, packet: NetworkPacket):
        """处理单个数据包"""
        # 确定入口交换机
        entry_switch_id = self.controller._ip_to_switch(packet.src_ip)

        if entry_switch_id in self.controller.switches:
            entry_switch = self.controller.switches[entry_switch_id]

            # 交换机处理数据包
            matched, actions = entry_switch.process_packet(packet)

            if not matched or "CONTROLLER" in actions:
                # 发送PacketIn到控制器
                self.controller.handle_packet_in(entry_switch_id, packet)

    def get_network_status(self) -> Dict:
        """获取网络状态"""
        return self.controller.get_network_stats()


# 测试函数
def test_sdn_network():
    """测试SDN网络功能"""
    network = SDNNetwork()
    network.setup_topology()
    network.start_network()

    # 模拟一些网络流量
    test_packets = [
        NetworkPacket("10.0.0.1", "10.0.0.3", 80, 8080, "TCP", 1500, time.time()),
        NetworkPacket("10.0.0.2", "10.0.0.4", 443, 9443, "TCP", 800, time.time()),
        NetworkPacket("10.0.0.5", "10.0.0.1", 22, 2222, "TCP", 200, time.time())
    ]

    for packet in test_packets:
        network.inject_packet(packet)
        time.sleep(0.1)

    time.sleep(2)  # 等待处理完成

    # 打印网络状态
    status = network.get_network_status()
    print("\nNetwork Status:")
    print(json.dumps(status, indent=2))

    network.stop_network()


if __name__ == "__main__":
    test_sdn_network()