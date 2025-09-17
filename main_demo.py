#!/usr/bin/env python3
"""
SDN + AI网络异常检测系统完整演示
整合所有组件：SDN网络模拟、流量生成、AI异常检测、Web监控界面
"""

import time
import threading
import signal
import sys
import os
from pathlib import Path

# 添加模块路径
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from network_sim.sdn_network import SDNNetwork, NetworkPacket
from traffic_gen.packet_generator import TrafficGenerator
from ai_detector.anomaly_detection import ComprehensiveAnomalyDetector
from monitoring.web_dashboard import NetworkMonitor, app as dashboard_app


class SDNAIDemo:
    """SDN + AI异常检测系统完整演示"""

    def __init__(self):
        self.sdn_network = SDNNetwork()
        self.traffic_generator = TrafficGenerator()
        self.anomaly_detector = ComprehensiveAnomalyDetector()
        self.network_monitor = NetworkMonitor()

        self.running = False
        self.demo_threads = []

        # 演示配置
        self.demo_phases = [
            {"name": "基线建立", "duration": 60, "patterns": ["normal_web", "normal_dns"]},
            {"name": "DDoS攻击", "duration": 45, "patterns": ["normal_web", "ddos_attack"]},
            {"name": "端口扫描", "duration": 30, "patterns": ["normal_dns", "port_scan"]},
            {"name": "大文件传输", "duration": 25, "patterns": ["normal_web", "large_transfer"]},
            {"name": "混合场景", "duration": 40, "patterns": ["normal_web", "ddos_attack", "port_scan"]}
        ]

    def setup_system(self):
        """初始化系统组件"""
        print("="*60)
        print("SDN + AI 网络异常检测系统初始化")
        print("="*60)

        # 1. 设置SDN网络拓扑
        print("1. 设置SDN网络拓扑...")
        self.sdn_network.setup_topology()

        # 2. 配置流量生成器
        print("2. 配置流量生成器...")
        self.traffic_generator.set_packet_callback(self._handle_generated_packet)

        # 3. 启动网络监控
        print("3. 启动网络监控...")
        self.network_monitor.start_monitoring()

        print("系统初始化完成！\n")

    def _handle_generated_packet(self, packet: NetworkPacket):
        """处理生成的数据包"""
        # 注入到SDN网络
        self.sdn_network.inject_packet(packet)

        # 批量处理检测（提高效率）
        if not hasattr(self, '_packet_batch'):
            self._packet_batch = []
            self._last_detection = time.time()

        self._packet_batch.append(packet)

        # 每100个包或每5秒进行一次检测
        if (len(self._packet_batch) >= 100 or
            time.time() - self._last_detection > 5):
            self._process_detection_batch()

    def _process_detection_batch(self):
        """批量处理异常检测"""
        if not hasattr(self, '_packet_batch') or not self._packet_batch:
            return

        # AI异常检测
        detection_result = self.anomaly_detector.process_packets(self._packet_batch)

        if detection_result:
            # 更新网络统计
            network_stats = self.sdn_network.get_network_status()
            self.network_monitor.add_network_stats(network_stats)

            # 如果检测到异常，记录事件
            if detection_result.anomaly_detected:
                anomaly_event = {
                    'type': detection_result.anomaly_type,
                    'confidence': detection_result.confidence,
                    'source_ip': self._packet_batch[-1].src_ip if self._packet_batch else "unknown",
                    'target_ip': self._packet_batch[-1].dst_ip if self._packet_batch else "unknown",
                    'details': f"检测到{detection_result.anomaly_type}，置信度: {detection_result.confidence:.2f}",
                    'actions_taken': detection_result.recommended_actions[:3]  # 前3个推荐动作
                }
                self.network_monitor.add_anomaly_event(anomaly_event)

                print(f"\n🚨 异常检测: {detection_result.anomaly_type}")
                print(f"   置信度: {detection_result.confidence:.2f}")
                print(f"   推荐动作: {', '.join(detection_result.recommended_actions[:2])}")

        # 清空批次
        self._packet_batch = []
        self._last_detection = time.time()

    def run_demo_phases(self):
        """运行演示阶段"""
        print("开始运行演示场景...")
        print("监控界面: http://localhost:8080")
        print("="*60)

        total_duration = sum(phase["duration"] for phase in self.demo_phases)
        print(f"总演示时间: {total_duration // 60}分{total_duration % 60}秒\n")

        for i, phase in enumerate(self.demo_phases, 1):
            if not self.running:
                break

            print(f"\n阶段 {i}/{len(self.demo_phases)}: {phase['name']}")
            print(f"   持续时间: {phase['duration']}秒")
            print(f"   流量模式: {', '.join(phase['patterns'])}")

            # 启动该阶段的流量生成
            self.traffic_generator.start_traffic_generation(
                phase['patterns'],
                sequential=False
            )

            # 等待该阶段完成
            start_time = time.time()
            while (time.time() - start_time) < phase['duration'] and self.running:
                elapsed = time.time() - start_time
                remaining = phase['duration'] - elapsed
                progress = (elapsed / phase['duration']) * 100

                print(f"\r   进度: {progress:.1f}% | 剩余: {remaining:.0f}s", end="", flush=True)
                time.sleep(1)

            print()  # 换行

            # 停止当前阶段的流量
            self.traffic_generator.stop_traffic_generation()

            # 阶段间休息
            if i < len(self.demo_phases) and self.running:
                print("   阶段间隔...")
                time.sleep(5)

        print("\n✅ 所有演示阶段完成")

    def run_performance_tests(self):
        """运行性能测试"""
        print("\n运行性能测试...")

        try:
            # 获取最终统计
            network_stats = self.sdn_network.get_network_status()
            traffic_stats = self.traffic_generator.get_traffic_stats()
            detection_summary = self.anomaly_detector.get_detection_summary()

            print("\n" + "="*60)
            print("系统性能报告")
            print("="*60)

            # 网络性能
            print("\n网络性能:")
            print(f"  总流量: {network_stats['traffic']['total_flows']} 流")
            print(f"  总数据包: {network_stats['traffic']['total_packets']} 包")
            print(f"  总字节数: {network_stats['traffic']['total_bytes']:,} 字节")
            print(f"  阻断流量: {network_stats['traffic']['blocked_flows']} 流")

            # 流量生成性能
            print("\n流量生成:")
            print(f"  生成数据包: {traffic_stats['total_packets_generated']} 包")
            print(f"  生成字节数: {traffic_stats['total_bytes_generated']:,} 字节")
            print(f"  活跃模式: {traffic_stats['active_patterns']} 个")

            # AI检测性能
            print("\nAI检测性能:")
            print(f"  处理数据包: {detection_summary['total_packets_processed']} 包")
            print(f"  检测异常: {detection_summary['total_anomalies_detected']} 次")
            print(f"  近期异常: {detection_summary['recent_anomalies']} 次")
            print(f"  基线建立: {'✅' if detection_summary['baseline_established'] else '❌'}")
            print(f"  模型训练: {'✅' if detection_summary['model_trained'] else '❌'}")

            # 异常类型统计
            if detection_summary['anomaly_types']:
                print("\n🔍 异常类型分布:")
                for anomaly_type, count in detection_summary['anomaly_types'].items():
                    print(f"  {anomaly_type}: {count} 次")

            print("\n" + "="*60)

        except Exception as e:
            print(f"性能测试错误: {e}")

    def start_web_dashboard(self):
        """启动Web监控界面"""
        def run_dashboard():
            try:
                dashboard_app.run(
                    host='0.0.0.0',
                    port=8080,
                    debug=False,
                    threaded=True,
                    use_reloader=False
                )
            except Exception as e:
                print(f"Dashboard error: {e}")

        dashboard_thread = threading.Thread(target=run_dashboard)
        dashboard_thread.daemon = True
        dashboard_thread.start()
        self.demo_threads.append(dashboard_thread)

        print("Web监控界面已启动: http://localhost:8080")

    def run_complete_demo(self):
        """运行完整演示"""
        try:
            print("启动SDN + AI网络异常检测系统演示")
            print("技术栈: SDN网络模拟 + 机器学习 + 实时监控")
            print("作业要求: ✅ SDN ✅ P4概念 ✅ AI ✅ 异常检测 ✅ 演示界面")

            # 设置信号处理
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)

            self.running = True

            # 1. 系统初始化
            self.setup_system()

            # 2. 启动Web界面
            self.start_web_dashboard()

            # 3. 启动SDN网络
            self.sdn_network.start_network()

            # 等待Web界面启动
            time.sleep(3)

            # 4. 运行演示阶段
            self.run_demo_phases()

            # 5. 性能测试
            self.run_performance_tests()

            # 6. 保持系统运行供查看
            print("\n演示完成！系统将继续运行以供查看...")
            print("   监控界面: http://localhost:8080")
            print("   按 Ctrl+C 退出演示")

            while self.running:
                time.sleep(1)

        except KeyboardInterrupt:
            print("\n收到中断信号，正在停止演示...")
        except Exception as e:
            print(f"\n演示过程中发生错误: {e}")
        finally:
            self.cleanup()

    def _signal_handler(self, signum, frame):
        """信号处理器"""
        print(f"\n收到信号 {signum}，正在优雅停止...")
        self.running = False

    def cleanup(self):
        """清理资源"""
        print("\n清理系统资源...")

        try:
            # 停止流量生成
            self.traffic_generator.stop_traffic_generation()

            # 停止SDN网络
            self.sdn_network.stop_network()

            # 停止监控
            self.network_monitor.stop_monitoring()

            # 保存演示数据
            self.save_demo_results()

            print("✅ 系统资源清理完成")

        except Exception as e:
            print(f"清理过程中发生错误: {e}")

    def save_demo_results(self):
        """保存演示结果"""
        try:
            timestamp = int(time.time())

            # 保存流量数据
            self.traffic_generator.save_traffic_data(f"demo_traffic_{timestamp}.json")

            # 保存网络统计
            network_stats = self.sdn_network.get_network_status()
            with open(f"demo_network_{timestamp}.json", 'w') as f:
                import json
                json.dump(network_stats, f, indent=2)

            # 保存检测摘要
            detection_summary = self.anomaly_detector.get_detection_summary()
            with open(f"demo_detection_{timestamp}.json", 'w') as f:
                import json
                json.dump(detection_summary, f, indent=2)

            print(f"演示结果已保存到 demo_*_{timestamp}.json 文件")

        except Exception as e:
            print(f"保存演示结果时发生错误: {e}")


def main():
    """主函数"""
    print("SDN + AI 网络异常检测系统")
    print("=" * 50)

    # 检查Python版本
    if sys.version_info < (3, 7):
        print("❌ 需要 Python 3.7 或更高版本")
        sys.exit(1)

    # 检查依赖包
    required_packages = ['numpy', 'pandas', 'sklearn', 'flask', 'plotly', 'networkx']
    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print(f"❌ 缺少依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        sys.exit(1)

    # 运行演示
    demo = SDNAIDemo()
    demo.run_complete_demo()


if __name__ == "__main__":
    main()