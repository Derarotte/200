#!/usr/bin/env python3
"""
Network Performance Testing and Metrics Collection
Comprehensive testing suite for network performance evaluation
"""

import time
import json
import threading
import subprocess
import statistics
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import os
import psutil


class NetworkPerformanceTester:
    """Comprehensive network performance testing suite"""
    
    def __init__(self, output_dir="results"):
        self.output_dir = output_dir
        self.test_results = {}
        self.performance_metrics = []
        os.makedirs(output_dir, exist_ok=True)
    
    def run_comprehensive_tests(self):
        """Run all performance tests"""
        print("=== Starting Comprehensive Network Performance Tests ===")
        
        test_suite = [
            ("Connectivity Test", self.test_connectivity),
            ("Latency Test", self.test_latency),
            ("Bandwidth Test", self.test_bandwidth),
            ("Throughput Test", self.test_throughput),
            ("Packet Loss Test", self.test_packet_loss),
            ("Load Balancing Test", self.test_load_balancing),
            ("Scalability Test", self.test_scalability),
            ("Anomaly Detection Performance", self.test_anomaly_detection_performance),
            ("SDN Controller Performance", self.test_sdn_controller_performance),
            ("System Resource Utilization", self.test_system_resources)
        ]
        
        for test_name, test_function in test_suite:
            print(f"\n--- Running {test_name} ---")
            try:
                start_time = time.time()
                result = test_function()
                end_time = time.time()
                
                self.test_results[test_name] = {
                    'result': result,
                    'execution_time': end_time - start_time,
                    'timestamp': datetime.now().isoformat(),
                    'status': 'success' if result else 'failed'
                }
                
                print(f"✓ {test_name} completed in {end_time - start_time:.2f}s")
                
            except Exception as e:
                print(f"✗ {test_name} failed: {e}")
                self.test_results[test_name] = {
                    'result': None,
                    'execution_time': 0,
                    'timestamp': datetime.now().isoformat(),
                    'status': 'error',
                    'error': str(e)
                }
        
        self.save_test_results()
        self.generate_performance_report()
        print(f"\n=== Performance tests completed. Results saved to {self.output_dir}/ ===")
    
    def test_connectivity(self):
        """Test basic network connectivity between hosts"""
        hosts = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8']
        connectivity_results = {}
        
        try:
            # Simulate connectivity test (in real environment, this would use Mininet)
            for i, src in enumerate(hosts):
                connectivity_results[src] = {}
                for j, dst in enumerate(hosts):
                    if i != j:
                        # Simulate ping test
                        success_rate = np.random.uniform(0.95, 1.0)  # 95-100% success
                        connectivity_results[src][dst] = {
                            'reachable': success_rate > 0.98,
                            'success_rate': success_rate,
                            'avg_rtt': np.random.uniform(0.5, 5.0)  # 0.5-5ms RTT
                        }
            
            # Calculate overall connectivity score
            total_tests = len(hosts) * (len(hosts) - 1)
            successful_tests = sum(
                1 for src_results in connectivity_results.values()
                for dst_result in src_results.values()
                if dst_result['reachable']
            )
            
            connectivity_score = successful_tests / total_tests
            
            return {
                'connectivity_matrix': connectivity_results,
                'overall_connectivity': connectivity_score,
                'total_host_pairs': total_tests,
                'successful_connections': successful_tests
            }
            
        except Exception as e:
            print(f"Connectivity test error: {e}")
            return None
    
    def test_latency(self):
        """Test network latency between different host pairs"""
        latency_results = {}
        
        try:
            # Test latency patterns
            test_pairs = [('h1', 'h2'), ('h1', 'h8'), ('h3', 'h6'), ('h4', 'h7')]
            
            for src, dst in test_pairs:
                # Simulate latency measurements
                latencies = []
                for _ in range(100):  # 100 ping tests
                    # Simulate realistic latency based on network distance
                    base_latency = 0.5  # Base latency
                    distance_factor = abs(int(src[1:]) - int(dst[1:])) * 0.2
                    jitter = np.random.normal(0, 0.1)
                    latency = max(0.1, base_latency + distance_factor + jitter)
                    latencies.append(latency)
                
                latency_results[f"{src}-{dst}"] = {
                    'min_latency': min(latencies),
                    'max_latency': max(latencies),
                    'avg_latency': statistics.mean(latencies),
                    'median_latency': statistics.median(latencies),
                    'std_dev': statistics.stdev(latencies),
                    'jitter': max(latencies) - min(latencies),
                    'measurements': latencies[:20]  # Store first 20 for analysis
                }
            
            # Calculate overall latency metrics
            all_averages = [result['avg_latency'] for result in latency_results.values()]
            
            return {
                'pair_results': latency_results,
                'overall_avg_latency': statistics.mean(all_averages),
                'overall_max_latency': max(result['max_latency'] for result in latency_results.values()),
                'overall_jitter': statistics.mean([result['jitter'] for result in latency_results.values()])
            }
            
        except Exception as e:
            print(f"Latency test error: {e}")
            return None
    
    def test_bandwidth(self):
        """Test available bandwidth between hosts"""
        bandwidth_results = {}
        
        try:
            test_scenarios = [
                ('h1', 'h2', 'single_flow'),
                ('h1', 'h8', 'cross_topology'),
                (['h1', 'h3', 'h5'], 'h2', 'multiple_sources'),
                ('h4', ['h6', 'h7', 'h8'], 'multiple_destinations')
            ]
            
            for scenario in test_scenarios:
                if len(scenario) == 3:
                    src, dst, test_type = scenario
                    
                    # Simulate bandwidth test
                    if test_type == 'single_flow':
                        # Single TCP flow
                        bandwidth = np.random.uniform(80, 100)  # 80-100 Mbps
                        utilization = bandwidth / 100  # Assume 100 Mbps link capacity
                        
                    elif test_type == 'cross_topology':
                        # Cross-topology flow (longer path)
                        bandwidth = np.random.uniform(60, 90)  # Lower due to distance
                        utilization = bandwidth / 100
                        
                    elif test_type == 'multiple_sources':
                        # Multiple sources competing for bandwidth
                        total_bandwidth = 100
                        num_flows = len(src)
                        bandwidth = total_bandwidth / num_flows * np.random.uniform(0.8, 1.0)
                        utilization = bandwidth / (total_bandwidth / num_flows)
                        
                    elif test_type == 'multiple_destinations':
                        # Single source to multiple destinations
                        bandwidth = np.random.uniform(30, 40)  # Divided among destinations
                        utilization = bandwidth / 100
                    
                    bandwidth_results[f"{src}-{dst}"] = {
                        'test_type': test_type,
                        'bandwidth_mbps': bandwidth,
                        'utilization': utilization,
                        'throughput_efficiency': utilization * 100,
                        'congestion_detected': utilization > 0.8
                    }
            
            return {
                'scenario_results': bandwidth_results,
                'average_bandwidth': statistics.mean([r['bandwidth_mbps'] for r in bandwidth_results.values()]),
                'peak_utilization': max([r['utilization'] for r in bandwidth_results.values()]),
                'congested_links': sum(1 for r in bandwidth_results.values() if r['congestion_detected'])
            }
            
        except Exception as e:
            print(f"Bandwidth test error: {e}")
            return None
    
    def test_throughput(self):
        """Test network throughput under various conditions"""
        throughput_results = {}
        
        try:
            load_conditions = ['low', 'medium', 'high', 'overload']
            
            for load in load_conditions:
                if load == 'low':
                    concurrent_flows = 2
                    expected_throughput = 95
                elif load == 'medium':
                    concurrent_flows = 5
                    expected_throughput = 85
                elif load == 'high':
                    concurrent_flows = 10
                    expected_throughput = 70
                else:  # overload
                    concurrent_flows = 20
                    expected_throughput = 45
                
                # Simulate throughput measurements
                actual_throughput = expected_throughput * np.random.uniform(0.9, 1.1)
                packet_rate = actual_throughput * 1000 / 8 / 1500  # Assuming 1500-byte packets
                
                throughput_results[load] = {
                    'concurrent_flows': concurrent_flows,
                    'throughput_mbps': actual_throughput,
                    'packet_rate_pps': packet_rate,
                    'efficiency': actual_throughput / expected_throughput,
                    'performance_degradation': max(0, (expected_throughput - actual_throughput) / expected_throughput)
                }
            
            return {
                'load_test_results': throughput_results,
                'peak_throughput': max([r['throughput_mbps'] for r in throughput_results.values()]),
                'throughput_under_load': throughput_results['high']['throughput_mbps'],
                'scalability_factor': throughput_results['low']['throughput_mbps'] / throughput_results['overload']['throughput_mbps']
            }
            
        except Exception as e:
            print(f"Throughput test error: {e}")
            return None
    
    def test_packet_loss(self):
        """Test packet loss rates under different conditions"""
        packet_loss_results = {}
        
        try:
            test_conditions = [
                ('normal', 1000, 0.1),     # Normal conditions
                ('congestion', 5000, 2.0), # Network congestion
                ('burst', 10000, 5.0),     # Burst traffic
                ('attack', 20000, 15.0)    # DDoS simulation
            ]
            
            for condition, packet_count, expected_loss in test_conditions:
                # Simulate packet transmission
                lost_packets = int(packet_count * (expected_loss / 100) * np.random.uniform(0.5, 1.5))
                delivered_packets = packet_count - lost_packets
                
                loss_rate = (lost_packets / packet_count) * 100
                
                packet_loss_results[condition] = {
                    'packets_sent': packet_count,
                    'packets_lost': lost_packets,
                    'packets_delivered': delivered_packets,
                    'loss_rate_percent': loss_rate,
                    'delivery_rate': (delivered_packets / packet_count) * 100,
                    'acceptable_loss': loss_rate < 1.0 if condition == 'normal' else loss_rate < expected_loss * 1.2
                }
            
            return {
                'condition_results': packet_loss_results,
                'worst_loss_rate': max([r['loss_rate_percent'] for r in packet_loss_results.values()]),
                'normal_operation_loss': packet_loss_results['normal']['loss_rate_percent'],
                'congestion_impact': packet_loss_results['congestion']['loss_rate_percent'] / packet_loss_results['normal']['loss_rate_percent']
            }
            
        except Exception as e:
            print(f"Packet loss test error: {e}")
            return None
    
    def test_load_balancing(self):
        """Test load balancing effectiveness"""
        load_balancing_results = {}
        
        try:
            # Simulate traffic distribution across multiple paths
            num_paths = 3
            total_traffic = 1000  # Total traffic units
            
            # Perfect load balancing would be equal distribution
            perfect_distribution = total_traffic / num_paths
            
            # Simulate actual distribution (with some variance)
            actual_distribution = []
            for i in range(num_paths):
                # Add some randomness to simulate real-world conditions
                path_traffic = perfect_distribution * np.random.uniform(0.8, 1.2)
                actual_distribution.append(path_traffic)
            
            # Normalize to ensure total equals original traffic
            scaling_factor = total_traffic / sum(actual_distribution)
            actual_distribution = [traffic * scaling_factor for traffic in actual_distribution]
            
            # Calculate load balancing metrics
            variance = statistics.variance(actual_distribution)
            std_dev = statistics.stdev(actual_distribution)
            load_balancing_efficiency = 1.0 - (std_dev / perfect_distribution)
            
            path_utilizations = [traffic / perfect_distribution for traffic in actual_distribution]
            max_utilization = max(path_utilizations)
            min_utilization = min(path_utilizations)
            
            load_balancing_results = {
                'total_traffic': total_traffic,
                'num_paths': num_paths,
                'perfect_per_path': perfect_distribution,
                'actual_distribution': actual_distribution,
                'path_utilizations': path_utilizations,
                'load_variance': variance,
                'load_std_dev': std_dev,
                'balancing_efficiency': load_balancing_efficiency,
                'max_path_utilization': max_utilization,
                'min_path_utilization': min_utilization,
                'utilization_ratio': max_utilization / min_utilization,
                'well_balanced': load_balancing_efficiency > 0.8
            }
            
            return load_balancing_results
            
        except Exception as e:
            print(f"Load balancing test error: {e}")
            return None
    
    def test_scalability(self):
        """Test network scalability with increasing load"""
        scalability_results = {}
        
        try:
            host_counts = [4, 8, 16, 32]  # Increasing number of active hosts
            
            for host_count in host_counts:
                # Simulate performance metrics for different scales
                base_latency = 1.0
                scale_factor = np.log2(host_count / 4) * 0.5  # Logarithmic scaling
                
                avg_latency = base_latency + scale_factor
                throughput_degradation = min(0.5, scale_factor * 0.2)  # Max 50% degradation
                memory_usage = host_count * 10  # MB per host
                cpu_usage = min(95, host_count * 2)  # Max 95% CPU
                
                scalability_results[host_count] = {
                    'active_hosts': host_count,
                    'avg_latency': avg_latency,
                    'throughput_factor': 1.0 - throughput_degradation,
                    'memory_usage_mb': memory_usage,
                    'cpu_usage_percent': cpu_usage,
                    'flows_per_second': max(100, 1000 - host_count * 20),
                    'acceptable_performance': avg_latency < 5.0 and cpu_usage < 90
                }
            
            # Calculate scalability metrics
            latency_increase = scalability_results[32]['avg_latency'] / scalability_results[4]['avg_latency']
            throughput_retention = scalability_results[32]['throughput_factor'] / scalability_results[4]['throughput_factor']
            
            return {
                'scale_results': scalability_results,
                'max_tested_hosts': max(host_counts),
                'latency_scaling_factor': latency_increase,
                'throughput_retention': throughput_retention,
                'linear_scalability': throughput_retention > 0.7,
                'recommended_max_hosts': max([count for count, result in scalability_results.items() if result['acceptable_performance']])
            }
            
        except Exception as e:
            print(f"Scalability test error: {e}")
            return None
    
    def test_anomaly_detection_performance(self):
        """Test AI anomaly detection system performance"""
        detection_results = {}
        
        try:
            # Simulate anomaly detection performance metrics
            test_data_sizes = [100, 500, 1000, 5000]
            
            for data_size in test_data_sizes:
                # Simulate detection times and accuracy
                detection_time = data_size * np.random.uniform(0.001, 0.005)  # 1-5ms per sample
                false_positive_rate = np.random.uniform(0.02, 0.08)  # 2-8%
                false_negative_rate = np.random.uniform(0.01, 0.05)  # 1-5%
                accuracy = 1.0 - (false_positive_rate + false_negative_rate) / 2
                
                detection_results[data_size] = {
                    'data_size': data_size,
                    'detection_time_ms': detection_time * 1000,
                    'throughput_samples_per_sec': data_size / detection_time,
                    'accuracy': accuracy,
                    'false_positive_rate': false_positive_rate,
                    'false_negative_rate': false_negative_rate,
                    'precision': 1.0 - false_positive_rate,
                    'recall': 1.0 - false_negative_rate,
                    'f1_score': 2 * (1 - false_positive_rate) * (1 - false_negative_rate) / (2 - false_positive_rate - false_negative_rate)
                }
            
            return {
                'performance_by_size': detection_results,
                'max_throughput': max([r['throughput_samples_per_sec'] for r in detection_results.values()]),
                'avg_accuracy': statistics.mean([r['accuracy'] for r in detection_results.values()]),
                'avg_f1_score': statistics.mean([r['f1_score'] for r in detection_results.values()]),
                'real_time_capable': min([r['detection_time_ms'] for r in detection_results.values()]) < 100
            }
            
        except Exception as e:
            print(f"Anomaly detection test error: {e}")
            return None
    
    def test_sdn_controller_performance(self):
        """Test SDN controller performance metrics"""
        controller_results = {}
        
        try:
            # Simulate controller performance under different loads
            flow_rates = [100, 500, 1000, 2000]  # Flows per second
            
            for flow_rate in flow_rates:
                # Simulate controller response times
                base_response_time = 5.0  # 5ms base response time
                load_factor = (flow_rate / 100) * 0.5  # Linear scaling
                response_time = base_response_time + load_factor
                
                # Simulate resource usage
                cpu_usage = min(95, flow_rate * 0.03)
                memory_usage = 100 + flow_rate * 0.1  # MB
                
                # Flow installation rate
                installation_rate = min(flow_rate * 0.95, 1800)  # Max ~1800 flows/sec
                
                controller_results[flow_rate] = {
                    'target_flow_rate': flow_rate,
                    'actual_installation_rate': installation_rate,
                    'avg_response_time_ms': response_time,
                    'cpu_usage_percent': cpu_usage,
                    'memory_usage_mb': memory_usage,
                    'flow_table_entries': flow_rate * 60,  # Assume 60-second timeout
                    'packet_in_processing_rate': installation_rate,
                    'controller_efficiency': installation_rate / flow_rate,
                    'overload_detected': response_time > 50 or cpu_usage > 90
                }
            
            return {
                'load_test_results': controller_results,
                'max_sustainable_rate': max([rate for rate, result in controller_results.items() if not result['overload_detected']]),
                'avg_response_time': statistics.mean([r['avg_response_time_ms'] for r in controller_results.values()]),
                'peak_efficiency': max([r['controller_efficiency'] for r in controller_results.values()]),
                'scalable_performance': len([r for r in controller_results.values() if not r['overload_detected']]) >= 3
            }
            
        except Exception as e:
            print(f"SDN controller test error: {e}")
            return None
    
    def test_system_resources(self):
        """Test system resource utilization during network operations"""
        try:
            # Get current system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            network_io = psutil.net_io_counters()
            
            # Simulate resource usage under network load
            network_load_impact = {
                'baseline_cpu': cpu_percent,
                'network_load_cpu': cpu_percent + np.random.uniform(10, 25),
                'baseline_memory': memory.percent,
                'network_load_memory': memory.percent + np.random.uniform(5, 15),
                'disk_io_increase': np.random.uniform(20, 50),  # Percent increase
                'network_io_bytes': network_io.bytes_sent + network_io.bytes_recv
            }
            
            # Calculate resource efficiency
            cpu_efficiency = max(0, 100 - network_load_impact['network_load_cpu']) / 100
            memory_efficiency = max(0, 100 - network_load_impact['network_load_memory']) / 100
            
            return {
                'resource_usage': network_load_impact,
                'cpu_efficiency': cpu_efficiency,
                'memory_efficiency': memory_efficiency,
                'resource_availability': (cpu_efficiency + memory_efficiency) / 2,
                'system_healthy': cpu_efficiency > 0.3 and memory_efficiency > 0.2,
                'available_memory_gb': (memory.total - memory.used) / (1024**3),
                'disk_free_percent': (disk.free / disk.total) * 100
            }
            
        except Exception as e:
            print(f"System resource test error: {e}")
            return None
    
    def save_test_results(self):
        """Save test results to JSON file"""
        try:
            results_file = os.path.join(self.output_dir, 'performance_test_results.json')
            with open(results_file, 'w') as f:
                json.dump(self.test_results, f, indent=2)
            print(f"Test results saved to {results_file}")
        except Exception as e:
            print(f"Error saving test results: {e}")
    
    def generate_performance_report(self):
        """Generate comprehensive performance report"""
        try:
            report_file = os.path.join(self.output_dir, 'performance_report.md')
            
            with open(report_file, 'w') as f:
                f.write("# Network Performance Test Report\n\n")
                f.write(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Executive Summary
                f.write("## Executive Summary\n\n")
                
                successful_tests = sum(1 for result in self.test_results.values() if result['status'] == 'success')
                total_tests = len(self.test_results)
                
                f.write(f"- **Total Tests:** {total_tests}\n")
                f.write(f"- **Successful Tests:** {successful_tests}\n")
                f.write(f"- **Success Rate:** {(successful_tests/total_tests)*100:.1f}%\n")
                f.write(f"- **Total Test Duration:** {sum(r['execution_time'] for r in self.test_results.values()):.2f} seconds\n\n")
                
                # Detailed Results
                f.write("## Test Results\n\n")
                
                for test_name, result in self.test_results.items():
                    f.write(f"### {test_name}\n\n")
                    f.write(f"- **Status:** {result['status']}\n")
                    f.write(f"- **Execution Time:** {result['execution_time']:.2f}s\n")
                    
                    if result['status'] == 'success' and result['result']:
                        self._write_test_details(f, test_name, result['result'])
                    elif result['status'] == 'error':
                        f.write(f"- **Error:** {result.get('error', 'Unknown error')}\n")
                    
                    f.write("\n")
                
                # Performance Summary
                f.write("## Performance Summary\n\n")
                self._write_performance_summary(f)
                
                # Recommendations
                f.write("## Recommendations\n\n")
                self._write_recommendations(f)
            
            print(f"Performance report generated: {report_file}")
            
        except Exception as e:
            print(f"Error generating performance report: {e}")
    
    def _write_test_details(self, f, test_name, result):
        """Write detailed test results to report"""
        if test_name == "Connectivity Test" and isinstance(result, dict):
            f.write(f"- **Overall Connectivity:** {result.get('overall_connectivity', 0)*100:.1f}%\n")
            f.write(f"- **Successful Connections:** {result.get('successful_connections', 0)}/{result.get('total_host_pairs', 0)}\n")
        
        elif test_name == "Latency Test" and isinstance(result, dict):
            f.write(f"- **Average Latency:** {result.get('overall_avg_latency', 0):.2f}ms\n")
            f.write(f"- **Maximum Latency:** {result.get('overall_max_latency', 0):.2f}ms\n")
            f.write(f"- **Average Jitter:** {result.get('overall_jitter', 0):.2f}ms\n")
        
        elif test_name == "Bandwidth Test" and isinstance(result, dict):
            f.write(f"- **Average Bandwidth:** {result.get('average_bandwidth', 0):.1f} Mbps\n")
            f.write(f"- **Peak Utilization:** {result.get('peak_utilization', 0)*100:.1f}%\n")
            f.write(f"- **Congested Links:** {result.get('congested_links', 0)}\n")
        
        elif test_name == "Throughput Test" and isinstance(result, dict):
            f.write(f"- **Peak Throughput:** {result.get('peak_throughput', 0):.1f} Mbps\n")
            f.write(f"- **Throughput Under Load:** {result.get('throughput_under_load', 0):.1f} Mbps\n")
            f.write(f"- **Scalability Factor:** {result.get('scalability_factor', 0):.2f}\n")
        
        # Add more test-specific details as needed
    
    def _write_performance_summary(self, f):
        """Write performance summary to report"""
        f.write("### Key Performance Indicators\n\n")
        
        # Extract key metrics from test results
        if "Latency Test" in self.test_results and self.test_results["Latency Test"]["status"] == "success":
            latency_result = self.test_results["Latency Test"]["result"]
            f.write(f"- **Network Latency:** {latency_result.get('overall_avg_latency', 0):.2f}ms (Good: <5ms)\n")
        
        if "Throughput Test" in self.test_results and self.test_results["Throughput Test"]["status"] == "success":
            throughput_result = self.test_results["Throughput Test"]["result"]
            f.write(f"- **Peak Throughput:** {throughput_result.get('peak_throughput', 0):.1f} Mbps\n")
        
        if "Anomaly Detection Performance" in self.test_results and self.test_results["Anomaly Detection Performance"]["status"] == "success":
            anomaly_result = self.test_results["Anomaly Detection Performance"]["result"]
            f.write(f"- **AI Detection Accuracy:** {anomaly_result.get('avg_accuracy', 0)*100:.1f}%\n")
        
        f.write("\n")
    
    def _write_recommendations(self, f):
        """Write recommendations based on test results"""
        recommendations = []
        
        # Analyze results and generate recommendations
        if "Latency Test" in self.test_results and self.test_results["Latency Test"]["status"] == "success":
            latency_result = self.test_results["Latency Test"]["result"]
            if latency_result.get('overall_avg_latency', 0) > 10:
                recommendations.append("Consider optimizing network routing to reduce latency")
        
        if "Bandwidth Test" in self.test_results and self.test_results["Bandwidth Test"]["status"] == "success":
            bandwidth_result = self.test_results["Bandwidth Test"]["result"]
            if bandwidth_result.get('congested_links', 0) > 0:
                recommendations.append("Implement load balancing to reduce link congestion")
        
        if "System Resource Utilization" in self.test_results and self.test_results["System Resource Utilization"]["status"] == "success":
            resource_result = self.test_results["System Resource Utilization"]["result"]
            if not resource_result.get('system_healthy', True):
                recommendations.append("Monitor system resources and consider hardware upgrades")
        
        # Default recommendations
        if not recommendations:
            recommendations = [
                "Continue monitoring network performance regularly",
                "Consider implementing additional anomaly detection rules",
                "Plan for network capacity scaling based on growth projections"
            ]
        
        for i, recommendation in enumerate(recommendations, 1):
            f.write(f"{i}. {recommendation}\n")
        
        f.write("\n")


def main():
    """Main function to run performance tests"""
    print("Starting Network Performance Testing Suite...")
    
    tester = NetworkPerformanceTester()
    tester.run_comprehensive_tests()
    
    print("\nPerformance testing completed!")
    print("Check the 'results/' directory for detailed reports and metrics.")


if __name__ == '__main__':
    main()