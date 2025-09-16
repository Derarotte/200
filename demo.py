#!/usr/bin/env python3
"""
Comprehensive Demo Script for Network Intelligence System
Automates the complete demonstration of all system components
"""

import os
import sys
import time
import json
import threading
import subprocess
import signal
from datetime import datetime
import argparse


class NetworkIntelligenceDemo:
    """Complete demo orchestration for the network intelligence system"""
    
    def __init__(self):
        self.processes = []  # Keep track of running processes
        self.demo_running = True
        self.components_status = {
            'topology': 'stopped',
            'controller': 'stopped', 
            'traffic_generator': 'stopped',
            'ai_detection': 'stopped',
            'dashboard': 'stopped',
            'performance_tests': 'stopped'
        }
        
        # Setup signal handlers for clean shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print("\n🛑 Shutdown signal received. Cleaning up...")
        self.cleanup_processes()
        sys.exit(0)
    
    def print_banner(self):
        """Print demo banner"""
        print("=" * 80)
        print("🌐 NETWORK INTELLIGENCE SYSTEM - COMPREHENSIVE DEMO")
        print("=" * 80)
        print("This demo showcases:")
        print("• Multi-tier SDN network topology with Mininet")
        print("• AI-powered anomaly detection system")  
        print("• Intelligent SDN controller with Ryu")
        print("• Real-time monitoring dashboard")
        print("• Automated performance testing")
        print("• Network security and threat detection")
        print("=" * 80)
        print()
    
    def check_prerequisites(self):
        """Check if required software is installed"""
        print("🔍 Checking prerequisites...")
        
        requirements = [
            ('python3', 'Python 3'),
            ('mn', 'Mininet'),
            ('ryu-manager', 'Ryu SDN Controller'),
            ('ovs-vsctl', 'Open vSwitch')
        ]
        
        missing_requirements = []
        
        for cmd, name in requirements:
            try:
                result = subprocess.run(['which', cmd], capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"  ✅ {name}: {result.stdout.strip()}")
                else:
                    print(f"  ❌ {name}: Not found")
                    missing_requirements.append(name)
            except Exception as e:
                print(f"  ❌ {name}: Error checking - {e}")
                missing_requirements.append(name)
        
        if missing_requirements:
            print(f"\n❌ Missing requirements: {', '.join(missing_requirements)}")
            print("Please install the missing components before running the demo.")
            return False
        
        print("✅ All prerequisites satisfied!\n")
        return True
    
    def setup_directories(self):
        """Create necessary directories"""
        print("📁 Setting up directories...")
        
        directories = ['data', 'logs', 'models', 'results', 'templates']
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"  📂 Created/verified: {directory}/")
        
        print()
    
    def start_network_topology(self):
        """Start the Mininet network topology"""
        print("🌐 Starting network topology...")
        
        try:
            # Check if Mininet is already running
            result = subprocess.run(['sudo', 'mn', '-c'], capture_output=True, text=True)
            
            # Start the custom topology
            cmd = ['sudo', 'python3', 'src/mininet_topology/custom_topology.py']
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            self.processes.append(('topology', process))
            self.components_status['topology'] = 'starting'
            
            # Give it time to start
            time.sleep(5)
            
            # Check if process is still running
            if process.poll() is None:
                self.components_status['topology'] = 'running'
                print("  ✅ Network topology started successfully")
            else:
                self.components_status['topology'] = 'failed'
                print("  ❌ Failed to start network topology")
                return False
            
        except Exception as e:
            print(f"  ❌ Error starting topology: {e}")
            self.components_status['topology'] = 'failed'
            return False
        
        return True
    
    def start_sdn_controller(self):
        """Start the SDN controller"""
        print("🎛️  Starting SDN controller...")
        
        try:
            cmd = ['ryu-manager', 'src/sdn_controller/intelligent_controller.py']
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            self.processes.append(('controller', process))
            self.components_status['controller'] = 'starting'
            
            # Give controller time to start
            time.sleep(3)
            
            if process.poll() is None:
                self.components_status['controller'] = 'running'
                print("  ✅ SDN controller started successfully")
            else:
                self.components_status['controller'] = 'failed'
                print("  ❌ Failed to start SDN controller")
                return False
                
        except Exception as e:
            print(f"  ❌ Error starting controller: {e}")
            self.components_status['controller'] = 'failed'
            return False
        
        return True
    
    def start_traffic_generator(self):
        """Start traffic generation"""
        print("🚦 Starting traffic generator...")
        
        try:
            cmd = ['python3', 'src/mininet_topology/traffic_generator.py']
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            self.processes.append(('traffic_generator', process))
            self.components_status['traffic_generator'] = 'running'
            print("  ✅ Traffic generator started")
            
        except Exception as e:
            print(f"  ❌ Error starting traffic generator: {e}")
            self.components_status['traffic_generator'] = 'failed'
            return False
        
        return True
    
    def train_ai_models(self):
        """Train AI anomaly detection models"""
        print("🤖 Training AI anomaly detection models...")
        
        try:
            # Wait a bit for traffic data to be generated
            print("  ⏳ Waiting for traffic data generation...")
            time.sleep(30)
            
            cmd = ['python3', 'src/ai_detection/anomaly_detector.py']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                self.components_status['ai_detection'] = 'completed'
                print("  ✅ AI models trained successfully")
            else:
                print(f"  ❌ AI training failed: {result.stderr}")
                self.components_status['ai_detection'] = 'failed'
                return False
                
        except subprocess.TimeoutExpired:
            print("  ⚠️  AI training timeout - continuing with demo")
            self.components_status['ai_detection'] = 'timeout'
        except Exception as e:
            print(f"  ❌ Error training AI models: {e}")
            self.components_status['ai_detection'] = 'failed'
            return False
        
        return True
    
    def start_monitoring_dashboard(self):
        """Start the monitoring dashboard"""
        print("📊 Starting monitoring dashboard...")
        
        try:
            cmd = ['python3', 'src/monitoring/dashboard.py']
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            self.processes.append(('dashboard', process))
            self.components_status['dashboard'] = 'starting'
            
            # Give dashboard time to start
            time.sleep(5)
            
            if process.poll() is None:
                self.components_status['dashboard'] = 'running'
                print("  ✅ Monitoring dashboard started")
                print("  🌐 Dashboard available at: http://localhost:5000")
            else:
                self.components_status['dashboard'] = 'failed'
                print("  ❌ Failed to start monitoring dashboard")
                return False
                
        except Exception as e:
            print(f"  ❌ Error starting dashboard: {e}")
            self.components_status['dashboard'] = 'failed'
            return False
        
        return True
    
    def run_performance_tests(self):
        """Run performance tests"""
        print("⚡ Running performance tests...")
        
        try:
            cmd = ['python3', 'src/utils/performance_tester.py']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                self.components_status['performance_tests'] = 'completed'
                print("  ✅ Performance tests completed")
                print("  📊 Results saved to results/ directory")
            else:
                print(f"  ❌ Performance tests failed: {result.stderr}")
                self.components_status['performance_tests'] = 'failed'
                return False
                
        except subprocess.TimeoutExpired:
            print("  ⚠️  Performance tests timeout - check results/ directory")
            self.components_status['performance_tests'] = 'timeout'
        except Exception as e:
            print(f"  ❌ Error running performance tests: {e}")
            self.components_status['performance_tests'] = 'failed'
            return False
        
        return True
    
    def show_system_status(self):
        """Display current system status"""
        print("\n" + "="*60)
        print("📋 SYSTEM STATUS")
        print("="*60)
        
        status_icons = {
            'stopped': '⚪',
            'starting': '🟡',
            'running': '🟢',
            'completed': '✅',
            'failed': '❌',
            'timeout': '⚠️'
        }
        
        components = {
            'topology': 'Network Topology',
            'controller': 'SDN Controller',
            'traffic_generator': 'Traffic Generator',
            'ai_detection': 'AI Detection Models',
            'dashboard': 'Monitoring Dashboard',
            'performance_tests': 'Performance Tests'
        }
        
        for component, name in components.items():
            status = self.components_status[component]
            icon = status_icons.get(status, '❓')
            print(f"{icon} {name:<25} {status.upper()}")
        
        print("="*60)
    
    def show_demo_results(self):
        """Show demo results and access information"""
        print("\n" + "="*60)
        print("🎉 DEMO RESULTS & ACCESS INFORMATION")
        print("="*60)
        
        print("\n📊 DASHBOARD ACCESS:")
        if self.components_status['dashboard'] == 'running':
            print("  🌐 Web Dashboard: http://localhost:5000")
            print("  📈 Real-time metrics, anomaly alerts, topology view")
        else:
            print("  ❌ Dashboard not running")
        
        print("\n📁 GENERATED DATA:")
        print("  📋 Traffic logs: data/traffic_log.json")
        print("  📊 Network stats: data/network_stats.json")
        print("  🔄 Flow stats: data/flow_stats.json")
        
        print("\n🤖 AI MODELS:")
        if os.path.exists('models'):
            model_files = [f for f in os.listdir('models') if f.endswith('.pkl') or f.endswith('.h5')]
            if model_files:
                print("  ✅ Trained models:")
                for model in model_files:
                    print(f"    - {model}")
            else:
                print("  ⚠️  No trained models found")
        else:
            print("  ❌ Models directory not found")
        
        print("\n📈 PERFORMANCE RESULTS:")
        if os.path.exists('results'):
            result_files = [f for f in os.listdir('results') if f.endswith('.json') or f.endswith('.md')]
            if result_files:
                print("  ✅ Test results:")
                for result in result_files:
                    print(f"    - results/{result}")
            else:
                print("  ⚠️  No test results found")
        else:
            print("  ❌ Results directory not found")
        
        print("\n🔧 SYSTEM COMMANDS:")
        print("  🛑 Stop demo: Ctrl+C")
        print("  📊 View logs: tail -f logs/*.log")
        print("  🔍 Check processes: ps aux | grep python")
        
        print("="*60)
    
    def cleanup_processes(self):
        """Clean up all running processes"""
        print("\n🧹 Cleaning up processes...")
        
        for name, process in self.processes:
            try:
                if process.poll() is None:  # Process is still running
                    print(f"  🛑 Stopping {name}...")
                    process.terminate()
                    
                    # Wait for graceful termination
                    try:
                        process.wait(timeout=5)
                        print(f"    ✅ {name} stopped gracefully")
                    except subprocess.TimeoutExpired:
                        print(f"    ⚠️  Force killing {name}...")
                        process.kill()
                        process.wait()
                else:
                    print(f"  ✅ {name} already stopped")
                    
            except Exception as e:
                print(f"  ❌ Error stopping {name}: {e}")
        
        # Clean up Mininet
        try:
            print("  🧹 Cleaning up Mininet...")
            subprocess.run(['sudo', 'mn', '-c'], capture_output=True, text=True)
            print("    ✅ Mininet cleaned up")
        except Exception as e:
            print(f"    ❌ Error cleaning Mininet: {e}")
        
        print("✅ Cleanup completed")
    
    def run_interactive_demo(self):
        """Run interactive demo with user prompts"""
        self.print_banner()
        
        if not self.check_prerequisites():
            return False
        
        self.setup_directories()
        
        print("🎬 Starting interactive demo...")
        print("Press Enter to continue at each step, or Ctrl+C to exit\n")
        
        # Step 1: Network Topology
        input("📍 Step 1: Start network topology (Press Enter)")
        if not self.start_network_topology():
            return False
        
        # Step 2: SDN Controller  
        input("\n📍 Step 2: Start SDN controller (Press Enter)")
        if not self.start_sdn_controller():
            return False
        
        # Step 3: Traffic Generation
        input("\n📍 Step 3: Start traffic generation (Press Enter)")
        if not self.start_traffic_generator():
            return False
        
        # Step 4: AI Training
        input("\n📍 Step 4: Train AI models (Press Enter)")
        self.train_ai_models()
        
        # Step 5: Dashboard
        input("\n📍 Step 5: Start monitoring dashboard (Press Enter)")
        if not self.start_monitoring_dashboard():
            return False
        
        # Step 6: Performance Tests
        input("\n📍 Step 6: Run performance tests (Press Enter)")
        self.run_performance_tests()
        
        # Show final status
        self.show_system_status()
        self.show_demo_results()
        
        print("\n🎉 Demo setup complete!")
        print("The system is now running. Press Ctrl+C to stop all components.")
        
        # Keep running until interrupted
        try:
            while self.demo_running:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        
        return True
    
    def run_automated_demo(self, duration=300):
        """Run automated demo without user interaction"""
        self.print_banner()
        
        if not self.check_prerequisites():
            return False
        
        self.setup_directories()
        
        print(f"🤖 Running automated demo for {duration} seconds...")
        
        # Start all components
        steps = [
            ("Network Topology", self.start_network_topology),
            ("SDN Controller", self.start_sdn_controller),
            ("Traffic Generator", self.start_traffic_generator),
            ("AI Training", self.train_ai_models),
            ("Monitoring Dashboard", self.start_monitoring_dashboard),
            ("Performance Tests", self.run_performance_tests)
        ]
        
        for step_name, step_function in steps:
            print(f"\n📍 {step_name}...")
            step_function()
            time.sleep(2)
        
        # Show status and run for specified duration
        self.show_system_status()
        self.show_demo_results()
        
        print(f"\n⏰ Demo running for {duration} seconds...")
        
        start_time = time.time()
        while time.time() - start_time < duration and self.demo_running:
            time.sleep(10)
            print(f"  ⏱️  Demo time remaining: {int(duration - (time.time() - start_time))} seconds")
        
        print("\n🏁 Automated demo completed!")
        return True


def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(description='Network Intelligence System Demo')
    parser.add_argument('--mode', choices=['interactive', 'automated'], default='interactive',
                       help='Demo mode: interactive (default) or automated')
    parser.add_argument('--duration', type=int, default=300,
                       help='Duration for automated demo in seconds (default: 300)')
    
    args = parser.parse_args()
    
    demo = NetworkIntelligenceDemo()
    
    try:
        if args.mode == 'interactive':
            success = demo.run_interactive_demo()
        else:
            success = demo.run_automated_demo(args.duration)
        
        if success:
            print("\n✅ Demo completed successfully!")
        else:
            print("\n❌ Demo encountered errors!")
            return 1
            
    except KeyboardInterrupt:
        print("\n\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        return 1
    finally:
        demo.cleanup_processes()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())