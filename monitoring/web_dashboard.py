#!/usr/bin/env python3
"""
Web监控仪表盘
实时展示SDN网络状态、流量统计、异常检测结果
使用Flask + Chart.js实现交互式Web界面
"""

from flask import Flask, render_template_string, jsonify, request
import json
import time
import threading
from collections import defaultdict, deque
import plotly.graph_objs as go
import plotly.utils

# 导入其他模块
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class NetworkMonitor:
    """网络监控数据收集器"""

    def __init__(self):
        self.network_stats = deque(maxlen=1000)
        self.anomaly_events = deque(maxlen=500)
        self.traffic_flows = defaultdict(lambda: deque(maxlen=100))
        self.performance_metrics = {
            'latency': deque(maxlen=100),
            'throughput': deque(maxlen=100),
            'packet_loss': deque(maxlen=100)
        }

        # 模拟数据生成
        self.simulation_active = False
        self.simulation_thread = None

    def start_monitoring(self):
        """启动监控数据收集"""
        print("开始网络监控...")
        self.simulation_active = True
        self.simulation_thread = threading.Thread(target=self._simulate_data)
        self.simulation_thread.daemon = True
        self.simulation_thread.start()

    def stop_monitoring(self):
        """停止监控"""
        self.simulation_active = False
        if self.simulation_thread:
            self.simulation_thread.join()

    def add_network_stats(self, stats):
        """添加网络统计数据"""
        stats['timestamp'] = time.time()
        self.network_stats.append(stats)

    def add_anomaly_event(self, event):
        """添加异常事件"""
        event['timestamp'] = time.time()
        self.anomaly_events.append(event)

    def _simulate_data(self):
        """模拟实时数据生成"""
        import random

        while self.simulation_active:
            try:
                current_time = time.time()

                # 模拟网络统计数据
                base_flows = 50 + len(self.network_stats) % 100
                base_packets = base_flows * random.randint(10, 50)
                base_bytes = base_packets * random.randint(500, 1500)

                # 添加一些随机波动
                if random.random() < 0.1:  # 10%概率产生流量突增
                    base_packets *= random.uniform(2, 5)
                    base_bytes *= random.uniform(2, 5)

                network_stat = {
                    'timestamp': current_time,
                    'topology': {
                        'switches': 4,
                        'links': 6,
                        'hosts': 6
                    },
                    'traffic': {
                        'total_flows': base_flows,
                        'total_packets': base_packets,
                        'total_bytes': base_bytes,
                        'blocked_flows': random.randint(0, 5)
                    },
                    'switches': {
                        's1': {'packets_processed': base_packets // 4, 'flows_installed': base_flows // 4},
                        's2': {'packets_processed': base_packets // 4, 'flows_installed': base_flows // 4},
                        's3': {'packets_processed': base_packets // 4, 'flows_installed': base_flows // 4},
                        's4': {'packets_processed': base_packets // 4, 'flows_installed': base_flows // 4}
                    }
                }

                self.add_network_stats(network_stat)

                # 模拟性能指标
                self.performance_metrics['latency'].append({
                    'timestamp': current_time,
                    'value': random.uniform(1, 10) + (5 if base_packets > 5000 else 0)
                })

                self.performance_metrics['throughput'].append({
                    'timestamp': current_time,
                    'value': base_bytes / 1024 / 1024  # MB/s
                })

                self.performance_metrics['packet_loss'].append({
                    'timestamp': current_time,
                    'value': random.uniform(0, 0.1) + (0.2 if base_packets > 8000 else 0)
                })

                # 模拟异常事件
                if random.random() < 0.05:  # 5%概率产生异常
                    anomaly_types = ['ddos_attack', 'port_scan', 'large_transfer', 'statistical_anomaly']
                    anomaly_type = random.choice(anomaly_types)

                    anomaly_event = {
                        'timestamp': current_time,
                        'type': anomaly_type,
                        'confidence': random.uniform(0.6, 0.95),
                        'source_ip': f"10.0.0.{random.randint(1, 6)}",
                        'target_ip': f"10.0.0.{random.randint(1, 6)}",
                        'details': f"Detected {anomaly_type} with high confidence",
                        'actions_taken': ['Traffic monitoring increased', 'Security alert sent']
                    }

                    self.add_anomaly_event(anomaly_event)

                time.sleep(2)  # 每2秒更新一次

            except Exception as e:
                print(f"Simulation error: {e}")
                time.sleep(5)

    def get_current_stats(self):
        """获取当前网络统计"""
        if not self.network_stats:
            return {}

        latest = self.network_stats[-1]
        recent_anomalies = [a for a in self.anomaly_events if time.time() - a['timestamp'] < 300]

        return {
            'network': latest,
            'anomalies': {
                'recent_count': len(recent_anomalies),
                'total_count': len(self.anomaly_events),
                'latest': list(self.anomaly_events)[-5:] if self.anomaly_events else []
            },
            'performance': {
                'current_latency': self.performance_metrics['latency'][-1]['value'] if self.performance_metrics['latency'] else 0,
                'current_throughput': self.performance_metrics['throughput'][-1]['value'] if self.performance_metrics['throughput'] else 0,
                'current_packet_loss': self.performance_metrics['packet_loss'][-1]['value'] if self.performance_metrics['packet_loss'] else 0
            },
            'uptime_minutes': len(self.network_stats) * 2 / 60  # 2秒间隔
        }

    def generate_traffic_plot(self):
        """生成流量图表数据"""
        if len(self.network_stats) < 2:
            return {}

        recent_stats = list(self.network_stats)[-50:]  # 最近50个数据点
        times = [(s['timestamp'] - self.network_stats[0]['timestamp']) / 60 for s in recent_stats]
        packets = [s['traffic']['total_packets'] for s in recent_stats]
        bytes_data = [s['traffic']['total_bytes'] / 1024 / 1024 for s in recent_stats]  # MB

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=times,
            y=packets,
            mode='lines+markers',
            name='Packets/sec',
            line=dict(color='blue', width=2),
            yaxis='y'
        ))

        fig.add_trace(go.Scatter(
            x=times,
            y=bytes_data,
            mode='lines+markers',
            name='Traffic (MB)',
            line=dict(color='red', width=2),
            yaxis='y2'
        ))

        fig.update_layout(
            title='Network Traffic Over Time',
            xaxis=dict(title='Time (minutes)'),
            yaxis=dict(title='Packets', side='left'),
            yaxis2=dict(title='Traffic (MB)', side='right', overlaying='y'),
            hovermode='x unified',
            showlegend=True
        )

        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    def generate_anomaly_plot(self):
        """生成异常检测图表数据"""
        if len(self.anomaly_events) < 2:
            return {}

        # 按时间统计异常数量
        anomaly_counts = defaultdict(int)
        for event in self.anomaly_events:
            time_bucket = int(event['timestamp'] / 60) * 60  # 按分钟分组
            anomaly_counts[time_bucket] += 1

        if not anomaly_counts:
            return {}

        times = sorted(anomaly_counts.keys())
        start_time = self.network_stats[0]['timestamp'] if self.network_stats else times[0]
        relative_times = [(t - start_time) / 60 for t in times]
        counts = [anomaly_counts[t] for t in times]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=relative_times,
            y=counts,
            name='Anomalies per minute',
            marker=dict(color='orange'),
            opacity=0.7
        ))

        fig.update_layout(
            title='Anomaly Detection Timeline',
            xaxis=dict(title='Time (minutes)'),
            yaxis=dict(title='Anomaly Count'),
            hovermode='x'
        )

        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    def generate_performance_plot(self):
        """生成性能指标图表"""
        if not any(self.performance_metrics.values()):
            return {}

        fig = go.Figure()

        for metric_name, metric_data in self.performance_metrics.items():
            if metric_data:
                recent_data = list(metric_data)[-30:]  # 最近30个点
                start_time = self.network_stats[0]['timestamp'] if self.network_stats else recent_data[0]['timestamp']
                times = [(d['timestamp'] - start_time) / 60 for d in recent_data]
                values = [d['value'] for d in recent_data]

                fig.add_trace(go.Scatter(
                    x=times,
                    y=values,
                    mode='lines+markers',
                    name=metric_name.title(),
                    line=dict(width=2)
                ))

        fig.update_layout(
            title='Network Performance Metrics',
            xaxis=dict(title='Time (minutes)'),
            yaxis=dict(title='Value'),
            hovermode='x unified'
        )

        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


# Flask应用
app = Flask(__name__)
monitor = NetworkMonitor()

# HTML模板
DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>SDN + AI Network Monitoring Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f8f9fa;
            color: #333;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .header h1 {
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }
        .header p {
            margin: 10px 0 0 0;
            opacity: 0.9;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .stat-card {
            background: white;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.08);
            text-align: center;
            transition: transform 0.2s ease;
        }
        .stat-card:hover {
            transform: translateY(-2px);
        }
        .stat-value {
            font-size: 2.5em;
            font-weight: 700;
            color: #667eea;
            margin-bottom: 5px;
        }
        .stat-label {
            color: #666;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .chart-container {
            background: white;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.08);
            margin-bottom: 25px;
        }
        .alert {
            background: #ff6b6b;
            color: white;
            padding: 15px 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            text-align: center;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.7; }
            100% { opacity: 1; }
        }
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .status-online { background-color: #51cf66; }
        .status-warning { background-color: #ffd43b; }
        .status-offline { background-color: #ff6b6b; }
        .control-panel {
            background: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.08);
            margin-bottom: 25px;
            text-align: center;
        }
        .btn {
            background: #667eea;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 6px;
            cursor: pointer;
            margin: 0 10px;
            font-size: 14px;
            transition: background 0.2s ease;
        }
        .btn:hover {
            background: #5a67d8;
        }
        .btn-danger {
            background: #ff6b6b;
        }
        .btn-danger:hover {
            background: #ff5252;
        }
        .anomaly-list {
            max-height: 300px;
            overflow-y: auto;
            background: #f8f9fa;
            border-radius: 8px;
            padding: 15px;
        }
        .anomaly-item {
            background: white;
            padding: 10px 15px;
            margin: 5px 0;
            border-radius: 6px;
            border-left: 4px solid #ff6b6b;
            font-size: 0.9em;
        }
        .timestamp {
            color: #666;
            font-size: 0.8em;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>SDN + AI Network Monitor</h1>
        <p>Real-time Network Traffic Analysis & Anomaly Detection</p>
    </div>

    <div class="container">
        <div id="anomaly-alert" class="alert" style="display: none;">
            <strong>网络异常检测!</strong> <span id="alert-details"></span>
        </div>

        <div class="control-panel">
            <button class="btn" onclick="refreshData()">Refresh Data</button>
            <button class="btn" onclick="toggleMonitoring()" id="monitor-btn">⏸️ Pause Monitoring</button>
            <button class="btn btn-danger" onclick="clearAlerts()">🧹 Clear Alerts</button>
            <span style="margin-left: 20px;">
                <span class="status-indicator" id="system-status"></span>
                <span id="status-text">System Status</span>
            </span>
        </div>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value" id="switches-count">-</div>
                <div class="stat-label">Active Switches</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="flows-count">-</div>
                <div class="stat-label">Network Flows</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="packets-count">-</div>
                <div class="stat-label">Packets/sec</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="anomalies-count">-</div>
                <div class="stat-label">Anomalies Detected</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="throughput">-</div>
                <div class="stat-label">Throughput (MB/s)</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="latency">-</div>
                <div class="stat-label">Latency (ms)</div>
            </div>
        </div>

        <div class="chart-container">
            <div id="traffic-plot"></div>
        </div>

        <div class="chart-container">
            <div id="anomaly-plot"></div>
        </div>

        <div class="chart-container">
            <div id="performance-plot"></div>
        </div>

        <div class="chart-container">
            <h3>Recent Anomaly Events</h3>
            <div id="anomaly-list" class="anomaly-list"></div>
        </div>
    </div>

    <script>
        let monitoringActive = true;
        let refreshInterval;

        function updateStats() {
            fetch('/api/stats')
                .then(response => response.json())
                .then(data => {
                    const network = data.network || {};
                    const anomalies = data.anomalies || {};
                    const performance = data.performance || {};

                    // Update basic stats
                    document.getElementById('switches-count').textContent = network.topology?.switches || '-';
                    document.getElementById('flows-count').textContent = network.traffic?.total_flows || '-';
                    document.getElementById('packets-count').textContent = network.traffic?.total_packets || '-';
                    document.getElementById('anomalies-count').textContent = anomalies.total_count || '-';
                    document.getElementById('throughput').textContent = (performance.current_throughput || 0).toFixed(2);
                    document.getElementById('latency').textContent = (performance.current_latency || 0).toFixed(1);

                    // Update system status
                    const statusEl = document.getElementById('system-status');
                    const statusText = document.getElementById('status-text');

                    if (anomalies.recent_count > 5) {
                        statusEl.className = 'status-indicator status-offline';
                        statusText.textContent = 'High Alert';
                    } else if (anomalies.recent_count > 0) {
                        statusEl.className = 'status-indicator status-warning';
                        statusText.textContent = 'Warning';
                    } else {
                        statusEl.className = 'status-indicator status-online';
                        statusText.textContent = 'Normal';
                    }

                    // Show anomaly alert
                    const alertEl = document.getElementById('anomaly-alert');
                    const alertDetails = document.getElementById('alert-details');

                    if (anomalies.recent_count > 0 && anomalies.latest.length > 0) {
                        const latestAnomaly = anomalies.latest[anomalies.latest.length - 1];
                        alertDetails.textContent = `${latestAnomaly.type} detected (Confidence: ${(latestAnomaly.confidence * 100).toFixed(1)}%)`;
                        alertEl.style.display = 'block';
                    } else {
                        alertEl.style.display = 'none';
                    }

                    // Update anomaly list
                    updateAnomalyList(anomalies.latest || []);
                })
                .catch(error => console.error('Error updating stats:', error));
        }

        function updateAnomalyList(anomalies) {
            const listEl = document.getElementById('anomaly-list');

            if (anomalies.length === 0) {
                listEl.innerHTML = '<p style="text-align: center; color: #666;">No recent anomalies detected</p>';
                return;
            }

            const html = anomalies.map(anomaly => {
                const date = new Date(anomaly.timestamp * 1000);
                return `
                    <div class="anomaly-item">
                        <strong>${anomaly.type.replace('_', ' ').toUpperCase()}</strong>
                        (${(anomaly.confidence * 100).toFixed(1)}% confidence)
                        <br>
                        <span style="font-size: 0.8em; color: #666;">
                            ${anomaly.source_ip} → ${anomaly.target_ip}
                        </span>
                        <div class="timestamp">${date.toLocaleString()}</div>
                    </div>
                `;
            }).join('');

            listEl.innerHTML = html;
        }

        function updatePlots() {
            // Update traffic plot
            fetch('/api/traffic_plot')
                .then(response => response.json())
                .then(data => {
                    if (data && Object.keys(data).length > 0) {
                        Plotly.newPlot('traffic-plot', data.data, data.layout, {responsive: true});
                    }
                })
                .catch(error => console.error('Error updating traffic plot:', error));

            // Update anomaly plot
            fetch('/api/anomaly_plot')
                .then(response => response.json())
                .then(data => {
                    if (data && Object.keys(data).length > 0) {
                        Plotly.newPlot('anomaly-plot', data.data, data.layout, {responsive: true});
                    }
                })
                .catch(error => console.error('Error updating anomaly plot:', error));

            // Update performance plot
            fetch('/api/performance_plot')
                .then(response => response.json())
                .then(data => {
                    if (data && Object.keys(data).length > 0) {
                        Plotly.newPlot('performance-plot', data.data, data.layout, {responsive: true});
                    }
                })
                .catch(error => console.error('Error updating performance plot:', error));
        }

        function refreshData() {
            updateStats();
            updatePlots();
        }

        function toggleMonitoring() {
            monitoringActive = !monitoringActive;
            const btn = document.getElementById('monitor-btn');

            if (monitoringActive) {
                btn.textContent = '⏸️ Pause Monitoring';
                startAutoRefresh();
            } else {
                btn.textContent = '▶️ Resume Monitoring';
                clearInterval(refreshInterval);
            }
        }

        function clearAlerts() {
            document.getElementById('anomaly-alert').style.display = 'none';
        }

        function startAutoRefresh() {
            if (refreshInterval) clearInterval(refreshInterval);
            refreshInterval = setInterval(refreshData, 5000); // 每5秒刷新
        }

        // Initialize dashboard
        document.addEventListener('DOMContentLoaded', function() {
            refreshData();
            startAutoRefresh();
        });
    </script>
</body>
</html>
"""

@app.route('/')
def dashboard():
    """主仪表盘页面"""
    return render_template_string(DASHBOARD_HTML)

@app.route('/api/stats')
def api_stats():
    """获取当前统计数据"""
    return jsonify(monitor.get_current_stats())

@app.route('/api/traffic_plot')
def api_traffic_plot():
    """获取流量图表数据"""
    plot_json = monitor.generate_traffic_plot()
    if plot_json:
        return json.loads(plot_json)
    return {}

@app.route('/api/anomaly_plot')
def api_anomaly_plot():
    """获取异常检测图表数据"""
    plot_json = monitor.generate_anomaly_plot()
    if plot_json:
        return json.loads(plot_json)
    return {}

@app.route('/api/performance_plot')
def api_performance_plot():
    """获取性能指标图表数据"""
    plot_json = monitor.generate_performance_plot()
    if plot_json:
        return json.loads(plot_json)
    return {}

@app.route('/api/health')
def api_health():
    """健康检查"""
    return jsonify({
        'status': 'healthy',
        'monitoring_active': monitor.simulation_active,
        'data_points': len(monitor.network_stats)
    })

def run_dashboard(host='0.0.0.0', port=8080, debug=False):
    """启动监控仪表盘"""
    print(f"开始启动网络监控面板 http://{host}:{port}")
    print("Dashboard features:")
    print("- Real-time network traffic visualization")
    print("- Anomaly detection alerts and timeline")
    print("- Performance metrics monitoring")
    print("- Interactive charts and controls")
    print("- System status indicators")

    # 启动数据监控
    monitor.start_monitoring()

    try:
        app.run(host=host, port=port, debug=debug, threaded=True)
    except KeyboardInterrupt:
        print("\nStopping dashboard...")
    finally:
        monitor.stop_monitoring()

if __name__ == '__main__':
    run_dashboard()