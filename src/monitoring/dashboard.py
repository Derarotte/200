#!/usr/bin/env python3
"""
Real-time Network Monitoring Dashboard
Web-based interface for monitoring network health and anomalies
"""

from flask import Flask, render_template, jsonify, request
import json
import os
import time
from datetime import datetime, timedelta
import threading
import pandas as pd
import plotly.graph_objs as go
import plotly.utils
from collections import defaultdict, deque
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.data_collector import NetworkDataCollector
from ai_detection.anomaly_detector import NetworkAnomalyDetector


app = Flask(__name__)
app.config['SECRET_KEY'] = 'network-monitoring-dashboard'

# Global variables for real-time data
network_stats = deque(maxlen=100)  # Keep last 100 data points
anomaly_alerts = deque(maxlen=50)   # Keep last 50 alerts
flow_data = deque(maxlen=100)       # Keep last 100 flow stats
topology_data = {}
performance_metrics = {}

# Initialize components
data_collector = NetworkDataCollector()
anomaly_detector = NetworkAnomalyDetector()

# Load trained models if available
if os.path.exists('models'):
    try:
        anomaly_detector.load_models()
        print("Loaded pre-trained anomaly detection models")
    except:
        print("No pre-trained models found")


@app.route('/')
def dashboard():
    """Main dashboard page"""
    return render_template('dashboard.html')


@app.route('/api/network-stats')
def get_network_stats():
    """API endpoint for network statistics"""
    try:
        current_stats = data_collector.collect_system_stats()
        if current_stats:
            network_stats.append(current_stats)
        
        # Return last 20 data points for chart
        return jsonify({
            'stats': list(network_stats)[-20:],
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/flow-stats')
def get_flow_stats():
    """API endpoint for OpenFlow statistics"""
    try:
        current_flows = data_collector.collect_ovs_flow_stats()
        if current_flows:
            flow_data.append(current_flows)
        
        return jsonify({
            'flows': list(flow_data)[-10:],
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/topology')
def get_topology():
    """API endpoint for network topology"""
    try:
        # Simulate topology data - in real implementation, this would come from controller
        topology = {
            'nodes': [
                {'id': 's1', 'type': 'switch', 'label': 'Core Switch 1'},
                {'id': 's2', 'type': 'switch', 'label': 'Core Switch 2'},
                {'id': 's3', 'type': 'switch', 'label': 'Agg Switch 1'},
                {'id': 's4', 'type': 'switch', 'label': 'Agg Switch 2'},
                {'id': 's5', 'type': 'switch', 'label': 'Agg Switch 3'},
                {'id': 's6', 'type': 'switch', 'label': 'Access Switch 1'},
                {'id': 's7', 'type': 'switch', 'label': 'Access Switch 2'},
                {'id': 's8', 'type': 'switch', 'label': 'Access Switch 3'},
                {'id': 's9', 'type': 'switch', 'label': 'Access Switch 4'},
                {'id': 'h1', 'type': 'host', 'label': 'Host 1'},
                {'id': 'h2', 'type': 'host', 'label': 'Host 2'},
                {'id': 'h3', 'type': 'host', 'label': 'Host 3'},
                {'id': 'h4', 'type': 'host', 'label': 'Host 4'},
                {'id': 'h5', 'type': 'host', 'label': 'Host 5'},
                {'id': 'h6', 'type': 'host', 'label': 'Host 6'},
                {'id': 'h7', 'type': 'host', 'label': 'Host 7'},
                {'id': 'h8', 'type': 'host', 'label': 'Host 8'},
            ],
            'edges': [
                {'source': 's1', 'target': 's2', 'type': 'core'},
                {'source': 's1', 'target': 's3', 'type': 'core-agg'},
                {'source': 's1', 'target': 's4', 'type': 'core-agg'},
                {'source': 's2', 'target': 's4', 'type': 'core-agg'},
                {'source': 's2', 'target': 's5', 'type': 'core-agg'},
                {'source': 's3', 'target': 's6', 'type': 'agg-access'},
                {'source': 's3', 'target': 's7', 'type': 'agg-access'},
                {'source': 's4', 'target': 's7', 'type': 'agg-access'},
                {'source': 's4', 'target': 's8', 'type': 'agg-access'},
                {'source': 's5', 'target': 's8', 'type': 'agg-access'},
                {'source': 's5', 'target': 's9', 'type': 'agg-access'},
                {'source': 's6', 'target': 'h1', 'type': 'access-host'},
                {'source': 's6', 'target': 'h2', 'type': 'access-host'},
                {'source': 's7', 'target': 'h3', 'type': 'access-host'},
                {'source': 's7', 'target': 'h4', 'type': 'access-host'},
                {'source': 's8', 'target': 'h5', 'type': 'access-host'},
                {'source': 's8', 'target': 'h6', 'type': 'access-host'},
                {'source': 's9', 'target': 'h7', 'type': 'access-host'},
                {'source': 's9', 'target': 'h8', 'type': 'access-host'},
            ]
        }
        
        return jsonify(topology)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/anomaly-detection', methods=['POST'])
def detect_anomalies():
    """API endpoint for anomaly detection"""
    try:
        if not anomaly_detector.is_trained:
            return jsonify({'error': 'Anomaly detection model not trained'}), 400
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Detect anomalies
        results = anomaly_detector.detect_anomalies(data)
        
        # Store alerts for anomalous traffic
        for i, result in enumerate(results):
            if result['ensemble'] == 1:
                alert = {
                    'timestamp': datetime.now().isoformat(),
                    'data': data[i] if isinstance(data, list) else data,
                    'detection_result': result,
                    'severity': 'high' if result['anomaly_score'] > 0.8 else 'medium'
                }
                anomaly_alerts.append(alert)
        
        return jsonify({'results': results})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/alerts')
def get_alerts():
    """API endpoint for anomaly alerts"""
    try:
        return jsonify({
            'alerts': list(anomaly_alerts)[-20:],
            'total_alerts': len(anomaly_alerts)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/performance-metrics')
def get_performance_metrics():
    """API endpoint for performance metrics"""
    try:
        # Calculate performance metrics from recent data
        if network_stats:
            recent_stats = list(network_stats)[-10:]
            
            # Calculate averages
            total_connections = sum(s.get('active_connections', 0) for s in recent_stats)
            avg_connections = total_connections / len(recent_stats)
            
            # Calculate bandwidth utilization
            total_bytes = 0
            interface_count = 0
            
            for stats in recent_stats:
                for interface, counters in stats.get('interfaces', {}).items():
                    total_bytes += counters.get('bytes_recv', 0) + counters.get('bytes_sent', 0)
                    interface_count += 1
            
            avg_bandwidth = (total_bytes / interface_count) if interface_count > 0 else 0
            
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'average_connections': avg_connections,
                'average_bandwidth': avg_bandwidth,
                'total_alerts': len(anomaly_alerts),
                'active_flows': len(flow_data),
                'network_health': 'healthy' if len(anomaly_alerts) < 5 else 'warning'
            }
            
            return jsonify(metrics)
        
        return jsonify({
            'timestamp': datetime.now().isoformat(),
            'average_connections': 0,
            'average_bandwidth': 0,
            'total_alerts': 0,
            'active_flows': 0,
            'network_health': 'unknown'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/traffic-analysis')
def get_traffic_analysis():
    """API endpoint for traffic analysis charts"""
    try:
        # Load traffic data if available
        traffic_file = 'data/traffic_log.json'
        if os.path.exists(traffic_file):
            with open(traffic_file, 'r') as f:
                traffic_data = json.load(f)
            
            df = pd.DataFrame(traffic_data)
            
            if not df.empty:
                # Create charts
                charts = {}
                
                # Traffic type distribution
                type_counts = df['type'].value_counts()
                charts['traffic_types'] = {
                    'labels': type_counts.index.tolist(),
                    'values': type_counts.values.tolist()
                }
                
                # Protocol distribution
                protocol_counts = df['protocol'].value_counts().head(10)
                charts['protocols'] = {
                    'labels': protocol_counts.index.tolist(),
                    'values': protocol_counts.values.tolist()
                }
                
                # Bandwidth over time
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df_resampled = df.set_index('timestamp').resample('1T')['bandwidth'].mean()
                charts['bandwidth_time'] = {
                    'timestamps': [t.isoformat() for t in df_resampled.index],
                    'values': df_resampled.fillna(0).values.tolist()
                }
                
                # Packet size distribution
                charts['packet_sizes'] = {
                    'normal': df[df['type'] == 'normal']['packet_size'].tolist(),
                    'anomaly': df[df['type'] == 'anomaly']['packet_size'].tolist()
                }
                
                return jsonify(charts)
        
        return jsonify({'error': 'No traffic data available'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/system-info')
def get_system_info():
    """API endpoint for system information"""
    try:
        import psutil
        
        system_info = {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent,
            'network_connections': len(psutil.net_connections()),
            'uptime': time.time() - psutil.boot_time(),
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(system_info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def create_dashboard_template():
    """Create HTML template for dashboard"""
    template_dir = 'templates'
    os.makedirs(template_dir, exist_ok=True)
    
    dashboard_html = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Network Intelligence Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        .metric-card { margin-bottom: 20px; }
        .alert-high { border-left: 4px solid #dc3545; }
        .alert-medium { border-left: 4px solid #ffc107; }
        .network-topology { height: 400px; border: 1px solid #ddd; }
        .chart-container { height: 300px; margin-bottom: 20px; }
    </style>
</head>
<body>
    <nav class="navbar navbar-dark bg-dark">
        <div class="container-fluid">
            <span class="navbar-brand mb-0 h1">🌐 Network Intelligence Dashboard</span>
            <span class="navbar-text" id="last-updated">Last updated: --</span>
        </div>
    </nav>

    <div class="container-fluid mt-3">
        <!-- Performance Metrics Row -->
        <div class="row">
            <div class="col-md-3">
                <div class="card metric-card">
                    <div class="card-body">
                        <h6 class="card-title">Network Health</h6>
                        <h3 class="text-success" id="network-health">Healthy</h3>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card metric-card">
                    <div class="card-body">
                        <h6 class="card-title">Active Connections</h6>
                        <h3 id="active-connections">0</h3>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card metric-card">
                    <div class="card-body">
                        <h6 class="card-title">Total Alerts</h6>
                        <h3 class="text-warning" id="total-alerts">0</h3>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card metric-card">
                    <div class="card-body">
                        <h6 class="card-title">Active Flows</h6>
                        <h3 id="active-flows">0</h3>
                    </div>
                </div>
            </div>
        </div>

        <!-- Charts Row -->
        <div class="row">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">Network Bandwidth</div>
                    <div class="card-body">
                        <div id="bandwidth-chart" class="chart-container"></div>
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">Traffic Distribution</div>
                    <div class="card-body">
                        <div id="traffic-chart" class="chart-container"></div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Network Topology and Alerts Row -->
        <div class="row">
            <div class="col-md-8">
                <div class="card">
                    <div class="card-header">Network Topology</div>
                    <div class="card-body">
                        <div id="topology-viz" class="network-topology"></div>
                    </div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="card">
                    <div class="card-header">Recent Alerts</div>
                    <div class="card-body">
                        <div id="alerts-list" style="height: 400px; overflow-y: auto;"></div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Auto-refresh dashboard
        setInterval(updateDashboard, 5000);
        updateDashboard();

        function updateDashboard() {
            updateMetrics();
            updateCharts();
            updateAlerts();
            updateTopology();
            document.getElementById('last-updated').textContent = 'Last updated: ' + new Date().toLocaleTimeString();
        }

        function updateMetrics() {
            fetch('/api/performance-metrics')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('network-health').textContent = data.network_health || 'Unknown';
                    document.getElementById('active-connections').textContent = Math.round(data.average_connections || 0);
                    document.getElementById('total-alerts').textContent = data.total_alerts || 0;
                    document.getElementById('active-flows').textContent = data.active_flows || 0;
                });
        }

        function updateCharts() {
            // Update bandwidth chart
            fetch('/api/network-stats')
                .then(response => response.json())
                .then(data => {
                    if (data.stats && data.stats.length > 0) {
                        const timestamps = data.stats.map(s => new Date(s.timestamp));
                        const connections = data.stats.map(s => s.active_connections || 0);
                        
                        const trace = {
                            x: timestamps,
                            y: connections,
                            type: 'scatter',
                            mode: 'lines+markers',
                            name: 'Active Connections'
                        };

                        Plotly.newPlot('bandwidth-chart', [trace], {
                            title: 'Network Activity',
                            xaxis: { title: 'Time' },
                            yaxis: { title: 'Connections' }
                        });
                    }
                });

            // Update traffic distribution chart
            fetch('/api/traffic-analysis')
                .then(response => response.json())
                .then(data => {
                    if (data.traffic_types) {
                        const trace = {
                            labels: data.traffic_types.labels,
                            values: data.traffic_types.values,
                            type: 'pie'
                        };

                        Plotly.newPlot('traffic-chart', [trace], {
                            title: 'Traffic Type Distribution'
                        });
                    }
                });
        }

        function updateAlerts() {
            fetch('/api/alerts')
                .then(response => response.json())
                .then(data => {
                    const alertsList = document.getElementById('alerts-list');
                    alertsList.innerHTML = '';
                    
                    if (data.alerts && data.alerts.length > 0) {
                        data.alerts.forEach(alert => {
                            const alertDiv = document.createElement('div');
                            alertDiv.className = `alert alert-${alert.severity} alert-${alert.severity}`;
                            alertDiv.innerHTML = `
                                <small>${new Date(alert.timestamp).toLocaleString()}</small><br>
                                <strong>Anomaly Detected</strong><br>
                                Score: ${alert.detection_result.anomaly_score.toFixed(3)}<br>
                                Source: ${alert.data.src} → ${alert.data.dst}
                            `;
                            alertsList.appendChild(alertDiv);
                        });
                    } else {
                        alertsList.innerHTML = '<div class="text-muted">No recent alerts</div>';
                    }
                });
        }

        function updateTopology() {
            fetch('/api/topology')
                .then(response => response.json())
                .then(data => {
                    // Simple topology visualization would go here
                    // For now, just show node count
                    const topologyDiv = document.getElementById('topology-viz');
                    topologyDiv.innerHTML = `
                        <div class="text-center mt-5">
                            <h4>Network Topology</h4>
                            <p>${data.nodes.length} nodes, ${data.edges.length} connections</p>
                            <small class="text-muted">Detailed visualization requires D3.js implementation</small>
                        </div>
                    `;
                });
        }
    </script>
</body>
</html>
'''
    
    with open(os.path.join(template_dir, 'dashboard.html'), 'w') as f:
        f.write(dashboard_html)


def start_background_collection():
    """Start background data collection"""
    def collect_data():
        while True:
            try:
                # Collect network stats
                stats = data_collector.collect_system_stats()
                if stats:
                    network_stats.append(stats)
                
                # Collect flow stats
                flows = data_collector.collect_ovs_flow_stats()
                if flows:
                    flow_data.append(flows)
                
                time.sleep(5)
            except Exception as e:
                print(f"Background collection error: {e}")
                time.sleep(10)
    
    thread = threading.Thread(target=collect_data, daemon=True)
    thread.start()


def main():
    """Main function to start the dashboard"""
    print("Starting Network Intelligence Dashboard...")
    
    # Create HTML template
    create_dashboard_template()
    
    # Start background data collection
    start_background_collection()
    
    # Start Flask app
    print("Dashboard available at: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)


if __name__ == '__main__':
    main()