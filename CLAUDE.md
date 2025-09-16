# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an AI-powered SDN (Software Defined Networking) network monitoring and management system called "Network Intelligence System". It combines Mininet network simulation, Ryu SDN controller, machine learning-based anomaly detection, and real-time web monitoring.

## Commands

### Demo and Testing
- **Run interactive demo**: `python3 demo.py --mode interactive`
- **Run automated demo**: `python3 demo.py --mode automated --duration 300`
- **Run shell script demo**: `sudo ./run_demo.sh` (requires root for Mininet)

### Individual Components
- **Network topology**: `sudo python3 src/mininet_topology/custom_topology.py`
- **SDN controller**: `ryu-manager src/sdn_controller/intelligent_controller.py`
- **Traffic generator**: `python3 src/mininet_topology/traffic_generator.py`
- **AI anomaly detection**: `python3 src/ai_detection/anomaly_detector.py`
- **Web dashboard**: `python3 src/monitoring/dashboard.py`
- **Performance tests**: `python3 src/utils/performance_tester.py`

### Setup
- **Install dependencies**: `pip3 install -r requirements.txt`
- **Clean Mininet**: `sudo mn -c`
- **Create directories**: `mkdir -p data logs models results templates`

### Accessing Services
- **Web dashboard**: http://localhost:5000
- **API endpoints**: http://localhost:5000/api/

## Architecture

### Core Components Structure
- **src/mininet_topology/**: Network topology simulation and traffic generation
  - `custom_topology.py`: Multi-tier network (core/aggregation/access layers)
  - `traffic_generator.py`: Generates various traffic patterns for testing
- **src/sdn_controller/**: Ryu-based intelligent SDN controller
  - `intelligent_controller.py`: AI-integrated controller with load balancing and anomaly response
- **src/ai_detection/**: Machine learning anomaly detection system
  - `anomaly_detector.py`: Ensemble models (Isolation Forest, Random Forest, Neural Networks)
- **src/monitoring/**: Real-time web dashboard and data visualization
  - `dashboard.py`: Flask web interface with Plotly charts
- **src/utils/**: Support utilities for data collection and performance testing
  - `data_collector.py`: Network statistics collection
  - `performance_tester.py`: Automated performance benchmarking

### Data Flow
1. Mininet creates network topology with multiple switches and hosts
2. SDN controller manages flow rules and monitors network state
3. Traffic generator creates diverse traffic patterns
4. Data collector gathers statistics (flow stats, port stats, topology info)
5. AI models analyze traffic patterns and detect anomalies
6. Web dashboard displays real-time metrics and alerts
7. Performance tester validates system capabilities

### Key Dependencies
- **Mininet**: Network emulation platform
- **Ryu**: SDN controller framework
- **scikit-learn & TensorFlow**: ML models for anomaly detection
- **Flask**: Web dashboard framework
- **NetworkX**: Graph analysis for topology management

### Demo Integration
The `demo.py` orchestrates all components in sequence, providing both interactive and automated modes. It handles process management, cleanup, and provides comprehensive status reporting.

### Model Storage
- AI models saved to `models/` directory after training
- Traffic data logged to `data/` directory as JSON files
- Performance results stored in `results/` directory