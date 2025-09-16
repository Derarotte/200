#!/bin/bash
# Network Intelligence System Demo Runner
# Automated script to run the complete demonstration

set -e  # Exit on any error

echo "🌐 Network Intelligence System Demo"
echo "=================================="
echo

# Check if running as root (needed for Mininet)
if [[ $EUID -ne 0 ]]; then
   echo "❌ This demo requires root privileges for Mininet"
   echo "Please run with: sudo ./run_demo.sh"
   exit 1
fi

# Set script directory as working directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "📁 Working directory: $SCRIPT_DIR"
echo

# Check Python version
echo "🐍 Checking Python version..."
python3 --version
echo

# Install Python dependencies if requirements.txt exists
if [ -f "requirements.txt" ]; then
    echo "📦 Installing Python dependencies..."
    pip3 install -r requirements.txt
    echo
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data logs models results templates
echo "  ✅ Directories created"
echo

# Clean up any previous Mininet instances
echo "🧹 Cleaning up previous instances..."
mn -c > /dev/null 2>&1 || true
echo "  ✅ Cleanup completed"
echo

# Check available demo modes
echo "🎬 Available demo modes:"
echo "  1. Interactive Demo (step-by-step with user prompts)"
echo "  2. Automated Demo (runs automatically for 5 minutes)"
echo "  3. Quick Test (basic functionality test)"
echo

# Get user choice
read -p "Select demo mode (1-3): " choice

case $choice in
    1)
        echo "🎭 Starting Interactive Demo..."
        echo "Follow the prompts to proceed through each step"
        echo
        python3 demo.py --mode interactive
        ;;
    2)
        echo "🤖 Starting Automated Demo..."
        echo "Demo will run automatically for 5 minutes"
        echo
        python3 demo.py --mode automated --duration 300
        ;;
    3)
        echo "⚡ Starting Quick Test..."
        echo "Basic functionality test (2 minutes)"
        echo
        python3 demo.py --mode automated --duration 120
        ;;
    *)
        echo "❌ Invalid choice. Please run the script again."
        exit 1
        ;;
esac

echo
echo "🎉 Demo completed!"
echo
echo "📊 Generated files:"
find data/ -name "*.json" 2>/dev/null | head -5
echo
find results/ -name "*.*" 2>/dev/null | head -5
echo

echo "📝 To run individual components:"
echo "  Network Topology:    sudo python3 src/mininet_topology/custom_topology.py"
echo "  SDN Controller:      ryu-manager src/sdn_controller/intelligent_controller.py"
echo "  Traffic Generator:   python3 src/mininet_topology/traffic_generator.py"
echo "  AI Detection:        python3 src/ai_detection/anomaly_detector.py"
echo "  Dashboard:           python3 src/monitoring/dashboard.py"
echo "  Performance Tests:   python3 src/utils/performance_tester.py"
echo

echo "🌐 Access points:"
echo "  Dashboard: http://localhost:5000"
echo "  API:       http://localhost:5000/api/"
echo

echo "✅ Demo script completed successfully!"