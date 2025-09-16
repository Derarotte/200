#!/usr/bin/env python3
"""
Custom Mininet topology for network monitoring demonstration
Implements a multi-tier network with varying traffic patterns
"""

from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import RemoteController, OVSKernelSwitch
from mininet.cli import CLI
from mininet.log import setLogLevel, info
from mininet.link import TCLink
import time


class NetworkTopology(Topo):
    """Custom topology with core, aggregation, and access layers"""
    
    def __init__(self):
        super().__init__()
        
        # Core switches
        core1 = self.addSwitch('s1', dpid='0000000000000001')
        core2 = self.addSwitch('s2', dpid='0000000000000002')
        
        # Aggregation switches  
        agg1 = self.addSwitch('s3', dpid='0000000000000003')
        agg2 = self.addSwitch('s4', dpid='0000000000000004')
        agg3 = self.addSwitch('s5', dpid='0000000000000005')
        
        # Access switches
        acc1 = self.addSwitch('s6', dpid='0000000000000006')
        acc2 = self.addSwitch('s7', dpid='0000000000000007')
        acc3 = self.addSwitch('s8', dpid='0000000000000008')
        acc4 = self.addSwitch('s9', dpid='0000000000000009')
        
        # Core layer interconnections
        self.addLink(core1, core2, bw=1000, delay='1ms', loss=0)
        
        # Core to aggregation links
        self.addLink(core1, agg1, bw=500, delay='2ms', loss=0)
        self.addLink(core1, agg2, bw=500, delay='2ms', loss=0)
        self.addLink(core2, agg2, bw=500, delay='2ms', loss=0)
        self.addLink(core2, agg3, bw=500, delay='2ms', loss=0)
        
        # Aggregation to access links
        self.addLink(agg1, acc1, bw=100, delay='5ms', loss=0)
        self.addLink(agg1, acc2, bw=100, delay='5ms', loss=0)
        self.addLink(agg2, acc2, bw=100, delay='5ms', loss=0)
        self.addLink(agg2, acc3, bw=100, delay='5ms', loss=0)
        self.addLink(agg3, acc3, bw=100, delay='5ms', loss=0)
        self.addLink(agg3, acc4, bw=100, delay='5ms', loss=0)
        
        # Add hosts
        for i in range(1, 9):
            host = self.addHost(f'h{i}', ip=f'10.0.0.{i}/24')
            # Distribute hosts across access switches
            if i <= 2:
                self.addLink(host, acc1, bw=10, delay='1ms')
            elif i <= 4:
                self.addLink(host, acc2, bw=10, delay='1ms')
            elif i <= 6:
                self.addLink(host, acc3, bw=10, delay='1ms')
            else:
                self.addLink(host, acc4, bw=10, delay='1ms')


def create_network():
    """Create and configure the Mininet network"""
    topo = NetworkTopology()
    
    # Use remote controller (Ryu)
    net = Mininet(
        topo=topo,
        controller=RemoteController,
        switch=OVSKernelSwitch,
        link=TCLink,
        autoSetMacs=True,
        autoStaticArp=True
    )
    
    return net


def start_network():
    """Start the network and run basic connectivity tests"""
    setLogLevel('info')
    
    net = create_network()
    net.start()
    
    info("*** Network started\n")
    info("*** Testing connectivity\n")
    
    # Basic connectivity test
    net.pingAll()
    
    info("*** Network topology created successfully\n")
    info("*** Run 'python3 traffic_generator.py' in another terminal to generate traffic\n")
    info("*** Run 'python3 sdn_controller.py' to start the SDN controller\n")
    
    CLI(net)
    net.stop()


if __name__ == '__main__':
    start_network()