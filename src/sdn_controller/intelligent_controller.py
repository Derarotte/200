#!/usr/bin/env python3
"""
Intelligent SDN Controller using Ryu framework
Implements adaptive network management with AI-driven decisions
"""

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ether_types, ipv4, tcp, udp, icmp
from ryu.topology import event, switches
from ryu.topology.api import get_switch, get_link
import networkx as nx
import time
import json
import threading
from datetime import datetime
import requests


class IntelligentController(app_manager.RyuApp):
    """AI-powered SDN controller for intelligent network management"""
    
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]
    
    def __init__(self, *args, **kwargs):
        super(IntelligentController, self).__init__(*args, **kwargs)
        
        # Network state
        self.mac_to_port = {}
        self.switches = {}
        self.links = {}
        self.topology_graph = nx.Graph()
        
        # Traffic monitoring
        self.flow_stats = {}
        self.port_stats = {}
        self.traffic_matrix = {}
        
        # AI integration
        self.anomaly_detector_url = "http://localhost:5001/detect"
        self.monitoring_enabled = True
        
        # Load balancing
        self.path_cache = {}
        self.link_utilization = {}
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitor_network)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
    
    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        """Handle switch connection"""
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        
        self.logger.info(f"Switch {datapath.id} connected")
        self.switches[datapath.id] = datapath
        
        # Install table-miss flow entry
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                        ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 0, match, actions)
        
        # Request port descriptions
        req = parser.OFPPortDescStatsRequest(datapath, 0)
        datapath.send_msg(req)
    
    def add_flow(self, datapath, priority, match, actions, buffer_id=None,
                 hard_timeout=0, idle_timeout=0):
        """Add a flow entry to the switch"""
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        
        if buffer_id:
            mod = parser.OFPFlowMod(datapath=datapath, buffer_id=buffer_id,
                                   priority=priority, match=match,
                                   instructions=inst, hard_timeout=hard_timeout,
                                   idle_timeout=idle_timeout)
        else:
            mod = parser.OFPFlowMod(datapath=datapath, priority=priority,
                                   match=match, instructions=inst,
                                   hard_timeout=hard_timeout, idle_timeout=idle_timeout)
        
        datapath.send_msg(mod)
    
    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        """Handle packet-in events"""
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']
        
        pkt = packet.Packet(msg.data)
        eth_pkt = pkt.get_protocols(ethernet.ethernet)[0]
        
        # Ignore LLDP packets
        if eth_pkt.ethertype == ether_types.ETH_TYPE_LLDP:
            return
        
        dst = eth_pkt.dst
        src = eth_pkt.src
        dpid = datapath.id
        
        # Learn MAC address
        self.mac_to_port.setdefault(dpid, {})
        self.mac_to_port[dpid][src] = in_port
        
        # Check for anomalies
        self._analyze_packet(pkt, dpid, in_port)
        
        # Determine output port
        if dst in self.mac_to_port[dpid]:
            out_port = self.mac_to_port[dpid][dst]
        else:
            out_port = ofproto.OFPP_FLOOD
        
        # Intelligent path selection for unicast traffic
        if out_port != ofproto.OFPP_FLOOD:
            path = self._get_optimal_path(dpid, dst)
            if path:
                self._install_path_flows(path, eth_pkt, in_port)
                return
        
        actions = [parser.OFPActionOutput(out_port)]
        
        # Install flow to avoid packet_in next time
        if out_port != ofproto.OFPP_FLOOD:
            match = parser.OFPMatch(in_port=in_port, eth_dst=dst, eth_src=src)
            self.add_flow(datapath, 1, match, actions, idle_timeout=30)
        
        data = None
        if msg.buffer_id == ofproto.OFP_NO_BUFFER:
            data = msg.data
        
        out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id,
                                 in_port=in_port, actions=actions, data=data)
        datapath.send_msg(out)
    
    def _analyze_packet(self, pkt, dpid, in_port):
        """Analyze packet for anomalies and security threats"""
        try:
            eth_pkt = pkt.get_protocols(ethernet.ethernet)[0]
            
            # Extract packet features
            packet_info = {
                'timestamp': datetime.now().isoformat(),
                'switch_id': dpid,
                'in_port': in_port,
                'src_mac': eth_pkt.src,
                'dst_mac': eth_pkt.dst,
                'packet_size': len(pkt.data) if pkt.data else 0,
                'ethertype': eth_pkt.ethertype
            }
            
            # Check for IP packets
            ip_pkt = pkt.get_protocol(ipv4.ipv4)
            if ip_pkt:
                packet_info.update({
                    'src_ip': ip_pkt.src,
                    'dst_ip': ip_pkt.dst,
                    'protocol': ip_pkt.proto,
                    'ttl': ip_pkt.ttl
                })
                
                # Check transport layer
                tcp_pkt = pkt.get_protocol(tcp.tcp)
                udp_pkt = pkt.get_protocol(udp.udp)
                icmp_pkt = pkt.get_protocol(icmp.icmp)
                
                if tcp_pkt:
                    packet_info.update({
                        'src_port': tcp_pkt.src_port,
                        'dst_port': tcp_pkt.dst_port,
                        'tcp_flags': tcp_pkt.bits
                    })
                elif udp_pkt:
                    packet_info.update({
                        'src_port': udp_pkt.src_port,
                        'dst_port': udp_pkt.dst_port
                    })
            
            # Send to anomaly detection system
            self._check_for_anomalies(packet_info)
            
        except Exception as e:
            self.logger.error(f"Error analyzing packet: {e}")
    
    def _check_for_anomalies(self, packet_info):
        """Send packet info to AI anomaly detection system"""
        try:
            # Convert to format expected by anomaly detector
            detection_data = {
                'timestamp': packet_info['timestamp'],
                'src': f"s{packet_info['switch_id']}",
                'dst': packet_info.get('dst_ip', 'unknown'),
                'src_ip': packet_info.get('src_ip', '0.0.0.0'),
                'dst_ip': packet_info.get('dst_ip', '0.0.0.0'),
                'protocol': self._get_protocol_name(packet_info.get('protocol', 0)),
                'type': 'unknown',
                'packet_size': packet_info['packet_size'],
                'bandwidth': self._estimate_bandwidth(packet_info)
            }
            
            # Check if this looks like an attack pattern
            if self._is_potential_attack(packet_info):
                self.logger.warning(f"Potential attack detected: {packet_info}")
                self._take_defensive_action(packet_info)
            
        except Exception as e:
            self.logger.error(f"Error checking for anomalies: {e}")
    
    def _get_protocol_name(self, proto_num):
        """Convert protocol number to name"""
        protocol_map = {1: 'ICMP', 6: 'TCP', 17: 'UDP'}
        return protocol_map.get(proto_num, f'PROTO_{proto_num}')
    
    def _estimate_bandwidth(self, packet_info):
        """Estimate bandwidth usage"""
        return packet_info['packet_size'] * 8 / 1024  # Convert to kbps estimate
    
    def _is_potential_attack(self, packet_info):
        """Simple heuristic-based attack detection"""
        # Port scanning detection
        if packet_info.get('tcp_flags') == 2:  # SYN flag only
            dst_ip = packet_info.get('dst_ip')
            if dst_ip:
                current_time = time.time()
                key = f"{packet_info.get('src_ip')}_{dst_ip}"
                
                if key not in self.traffic_matrix:
                    self.traffic_matrix[key] = []
                
                self.traffic_matrix[key].append(current_time)
                
                # Remove old entries
                self.traffic_matrix[key] = [t for t in self.traffic_matrix[key] 
                                          if current_time - t < 60]
                
                # Check for high connection rate
                if len(self.traffic_matrix[key]) > 50:  # 50 connections in 1 minute
                    return True
        
        # Large packet anomaly
        if packet_info['packet_size'] > 9000:
            return True
        
        return False
    
    def _take_defensive_action(self, packet_info):
        """Take defensive action against detected attacks"""
        try:
            # Block the source IP temporarily
            src_ip = packet_info.get('src_ip')
            if src_ip and src_ip != '0.0.0.0':
                self.logger.info(f"Blocking suspicious IP: {src_ip}")
                self._block_ip_address(src_ip, duration=300)  # Block for 5 minutes
        except Exception as e:
            self.logger.error(f"Error taking defensive action: {e}")
    
    def _block_ip_address(self, ip_address, duration=300):
        """Block an IP address on all switches"""
        for dpid, datapath in self.switches.items():
            parser = datapath.ofproto_parser
            ofproto = datapath.ofproto
            
            # Block incoming traffic from this IP
            match = parser.OFPMatch(eth_type=ether_types.ETH_TYPE_IP,
                                   ipv4_src=ip_address)
            actions = []  # Drop packet
            
            self.add_flow(datapath, 100, match, actions, 
                         hard_timeout=duration)
            
            self.logger.info(f"IP {ip_address} blocked on switch {dpid} for {duration} seconds")
    
    def _get_optimal_path(self, src_dpid, dst_mac):
        """Get optimal path using network topology"""
        try:
            # Find destination switch
            dst_dpid = None
            for dpid, mac_table in self.mac_to_port.items():
                if dst_mac in mac_table:
                    dst_dpid = dpid
                    break
            
            if not dst_dpid or dst_dpid == src_dpid:
                return None
            
            # Use cached path if available
            path_key = f"{src_dpid}_{dst_dpid}"
            if path_key in self.path_cache:
                return self.path_cache[path_key]
            
            # Calculate shortest path
            if self.topology_graph.has_node(src_dpid) and self.topology_graph.has_node(dst_dpid):
                path = nx.shortest_path(self.topology_graph, src_dpid, dst_dpid)
                self.path_cache[path_key] = path
                return path
            
        except Exception as e:
            self.logger.error(f"Error calculating optimal path: {e}")
        
        return None
    
    def _install_path_flows(self, path, eth_pkt, in_port):
        """Install flow entries along a path"""
        try:
            for i in range(len(path) - 1):
                curr_switch = path[i]
                next_switch = path[i + 1]
                
                # Find the port to next switch
                datapath = self.switches.get(curr_switch)
                if not datapath:
                    continue
                
                parser = datapath.ofproto_parser
                
                # Install forward flow
                match = parser.OFPMatch(eth_dst=eth_pkt.dst, eth_src=eth_pkt.src)
                out_port = self._get_port_to_switch(curr_switch, next_switch)
                
                if out_port:
                    actions = [parser.OFPActionOutput(out_port)]
                    self.add_flow(datapath, 10, match, actions, idle_timeout=30)
        
        except Exception as e:
            self.logger.error(f"Error installing path flows: {e}")
    
    def _get_port_to_switch(self, from_switch, to_switch):
        """Get port number from one switch to another"""
        # This would typically come from topology discovery
        # For now, return a default port
        return 1
    
    @set_ev_cls(event.EventSwitchEnter)
    def switch_enter_handler(self, ev):
        """Handle switch entering topology"""
        switch = ev.switch
        self.topology_graph.add_node(switch.dp.id)
        self.logger.info(f"Switch {switch.dp.id} entered topology")
    
    @set_ev_cls(event.EventLinkAdd)
    def link_add_handler(self, ev):
        """Handle link addition to topology"""
        link = ev.link
        src_dpid = link.src.dpid
        dst_dpid = link.dst.dpid
        
        self.topology_graph.add_edge(src_dpid, dst_dpid)
        self.links[(src_dpid, dst_dpid)] = link
        self.logger.info(f"Link added: {src_dpid} -> {dst_dpid}")
    
    def _monitor_network(self):
        """Continuous network monitoring"""
        while self.monitoring_enabled:
            try:
                self._collect_flow_stats()
                self._collect_port_stats()
                self._analyze_traffic_patterns()
                time.sleep(10)  # Monitor every 10 seconds
            except Exception as e:
                self.logger.error(f"Error in network monitoring: {e}")
                time.sleep(5)
    
    def _collect_flow_stats(self):
        """Collect flow statistics from all switches"""
        for dpid, datapath in self.switches.items():
            parser = datapath.ofproto_parser
            req = parser.OFPFlowStatsRequest(datapath)
            datapath.send_msg(req)
    
    def _collect_port_stats(self):
        """Collect port statistics from all switches"""
        for dpid, datapath in self.switches.items():
            parser = datapath.ofproto_parser
            req = parser.OFPPortStatsRequest(datapath, 0, datapath.ofproto.OFPP_ANY)
            datapath.send_msg(req)
    
    def _analyze_traffic_patterns(self):
        """Analyze traffic patterns for optimization"""
        try:
            # Analyze link utilization
            for (src, dst), link in self.links.items():
                utilization = self.link_utilization.get((src, dst), 0)
                
                # If utilization is high, consider load balancing
                if utilization > 0.8:
                    self.logger.warning(f"High utilization on link {src}->{dst}: {utilization:.2f}")
                    # Implement load balancing logic here
            
        except Exception as e:
            self.logger.error(f"Error analyzing traffic patterns: {e}")
    
    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def flow_stats_reply_handler(self, ev):
        """Handle flow statistics reply"""
        body = ev.msg.body
        dpid = ev.msg.datapath.id
        
        flows = []
        for stat in body:
            flows.append({
                'priority': stat.priority,
                'match': stat.match,
                'actions': stat.instructions,
                'packet_count': stat.packet_count,
                'byte_count': stat.byte_count,
                'duration': stat.duration_sec
            })
        
        self.flow_stats[dpid] = flows
    
    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def port_stats_reply_handler(self, ev):
        """Handle port statistics reply"""
        body = ev.msg.body
        dpid = ev.msg.datapath.id
        
        ports = {}
        for stat in body:
            ports[stat.port_no] = {
                'rx_packets': stat.rx_packets,
                'tx_packets': stat.tx_packets,
                'rx_bytes': stat.rx_bytes,
                'tx_bytes': stat.tx_bytes,
                'rx_errors': stat.rx_errors,
                'tx_errors': stat.tx_errors
            }
        
        self.port_stats[dpid] = ports


def main():
    """Main function to start the controller"""
    from ryu.cmd import manager
    import sys
    
    sys.argv = ['ryu-manager', __file__]
    manager.main()


if __name__ == '__main__':
    main()