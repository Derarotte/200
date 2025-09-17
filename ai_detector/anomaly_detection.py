#!/usr/bin/env python3
"""
AI异常检测系统
使用机器学习算法检测网络流量中的异常行为
包括：Isolation Forest、统计分析、规则引擎
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import time
import json
import threading
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import joblib
import os

# 导入网络模拟模块
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from network_sim.sdn_network import NetworkPacket


@dataclass
class AnomalyResult:
    """异常检测结果"""
    timestamp: float
    anomaly_detected: bool
    anomaly_type: str
    confidence: float
    details: Dict
    affected_flows: List[str]
    recommended_actions: List[str]


class FlowFeatureExtractor:
    """流量特征提取器"""

    def __init__(self, window_size=60):
        self.window_size = window_size
        self.flow_windows = defaultdict(lambda: deque(maxlen=1000))
        self.global_stats = {
            'total_packets': 0,
            'total_bytes': 0,
            'unique_flows': set(),
            'start_time': time.time()
        }

    def extract_features(self, packets: List[NetworkPacket]) -> np.ndarray:
        """从数据包列表提取特征向量"""
        if not packets:
            return np.zeros(15)

        current_time = time.time()

        # 基本统计特征
        total_packets = len(packets)
        total_bytes = sum(p.size for p in packets)
        avg_packet_size = total_bytes / total_packets if total_packets > 0 else 0

        # 时间窗口特征
        window_packets = [p for p in packets if current_time - p.timestamp <= self.window_size]
        packet_rate = len(window_packets) / self.window_size
        byte_rate = sum(p.size for p in window_packets) / self.window_size

        # 流多样性特征
        unique_flows = len(set(p.flow_id for p in packets))
        flow_diversity = unique_flows / total_packets if total_packets > 0 else 0

        # 协议分布特征
        protocols = [p.protocol for p in packets]
        tcp_ratio = protocols.count('TCP') / len(protocols) if protocols else 0
        udp_ratio = protocols.count('UDP') / len(protocols) if protocols else 0

        # 端口分布特征
        dst_ports = [p.dst_port for p in packets]
        unique_ports = len(set(dst_ports))
        port_diversity = unique_ports / len(dst_ports) if dst_ports else 0

        # 包大小分布特征
        sizes = [p.size for p in packets]
        if sizes:
            size_std = np.std(sizes)
            size_variance = np.var(sizes)
            large_packet_ratio = len([s for s in sizes if s > 1000]) / len(sizes)
            small_packet_ratio = len([s for s in sizes if s < 100]) / len(sizes)
        else:
            size_std = size_variance = large_packet_ratio = small_packet_ratio = 0

        # 时间间隔特征
        if len(packets) > 1:
            timestamps = sorted([p.timestamp for p in packets])
            intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
            avg_interval = np.mean(intervals) if intervals else 0
            interval_std = np.std(intervals) if intervals else 0
        else:
            avg_interval = interval_std = 0

        # 构建特征向量
        features = np.array([
            packet_rate,           # 0: 包速率
            byte_rate,            # 1: 字节速率
            avg_packet_size,      # 2: 平均包大小
            flow_diversity,       # 3: 流多样性
            tcp_ratio,            # 4: TCP比例
            udp_ratio,            # 5: UDP比例
            port_diversity,       # 6: 端口多样性
            size_std,             # 7: 包大小标准差
            size_variance,        # 8: 包大小方差
            large_packet_ratio,   # 9: 大包比例
            small_packet_ratio,   # 10: 小包比例
            avg_interval,         # 11: 平均时间间隔
            interval_std,         # 12: 时间间隔标准差
            unique_flows,         # 13: 唯一流数量
            total_packets         # 14: 总包数
        ])

        return features

    def update_global_stats(self, packets: List[NetworkPacket]):
        """更新全局统计信息"""
        self.global_stats['total_packets'] += len(packets)
        self.global_stats['total_bytes'] += sum(p.size for p in packets)

        for packet in packets:
            self.global_stats['unique_flows'].add(packet.flow_id)
            self.flow_windows[packet.flow_id].append(packet)


class StatisticalAnomalyDetector:
    """基于统计的异常检测器"""

    def __init__(self):
        self.baseline_stats = {}
        self.thresholds = {
            'packet_rate_multiplier': 5.0,
            'byte_rate_multiplier': 5.0,
            'flow_diversity_threshold': 0.1,
            'port_scan_threshold': 20,
            'ddos_packet_rate': 100,
            'large_packet_threshold': 0.3
        }

    def establish_baseline(self, feature_history: List[np.ndarray]):
        """建立正常流量基线"""
        if len(feature_history) < 10:
            return

        features_matrix = np.vstack(feature_history)

        self.baseline_stats = {
            'means': np.mean(features_matrix, axis=0),
            'stds': np.std(features_matrix, axis=0) + 1e-6,  # 避免除零
            'mins': np.min(features_matrix, axis=0),
            'maxs': np.max(features_matrix, axis=0)
        }

        print("Statistical baseline established")
        print(f"Baseline packet rate: {self.baseline_stats['means'][0]:.2f} ± {self.baseline_stats['stds'][0]:.2f}")
        print(f"Baseline byte rate: {self.baseline_stats['means'][1]:.2f} ± {self.baseline_stats['stds'][1]:.2f}")

    def detect_anomalies(self, features: np.ndarray) -> List[Dict]:
        """检测统计异常"""
        anomalies = []

        if not self.baseline_stats:
            return anomalies

        means = self.baseline_stats['means']
        stds = self.baseline_stats['stds']

        # Z-score异常检测
        z_scores = np.abs((features - means) / stds)

        # 特定规则检测
        packet_rate = features[0]
        byte_rate = features[1]
        flow_diversity = features[3]
        port_diversity = features[6]
        large_packet_ratio = features[9]
        unique_flows = features[13]

        # DDoS攻击检测
        if packet_rate > self.thresholds['ddos_packet_rate']:
            anomalies.append({
                'type': 'ddos_attack',
                'confidence': min(packet_rate / self.thresholds['ddos_packet_rate'], 2.0),
                'details': f'High packet rate: {packet_rate:.2f} pps'
            })

        # 端口扫描检测
        if (port_diversity > self.thresholds['port_scan_threshold'] and
            unique_flows > 10 and flow_diversity < self.thresholds['flow_diversity_threshold']):
            anomalies.append({
                'type': 'port_scan',
                'confidence': port_diversity / self.thresholds['port_scan_threshold'],
                'details': f'Port diversity: {port_diversity}, Flow diversity: {flow_diversity:.3f}'
            })

        # 大包传输异常
        if large_packet_ratio > self.thresholds['large_packet_threshold']:
            anomalies.append({
                'type': 'large_transfer',
                'confidence': large_packet_ratio / self.thresholds['large_packet_threshold'],
                'details': f'Large packet ratio: {large_packet_ratio:.3f}'
            })

        # 通用统计异常
        high_z_indices = np.where(z_scores > 3.0)[0]
        if len(high_z_indices) > 2:
            anomalies.append({
                'type': 'statistical_anomaly',
                'confidence': np.mean(z_scores[high_z_indices]) / 3.0,
                'details': f'High Z-scores for features: {high_z_indices.tolist()}'
            })

        return anomalies


class MLAnomalyDetector:
    """基于机器学习的异常检测器"""

    def __init__(self, model_path="models"):
        self.model_path = model_path
        self.isolation_forest = None
        self.scaler = StandardScaler()
        self.dbscan = None
        self.is_trained = False

        # 创建模型目录
        if not os.path.exists(model_path):
            os.makedirs(model_path)

    def train_model(self, training_features: List[np.ndarray]):
        """训练异常检测模型"""
        if len(training_features) < 50:
            print("Insufficient training data for ML model")
            return False

        print(f"Training ML model with {len(training_features)} samples...")

        # 准备训练数据
        X = np.vstack(training_features)

        # 数据标准化
        X_scaled = self.scaler.fit_transform(X)

        # 训练Isolation Forest
        self.isolation_forest = IsolationForest(
            contamination=0.1,  # 预期异常比例
            random_state=42,
            n_estimators=100,
            max_features=1.0
        )
        self.isolation_forest.fit(X_scaled)

        # 训练DBSCAN聚类（用于识别异常模式）
        self.dbscan = DBSCAN(eps=0.5, min_samples=5)
        self.dbscan.fit(X_scaled)

        self.is_trained = True
        self.save_models()

        print("ML model training completed")
        return True

    def detect_anomalies(self, features: np.ndarray) -> Optional[Dict]:
        """使用ML模型检测异常"""
        if not self.is_trained:
            return None

        # 标准化特征
        features_scaled = self.scaler.transform(features.reshape(1, -1))

        # Isolation Forest预测
        anomaly_score = self.isolation_forest.decision_function(features_scaled)[0]
        is_anomaly = self.isolation_forest.predict(features_scaled)[0] == -1

        # DBSCAN聚类预测
        cluster_label = self.dbscan.fit_predict(features_scaled)[0]
        is_outlier = cluster_label == -1

        # 综合判断
        confidence = abs(anomaly_score)
        final_anomaly = is_anomaly or (is_outlier and confidence > 0.3)

        return {
            'is_anomaly': final_anomaly,
            'anomaly_score': float(anomaly_score),
            'confidence': float(confidence),
            'cluster_label': int(cluster_label),
            'details': {
                'isolation_forest_anomaly': bool(is_anomaly),
                'dbscan_outlier': bool(is_outlier)
            }
        }

    def save_models(self):
        """保存训练好的模型"""
        try:
            model_file = os.path.join(self.model_path, 'isolation_forest.pkl')
            scaler_file = os.path.join(self.model_path, 'scaler.pkl')
            dbscan_file = os.path.join(self.model_path, 'dbscan.pkl')

            joblib.dump(self.isolation_forest, model_file)
            joblib.dump(self.scaler, scaler_file)
            joblib.dump(self.dbscan, dbscan_file)

            print("ML models saved successfully")
        except Exception as e:
            print(f"Error saving models: {e}")

    def load_models(self):
        """加载预训练模型"""
        try:
            model_file = os.path.join(self.model_path, 'isolation_forest.pkl')
            scaler_file = os.path.join(self.model_path, 'scaler.pkl')
            dbscan_file = os.path.join(self.model_path, 'dbscan.pkl')

            if all(os.path.exists(f) for f in [model_file, scaler_file, dbscan_file]):
                self.isolation_forest = joblib.load(model_file)
                self.scaler = joblib.load(scaler_file)
                self.dbscan = joblib.load(dbscan_file)
                self.is_trained = True
                print("ML models loaded successfully")
                return True
        except Exception as e:
            print(f"Error loading models: {e}")

        return False


class ComprehensiveAnomalyDetector:
    """综合异常检测系统"""

    def __init__(self):
        self.feature_extractor = FlowFeatureExtractor()
        self.statistical_detector = StatisticalAnomalyDetector()
        self.ml_detector = MLAnomalyDetector()

        self.packet_buffer = deque(maxlen=10000)
        self.feature_history = deque(maxlen=1000)
        self.anomaly_history = deque(maxlen=100)

        self.baseline_established = False
        self.model_trained = False

        # 尝试加载预训练模型
        self.ml_detector.load_models()

    def process_packets(self, packets: List[NetworkPacket]) -> Optional[AnomalyResult]:
        """处理数据包并进行异常检测"""
        if not packets:
            return None

        # 添加到缓冲区
        self.packet_buffer.extend(packets)

        # 提取特征
        recent_packets = list(self.packet_buffer)[-1000:]  # 最近1000个包
        features = self.feature_extractor.extract_features(recent_packets)
        self.feature_history.append(features)

        # 更新全局统计
        self.feature_extractor.update_global_stats(packets)

        # 建立基线（需要足够的历史数据）
        if not self.baseline_established and len(self.feature_history) >= 50:
            self.statistical_detector.establish_baseline(list(self.feature_history)[:30])
            self.baseline_established = True

        # 训练ML模型（需要更多数据）
        if not self.model_trained and len(self.feature_history) >= 100:
            self.ml_detector.train_model(list(self.feature_history)[:80])
            self.model_trained = True

        # 执行异常检测
        return self._detect_anomalies(features, recent_packets)

    def _detect_anomalies(self, features: np.ndarray, packets: List[NetworkPacket]) -> AnomalyResult:
        """综合异常检测"""
        current_time = time.time()

        # 统计异常检测
        statistical_anomalies = self.statistical_detector.detect_anomalies(features)

        # ML异常检测
        ml_result = self.ml_detector.detect_anomalies(features)

        # 综合判断
        has_statistical_anomaly = len(statistical_anomalies) > 0
        has_ml_anomaly = ml_result and ml_result['is_anomaly'] if ml_result else False

        anomaly_detected = has_statistical_anomaly or has_ml_anomaly

        # 确定异常类型和置信度
        if anomaly_detected:
            if statistical_anomalies:
                # 使用统计检测的结果
                primary_anomaly = max(statistical_anomalies, key=lambda x: x['confidence'])
                anomaly_type = primary_anomaly['type']
                confidence = primary_anomaly['confidence']
            else:
                # 使用ML检测的结果
                anomaly_type = 'ml_detected_anomaly'
                confidence = ml_result['confidence']

            # 生成推荐动作
            recommended_actions = self._generate_recommendations(anomaly_type, confidence)

            # 获取受影响的流
            affected_flows = list(set([p.flow_id for p in packets[-100:]]))  # 最近的流

        else:
            anomaly_type = 'normal'
            confidence = 0.0
            recommended_actions = []
            affected_flows = []

        # 创建检测结果
        result = AnomalyResult(
            timestamp=current_time,
            anomaly_detected=anomaly_detected,
            anomaly_type=anomaly_type,
            confidence=min(confidence, 1.0),
            details={
                'statistical_anomalies': statistical_anomalies,
                'ml_result': ml_result,
                'feature_vector': features.tolist(),
                'packet_count': len(packets)
            },
            affected_flows=affected_flows,
            recommended_actions=recommended_actions
        )

        # 记录到历史
        self.anomaly_history.append(result)

        return result

    def _generate_recommendations(self, anomaly_type: str, confidence: float) -> List[str]:
        """生成推荐的响应动作"""
        actions = []

        if anomaly_type == 'ddos_attack':
            actions.extend([
                "实施流量限速 (Rate Limiting)",
                "启用DDoS防护机制",
                "阻断可疑源IP地址",
                "增加服务器资源",
                "启用CDN分流"
            ])
        elif anomaly_type == 'port_scan':
            actions.extend([
                "阻断扫描源IP",
                "启用端口敲门机制",
                "增强防火墙规则",
                "监控后续攻击行为",
                "记录安全事件日志"
            ])
        elif anomaly_type == 'large_transfer':
            actions.extend([
                "检查传输内容合法性",
                "实施带宽限制",
                "监控文件传输行为",
                "验证用户权限"
            ])
        else:
            actions.extend([
                "增强网络监控",
                "收集更多数据分析",
                "验证检测结果"
            ])

        if confidence > 0.8:
            actions.insert(0, "立即响应 - 高置信度威胁")
        elif confidence > 0.5:
            actions.insert(0, "警惕监控 - 中等威胁")

        return actions

    def get_detection_summary(self) -> Dict:
        """获取检测摘要"""
        recent_anomalies = [a for a in self.anomaly_history if a.anomaly_detected]

        anomaly_types = defaultdict(int)
        for anomaly in recent_anomalies:
            anomaly_types[anomaly.anomaly_type] += 1

        return {
            'timestamp': time.time(),
            'total_packets_processed': len(self.packet_buffer),
            'total_anomalies_detected': len(recent_anomalies),
            'recent_anomalies': len([a for a in recent_anomalies if time.time() - a.timestamp < 300]),
            'anomaly_types': dict(anomaly_types),
            'baseline_established': self.baseline_established,
            'model_trained': self.model_trained,
            'detection_accuracy': {
                'statistical_detector_active': self.baseline_established,
                'ml_detector_active': self.ml_detector.is_trained
            }
        }


# 测试函数
def test_anomaly_detector():
    """测试异常检测系统"""
    detector = ComprehensiveAnomalyDetector()

    # 生成测试数据包
    normal_packets = []
    for i in range(100):
        packet = NetworkPacket(
            src_ip=f"10.0.0.{(i % 4) + 1}",
            dst_ip=f"10.0.0.{((i + 1) % 4) + 1}",
            src_port=30000 + i,
            dst_port=80,
            protocol="TCP",
            size=random.randint(500, 1500),
            timestamp=time.time() + i * 0.1
        )
        normal_packets.append(packet)

    # 处理正常流量
    print("Processing normal traffic...")
    for i in range(0, len(normal_packets), 10):
        batch = normal_packets[i:i+10]
        result = detector.process_packets(batch)
        if result and result.anomaly_detected:
            print(f"Anomaly detected: {result.anomaly_type} (confidence: {result.confidence:.2f})")

    # 生成异常流量（DDoS模拟）
    print("\nGenerating DDoS attack...")
    attack_packets = []
    for i in range(200):
        packet = NetworkPacket(
            src_ip="10.0.0.1",
            dst_ip="10.0.0.3",
            src_port=30000 + i,
            dst_port=80,
            protocol="TCP",
            size=64,
            timestamp=time.time() + i * 0.01
        )
        attack_packets.append(packet)

    # 处理攻击流量
    result = detector.process_packets(attack_packets)
    if result:
        print(f"Detection result: {result.anomaly_type}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Recommended actions: {result.recommended_actions}")

    # 打印检测摘要
    summary = detector.get_detection_summary()
    print("\nDetection Summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    import random
    test_anomaly_detector()