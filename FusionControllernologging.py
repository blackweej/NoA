"""
FusionController 클래스 구현
두 번의 MoE 출력을 비교하고 fusion degree를 동적으로 조정하는 핵심 시스템

파일 위치: src/mistral_inference/fusion_controller.py
"""

import torch
import torch.nn.functional as F
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import numpy as np
import json
import os


class FusionMetrics:
    """개별 Expert의 Fusion 관련 메트릭"""
    def __init__(self, expert_id: int, similarity_score: float, novelty_score: float, fusion_degree: float):
        self.expert_id = expert_id
        self.similarity_score = similarity_score  # 0-1 사이
        self.novelty_score = novelty_score  # 0-1 사이
        self.fusion_degree = fusion_degree  # 0-1 사이
        self.last_updated = datetime.now()
        self.activation_weight = 0.0  # Expert의 활성화 가중치
        self.historical_performance = []  # 과거 성능 기록

    def to_dict(self) -> Dict:
        """직렬화를 위한 딕셔너리 변환"""
        return {
            'expert_id': self.expert_id,
            'similarity_score': self.similarity_score,
            'novelty_score': self.novelty_score,
            'fusion_degree': self.fusion_degree,
            'last_updated': self.last_updated.isoformat(),
            'activation_weight': self.activation_weight,
            'historical_performance': self.historical_performance
        }


class FusionResult:
    """최종 Fusion 결과"""
    def __init__(self, fused_output: torch.Tensor, fusion_weights: Dict[int, float], metrics: List[FusionMetrics]):
        self.fused_output = fused_output
        self.fusion_weights = fusion_weights  # expert_id: weight 매핑
        self.metrics = metrics  # List[FusionMetrics]
        self.total_fusion_strength = sum(fusion_weights.values())
        self.processing_time = 0.0

    def get_top_contributing_experts(self, k: int = 3) -> List[Tuple[int, float]]:
        """가장 큰 기여도를 가진 Expert들 반환"""
        sorted_experts = sorted(self.fusion_weights.items(), key=lambda x: x[1], reverse=True)
        return sorted_experts[:k]


class FusionController:
    """
    Fusion degree 계산 및 관리 시스템
    두 번의 MoE 출력을 비교하여 동적으로 fusion degree를 조정
    """
    
    def __init__(self, config: Optional[Dict] = None):
        # 기본 설정
        self.config = config or {
            'similarity_threshold': 0.7,
            'novelty_threshold': 0.3,
            'min_fusion_degree': 0.1,
            'max_fusion_degree': 0.9,
            'adjustment_rate': 0.1,
            'decay_factor': 0.95,
            'history_length': 10
        }
        
        # Fusion degree 저장소 (expert_id: fusion_degree)
        self.fusion_degrees: Dict[int, float] = {}
        
        # 과거 성능 기록
        self.performance_history: Dict[int, List[float]] = {}
        
        # 통계 정보
        self.stats = {
            'total_calculations': 0,
            'avg_similarity_score': 0.0,
            'avg_novelty_score': 0.0,
            'fusion_adjustments': 0
        }
        
        # 저장 경로
        self.save_path = "fusion_degrees.json"
        
        # 기존 데이터 로드
        self._load_fusion_degrees()

    def calculate_fusion_degree(self, first_output: torch.Tensor, second_output: torch.Tensor, 
                              max_experts) -> List[FusionMetrics]:
        """
        두 출력 간 비교를 통한 fusion_degree 계산
        
        Args:
            first_output: 첫 번째 MoE 출력 [batch_size, seq_len, hidden_dim]
            second_output: 두 번째 MoE 출력 [batch_size, seq_len, hidden_dim]
            max_experts: MaxExpertsList 객체
        
        Returns:
            fusion_metrics: List[FusionMetrics]
        """
        fusion_metrics = []
        
        # 전체 출력 차이 계산
        output_diff = self._calculate_global_output_difference(first_output, second_output)
        
        for expert_output in max_experts.experts:
            expert_id = expert_output.expert_id
            
            # 1. 유사도 점수 계산
            similarity_score = self._calculate_output_similarity(
                first_output, second_output, expert_output
            )
            
            # 2. 참신성 점수 계산
            novelty_score = self._calculate_novelty_score(
                first_output, second_output, expert_output, output_diff
            )
            
            # 3. 기존 fusion_degree 가져오기
            current_fusion_degree = self.fusion_degrees.get(expert_id, 1.0)
            
            # 4. 동적 조정 공식 적용
            new_fusion_degree = self._adjust_fusion_degree(
                current_fusion_degree,
                similarity_score,
                novelty_score,
                expert_output.activation_score
            )
            
            # 5. 메트릭 객체 생성
            metrics = FusionMetrics(
                expert_id=expert_id,
                similarity_score=similarity_score,
                novelty_score=novelty_score,
                fusion_degree=new_fusion_degree
            )
            metrics.activation_weight = expert_output.weight
            
            # 6. 과거 성능 기록 업데이트
            self._update_performance_history(expert_id, novelty_score)
            
            fusion_metrics.append(metrics)
            
            # 7. 새로운 fusion_degree 저장
            self.fusion_degrees[expert_id] = new_fusion_degree
        
        # 8. 통계 업데이트
        self._update_statistics(fusion_metrics)
        
        return fusion_metrics

    def apply_fusion_weights(self, first_output: torch.Tensor, second_output: torch.Tensor, 
                           fusion_metrics: List[FusionMetrics]) -> FusionResult:
        """
        Fusion degree를 적용한 최종 출력 생성
        
        Args:
            first_output: 첫 번째 MoE 출력
            second_output: 두 번째 MoE 출력
            fusion_metrics: FusionMetrics 리스트
        
        Returns:
            fusion_result: FusionResult 객체
        """
        start_time = datetime.now()
        
        # 1. Expert별 가중치 정규화
        fusion_weights = self._normalize_fusion_weights(fusion_metrics)
        
        # 2. 전체 fusion 강도 계산
        total_fusion_strength = sum(fusion_weights.values())
        
        # 3. 안전 장치: 모든 fusion_degree가 너무 낮은 경우
        if total_fusion_strength < self.config['min_fusion_degree']:
            return FusionResult(first_output, {}, fusion_metrics)
        
        # 4. 적응적 가중치 계산
        alpha = self._calculate_adaptive_alpha(total_fusion_strength, fusion_metrics)
        beta = 1.0 - alpha
        
        # 5. 공간별 가중 조합 (attention 기반)
        fused_output = self._spatial_weighted_fusion(
            first_output, second_output, fusion_weights, alpha, beta
        )
        
        # 6. 결과 패키징
        fusion_result = FusionResult(fused_output, fusion_weights, fusion_metrics)
        fusion_result.processing_time = (datetime.now() - start_time).total_seconds()
        
        return fusion_result

    def _calculate_global_output_difference(self, first_output: torch.Tensor, 
                                          second_output: torch.Tensor) -> torch.Tensor:
        """전체 출력 차이의 공간적 분포 계산"""
        # L2 거리 기반 차이 계산
        diff = torch.norm(first_output - second_output, p=2, dim=-1)  # [batch_size, seq_len]
        
        # 정규화
        diff_normalized = diff / (torch.max(diff) + 1e-8)
        
        return diff_normalized

    def _calculate_output_similarity(self, first_output: torch.Tensor, second_output: torch.Tensor, 
                                   expert_output) -> float:
        """두 출력 간의 유사도 계산"""
        # 1. 코사인 유사도 계산
        first_flat = first_output.flatten()
        second_flat = second_output.flatten()
        
        cosine_sim = F.cosine_similarity(first_flat.unsqueeze(0), second_flat.unsqueeze(0))
        
        # 2. Expert 활성화 영역에서의 유사도 가중치 적용
        activation_weight = expert_output.weight
        weighted_similarity = cosine_sim * activation_weight
        
        # 3. 정규화 (0-1 범위)
        similarity_score = torch.clamp(weighted_similarity, 0.0, 1.0).item()
        
        return similarity_score

    def _calculate_novelty_score(self, first_output: torch.Tensor, second_output: torch.Tensor, 
                               expert_output, output_diff: torch.Tensor) -> float:
        """참신성 점수 계산"""
        expert_id = expert_output.expert_id
        
        # 1. 출력 차이 기반 참신성
        diff_magnitude = torch.mean(output_diff).item()
        
        # 2. Expert의 과거 성능 기록 고려
        historical_performance = self.performance_history.get(expert_id, [0.5])
        avg_historical = np.mean(historical_performance)
        
        # 3. 활성화 패턴 기반 참신성
        activation_novelty = self._calculate_activation_novelty(expert_output)
        
        # 4. 종합 참신성 점수
        novelty_components = {
            'diff_magnitude': diff_magnitude * 0.4,
            'historical_deviation': abs(diff_magnitude - avg_historical) * 0.3,
            'activation_novelty': activation_novelty * 0.3
        }
        
        novelty_score = sum(novelty_components.values())
        
        # 5. 정규화 (0-1 범위)
        novelty_score = np.clip(novelty_score, 0.0, 1.0)
        
        return float(novelty_score)

    def _calculate_activation_novelty(self, expert_output) -> float:
        """Expert 활성화 패턴의 참신성 계산"""
        # 현재 활성화 점수와 과거 평균 비교
        current_activation = expert_output.activation_score
        expert_id = expert_output.expert_id
        
        # 과거 활성화 점수 기록이 있는 경우
        if expert_id in self.performance_history:
            historical_activations = self.performance_history[expert_id]
            if len(historical_activations) > 0:
                avg_historical_activation = np.mean(historical_activations)
                activation_novelty = abs(current_activation - avg_historical_activation)
                return min(activation_novelty, 1.0)
        
        # 기본값 반환
        return 0.5

    def _adjust_fusion_degree(self, current_degree: float, similarity: float, 
                            novelty: float, activation: float) -> float:
        """
        동적 fusion_degree 조정 공식
        
        조정 원칙:
        - 높은 참신성 + 적당한 유사도 = 높은 fusion_degree
        - 낮은 참신성 + 높은 유사도 = 낮은 fusion_degree
        - 활성화 점수도 고려
        """
        # 1. 기본 조정 인자들
        novelty_factor = novelty * 0.5  # 참신성 가중치
        similarity_factor = (1.0 - similarity) * 0.3  # 차이점 가중치 (유사도가 낮을수록 높은 가중치)
        activation_factor = activation * 0.2  # 활성화 가중치
        
        # 2. 종합 조정 점수
        adjustment_score = novelty_factor + similarity_factor + activation_factor
        
        # 3. 조정 방향 결정
        target_adjustment = (adjustment_score - 0.5) * self.config['adjustment_rate']
        
        # 4. 점진적 조정 (급격한 변화 방지)
        adjustment_with_inertia = target_adjustment * 0.7 + (current_degree - 0.5) * 0.3
        
        # 5. 새로운 degree 계산
        new_degree = current_degree + adjustment_with_inertia
        
        # 6. 범위 제한
        new_degree = np.clip(new_degree, 
                           self.config['min_fusion_degree'], 
                           self.config['max_fusion_degree'])
        
        return float(new_degree)

    def _normalize_fusion_weights(self, fusion_metrics: List[FusionMetrics]) -> Dict[int, float]:
        """Fusion weights 정규화"""
        fusion_weights = {}
        
        # 1. 기본 가중치 계산
        total_weight = 0.0
        for metric in fusion_metrics:
            weight = metric.fusion_degree * metric.activation_weight
            fusion_weights[metric.expert_id] = weight
            total_weight += weight
        
        # 2. 정규화
        if total_weight > 0:
            for expert_id in fusion_weights:
                fusion_weights[expert_id] /= total_weight
        
        return fusion_weights

    def _calculate_adaptive_alpha(self, total_fusion_strength: float, 
                                fusion_metrics: List[FusionMetrics]) -> float:
        """적응적 alpha 값 계산 (Assistant 출력의 영향도)"""
        # 1. 기본 alpha는 전체 fusion 강도에 비례
        base_alpha = min(total_fusion_strength, 0.8)
        
        # 2. 참신성 점수 기반 조정
        avg_novelty = np.mean([m.novelty_score for m in fusion_metrics])
        novelty_boost = avg_novelty * 0.2
        
        # 3. 최종 alpha 계산
        alpha = base_alpha + novelty_boost
        alpha = np.clip(alpha, 0.1, 0.9)
        
        return float(alpha)

    def _spatial_weighted_fusion(self, first_output: torch.Tensor, second_output: torch.Tensor,
                               fusion_weights: Dict[int, float], alpha: float, beta: float) -> torch.Tensor:
        """공간별 가중 조합을 통한 융합"""
        # 1. 기본 가중 조합
        basic_fusion = beta * first_output + alpha * second_output
        
        # 2. Expert별 가중치를 활용한 공간적 조정
        if len(fusion_weights) > 0:
            # 각 위치별로 다른 가중치 적용 (simplified spatial weighting)
            spatial_weights = torch.ones_like(first_output) * beta
            
            # 높은 fusion weight를 가진 Expert들이 영향을 미치는 영역에서 alpha 증가
            max_weight = max(fusion_weights.values()) if fusion_weights else 0.0
            if max_weight > 0:
                spatial_alpha = alpha * (1.0 + max_weight * 0.5)
                spatial_weights = spatial_weights * (1.0 - spatial_alpha) + spatial_alpha
            
            # 공간별 가중 적용
            fused_output = spatial_weights * first_output + (1.0 - spatial_weights) * second_output
        else:
            fused_output = basic_fusion
        
        return fused_output

    def _update_performance_history(self, expert_id: int, novelty_score: float):
        """Expert의 성능 기록 업데이트"""
        if expert_id not in self.performance_history:
            self.performance_history[expert_id] = []
        
        self.performance_history[expert_id].append(novelty_score)
        
        # 기록 길이 제한
        if len(self.performance_history[expert_id]) > self.config['history_length']:
            self.performance_history[expert_id].pop(0)

    def _update_statistics(self, fusion_metrics: List[FusionMetrics]):
        """통계 정보 업데이트"""
        self.stats['total_calculations'] += 1
        
        if fusion_metrics:
            avg_similarity = np.mean([m.similarity_score for m in fusion_metrics])
            avg_novelty = np.mean([m.novelty_score for m in fusion_metrics])
            
            # 이동 평균 업데이트
            decay = self.config['decay_factor']
            self.stats['avg_similarity_score'] = (
                self.stats['avg_similarity_score'] * decay + avg_similarity * (1 - decay)
            )
            self.stats['avg_novelty_score'] = (
                self.stats['avg_novelty_score'] * decay + avg_novelty * (1 - decay)
            )

    def _load_fusion_degrees(self):
        """저장된 fusion degree 로드"""
        try:
            if os.path.exists(self.save_path):
                with open(self.save_path, 'r') as f:
                    data = json.load(f)
                    self.fusion_degrees = {int(k): v for k, v in data.get('fusion_degrees', {}).items()}
                    self.performance_history = {int(k): v for k, v in data.get('performance_history', {}).items()}
                    print(f"Loaded {len(self.fusion_degrees)} fusion degrees from {self.save_path}")
        except Exception as e:
            print(f"Error loading fusion degrees: {e}")

    def save_fusion_degrees(self):
        """현재 fusion degree 저장"""
        try:
            data = {
                'fusion_degrees': self.fusion_degrees,
                'performance_history': self.performance_history,
                'stats': self.stats,
                'config': self.config,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.save_path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Saved fusion degrees to {self.save_path}")
        except Exception as e:
            print(f"Error saving fusion degrees: {e}")

    def get_fusion_statistics(self) -> Dict:
        """현재 fusion 통계 반환"""
        return {
            'total_experts': len(self.fusion_degrees),
            'avg_fusion_degree': np.mean(list(self.fusion_degrees.values())) if self.fusion_degrees else 0.0,
            'min_fusion_degree': min(self.fusion_degrees.values()) if self.fusion_degrees else 0.0,
            'max_fusion_degree': max(self.fusion_degrees.values()) if self.fusion_degrees else 0.0,
            'total_calculations': self.stats['total_calculations'],
            'avg_similarity_score': self.stats['avg_similarity_score'],
            'avg_novelty_score': self.stats['avg_novelty_score'],
            'experts_with_history': len(self.performance_history)
        }

    def reset_fusion_degrees(self):
        """모든 fusion degree 초기화"""
        self.fusion_degrees.clear()
        self.performance_history.clear()
        self.stats = {
            'total_calculations': 0,
            'avg_similarity_score': 0.0,
            'avg_novelty_score': 0.0,
            'fusion_adjustments': 0
        }
        print("All fusion degrees reset to default values")

    def __del__(self):
        """소멸자에서 자동 저장"""
        try:
            self.save_fusion_degrees()
        except:
            pass