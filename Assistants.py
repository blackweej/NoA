"""
Assistant Router Implementation for Dual MoE System
Based on mistral-inference architecture with enhanced routing capabilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from sklearn.cluster import KMeans
import logging

from mistral_inference.generate import generate
from mistral_inference.transformer import Transformer
from mistral_inference.tokenizer import MistralTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ExpertOutput:
    """Expert의 출력 정보를 담는 데이터 클래스"""
    expert_id: int
    output_tensor: torch.Tensor
    weight: float
    activation_score: float
    fusion_degree: float = 1.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MaxExpertsList:
    """최대 활성화 Expert들의 리스트"""
    max_count: int = 8
    experts: List[ExpertOutput] = field(default_factory=list)
    threshold: float = 0.1
    
    def add_expert(self, expert_output: ExpertOutput):
        """Expert 추가 (정렬된 상태 유지)"""
        if expert_output.activation_score >= self.threshold:
            self.experts.append(expert_output)
            self.experts.sort(key=lambda x: x.activation_score, reverse=True)
            if len(self.experts) > self.max_count:
                self.experts = self.experts[:self.max_count]
    
    def get_top_experts(self, count: int) -> List[ExpertOutput]:
        """상위 N개 Expert 반환"""
        return self.experts[:min(count, len(self.experts))]


@dataclass
class AssistantConfig:
    """Assistant의 설정 정보"""
    assistant_id: str
    prompt_template: str
    classification_threshold: float
    expert_affinity: Dict[int, float] = field(default_factory=dict)
    specialization_area: str = ""
    activation_pattern: List[float] = field(default_factory=list)
    
    def calculate_affinity_score(self, expert_id: int) -> float:
        """특정 Expert와의 친화도 점수 계산"""
        return self.expert_affinity.get(expert_id, 0.0)


@dataclass
class AssistantOutput:
    """Assistant의 출력 정보"""
    assistant_id: str
    comment_text: str
    confidence_score: float
    related_experts: List[int]
    generation_time: float = 0.0
    token_count: int = 0
    embedding_vector: Optional[torch.Tensor] = None
    
    def __post_init__(self):
        self.token_count = len(self.comment_text.split())


class AssistantRouter:
    """
    Assistant 라우터 시스템
    max_experts를 기반으로 동적으로 Assistant를 생성하고 관리
    """
    
    def __init__(self, 
                 tokenizer: MistralTokenizer,
                 model: Transformer,
                 max_assistants: int = 5,
                 comment_max_tokens: int = 256,
                 temperature: float = 0.7,
                 clustering_method: str = 'kmeans'):
        
        self.tokenizer = tokenizer
        self.model = model
        self.max_assistants = max_assistants
        self.comment_max_tokens = comment_max_tokens
        self.temperature = temperature
        self.clustering_method = clustering_method
        
        # 동적으로 생성된 Assistant들 저장
        self.active_assistants: Dict[str, AssistantConfig] = {}
        self.assistant_history: List[AssistantOutput] = []
        
        # Assistant 생성을 위한 프롬프트 템플릿들
        self.base_prompt_templates = {
            'analytical': "당신은 분석적 사고를 전문으로 하는 Assistant입니다. 주어진 질문에 대해 논리적이고 체계적인 관점에서 보완적인 코멘트를 제공해주세요.",
            'creative': "당신은 창의적 사고를 전문으로 하는 Assistant입니다. 주어진 질문에 대해 독창적이고 혁신적인 관점에서 보완적인 코멘트를 제공해주세요.",
            'practical': "당신은 실용적 접근을 전문으로 하는 Assistant입니다. 주어진 질문에 대해 현실적이고 적용 가능한 관점에서 보완적인 코멘트를 제공해주세요.",
            'critical': "당신은 비판적 사고를 전문으로 하는 Assistant입니다. 주어진 질문에 대해 문제점을 찾고 개선점을 제시하는 관점에서 보완적인 코멘트를 제공해주세요.",
            'synthetic': "당신은 종합적 사고를 전문으로 하는 Assistant입니다. 주어진 질문에 대해 다양한 관점을 통합하여 보완적인 코멘트를 제공해주세요."
        }
        
        logger.info(f"AssistantRouter initialized with {max_assistants} max assistants")
    
    def bridge_through_assistants(self,
                            original_question: str,
                            first_output_text: str,
                            max_experts: MaxExpertsList) -> List[AssistantOutput]:
        """
        Assistant 시스템의 메인 처리 함수
        
        Args:
            original_question: 원본 질문
            first_output_text: 첫 번째 MoE 출력
            max_experts: 최대 활성화 Expert 리스트
            
        Returns:
            assistant_outputs: Assistant들의 출력 리스트
        """
        start_time = datetime.now()
        
        try:
            # 1. 동적 Assistant 생성
            assistants = self._create_assistants(max_experts)
            logger.info(f"Created {len(assistants)} assistants based on expert patterns")
            
            # 2. 각 Assistant별 코멘트 생성
            assistant_outputs = []
            for assistant in assistants:
                try:
                    # 코멘트 생성 시간 측정
                    comment_start = datetime.now()
                    
                    # 프롬프트 구성
                    prompt = self._build_comment_prompt(
                        assistant.prompt_template,
                        original_question,
                        first_output_text,
                        assistant.expert_affinity
                    )
                    
                    # Generate 함수 호출
                    comment_output = self._generate_comment(prompt, assistant.assistant_id)
                    
                    # 신뢰도 점수 계산
                    confidence_score = self._calculate_confidence(comment_output, assistant.expert_affinity)
                    
                    # 관련 Expert 식별
                    related_experts = [
                        expert_id for expert_id, affinity in assistant.expert_affinity.items()
                        if affinity > assistant.classification_threshold
                    ]
                    
                    # Assistant 출력 생성
                    generation_time = (datetime.now() - comment_start).total_seconds()
                    
                    assistant_output = AssistantOutput(
                        assistant_id=assistant.assistant_id,
                        comment_text=comment_output,
                        confidence_score=confidence_score,
                        related_experts=related_experts,
                        generation_time=generation_time
                    )
                    
                    assistant_outputs.append(assistant_output)
                    logger.info(f"Generated comment from {assistant.assistant_id} in {generation_time:.2f}s")
                    
                except Exception as e:
                    logger.error(f"Error generating comment for {assistant.assistant_id}: {str(e)}")
                    continue
            
            # 3. 출력 품질 필터링
            filtered_outputs = self._filter_assistant_outputs(assistant_outputs)
            
            # 4. 히스토리 업데이트
            self.assistant_history.extend(filtered_outputs)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Assistant processing completed in {processing_time:.2f}s")
            
            return filtered_outputs
            
        except Exception as e:
            logger.error(f"Error in bridge_through_assistants: {str(e)}")
            return []
    
    def _create_assistants(self, max_experts: MaxExpertsList) -> List[AssistantConfig]:
        """
        max_experts 기반으로 동적 Assistant 생성
        
        Args:
            max_experts: 최대 활성화 Expert 리스트
            
        Returns:
            assistants: 생성된 Assistant 설정 리스트
        """
        try:
            if not max_experts.experts:
                logger.warning("No experts available for assistant creation")
                return []
            
            # 1. Expert 활성화 패턴 분석
            expert_patterns = self._analyze_expert_patterns(max_experts)
            
            # 2. 클러스터링을 통한 Expert 그룹핑
            expert_clusters = self._cluster_experts_by_activation(max_experts)
            
            assistants = []
            template_keys = list(self.base_prompt_templates.keys())
            
            # 3. 각 클러스터별로 Assistant 생성
            for cluster_id, expert_group in expert_clusters.items():
                if not expert_group:
                    continue
                    
                # 클러스터 특성 분석
                cluster_characteristics = self._analyze_cluster_characteristics(expert_group)
                
                # 적합한 템플릿 선택
                template_key = template_keys[cluster_id % len(template_keys)]
                
                assistant_config = AssistantConfig(
                    assistant_id=f"assistant_{cluster_id}_{template_key}",
                    prompt_template=self.base_prompt_templates[template_key],
                    classification_threshold=self._calculate_dynamic_threshold(expert_group),
                    specialization_area=template_key,
                    activation_pattern=cluster_characteristics['pattern']
                )
                
                # 4. Expert와의 친화도 계산
                for expert_output in expert_group:
                    affinity_score = self._calculate_expert_affinity(
                        expert_output, 
                        cluster_characteristics
                    )
                    assistant_config.expert_affinity[expert_output.expert_id] = affinity_score
                
                assistants.append(assistant_config)
                
                # 활성 Assistant 저장
                self.active_assistants[assistant_config.assistant_id] = assistant_config
                
                logger.info(f"Created assistant {assistant_config.assistant_id} for cluster {cluster_id}")
            
            return assistants[:self.max_assistants]
            
        except Exception as e:
            logger.error(f"Error creating assistants: {str(e)}")
            return []
    
    def _analyze_expert_patterns(self, max_experts: MaxExpertsList) -> Dict[str, Any]:
        """Expert들의 활성화 패턴 분석"""
        if not max_experts.experts:
            return {}
        
        activation_scores = [expert.activation_score for expert in max_experts.experts]
        weights = [expert.weight for expert in max_experts.experts]
        
        patterns = {
            'mean_activation': np.mean(activation_scores),
            'std_activation': np.std(activation_scores),
            'max_activation': np.max(activation_scores),
            'min_activation': np.min(activation_scores),
            'mean_weight': np.mean(weights),
            'activation_distribution': self._categorize_activation_distribution(activation_scores),
            'dominant_experts': [expert.expert_id for expert in max_experts.get_top_experts(3)]
        }
        
        return patterns
    
    def _cluster_experts_by_activation(self, max_experts: MaxExpertsList) -> Dict[int, List[ExpertOutput]]:
        """활성화 점수 기반 Expert 클러스터링"""
        if len(max_experts.experts) <= 1:
            return {0: max_experts.experts}
        
        # 특성 벡터 생성 (activation_score, weight, fusion_degree)
        features = []
        for expert in max_experts.experts:
            features.append([
                expert.activation_score,
                expert.weight,
                expert.fusion_degree
            ])
        
        features_array = np.array(features)
        
        # 클러스터 수 동적 결정
        n_clusters = min(self.max_assistants, len(max_experts.experts))
        
        if self.clustering_method == 'kmeans':
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features_array)
        else:
            # 간단한 임계값 기반 클러스터링
            cluster_labels = self._threshold_based_clustering(features_array, n_clusters)
        
        # 클러스터별 Expert 그룹핑
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(max_experts.experts[i])
        
        return clusters
    
    def _threshold_based_clustering(self, features: np.ndarray, n_clusters: int) -> np.ndarray:
        """임계값 기반 간단한 클러스터링"""
        activation_scores = features[:, 0]
        sorted_indices = np.argsort(activation_scores)[::-1]  # 내림차순 정렬
        
        cluster_labels = np.zeros(len(features), dtype=int)
        experts_per_cluster = len(features) // n_clusters
        remainder = len(features) % n_clusters
        
        start_idx = 0
        for cluster_id in range(n_clusters):
            end_idx = start_idx + experts_per_cluster + (1 if cluster_id < remainder else 0)
            cluster_indices = sorted_indices[start_idx:end_idx]
            cluster_labels[cluster_indices] = cluster_id
            start_idx = end_idx
        
        return cluster_labels
    
    def _analyze_cluster_characteristics(self, expert_group: List[ExpertOutput]) -> Dict[str, Any]:
        """클러스터 특성 분석"""
        if not expert_group:
            return {'pattern': []}
        
        activation_scores = [expert.activation_score for expert in expert_group]
        weights = [expert.weight for expert in expert_group]
        
        characteristics = {
            'pattern': activation_scores,
            'avg_activation': np.mean(activation_scores),
            'activation_variance': np.var(activation_scores),
            'weight_distribution': weights,
            'cluster_size': len(expert_group),
            'dominant_expert': max(expert_group, key=lambda x: x.activation_score).expert_id,
            'activation_range': np.max(activation_scores) - np.min(activation_scores)
        }
        
        return characteristics
    
    def _calculate_dynamic_threshold(self, expert_group: List[ExpertOutput]) -> float:
        """클러스터 특성에 따른 동적 임계값 계산"""
        if not expert_group:
            return 0.1
        
        activation_scores = [expert.activation_score for expert in expert_group]
        mean_activation = np.mean(activation_scores)
        std_activation = np.std(activation_scores)
        
        # 표준편차가 큰 경우 더 높은 임계값 사용
        dynamic_threshold = max(0.05, mean_activation - 0.5 * std_activation)
        
        return min(dynamic_threshold, 0.8)  # 최대 0.8로 제한
    
    def _calculate_expert_affinity(self, 
                                 expert_output: ExpertOutput,
                                 cluster_characteristics: Dict[str, Any]) -> float:
        """Expert와 클러스터 간의 친화도 계산"""
        # 활성화 점수 기반 친화도
        activation_affinity = expert_output.activation_score / cluster_characteristics['avg_activation']
        
        # 가중치 기반 친화도
        weight_affinity = expert_output.weight
        
        # Fusion degree 기반 친화도
        fusion_affinity = expert_output.fusion_degree
        
        # 가중 평균으로 최종 친화도 계산
        total_affinity = (
            0.4 * activation_affinity +
            0.3 * weight_affinity +
            0.3 * fusion_affinity
        )
        
        return min(total_affinity, 1.0)
    
    def _categorize_activation_distribution(self, activation_scores: List[float]) -> str:
        """활성화 점수 분포 특성 분류"""
        if not activation_scores:
            return "empty"
        
        mean_score = np.mean(activation_scores)
        std_score = np.std(activation_scores)
        
        if std_score < 0.1:
            return "uniform"
        elif mean_score > 0.7:
            return "high_activation"
        elif mean_score < 0.3:
            return "low_activation"
        else:
            return "mixed"
    
    def _build_comment_prompt(self,
                            template: str,
                            original_question: str,
                            first_output_text: str, # 입력 파라미터 변경: torch.Tensor -> str [cite: 14]
                            expert_affinity: Dict[int, float]) -> str:
        """코멘트 생성을 위한 프롬프트 구성"""
        
        top_experts = sorted(expert_affinity.items(), key=lambda x: x[1], reverse=True)[:3]
        expert_info = ", ".join([f"Expert{expert_id}(친화도:{affinity:.2f})" 
                               for expert_id, affinity in top_experts])
        
        prompt = f"""
{template}

원본 질문: {original_question}

첫 번째 답변: {first_output_text} # 수정: 텐서 요약 대신 디코딩된 텍스트 사용 [cite: 15]

관련 Expert 정보: {expert_info}

위 정보를 바탕으로 원본 질문에 대한 보완적 관점의 코멘트를 생성해주세요.
코멘트는 구체적이고 실용적이어야 하며, 원본 답변과 차별화된 새로운 통찰을 제공해야 합니다.

코멘트:"""
        
        return prompt
    
    def _generate_comment(self, prompt: str, assistant_id: str) -> str:
        """실제 Generate 함수 호출을 통한 코멘트 생성"""
        try:
            # 1. 프롬프트 토크나이징
            tokenized_prompt = self.tokenizer.encode(prompt)
            
            # 2. Generate 함수 호출
            output_tokens, _ = generate(
                [tokenized_prompt],
                self.model,
                max_tokens=self.comment_max_tokens,
                temperature=self.temperature,
                eos_id=self.tokenizer.eos_id
            )
            
            # 3. 디코딩
            raw_comment = self.tokenizer.decode(output_tokens[0])
            
            # 4. 코멘트 후처리
            processed_comment = self._post_process_comment(raw_comment)
            
            # 5. 포맷팅
            formatted_comment = self._format_as_comment(processed_comment, assistant_id)
            
            return formatted_comment
            
        except Exception as e:
            logger.error(f"Error generating comment for {assistant_id}: {str(e)}")
            return f"[{assistant_id}] 코멘트 생성 중 오류가 발생했습니다."
    
    def _post_process_comment(self, raw_comment: str) -> str:
        """생성된 코멘트 후처리"""
        # 불필요한 토큰 제거
        comment = raw_comment.strip()
        
        # 프롬프트 반복 제거
        if "코멘트:" in comment:
            comment = comment.split("코멘트:")[-1].strip()
        
        # 길이 제한
        if len(comment) > 500:
            comment = comment[:500] + "..."
        
        # 빈 코멘트 처리
        if not comment:
            comment = "추가적인 관점에서 고려해볼 점이 있습니다."
        
        return comment
    
    def _format_as_comment(self, comment_text: str, assistant_id: str) -> str:
        """코멘트를 지정된 형식으로 포맷팅"""
        return f"[{assistant_id}] 원래 질문에 대한 보완적 관점: {comment_text}"
    
    def _calculate_confidence(self, comment_text: str, expert_affinity: Dict[int, float]) -> float:
        """코멘트의 신뢰도 점수 계산"""
        # 1. 텍스트 길이 기반 신뢰도
        length_score = min(len(comment_text.split()) / 50, 1.0)
        
        # 2. Expert 친화도 기반 신뢰도
        affinity_score = np.mean(list(expert_affinity.values())) if expert_affinity else 0.0
        
        # 3. 키워드 기반 신뢰도 (간단한 휴리스틱)
        confidence_keywords = ['분석', '고려', '관점', '제안', '방법', '접근', '개선']
        keyword_score = sum(1 for keyword in confidence_keywords if keyword in comment_text) / len(confidence_keywords)
        
        # 4. 종합 신뢰도 계산
        total_confidence = (
            0.3 * length_score +
            0.4 * affinity_score +
            0.3 * keyword_score
        )
        
        return min(total_confidence, 1.0)
    
    def _filter_assistant_outputs(self, assistant_outputs: List[AssistantOutput]) -> List[AssistantOutput]:
        """Assistant 출력 품질 필터링"""
        if not assistant_outputs:
            return []
        
        # 1. 신뢰도 점수 기반 필터링
        min_confidence = 0.3
        filtered_outputs = [
            output for output in assistant_outputs 
            if output.confidence_score >= min_confidence
        ]
        
        # 2. 중복 제거 (유사한 코멘트 필터링)
        unique_outputs = self._remove_duplicate_comments(filtered_outputs)
        
        # 3. 상위 N개 선택
        sorted_outputs = sorted(unique_outputs, key=lambda x: x.confidence_score, reverse=True)
        
        return sorted_outputs[:self.max_assistants]
    
    def _remove_duplicate_comments(self, outputs: List[AssistantOutput]) -> List[AssistantOutput]:
        """중복 코멘트 제거"""
        if len(outputs) <= 1:
            return outputs
        
        unique_outputs = []
        seen_comments = set()
        
        for output in outputs:
            # 간단한 중복 검사 (실제로는 더 정교한 유사도 검사 필요)
            comment_key = output.comment_text[:100].lower()  # 첫 100자를 키로 사용
            
            if comment_key not in seen_comments:
                seen_comments.add(comment_key)
                unique_outputs.append(output)
        
        return unique_outputs
    
    def get_assistant_statistics(self) -> Dict[str, Any]:
        """Assistant 시스템 통계 정보"""
        if not self.assistant_history:
            return {"message": "No assistant history available"}
        
        total_outputs = len(self.assistant_history)
        avg_confidence = np.mean([output.confidence_score for output in self.assistant_history])
        avg_generation_time = np.mean([output.generation_time for output in self.assistant_history])
        
        assistant_usage = {}
        for output in self.assistant_history:
            assistant_id = output.assistant_id
            if assistant_id not in assistant_usage:
                assistant_usage[assistant_id] = 0
            assistant_usage[assistant_id] += 1
        
        return {
            "total_outputs": total_outputs,
            "average_confidence": avg_confidence,
            "average_generation_time": avg_generation_time,
            "assistant_usage": assistant_usage,
            "active_assistants": len(self.active_assistants)
        }
    
    def update_assistant_fusion_degrees(self, feedback_data: Dict[str, float]):
        """피드백 기반 Assistant의 fusion degree 업데이트"""
        for assistant_id, assistant_config in self.active_assistants.items():
            if assistant_id in feedback_data:
                feedback_score = feedback_data[assistant_id]
                
                # 피드백 점수에 따른 expert affinity 조정
                adjustment_factor = (feedback_score - 0.5) * 0.1  # -0.05 ~ 0.05 범위
                
                for expert_id in assistant_config.expert_affinity:
                    current_affinity = assistant_config.expert_affinity[expert_id]
                    new_affinity = max(0.0, min(1.0, current_affinity + adjustment_factor))
                    assistant_config.expert_affinity[expert_id] = new_affinity
                
                logger.info(f"Updated fusion degrees for {assistant_id} based on feedback: {feedback_score}")
    
    def reset_assistants(self):
        """Assistant 상태 초기화"""
        self.active_assistants.clear()
        self.assistant_history.clear()
        logger.info("Assistant system reset completed")


# 사용 예시 및 테스트 함수
def test_assistant_router():
    """Assistant Router 테스트 함수"""
    # 이는 실제 사용 시 적절한 tokenizer와 model로 대체되어야 함
    print("AssistantRouter 테스트 시작...")
    
    # Mock 데이터 생성
    mock_experts = MaxExpertsList(max_count=6)
    for i in range(6):
        expert = ExpertOutput(
            expert_id=i,
            output_tensor=torch.randn(1, 10, 512),
            weight=np.random.random(),
            activation_score=np.random.random(),
            fusion_degree=np.random.random()
        )
        mock_experts.add_expert(expert)
    
    print(f"Mock experts 생성 완료: {len(mock_experts.experts)}개")
    
    # 실제 사용 시에는 아래와 같이 초기화
    # tokenizer = MistralTokenizer.from_file("tokenizer.model")
    # model = Transformer.from_folder("model_path")
    # router = AssistantRouter(tokenizer, model)
    
    print("AssistantRouter 테스트 완료")


if __name__ == "__main__":
    test_assistant_router()