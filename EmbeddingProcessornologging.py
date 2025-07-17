"""
EmbeddingProcessor for Dual MoE System
Assistant 출력을 벡터화하고 Expert 시스템으로 라우팅하는 핵심 모듈

기존 mistral-inference 구조를 활용하여 임베딩 처리 및 라우팅 기능 제공
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from datetime import datetime

# mistral-inference 모듈들
from mistral_inference.transformer import Transformer
from mistral_inference.tokenizer import MistralTokenizer


@dataclass
class AssistantOutput:
    """Assistant 시스템의 출력 구조"""
    assistant_id: str
    comment_text: str
    confidence_score: float
    related_experts: List[int]
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class EmbeddedAssistantOutput:
    """임베딩 처리된 Assistant 출력"""
    assistant_output: AssistantOutput
    embedding_vector: torch.Tensor  # [hidden_dim]
    similarity_scores: Dict[int, float]  # expert_id: similarity_score
    routing_weights: Dict[int, float]  # expert_id: routing_weight


@dataclass
class ExpertOutput:
    """Expert 출력 구조 (기존 시스템과 호환)"""
    expert_id: int
    output_tensor: torch.Tensor
    weight: float
    activation_score: float
    fusion_degree: float = 1.0


@dataclass
class MaxExpertsList:
    """최대 활성화 Expert 리스트"""
    experts: List[ExpertOutput]
    max_count: int = 8
    threshold: float = 0.1


class EmbeddingProcessor:
    """
    Assistant 출력을 벡터화하고 Expert 시스템으로 라우팅하는 프로세서
    """
    
    def __init__(self, model: Transformer, tokenizer: MistralTokenizer, 
                 hidden_dim: int = 4096, similarity_threshold: float = 0.1):
        """
        Args:
            model: Mistral Transformer 모델
            tokenizer: Mistral 토크나이저
            hidden_dim: 모델의 hidden dimension
            similarity_threshold: 유사도 계산 임계값
        """
        self.model = model
        self.tokenizer = tokenizer
        self.hidden_dim = hidden_dim
        self.similarity_threshold = similarity_threshold
        
        # 임베딩 캐시 (성능 최적화)
        self.embedding_cache = {}
        
        # --- 수정된 부분 ---
        # Expert 임베딩을 더 이상 빈 딕셔너리로 시작하지 않습니다. [cite: 25]
        self.expert_embeddings = self._precompute_expert_embeddings()
        # --- 수정 끝 ---
        
        # 라우팅 히스토리 (분석용)
        self.routing_history = []
        
        print(f"EmbeddingProcessor initialized with hidden_dim={hidden_dim}")
    
    def _precompute_expert_embeddings(self) -> Dict[int, torch.Tensor]:
        """
        모델의 모든 Expert에 대한 대표 임베딩을 미리 계산합니다.
        Expert의 가중치 평균을 사용하여 정체성 벡터를 생성합니다. [cite: 27]
        """
        expert_embeddings = {}
        with torch.no_grad():
            # 모델의 모든 레이어를 순회
            for layer in self.model.layers:
                if hasattr(layer, 'feed_forward') and hasattr(layer.feed_forward, 'experts'):
                    # 각 Expert의 가중치를 평균내어 대표 벡터로 사용 [cite: 28]
                    for i, expert_module in enumerate(layer.feed_forward.experts):
                        # w1, w2, w3 등 모든 파라미터 텐서를 가져옴
                        param_vectors = [p.view(-1) for p in expert_module.parameters()]
                        
                        if not param_vectors:
                            continue
                            
                        # 모든 파라미터를 이어 붙인 후 평균내어 하나의 벡터로 만듦
                        # 주의: Mixtral의 경우 모든 파라미터를 cat하는 것은 매우 큰 텐서를 만듭니다.
                        # 여기서는 각 파라미터의 평균을 먼저 구하고, 그 벡터들의 평균을 내는 방식으로 변경하여 메모리 효율을 높일 수 있습니다.
                        # 아래는 원본 코드의 의도를 살린 구현입니다.
                        avg_vector = torch.cat(param_vectors).mean(dim=0, keepdim=True)
                        
                        # Mixtral은 모든 레이어의 Expert가 동일한 역할을 하므로, 레이어별 구분 없이 Expert 인덱스(i)를 사용합니다. [cite: 31, 32]
                        if i not in expert_embeddings:
                            # 정규화하여 저장
                            expert_embeddings[i] = F.normalize(avg_vector, p=2, dim=-1).squeeze()
        return expert_embeddings
    
    
    def vectorize_assistant_output(self, assistant_outputs: List[AssistantOutput]) -> List[EmbeddedAssistantOutput]:
        """
        Assistant 출력들을 벡터화
        
        Args:
            assistant_outputs: Assistant 시스템의 출력 리스트
            
        Returns:
            embedded_outputs: 임베딩 처리된 출력 리스트
        """
        embedded_outputs = []
        
        print(f"Processing {len(assistant_outputs)} assistant outputs for embedding...")
        
        for i, assistant_output in enumerate(assistant_outputs):
            try:
                # 1. 텍스트 임베딩 생성
                embedding_vector = self._create_embedding(assistant_output.comment_text)
                
                # 2. 관련 Expert들과의 유사도 계산
                similarity_scores = self._calculate_expert_similarities(
                    embedding_vector, 
                    assistant_output.related_experts
                )
                
                # 3. 라우팅 가중치 계산
                routing_weights = self._calculate_routing_weights(
                    similarity_scores, 
                    assistant_output.confidence_score
                )
                
                # 4. 결과 구성
                embedded_output = EmbeddedAssistantOutput(
                    assistant_output=assistant_output,
                    embedding_vector=embedding_vector,
                    similarity_scores=similarity_scores,
                    routing_weights=routing_weights
                )
                
                embedded_outputs.append(embedded_output)
                
                print(f"  Assistant {i+1}/{len(assistant_outputs)}: "
                      f"embedding_dim={embedding_vector.shape}, "
                      f"similarities={len(similarity_scores)}, "
                      f"routing_weights={len(routing_weights)}")
                
            except Exception as e:
                print(f"Error processing assistant output {i}: {e}")
                continue
        
        return embedded_outputs
    
    def route_to_experts(self, embedded_outputs: List[EmbeddedAssistantOutput], 
                        max_experts: MaxExpertsList) -> torch.Tensor:
        """
        임베딩된 Assistant 출력을 Expert 시스템으로 라우팅
        
        Args:
            embedded_outputs: 임베딩 처리된 Assistant 출력들
            max_experts: 최대 활성화 Expert 리스트
            
        Returns:
            routing_vector: Expert 시스템으로 전달될 라우팅 벡터 [1, 1, hidden_dim]
        """
        if not embedded_outputs:
            print("No embedded outputs to route, returning zero vector")
            return torch.zeros(1, 1, self.hidden_dim)
        
        print(f"Routing {len(embedded_outputs)} embedded outputs to {len(max_experts.experts)} experts...")
        
        # 1. 모든 Assistant 임베딩을 가중평균으로 결합
        combined_embedding = self._weighted_combine_embeddings(embedded_outputs)
        
        # 2. Expert 친화도 기반 가중치 조정
        expert_weights = self._calculate_expert_routing_weights(
            combined_embedding, 
            max_experts
        )
        
        # 3. 라우팅 벡터 생성 (batch_size=1, seq_len=1을 가정)
        routing_vector = self._create_routing_vector(combined_embedding, expert_weights)
        
        # 4. 라우팅 히스토리 저장
        self._save_routing_history(embedded_outputs, expert_weights, routing_vector)
        
        print(f"Generated routing vector: shape={routing_vector.shape}, "
              f"norm={torch.norm(routing_vector).item():.4f}")
        
        return routing_vector
    
    def _create_embedding(self, text: str) -> torch.Tensor:
        """
        텍스트를 임베딩 벡터로 변환
        
        Args:
            text: 임베딩할 텍스트
            
        Returns:
            embedding: 문장 임베딩 벡터 [hidden_dim]
        """
        # 캐시 확인
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        try:
            # 1. 토크나이제이션
            tokens = self.tokenizer.encode(text, add_bos=True, add_eos=True)
            
            # 2. 텐서 변환
            token_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)  # [1, seq_len]
            
            # 3. 모델의 임베딩 레이어 사용
            with torch.no_grad():
                # mistral-inference의 embed_tokens 사용
                token_embeddings = self.model.embed_tokens(token_tensor)  # [1, seq_len, hidden_dim]
                
                # 4. 평균 풀링으로 문장 임베딩 생성 (패딩 토큰 제외)
                # 실제 토큰 길이 계산 (BOS, EOS 포함)
                valid_length = len(tokens)
                sentence_embedding = token_embeddings[0, :valid_length].mean(dim=0)  # [hidden_dim]
                
                # 5. 정규화
                sentence_embedding = F.normalize(sentence_embedding, p=2, dim=0)
            
            # 캐시 저장
            self.embedding_cache[text] = sentence_embedding
            
            return sentence_embedding
            
        except Exception as e:
            print(f"Error creating embedding for text: {e}")
            # 오류 시 영벡터 반환
            return torch.zeros(self.hidden_dim)
    
    def _calculate_expert_similarities(self, embedding_vector: torch.Tensor, 
                                     related_experts: List[int]) -> Dict[int, float]:
        """
        임베딩 벡터와 관련 Expert들 간의 유사도 계산
        
        Args:
            embedding_vector: 임베딩 벡터 [hidden_dim]
            related_experts: 관련 Expert ID 리스트
            
        Returns:
            similarity_scores: expert_id: similarity_score 매핑
        """
        similarity_scores = {}
        
        for expert_id in related_experts:
            # Expert 임베딩 벡터 가져오기 (없으면 랜덤 생성)
            expert_embedding = self._get_expert_embedding(expert_id)
            
            # 코사인 유사도 계산
            similarity = F.cosine_similarity(
                embedding_vector.unsqueeze(0), 
                expert_embedding.unsqueeze(0)
            ).item()
            
            # 임계값 적용
            if similarity >= self.similarity_threshold:
                similarity_scores[expert_id] = similarity
        
        return similarity_scores
    
    def _calculate_routing_weights(self, similarity_scores: Dict[int, float], 
                                 confidence_score: float) -> Dict[int, float]:
        """
        유사도와 신뢰도 기반 라우팅 가중치 계산
        
        Args:
            similarity_scores: Expert 유사도 점수들
            confidence_score: Assistant 신뢰도 점수
            
        Returns:
            routing_weights: expert_id: routing_weight 매핑
        """
        routing_weights = {}
        
        if not similarity_scores:
            return routing_weights
        
        # 1. 신뢰도 가중치 적용
        for expert_id, similarity in similarity_scores.items():
            weighted_score = similarity * confidence_score
            routing_weights[expert_id] = weighted_score
        
        # 2. 소프트맥스 정규화
        if routing_weights:
            scores = list(routing_weights.values())
            softmax_scores = F.softmax(torch.tensor(scores), dim=0)
            
            for i, expert_id in enumerate(routing_weights.keys()):
                routing_weights[expert_id] = softmax_scores[i].item()
        
        return routing_weights
    
    def _weighted_combine_embeddings(self, embedded_outputs: List[EmbeddedAssistantOutput]) -> torch.Tensor:
        """
        여러 Assistant 임베딩을 가중평균으로 결합
        
        Args:
            embedded_outputs: 임베딩 처리된 출력들
            
        Returns:
            combined_embedding: 결합된 임베딩 벡터 [hidden_dim]
        """
        if not embedded_outputs:
            return torch.zeros(self.hidden_dim)
        
        # 1. 신뢰도 기반 가중치 계산
        weights = []
        embeddings = []
        
        for embedded_output in embedded_outputs:
            weight = embedded_output.assistant_output.confidence_score
            weights.append(weight)
            embeddings.append(embedded_output.embedding_vector)
        
        # 2. 가중치 정규화
        weights = torch.tensor(weights)
        weights = F.softmax(weights, dim=0)
        
        # 3. 가중평균 계산
        combined_embedding = torch.zeros(self.hidden_dim)
        for i, embedding in enumerate(embeddings):
            combined_embedding += weights[i] * embedding
        
        # 4. 정규화
        combined_embedding = F.normalize(combined_embedding, p=2, dim=0)
        
        return combined_embedding
    
    def _calculate_expert_routing_weights(self, combined_embedding: torch.Tensor, 
                                        max_experts: MaxExpertsList) -> Dict[int, float]:
        """
        결합된 임베딩을 기반으로 Expert 라우팅 가중치 계산
        
        Args:
            combined_embedding: 결합된 임베딩 벡터
            max_experts: 최대 활성화 Expert 리스트
            
        Returns:
            expert_weights: expert_id: weight 매핑
        """
        expert_weights = {}
        
        for expert_output in max_experts.experts:
            expert_id = expert_output.expert_id
            
            # Expert 임베딩과의 유사도 계산
            expert_embedding = self._get_expert_embedding(expert_id)
            similarity = F.cosine_similarity(
                combined_embedding.unsqueeze(0),
                expert_embedding.unsqueeze(0)
            ).item()
            
            # 기존 activation_score와 결합
            combined_score = (similarity + expert_output.activation_score) / 2.0
            
            # fusion_degree 적용
            final_weight = combined_score * expert_output.fusion_degree
            
            expert_weights[expert_id] = final_weight
        
        # 소프트맥스 정규화
        if expert_weights:
            scores = list(expert_weights.values())
            softmax_scores = F.softmax(torch.tensor(scores), dim=0)
            
            for i, expert_id in enumerate(expert_weights.keys()):
                expert_weights[expert_id] = softmax_scores[i].item()
        
        return expert_weights
    
    def _create_routing_vector(self, combined_embedding: torch.Tensor, 
                             expert_weights: Dict[int, float]) -> torch.Tensor:
        """
        라우팅 벡터 생성 (MoE 시스템 입력 형태)
        
        Args:
            combined_embedding: 결합된 임베딩 벡터
            expert_weights: Expert 가중치들
            
        Returns:
            routing_vector: [1, 1, hidden_dim] 형태의 라우팅 벡터
        """
        # 1. Expert 가중치 기반 임베딩 조정
        if expert_weights:
            # 가중치를 사용해 임베딩 스케일링
            total_weight = sum(expert_weights.values())
            scaled_embedding = combined_embedding * total_weight
        else:
            scaled_embedding = combined_embedding
        
        # 2. MoE 입력 형태로 변환 [batch_size=1, seq_len=1, hidden_dim]
        routing_vector = scaled_embedding.unsqueeze(0).unsqueeze(0)
        
        return routing_vector
    
    def _get_expert_embedding(self, expert_id: int) -> torch.Tensor:
        """
        Expert ID에 해당하는 임베딩 벡터 가져오기
        
        Args:
            expert_id: Expert ID
            
        Returns:
            expert_embedding: Expert 임베딩 벡터 [hidden_dim]
        """
        if expert_id not in self.expert_embeddings:
            print(f"Warning: Expert ID {expert_id}에 대한 사전 계산된 임베딩이 없습니다. 영벡터를 반환합니다.") [cite: 35]
            return torch.zeros(self.hidden_dim)
        
        return self.expert_embeddings[expert_id]
    
    def _save_routing_history(self, embedded_outputs: List[EmbeddedAssistantOutput], 
                            expert_weights: Dict[int, float], 
                            routing_vector: torch.Tensor):
        """
        라우팅 히스토리 저장 (분석 및 디버깅용)
        
        Args:
            embedded_outputs: 임베딩 출력들
            expert_weights: Expert 가중치들
            routing_vector: 라우팅 벡터
        """
        history_entry = {
            'timestamp': datetime.now(),
            'num_assistants': len(embedded_outputs),
            'expert_weights': expert_weights.copy(),
            'routing_norm': torch.norm(routing_vector).item(),
            'assistant_ids': [out.assistant_output.assistant_id for out in embedded_outputs]
        }
        
        self.routing_history.append(history_entry)
        
        # 히스토리 크기 제한 (메모리 관리)
        if len(self.routing_history) > 1000:
            self.routing_history = self.routing_history[-1000:]
    
    def get_routing_statistics(self) -> Dict:
        """
        라우팅 통계 정보 반환
        
        Returns:
            stats: 라우팅 통계 딕셔너리
        """
        if not self.routing_history:
            return {}
        
        # 통계 계산
        recent_history = self.routing_history[-100:]  # 최근 100개
        
        expert_usage = {}
        for entry in recent_history:
            for expert_id, weight in entry['expert_weights'].items():
                if expert_id not in expert_usage:
                    expert_usage[expert_id] = []
                expert_usage[expert_id].append(weight)
        
        stats = {
            'total_routings': len(self.routing_history),
            'cache_size': len(self.embedding_cache),
            'expert_embeddings': len(self.expert_embeddings),
            'expert_usage_stats': {
                expert_id: {
                    'count': len(weights),
                    'avg_weight': np.mean(weights),
                    'std_weight': np.std(weights)
                } for expert_id, weights in expert_usage.items()
            }
        }
        
        return stats
    
    def clear_cache(self):
        """캐시 및 히스토리 정리"""
        self.embedding_cache.clear()
        self.routing_history.clear()
        print("EmbeddingProcessor cache cleared")


# 사용 예제 함수들
def create_test_assistant_outputs() -> List[AssistantOutput]:
    """테스트용 Assistant 출력 생성"""
    return [
        AssistantOutput(
            assistant_id="assistant_1",
            comment_text="이 질문은 기술적 측면에서 매우 흥미롭습니다. 특히 구현 방법론에 대한 깊은 이해가 필요합니다.",
            confidence_score=0.8,
            related_experts=[1, 3, 5]
        ),
        AssistantOutput(
            assistant_id="assistant_2", 
            comment_text="창의적 관점에서 보면 이 문제는 새로운 접근 방식을 요구합니다.",
            confidence_score=0.7,
            related_experts=[2, 4, 6]
        )
    ]


def create_test_max_experts() -> MaxExpertsList:
    """테스트용 MaxExpertsList 생성"""
    experts = []
    for i in range(1, 7):
        expert = ExpertOutput(
            expert_id=i,
            output_tensor=torch.randn(1, 1, 4096),
            weight=0.1 + i * 0.05,
            activation_score=0.2 + i * 0.1,
            fusion_degree=0.8 + i * 0.02
        )
        experts.append(expert)
    
    return MaxExpertsList(experts=experts)


# 테스트 실행 함수
def test_embedding_processor():
    """EmbeddingProcessor 테스트"""
    # 더미 모델과 토크나이저 (실제 사용 시에는 실제 모델 로드)
    class DummyModel:
        def embed_tokens(self, tokens):
            return torch.randn(tokens.shape[0], tokens.shape[1], 4096)
    
    class DummyTokenizer:
        def encode(self, text, add_bos=True, add_eos=True):
            return [1, 2, 3, 4, 5]  # 더미 토큰
    
    # 프로세서 초기화
    processor = EmbeddingProcessor(
        model=DummyModel(),
        tokenizer=DummyTokenizer(),
        hidden_dim=4096
    )
    
    # 테스트 데이터
    assistant_outputs = create_test_assistant_outputs()
    max_experts = create_test_max_experts()
    
    # 처리 실행
    print("=== EmbeddingProcessor 테스트 ===")
    embedded_outputs = processor.vectorize_assistant_output(assistant_outputs)
    routing_vector = processor.route_to_experts(embedded_outputs, max_experts)
    
    # 결과 출력
    print(f"임베딩 출력 수: {len(embedded_outputs)}")
    print(f"라우팅 벡터 형태: {routing_vector.shape}")
    print(f"라우팅 통계: {processor.get_routing_statistics()}")
    
    return processor, embedded_outputs, routing_vector


if __name__ == "__main__":
    test_embedding_processor()