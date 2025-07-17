"""
Dual Output MoE System for Mistral-based Architecture
이중 패스 MoE 시스템 - 첫 번째 패스에서 max_experts 생성, 두 번째 패스에서 Assistant 피드백 기반 재라우팅

기존 Mistral MoE 구조를 확장하여 두 번의 출력을 생성하는 시스템
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import json
import numpy as np


@dataclass
class ExpertOutput:
    """Expert 출력 정보를 저장하는 데이터 클래스"""
    expert_id: int
    output_tensor: torch.Tensor # 이 필드는 더 이상 사용되지 않을 수 있으나, 호환성을 위해 유지
    weight: float
    activation_score: float # Gate의 라우팅 점수
    fusion_degree: float = 1.0

@dataclass
class MaxExpertsList:
    """max_experts 리스트를 관리하는 클래스"""
    experts: List[ExpertOutput]
    max_count: int = 8
    threshold: float = 0.1

    def add_expert(self, expert_output: ExpertOutput):
        """Expert 추가 (활성화 점수 기준)"""
        if expert_output.activation_score >= self.threshold:
            self.experts.append(expert_output)
            self.experts.sort(key=lambda x: x.activation_score, reverse=True)
            if len(self.experts) > self.max_count:
                self.experts = self.experts[:self.max_count]

class DualOutputMoE(nn.Module):
    """
    기존 MoE 모델을 제어하는 이중 출력 시스템.
    PyTorch Hook을 사용하여 라우팅 정보를 캡처합니다.
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.num_experts = self.model.args.moe.num_experts
        
        # Hook을 통해 캡처된 라우팅 정보를 저장할 변수
        self.captured_routing_logits = []
        
        # MoE 레이어의 gate에 Hook 등록
        self._register_hooks()
        
        # 내부 상태
        self.max_experts_cache = None

    def _register_hooks(self):
        """모델의 모든 MoE Gate에 Forward Hook을 등록합니다."""
        
        def hook_fn(module, input, output):
            # output은 라우팅 로짓(logits) 텐서입니다.
            self.captured_routing_logits.append(output)

        for layer in self.model.layers:
            # Mixtral 모델의 feed_forward.gate 구조에 접근
            if hasattr(layer, 'feed_forward') and hasattr(layer.feed_forward, 'gate'):
                layer.feed_forward.gate.register_forward_hook(hook_fn)

    def _clear_captured_data(self):
        """다음 Forward Pass를 위해 캡처된 데이터를 초기화합니다."""
        self.captured_routing_logits = []

    def forward_first_pass(self, tokens: torch.Tensor, seqlens: List[int]) -> Tuple[torch.Tensor, MaxExpertsList]:
        """
        첫 번째 패스: 일반적인 MoE 출력 + max_experts 생성
        
        Args:
            tokens: 입력 토큰 ID [num_tokens]
            seqlens: 각 시퀀스의 길이 [batch_size]

        Returns:
            first_output: 모델의 최종 출력 (임베딩)
            max_experts: MaxExpertsList 객체
        """
        self._clear_captured_data()
        
        # 1. 모델의 정식 forward pass 실행
        # mistral-inference의 Transformer.forward는 토큰 ID와 seqlens를 받습니다.
        first_output = self.model.forward(tokens, seqlens)
        
        # 2. Hook으로 캡처된 라우팅 정보 분석
        if not self.captured_routing_logits:
            raise RuntimeError("MoE routing 정보를 캡처하지 못했습니다. 모델 구조를 확인하세요.")
            
        # 모든 MoE 레이어의 라우팅 점수를 평균내어 각 Expert의 중요도 계산
        # captured_routing_logits: List[Tensor] (각 요소는 [num_tokens, num_experts] 형태)
        avg_routing_probs = torch.zeros(self.num_experts)
        
        for logits in self.captured_routing_logits:
            probs = F.softmax(logits, dim=-1) # [num_tokens, num_experts]
            # 각 Expert의 평균 활성화 확률
            avg_routing_probs += probs.mean(dim=0).cpu()
            
        avg_routing_probs /= len(self.captured_routing_logits)
        
        # 3. max_experts 리스트 생성
        max_experts = MaxExpertsList(experts=[], max_count=self.model.args.moe.num_experts)
        all_experts = []
        for expert_id in range(self.num_experts):
            activation_score = avg_routing_probs[expert_id].item()
            # weight와 output_tensor는 이 설계에서 더 이상 직접적인 의미가 없지만,
            # 하위 시스템과의 호환성을 위해 activation_score를 weight로 사용합니다.
            expert_out = ExpertOutput(
                expert_id=expert_id,
                output_tensor=None, # 더 이상 사용되지 않음
                weight=activation_score,
                activation_score=activation_score
            )
            all_experts.append(expert_out)
            
        # 활성화 점수 기준으로 정렬하여 max_experts에 추가
        all_experts.sort(key=lambda x: x.activation_score, reverse=True)
        for expert in all_experts:
            max_experts.add_expert(expert) # add_expert 내부의 threshold 조건에 따라 추가됨
        
        # 4. 캐시 저장 및 반환
        self.max_experts_cache = max_experts
        return first_output, max_experts
    
    def forward_second_pass(self, assistant_embedded_vector: torch.Tensor) -> torch.Tensor:
        """
        두 번째 패스: Assistant의 임베딩 벡터로 MoE 레이어를 직접 실행하여 새로운 출력을 생성합니다.
        
        Args:
            assistant_embedded_vector: [batch_size, seq_len, hidden_dim]

        Returns:
            second_output: 두 번째 MoE 패스의 최종 출력 임베딩 [cite: 111, 112]
        """
        if self.max_experts_cache is None:
            raise ValueError("두 번째 패스 실행 전에 첫 번째 패스가 실행되어야 합니다.") [cite: 112]

        # --- 수정된 핵심 로직 ---
        # 1. 입력 임베딩을 h(hidden_state)로 설정합니다. [cite: 113]
        h = assistant_embedded_vector
        
        # 2. 어텐션 마스크를 생성합니다. [cite: 114]
        #    입력 시퀀스 전체를 보는 마스크(Causal Mask)를 생성합니다. [cite: 115]
        mask = torch.full((h.shape[1], h.shape[1]), float("-inf"), device=h.device)
        mask = torch.triu(mask, diagonal=1) [cite: 116]
        
        # 3. 모델의 모든 트랜스포머 레이어를 순차적으로 통과시킵니다. [cite: 117]
        #    이 과정에서 모델 내부의 MoE Gate가 새로운 입력(h)에 따라 다시 동작합니다. [cite: 117]
        for layer in self.model.layers:
            h = layer(h, mask)
        
        # 4. 최종 Layer Normalization을 적용하여 두 번째 결과물을 완성합니다. [cite: 119]
        second_output = self.model.norm(h)
        # --- 수정 끝 ---
        
        return second_output
    
    def _broadcast_expert_output(self, expert_output: torch.Tensor, 
                               target_shape: torch.Size, weight: float) -> torch.Tensor:
        """
        Expert 출력을 타겟 형태로 브로드캐스팅
        
        Args:
            expert_output: Expert의 출력 텐서
            target_shape: 타겟 형태 [total_tokens, hidden_dim]
            weight: 가중치
        
        Returns:
            브로드캐스트된 출력 텐서
        """
        total_tokens, hidden_dim = target_shape
        
        if expert_output.shape[0] == total_tokens:
            # 이미 올바른 형태
            return expert_output * weight
        else:
            # 평균 풀링 후 브로드캐스팅
            avg_output = expert_output.mean(dim=0, keepdim=True)  # [1, hidden_dim]
            return avg_output.expand(total_tokens, hidden_dim) * weight
    
    def update_fusion_degree(self, expert_id: int, new_degree: float):
        """
        특정 Expert의 fusion_degree 업데이트
        
        Args:
            expert_id: Expert ID
            new_degree: 새로운 fusion degree (0.0 ~ 1.0)
        """
        self.fusion_degrees[expert_id] = max(0.0, min(1.0, new_degree))
        
        # 캐시된 max_experts도 업데이트
        if self.max_experts_cache:
            for expert in self.max_experts_cache.experts:
                if expert.expert_id == expert_id:
                    expert.fusion_degree = new_degree
                    break
    
    def get_expert_metrics(self) -> Dict:
        """
        Expert 활용도 및 성능 메트릭 반환
        
        Returns:
            메트릭 딕셔너리
        """
        return {
            'first_pass_calls': self.metrics['first_pass_calls'],
            'second_pass_calls': self.metrics['second_pass_calls'],
            'expert_utilization': self.metrics['expert_utilization'].tolist(),
            'routing_entropy_avg': np.mean(self.metrics['routing_entropy']) if self.metrics['routing_entropy'] else 0.0,
            'fusion_degrees': dict(self.fusion_degrees),
            'active_experts': len([d for d in self.fusion_degrees.values() if d > 0.1])
        }
    
    def reset_metrics(self):
        """메트릭 초기화"""
        self.metrics = {
            'first_pass_calls': 0,
            'second_pass_calls': 0,
            'expert_utilization': torch.zeros(self.num_experts),
            'routing_entropy': []
        }
    
    def forward(self, input_tensor: torch.Tensor, 
                assistant_vector: Optional[torch.Tensor] = None,
                return_max_experts: bool = False) -> torch.Tensor:
        """
        통합 forward 함수 (기존 MoE 호환성 유지)
        
        Args:
            input_tensor: 입력 텐서
            assistant_vector: Assistant 벡터 (선택적)
            return_max_experts: max_experts 반환 여부
        
        Returns:
            출력 텐서 (또는 튜플)
        """
        if assistant_vector is None:
            # 첫 번째 패스만 실행
            first_output, max_experts = self.forward_first_pass(input_tensor)
            if return_max_experts:
                return first_output, max_experts
            return first_output
        else:
            # 두 번째 패스 실행
            if self.max_experts_cache is None:
                raise ValueError("두 번째 패스 실행 전에 첫 번째 패스가 실행되어야 합니다.")
            return self.forward_second_pass(assistant_vector, self.max_experts_cache)
    
    def save_state(self, filepath: str):
        """상태 저장"""
        state = {
            'fusion_degrees': self.fusion_degrees,
            'metrics': self.get_expert_metrics(),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self, filepath: str):
        """상태 로드"""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.fusion_degrees = state.get('fusion_degrees', {})
        # 문자열 키를 정수로 변환
        self.fusion_degrees = {int(k): v for k, v in self.fusion_degrees.items()}


# 사용 예시 및 테스트 코드
def create_test_dual_moe():
    """테스트용 DualOutputMoE 생성"""
    
    # 더미 Expert 생성
    class DummyExpert(nn.Module):
        def __init__(self, hidden_dim):
            super().__init__()
            self.linear = nn.Linear(hidden_dim, hidden_dim)
            
        def forward(self, x):
            return self.linear(x)
    
    # 더미 Gate 생성
    class DummyGate(nn.Module):
        def __init__(self, hidden_dim, num_experts):
            super().__init__()
            self.linear = nn.Linear(hidden_dim, num_experts)
            
        def forward(self, x):
            return self.linear(x)
    
    # 파라미터 설정
    hidden_dim = 512
    num_experts = 8
    
    # 컴포넌트 생성
    experts = [DummyExpert(hidden_dim) for _ in range(num_experts)]
    gate = DummyGate(hidden_dim, num_experts)
    
    # DualOutputMoE 생성
    args = type('Args', (), {'hidden_dim': hidden_dim})()
    dual_moe = DualOutputMoE(experts, gate, args, top_k=2)
    
    return dual_moe


if __name__ == "__main__":
    # 테스트 실행
    print("DualOutputMoE 테스트 시작...")
    
    # 1. 시스템 생성
    dual_moe = create_test_dual_moe()
    
    # 2. 테스트 입력
    batch_size, seq_len, hidden_dim = 2, 10, 512
    input_tensor = torch.randn(batch_size, seq_len, hidden_dim)
    
    # 3. 첫 번째 패스 실행
    print("첫 번째 패스 실행...")
    first_output, max_experts = dual_moe.forward_first_pass(input_tensor)
    print(f"첫 번째 출력 형태: {first_output.shape}")
    print(f"max_experts 개수: {len(max_experts.experts)}")
    
    # 4. 더미 Assistant 벡터 생성
    assistant_vector = torch.randn(batch_size, seq_len, hidden_dim)
    
    # 5. 두 번째 패스 실행
    print("두 번째 패스 실행...")
    second_output = dual_moe.forward_second_pass(assistant_vector, max_experts)
    print(f"두 번째 출력 형태: {second_output.shape}")
    
    # 6. 메트릭 확인
    metrics = dual_moe.get_expert_metrics()
    print(f"메트릭: {metrics}")
    
    # 7. Fusion degree 업데이트 테스트
    dual_moe.update_fusion_degree(0, 0.5)
    print("Fusion degree 업데이트 완료")
    
    print("테스트 완료!")