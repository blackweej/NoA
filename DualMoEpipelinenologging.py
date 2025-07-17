"""
DualMoE Pipeline Implementation based on mistral-inference
Implements dual-pass MoE with Assistant system and fusion degree calculation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import json
import logging
from pathlib import Path

# mistral-inference imports
from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from DualMoEoutput import DualOutputMoE

@dataclass
class ExpertOutput:
    """Expert 출력 정보를 담는 데이터 클래스"""
    expert_id: int
    output_tensor: torch.Tensor
    weight: float
    activation_score: float
    fusion_degree: float = 1.0


@dataclass
class AssistantOutput:
    """Assistant 출력 정보를 담는 데이터 클래스"""
    assistant_id: str
    comment_text: str
    confidence_score: float
    related_experts: List[int]
    embedding_vector: Optional[torch.Tensor] = None


@dataclass
class FusionMetrics:
    """Fusion 계산 결과를 담는 데이터 클래스"""
    expert_id: int
    similarity_score: float
    novelty_score: float
    fusion_degree: float
    last_updated: datetime


@dataclass
class SurveyResponse:
    """설문 응답 데이터 클래스"""
    question: str
    relevance_scores: List[int]  # 1-5 점수
    timestamp: datetime
    user_id: Optional[str] = None


class MaxExpertsList:
    """최대 활성화된 Expert들의 리스트"""
    def __init__(self, max_count: int = 8):
        self.max_count = max_count
        self.experts: List[ExpertOutput] = []
        self.threshold = 0.1  # 최소 활성화 임계값
    
    def add_expert(self, expert_output: ExpertOutput):
        if expert_output.activation_score > self.threshold:
            self.experts.append(expert_output)
            # 활성화 점수 기준으로 정렬하고 상위 max_count개만 유지
            self.experts.sort(key=lambda x: x.activation_score, reverse=True)
            self.experts = self.experts[:self.max_count]
    
    def get_expert_ids(self) -> List[int]:
        return [expert.expert_id for expert in self.experts]

class AssistantRouter:
    """Assistant 시스템 라우터"""
    
    def __init__(self, tokenizer: MistralTokenizer, model: Transformer, classification_threshold: float = 0.3):
        self.tokenizer = tokenizer
        self.model = model
        self.classification_threshold = classification_threshold
        self.logger = logging.getLogger(__name__)
    
    def _cluster_experts_by_activation(self, max_experts: MaxExpertsList) -> Dict[int, List[ExpertOutput]]:
        """Expert들을 활성화 패턴으로 클러스터링"""
        # 간단한 구현: 활성화 점수 기준으로 그룹화
        clusters = {}
        
        for i, expert in enumerate(max_experts.experts):
            cluster_id = i // 2  # 2개씩 그룹화
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(expert)
        
        return clusters
    
    def _generate_prompt_template(self, expert_group: List[ExpertOutput]) -> str:
        """Expert 그룹 기반 프롬프트 템플릿 생성"""
        expert_ids = [expert.expert_id for expert in expert_group]
        return f"""
        당신은 Expert {expert_ids}의 전문성을 가진 AI Assistant입니다.
        다음 질문에 대해 보완적인 관점에서 코멘트를 작성해주세요.
        
        질문: {{question}}
        
        기존 답변의 맥락을 고려하여 새로운 통찰이나 관점을 제공해주세요.
        코멘트는 간결하고 명확하게 작성해주세요.
        """
    
    def _generate_comment(self, prompt: str, assistant_id: str) -> str:
        """실제 Generate 함수 호출"""
        try:
            # 1. 프롬프트 토크나이징
            completion_request = ChatCompletionRequest(
                messages=[UserMessage(content=prompt)]
            )
            tokens = self.tokenizer.encode_chat_completion(completion_request).tokens
            
            # 2. Generate 함수 호출
            out_tokens, _ = generate(
                [tokens],
                self.model,
                max_tokens=256,
                temperature=0.7,
                eos_id=self.tokenizer.instruct_tokenizer.tokenizer.eos_id
            )
            
            # 3. 디코딩
            comment_text = self.tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])
            
            # 4. 코멘트 포맷팅
            formatted_comment = f"[Assistant_{assistant_id}] 보완적 관점: {comment_text}"
            
            return formatted_comment
            
        except Exception as e:
            self.logger.error(f"Error generating comment: {e}")
            return f"[Assistant_{assistant_id}] 코멘트 생성 중 오류가 발생했습니다."
    
    def bridge_through_assistants(self, original_question: str, first_output: torch.Tensor, max_experts: MaxExpertsList) -> List[AssistantOutput]:
        """Assistant 시스템 메인 처리 함수"""
        self.logger.info(f"Processing question through {len(max_experts.experts)} experts")
        
        # 1. Expert 클러스터링
        expert_clusters = self._cluster_experts_by_activation(max_experts)
        
        # 2. 각 클러스터별 Assistant 생성 및 코멘트 생성
        assistant_outputs = []
        
        for cluster_id, expert_group in expert_clusters.items():
            # Assistant 설정
            assistant_id = f"cluster_{cluster_id}"
            prompt_template = self._generate_prompt_template(expert_group)
            
            # 프롬프트 구성
            prompt = prompt_template.format(question=original_question)
            
            # 코멘트 생성
            comment_text = self._generate_comment(prompt, assistant_id)
            
            # 신뢰도 점수 계산 (활성화 점수 평균)
            confidence_score = sum(expert.activation_score for expert in expert_group) / len(expert_group)
            
            # 관련 Expert 리스트
            related_experts = [expert.expert_id for expert in expert_group 
                             if expert.activation_score > self.classification_threshold]
            
            assistant_output = AssistantOutput(
                assistant_id=assistant_id,
                comment_text=comment_text,
                confidence_score=confidence_score,
                related_experts=related_experts
            )
            
            assistant_outputs.append(assistant_output)
        
        self.logger.info(f"Generated {len(assistant_outputs)} assistant comments")
        return assistant_outputs


class EmbeddingProcessor:
    """임베딩 처리 시스템"""
    
    def __init__(self, model: Transformer, tokenizer: MistralTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.logger = logging.getLogger(__name__)
    
    def _create_embedding(self, text: str) -> torch.Tensor:
        """텍스트를 임베딩 벡터로 변환"""
        try:
            # 토크나이징
            completion_request = ChatCompletionRequest(
                messages=[UserMessage(content=text)]
            )
            tokens = self.tokenizer.encode_chat_completion(completion_request).tokens
            
            # 임베딩 레이어 통과
            with torch.no_grad():
                token_embeddings = self.model.tok_embeddings(torch.tensor(tokens))
                # 평균 풀링으로 문장 임베딩 생성
                sentence_embedding = token_embeddings.mean(dim=0)
            
            return sentence_embedding
            
        except Exception as e:
            self.logger.error(f"Error creating embedding: {e}")
            # 기본값 반환
            return torch.zeros(self.model.args.dim)
    
    def vectorize_assistant_output(self, assistant_outputs: List[AssistantOutput]) -> List[AssistantOutput]:
        """Assistant 출력들을 벡터화"""
        for assistant_output in assistant_outputs:
            embedding_vector = self._create_embedding(assistant_output.comment_text)
            assistant_output.embedding_vector = embedding_vector
        
        self.logger.info(f"Vectorized {len(assistant_outputs)} assistant outputs")
        return assistant_outputs
    
    def route_to_experts(self, assistant_outputs: List[AssistantOutput], max_experts: MaxExpertsList) -> torch.Tensor:
        """임베딩된 Assistant 출력을 Expert 시스템으로 라우팅"""
        # 모든 Assistant 임베딩을 가중평균으로 결합
        if not assistant_outputs:
            return torch.zeros(1, 1, self.model.args.dim)
        
        combined_embedding = torch.zeros_like(assistant_outputs[0].embedding_vector)
        total_weight = 0
        
        for assistant_output in assistant_outputs:
            if assistant_output.embedding_vector is not None:
                weight = assistant_output.confidence_score
                combined_embedding += assistant_output.embedding_vector * weight
                total_weight += weight
        
        if total_weight > 0:
            combined_embedding /= total_weight
        
        # 배치 차원 추가
        routing_vector = combined_embedding.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden_dim]
        
        return routing_vector


class FusionController:
    """Fusion 제어 시스템"""
    
    def __init__(self):
        self.fusion_degrees: Dict[int, float] = {}
        self.logger = logging.getLogger(__name__)
    
    def _calculate_output_similarity(self, first_output: torch.Tensor, second_output: torch.Tensor) -> float:
        """두 출력 간의 유사도 계산"""
        # 코사인 유사도 계산
        first_flat = first_output.reshape(-1)
        second_flat = second_output.reshape(-1)
        
        similarity = F.cosine_similarity(first_flat, second_flat, dim=0)
        return similarity.item()
    
    def _calculate_novelty_score(self, first_output: torch.Tensor, second_output: torch.Tensor) -> float:
        """참신성 점수 계산"""
        # L2 거리 기반 참신성 계산
        l2_distance = torch.norm(first_output - second_output, p=2)
        max_distance = torch.norm(first_output, p=2) + torch.norm(second_output, p=2)
        
        if max_distance == 0:
            return 0.0
        
        novelty_score = (l2_distance / max_distance).item()
        return min(1.0, novelty_score)
    
    def calculate_fusion_degree(self, first_output: torch.Tensor, second_output: torch.Tensor, max_experts: MaxExpertsList) -> List[FusionMetrics]:
        """두 출력 간 비교를 통한 fusion_degree 계산"""
        fusion_metrics = []
        
        # 전체 출력 간 유사도 및 참신성 계산
        similarity_score = self._calculate_output_similarity(first_output, second_output)
        novelty_score = self._calculate_novelty_score(first_output, second_output)
        
        for expert_output in max_experts.experts:
            expert_id = expert_output.expert_id
            
            # 기존 fusion_degree 가져오기
            current_fusion_degree = self.fusion_degrees.get(expert_id, 1.0)
            
            # 동적 조정 공식
            new_fusion_degree = self._adjust_fusion_degree(
                current_fusion_degree,
                similarity_score,
                novelty_score,
                expert_output.activation_score
            )
            
            fusion_metric = FusionMetrics(
                expert_id=expert_id,
                similarity_score=similarity_score,
                novelty_score=novelty_score,
                fusion_degree=new_fusion_degree,
                last_updated=datetime.now()
            )
            
            fusion_metrics.append(fusion_metric)
            
            # 업데이트된 fusion_degree 저장
            self.fusion_degrees[expert_id] = new_fusion_degree
            expert_output.fusion_degree = new_fusion_degree
        
        self.logger.info(f"Calculated fusion degrees for {len(fusion_metrics)} experts")
        return fusion_metrics
    
    def _adjust_fusion_degree(self, current_degree: float, similarity: float, novelty: float, activation: float) -> float:
        """동적 fusion_degree 조정 공식"""
        # 높은 참신성 + 적당한 유사도 = 높은 fusion_degree
        # 낮은 참신성 + 높은 유사도 = 낮은 fusion_degree
        
        novelty_factor = novelty * 0.6  # 참신성 가중치
        similarity_factor = (1.0 - similarity) * 0.3  # 차이점 가중치
        activation_factor = activation * 0.1  # 활성화 가중치
        
        adjustment = novelty_factor + similarity_factor + activation_factor
        
        # 현재 degree에 조정값 적용
        new_degree = current_degree + (adjustment - 0.5) * 0.1
        
        # 범위 제한 (0-1)
        new_degree = max(0.0, min(1.0, new_degree))
        
        return new_degree
    
    def apply_fusion_weights(self, first_output: torch.Tensor, second_output: torch.Tensor, fusion_metrics: List[FusionMetrics]) -> torch.Tensor:
        """Fusion degree를 적용한 최종 출력 생성"""
        if not fusion_metrics:
            return first_output
        
        # 전체 fusion 가중치 계산
        total_fusion_weight = sum(metric.fusion_degree for metric in fusion_metrics)
        
        if total_fusion_weight == 0:
            return first_output
        
        # 정규화된 가중치 계산
        alpha = total_fusion_weight / len(fusion_metrics)  # 평균 fusion degree
        beta = 1.0 - alpha  # 원본 출력 영향도
        
        # 최종 출력 생성
        fused_output = beta * first_output + alpha * second_output
        
        self.logger.info(f"Applied fusion with alpha={alpha:.3f}, beta={beta:.3f}")
        return fused_output


class SurveySystem:
    """설문 조사 시스템"""
    
    def __init__(self, fusion_controller: FusionController):
        self.fusion_controller = fusion_controller
        self.survey_responses: List[SurveyResponse] = []
        self.adjustment_rates = {
            'positive': 0.05,  # 긍정적 피드백 시 증가율
            'negative': -0.1   # 부정적 피드백 시 감소율 (더 강하게)
        }
        self.logger = logging.getLogger(__name__)
    
    def collect_survey_response(self, question: str, relevance_scores: List[int], user_id: Optional[str] = None) -> SurveyResponse:
        """사용자로부터 설문 응답 수집"""
        survey_response = SurveyResponse(
            question=question,
            relevance_scores=relevance_scores,
            timestamp=datetime.now(),
            user_id=user_id
        )
        
        self.survey_responses.append(survey_response)
        self.logger.info(f"Collected survey response with scores: {relevance_scores}")
        
        return survey_response
    
    def update_fusion_degrees(self, recent_responses: List[SurveyResponse]) -> Dict[int, float]:
        """설문 결과를 기반으로 fusion_degree 업데이트"""
        updated_degrees = {}
        
        for response in recent_responses:
            avg_relevance = sum(response.relevance_scores) / len(response.relevance_scores)
            
            # 평균 점수 기반 조정
            if avg_relevance >= 4.0:  # 긍정적 피드백
                adjustment_rate = self.adjustment_rates['positive']
            elif avg_relevance <= 2.0:  # 부정적 피드백
                adjustment_rate = self.adjustment_rates['negative']
            else:  # 중립
                adjustment_rate = 0.0
            
            # 모든 Expert의 fusion_degree 업데이트
            for expert_id in self.fusion_controller.fusion_degrees:
                current_degree = self.fusion_controller.fusion_degrees[expert_id]
                new_degree = max(0.0, min(1.0, current_degree + adjustment_rate))
                self.fusion_controller.fusion_degrees[expert_id] = new_degree
                updated_degrees[expert_id] = new_degree
        
        self.logger.info(f"Updated fusion degrees based on {len(recent_responses)} responses")
        return updated_degrees


class DualMoEPipeline:
    """DualMoE 파이프라인 메인 클래스"""
    
    def __init__(self, model_path: str, tokenizer_path: str):
        # 모델 로딩
        self.tokenizer = MistralTokenizer.from_file(tokenizer_path)
        self.model = Transformer.from_folder(model_path)
        
        # --- 수정된 부분 ---
        # 더 이상 가짜 Expert와 Gate를 만들지 않습니다.
        # 실제 모델을 DualOutputMoE 컨트롤러에 전달합니다.
        self.experts_system = DualOutputMoE(self.model)
        # --- 수정 끝 ---
        
        
        
        self.assistants_system = AssistantRouter(self.tokenizer, self.model)
        self.embedding_processor = EmbeddingProcessor(self.model, self.tokenizer)
        self.fusion_controller = FusionController()
        self.survey_system = SurveySystem(self.fusion_controller)
        
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
    
    def process_query(self, user_query: str, enable_survey: bool = False) -> Dict[str, Any]:
        """전체 파이프라인 실행"""
        self.logger.info(f"Processing query: {user_query}")
        
        # 1. 입력 전처리 (토큰 ID와 시퀀스 길이 반환)
        tokens, seqlens = self._preprocess_query(user_query)
        first_output_embedding, max_experts = self.experts_system.forward_first_pass(tokens, seqlens)
        
        # 2. 첫 번째 MoE 패스 (입력이 토큰 ID로 변경됨)
        # first_output은 이제 모델의 최종 로짓(logits) 또는 임베딩입니다.
        first_output_embedding, max_experts = self.experts_system.forward_first_pass(tokens, seqlens)
        first_output_text = self._decode_output_to_text(first_output_embedding)
        # 3. Assistant 시스템 처리
        assistant_outputs = self.assistants_system.bridge_through_assistants(
            user_query, 
            first_output_text, # 수정: 임베딩 대신 텍스트 전달 [cite: 18]
            max_experts
        )
        
        # 5. 임베딩 처리
        second_pass_input_embedding = self._prepare_second_pass_input(
            user_query, first_output_text, assistant_outputs
        )
        
        # 6. 두 번째 MoE 패스 실행
        second_output = self.experts_system.forward_second_pass(second_pass_input_embedding)
        # 8. Fusion 처리 (외부 FusionController 호출)
        #    더 이상 파이프라인 내부의 간단한 함수를 사용하지 않습니다.
        fusion_metrics = self.fusion_controller.calculate_fusion_degree(
            first_output_embedding, second_output, max_experts
        )
        fusion_result = self.fusion_controller.apply_fusion_weights(
            first_output_embedding, second_output, fusion_metrics
        )
        
        # 9. 결과 패키징
        result = {
            'user_query': user_query,
            'first_output': first_output_embedding,
            'second_output': second_output,
            'fused_output': fusion_result.fused_output, # FusionResult 객체에서 최종 출력을 가져옴
            'assistant_outputs': assistant_outputs,
            'fusion_metrics': fusion_metrics,
            'max_experts': max_experts,
            'survey_ready': enable_survey
        }
        
        self._last_query_result = result # 나중에 설문에서 참조할 수 있도록 마지막 결과를 저장
        
        # 10. 설문조사 (옵션)
        if enable_survey:
            # SurveySystem은 텍스트 출력을 필요로 하므로, 최종 결과들을 디코딩합니다.
            first_text = self._decode_output_to_text(result['first_output'])
            second_text = self._decode_output_to_text(result['second_output'])
            fused_text = self._decode_output_to_text(result['fused_output'])
            
            # SurveySystem의 설문 수집 인터페이스를 호출합니다.
            self.survey_system.collect_survey_response(
                question=user_query,
                outputs={
                    'first': first_text,
                    'second': second_text,
                    'fused': fused_text,
                },
                # auto_mode=True로 설정하면 테스트 시 자동으로 점수를 매깁니다.
                # 사용자에게 직접 입력을 받으려면 False로 두거나 생략합니다.
                auto_mode=False 
            )

        self.logger.info("Pipeline processing completed")
        return result
    
    
    def _prepare_second_pass_input(self, user_query: str, first_output_text: str, assistant_outputs: List[Any]) -> torch.Tensor:
        """
        Assistant의 코멘트를 바탕으로 두 번째 패스의 입력 텐서를 준비합니다.
        하나의 '개념 벡터'가 아닌, 토큰 '시퀀스 임베딩'을 생성합니다.
        """
        self.logger.info("Preparing input for the second pass based on assistant comments...")

        # 1. Assistant 코멘트들을 하나의 텍스트로 결합
        assistant_comments = "\n".join([f"- {out.comment_text}" for out in assistant_outputs])

        # 2. 두 번째 추론을 위한 새로운 프롬프트 구성
        second_pass_prompt = f"""
## 원본 질문
{user_query}

## 1차 답변
{first_output_text}

## 전문가 코멘트 (새로운 관점)
{assistant_comments}

## 종합적 재검토 및 최종 답변
"""

        # 3. 프롬프트를 토큰화
        # mistral-inference의 토크나이저를 사용하여 인코딩합니다.
        completion_request = ChatCompletionRequest(
            messages=[UserMessage(content=second_pass_prompt)]
        )
        tokens = self.tokenizer.encode_chat_completion(completion_request).tokens
        token_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0) # [1, seq_len] 형태

        # 4. 토큰 시퀀스를 임베딩 시퀀스로 변환
        # 모델의 임베딩 레이어를 직접 사용합니다.
        with torch.no_grad():
            # Transformer 모델의 embed_tokens 메서드를 사용해야 합니다.
            # 이 부분이 모델 구조에 따라 다를 수 있으니 확인이 필요합니다.
            # 예시: self.model.tok_embeddings(token_tensor) 또는 self.model.embed_tokens(token_tensor)
            input_embedding = self.model.tok_embeddings(token_tensor) # [1, seq_len, hidden_dim]

        self.logger.info(f"Second pass input prepared with shape: {input_embedding.shape}")
        return input_embedding
    
    def submit_survey(self, relevance_scores: List[int], user_id: Optional[str] = None) -> Dict[int, float]:
        """
        (이 함수는 외부에서 수동으로 설문 점수를 제출할 때 사용될 수 있습니다.)
        설문 제출 및 fusion_degree 업데이트
        """
        if not hasattr(self, '_last_query_result'):
            raise ValueError("평가할 최근 쿼리가 없습니다.")
        
        last_query = self._last_query_result['user_query']
        
        # SurveySystem의 응답 수집 및 업데이트 로직을 직접 호출할 수도 있습니다.
        survey_response = self.survey_system.collect_survey_response(
            question=last_query,
            relevance_scores=relevance_scores,
            user_id=user_id
        )
        
        # 설문 결과를 기반으로 fusion_degree 업데이트
        updated_degrees = self.survey_system.update_fusion_degrees([survey_response])
        
        return updated_degrees
    
    def _decode_output_to_text(self, output_embedding: torch.Tensor) -> str:
        """모델의 최종 출력 임베딩을 텍스트로 디코딩합니다.""" [cite: 8]
        # 모델의 출력 임베딩을 단어 사전에 대한 점수(logits)로 변환
        logits = F.linear(output_embedding, self.model.tok_embeddings.weight) # [batch, seq_len, vocab_size] [cite: 9]
        # 가장 높은 점수를 가진 토큰 ID를 선택
        next_token_ids = torch.argmax(logits, dim=-1) # [batch, seq_len] [cite: 10]

        # 배치 중 첫 번째 결과만 디코딩
        decoded_text = self.tokenizer.decode(next_token_ids[0].tolist()) [cite: 10]
        return decoded_text

    def _preprocess_query(self, query: str) -> Tuple[torch.Tensor, List[int]]:
        """쿼리를 토큰 ID와 시퀀스 길이로 전처리합니다."""
        completion_request = ChatCompletionRequest(
            messages=[UserMessage(content=query)]
        )
        tokens = self.tokenizer.encode_chat_completion(completion_request).tokens
        
        # seqlens는 배치 처리를 위한 각 시퀀스의 길이를 담은 리스트입니다.
        # 여기서는 배치가 1이므로, 전체 토큰 길이가 됩니다.
        seqlens = [len(tokens)]
        
        return torch.tensor(tokens, dtype=torch.long), seqlens


# 사용 예시
if __name__ == "__main__":
    # 모델 경로 설정
    model_path = "C:\\Users\\jungh\\Desktop\\fuck\\mistral-inference-main\\model"  # 실제 모델 경로로 변경
    # model 폴더에 있는 실제 토크나이저 파일(예: tokenizer.model 또는 tekken.json)로 경로를 수정해야 합니다.
    tokenizer_path = "C:\\Users\\jungh\\Desktop\\fuck\\mistral-inference-main\\model\\tokenizer.model"  # 실제 토크나이저 경로로 변경
    
    try:
        # 파이프라인 초기화
        pipeline = DualMoEPipeline(model_path, tokenizer_path)
        
        # 쿼리 처리
        user_query = "recommend the report topic about ai"
        result = pipeline.process_query(user_query, enable_survey=True)
        
        # 결과 출력
        print(f"Query: {result['user_query']}")
        print(f"Active Experts: {len(result['max_experts'].experts)}")
        print(f"Assistant Comments: {len(result['assistant_outputs'])}")
        
        for assistant_output in result['assistant_outputs']:
            print(f"- {assistant_output.comment_text}")
        
        # 설문 제출 예시
        relevance_scores = [4, 5, 3, 4, 5]  # 1-5 점수
        updated_degrees = pipeline.submit_survey(relevance_scores)
        print(f"Updated fusion degrees: {updated_degrees}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please ensure mistral-inference is installed and model paths are correct")