"""
Survey System for Dual MoE Pipeline
사용자 피드백을 수집하고 fusion_degree를 동적으로 조정하는 시스템

작성자: Assistant
날짜: 2025년 7월
"""

import json
import time
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
from pathlib import Path
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SurveyResponse:
    """설문 응답 데이터 클래스"""
    question: str
    first_output_text: str
    second_output_text: str
    fused_output_text: str
    relevance_scores: List[int]  # 5개의 점수 (1-5)
    timestamp: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

@dataclass
class FeedbackMetrics:
    """피드백 메트릭 데이터 클래스"""
    expert_id: int
    avg_relevance: float
    feedback_count: int
    trend: str  # 'positive', 'negative', 'neutral'
    last_updated: str
    confidence_score: float
    
    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.now().isoformat()

class SurveySystem:
    """
    사용자 피드백 수집 및 fusion_degree 동적 조정 시스템
    """
    
    def __init__(self, 
                 fusion_controller=None,
                 survey_storage_path: str = "survey_data.json",
                 adjustment_rates: Dict[str, float] = None):
        """
        초기화
        
        Args:
            fusion_controller: FusionController 인스턴스
            survey_storage_path: 설문 데이터 저장 경로
            adjustment_rates: 조정율 설정
        """
        self.fusion_controller = fusion_controller
        self.survey_storage_path = Path(survey_storage_path)
        
        # 기본 조정율 설정
        self.adjustment_rates = adjustment_rates or {
            'positive': 0.05,  # 긍정적 피드백 시 증가율
            'negative': -0.08, # 부정적 피드백 시 감소율
            'neutral': 0.0
        }
        
        # 설문 응답 저장소
        self.survey_responses: List[SurveyResponse] = []
        
        # 피드백 메트릭 저장소
        self.feedback_metrics: Dict[int, FeedbackMetrics] = {}
        
        # 데이터 로딩
        self._load_survey_data()
        
        logger.info(f"Survey System initialized with storage path: {self.survey_storage_path}")

    def collect_survey_response(self, 
                              question: str,
                              outputs: Dict[str, str],
                              user_id: Optional[str] = None,
                              session_id: Optional[str] = None,
                              auto_mode: bool = False) -> SurveyResponse:
        """
        사용자로부터 설문 응답 수집
        
        Args:
            question: 원본 질문
            outputs: {'first': str, 'second': str, 'fused': str}
            user_id: 사용자 ID (선택적)
            session_id: 세션 ID (선택적)
            auto_mode: 자동 모드 (테스트용)
        
        Returns:
            SurveyResponse 인스턴스
        """
        logger.info(f"Collecting survey response for question: {question[:50]}...")
        
        # 설문 응답 객체 생성
        survey_response = SurveyResponse(
            question=question,
            first_output_text=outputs.get('first', ''),
            second_output_text=outputs.get('second', ''),
            fused_output_text=outputs.get('fused', ''),
            relevance_scores=[],
            timestamp=datetime.now().isoformat(),
            user_id=user_id,
            session_id=session_id
        )
        
        if auto_mode:
            # 자동 모드: 랜덤 점수 생성 (테스트용)
            survey_response.relevance_scores = self._generate_auto_scores(outputs)
        else:
            # 실제 사용자 인터페이스 표시
            relevance_scores = self._display_survey_interface(survey_response)
            survey_response.relevance_scores = relevance_scores
        
        # 응답 저장
        self.survey_responses.append(survey_response)
        
        # 즉시 fusion_degree 업데이트
        self._update_fusion_degrees_from_response(survey_response)
        
        # 데이터 저장
        self._save_survey_data()
        
        logger.info(f"Survey response collected with scores: {survey_response.relevance_scores}")
        return survey_response

    def _display_survey_interface(self, survey_response: SurveyResponse) -> List[int]:
        """
        사용자 인터페이스를 통한 설문 진행
        
        Args:
            survey_response: SurveyResponse 인스턴스
        
        Returns:
            List[int]: 5개의 연관성 점수 (1-5)
        """
        print("\n" + "="*80)
        print("📋 SURVEY: 응답 품질 평가")
        print("="*80)
        
        print(f"\n💭 원본 질문: {survey_response.question}")
        print("\n" + "-"*50)
        
        # 3가지 출력 표시
        outputs = [
            ("첫 번째 출력 (Original MoE)", survey_response.first_output_text),
            ("두 번째 출력 (Assistant Enhanced)", survey_response.second_output_text),
            ("융합 출력 (Fused Output)", survey_response.fused_output_text)
        ]
        
        for i, (title, text) in enumerate(outputs, 1):
            print(f"\n{i}. {title}:")
            print(f"   {text[:200]}..." if len(text) > 200 else f"   {text}")
        
        print("\n" + "-"*50)
        print("📊 다음 각 항목에 대해 1-5점으로 평가해주세요:")
        print("   (1=매우 낮음, 2=낮음, 3=보통, 4=높음, 5=매우 높음)")
        
        questions = [
            "질문과의 연관성",
            "응답의 유용성",
            "정보의 정확성",
            "창의성/참신성",
            "전체적 만족도"
        ]
        
        scores = []
        for i, question in enumerate(questions, 1):
            while True:
                try:
                    score = int(input(f"{i}. {question}: "))
                    if 1 <= score <= 5:
                        scores.append(score)
                        break
                    else:
                        print("   ❌ 1-5 사이의 숫자를 입력해주세요.")
                except ValueError:
                    print("   ❌ 숫자를 입력해주세요.")
        
        print(f"\n✅ 평가 완료! 점수: {scores}")
        print("="*80)
        
        return scores

    def _generate_auto_scores(self, outputs: Dict[str, str]) -> List[int]:
        """
        자동 점수 생성 (테스트용)
        
        Args:
            outputs: 출력 텍스트들
        
        Returns:
            List[int]: 5개의 연관성 점수
        """
        # 텍스트 길이 기반 간단한 점수 생성
        fused_len = len(outputs.get('fused', ''))
        first_len = len(outputs.get('first', ''))
        second_len = len(outputs.get('second', ''))
        
        # 기본 점수 (3) + 길이 기반 보정
        base_score = 3
        
        # 융합 출력이 더 길면 좋은 점수
        if fused_len > max(first_len, second_len):
            bonus = 1
        elif fused_len < min(first_len, second_len):
            bonus = -1
        else:
            bonus = 0
        
        # 랜덤 노이즈 추가
        import random
        scores = []
        for _ in range(5):
            score = base_score + bonus + random.randint(-1, 1)
            scores.append(max(1, min(5, score)))
        
        return scores

    def _update_fusion_degrees_from_response(self, survey_response: SurveyResponse):
        """
        단일 설문 응답으로부터 fusion_degree 업데이트
        
        Args:
            survey_response: SurveyResponse 인스턴스
        """
        if not self.fusion_controller:
            logger.warning("FusionController not available, skipping fusion degree update")
            return
        
        # 평균 점수 계산
        avg_relevance = sum(survey_response.relevance_scores) / len(survey_response.relevance_scores)
        
        # 모든 active expert에 대해 업데이트
        for expert_id in self.fusion_controller.fusion_degrees:  # 기본 8개 expert 가정
            # 현재 fusion_degree 가져오기
            current_degree = getattr(self.fusion_controller, 'fusion_degrees', {}).get(expert_id, 1.0)
            
            # 조정율 계산
            adjustment_rate = self._calculate_adjustment_rate(avg_relevance, 'neutral')
            
            # 새로운 fusion_degree 계산
            new_degree = self._apply_adjustment(current_degree, adjustment_rate)
            
            # 업데이트 (FusionController가 있을 경우)
            if hasattr(self.fusion_controller, 'fusion_degrees'):
                self.fusion_controller.fusion_degrees[expert_id] = new_degree
            
            # 메트릭 업데이트
            self._update_feedback_metrics(expert_id, avg_relevance)
        
        logger.info(f"Fusion degrees updated based on survey response (avg score: {avg_relevance:.2f})")

    def update_fusion_degrees_batch(self, survey_responses: List[SurveyResponse]) -> List[FeedbackMetrics]:
        """
        배치 설문 결과를 기반으로 fusion_degree 업데이트
        
        Args:
            survey_responses: 설문 응답 리스트
        
        Returns:
            List[FeedbackMetrics]: 업데이트된 메트릭들
        """
        logger.info(f"Updating fusion degrees from {len(survey_responses)} survey responses")
        
        # Expert별 피드백 집계
        expert_feedback = self._aggregate_feedback_by_expert(survey_responses)
        
        updated_metrics = []
        for expert_id, feedback_data in expert_feedback.items():
            # 평균 연관성 점수 계산
            avg_relevance = sum(feedback_data['scores']) / len(feedback_data['scores'])
            
            # 트렌드 분석
            trend = self._analyze_trend(feedback_data['scores'])
            
            # Fusion degree 조정
            adjustment_rate = self._calculate_adjustment_rate(avg_relevance, trend)
            
            current_degree = getattr(self.fusion_controller, 'fusion_degrees', {}).get(expert_id, 1.0)
            new_degree = self._apply_adjustment(current_degree, adjustment_rate)
            
            # 업데이트
            if hasattr(self.fusion_controller, 'fusion_degrees'):
                self.fusion_controller.fusion_degrees[expert_id] = new_degree
            
            # 메트릭 생성
            metrics = FeedbackMetrics(
                expert_id=expert_id,
                avg_relevance=avg_relevance,
                feedback_count=len(feedback_data['scores']),
                trend=trend,
                last_updated=datetime.now().isoformat(),
                confidence_score=self._calculate_confidence_score(feedback_data['scores'])
            )
            
            updated_metrics.append(metrics)
            self.feedback_metrics[expert_id] = metrics
        
        # 데이터 저장
        self._save_survey_data()
        
        logger.info(f"Batch fusion degree update completed for {len(updated_metrics)} experts")
        return updated_metrics

    def _aggregate_feedback_by_expert(self, survey_responses: List[SurveyResponse]) -> Dict[int, Dict]:
        """
        Expert별 피드백 집계
        
        Args:
            survey_responses: 설문 응답 리스트
        
        Returns:
            Dict[int, Dict]: Expert ID별 집계 데이터
        """
        expert_feedback = {}
        
        for response in survey_responses:
            avg_score = sum(response.relevance_scores) / len(response.relevance_scores)
            
            for expert_id in self.fusion_controller.fusion_degrees:
                if expert_id not in expert_feedback:
                    expert_feedback[expert_id] = {'scores': [], 'timestamps': []}
                
                expert_feedback[expert_id]['scores'].append(avg_score)
                expert_feedback[expert_id]['timestamps'].append(response.timestamp)
        
        return expert_feedback

    def _analyze_trend(self, scores: List[float]) -> str:
        """
        점수 트렌드 분석
        
        Args:
            scores: 점수 리스트
        
        Returns:
            str: 'positive', 'negative', 'neutral'
        """
        if len(scores) < 2:
            return 'neutral'
        
        # 최근 5개 점수 기준으로 트렌드 분석
        recent_scores = scores[-5:]
        
        if len(recent_scores) < 2:
            return 'neutral'
        
        # 선형 회귀를 통한 트렌드 계산
        x = np.arange(len(recent_scores))
        y = np.array(recent_scores)
        
        if len(x) > 1:
            slope = np.polyfit(x, y, 1)[0]
            
            if slope > 0.1:
                return 'positive'
            elif slope < -0.1:
                return 'negative'
            else:
                return 'neutral'
        
        return 'neutral'

    def _calculate_adjustment_rate(self, avg_relevance: float, trend: str) -> float:
        """
        평균 연관성과 트렌드 기반 조정율 계산
        
        Args:
            avg_relevance: 평균 연관성 점수 (1-5)
            trend: 트렌드 ('positive', 'negative', 'neutral')
        
        Returns:
            float: 조정율
        """
        # 기본 조정율 (5점 만점 기준)
        base_rate = (avg_relevance - 3.0) / 2.0  # -1 ~ 1 범위로 정규화
        
        # 트렌드 가중치 적용
        trend_multiplier = {
            'positive': 1.2,
            'negative': 1.5,  # 부정적 피드백에 더 민감하게 반응
            'neutral': 1.0
        }
        
        adjustment_rate = base_rate * trend_multiplier[trend]
        
        # 조정율 제한
        if adjustment_rate > 0:
            adjustment_rate = min(adjustment_rate, self.adjustment_rates['positive'])
        else:
            adjustment_rate = max(adjustment_rate, self.adjustment_rates['negative'])
        
        return adjustment_rate

    def _apply_adjustment(self, current_degree: float, adjustment_rate: float) -> float:
        """
        현재 fusion_degree에 조정율 적용
        
        Args:
            current_degree: 현재 fusion_degree (0-1)
            adjustment_rate: 조정율
        
        Returns:
            float: 새로운 fusion_degree (0-1)
        """
        new_degree = current_degree + adjustment_rate
        
        # 범위 제한
        new_degree = max(0.0, min(1.0, new_degree))
        
        return new_degree

    def _calculate_confidence_score(self, scores: List[float]) -> float:
        """
        피드백 신뢰도 점수 계산
        
        Args:
            scores: 점수 리스트
        
        Returns:
            float: 신뢰도 점수 (0-1)
        """
        if len(scores) < 2:
            return 0.5
        
        # 분산 기반 신뢰도 계산 (분산이 낮을수록 신뢰도 높음)
        variance = np.var(scores)
        max_variance = 4.0  # 1-5 점수에서 최대 분산
        
        confidence = 1.0 - (variance / max_variance)
        return max(0.0, min(1.0, confidence))

    def _update_feedback_metrics(self, expert_id: int, avg_relevance: float):
        """
        개별 Expert의 피드백 메트릭 업데이트
        
        Args:
            expert_id: Expert ID
            avg_relevance: 평균 연관성 점수
        """
        if expert_id not in self.feedback_metrics:
            self.feedback_metrics[expert_id] = FeedbackMetrics(
                expert_id=expert_id,
                avg_relevance=avg_relevance,
                feedback_count=1,
                trend='neutral',
                last_updated=datetime.now().isoformat(),
                confidence_score=0.5
            )
        else:
            metrics = self.feedback_metrics[expert_id]
            # 이동 평균 업데이트
            metrics.avg_relevance = (metrics.avg_relevance * metrics.feedback_count + avg_relevance) / (metrics.feedback_count + 1)
            metrics.feedback_count += 1
            metrics.last_updated = datetime.now().isoformat()

    def _save_survey_data(self):
        """설문 데이터 저장"""
        try:
            data = {
                'survey_responses': [asdict(response) for response in self.survey_responses],
                'feedback_metrics': {str(k): asdict(v) for k, v in self.feedback_metrics.items()},
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.survey_storage_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Survey data saved to {self.survey_storage_path}")
        except Exception as e:
            logger.error(f"Failed to save survey data: {e}")

    def _load_survey_data(self):
        """설문 데이터 로딩"""
        try:
            if self.survey_storage_path.exists():
                with open(self.survey_storage_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 설문 응답 로딩
                self.survey_responses = [
                    SurveyResponse(**response) 
                    for response in data.get('survey_responses', [])
                ]
                
                # 피드백 메트릭 로딩
                metrics_data = data.get('feedback_metrics', {})
                self.feedback_metrics = {
                    int(k): FeedbackMetrics(**v) 
                    for k, v in metrics_data.items()
                }
                
                logger.info(f"Survey data loaded: {len(self.survey_responses)} responses, {len(self.feedback_metrics)} metrics")
            else:
                logger.info("No existing survey data found, starting fresh")
        except Exception as e:
            logger.error(f"Failed to load survey data: {e}")

    def get_survey_statistics(self) -> Dict:
        """설문 통계 반환"""
        if not self.survey_responses:
            return {'total_responses': 0, 'average_score': 0.0, 'expert_metrics': {}}
        
        total_responses = len(self.survey_responses)
        all_scores = []
        
        for response in self.survey_responses:
            all_scores.extend(response.relevance_scores)
        
        average_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
        
        return {
            'total_responses': total_responses,
            'average_score': average_score,
            'expert_metrics': {k: asdict(v) for k, v in self.feedback_metrics.items()},
            'recent_trends': self._get_recent_trends()
        }

    def _get_recent_trends(self) -> Dict:
        """최근 트렌드 분석"""
        if len(self.survey_responses) < 5:
            return {'trend': 'insufficient_data', 'confidence': 0.0}
        
        recent_responses = self.survey_responses[-10:]  # 최근 10개 응답
        recent_scores = []
        
        for response in recent_responses:
            avg_score = sum(response.relevance_scores) / len(response.relevance_scores)
            recent_scores.append(avg_score)
        
        trend = self._analyze_trend(recent_scores)
        confidence = self._calculate_confidence_score(recent_scores)
        
        return {
            'trend': trend,
            'confidence': confidence,
            'recent_average': sum(recent_scores) / len(recent_scores),
            'score_variance': np.var(recent_scores)
        }

    def export_survey_data(self, export_path: str = None) -> str:
        """설문 데이터 내보내기"""
        if export_path is None:
            export_path = f"survey_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        statistics = self.get_survey_statistics()
        
        export_data = {
            'metadata': {
                'export_time': datetime.now().isoformat(),
                'total_responses': len(self.survey_responses),
                'system_version': '1.0.0'
            },
            'statistics': statistics,
            'raw_data': {
                'survey_responses': [asdict(response) for response in self.survey_responses],
                'feedback_metrics': {str(k): asdict(v) for k, v in self.feedback_metrics.items()}
            }
        }
        
        with open(export_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Survey data exported to {export_path}")
        return export_path

# 사용 예제 및 테스트 코드
if __name__ == "__main__":
    # 테스트용 간단한 FusionController 모킹
    class MockFusionController:
        def __init__(self):
            self.fusion_degrees = {i: 1.0 for i in self.fusion_controller.fusion_degrees}
    
    # 시스템 초기화
    fusion_controller = MockFusionController()
    survey_system = SurveySystem(fusion_controller)
    
    # 테스트 데이터
    test_outputs = {
        'first': "이것은 첫 번째 MoE 출력입니다. 기본적인 답변을 제공합니다.",
        'second': "이것은 Assistant가 강화한 두 번째 출력입니다. 더 창의적이고 상세한 답변을 제공합니다.",
        'fused': "이것은 융합된 최종 출력입니다. 두 출력의 장점을 결합하여 최적의 답변을 제공합니다."
    }
    
    print("🔧 Survey System 테스트 시작")
    
    # 자동 모드로 설문 응답 수집
    response = survey_system.collect_survey_response(
        question="인공지능의 미래에 대해 설명해주세요.",
        outputs=test_outputs,
        user_id="test_user",
        auto_mode=True
    )
    
    print(f"\n📊 수집된 응답: {response.relevance_scores}")
    
    # 통계 확인
    stats = survey_system.get_survey_statistics()
    print(f"\n📈 현재 통계: {stats}")
    
    # 데이터 내보내기
    export_path = survey_system.export_survey_data()
    print(f"\n💾 데이터 내보내기 완료: {export_path}")
    
    print("\n✅ Survey System 테스트 완료")