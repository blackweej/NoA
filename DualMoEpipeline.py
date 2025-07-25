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

# Opsætning af grundlæggende logningskonfiguration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


@dataclass
class ExpertOutput:
    """Dataklasse, der indeholder ekspertoutputoplysninger"""
    expert_id: int
    output_tensor: torch.Tensor
    weight: float
    activation_score: float
    fusion_degree: float = 1.0


@dataclass
class AssistantOutput:
    """Dataklasse, der indeholder assistentoutputoplysninger"""
    assistant_id: str
    comment_text: str
    confidence_score: float
    related_experts: List[int]
    embedding_vector: Optional[torch.Tensor] = None


@dataclass
class FusionMetrics:
    """Dataklasse, der indeholder fusionsberegningsresultater"""
    expert_id: int
    similarity_score: float
    novelty_score: float
    fusion_degree: float
    last_updated: datetime


@dataclass
class SurveyResponse:
    """Dataklasse for undersøgelsessvar"""
    question: str
    relevance_scores: List[int]  # 1-5 point
    timestamp: datetime
    user_id: Optional[str] = None


class MaxExpertsList:
    """Liste over maksimalt aktiverede eksperter"""
    def __init__(self, max_count: int = 8):
        self.max_count = max_count
        self.experts: List[ExpertOutput] = []
        self.threshold = 0.1  # Minimumsaktiveringstærskel
        self.logger = logging.getLogger(__name__)
    
    def add_expert(self, expert_output: ExpertOutput):
        if expert_output.activation_score > self.threshold:
            self.logger.info(f"Tilføjer ekspert {expert_output.expert_id} med aktiveringsscore {expert_output.activation_score:.4f}")
            self.experts.append(expert_output)
            # Sorter efter aktiveringsscore og behold kun de øverste max_count
            self.experts.sort(key=lambda x: x.activation_score, reverse=True)
            self.experts = self.experts[:self.max_count]
        else:
            self.logger.warning(f"Ekspert {expert_output.expert_id} med aktiveringsscore {expert_output.activation_score:.4f} under tærsklen på {self.threshold}")
    
    def get_expert_ids(self) -> List[int]:
        return [expert.expert_id for expert in self.experts]

class AssistantRouter:
    """Assistent system router"""
    
    def __init__(self, tokenizer: MistralTokenizer, model: Transformer, classification_threshold: float = 0.3):
        self.tokenizer = tokenizer
        self.model = model
        self.classification_threshold = classification_threshold
        self.logger = logging.getLogger(__name__)
    
    def _cluster_experts_by_activation(self, max_experts: MaxExpertsList) -> Dict[int, List[ExpertOutput]]:
        """Klyng eksperter efter aktiveringsmønster"""
        self.logger.info("Klynger eksperter efter aktivering.")
        # Simpel implementering: grupper baseret på aktiveringsscore
        clusters = {}
        
        for i, expert in enumerate(max_experts.experts):
            cluster_id = i // 2  # Grupper med 2
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(expert)
        
        self.logger.info(f"Oprettet {len(clusters)} ekspertklynger.")
        return clusters
    
    def _generate_prompt_template(self, expert_group: List[ExpertOutput]) -> str:
        """Generer promptskabelon baseret på ekspertgruppe"""
        expert_ids = [expert.expert_id for expert in expert_group]
        self.logger.info(f"Genererer promptskabelon for ekspertgruppe: {expert_ids}")
        return f"""
        Du er en AI-assistent med ekspertise fra Eksperter {expert_ids}.
        Skriv en kommentar fra et supplerende perspektiv på følgende spørgsmål.
        
        Spørgsmål: {{question}}
        
        Giv ny indsigt eller et nyt perspektiv i betragtning af konteksten i det eksisterende svar.
        Skriv kommentaren kortfattet og klart.
        """
    
    def _generate_comment(self, prompt: str, assistant_id: str) -> str:
        """Kald til den faktiske genereringsfunktion"""
        self.logger.info(f"Genererer kommentar for assistent {assistant_id}.")
        try:
            # 1. Tokenisering af prompt
            completion_request = ChatCompletionRequest(
                messages=[UserMessage(content=prompt)]
            )
            tokens = self.tokenizer.encode_chat_completion(completion_request).tokens
            
            # 2. Kald til genereringsfunktion
            out_tokens, _ = generate(
                [tokens],
                self.model,
                max_tokens=256,
                temperature=0.7,
                eos_id=self.tokenizer.instruct_tokenizer.tokenizer.eos_id
            )
            
            # 3. Dekodning
            comment_text = self.tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])
            
            # 4. Formatering af kommentar
            formatted_comment = f"[Assistant_{assistant_id}] Supplerende perspektiv: {comment_text}"
            self.logger.info(f"Kommentar genereret for assistent {assistant_id}.")
            return formatted_comment
            
        except Exception as e:
            self.logger.error(f"Fejl under generering af kommentar: {e}", exc_info=True)
            return f"[Assistant_{assistant_id}] Der opstod en fejl under generering af kommentar."
    
    def bridge_through_assistants(self, original_question: str, first_output: torch.Tensor, max_experts: MaxExpertsList) -> List[AssistantOutput]:
        """Assistent systemets hovedbehandlingsfunktion"""
        self.logger.info(f"Behandler spørgsmål gennem {len(max_experts.experts)} eksperter")
        
        # 1. Klyngning af eksperter
        expert_clusters = self._cluster_experts_by_activation(max_experts)
        
        # 2. Generering af assistenter og kommentarer for hver klynge
        assistant_outputs = []
        
        for cluster_id, expert_group in expert_clusters.items():
            # Opsætning af assistent
            assistant_id = f"cluster_{cluster_id}"
            self.logger.info(f"Behandler klynge {cluster_id} for at generere assistentoutput.")
            prompt_template = self._generate_prompt_template(expert_group)
            
            # Sammensætning af prompt
            prompt = prompt_template.format(question=original_question)
            
            # Generering af kommentar
            comment_text = self._generate_comment(prompt, assistant_id)
            
            # Beregning af tillidsscore (gennemsnit af aktiveringsscorer)
            confidence_score = sum(expert.activation_score for expert in expert_group) / len(expert_group)
            self.logger.info(f"Beregnet tillidsscore for assistent {assistant_id}: {confidence_score:.4f}")
            
            # Liste over relaterede eksperter
            related_experts = [expert.expert_id for expert in expert_group 
                             if expert.activation_score > self.classification_threshold]
            
            assistant_output = AssistantOutput(
                assistant_id=assistant_id,
                comment_text=comment_text,
                confidence_score=confidence_score,
                related_experts=related_experts
            )
            
            assistant_outputs.append(assistant_output)
        
        self.logger.info(f"Genereret {len(assistant_outputs)} assistentkommentarer")
        return assistant_outputs


class EmbeddingProcessor:
    """Indlejringsbehandlingssystem"""
    
    def __init__(self, model: Transformer, tokenizer: MistralTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.logger = logging.getLogger(__name__)
    
    def _create_embedding(self, text: str) -> torch.Tensor:
        """Konverter tekst til en indlejringsvektor"""
        self.logger.info("Opretter indlejring for tekst...")
        try:
            # Tokenisering
            completion_request = ChatCompletionRequest(
                messages=[UserMessage(content=text)]
            )
            tokens = self.tokenizer.encode_chat_completion(completion_request).tokens
            
            # Gennem indlejringslag
            with torch.no_grad():
                token_embeddings = self.model.tok_embeddings(torch.tensor(tokens))
                # Opret sætningsindlejring med gennemsnitlig pooling
                sentence_embedding = token_embeddings.mean(dim=0)
            
            self.logger.info(f"Indlejring oprettet med form: {sentence_embedding.shape}")
            return sentence_embedding
            
        except Exception as e:
            self.logger.error(f"Fejl under oprettelse af indlejring: {e}", exc_info=True)
            # Returner en standardværdi
            return torch.zeros(self.model.args.dim)
    
    def vectorize_assistant_output(self, assistant_outputs: List[AssistantOutput]) -> List[AssistantOutput]:
        """Vektorisér assistentoutput"""
        self.logger.info("Vektoriserer assistentoutput...")
        for assistant_output in assistant_outputs:
            self.logger.info(f"Vektoriserer output for assistent {assistant_output.assistant_id}")
            embedding_vector = self._create_embedding(assistant_output.comment_text)
            assistant_output.embedding_vector = embedding_vector
        
        self.logger.info(f"Vektoriseret {len(assistant_outputs)} assistentoutput")
        return assistant_outputs
    
    def route_to_experts(self, assistant_outputs: List[AssistantOutput], max_experts: MaxExpertsList) -> torch.Tensor:
        """Rut indlejret assistentoutput til ekspertsystemet"""
        self.logger.info("Ruter assistentoutput til eksperter.")
        # Kombiner alle assistentindlejringer med vægtet gennemsnit
        if not assistant_outputs:
            self.logger.warning("Ingen assistentoutput at rute. Returnerer en nul-tensor.")
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
        
        self.logger.info(f"Kombineret assistentindlejringer med samlet vægt {total_weight:.4f}")
        # Tilføj batch-dimension
        routing_vector = combined_embedding.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden_dim]
        
        return routing_vector


class FusionController:
    """Fusionskontrolsystem"""
    
    def __init__(self):
        self.fusion_degrees: Dict[int, float] = {}
        self.logger = logging.getLogger(__name__)
    
    def _calculate_output_similarity(self, first_output: torch.Tensor, second_output: torch.Tensor) -> float:
        """Beregn lighed mellem to output"""
        self.logger.info("Beregner outputlighed.")
        # Beregn cosinus-lighed
        first_flat = first_output.reshape(-1)
        second_flat = second_output.reshape(-1)
        
        similarity = F.cosine_similarity(first_flat, second_flat, dim=0)
        self.logger.info(f"Beregnet cosinus-lighed: {similarity.item():.4f}")
        return similarity.item()
    
    def _calculate_novelty_score(self, first_output: torch.Tensor, second_output: torch.Tensor) -> float:
        """Beregn nyhedsscore"""
        self.logger.info("Beregner nyhedsscore.")
        # Beregn nyhed baseret på L2-afstand
        l2_distance = torch.norm(first_output - second_output, p=2)
        max_distance = torch.norm(first_output, p=2) + torch.norm(second_output, p=2)
        
        if max_distance == 0:
            self.logger.warning("Maksimal afstand er nul, kan ikke beregne nyhedsscore.")
            return 0.0
        
        novelty_score = (l2_distance / max_distance).item()
        self.logger.info(f"Beregnet nyhedsscore: {novelty_score:.4f}")
        return min(1.0, novelty_score)
    
    def calculate_fusion_degree(self, first_output: torch.Tensor, second_output: torch.Tensor, max_experts: MaxExpertsList) -> List[FusionMetrics]:
        """Beregn fusion_degree ved at sammenligne to output"""
        self.logger.info("Beregner fusionsgrad for aktive eksperter.")
        fusion_metrics = []
        
        # Beregn samlet outputlighed og nyhed
        similarity_score = self._calculate_output_similarity(first_output, second_output)
        novelty_score = self._calculate_novelty_score(first_output, second_output)
        
        for expert_output in max_experts.experts:
            expert_id = expert_output.expert_id
            
            # Hent eksisterende fusion_degree
            current_fusion_degree = self.fusion_degrees.get(expert_id, 1.0)
            
            # Dynamisk justeringsformel
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
            
            # Gem opdateret fusion_degree
            self.fusion_degrees[expert_id] = new_fusion_degree
            expert_output.fusion_degree = new_fusion_degree
            self.logger.info(f"Ekspert {expert_id}: Gammel fusionsgrad={current_fusion_degree:.4f}, Ny fusionsgrad={new_fusion_degree:.4f}")
        
        self.logger.info(f"Beregnet fusionsgrader for {len(fusion_metrics)} eksperter")
        return fusion_metrics
    
    def _adjust_fusion_degree(self, current_degree: float, similarity: float, novelty: float, activation: float) -> float:
        """Dynamisk justeringsformel for fusion_degree"""
        # Høj nyhed + moderat lighed = høj fusion_degree
        # Lav nyhed + høj lighed = lav fusion_degree
        
        novelty_factor = novelty * 0.6  # Nyhedsvægt
        similarity_factor = (1.0 - similarity) * 0.3  # Forskelvægt
        activation_factor = activation * 0.1  # Aktiveringsvægt
        
        adjustment = novelty_factor + similarity_factor + activation_factor
        
        # Anvend justering på nuværende grad
        new_degree = current_degree + (adjustment - 0.5) * 0.1
        
        # Begræns interval (0-1)
        new_degree = max(0.0, min(1.0, new_degree))
        
        return new_degree
    
    def apply_fusion_weights(self, first_output: torch.Tensor, second_output: torch.Tensor, fusion_metrics: List[FusionMetrics]) -> torch.Tensor:
        """Generer det endelige output ved at anvende fusionsgrad"""
        self.logger.info("Anvender fusionsvægte til at generere det endelige output.")
        if not fusion_metrics:
            self.logger.warning("Ingen fusionsmetrikker tilgængelige. Returnerer første output.")
            return first_output
        
        # Beregn samlet fusionsvægt
        total_fusion_weight = sum(metric.fusion_degree for metric in fusion_metrics)
        
        if total_fusion_weight == 0:
            self.logger.warning("Samlet fusionsvægt er nul. Returnerer første output.")
            return first_output
        
        # Beregn normaliserede vægte
        alpha = total_fusion_weight / len(fusion_metrics)  # gennemsnitlig fusionsgrad
        beta = 1.0 - alpha  # oprindelig outputindflydelse
        
        # Generer det endelige output
        fused_output = beta * first_output + alpha * second_output
        
        self.logger.info(f"Anvendt fusion med alpha={alpha:.3f}, beta={beta:.3f}")
        return fused_output


class SurveySystem:
    """Undersøgelsessystem"""
    
    def __init__(self, fusion_controller: FusionController):
        self.fusion_controller = fusion_controller
        self.survey_responses: List[SurveyResponse] = []
        self.adjustment_rates = {
            'positive': 0.05,  # Stigningsrate for positiv feedback
            'negative': -0.1   # Faldrate for negativ feedback (stærkere)
        }
        self.logger = logging.getLogger(__name__)
    
    def collect_survey_response(self, question: str, relevance_scores: List[int], user_id: Optional[str] = None) -> SurveyResponse:
        """Indsaml undersøgelsessvar fra brugeren"""
        self.logger.info(f"Indsamler undersøgelsessvar for spørgsmål: '{question}'")
        survey_response = SurveyResponse(
            question=question,
            relevance_scores=relevance_scores,
            timestamp=datetime.now(),
            user_id=user_id
        )
        
        self.survey_responses.append(survey_response)
        self.logger.info(f"Indsamlet undersøgelsessvar med scores: {relevance_scores}")
        
        return survey_response
    
    def update_fusion_degrees(self, recent_responses: List[SurveyResponse]) -> Dict[int, float]:
        """Opdater fusion_degree baseret på undersøgelsesresultater"""
        self.logger.info("Opdaterer fusionsgrader baseret på undersøgelsessvar.")
        updated_degrees = {}
        
        for response in recent_responses:
            avg_relevance = sum(response.relevance_scores) / len(response.relevance_scores)
            self.logger.info(f"Gennemsnitlig relevansscore fra svar: {avg_relevance:.2f}")
            
            # Justering baseret på gennemsnitlig score
            if avg_relevance >= 4.0:  # Positiv feedback
                adjustment_rate = self.adjustment_rates['positive']
                self.logger.info(f"Positiv feedback registreret. Anvender justeringsrate: {adjustment_rate}")
            elif avg_relevance <= 2.0:  # Negativ feedback
                adjustment_rate = self.adjustment_rates['negative']
                self.logger.info(f"Negativ feedback registreret. Anvender justeringsrate: {adjustment_rate}")
            else:  # Neutral
                adjustment_rate = 0.0
                self.logger.info("Neutral feedback registreret. Ingen justering.")
            
            # Opdater fusion_degree for alle eksperter
            for expert_id in self.fusion_controller.fusion_degrees:
                current_degree = self.fusion_controller.fusion_degrees[expert_id]
                new_degree = max(0.0, min(1.0, current_degree + adjustment_rate))
                self.fusion_controller.fusion_degrees[expert_id] = new_degree
                updated_degrees[expert_id] = new_degree
                self.logger.info(f"Ekspert {expert_id} fusionsgrad opdateret til {new_degree:.4f}")
        
        self.logger.info(f"Opdateret fusionsgrader baseret på {len(recent_responses)} svar")
        return updated_degrees


class DualMoEPipeline:
    """DualMoE pipeline hovedklasse"""
    
    def __init__(self, model_path: str, tokenizer_path: str):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initialiserer DualMoE Pipeline...")
        # Indlæs model
        self.tokenizer = MistralTokenizer.from_file(tokenizer_path)
        self.model = Transformer.from_folder(model_path)
        self.logger.info("Model og tokenizer indlæst succesfuldt.")
        
        # --- Modificeret sektion ---
        # Opretter ikke længere falske Eksperter og Gate.
        # Sender den rigtige model til DualOutputMoE controlleren.
        self.experts_system = DualOutputMoE(self.model)
        # --- Modificering slut ---
        
        self.assistants_system = AssistantRouter(self.tokenizer, self.model)
        self.embedding_processor = EmbeddingProcessor(self.model, self.tokenizer)
        self.fusion_controller = FusionController()
        self.survey_system = SurveySystem(self.fusion_controller)
        
        self.logger.info("Alle pipeline-komponenter initialiseret.")
    
    def process_query(self, user_query: str, enable_survey: bool = False) -> Dict[str, Any]:
        """Kør hele pipelinen"""
        self.logger.info(f"Begynder behandling af forespørgsel: {user_query}")
        
        # 1. Forbehandling af input (returnerer token-ID'er og sekvenslængder)
        self.logger.info("Trin 1: Forbehandler brugerforespørgsel.")
        tokens, seqlens = self._preprocess_query(user_query)
        
        # 2. Første MoE-pas (input er ændret til token-ID'er)
        self.logger.info("Trin 2: Udfører første MoE-pas.")
        first_output_embedding, max_experts = self.experts_system.forward_first_pass(tokens, seqlens)
        first_output_text = self._decode_output_to_text(first_output_embedding)
        self.logger.info(f"Første pas afsluttet. Aktive eksperter: {[e.expert_id for e in max_experts.experts]}")

        # 3. Behandling i assistentsystemet
        self.logger.info("Trin 3: Behandler gennem assistentsystemet.")
        assistant_outputs = self.assistants_system.bridge_through_assistants(
            user_query, 
            first_output_text,
            max_experts
        )
        
        # 5. Indlejringsbehandling
        self.logger.info("Trin 5: Forbereder input til andet pas.")
        second_pass_input_embedding = self._prepare_second_pass_input(
            user_query, first_output_text, assistant_outputs
        )
        
        # 6. Kørsel af andet MoE-pas
        self.logger.info("Trin 6: Udfører andet MoE-pas.")
        second_output = self.experts_system.forward_second_pass(second_pass_input_embedding)
        
        # 8. Fusionsbehandling (kald til ekstern FusionController)
        self.logger.info("Trin 8: Beregner og anvender fusionsgrader.")
        fusion_metrics = self.fusion_controller.calculate_fusion_degree(
            first_output_embedding, second_output, max_experts
        )
        fusion_result = self.fusion_controller.apply_fusion_weights(
            first_output_embedding, second_output, fusion_metrics
        )
        
        # 9. Pakning af resultater
        self.logger.info("Trin 9: Pakker de endelige resultater.")
        result = {
            'user_query': user_query,
            'first_output': first_output_embedding,
            'second_output': second_output,
            'fused_output': fusion_result,
            'assistant_outputs': assistant_outputs,
            'fusion_metrics': fusion_metrics,
            'max_experts': max_experts,
            'survey_ready': enable_survey
        }
        
        self._last_query_result = result # Gem det seneste resultat til reference i undersøgelsen
        
        # 10. Undersøgelse (valgfrit)
        if enable_survey:
            self.logger.info("Trin 10: Undersøgelsestilstand aktiveret. Forbereder indsamling af feedback.")
            # SurveySystem kræver tekstoutput, så dekod de endelige resultater.
            first_text = self._decode_output_to_text(result['first_output'])
            second_text = self._decode_output_to_text(result['second_output'])
            fused_text = self._decode_output_to_text(result['fused_output'])
            
            # Kald SurveySystems grænseflade til indsamling af undersøgelser.
            # Dette er kun en demonstration; i en rigtig app ville du indsamle dette fra en bruger.
            self.logger.info("Simulerer indsamling af undersøgelsessvar (manuel indsendelse kræves).")

        self.logger.info("Pipelinebehandling afsluttet.")
        return result
    
    
    def _prepare_second_pass_input(self, user_query: str, first_output_text: str, assistant_outputs: List[Any]) -> torch.Tensor:
        """
        Forbered input-tensoren til det andet pas baseret på assistentens kommentarer.
        Generer en 'sekvensindlejring' af tokens i stedet for en enkelt 'konceptvektor'.
        """
        self.logger.info("Forbereder input til det andet pas baseret på assistentkommentarer...")

        # 1. Kombiner assistentkommentarer til en enkelt tekst
        assistant_comments = "\n".join([f"- {out.comment_text}" for out in assistant_outputs])
        self.logger.info(f"Kombinerede assistentkommentarer: \n{assistant_comments}")

        # 2. Sammensæt en ny prompt til den anden inferens
        second_pass_prompt = f"""
## Oprindeligt spørgsmål
{user_query}

## Første svar
{first_output_text}

## Ekspertkommentarer (nye perspektiver)
{assistant_comments}

## Omfattende gennemgang og endeligt svar
"""
        self.logger.info("Sammensat prompt til andet pas.")

        # 3. Tokeniser prompten
        completion_request = ChatCompletionRequest(
            messages=[UserMessage(content=second_pass_prompt)]
        )
        tokens = self.tokenizer.encode_chat_completion(completion_request).tokens
        token_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0) # [1, seq_len] form
        self.logger.info(f"Tokeniseret prompt til andet pas til {token_tensor.shape[1]} tokens.")

        # 4. Konverter token-sekvens til indlejringssekvens
        with torch.no_grad():
            input_embedding = self.model.tok_embeddings(token_tensor) # [1, seq_len, hidden_dim]

        self.logger.info(f"Input til andet pas forberedt med form: {input_embedding.shape}")
        return input_embedding
    
    def submit_survey(self, relevance_scores: List[int], user_id: Optional[str] = None) -> Dict[int, float]:
        """
        (Denne funktion kan bruges til manuelt at indsende undersøgelsesscores udefra.)
        Indsend undersøgelse og opdater fusion_degree
        """
        self.logger.info(f"Indsender manuel undersøgelse med scores: {relevance_scores}")
        if not hasattr(self, '_last_query_result'):
            self.logger.error("Forsøgte at indsende undersøgelse, men ingen seneste forespørgsel fundet.")
            raise ValueError("Ingen seneste forespørgsel at evaluere.")
        
        last_query = self._last_query_result['user_query']
        
        # Kald SurveySystems svarindsamlings- og opdateringslogik direkte
        survey_response = self.survey_system.collect_survey_response(
            question=last_query,
            relevance_scores=relevance_scores,
            user_id=user_id
        )
        
        # Opdater fusion_degree baseret på undersøgelsesresultaterne
        updated_degrees = self.survey_system.update_fusion_degrees([survey_response])
        self.logger.info(f"Fusionsgrader opdateret via manuel undersøgelsesindsendelse: {updated_degrees}")
        
        return updated_degrees
    
    def _decode_output_to_text(self, output_embedding: torch.Tensor) -> str:
        """Dekoder modellens endelige outputindlejring til tekst."""
        self.logger.info("Dekoder outputindlejring til tekst.")
        # Konverter modellens outputindlejring til scores (logits) over ordforrådet
        logits = F.linear(output_embedding, self.model.tok_embeddings.weight) # [batch, seq_len, vocab_size]
        # Vælg token-ID'erne med de højeste scores
        next_token_ids = torch.argmax(logits, dim=-1) # [batch, seq_len]

        # Dekod kun det første resultat i batchen
        decoded_text = self.tokenizer.decode(next_token_ids[0].tolist())
        self.logger.info("Dekodning afsluttet.")
        return decoded_text

    def _preprocess_query(self, query: str) -> Tuple[torch.Tensor, List[int]]:
        """Forbehandler forespørgsel til token-ID'er og sekvenslængder."""
        self.logger.info("Forbehandler forespørgsel til token-ID'er.")
        completion_request = ChatCompletionRequest(
            messages=[UserMessage(content=query)]
        )
        tokens = self.tokenizer.encode_chat_completion(completion_request).tokens
        
        # seqlens er en liste over længderne af hver sekvens til batchbehandling.
        # Her er batchen 1, så det er den samlede token-længde.
        seqlens = [len(tokens)]
        self.logger.info(f"Forespørgsel forbehandlet til {seqlens[0]} tokens.")
        
        return torch.tensor(tokens, dtype=torch.long), seqlens


# Eksempel på brug
if __name__ == "__main__":
    # Opsæt modelstier
    # BEMÆRK: Opdater disse stier, så de peger på din faktiske model og tokenizer-mappe.
    model_path = "C:\\Users\\jungh\\Desktop\\fuck\\mistral-inference-main\\model"  # Skift til den faktiske modelsti
    tokenizer_path = "C:\\Users\\jungh\\Desktop\\fuck\\mistral-inference-main\\model\\tokenizer.model"  # Skift til den faktiske tokenizer-sti
    
    logger = logging.getLogger()
    
    try:
        # Initialiser pipeline
        pipeline = DualMoEPipeline(model_path, tokenizer_path)
        
        # Behandl forespørgsel
        user_query = "recommend a report topic about ai"
        result = pipeline.process_query(user_query, enable_survey=True)
        
        # Udskriv resultater
        logger.info("\n--- PIPELINE-RESULTATER ---")
        logger.info(f"Forespørgsel: {result['user_query']}")
        logger.info(f"Aktive eksperter: {[expert.expert_id for expert in result['max_experts'].experts]}")
        logger.info(f"Antal assistentkommentarer: {len(result['assistant_outputs'])}")
        
        for i, assistant_output in enumerate(result['assistant_outputs']):
            logger.info(f"--- Assistentkommentar {i+1} ---")
            logger.info(assistant_output.comment_text)
        
        logger.info("\n--- Fusionsmetrikker ---")
        for metric in result['fusion_metrics']:
            logger.info(f"Ekspert {metric.expert_id}: Lighed={metric.similarity_score:.4f}, Nyhed={metric.novelty_score:.4f}, Fusionsgrad={metric.fusion_degree:.4f}")

        # Eksempel på indsendelse af undersøgelse
        logger.info("\n--- SIMULERER INDSENDELSE AF UNDERSØGELSE ---")
        # I en rigtig app ville disse scores komme fra brugerinput.
        relevance_scores = [4, 5, 3, 4, 5]  # 1-5 point
        logger.info(f"Indsender manuelle relevansscores: {relevance_scores}")
        updated_degrees = pipeline.submit_survey(relevance_scores)
        logger.info(f"Opdaterede fusionsgrader efter undersøgelse: {updated_degrees}")
        
    except FileNotFoundError as e:
        logger.error(f"Fejl: Model- eller tokenizer-fil ikke fundet. Sørg for, at stierne er korrekte.")
        logger.error(f"Detaljer: {e}")
    except Exception as e:
        logger.error(f"Der opstod en uventet fejl under pipeline-kørsel: {e}", exc_info=True)
        logger.error("Sørg for, at mistral-inference er installeret ('pip install mistral-inference'), og at modelstierne er korrekte.")