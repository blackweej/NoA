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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ExpertOutput:
    """Dataklasse, der indeholder ekspertoutputoplysninger"""
    expert_id: int
    output_tensor: torch.Tensor
    weight: float
    activation_score: float
    fusion_degree: float = 1.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MaxExpertsList:
    """Liste over maksimalt aktiverede eksperter"""
    max_count: int = 8
    experts: List[ExpertOutput] = field(default_factory=list)
    threshold: float = 0.1
    
    def add_expert(self, expert_output: ExpertOutput):
        """Tilføj ekspert (oprethold sorteret tilstand)"""
        if expert_output.activation_score >= self.threshold:
            self.experts.append(expert_output)
            self.experts.sort(key=lambda x: x.activation_score, reverse=True)
            if len(self.experts) > self.max_count:
                self.experts = self.experts[:self.max_count]
    
    def get_top_experts(self, count: int) -> List[ExpertOutput]:
        """Returner de øverste N eksperter"""
        return self.experts[:min(count, len(self.experts))]


@dataclass
class AssistantConfig:
    """Assistentens konfigurationsoplysninger"""
    assistant_id: str
    prompt_template: str
    classification_threshold: float
    expert_affinity: Dict[int, float] = field(default_factory=dict)
    specialization_area: str = ""
    activation_pattern: List[float] = field(default_factory=list)
    
    def calculate_affinity_score(self, expert_id: int) -> float:
        """Beregn affinitetsscore med en specifik ekspert"""
        return self.expert_affinity.get(expert_id, 0.0)


@dataclass
class AssistantOutput:
    """Assistentens outputoplysninger"""
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
    Assistent-routersystem
    Dynamisk oprettelse og styring af assistenter baseret på max_experts
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
        
        # Gem dynamisk oprettede assistenter
        self.active_assistants: Dict[str, AssistantConfig] = {}
        self.assistant_history: List[AssistantOutput] = []
        
        # Prompt-skabeloner til oprettelse af assistenter
        self.base_prompt_templates = {
            'analytical': "Du er en assistent specialiseret i analytisk tænkning. Giv venligst en supplerende kommentar til det givne spørgsmål fra et logisk og systematisk perspektiv.",
            'creative': "Du er en assistent specialiseret i kreativ tænkning. Giv venligst en supplerende kommentar til det givne spørgsmål fra et originalt og innovativt perspektiv.",
            'practical': "Du er en assistent specialiseret i en praktisk tilgang. Giv venligst en supplerende kommentar til det givne spørgsmål fra et realistisk og anvendeligt perspektiv.",
            'critical': "Du er en assistent specialiseret i kritisk tænkning. Giv venligst en supplerende kommentar til det givne spørgsmål fra et perspektiv med at finde problemer og foreslå forbedringer.",
            'synthetic': "Du er en assistent specialiseret i syntetisk tænkning. Giv venligst en supplerende kommentar til det givne spørgsmål ved at integrere forskellige perspektiver."
        }
        
        logger.info(f"AssistantRouter initialiseret med {max_assistants} maks. assistenter og '{clustering_method}' klyngemetode.")
    
    def bridge_through_assistants(self,
                            original_question: str,
                            first_output_text: str,
                            max_experts: MaxExpertsList) -> List[AssistantOutput]:
        """
        Assistent-systemets hovedbehandlingsfunktion
        
        Args:
            original_question: Oprindeligt spørgsmål
            first_output_text: Første MoE-output
            max_experts: Liste over maksimalt aktiverede eksperter
            
        Returns:
            assistant_outputs: Liste over assistenternes output
        """
        logger.info("Starter bridge_through_assistants-processen...")
        start_time = datetime.now()
        
        try:
            # 1. Dynamisk oprettelse af assistenter
            logger.info("Trin 1: Opretter assistenter dynamisk.")
            assistants = self._create_assistants(max_experts)
            if not assistants:
                logger.warning("Ingen assistenter blev oprettet. Afslutter processen.")
                return []
            logger.info(f"Oprettede {len(assistants)} assistenter baseret på ekspertmønstre.")
            
            # 2. Generering af kommentarer for hver assistent
            logger.info("Trin 2: Genererer kommentarer for hver assistent.")
            assistant_outputs = []
            for assistant in assistants:
                try:
                    comment_start = datetime.now()
                    
                    # Sammensætning af prompt
                    prompt = self._build_comment_prompt(
                        assistant.prompt_template,
                        original_question,
                        first_output_text,
                        assistant.expert_affinity
                    )
                    
                    # Kald til genereringsfunktion
                    comment_output = self._generate_comment(prompt, assistant.assistant_id)
                    
                    # Beregning af tillidsscore
                    confidence_score = self._calculate_confidence(comment_output, assistant.expert_affinity)
                    
                    # Identifikation af relaterede eksperter
                    related_experts = [
                        expert_id for expert_id, affinity in assistant.expert_affinity.items()
                        if affinity > assistant.classification_threshold
                    ]
                    
                    # Oprettelse af assistent-output
                    generation_time = (datetime.now() - comment_start).total_seconds()
                    
                    assistant_output = AssistantOutput(
                        assistant_id=assistant.assistant_id,
                        comment_text=comment_output,
                        confidence_score=confidence_score,
                        related_experts=related_experts,
                        generation_time=generation_time
                    )
                    
                    assistant_outputs.append(assistant_output)
                    logger.info(f"Genereret kommentar fra {assistant.assistant_id} på {generation_time:.2f}s med tillidsscore {confidence_score:.2f}.")
                    
                except Exception as e:
                    logger.error(f"Fejl under generering af kommentar for {assistant.assistant_id}: {str(e)}", exc_info=True)
                    continue
            
            # 3. Filtrering af outputkvalitet
            logger.info("Trin 3: Filtrerer assistent-output.")
            filtered_outputs = self._filter_assistant_outputs(assistant_outputs)
            logger.info(f"Filtrerede outputs fra {len(assistant_outputs)} til {len(filtered_outputs)}.")
            
            # 4. Opdatering af historik
            self.assistant_history.extend(filtered_outputs)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Assistentbehandling afsluttet på {processing_time:.2f}s.")
            
            return filtered_outputs
            
        except Exception as e:
            logger.error(f"Fejl i bridge_through_assistants: {str(e)}", exc_info=True)
            return []
    
    def _create_assistants(self, max_experts: MaxExpertsList) -> List[AssistantConfig]:
        """
        Dynamisk oprettelse af assistenter baseret på max_experts
        
        Args:
            max_experts: Liste over maksimalt aktiverede eksperter
            
        Returns:
            assistants: Liste over oprettede assistentkonfigurationer
        """
        try:
            if not max_experts.experts:
                logger.warning("Ingen eksperter tilgængelige for oprettelse af assistenter.")
                return []
            
            # 1. Analyse af ekspertaktiveringsmønstre
            self._analyze_expert_patterns(max_experts)
            
            # 2. Klyngning af eksperter til gruppering
            logger.info("Klynger eksperter for at danne assistentgrupper...")
            expert_clusters = self._cluster_experts_by_activation(max_experts)
            logger.info(f"Identificerede {len(expert_clusters)} ekspertklynger.")
            
            assistants = []
            template_keys = list(self.base_prompt_templates.keys())
            
            # 3. Oprettelse af en assistent for hver klynge
            for cluster_id, expert_group in expert_clusters.items():
                if not expert_group:
                    continue
                    
                # Analyse af klyngekarakteristika
                cluster_characteristics = self._analyze_cluster_characteristics(expert_group)
                
                # Valg af passende skabelon
                template_key = template_keys[cluster_id % len(template_keys)]
                
                assistant_config = AssistantConfig(
                    assistant_id=f"assistant_{cluster_id}_{template_key}",
                    prompt_template=self.base_prompt_templates[template_key],
                    classification_threshold=self._calculate_dynamic_threshold(expert_group),
                    specialization_area=template_key,
                    activation_pattern=cluster_characteristics['pattern']
                )
                
                # 4. Beregning af affinitet med eksperter
                for expert_output in expert_group:
                    affinity_score = self._calculate_expert_affinity(
                        expert_output, 
                        cluster_characteristics
                    )
                    assistant_config.expert_affinity[expert_output.expert_id] = affinity_score
                
                assistants.append(assistant_config)
                
                # Gem aktiv assistent
                self.active_assistants[assistant_config.assistant_id] = assistant_config
                
                logger.info(f"Oprettet assistent {assistant_config.assistant_id} for klynge {cluster_id} med {len(expert_group)} eksperter.")
            
            return assistants[:self.max_assistants]
            
        except Exception as e:
            logger.error(f"Fejl under oprettelse af assistenter: {str(e)}", exc_info=True)
            return []
    
    def _analyze_expert_patterns(self, max_experts: MaxExpertsList) -> Dict[str, Any]:
        """Analyse af eksperters aktiveringsmønstre"""
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
        logger.info(f"Analyse af ekspertmønstre: Gennemsnitlig aktivering={patterns['mean_activation']:.2f}, Fordeling='{patterns['activation_distribution']}'")
        return patterns
    
    def _cluster_experts_by_activation(self, max_experts: MaxExpertsList) -> Dict[int, List[ExpertOutput]]:
        """Klyngning af eksperter baseret på aktiveringsscore"""
        if len(max_experts.experts) <= 1:
            logger.info("Kun én eller færre eksperter; springer klyngning over.")
            return {0: max_experts.experts}
        
        # Oprettelse af funktionsvektor (activation_score, weight, fusion_degree)
        features = []
        for expert in max_experts.experts:
            features.append([
                expert.activation_score,
                expert.weight,
                expert.fusion_degree
            ])
        
        features_array = np.array(features)
        
        # Dynamisk bestemmelse af antal klynger
        n_clusters = min(self.max_assistants, len(max_experts.experts))
        
        logger.info(f"Udfører '{self.clustering_method}' klyngning med n_clusters={n_clusters}.")
        if self.clustering_method == 'kmeans':
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features_array)
        else:
            # Simpel tærskelbaseret klyngning
            cluster_labels = self._threshold_based_clustering(features_array, n_clusters)
        
        # Gruppering af eksperter efter klynge
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(max_experts.experts[i])
        
        return clusters
    
    def _threshold_based_clustering(self, features: np.ndarray, n_clusters: int) -> np.ndarray:
        """Simpel tærskelbaseret klyngning"""
        activation_scores = features[:, 0]
        sorted_indices = np.argsort(activation_scores)[::-1]  # Faldende sortering
        
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
        """Analyse af klyngekarakteristika"""
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
        """Beregning af dynamisk tærskel baseret på klyngekarakteristika"""
        if not expert_group:
            return 0.1
        
        activation_scores = [expert.activation_score for expert in expert_group]
        mean_activation = np.mean(activation_scores)
        std_activation = np.std(activation_scores)
        
        # Brug en højere tærskel, hvis standardafvigelsen er stor
        dynamic_threshold = max(0.05, mean_activation - 0.5 * std_activation)
        
        return min(dynamic_threshold, 0.8)  # Begræns til maksimalt 0.8
    
    def _calculate_expert_affinity(self, 
                                 expert_output: ExpertOutput,
                                 cluster_characteristics: Dict[str, Any]) -> float:
        """Beregning af affinitet mellem ekspert og klynge"""
        # Affinitet baseret på aktiveringsscore
        activation_affinity = expert_output.activation_score / cluster_characteristics['avg_activation']
        
        # Affinitet baseret på vægt
        weight_affinity = expert_output.weight
        
        # Affinitet baseret på fusionsgrad
        fusion_affinity = expert_output.fusion_degree
        
        # Beregning af endelig affinitet med vægtet gennemsnit
        total_affinity = (
            0.4 * activation_affinity +
            0.3 * weight_affinity +
            0.3 * fusion_affinity
        )
        
        return min(total_affinity, 1.0)
    
    def _categorize_activation_distribution(self, activation_scores: List[float]) -> str:
        """Kategorisering af aktiveringsscorefordeling"""
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
                            first_output_text: str,
                            expert_affinity: Dict[int, float]) -> str:
        """Sammensætning af prompt til generering af kommentar"""
        
        top_experts = sorted(expert_affinity.items(), key=lambda x: x[1], reverse=True)[:3]
        expert_info = ", ".join([f"Ekspert{expert_id}(affinitet:{affinity:.2f})" 
                               for expert_id, affinity in top_experts])
        
        prompt = f"""
{template}

Oprindeligt spørgsmål: {original_question}

Første svar: {first_output_text}

Relaterede ekspertinformationer: {expert_info}

Baseret på ovenstående informationer, generer venligst en kommentar fra et supplerende perspektiv på det oprindelige spørgsmål.
Kommentaren skal være specifik og praktisk og give ny indsigt, der adskiller sig fra det oprindelige svar.

Kommentar:"""
        
        logger.debug(f"Bygget prompt for assistent med skabelon: {template.split('.')[0]}")
        return prompt
    
    def _generate_comment(self, prompt: str, assistant_id: str) -> str:
        """Generering af kommentar ved kald til den faktiske genereringsfunktion"""
        logger.debug(f"Starter generering af kommentar for {assistant_id}...")
        try:
            # 1. Tokenisering af prompt
            tokenized_prompt = self.tokenizer.encode(prompt)
            
            # 2. Kald til genereringsfunktion
            output_tokens, _ = generate(
                [tokenized_prompt],
                self.model,
                max_tokens=self.comment_max_tokens,
                temperature=self.temperature,
                eos_id=self.tokenizer.eos_id
            )
            
            # 3. Dekodning
            raw_comment = self.tokenizer.decode(output_tokens[0])
            
            # 4. Efterbehandling af kommentar
            processed_comment = self._post_process_comment(raw_comment)
            
            # 5. Formatering
            formatted_comment = self._format_as_comment(processed_comment, assistant_id)
            
            logger.debug(f"Kommentar genereret og formateret for {assistant_id}.")
            return formatted_comment
            
        except Exception as e:
            logger.error(f"Fejl under generering af kommentar for {assistant_id}: {str(e)}", exc_info=True)
            return f"[{assistant_id}] Der opstod en fejl under generering af kommentar."
    
    def _post_process_comment(self, raw_comment: str) -> str:
        """Efterbehandling af genereret kommentar"""
        # Fjern unødvendige tokens
        comment = raw_comment.strip()
        
        # Fjern prompt-gentagelse
        if "Kommentar:" in comment:
            comment = comment.split("Kommentar:")[-1].strip()
        
        # Længdebegrænsning
        if len(comment) > 500:
            comment = comment[:500] + "..."
        
        # Håndtering af tom kommentar
        if not comment:
            comment = "Der er yderligere aspekter at overveje."
        
        return comment
    
    def _format_as_comment(self, comment_text: str, assistant_id: str) -> str:
        """Formatering af kommentar til det angivne format"""
        return f"[{assistant_id}] Supplerende perspektiv på det oprindelige spørgsmål: {comment_text}"
    
    def _calculate_confidence(self, comment_text: str, expert_affinity: Dict[int, float]) -> float:
        """Beregning af kommentarens tillidsscore"""
        # 1. Tillid baseret på tekstlængde
        length_score = min(len(comment_text.split()) / 50, 1.0)
        
        # 2. Tillid baseret på ekspertaffinitet
        affinity_score = np.mean(list(expert_affinity.values())) if expert_affinity else 0.0
        
        # 3. Tillid baseret på nøgleord (simpel heuristik)
        confidence_keywords = ['analyse', 'overvejelse', 'perspektiv', 'forslag', 'metode', 'tilgang', 'forbedring']
        keyword_score = sum(1 for keyword in confidence_keywords if keyword in comment_text) / len(confidence_keywords)
        
        # 4. Beregning af samlet tillid
        total_confidence = (
            0.3 * length_score +
            0.4 * affinity_score +
            0.3 * keyword_score
        )
        
        logger.debug(f"Beregnet tillidsscore: Længde={length_score:.2f}, Affinitet={affinity_score:.2f}, Nøgleord={keyword_score:.2f} -> Samlet={total_confidence:.2f}")
        return min(total_confidence, 1.0)
    
    def _filter_assistant_outputs(self, assistant_outputs: List[AssistantOutput]) -> List[AssistantOutput]:
        """Filtrering af assistent-outputkvalitet"""
        if not assistant_outputs:
            return []
        
        # 1. Filtrering baseret på tillidsscore
        min_confidence = 0.3
        logger.debug(f"Filtrerer outputs under tillidstærsklen på {min_confidence}.")
        filtered_outputs = [
            output for output in assistant_outputs 
            if output.confidence_score >= min_confidence
        ]
        
        # 2. Fjernelse af dubletter (filtrering af lignende kommentarer)
        logger.debug("Fjerner duplikerede eller meget ens kommentarer.")
        unique_outputs = self._remove_duplicate_comments(filtered_outputs)
        
        # 3. Valg af de øverste N
        sorted_outputs = sorted(unique_outputs, key=lambda x: x.confidence_score, reverse=True)
        
        return sorted_outputs[:self.max_assistants]
    
    def _remove_duplicate_comments(self, outputs: List[AssistantOutput]) -> List[AssistantOutput]:
        """Fjernelse af duplikerede kommentarer"""
        if len(outputs) <= 1:
            return outputs
        
        unique_outputs = []
        seen_comments = set()
        
        for output in outputs:
            # Simpel duplikatkontrol (i praksis kræves en mere avanceret lighedskontrol)
            comment_key = output.comment_text[:100].lower()  # Brug de første 100 tegn som nøgle
            
            if comment_key not in seen_comments:
                seen_comments.add(comment_key)
                unique_outputs.append(output)
        
        return unique_outputs
    
    def get_assistant_statistics(self) -> Dict[str, Any]:
        """Statistiske oplysninger om assistent-systemet"""
        logger.info("Henter statistikker for assistent-systemet.")
        if not self.assistant_history:
            logger.warning("Ingen assistenthistorik tilgængelig for at generere statistikker.")
            return {"message": "Ingen assistenthistorik tilgængelig"}
        
        total_outputs = len(self.assistant_history)
        avg_confidence = np.mean([output.confidence_score for output in self.assistant_history])
        avg_generation_time = np.mean([output.generation_time for output in self.assistant_history])
        
        assistant_usage = {}
        for output in self.assistant_history:
            assistant_id = output.assistant_id
            if assistant_id not in assistant_usage:
                assistant_usage[assistant_id] = 0
            assistant_usage[assistant_id] += 1
        
        stats = {
            "total_outputs": total_outputs,
            "average_confidence": avg_confidence,
            "average_generation_time": avg_generation_time,
            "assistant_usage": assistant_usage,
            "active_assistants": len(self.active_assistants)
        }
        logger.info(f"Statistikker genereret: Samlet outputs={total_outputs}, Gennemsnitlig tillid={avg_confidence:.2f}")
        return stats
    
    def update_assistant_fusion_degrees(self, feedback_data: Dict[str, float]):
        """Opdatering af assistentens fusionsgrad baseret på feedback"""
        logger.info("Opdaterer assistentens fusionsgrader baseret på feedback.")
        for assistant_id, assistant_config in self.active_assistants.items():
            if assistant_id in feedback_data:
                feedback_score = feedback_data[assistant_id]
                
                # Justering af ekspertaffinitet baseret på feedbackscore
                adjustment_factor = (feedback_score - 0.5) * 0.1  # Område -0.05 ~ 0.05
                
                for expert_id in assistant_config.expert_affinity:
                    current_affinity = assistant_config.expert_affinity[expert_id]
                    new_affinity = max(0.0, min(1.0, current_affinity + adjustment_factor))
                    assistant_config.expert_affinity[expert_id] = new_affinity
                
                logger.info(f"Opdaterede fusionsgrader for {assistant_id} baseret på feedback: {feedback_score}")
    
    def reset_assistants(self):
        """Nulstilling af assistent-tilstand"""
        self.active_assistants.clear()
        self.assistant_history.clear()
        logger.info("Assistent-systemet er blevet nulstillet.")


# Eksempel på brug og testfunktion
def test_assistant_router():
    """Testfunktion for Assistant Router"""
    # Dette skal erstattes med en passende tokenizer og model ved faktisk brug
    print("Starter test af AssistantRouter...")
    
    # Oprettelse af mock-data
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
    
    print(f"Oprettelse af mock-eksperter fuldført: {len(mock_experts.experts)} stk.")
    
    # Ved faktisk brug initialiseres som følger:
    # tokenizer = MistralTokenizer.from_file("tokenizer.model")
    # model = Transformer.from_folder("model_path")
    # router = AssistantRouter(tokenizer, model)
    
    print("Test af AssistantRouter afsluttet.")


if __name__ == "__main__":
    test_assistant_router()