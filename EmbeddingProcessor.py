"""
EmbeddingProcessor for Dual MoE System
Kerne-modul til vektorisering af Assistant-output og routing til Expert-systemet

Leverer indlejringsbehandling og routing-funktionalitet ved hjælp af den eksisterende mistral-inference-struktur
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from datetime import datetime
import logging

# mistral-inference-moduler
from mistral_inference.transformer import Transformer
from mistral_inference.tokenizer import MistralTokenizer

# Konfigurer grundlæggende logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class AssistantOutput:
    """Outputstruktur for Assistant-systemet"""
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
    """Indlejringsbehandlet Assistant-output"""
    assistant_output: AssistantOutput
    embedding_vector: torch.Tensor  # [hidden_dim]
    similarity_scores: Dict[int, float]  # expert_id: similarity_score
    routing_weights: Dict[int, float]  # expert_id: routing_weight


@dataclass
class ExpertOutput:
    """Expert-outputstruktur (kompatibel med eksisterende system)"""
    expert_id: int
    output_tensor: torch.Tensor
    weight: float
    activation_score: float
    fusion_degree: float = 1.0


@dataclass
class MaxExpertsList:
    """Liste over maksimalt aktiverede Eksperter"""
    experts: List[ExpertOutput]
    max_count: int = 8
    threshold: float = 0.1


class EmbeddingProcessor:
    """
    Processor til vektorisering af Assistant-output og routing til Expert-systemet
    """
    
    def __init__(self, model: Transformer, tokenizer: MistralTokenizer, 
                 hidden_dim: int = 4096, similarity_threshold: float = 0.1):
        """
        Args:
            model: Mistral Transformer-model
            tokenizer: Mistral-tokenizer
            hidden_dim: Modellens skjulte dimension
            similarity_threshold: Tærskel for lighedsberegning
        """
        self.model = model
        self.tokenizer = tokenizer
        self.hidden_dim = hidden_dim
        self.similarity_threshold = similarity_threshold
        
        # Indlejringscache (ydelsesoptimering)
        self.embedding_cache = {}
        
        # --- Modificeret sektion ---
        # Ekspert-indlejringer starter ikke længere som en tom ordbog.
        self.expert_embeddings = self._precompute_expert_embeddings()
        # --- Modificering slut ---
        
        # Routinghistorik (til analyse)
        self.routing_history = []
        
        logger.info(f"EmbeddingProcessor initialiseret med hidden_dim={hidden_dim} og similarity_threshold={similarity_threshold}.")
    
    def _precompute_expert_embeddings(self) -> Dict[int, torch.Tensor]:
        """
        Forudberegner repræsentative indlejringer for alle Eksperter i modellen.
        Genererer identitetsvektorer ved hjælp af et vægtet gennemsnit af Ekspertens vægte.
        """
        logger.info("Starter forudberegning af Ekspert-indlejringer...")
        expert_embeddings = {}
        with torch.no_grad():
            # Itererer gennem alle lag i modellen
            for layer_idx, layer in enumerate(self.model.layers):
                if hasattr(layer, 'feed_forward') and hasattr(layer.feed_forward, 'experts'):
                    # Gennemsnit af vægtene for hver Ekspert til brug som en repræsentativ vektor
                    for expert_idx, expert_module in enumerate(layer.feed_forward.experts):
                        # Hent alle parametertensorer som w1, w2, w3 osv.
                        param_vectors = [p.view(-1) for p in expert_module.parameters()]
                        
                        if not param_vectors:
                            logger.warning(f"Ekspert {expert_idx} i lag {layer_idx} har ingen parametre, springer over.")
                            continue
                            
                        # Sammenkæd og gennemsnit alle parametre til en enkelt vektor
                        avg_vector = torch.cat(param_vectors).mean(dim=0, keepdim=True)
                        
                        # Da alle lag i Mixtral deler Eksperter med samme rolle, bruges Ekspert-indekset (i) uden lagadskillelse.
                        if expert_idx not in expert_embeddings:
                            # Normaliser og gem
                            expert_embeddings[expert_idx] = F.normalize(avg_vector, p=2, dim=-1).squeeze()
                            logger.debug(f"Forudberegnet indlejring for Ekspert {expert_idx} (fra lag {layer_idx}).")
        
        logger.info(f"Forudberegning afsluttet. {len(expert_embeddings)} unikke Ekspert-indlejringer blev oprettet.")
        return expert_embeddings
    
    
    def vectorize_assistant_output(self, assistant_outputs: List[AssistantOutput]) -> List[EmbeddedAssistantOutput]:
        """
        Vektoriserer Assistant-outputs
        
        Args:
            assistant_outputs: Liste over outputs fra Assistant-systemet
            
        Returns:
            embedded_outputs: Liste over indlejringsbehandlede outputs
        """
        embedded_outputs = []
        
        logger.info(f"Behandler {len(assistant_outputs)} Assistant-outputs til indlejring...")
        
        for i, assistant_output in enumerate(assistant_outputs):
            try:
                # 1. Opret tekstindlejring
                embedding_vector = self._create_embedding(assistant_output.comment_text)
                
                # 2. Beregn ligheder med relaterede Eksperter
                similarity_scores = self._calculate_expert_similarities(
                    embedding_vector, 
                    assistant_output.related_experts
                )
                
                # 3. Beregn routing-vægte
                routing_weights = self._calculate_routing_weights(
                    similarity_scores, 
                    assistant_output.confidence_score
                )
                
                # 4. Sammensæt resultat
                embedded_output = EmbeddedAssistantOutput(
                    assistant_output=assistant_output,
                    embedding_vector=embedding_vector,
                    similarity_scores=similarity_scores,
                    routing_weights=routing_weights
                )
                
                embedded_outputs.append(embedded_output)
                
                logger.info(f"  Assistent {i+1}/{len(assistant_outputs)} ({assistant_output.assistant_id}): "
                      f"indlejringsdim={embedding_vector.shape}, "
                      f"ligheder={len(similarity_scores)}, "
                      f"routing_vægte={len(routing_weights)}")
                
            except Exception as e:
                logger.error(f"Fejl under behandling af Assistant-output {i}: {e}", exc_info=True)
                continue
        
        logger.info(f"Vektorisering af {len(embedded_outputs)} Assistant-outputs afsluttet.")
        return embedded_outputs
    
    def route_to_experts(self, embedded_outputs: List[EmbeddedAssistantOutput], 
                        max_experts: MaxExpertsList) -> torch.Tensor:
        """
        Router indlejrede Assistant-outputs til Expert-systemet
        
        Args:
            embedded_outputs: Indlejringsbehandlede Assistant-outputs
            max_experts: Liste over maksimalt aktiverede Eksperter
            
        Returns:
            routing_vector: Routing-vektor, der skal sendes til Expert-systemet [1, 1, hidden_dim]
        """
        if not embedded_outputs:
            logger.warning("Ingen indlejrede outputs at route, returnerer nulvektor.")
            return torch.zeros(1, 1, self.hidden_dim)
        
        logger.info(f"Router {len(embedded_outputs)} indlejrede outputs til {len(max_experts.experts)} Eksperter...")
        
        # 1. Kombiner alle Assistant-indlejringer med et vægtet gennemsnit
        combined_embedding = self._weighted_combine_embeddings(embedded_outputs)
        
        # 2. Juster vægte baseret på Ekspert-affinitet
        expert_weights = self._calculate_expert_routing_weights(
            combined_embedding, 
            max_experts
        )
        
        # 3. Opret routing-vektor (antager batch_size=1, seq_len=1)
        routing_vector = self._create_routing_vector(combined_embedding, expert_weights)
        
        # 4. Gem routinghistorik
        self._save_routing_history(embedded_outputs, expert_weights, routing_vector)
        
        logger.info(f"Genereret routing-vektor: form={routing_vector.shape}, "
              f"norm={torch.norm(routing_vector).item():.4f}")
        
        return routing_vector
    
    def _create_embedding(self, text: str) -> torch.Tensor:
        """
        Konverterer tekst til en indlejringsvektor
        
        Args:
            text: Tekst, der skal indlejres
            
        Returns:
            embedding: Sætningsindlejringsvektor [hidden_dim]
        """
        # Tjek cache
        text_key = text[:200] # Brug en del af teksten som nøgle for at undgå for lange nøgler
        if text_key in self.embedding_cache:
            logger.debug(f"Indlæser indlejring fra cache for tekst: '{text_key[:50]}...'")
            return self.embedding_cache[text_key]
        
        logger.debug(f"Opretter ny indlejring for tekst: '{text_key[:50]}...'")
        try:
            # 1. Tokenisering
            tokens = self.tokenizer.encode(text, add_bos=True, add_eos=True)
            
            # 2. Tensorkonvertering
            token_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)  # [1, seq_len]
            
            # 3. Brug af modellens indlejringslag
            with torch.no_grad():
                token_embeddings = self.model.embed_tokens(token_tensor)  # [1, seq_len, hidden_dim]
                
                # 4. Opret sætningsindlejring med gennemsnitlig pooling (ekskluderer padding-tokens)
                valid_length = len(tokens)
                sentence_embedding = token_embeddings[0, :valid_length].mean(dim=0)  # [hidden_dim]
                
                # 5. Normalisering
                sentence_embedding = F.normalize(sentence_embedding, p=2, dim=0)
            
            # Gem i cache
            self.embedding_cache[text_key] = sentence_embedding
            
            return sentence_embedding
            
        except Exception as e:
            logger.error(f"Fejl under oprettelse af indlejring for tekst: {e}", exc_info=True)
            # Returner nulvektor ved fejl
            return torch.zeros(self.hidden_dim)
    
    def _calculate_expert_similarities(self, embedding_vector: torch.Tensor, 
                                     related_experts: List[int]) -> Dict[int, float]:
        """
        Beregner ligheder mellem indlejringsvektor og relaterede Eksperter
        
        Args:
            embedding_vector: Indlejringsvektor [hidden_dim]
            related_experts: Liste over relaterede Ekspert-ID'er
            
        Returns:
            similarity_scores: mapping af expert_id: similarity_score
        """
        similarity_scores = {}
        logger.debug(f"Beregner ligheder for {len(related_experts)} relaterede Eksperter.")
        
        for expert_id in related_experts:
            expert_embedding = self._get_expert_embedding(expert_id)
            
            # Beregn cosinuslighed
            similarity = F.cosine_similarity(
                embedding_vector.unsqueeze(0), 
                expert_embedding.unsqueeze(0)
            ).item()
            
            # Anvend tærskel
            if similarity >= self.similarity_threshold:
                similarity_scores[expert_id] = similarity
        
        logger.debug(f"Fundet {len(similarity_scores)} Eksperter over lighedstærsklen.")
        return similarity_scores
    
    def _calculate_routing_weights(self, similarity_scores: Dict[int, float], 
                                 confidence_score: float) -> Dict[int, float]:
        """
        Beregner routing-vægte baseret på lighed og tillid
        
        Args:
            similarity_scores: Ekspertlighedsscores
            confidence_score: Assistant-tillidsscore
            
        Returns:
            routing_weights: mapping af expert_id: routing_weight
        """
        routing_weights = {}
        
        if not similarity_scores:
            return routing_weights
        
        # 1. Anvend tillidsvægt
        for expert_id, similarity in similarity_scores.items():
            weighted_score = similarity * confidence_score
            routing_weights[expert_id] = weighted_score
        
        # 2. Softmax-normalisering
        if routing_weights:
            scores = list(routing_weights.values())
            softmax_scores = F.softmax(torch.tensor(scores), dim=0)
            
            for i, expert_id in enumerate(routing_weights.keys()):
                routing_weights[expert_id] = softmax_scores[i].item()
        
        logger.debug(f"Beregnet routing-vægte for {len(routing_weights)} Eksperter.")
        return routing_weights
    
    def _weighted_combine_embeddings(self, embedded_outputs: List[EmbeddedAssistantOutput]) -> torch.Tensor:
        """
        Kombinerer flere Assistant-indlejringer med et vægtet gennemsnit
        
        Args:
            embedded_outputs: Indlejringsbehandlede outputs
            
        Returns:
            combined_embedding: Kombineret indlejringsvektor [hidden_dim]
        """
        if not embedded_outputs:
            return torch.zeros(self.hidden_dim)
        
        logger.debug(f"Kombinerer {len(embedded_outputs)} Assistant-indlejringer...")
        weights = [out.assistant_output.confidence_score for out in embedded_outputs]
        embeddings = [out.embedding_vector for out in embedded_outputs]
        
        # Normaliser vægte
        weights_tensor = torch.tensor(weights)
        weights_tensor = F.softmax(weights_tensor, dim=0)
        
        # Beregn vægtet gennemsnit
        combined_embedding = torch.zeros(self.hidden_dim)
        for i, embedding in enumerate(embeddings):
            combined_embedding += weights_tensor[i] * embedding
        
        # Normaliser
        combined_embedding = F.normalize(combined_embedding, p=2, dim=0)
        
        logger.debug(f"Kombineret indlejring oprettet med norm: {torch.norm(combined_embedding).item():.4f}")
        return combined_embedding
    
    def _calculate_expert_routing_weights(self, combined_embedding: torch.Tensor, 
                                        max_experts: MaxExpertsList) -> Dict[int, float]:
        """
        Beregner Ekspert-routing-vægte baseret på den kombinerede indlejring
        
        Args:
            combined_embedding: Kombineret indlejringsvektor
            max_experts: Liste over maksimalt aktiverede Eksperter
            
        Returns:
            expert_weights: mapping af expert_id: weight
        """
        expert_weights = {}
        logger.debug(f"Beregner endelige routing-vægte for {len(max_experts.experts)} aktive Eksperter.")
        
        for expert_output in max_experts.experts:
            expert_id = expert_output.expert_id
            expert_embedding = self._get_expert_embedding(expert_id)
            similarity = F.cosine_similarity(
                combined_embedding.unsqueeze(0),
                expert_embedding.unsqueeze(0)
            ).item()
            
            # Kombiner med oprindelig activation_score
            combined_score = (similarity + expert_output.activation_score) / 2.0
            
            # Anvend fusion_degree
            final_weight = combined_score * expert_output.fusion_degree
            expert_weights[expert_id] = final_weight
        
        # Softmax-normalisering
        if expert_weights:
            scores = list(expert_weights.values())
            softmax_scores = F.softmax(torch.tensor(scores), dim=0)
            for i, expert_id in enumerate(expert_weights.keys()):
                expert_weights[expert_id] = softmax_scores[i].item()
        
        return expert_weights
    
    def _create_routing_vector(self, combined_embedding: torch.Tensor, 
                             expert_weights: Dict[int, float]) -> torch.Tensor:
        """
        Opretter routing-vektor (MoE-systemets inputformat)
        
        Args:
            combined_embedding: Kombineret indlejringsvektor
            expert_weights: Ekspertvægte
            
        Returns:
            routing_vector: Routing-vektor i formen [1, 1, hidden_dim]
        """
        # Skaler indlejring baseret på Ekspertvægte
        if expert_weights:
            total_weight = sum(expert_weights.values())
            scaled_embedding = combined_embedding * total_weight
        else:
            scaled_embedding = combined_embedding
        
        # Konverter til MoE-inputformat
        routing_vector = scaled_embedding.unsqueeze(0).unsqueeze(0)
        return routing_vector
    
    def _get_expert_embedding(self, expert_id: int) -> torch.Tensor:
        """
        Henter indlejringsvektor for et givet Ekspert-ID
        
        Args:
            expert_id: Ekspert-ID
            
        Returns:
            expert_embedding: Ekspert-indlejringsvektor [hidden_dim]
        """
        if expert_id not in self.expert_embeddings:
            logger.warning(f"Ingen forudberegnet indlejring for Ekspert-ID {expert_id}. Returnerer nulvektor.")
            return torch.zeros(self.hidden_dim)
        
        return self.expert_embeddings[expert_id]
    
    def _save_routing_history(self, embedded_outputs: List[EmbeddedAssistantOutput], 
                            expert_weights: Dict[int, float], 
                            routing_vector: torch.Tensor):
        """
        Gemmer routinghistorik (til analyse og fejlfinding)
        
        Args:
            embedded_outputs: Indlejrings-outputs
            expert_weights: Ekspertvægte
            routing_vector: Routing-vektor
        """
        history_entry = {
            'timestamp': datetime.now(),
            'num_assistants': len(embedded_outputs),
            'expert_weights': expert_weights.copy(),
            'routing_norm': torch.norm(routing_vector).item(),
            'assistant_ids': [out.assistant_output.assistant_id for out in embedded_outputs]
        }
        self.routing_history.append(history_entry)
        
        # Begræns historikstørrelse (hukommelsesstyring)
        if len(self.routing_history) > 1000:
            self.routing_history = self.routing_history[-1000:]
    
    def get_routing_statistics(self) -> Dict:
        """
        Returnerer routingstatistik
        
        Returns:
            stats: Ordbog med routingstatistik
        """
        if not self.routing_history:
            return {}
        
        recent_history = self.routing_history[-100:]  # Seneste 100
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
        """Rydder cache og historik"""
        self.embedding_cache.clear()
        self.routing_history.clear()
        logger.info("EmbeddingProcessor-cache og -historik er blevet ryddet.")


# Eksempelfunktioner til brug
def create_test_assistant_outputs() -> List[AssistantOutput]:
    """Opretter test-Assistant-outputs"""
    return [
        AssistantOutput(
            assistant_id="assistant_1",
            comment_text="Dette spørgsmål er teknisk set meget interessant. Det kræver en dyb forståelse af implementeringsmetodikken.",
            confidence_score=0.8,
            related_experts=[1, 3, 5]
        ),
        AssistantOutput(
            assistant_id="assistant_2", 
            comment_text="Fra et kreativt perspektiv kræver dette problem en ny tilgang.",
            confidence_score=0.7,
            related_experts=[2, 4, 6]
        )
    ]


def create_test_max_experts() -> MaxExpertsList:
    """Opretter en test-MaxExpertsList"""
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


# Testkørselsfunktion
def test_embedding_processor():
    """Tester EmbeddingProcessor"""
    # Dummy-model og -tokenizer (erstattes med faktiske modeller ved rigtig brug)
    class DummyModel:
        def __init__(self):
            # Simuler lag med Eksperter til forudberegning
            self.layers = [type('DummyLayer', (), {
                'feed_forward': type('DummyFF', (), {
                    'experts': [torch.nn.Linear(10,10) for _ in range(8)]
                })
            })()]
        def embed_tokens(self, tokens):
            return torch.randn(tokens.shape[0], tokens.shape[1], 4096)
    
    class DummyTokenizer:
        def encode(self, text, add_bos=True, add_eos=True):
            return [1, 2, 3, 4, 5]  # Dummy-tokens
    
    # Initialiser processor
    processor = EmbeddingProcessor(
        model=DummyModel(),
        tokenizer=DummyTokenizer(),
        hidden_dim=4096
    )
    
    # Testdata
    assistant_outputs = create_test_assistant_outputs()
    max_experts = create_test_max_experts()
    
    # Kør behandling
    logger.info("=== EmbeddingProcessor Test ===")
    embedded_outputs = processor.vectorize_assistant_output(assistant_outputs)
    routing_vector = processor.route_to_experts(embedded_outputs, max_experts)
    
    # Udskriv resultater
    logger.info(f"Antal indlejrede outputs: {len(embedded_outputs)}")
    logger.info(f"Routing-vektorform: {routing_vector.shape}")
    logger.info(f"Routingstatistik: {processor.get_routing_statistics()}")
    
    processor.clear_cache()
    return processor, embedded_outputs, routing_vector


if __name__ == "__main__":
    test_embedding_processor()