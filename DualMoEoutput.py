"""
Dual Output MoE System for Mistral-based Architecture
Dobbeltpas MoE-system - genererer max_experts i første pas, rerouter baseret på Assistant-feedback i andet pas

Et system, der udvider den eksisterende Mistral MoE-struktur til at generere to output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import json
import numpy as np
import logging

# Opsætning af grundlæggende logningskonfiguration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


@dataclass
class ExpertOutput:
    """Dataklasse til lagring af ekspertoutputoplysninger"""
    expert_id: int
    output_tensor: torch.Tensor # Dette felt bruges muligvis ikke længere, men bevares for kompatibilitet
    weight: float
    activation_score: float # Routing-score fra gaten
    fusion_degree: float = 1.0

@dataclass
class MaxExpertsList:
    """Klasse til styring af max_experts-listen"""
    experts: List[ExpertOutput]
    max_count: int = 8
    threshold: float = 0.1

    def add_expert(self, expert_output: ExpertOutput):
        """Tilføj ekspert (baseret på aktiveringsscore)"""
        if expert_output.activation_score >= self.threshold:
            self.experts.append(expert_output)
            self.experts.sort(key=lambda x: x.activation_score, reverse=True)
            if len(self.experts) > self.max_count:
                self.experts = self.experts[:self.max_count]

class DualOutputMoE(nn.Module):
    """
    Et dobbelt output-system, der styrer en eksisterende MoE-model.
    Bruger PyTorch Hooks til at fange routing-oplysninger.
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.num_experts = self.model.args.moe.num_experts
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialiserer DualOutputMoE med {self.num_experts} eksperter.")

        # Variabel til lagring af routing-oplysninger fanget via Hook
        self.captured_routing_logits = []

        # Registrer Hook på gaten i MoE-laget
        self._register_hooks()

        # Intern tilstand
        self.max_experts_cache = None

    def _register_hooks(self):
        """Registrerer en Forward Hook på alle MoE Gates i modellen."""
        self.logger.info("Registrerer forward hooks på MoE-gates...")
        
        def hook_fn(module, input, output):
            # output er routing logits-tensoren.
            self.logger.debug(f"Hook fanget routing logits med form: {output.shape}")
            self.captured_routing_logits.append(output)

        hook_count = 0
        for layer in self.model.layers:
            # Få adgang til feed_forward.gate-strukturen i Mixtral-modellen
            if hasattr(layer, 'feed_forward') and hasattr(layer.feed_forward, 'gate'):
                layer.feed_forward.gate.register_forward_hook(hook_fn)
                hook_count += 1
        self.logger.info(f"Registrerede {hook_count} hooks på MoE-gates.")

    def _clear_captured_data(self):
        """Rydder de fangede data til det næste Forward Pass."""
        self.logger.debug("Rydder fangede routing-data.")
        self.captured_routing_logits = []

    def forward_first_pass(self, tokens: torch.Tensor, seqlens: List[int]) -> Tuple[torch.Tensor, MaxExpertsList]:
        """
        Første pas: normalt MoE-output + generering af max_experts
        
        Args:
            tokens: Input token-ID'er [num_tokens]
            seqlens: Længde af hver sekvens [batch_size]

        Returns:
            first_output: Modellens endelige output (indlejring)
            max_experts: MaxExpertsList-objekt
        """
        self.logger.info("Starter første fremadrettede pas.")
        self._clear_captured_data()
        
        # 1. Kør modellens formelle forward pass
        # mistral-inferences Transformer.forward tager token-ID'er og seqlens.
        self.logger.info("Udfører primær modelfremadrettet pas...")
        first_output = self.model.forward(tokens, seqlens)
        self.logger.info(f"Første pas output genereret med form: {first_output.shape}")
        
        # 2. Analyser routing-oplysninger fanget af Hook
        if not self.captured_routing_logits:
            self.logger.error("Kunne ikke fange MoE-routing-oplysninger. Kontroller modelstrukturen.")
            raise RuntimeError("Kunne ikke fange MoE-routing-oplysninger. Kontroller modelstrukturen.")
            
        self.logger.info(f"Analyse af routing-logits fra {len(self.captured_routing_logits)} lag.")
        # Gennemsnit af routing-scores for alle MoE-lag for at beregne vigtigheden af hver ekspert
        # captured_routing_logits: List[Tensor] (hvert element er [num_tokens, num_experts] form)
        avg_routing_probs = torch.zeros(self.num_experts)
        
        for logits in self.captured_routing_logits:
            probs = F.softmax(logits, dim=-1) # [num_tokens, num_experts]
            # Gennemsnitlig aktiveringssandsynlighed for hver ekspert
            avg_routing_probs += probs.mean(dim=0).cpu()
            
        avg_routing_probs /= len(self.captured_routing_logits)
        self.logger.info(f"Gennemsnitlige ekspertaktiveringssandsynligheder: {avg_routing_probs.numpy().round(4)}")
        
        # 3. Opret max_experts-listen
        max_experts = MaxExpertsList(experts=[], max_count=self.model.args.moe.num_experts)
        all_experts = []
        for expert_id in range(self.num_experts):
            activation_score = avg_routing_probs[expert_id].item()
            # weight og output_tensor har ikke længere direkte betydning i dette design,
            # men activation_score bruges som weight for kompatibilitet med undersystemer.
            expert_out = ExpertOutput(
                expert_id=expert_id,
                output_tensor=None, # bruges ikke længere
                weight=activation_score,
                activation_score=activation_score
            )
            all_experts.append(expert_out)
            
        # Sorter efter aktiveringsscore for at tilføje til max_experts
        all_experts.sort(key=lambda x: x.activation_score, reverse=True)
        for expert in all_experts:
            max_experts.add_expert(expert) # Tilføjes i henhold til threshold-betingelsen i add_expert
        
        self.logger.info(f"Oprettet max_experts-liste med {len(max_experts.experts)} aktive eksperter.")
        # 4. Gem i cache og returner
        self.max_experts_cache = max_experts
        return first_output, max_experts
    
    def forward_second_pass(self, assistant_embedded_vector: torch.Tensor) -> torch.Tensor:
        """
        Andet pas: Kører MoE-lagene direkte med Assistant's indlejringsvektor for at generere et nyt output.
        
        Args:
            assistant_embedded_vector: [batch_size, seq_len, hidden_dim]

        Returns:
            second_output: Det endelige outputindlejring fra det andet MoE-pas
        """
        self.logger.info("Starter andet fremadrettede pas.")
        if self.max_experts_cache is None:
            self.logger.error("Første pas skal køres før andet pas.")
            raise ValueError("Første pas skal køres før andet pas.")

        # --- Modificeret kernelogik ---
        # 1. Sæt inputindlejringen som h (hidden_state).
        h = assistant_embedded_vector
        self.logger.info(f"Input til andet pas modtaget med form: {h.shape}")
        
        # 2. Opret en opmærksomhedsmaske.
        #    Opret en maske (Causal Mask), der ser hele inputsekvensen.
        mask = torch.full((h.shape[1], h.shape[1]), float("-inf"), device=h.device)
        mask = torch.triu(mask, diagonal=1)
        
        # 3. Passer sekventielt gennem alle transformerlag i modellen.
        #    Under denne proces fungerer MoE Gate inde i modellen igen i henhold til det nye input (h).
        self.logger.info(f"Behandler input gennem {len(self.model.layers)} transformerlag...")
        for i, layer in enumerate(self.model.layers):
            self.logger.debug(f"Indtaster lag {i+1}/{len(self.model.layers)}")
            h = layer(h, mask)
        
        # 4. Anvend endelig Layer Normalization for at fuldføre det andet resultat.
        self.logger.info("Anvender endelig normalisering.")
        second_output = self.model.norm(h)
        self.logger.info(f"Andet pas output genereret med form: {second_output.shape}")
        # --- Modificering slut ---
        
        return second_output
    
    def _broadcast_expert_output(self, expert_output: torch.Tensor, 
                               target_shape: torch.Size, weight: float) -> torch.Tensor:
        """
        Broadcast ekspertoutput til målform
        
        Args:
            expert_output: Ekspertens outputtensor
            target_shape: Målform [total_tokens, hidden_dim]
            weight: Vægt
        
        Returns:
            Broadcastet outputtensor
        """
        total_tokens, hidden_dim = target_shape
        
        if expert_output.shape[0] == total_tokens:
            # Allerede i den korrekte form
            return expert_output * weight
        else:
            # Gennemsnitlig pooling og derefter broadcast
            avg_output = expert_output.mean(dim=0, keepdim=True)  # [1, hidden_dim]
            return avg_output.expand(total_tokens, hidden_dim) * weight
    
    def update_fusion_degree(self, expert_id: int, new_degree: float):
        """
        Opdater fusion_degree for en specifik ekspert
        
        Args:
            expert_id: Ekspert-ID
            new_degree: Ny fusionsgrad (0.0 ~ 1.0)
        """
        self.logger.info(f"Opdaterer fusionsgrad for ekspert {expert_id} til {new_degree:.4f}")
        self.fusion_degrees[expert_id] = max(0.0, min(1.0, new_degree))
        
        # Opdater også cachelagrede max_experts
        if self.max_experts_cache:
            for expert in self.max_experts_cache.experts:
                if expert.expert_id == expert_id:
                    expert.fusion_degree = new_degree
                    self.logger.debug(f"Opdateret fusionsgrad i cache for ekspert {expert_id}.")
                    break
    
    def get_expert_metrics(self) -> Dict:
        """
        Returner ekspertudnyttelses- og præstationsmetrikker
        
        Returns:
            Metrikordbog
        """
        self.logger.info("Henter ekspertmetrikker.")
        return {
            'first_pass_calls': self.metrics['first_pass_calls'],
            'second_pass_calls': self.metrics['second_pass_calls'],
            'expert_utilization': self.metrics['expert_utilization'].tolist(),
            'routing_entropy_avg': np.mean(self.metrics['routing_entropy']) if self.metrics['routing_entropy'] else 0.0,
            'fusion_degrees': dict(self.fusion_degrees),
            'active_experts': len([d for d in self.fusion_degrees.values() if d > 0.1])
        }
    
    def reset_metrics(self):
        """Nulstil metrikker"""
        self.logger.info("Nulstiller metrikker.")
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
        Integreret forward-funktion (opretholder eksisterende MoE-kompatibilitet)
        
        Args:
            input_tensor: Inputtensor
            assistant_vector: Assistant-vektor (valgfri)
            return_max_experts: Om max_experts skal returneres
        
        Returns:
            Outputtensor (eller tuple)
        """
        self.logger.info("Kalder den integrerede forward-funktion.")
        if assistant_vector is None:
            # Kør kun første pas
            self.logger.info("Ingen assistentvektor angivet. Kører kun første pas.")
            first_output, max_experts = self.forward_first_pass(input_tensor)
            if return_max_experts:
                return first_output, max_experts
            return first_output
        else:
            # Kør andet pas
            self.logger.info("Assistentvektor angivet. Kører andet pas.")
            if self.max_experts_cache is None:
                raise ValueError("Første pas skal køres før andet pas.")
            return self.forward_second_pass(assistant_vector, self.max_experts_cache)
    
    def save_state(self, filepath: str):
        """Gem tilstand"""
        self.logger.info(f"Gemmer tilstand til {filepath}")
        state = {
            'fusion_degrees': self.fusion_degrees,
            'metrics': self.get_expert_metrics(),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        self.logger.info("Tilstand gemt succesfuldt.")
    
    def load_state(self, filepath: str):
        """Indlæs tilstand"""
        self.logger.info(f"Indlæser tilstand fra {filepath}")
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.fusion_degrees = state.get('fusion_degrees', {})
        # Konverter strengnøgler til heltal
        self.fusion_degrees = {int(k): v for k, v in self.fusion_degrees.items()}
        self.logger.info("Tilstand indlæst succesfuldt.")


# Eksempel på brug og testkode
def create_test_dual_moe():
    """Opret en test-DualOutputMoE"""
    
    # Opret dummy-ekspert
    class DummyExpert(nn.Module):
        def __init__(self, hidden_dim):
            super().__init__()
            self.linear = nn.Linear(hidden_dim, hidden_dim)
            
        def forward(self, x):
            return self.linear(x)
    
    # Opret dummy-gate
    class DummyGate(nn.Module):
        def __init__(self, hidden_dim, num_experts):
            super().__init__()
            self.linear = nn.Linear(hidden_dim, num_experts)
            
        def forward(self, x):
            return self.linear(x)
    
    # Indstil parametre
    hidden_dim = 512
    num_experts = 8
    
    # Opret komponenter
    experts = [DummyExpert(hidden_dim) for _ in range(num_experts)]
    gate = DummyGate(hidden_dim, num_experts)
    
    # Opret DualOutputMoE
    args = type('Args', (), {'hidden_dim': hidden_dim})()
    dual_moe = DualOutputMoE(experts, gate, args, top_k=2)
    
    return dual_moe


if __name__ == "__main__":
    # Kør test
    logger = logging.getLogger()
    logger.info("Starter DualOutputMoE-test...")
    
    # 1. Opret system
    dual_moe = create_test_dual_moe()
    
    # 2. Testinput
    batch_size, seq_len, hidden_dim = 2, 10, 512
    input_tensor = torch.randn(batch_size, seq_len, hidden_dim)
    
    # 3. Kør første pas
    logger.info("Kører første pas...")
    first_output, max_experts = dual_moe.forward_first_pass(input_tensor)
    logger.info(f"Første outputform: {first_output.shape}")
    logger.info(f"Antal max_experts: {len(max_experts.experts)}")
    
    # 4. Opret dummy-assistentvektor
    assistant_vector = torch.randn(batch_size, seq_len, hidden_dim)
    
    # 5. Kør andet pas
    logger.info("Kører andet pas...")
    second_output = dual_moe.forward_second_pass(assistant_vector, max_experts)
    logger.info(f"Andet outputform: {second_output.shape}")
    
    # 6. Kontroller metrikker
    metrics = dual_moe.get_expert_metrics()
    logger.info(f"Metrikker: {metrics}")
    
    # 7. Test opdatering af fusionsgrad
    dual_moe.update_fusion_degree(0, 0.5)
    logger.info("Opdatering af fusionsgrad fuldført")
    
    logger.info("Test fuldført!")