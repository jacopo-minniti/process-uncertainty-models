# rely/inference/sbs/strategies.py

import logging
import random
import math
from typing import List, Dict, Any, TYPE_CHECKING

from .clients import UncertaintyClient

if TYPE_CHECKING:
    from rely.inference.sbs.search import StepBeamSearch
    from rely.inference.sbs.utils import SBSNode

import logging
import random
import math
from typing import List, Dict, Any, TYPE_CHECKING

from .clients import UncertaintyClient

if TYPE_CHECKING:
    from rely.inference.sbs.search import StepBeamSearch
    from rely.inference.sbs.utils import SBSNode

logger = logging.getLogger(__name__)


def _distribute_samples_proportionally(scores: List[float], total_samples: int, num_beams: int) -> List[int]:
    """Helper to distribute samples proportionally to a list of scores."""
    total_score = sum(scores)
    if total_score == 0:
        # Fallback to even distribution
        base_samples = total_samples // num_beams
        remainder = total_samples % num_beams
        distribution = [base_samples + (1 if i < remainder else 0) for i in range(num_beams)]
        return distribution

    normalized_scores = [score / total_score for score in scores]
    sample_distribution = [int(norm_score * total_samples) for norm_score in normalized_scores]

    total_assigned = sum(sample_distribution)
    remainder = total_samples - total_assigned
    if remainder > 0:
        sorted_indices = sorted(range(num_beams), key=lambda k: scores[k], reverse=True)
        for i in range(remainder):
            sample_distribution[sorted_indices[i % num_beams]] += 1
    
    # Ensure no beam gets 0 samples if possible
    zero_indices = [i for i, s in enumerate(sample_distribution) if s == 0]
    if zero_indices:
        for i in zero_indices:
            # Find a beam with > 1 sample to donate
            donatable_beams = sorted([(s, j) for j, s in enumerate(sample_distribution) if s > 1], reverse=True)
            if donatable_beams:
                donor_index = donatable_beams[0][1]
                sample_distribution[donor_index] -= 1
                sample_distribution[i] += 1
            else:
                # Cannot re-allocate, break to avoid infinite loops
                break
    return sample_distribution

def _distribute_uniformly(total_samples_budget: int, num_active_beams: int) -> List[int]:
    """Helper to distribute samples uniformly across beams."""
    if num_active_beams == 0:
        return []
    
    base_samples = total_samples_budget // num_active_beams
    remainder = total_samples_budget % num_active_beams
    
    samples_per_beam = [base_samples] * num_active_beams
    if remainder > 0:
        indices_for_extra = random.sample(range(num_active_beams), remainder)
        for idx in indices_for_extra:
            samples_per_beam[idx] += 1
    return samples_per_beam

def _format_last_steps(beams: List['SBSNode'], max_len: int = 120) -> List[str]:
    """Return a short, single-line summary of each beam's latest step text."""
    steps: List[str] = []
    for beam in beams:
        snippet = beam.text.strip()
        if not snippet:
            snippet = "[root]"
        snippet = snippet.replace("\n", " ")
        if len(snippet) > max_len:
            snippet = snippet[: max_len - 3] + "..."
        steps.append(snippet)
    return steps

def _apply_softmax_and_distribute(
    sbs_instance: 'StepBeamSearch',
    scores: List[float],
    score_name: str = "scores"
) -> List[int]:
    """
    Applies softmax (with temperature) to scores and distributes samples.
    Shared logic for all score-based strategies.
    """
    if not scores:
        return []

    temp = sbs_instance.config.uncertainty_temperature
    
    # Deterministic / Greedy (Temp <= 0)
    if temp <= 0:
        max_score = max(scores)
        max_indices = [i for i, score in enumerate(scores) if score == max_score]
        distribution = [0] * len(scores)
        if max_indices:
            base_samples = sbs_instance.config.n_total_samples // len(max_indices)
            remainder = sbs_instance.config.n_total_samples % len(max_indices)
            for i in range(len(max_indices)):
                distribution[max_indices[i]] = base_samples + (1 if i < remainder else 0)
        
        if sbs_instance.config.verbose:
            logger.info(f"[Rank {sbs_instance.worker_rank}] {score_name}: {[f'{s:.3f}' for s in scores]}")
            logger.info(f"[Rank {sbs_instance.worker_rank}] Latest steps: {_format_last_steps(sbs_instance.active_beams)}")
            logger.info(f"[Rank {sbs_instance.worker_rank}] Sample distribution (deterministic): {distribution}")
        return distribution

    # Softmax Sampling
    max_score = max(scores)
    distribution_scores = [math.exp((score - max_score) / temp) for score in scores]

    sample_distribution = _distribute_samples_proportionally(
        distribution_scores,
        sbs_instance.config.n_total_samples,
        len(sbs_instance.active_beams)
    )
    
    if sbs_instance.config.verbose:
        logger.info(f"[Rank {sbs_instance.worker_rank}] {score_name}: {[f'{s:.3f}' for s in scores]}")
        logger.info(f"[Rank {sbs_instance.worker_rank}] Latest steps: {_format_last_steps(sbs_instance.active_beams)}")
        logger.info(f"[Rank {sbs_instance.worker_rank}] Sample distribution: {sample_distribution}")

    return sample_distribution


class SamplingStrategy:
    """Base class for sampling strategies."""
    def distribute_samples(self, sbs_instance: 'StepBeamSearch', question: str) -> List[int]:
        raise NotImplementedError

    def rank_candidates(self, sbs_instance: 'StepBeamSearch', candidates: List['SBSNode']) -> List['SBSNode']:
        """Rank candidate nodes for beam selection."""
        return sorted(candidates, key=lambda x: x.value, reverse=True)

    def update_candidate_uncertainty(self, candidate_node: 'SBSNode', generation_result: Dict[str, Any]):
        """Update the uncertainty of a newly generated candidate node."""
        pass # Default is no-op

    def requires_logprobs(self) -> bool:
        return False

class UniformStrategy(SamplingStrategy):
    """Distributes samples uniformly."""
    def distribute_samples(self, sbs_instance: 'StepBeamSearch', question: str) -> List[int]:
        num_active_beams = len(sbs_instance.active_beams)
        total_samples_budget = sbs_instance.config.n_total_samples
        
        distribution = _distribute_uniformly(total_samples_budget, num_active_beams)
        
        if sbs_instance.config.verbose:
            logger.info(f"[Rank {sbs_instance.worker_rank}] Sample distribution (uniform): {distribution}")
            
        return distribution

class PumBasedStrategy(SamplingStrategy):
    """Base class for strategies that use the PUM uncertainty model."""
    def __init__(self, uncertainty_client: UncertaintyClient):
        self.uncertainty_client = uncertainty_client

    def _get_uncertainties(self, sbs_instance: 'StepBeamSearch', prompts: List[str]) -> List[float]:
        if not prompts:
            return []
        return self.uncertainty_client.get_uncertainties(prompts)

    def update_beams_with_uncertainty(self, sbs_instance: 'StepBeamSearch', question: str) -> None:
        """Fetches and updates uncertainty scores for active beams if needed."""
        if not sbs_instance.active_beams:
            return

        if sbs_instance.should_evaluate_current_step():
            uncertainty_prompts = [
                sbs_instance.create_prompt(question, beam.full_text)
                for beam in sbs_instance.active_beams
            ]
            uncertainty_scores = self._get_uncertainties(sbs_instance, uncertainty_prompts)
            
            for i, beam in enumerate(sbs_instance.active_beams):
                if i < len(uncertainty_scores):
                    beam.uncertainty = uncertainty_scores[i]
            if sbs_instance.should_log_scored_steps():
                logger.info(
                    f"[Rank {sbs_instance.worker_rank}] Uncertainty scoring: "
                    f"depths={[beam.depth for beam in sbs_instance.active_beams]}, "
                    f"scores={[round(u,4) for u in uncertainty_scores[:10]]}"
                )

class PumStrategy(PumBasedStrategy):
    """Distributes samples based on PUM uncertainty."""
    def distribute_samples(self, sbs_instance: 'StepBeamSearch', question: str) -> List[int]:
        if not sbs_instance.active_beams:
            return []

        # 1. Update uncertainties
        self.update_beams_with_uncertainty(sbs_instance, question)
        
        # 2. Get scores
        uncertainty_scores = [beam.uncertainty for beam in sbs_instance.active_beams]

        # 3. Distribute
        return _apply_softmax_and_distribute(sbs_instance, uncertainty_scores, "Uncertainty scores")



class MeanVarStrategy(SamplingStrategy):
    """
    Strategy that combines:
    1. A 'Mean' PRM score (from the primary value model)
    2. A 'Variance' or Normalized PRM score (from the secondary model, reusing uncertainty infrastructure)
    
    Score = sigmoid(logit(mu_t) + beta * z_t)
    where:
      mu_t: running average of step scores (p_t) from the primary model
      z_t: score from secondary model
      beta: weight
    """
    def __init__(self, second_prm_client: UncertaintyClient, weight: float = 0.5):
        self.second_client = second_prm_client
        self.weight = weight

    def _logit(self, p: float) -> float:
        """Safe logit function: log(p / (1 - p))."""
        p = max(1e-6, min(1 - 1e-6, p))
        return math.log(p / (1 - p))

    def _sigmoid(self, x: float) -> float:
        """Sigmoid function: 1 / (1 + exp(-x))."""
        return 1 / (1 + math.exp(-x))

    def _get_second_scores(self, sbs_instance: 'StepBeamSearch', prompts: List[str]) -> List[float]:
        if not prompts:
            return []
        return self.second_client.get_uncertainties(prompts) # Reuse 'get_uncertainties' method name on client

    def update_beams_with_second_score(self, sbs_instance: 'StepBeamSearch', question: str) -> None:
        """Fetches and updates secondary scores for active beams."""
        if not sbs_instance.active_beams:
            return

        if sbs_instance.should_evaluate_current_step():
            second_prompts = [
                sbs_instance.create_prompt(question, beam.full_text)
                for beam in sbs_instance.active_beams
            ]
            second_scores = self._get_second_scores(sbs_instance, second_prompts)
            
            for i, beam in enumerate(sbs_instance.active_beams):
                if i < len(second_scores):
                    beam.uncertainty = second_scores[i]
            
            if sbs_instance.should_log_scored_steps():
                 logger.info(
                    f"[Rank {sbs_instance.worker_rank}] Secondary PRM scoring: "
                    f"depths={[beam.depth for beam in sbs_instance.active_beams]}, "
                    f"scores={[round(s, 4) for s in second_scores[:10]]}"
                )

    def distribute_samples(self, sbs_instance: 'StepBeamSearch', question: str) -> List[int]:
        # User requested: "Ranking only in this strategy. the distribution remains uniform"
        num_active_beams = len(sbs_instance.active_beams)
        total_samples_budget = sbs_instance.config.n_total_samples
        
        distribution = _distribute_uniformly(total_samples_budget, num_active_beams)
        
        if sbs_instance.config.verbose:
            logger.info(f"[Rank {sbs_instance.worker_rank}] Sample distribution (uniform/mean_var): {distribution}")
            
        return distribution

    def rank_candidates(self, sbs_instance: 'StepBeamSearch', candidates: List['SBSNode']) -> List['SBSNode']:
        """
        Rank candidates based on the combined score.
        s_t = sigmoid(logit(mu_t) + beta * z_t)
        """
        # 1. Fetch z_t (secondary scores) for candidates
        prompts = [
            sbs_instance.create_prompt(sbs_instance.question, c.full_text)
            for c in candidates
        ]
        
        second_scores = self._get_second_scores(sbs_instance, prompts)
        
        # Assign secondary scores
        for c, s in zip(candidates, second_scores):
            c.uncertainty = s
            
        # Log if verbose
        if sbs_instance.should_log_scored_steps():
             logger.info(
                f"[Rank {sbs_instance.worker_rank}] MeanVar Ranking: "
                f"Sample z_t (Var): {[round(c.uncertainty, 3) for c in candidates[:5]]}..."
            )

        # 2. Compute s_t and update node value
        for node in candidates:
            # Retrieve parent stats (running average state)
            parent = node.parent
            if parent is None:
                p_sum = 0.0
                p_count = 0
            else:
                p_sum = getattr(parent, '_mean_sum', 0.0)
                p_count = getattr(parent, '_mean_count', 0)
            
            # Incorporate current node's p_t (primary model score)
            # prm_score is set by StepBeamSearch if evaluated, else None
            current_prm = node.prm_score
            
            if current_prm is not None:
                current_sum = p_sum + current_prm
                current_count = p_count + 1
            else:
                current_sum = p_sum
                current_count = p_count
            
            # Save state on this node for its children
            setattr(node, '_mean_sum', current_sum)
            setattr(node, '_mean_count', current_count)
            
            # Compute mu_t (running average)
            if current_count > 0:
                mu_t = current_sum / current_count
            else:
                mu_t = 0.0 # Should not happen if prm_scores are present
                
            # Compute s_t formula
            # s_t = sigmoid(logit(mu_t) + beta * z_t)
            z_t = node.uncertainty
            beta = self.weight
            
            logit_mu = self._logit(mu_t)
            combined_logit = logit_mu + beta * z_t
            s_t = self._sigmoid(combined_logit)
            
            # Set the ranking value
            node.value = s_t
            
            # Also helpful to store mu_t directly if needed, but not required
            # node.mean_score = mu_t 

        # Return sorted by the new s_t
        return sorted(candidates, key=lambda x: x.value, reverse=True)

