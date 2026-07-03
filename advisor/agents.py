"""Agent-oriented wrappers around the ADvisor pipeline stages.

The original research pipeline is implemented in ``advisor.utils``.
These light wrappers make each stage easier to import, test, and replace
without changing the CLI behavior.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import utils


@dataclass
class CrossBrandAgent:
    """Finds similar brands and builds cross-brand reasoning."""

    all_metadata: Optional[pd.DataFrame] = None
    all_embeddings: Optional[Dict[str, np.ndarray]] = None
    brand_embeddings: Optional[Dict[str, np.ndarray]] = None

    def find_samples(
        self,
        brand_id: str,
        metric_modes: List[str],
    ) -> Tuple[Optional[str], List[Dict[str, Any]]]:
        if self.all_metadata is None:
            return None, []
        return utils.find_most_similar_brand(
            brand_id,
            self.all_metadata,
            metric_modes,
            all_embeddings=self.all_embeddings,
            brand_embeddings=self.brand_embeddings,
        )

    def build_reasoning(self, brand_id: str, metric_modes: List[str]) -> str:
        similar_brand_id, samples = self.find_samples(brand_id, metric_modes)
        if not similar_brand_id or not samples:
            return ""
        return utils.build_cross_brand_reasoning(similar_brand_id, samples, metric_modes)


@dataclass
class FeatureSelectionAgent:
    """Selects brand-specific LLM scoring features."""

    num_features: int = 4

    def select(
        self,
        brand_id: str,
        metric_modes: List[str],
        fewshot_examples: List[Dict[str, Any]],
        brand_description: Optional[str] = None,
        cross_brand_reasoning: Optional[str] = None,
        brand_performance_samples: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, str]:
        return utils.select_features_for_brand(
            brand_id,
            num_features=self.num_features,
            brand_description=brand_description,
            cross_brand_reasoning=cross_brand_reasoning,
            brand_performance_samples=brand_performance_samples,
            metric_modes=metric_modes,
            fewshot_examples=fewshot_examples,
        )


@dataclass
class ScoringAgent:
    """Adds LLM-derived feature scores to ad rows."""

    use_critique: bool = True

    def augment(
        self,
        brand_id: str,
        df: pd.DataFrame,
        fewshot_examples: List[Dict[str, Any]],
        metric_modes: List[str],
        selected_features: Dict[str, str],
        cross_brand_reasoning: Optional[str] = None,
    ) -> pd.DataFrame:
        return utils.augment_with_llm_features_multimodel(
            brand_id,
            df,
            fewshot_examples=fewshot_examples,
            metric_modes=metric_modes,
            selected_features=selected_features,
            cross_brand_reasoning=cross_brand_reasoning,
            use_critique=self.use_critique,
        )


@dataclass
class RankingAgent:
    """Trains or invokes the configured ranker for scored ads."""

    model_type: str = "lgbm"

    def run_brand(
        self,
        brand_id: str,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        metric_modes: List[str],
        all_embeddings: Optional[Dict[str, np.ndarray]] = None,
        all_metadata: Optional[pd.DataFrame] = None,
        brand_embeddings: Optional[Dict[str, np.ndarray]] = None,
        use_cross_brand_for_features: bool = False,
        use_critique: bool = True,
    ):
        return utils.run_brand_pipeline(
            brand_id,
            train_df,
            test_df,
            metric_modes,
            all_embeddings=all_embeddings,
            all_metadata=all_metadata,
            brand_embeddings=brand_embeddings,
            use_cross_brand_for_features=use_cross_brand_for_features,
            use_critique=use_critique,
            model_type=self.model_type,
        )
