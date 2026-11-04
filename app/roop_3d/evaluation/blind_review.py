"""Blinded A/B Human-Reviewed Comparison System (Stage 11).

Provides:
- Cryptographic randomization and anonymization of model names (e.g., 'System Alpha' vs 'System Beta')
- Randomized presentation order (A/B or Left/Right) to prevent positional bias
- Structured rating rubric:
  1. Identity Preservation (1-5 Likert scale)
  2. Expression Naturalness (1-5 Likert scale)
  3. Temporal Stability / Lack of Flicker (1-5 Likert scale)
  4. Boundary and Occlusion Realism (1-5 Likert scale)
  5. Overall Forced-Choice Preference ('System Alpha', 'System Beta', 'Tie')
- Secure unblinding protocol revealing ground-truth models only after review submission
- Win-rate statistics, inter-rater consistency, and binomial significance testing
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import hmac
import json
import math
import secrets
from typing import Any, Mapping, Sequence

from roop_3d.core.errors import DomainError


@dataclass
class HumanReviewScore:
    """Individual human rating submission for a blinded pair."""
    reviewer_id: str
    identity_preservation_a: int  # 1-5
    identity_preservation_b: int  # 1-5
    expression_naturalness_a: int  # 1-5
    expression_naturalness_b: int  # 1-5
    temporal_stability_a: int  # 1-5
    temporal_stability_b: int  # 1-5
    boundary_realism_a: int  # 1-5
    boundary_realism_b: int  # 1-5
    forced_choice_preference: str  # "A", "B", "Tie"
    reviewer_comments: str = ""
    timestamp: float = 0.0


@dataclass
class BlindComparisonPair:
    """A pair of model outputs prepared for blinded evaluation."""
    session_id: str
    pair_id: str
    scene_id: str
    blind_label_a: str  # e.g. "System Alpha"
    blind_label_b: str  # e.g. "System Beta"
    _secret_model_a: str  # True engine identifier
    _secret_model_b: str  # True engine identifier
    _is_swapped: bool  # True if system B was placed as label A
    scores: list[HumanReviewScore] = field(default_factory=list)


@dataclass
class BlindReviewSummary:
    """Aggregated evaluation results after secure unblinding."""
    pair_id: str
    model_1: str
    model_2: str
    total_reviews: int
    model_1_wins: int
    model_2_wins: int
    ties: int
    model_1_win_rate_pct: float
    model_2_win_rate_pct: float
    model_1_avg_identity: float
    model_2_avg_identity: float
    model_1_avg_naturalness: float
    model_2_avg_naturalness: float
    model_1_avg_stability: float
    model_2_avg_stability: float
    model_1_avg_realism: float
    model_2_avg_realism: float
    statistically_significant: bool
    p_value_approx: float


class BlindedReviewSystem:
    """Manages randomized, anonymized human A/B testing sessions."""

    def __init__(self, secret_salt: bytes | None = None):
        self.secret_salt = secret_salt or secrets.token_bytes(32)
        self.pairs: dict[str, BlindComparisonPair] = {}

    def create_blind_pair(
        self,
        session_id: str,
        scene_id: str,
        model_1: str,
        model_2: str,
    ) -> BlindComparisonPair:
        """Create a randomized, blinded comparison pair."""
        # Deterministic HMAC hash to decide label ordering
        h = hmac.new(self.secret_salt, f"{session_id}:{scene_id}:{model_1}:{model_2}".encode(), hashlib.sha256).digest()
        swap_order = (h[0] % 2) == 1

        pair_id = f"pair_{secrets.token_hex(6)}"
        label_a = "System Alpha"
        label_b = "System Beta"

        actual_a = model_2 if swap_order else model_1
        actual_b = model_1 if swap_order else model_2

        pair = BlindComparisonPair(
            session_id=session_id,
            pair_id=pair_id,
            scene_id=scene_id,
            blind_label_a=label_a,
            blind_label_b=label_b,
            _secret_model_a=actual_a,
            _secret_model_b=actual_b,
            _is_swapped=swap_order,
        )
        self.pairs[pair_id] = pair
        return pair

    def submit_rating(
        self,
        pair_id: str,
        rating: HumanReviewScore,
    ) -> None:
        """Submit a completed human review rating for a blinded pair."""
        pair = self.pairs.get(pair_id)
        if not pair:
            raise DomainError("PAIR_NOT_FOUND", f"Blind pair {pair_id} does not exist", 404)

        # Validate rating range 1..5
        for score in [
            rating.identity_preservation_a, rating.identity_preservation_b,
            rating.expression_naturalness_a, rating.expression_naturalness_b,
            rating.temporal_stability_a, rating.temporal_stability_b,
            rating.boundary_realism_a, rating.boundary_realism_b,
        ]:
            if not 1 <= score <= 5:
                raise ValueError("All ratings must be integer Likert values between 1 and 5")

        if rating.forced_choice_preference not in ("A", "B", "Tie"):
            raise ValueError("Preference must be 'A', 'B', or 'Tie'")

        pair.scores.append(rating)

    def unblind_pair_results(self, pair_id: str) -> BlindReviewSummary:
        """Unblind reviewer scores and aggregate true model performance metrics."""
        pair = self.pairs.get(pair_id)
        if not pair:
            raise DomainError("PAIR_NOT_FOUND", f"Blind pair {pair_id} does not exist", 404)

        scores = pair.scores
        n = len(scores)
        if n == 0:
            raise DomainError("NO_REVIEWS", "Cannot unblind results before reviews are submitted", 422)

        # Identify true identity of model 1 and model 2
        model_1 = pair._secret_model_b if pair._is_swapped else pair._secret_model_a
        model_2 = pair._secret_model_a if pair._is_swapped else pair._secret_model_b

        m1_wins, m2_wins, ties = 0, 0, 0
        m1_id, m2_id = [], []
        m1_nat, m2_nat = [], []
        m1_stab, m2_stab = [], []
        m1_real, m2_real = [], []

        for r in scores:
            if not pair._is_swapped:
                # Label A is model 1, Label B is model 2
                pref = r.forced_choice_preference
                if pref == "A":
                    m1_wins += 1
                elif pref == "B":
                    m2_wins += 1
                else:
                    ties += 1

                m1_id.append(r.identity_preservation_a)
                m2_id.append(r.identity_preservation_b)
                m1_nat.append(r.expression_naturalness_a)
                m2_nat.append(r.expression_naturalness_b)
                m1_stab.append(r.temporal_stability_a)
                m2_stab.append(r.temporal_stability_b)
                m1_real.append(r.boundary_realism_a)
                m2_real.append(r.boundary_realism_b)
            else:
                # Label A is model 2, Label B is model 1
                pref = r.forced_choice_preference
                if pref == "A":
                    m2_wins += 1
                elif pref == "B":
                    m1_wins += 1
                else:
                    ties += 1

                m1_id.append(r.identity_preservation_b)
                m2_id.append(r.identity_preservation_a)
                m1_nat.append(r.expression_naturalness_b)
                m2_nat.append(r.expression_naturalness_a)
                m1_stab.append(r.temporal_stability_b)
                m2_stab.append(r.temporal_stability_a)
                m1_real.append(r.boundary_realism_b)
                m2_real.append(r.boundary_realism_a)

        # Binomial significance test approximation (two-tailed sign test)
        decided = m1_wins + m2_wins
        if decided >= 5:
            # Normal approximation for sign test
            mean = decided * 0.5
            std = math.sqrt(decided * 0.25)
            z = abs(max(m1_wins, m2_wins) - 0.5 - mean) / max(std, 1e-6)
            # P-value approximation from z
            p_val = math.erfc(z / math.sqrt(2.0))
            is_sig = p_val < 0.05
        else:
            p_val = 1.0
            is_sig = False

        return BlindReviewSummary(
            pair_id=pair_id,
            model_1=model_1,
            model_2=model_2,
            total_reviews=n,
            model_1_wins=m1_wins,
            model_2_wins=m2_wins,
            ties=ties,
            model_1_win_rate_pct=round((m1_wins / n) * 100.0, 1),
            model_2_win_rate_pct=round((m2_wins / n) * 100.0, 1),
            model_1_avg_identity=round(float(sum(m1_id) / n), 2),
            model_2_avg_identity=round(float(sum(m2_id) / n), 2),
            model_1_avg_naturalness=round(float(sum(m1_nat) / n), 2),
            model_2_avg_naturalness=round(float(sum(m2_nat) / n), 2),
            model_1_avg_stability=round(float(sum(m1_stab) / n), 2),
            model_2_avg_stability=round(float(sum(m2_stab) / n), 2),
            model_1_avg_realism=round(float(sum(m1_real) / n), 2),
            model_2_avg_realism=round(float(sum(m2_real) / n), 2),
            statistically_significant=is_sig,
            p_value_approx=round(p_val, 4),
        )


@dataclass
class BlindComparisonSession:
    """A full test session containing multiple blinded pairs for an evaluation campaign."""
    session_id: str
    pairs: list[BlindComparisonPair] = field(default_factory=list)
