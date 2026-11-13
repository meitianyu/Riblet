from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN, localcontext
from typing import Sequence

import numpy as np

from plaintext_numeric import S as SCALE
from experiment_contract import ALGORITHM_VERSION, DESIGN_BITS, DESIGN_INCLUSIVE, numeric_scope
from observed_trajectory import ERRORS


NUMERIC_FORMAT = "FM64/fxp18"
K_VALUES = tuple(range(9))


def common_feasible_ks(masks: Sequence[Sequence[bool]]) -> tuple[int, ...]:

    if len(masks) != 6:
        raise ValueError('per-K feasibility requires five folds and one full score')
    arrays = [np.asarray(mask) for mask in masks]
    if any(a.shape != (9,) or a.dtype != np.bool_ for a in arrays):
        raise ValueError('per-K feasible masks must be boolean vectors of length9')
    return tuple(int(k) for k in np.flatnonzero(np.logical_and.reduce(arrays)))


def _score_feasible_mask(score) -> np.ndarray:
    n = score.scoring_event_count
    return np.asarray([
        n > 0 and valid == n and passing == n
        for valid, passing in zip(score.per_K_valid_event_count,
                                  score.per_K_design_passing_event_count)
    ], dtype=np.bool_)


@dataclass(frozen=True)
class SupportGate:
    complete_distinct_requests: int = 919
    training_complement_distinct_requests: int = 100
    training_complement_events: int = 40

    def __post_init__(self) -> None:
        for name, value in (
            ("complete_distinct_requests", self.complete_distinct_requests),
            (
                "training_complement_distinct_requests",
                self.training_complement_distinct_requests,
            ),
            ("training_complement_events", self.training_complement_events),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class ArrivedEvents:
    request_ids: np.ndarray
    folds: np.ndarray
    effective_lengths: np.ndarray
    operands: np.ndarray
    event_ids: np.ndarray | None = None

    def subset(self, mask) -> "ArrivedEvents":
        mask = np.asarray(mask, dtype=np.bool_)
        if mask.ndim != 1 or len(mask) != len(self.operands):
            raise ValueError("event subset mask must align with operands")
        return ArrivedEvents(
            *(np.ascontiguousarray(value[mask]) for value in (
                self.request_ids,
                self.folds,
                self.effective_lengths,
                self.operands,
            )),
            event_ids=(None if self.event_ids is None else np.ascontiguousarray(self.event_ids[mask])),
        )


@dataclass(frozen=True)
class CandidateScore:
    held_out_fold: int | None
    scoring_event_count: int
    scoring_request_count: int
    raw_lower_encoded: int
    raw_upper_encoded: int
    L_encoded: int
    H_encoded: int
    a_encoded: int
    b_encoded: int
    minimum_K: int | None
    out_of_interval_event_count: int
    first_failure_code: int | None
    first_failure_kind: str | None
    per_K_valid_event_count: tuple[int, ...]
    per_K_p13_passing_event_count: tuple[int, ...]
    per_K_numeric_invalid_event_count: tuple[int, ...]
    maximum_decoded_residual_at_reported_K: float | None
    reported_residual_status: str
    per_K_design_passing_event_count: tuple[int, ...]
    design_bits: int = DESIGN_BITS
    design_inclusive: bool = DESIGN_INCLUSIVE
    first_failure_K: int | None = None
    execution_abi: str = 'G9_PLAINTEXT_STATEFUL_EXACT_RING_V1'


@dataclass(frozen=True)
class CandidateFit:
    candidate_id: str
    operator: str
    lineage_id: str
    model_or_task_id: str
    public_site_id: str
    source: str
    qualified: bool
    rejection_reason: str | None
    support_request_count: int
    support_event_count: int
    fold_scores: tuple[CandidateScore, ...]
    full_score: CandidateScore | None
    K_full: int | None
    K_min: int | None
    final_L_encoded: int | None
    final_H_encoded: int | None
    final_a_encoded: int | None
    final_b_encoded: int | None

    @property
    def feasible_Ks(self) -> tuple[int, ...]:
        if not self.qualified or self.full_score is None or len(self.fold_scores) != 5:
            return ()
        return common_feasible_ks([
            _score_feasible_mask(score) for score in (*self.fold_scores, self.full_score)
        ])

    def selection_record(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "numeric_seed_candidate_id": numeric_scope(self.candidate_id),
            "design_bits": DESIGN_BITS,
            "design_inclusive": DESIGN_INCLUSIVE,
            "operator": self.operator,
            "source": self.source,
            "qualified": self.qualified,
            "rejection_reason": self.rejection_reason,
            "K_min": self.K_min,
            "feasible_Ks": list(self.feasible_Ks),
            "L_encoded": self.final_L_encoded,
            "H_encoded": self.final_H_encoded,
            "a_encoded": self.final_a_encoded,
            "b_encoded": self.final_b_encoded,
            "support_request_count": self.support_request_count,
            "support_event_count": self.support_event_count,
        }

    def audit_record(self) -> dict[str, object]:
        def score_record(score: CandidateScore) -> dict[str, object]:
            record = dict(score.__dict__)
            for name in (
                "per_K_valid_event_count",
                "per_K_p13_passing_event_count",
                "per_K_design_passing_event_count",
                "per_K_numeric_invalid_event_count",
            ):
                record[name] = list(record[name])
            return record

        row = self.selection_record()
        row.update(
            {
                "algorithm_version": ALGORITHM_VERSION,
                "numeric_format": NUMERIC_FORMAT,
                "lineage_id": self.lineage_id,
                "model_or_task_id": self.model_or_task_id,
                "public_site_id": self.public_site_id,
                "K_full": self.K_full,
                "fold_K": [score.minimum_K for score in self.fold_scores],
                "fold_scores": [score_record(score) for score in self.fold_scores],
                "full_score": score_record(self.full_score) if self.full_score else None,
            }
        )
        return row


def _expanded_reciprocal_upper(raw_upper: int) -> int:

    return (17 * raw_upper + 9) // 10


def _reciprocal_coefficients(L_encoded: int, H_encoded: int) -> tuple[int, int]:
    q = (
        L_encoded * L_encoded
        + 6 * L_encoded * H_encoded
        + H_encoded * H_encoded
    )
    if q <= 0:
        raise ValueError("Reciprocal interval is invalid")
    a = -((8 * SCALE**3) // q)
    b = (8 * (L_encoded + H_encoded) * SCALE**2) // q
    return a, b


def _rsqrt_coefficients(L_encoded: int, H_encoded: int) -> tuple[int, int]:
    if not 0 < L_encoded <= H_encoded:
        raise ValueError("RSqrt interval must be positive")
    with localcontext() as context:
        context.prec = 96
        scale = Decimal(SCALE)
        lower = Decimal(L_encoded) / scale
        upper = Decimal(H_encoded) / scale
        u = lower.sqrt()
        v = upper.sqrt()
        s2 = u * u + u * v + v * v
        w = (s2 / Decimal(3)).sqrt()
        c = Decimal(2) / (
            u * v * (u + v) + (Decimal(2) / Decimal(3)) * s2 * w
        )
        a = int((-c * scale).to_integral_value(rounding=ROUND_DOWN))
        b = int((c * s2 * scale).to_integral_value(rounding=ROUND_DOWN))
    return a, b


def _parameters(
    operator: str, training_operands: np.ndarray
) -> tuple[int, int, int, int, int, int]:
    raw_lower = int(np.min(training_operands))
    raw_upper = int(np.max(training_operands))
    if operator == "reciprocal":
        lower, upper = raw_lower, _expanded_reciprocal_upper(raw_upper)
        a, b = _reciprocal_coefficients(lower, upper)
    elif operator == "rsqrt":
        lower, upper = raw_lower, raw_upper
        a, b = _rsqrt_coefficients(lower, upper)
    else:
        raise ValueError("operator must be reciprocal or rsqrt")
    return raw_lower, raw_upper, lower, upper, a, b


def _score_candidate(
    *,
    operator: str,
    training_operands: np.ndarray,
    scoring: ArrivedEvents,
    held_out_fold: int | None,
    scoring_backend=None,
    score_scope: str = '',
) -> CandidateScore:
    raw_lower, raw_upper, lower, upper, a, b = _parameters(
        operator, training_operands
    )
    if scoring_backend is None or scoring.event_ids is None:
        raise ValueError('explicit clear G9 backend and stable event IDs required')
    unique = scoring.operands
    event_multiplicity = np.ones(len(unique),dtype=np.int64)
    trajectory = scoring_backend.score(operator,unique,a_encoded=a,b_encoded=b,
        maximum_k=8,event_ids=scoring.event_ids,scope=f'{score_scope}:fold={held_out_fold}')
    minimum: int | None = None
    for K in K_VALUES:
        if bool(np.all(trajectory.valid[K] & trajectory.design_pass[K])):
            minimum = K
            break
    valid_event_counts = tuple(
        int(np.sum(event_multiplicity[trajectory.valid[K]], dtype=np.int64))
        for K in K_VALUES
    )
    p13_event_counts = tuple(
        int(np.sum(event_multiplicity[trajectory.p13_pass[K]], dtype=np.int64))
        for K in K_VALUES
    )
    design_event_counts = tuple(
        int(np.sum(event_multiplicity[trajectory.design_pass[K]], dtype=np.int64))
        for K in K_VALUES
    )
    numeric_invalid_counts = tuple(
        len(scoring.operands) - count for count in valid_event_counts
    )
    first_failure_code = None
    first_failure_kind = None
    first_failure_K = None
    if minimum is None:
        invalid = np.flatnonzero(~trajectory.valid[8])
        if len(invalid):
            first_failure_code = int(trajectory.error_code[8, int(invalid[0])])
            first_failure_kind = ERRORS.get(first_failure_code, "other_protocol_exception")
            first_failure_K = int(trajectory.first_failure_K[int(invalid[0])])
        else:
            first_failure_code = 0
            first_failure_kind = "finite_design_precision_threshold"
    reported_K = minimum if minimum is not None else 8
    reported_y = trajectory.y_encoded[reported_K].astype(np.float64)
    decoded_operand = unique.astype(np.float64) / SCALE
    decoded_y = reported_y / SCALE
    if np.any(~trajectory.valid[reported_K]):
        maximum_residual = None
        residual_status = "positive_infinity"
    elif operator == "reciprocal":
        maximum_residual = float(np.max(np.abs(1.0 - decoded_operand * decoded_y)))
        residual_status = "finite"
    else:
        maximum_residual = float(
            np.max(np.abs(1.0 - np.sqrt(decoded_operand) * decoded_y))
        )
        residual_status = "finite"
    return CandidateScore(
        held_out_fold=held_out_fold,
        scoring_event_count=len(scoring.operands),
        scoring_request_count=len(set(scoring.request_ids.tolist())),
        raw_lower_encoded=raw_lower,
        raw_upper_encoded=raw_upper,
        L_encoded=lower,
        H_encoded=upper,
        a_encoded=a,
        b_encoded=b,
        minimum_K=minimum,
        out_of_interval_event_count=int(
            np.count_nonzero(
                (scoring.operands < lower) | (scoring.operands > upper)
            )
        ),
        first_failure_code=first_failure_code,
        first_failure_kind=first_failure_kind,
        per_K_valid_event_count=valid_event_counts,
        per_K_p13_passing_event_count=p13_event_counts,
        per_K_design_passing_event_count=design_event_counts,
        per_K_numeric_invalid_event_count=numeric_invalid_counts,
        maximum_decoded_residual_at_reported_K=maximum_residual,
        reported_residual_status=residual_status,
        first_failure_K=first_failure_K,
        execution_abi=scoring_backend.abi_id,
    )


def _rejected_candidate(
    *,
    candidate_id: str,
    operator: str,
    lineage_id: str,
    model_or_task_id: str,
    public_site_id: str,
    source: str,
    support_requests: int,
    support_events: int,
    reason: str,
    fold_scores: Sequence[CandidateScore] = (),
    full_score: CandidateScore | None = None,
) -> CandidateFit:
    return CandidateFit(
        candidate_id,
        operator,
        lineage_id,
        model_or_task_id,
        public_site_id,
        source,
        False,
        reason,
        support_requests,
        support_events,
        tuple(fold_scores),
        full_score,
        full_score.minimum_K if full_score else None,
        None,
        full_score.L_encoded if full_score else None,
        full_score.H_encoded if full_score else None,
        full_score.a_encoded if full_score else None,
        full_score.b_encoded if full_score else None,
    )


def fit_candidate(
    events: ArrivedEvents,
    *,
    operator: str,
    lineage_id: str,
    model_or_task_id: str,
    public_site_id: str,
    support_gate: SupportGate | None = None,
    scoring_backend=None,
    fit_purpose: str = 'formal',
) -> CandidateFit:


    if operator not in {"reciprocal", "rsqrt"}:
        raise ValueError("operator must be reciprocal or rsqrt")
    if scoring_backend is None:
        raise ValueError('plaintext recovery requires an explicit scoring backend; no implicit nearest replay')
    if scoring_backend is not None:
        scoring_backend.require_purpose(fit_purpose)
    gate = support_gate or SupportGate()
    source = 'PUBLIC_PARENT'
    candidate_id = ':'.join((lineage_id,operator,model_or_task_id,public_site_id,source,'parent'))
    support_requests = len(set(events.request_ids.tolist()))
    support_events = len(events.operands)
    if np.any(events.operands <= 0):
        return _rejected_candidate(candidate_id=candidate_id,operator=operator,lineage_id=lineage_id,
            model_or_task_id=model_or_task_id,public_site_id=public_site_id,source=source,
            support_requests=support_requests,support_events=support_events,reason='nonpositive_observed_operand')
    if support_requests < gate.complete_distinct_requests:
        return _rejected_candidate(
            candidate_id=candidate_id,
            operator=operator,
            lineage_id=lineage_id,
            model_or_task_id=model_or_task_id,
            public_site_id=public_site_id,
            source=source,
            support_requests=support_requests,
            support_events=support_events,
            reason="insufficient_complete_request_support",
        )

    fold_scores: list[CandidateScore] = []
    for held_out in range(5):
        training_mask = events.folds != held_out
        heldout_mask = ~training_mask
        training_requests = len(set(events.request_ids[training_mask].tolist()))
        if not np.any(heldout_mask):
            return _rejected_candidate(
                candidate_id=candidate_id,
                operator=operator,
                lineage_id=lineage_id,
                model_or_task_id=model_or_task_id,
                public_site_id=public_site_id,
                source=source,
                support_requests=support_requests,
                support_events=support_events,
                reason=f"held_out_fold_{held_out}_empty",
                fold_scores=fold_scores,
            )
        if (
            int(np.count_nonzero(training_mask))
            < gate.training_complement_events
            or training_requests < gate.training_complement_distinct_requests
        ):
            return _rejected_candidate(
                candidate_id=candidate_id,
                operator=operator,
                lineage_id=lineage_id,
                model_or_task_id=model_or_task_id,
                public_site_id=public_site_id,
                source=source,
                support_requests=support_requests,
                support_events=support_events,
                reason=f"fold_{held_out}_training_support",
                fold_scores=fold_scores,
            )
        score = _score_candidate(
            operator=operator,
            training_operands=events.operands[training_mask],
            scoring=events.subset(heldout_mask),
            held_out_fold=held_out,
            scoring_backend=scoring_backend,
            score_scope=candidate_id,
        )
        fold_scores.append(score)

    full_score = _score_candidate(
        operator=operator,
        training_operands=events.operands,
        scoring=events,
        held_out_fold=None,
        scoring_backend=scoring_backend,
        score_scope=candidate_id,
    )
    failed_fold = next((s.held_out_fold for s in fold_scores if s.minimum_K is None), None)
    if failed_fold is not None or full_score.minimum_K is None:
        return _rejected_candidate(
            candidate_id=candidate_id,
            operator=operator,
            lineage_id=lineage_id,
            model_or_task_id=model_or_task_id,
            public_site_id=public_site_id,
            source=source,
            support_requests=support_requests,
            support_events=support_events,
            reason=(f"fold_{failed_fold}_design_precision_or_numeric_failure" if failed_fold is not None
                    else "full_development_design_precision_or_numeric_failure"),
            fold_scores=fold_scores,
            full_score=full_score,
        )
    feasible = common_feasible_ks([
        _score_feasible_mask(score) for score in (*fold_scores, full_score)
    ])
    if not feasible:
        return _rejected_candidate(
            candidate_id=candidate_id, operator=operator, lineage_id=lineage_id,
            model_or_task_id=model_or_task_id, public_site_id=public_site_id,
            source=source,support_requests=support_requests,support_events=support_events, reason='no_common_feasible_K',
            fold_scores=fold_scores, full_score=full_score,
        )
    K_min = min(feasible)
    return CandidateFit(
        candidate_id,
        operator,
        lineage_id,
        model_or_task_id,
        public_site_id,
        source,
        True,
        None,
        support_requests,
        support_events,
        tuple(fold_scores),
        full_score,
        full_score.minimum_K,
        K_min,
        full_score.L_encoded,
        full_score.H_encoded,
        full_score.a_encoded,
        full_score.b_encoded,
    )
