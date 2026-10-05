"""Causal temporal gates and run-start notifications; independent of labels/models."""
from __future__ import annotations
from dataclasses import dataclass
from collections.abc import Mapping
import numpy as np
import pandas as pd
from src.audit_event_evaluation_protocol import validate_stream
from src.data_quality import LONG_GAP_THRESHOLD_SEC

GATE_DEFINITIONS = {
    'G0': ('RAW', 1, 1), 'G1': ('CONSECUTIVE_2', 2, 2),
    'G2': ('CONSECUTIVE_3', 3, 3), 'G3': ('CONSECUTIVE_5', 5, 5),
    'G4': ('VOTE_2_OF_3', 3, 2), 'G5': ('VOTE_3_OF_5', 5, 3),
    'G6': ('VOTE_5_OF_10', 10, 5),
}
COMPLEXITY_ORDER = ('G0', 'G1', 'G4', 'G2', 'G5', 'G3', 'G6')
COOLDOWNS = (0, 60, 300)


@dataclass(frozen=True)
class AlertPolicy:
    gate_id: str
    cooldown_seconds: int

    def __post_init__(self):
        if self.gate_id not in GATE_DEFINITIONS or self.cooldown_seconds not in COOLDOWNS:
            raise ValueError('Policy outside frozen family.')

    @property
    def policy_id(self):
        return f'{self.gate_id}_C{self.cooldown_seconds}'

    @property
    def window_points(self):
        return GATE_DEFINITIONS[self.gate_id][1]

    @property
    def minimum_positive(self):
        return GATE_DEFINITIONS[self.gate_id][2]

    @property
    def complexity(self):
        return (COMPLEXITY_ORDER.index(self.gate_id), COOLDOWNS.index(self.cooldown_seconds))

    def to_dict(self):
        return {'policy_id': self.policy_id, 'gate_id': self.gate_id,
                'cooldown_seconds': self.cooldown_seconds,
                'window_points': self.window_points, 'minimum_positive': self.minimum_positive}

    @classmethod
    def from_dict(cls, record):
        if not isinstance(record, Mapping):
            raise ValueError('Expected concrete policy record.')
        result = cls(record['gate_id'], record['cooldown_seconds'])
        if record.get('policy_id') != result.policy_id:
            raise ValueError('Policy ID/parameters mismatch.')
        for key in ('window_points', 'minimum_positive'):
            if key in record and record[key] != getattr(result, key):
                raise ValueError('Gate definition changed.')
        return result


@dataclass(frozen=True)
class LockedPolicies:
    """Exactly one immutable, concrete policy per family; no candidate grid."""
    family_policies: tuple
    lock_sha256: str
    candidate_config_sha256: str

    def __post_init__(self):
        families = [family for family, policy in self.family_policies]
        if len(families) != 3 or set(families) != {'statistical_rule', 'isolation_forest', 'autoencoder'}:
            raise ValueError('Expected exactly one locked policy per family.')
        if not all(isinstance(policy, AlertPolicy) for family, policy in self.family_policies):
            raise TypeError('Lock cannot contain a candidate grid.')
        if not all(isinstance(digest, str) and len(digest) == 64 for digest in (self.lock_sha256, self.candidate_config_sha256)):
            raise ValueError('Missing lock/config hash.')

    def for_model(self, model):
        return dict(self.family_policies)[model]


def apply_policy(frame, policy):
    """Trailing full windows, then greedy cooldown on gate-run starts.

    Input must be ONE trajectory/sample. Unknown indices and >300 s intervals
    reset gate history and run state. Last emitted time remains across holes;
    only the outer caller's new trajectory invocation resets cooldown state.
    """
    if not isinstance(policy, AlertPolicy):
        raise TypeError('One concrete AlertPolicy required.')
    g = validate_stream(frame)
    raw = g.predicted_anomaly.to_numpy()
    if not np.isin(raw, [0, 1]).all():
        raise ValueError('Nonbinary/unknown raw prediction.')
    raw = raw.astype(np.int64)
    indices = g.source_point_index.to_numpy(np.int64)
    timestamps = g.timestamp.to_numpy(dtype='datetime64[ns]').astype(np.int64)
    links = np.r_[False, (np.diff(indices) == 1) & (np.diff(timestamps) > 0)
                  & (np.diff(timestamps) <= int(LONG_GAP_THRESHOLD_SEC * 1e9))]
    starts = np.flatnonzero(~links)
    ends = np.r_[starts[1:], len(g)]
    supported = np.zeros(len(g), dtype=bool)
    gate = np.zeros(len(g), dtype=bool)
    window, requirement = policy.window_points, policy.minimum_positive
    for start, end in zip(starts, ends):
        cumulative = np.r_[0, np.cumsum(raw[start:end])]
        count = end - start
        if count >= window:
            votes = cumulative[window:] - cumulative[:-window]
            supported[start + window - 1:end] = True
            gate[start + window - 1:end] = votes >= requirement
    candidates = gate & ~(np.r_[False, gate[:-1]] & links)
    emitted = np.zeros(len(g), dtype=bool)
    suppressed = np.zeros(len(g), dtype=bool)
    suppressing_index = np.full(len(g), np.nan)
    suppressing_time = np.full(len(g), np.datetime64('NaT'), dtype='datetime64[ns]')
    last_position = None
    for position in np.flatnonzero(candidates):
        if last_position is None or timestamps[position] - timestamps[last_position] >= policy.cooldown_seconds * 1_000_000_000:
            emitted[position] = True
            last_position = position
        else:
            suppressed[position] = True
            suppressing_index[position] = indices[last_position]
            suppressing_time[position] = g.timestamp.to_numpy(dtype='datetime64[ns]')[last_position]
    reasons = np.full(len(g), '', dtype=object)
    reasons[(raw == 1) & ~supported] = 'warm_up_after_boundary_or_quality_gap'
    reasons[(raw == 1) & supported & ~gate] = ('insufficient_consecutive_positives'
                                             if window == requirement else 'vote_requirement_failure')
    g['gate_window_supported'] = supported
    g['gate_positive'] = gate
    g['notification_candidate'] = candidates
    g['notification_emitted'] = emitted
    g['suppressed_by_cooldown'] = suppressed
    g['suppressing_notification_source_index'] = suppressing_index
    g['suppressing_notification_timestamp'] = suppressing_time
    g['gate_suppression_reason'] = reasons
    return g
