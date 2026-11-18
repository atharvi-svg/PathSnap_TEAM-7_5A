from __future__ import annotations

import numpy as np


def viterbi_decode(emission_log_probs: list[np.ndarray], transition_log_probs: list[np.ndarray]):
    """Decode variable-sized candidate layers; restart cleanly at impossible boundaries."""
    if not emission_log_probs:
        return [], {"sequence_log_score": 0.0, "restart_indices": []}
    segments: list[tuple[int, int, list[np.ndarray]]] = []
    start = 0
    delta = np.asarray(emission_log_probs[0], dtype=float)
    pointers: list[np.ndarray] = []
    restarts: list[int] = []
    for t in range(1, len(emission_log_probs)):
        trans = np.asarray(transition_log_probs[t - 1], dtype=float)
        joint = delta[:, None] + trans
        best_prev = np.argmax(joint, axis=0)
        next_delta = np.max(joint, axis=0) + np.asarray(emission_log_probs[t], dtype=float)
        if not np.isfinite(next_delta).any():
            segments.append((start, t, pointers))
            start = t
            delta = np.asarray(emission_log_probs[t], dtype=float)
            pointers = []
            restarts.append(t)
        else:
            pointers.append(best_prev.astype(np.int32))
            delta = next_delta
    segments.append((start, len(emission_log_probs), pointers))

    path = [0] * len(emission_log_probs)
    total_score = 0.0
    for seg_start, seg_end, seg_ptrs in segments:
        state = int(np.argmax(delta)) if seg_end == len(emission_log_probs) else None
        # Recompute this segment's terminal scores from its emission and pointers.
        scores = np.asarray(emission_log_probs[seg_start], dtype=float)
        for offset, t in enumerate(range(seg_start + 1, seg_end)):
            trans = np.asarray(transition_log_probs[t - 1], dtype=float)
            scores = np.max(scores[:, None] + trans, axis=0) + np.asarray(emission_log_probs[t], dtype=float)
        state = int(np.argmax(scores))
        total_score += float(scores[state])
        path[seg_end - 1] = state
        for t in range(seg_end - 1, seg_start, -1):
            ptr = seg_ptrs[t - seg_start - 1]
            path[t - 1] = int(ptr[path[t]])
    return path, {"sequence_log_score": total_score, "restart_indices": restarts}
