import os

import torch


def h2o_enabled():
    return os.environ.get("EFFICIENTNAV_USE_H2O", "1").lower() in {"1", "true", "yes", "on"}


def _env_int(name, default):
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        return max(0, int(raw_value))
    except ValueError:
        return default


def _env_float(name, default):
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError:
        return default


def h2o_config():
    budget = _env_int("EFFICIENTNAV_H2O_CACHE_BUDGET", 1024)
    recent_size = _env_int("EFFICIENTNAV_H2O_RECENT_SIZE", 256)
    heavy_size = _env_int("EFFICIENTNAV_H2O_HEAVY_SIZE", 256)
    protected_prefix = _env_int("EFFICIENTNAV_H2O_PROTECTED_PREFIX", 64)
    return budget, recent_size, heavy_size, protected_prefix


def h2o_protected_prefix_outside_budget():
    return os.environ.get("EFFICIENTNAV_H2O_PREFIX_OUTSIDE_BUDGET", "1").lower() in {"1", "true", "yes", "on"}


def h2o_use_attention_scores():
    return os.environ.get("EFFICIENTNAV_H2O_USE_ATTENTION_SCORES", "0").lower() in {"1", "true", "yes", "on"}


def h2o_score_config():
    return {
        "instruction_weight": _env_float("EFFICIENTNAV_H2O_INSTRUCTION_WEIGHT", 6.0),
        "decision_context_weight": _env_float("EFFICIENTNAV_H2O_DECISION_CONTEXT_WEIGHT", 8.0),
        "retrieved_memory_weight": _env_float("EFFICIENTNAV_H2O_RETRIEVED_MEMORY_WEIGHT", 5.0),
        "planner_context_weight": _env_float("EFFICIENTNAV_H2O_PLANNER_CONTEXT_WEIGHT", 4.0),
        "trajectory_weight": _env_float("EFFICIENTNAV_H2O_TRAJECTORY_WEIGHT", 3.5),
        "generation_suffix_weight": _env_float("EFFICIENTNAV_H2O_GENERATION_SUFFIX_WEIGHT", 4.0),
        "goal_weight": _env_float("EFFICIENTNAV_H2O_GOAL_WEIGHT", 4.0),
        "cluster_weight": _env_float("EFFICIENTNAV_H2O_CLUSTER_WEIGHT", 2.5),
        "frontier_weight": _env_float("EFFICIENTNAV_H2O_FRONTIER_WEIGHT", 2.5),
        "json_weight": _env_float("EFFICIENTNAV_H2O_JSON_WEIGHT", 1.5),
    }


H2O_SEGMENT_MARKERS = {
    "instruction": ("<|instruction|>", "<|/instruction|>"),
    "instruction_core": ("<|instruction_core|>", "<|/instruction_core|>"),
    "decision_context": ("<|decision_context|>", "<|/decision_context|>"),
    "planner_context": ("<|planner_context|>", "<|/planner_context|>"),
    "retrieved_memory": ("<|retrieved_memory|>", "<|/retrieved_memory|>"),
    "trajectory": ("<|trajectory|>", "<|/trajectory|>"),
    "generation_suffix": ("<|generation_suffix|>", "<|/generation_suffix|>"),
}


H2O_RUNTIME_STATS = {
    "events": 0,
    "applied_events": 0,
    "evicted_tokens": 0,
    "seq_before_sum": 0,
    "seq_after_sum": 0,
    "kept_recent_sum": 0,
    "kept_heavy_sum": 0,
}


def reset_h2o_runtime_stats():
    for key in H2O_RUNTIME_STATS:
        H2O_RUNTIME_STATS[key] = 0


def record_h2o_runtime_stats(stats):
    if not isinstance(stats, dict):
        return
    H2O_RUNTIME_STATS["events"] += 1
    seq_before = int(stats.get("seq_before", 0) or 0)
    seq_after = int(stats.get("seq_after", seq_before) or seq_before)
    H2O_RUNTIME_STATS["seq_before_sum"] += seq_before
    H2O_RUNTIME_STATS["seq_after_sum"] += seq_after
    if stats.get("applied"):
        H2O_RUNTIME_STATS["applied_events"] += 1
        H2O_RUNTIME_STATS["evicted_tokens"] += max(0, seq_before - seq_after)
        H2O_RUNTIME_STATS["kept_recent_sum"] += int(stats.get("kept_recent", 0) or 0)
        H2O_RUNTIME_STATS["kept_heavy_sum"] += int(stats.get("kept_heavy", 0) or 0)


def get_h2o_runtime_stats():
    return dict(H2O_RUNTIME_STATS)


def legacy_cache_seq_len(cache):
    if not isinstance(cache, tuple) or not cache:
        return 0
    first_layer = cache[0]
    if not isinstance(first_layer, tuple) or len(first_layer) != 2:
        return 0
    return int(first_layer[0].shape[-2])


def _safe_decode_token(tokenizer, token_id):
    try:
        return tokenizer.decode([int(token_id)], skip_special_tokens=True).lower()
    except Exception:
        return ""


def _tokenizer_input_ids(tokenizer, text):
    try:
        encoded = tokenizer(text, add_special_tokens=False, return_tensors=None)
    except TypeError:
        encoded = tokenizer(text)
    input_ids = encoded.get("input_ids", encoded)
    if isinstance(input_ids, torch.Tensor):
        return input_ids.detach().flatten().tolist()
    if input_ids and isinstance(input_ids[0], list):
        return list(input_ids[0])
    return list(input_ids)


def _find_subsequence_positions(sequence, subsequence):
    if not sequence or not subsequence or len(subsequence) > len(sequence):
        return []
    positions = []
    last_start = len(sequence) - len(subsequence) + 1
    for idx in range(last_start):
        if sequence[idx: idx + len(subsequence)] == subsequence:
            positions.append(idx)
    return positions


def protected_prefix_before_marker(tokenizer, input_ids, marker):
    if tokenizer is None or input_ids is None or not marker:
        return None
    token_ids = input_ids.detach().flatten().tolist() if isinstance(input_ids, torch.Tensor) else list(input_ids)
    marker_ids = _tokenizer_input_ids(tokenizer, marker)
    positions = _find_subsequence_positions(token_ids, marker_ids)
    if not positions:
        return None
    return max(0, int(positions[0]))


def protected_suffix_from_marker(tokenizer, input_ids, marker):
    if tokenizer is None or input_ids is None or not marker:
        return None
    token_ids = input_ids.detach().flatten().tolist() if isinstance(input_ids, torch.Tensor) else list(input_ids)
    marker_ids = _tokenizer_input_ids(tokenizer, marker)
    positions = _find_subsequence_positions(token_ids, marker_ids)
    if not positions:
        return None
    return max(0, len(token_ids) - int(positions[0]))


def _goal_alias_tokens(goal_label):
    base_tokens = {
        token.strip().lower()
        for token in str(goal_label or "").replace("_", " ").split()
        if token.strip()
    }
    alias_map = {
        "watch": {"clock", "watch"},
        "tv": {"tv", "television", "monitor"},
        "sofa": {"sofa", "couch"},
        "chair": {"chair", "armchair", "diningchair"},
        "plant": {"plant", "potted", "potted plant", "houseplant"},
        "toilet": {"toilet", "bathroom", "restroom"},
        "sink": {"sink", "bathroom sink", "faucet"},
    }
    expanded_tokens = set(base_tokens)
    for token in list(base_tokens):
        expanded_tokens.update(alias_map.get(token, set()))
    return expanded_tokens


def _goal_cluster_terms(goal_label):
    base_goal = str(goal_label or "").strip().lower()
    cluster_map = {
        "tv": {"remote", "remotecontrol", "monitor", "television", "sidetable", "sofa", "couch"},
        "toilet": {"sink", "faucet", "mirror", "bathtub", "garbagecan", "towel"},
        "watch": {"clock", "alarmclock", "table", "desk", "dresser", "sidetable"},
        "bed": {"pillow", "dresser", "sidetable", "lamp", "chair"},
        "chair": {"table", "desk", "diningtable", "sofa"},
        "sofa": {"tv", "television", "remote", "remotecontrol", "table"},
        "plant": {"window", "lamp", "chair", "table"},
        "laptop": {"desk", "table", "chair", "remotecontrol"},
    }
    return cluster_map.get(base_goal, set())


def build_goal_heavy_scores(tokenizer, input_ids, goal_label):
    """Score cache tokens that explicitly mention the navigation goal.

    This is a lightweight first-pass stand-in for attention heavy hitters. The
    rest of the H2O machinery can stay the same when we later replace these
    scores with accumulated attention mass.
    """
    if tokenizer is None or input_ids is None:
        return None
    if isinstance(input_ids, torch.Tensor):
        token_ids = input_ids.detach().flatten().tolist()
        device = input_ids.device
    else:
        token_ids = list(input_ids)
        device = None
    goal_tokens = _goal_alias_tokens(goal_label)
    if not goal_tokens:
        return torch.zeros(len(token_ids), dtype=torch.float32, device=device)
    goal_weight = h2o_score_config()["goal_weight"]
    scores = []
    for token_id in token_ids:
        token_text = _safe_decode_token(tokenizer, token_id)
        score = 0.0
        for goal_token in goal_tokens:
            if goal_token and goal_token in token_text:
                score += goal_weight
        scores.append(score)
    return torch.tensor(scores, dtype=torch.float32, device=device)


def build_segment_protection_scores(input_ids, weight=None):
    if input_ids is None:
        return None
    if isinstance(input_ids, torch.Tensor):
        seq_len = int(input_ids.detach().flatten().numel())
        device = input_ids.device
    else:
        seq_len = len(list(input_ids))
        device = None
    if seq_len <= 0:
        return None
    if weight is None:
        weight = h2o_score_config()["instruction_weight"]
    return torch.full((seq_len,), float(weight), dtype=torch.float32, device=device)


def build_semantic_heavy_scores(tokenizer, input_ids, goal_label=None):
    if tokenizer is None or input_ids is None:
        return None
    if isinstance(input_ids, torch.Tensor):
        token_ids = input_ids.detach().flatten().tolist()
        device = input_ids.device
    else:
        token_ids = list(input_ids)
        device = None

    score_cfg = h2o_score_config()
    frontier_terms = {"door", "doorway", "window", "frame", "hallway", "entrance"}
    json_terms = {"place", "angle", "objects", "{", "}", "[", "]", "\""}
    goal_terms = _goal_alias_tokens(goal_label)
    cluster_terms = _goal_cluster_terms(goal_label)

    scores = []
    for token_id in token_ids:
        token_text = _safe_decode_token(tokenizer, token_id)
        score = 0.0
        if any(goal_term and goal_term in token_text for goal_term in goal_terms):
            score += score_cfg["goal_weight"]
        if any(cluster_term and cluster_term in token_text for cluster_term in cluster_terms):
            score += score_cfg["cluster_weight"]
        if any(frontier_term in token_text for frontier_term in frontier_terms):
            score += score_cfg["frontier_weight"]
        if any(json_term in token_text for json_term in json_terms):
            score += score_cfg["json_weight"]
        scores.append(score)
    return torch.tensor(scores, dtype=torch.float32, device=device)


def build_segment_heavy_scores(tokenizer, input_ids):
    if tokenizer is None or input_ids is None:
        return None
    if isinstance(input_ids, torch.Tensor):
        token_ids = input_ids.detach().flatten().tolist()
        device = input_ids.device
    else:
        token_ids = list(input_ids)
        device = None
    if not token_ids:
        return None

    score_cfg = h2o_score_config()
    scores = torch.zeros(len(token_ids), dtype=torch.float32, device=device)
    segment_weights = {
        "instruction": score_cfg["instruction_weight"],
        "instruction_core": score_cfg["instruction_weight"],
        "decision_context": score_cfg["decision_context_weight"],
        "planner_context": score_cfg["planner_context_weight"],
        "retrieved_memory": score_cfg["retrieved_memory_weight"],
        "trajectory": score_cfg["trajectory_weight"],
        "generation_suffix": score_cfg["generation_suffix_weight"],
    }

    for segment_name, (start_marker, end_marker) in H2O_SEGMENT_MARKERS.items():
        start_ids = _tokenizer_input_ids(tokenizer, start_marker)
        end_ids = _tokenizer_input_ids(tokenizer, end_marker)
        if not start_ids or not end_ids:
            continue
        start_positions = _find_subsequence_positions(token_ids, start_ids)
        end_positions = _find_subsequence_positions(token_ids, end_ids)
        if not start_positions or not end_positions:
            continue
        for start_pos in start_positions:
            start_idx = start_pos + len(start_ids)
            end_idx = None
            for candidate_end in end_positions:
                if candidate_end >= start_idx:
                    end_idx = candidate_end
                    break
            if end_idx is None or end_idx <= start_idx:
                continue
            scores[start_idx:end_idx] += float(segment_weights.get(segment_name, 0.0))
    return scores


def build_query_conditioned_heavy_scores(attentions):
    if not attentions:
        return None
    total_scores = None
    for layer_attention in attentions:
        if layer_attention is None:
            continue
        conditioned_scores = layer_attention.detach().to(dtype=torch.float32)[:, :, -1, :].sum(dim=(0, 1))
        if total_scores is None:
            total_scores = conditioned_scores
        else:
            target_len = max(total_scores.numel(), conditioned_scores.numel())
            total_scores = _pad_scores_left(total_scores, target_len)
            conditioned_scores = _pad_scores_left(conditioned_scores, target_len)
            total_scores = total_scores + conditioned_scores
    return total_scores


def align_scores_to_seq_len(scores, seq_len):
    if scores is None:
        return None
    if not isinstance(scores, torch.Tensor):
        scores = torch.tensor(scores, dtype=torch.float32)
    scores = scores.detach().flatten().to(dtype=torch.float32)
    if scores.numel() == seq_len:
        return scores
    if scores.numel() > seq_len:
        return scores[-seq_len:]
    pad = torch.zeros(seq_len - scores.numel(), dtype=scores.dtype, device=scores.device)
    return torch.cat([pad, scores], dim=0)


def build_attention_heavy_scores(attentions):
    """Build per-cache-token scores from received attention mass.

    H2O keeps tokens that many later queries attend to. Hugging Face returns
    attentions as one tensor per layer with shape [batch, heads, query, key].
    Summing over layers, heads, and query positions gives a lightweight
    heavy-hitter score aligned to the current KV-cache key dimension.
    """
    if not attentions:
        return None
    total_scores = None
    for layer_attention in attentions:
        if layer_attention is None:
            continue
        layer_scores = layer_attention.detach().to(dtype=torch.float32).sum(dim=(0, 1, 2))
        if total_scores is None:
            total_scores = layer_scores
        else:
            target_len = max(total_scores.numel(), layer_scores.numel())
            total_scores = _pad_scores_left(total_scores, target_len)
            layer_scores = _pad_scores_left(layer_scores, target_len)
            total_scores = total_scores + layer_scores
    return total_scores


def _pad_scores_left(scores, target_len):
    scores = scores.detach().flatten().to(dtype=torch.float32)
    if scores.numel() >= target_len:
        return scores[-target_len:]
    pad = torch.zeros(target_len - scores.numel(), dtype=scores.dtype, device=scores.device)
    return torch.cat([pad, scores], dim=0)


def merge_heavy_scores(existing_scores, new_scores):
    if existing_scores is None:
        return None if new_scores is None else new_scores.detach().flatten().to(dtype=torch.float32)
    if new_scores is None:
        return existing_scores.detach().flatten().to(dtype=torch.float32)
    target_len = max(existing_scores.numel(), new_scores.numel())
    existing_scores = _pad_scores_left(existing_scores, target_len)
    new_scores = _pad_scores_left(new_scores, target_len).to(existing_scores.device)
    return existing_scores + new_scores


def trim_heavy_scores(heavy_scores, keep_indices):
    if heavy_scores is None or keep_indices is None:
        return heavy_scores
    heavy_scores = heavy_scores.detach().flatten().to(dtype=torch.float32)
    if heavy_scores.numel() == 0:
        return heavy_scores
    index_tensor = torch.tensor(keep_indices, dtype=torch.long, device=heavy_scores.device)
    index_tensor = index_tensor[index_tensor < heavy_scores.numel()]
    if index_tensor.numel() == 0:
        return torch.zeros(0, dtype=heavy_scores.dtype, device=heavy_scores.device)
    return heavy_scores.index_select(0, index_tensor)


def _normalize_scores(heavy_scores, seq_len):
    if heavy_scores is None:
        return None
    if not isinstance(heavy_scores, torch.Tensor):
        heavy_scores = torch.tensor(heavy_scores, dtype=torch.float32)
    heavy_scores = heavy_scores.detach().flatten().to(dtype=torch.float32)
    if heavy_scores.numel() < seq_len:
        pad = torch.zeros(seq_len - heavy_scores.numel(), dtype=heavy_scores.dtype, device=heavy_scores.device)
        heavy_scores = torch.cat([pad, heavy_scores], dim=0)
    elif heavy_scores.numel() > seq_len:
        heavy_scores = heavy_scores[-seq_len:]
    return heavy_scores


def apply_h2o_to_legacy_cache(
    cache,
    heavy_scores=None,
    budget=None,
    recent_size=None,
    heavy_size=None,
    protected_prefix=None,
    protected_suffix=0,
    label="",
):
    if not h2o_enabled() or not isinstance(cache, tuple) or not cache:
        return cache, {"applied": False, "reason": "disabled"}
    default_budget, default_recent, default_heavy, default_protected = h2o_config()
    budget = default_budget if budget is None else max(0, int(budget))
    recent_size = default_recent if recent_size is None else max(0, int(recent_size))
    heavy_size = default_heavy if heavy_size is None else max(0, int(heavy_size))
    protected_prefix = default_protected if protected_prefix is None else max(0, int(protected_prefix))
    protected_suffix = max(0, int(protected_suffix or 0))

    seq_len = legacy_cache_seq_len(cache)
    prefix_outside_budget = h2o_protected_prefix_outside_budget()
    if prefix_outside_budget:
        protected_prefix = min(protected_prefix, seq_len)
        protected_suffix = min(protected_suffix, max(0, seq_len - protected_prefix))
    else:
        protected_prefix = min(protected_prefix, budget, seq_len)
        protected_suffix = min(protected_suffix, max(0, budget - protected_prefix, seq_len - protected_prefix))
    keep = set(range(protected_prefix))
    suffix_start = max(protected_prefix, seq_len - protected_suffix)
    suffix_indices = list(range(suffix_start, seq_len))
    keep.update(suffix_indices)
    protected_total = len(keep)
    effective_budget = min(budget, max(0, seq_len - protected_total)) if prefix_outside_budget else max(0, budget - protected_total)
    target_keep_count = min(seq_len, protected_total + effective_budget)

    if budget <= 0 or seq_len <= target_keep_count:
        stats = {
            "applied": False,
            "reason": "within_budget",
            "label": label,
            "seq_before": seq_len,
            "seq_after": seq_len,
            "budget": budget,
            "protected_prefix": protected_prefix,
            "protected_suffix": protected_suffix,
            "protected_prefix_outside_budget": prefix_outside_budget,
        }
        record_h2o_runtime_stats(stats)
        return cache, stats

    recent_budget = min(recent_size, max(0, target_keep_count - len(keep)), seq_len - len(keep))
    recent_indices = list(range(max(protected_prefix, seq_len - recent_budget), seq_len))
    keep.update(recent_indices)

    heavy_indices = []
    remaining_budget = target_keep_count - len(keep)
    scores = _normalize_scores(heavy_scores, seq_len)
    if scores is not None and remaining_budget > 0 and heavy_size > 0:
        scores = scores.cpu()
        candidate_indices = [idx for idx in range(protected_prefix, seq_len) if idx not in keep]
        candidate_indices.sort(key=lambda idx: (float(scores[idx]), idx), reverse=True)
        heavy_indices = candidate_indices[: min(heavy_size, remaining_budget, len(candidate_indices))]
        keep.update(heavy_indices)

    remaining_budget = target_keep_count - len(keep)
    if remaining_budget > 0:
        fill_indices = [
            idx for idx in range(seq_len - 1, protected_prefix - 1, -1)
            if idx not in keep
        ][:remaining_budget]
        keep.update(fill_indices)

    keep_indices = sorted(keep)
    trimmed_cache = []
    for layer_cache in cache:
        key_states, value_states = layer_cache
        index_tensor = torch.tensor(keep_indices, dtype=torch.long, device=key_states.device)
        trimmed_cache.append(
            (
                key_states.index_select(-2, index_tensor).detach(),
                value_states.index_select(-2, index_tensor.to(value_states.device)).detach(),
            )
        )

    stats = {
        "applied": True,
        "label": label,
        "seq_before": seq_len,
        "seq_after": len(keep_indices),
        "budget": budget,
        "kept_recent": len(recent_indices),
        "kept_heavy": len(heavy_indices),
        "protected_prefix": protected_prefix,
        "protected_suffix": protected_suffix,
        "protected_prefix_outside_budget": prefix_outside_budget,
        "keep_indices": keep_indices,
    }
    record_h2o_runtime_stats(stats)
    return tuple(trimmed_cache), stats
