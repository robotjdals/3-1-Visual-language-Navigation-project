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


def h2o_config():
    budget = _env_int("EFFICIENTNAV_H2O_CACHE_BUDGET", 2048)
    recent_size = _env_int("EFFICIENTNAV_H2O_RECENT_SIZE", 256)
    heavy_size = _env_int("EFFICIENTNAV_H2O_HEAVY_SIZE", 256)
    protected_prefix = _env_int("EFFICIENTNAV_H2O_PROTECTED_PREFIX", 0)
    return budget, recent_size, heavy_size, protected_prefix


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
    goal_tokens = {
        token.strip().lower()
        for token in str(goal_label or "").replace("_", " ").split()
        if token.strip()
    }
    if not goal_tokens:
        return torch.zeros(len(token_ids), dtype=torch.float32, device=device)
    scores = []
    for token_id in token_ids:
        token_text = _safe_decode_token(tokenizer, token_id)
        score = 0.0
        for goal_token in goal_tokens:
            if goal_token and goal_token in token_text:
                score += 1.0
        scores.append(score)
    return torch.tensor(scores, dtype=torch.float32, device=device)


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
    label="",
):
    if not h2o_enabled() or not isinstance(cache, tuple) or not cache:
        return cache, {"applied": False, "reason": "disabled"}
    default_budget, default_recent, default_heavy, default_protected = h2o_config()
    budget = default_budget if budget is None else max(0, int(budget))
    recent_size = default_recent if recent_size is None else max(0, int(recent_size))
    heavy_size = default_heavy if heavy_size is None else max(0, int(heavy_size))
    protected_prefix = default_protected if protected_prefix is None else max(0, int(protected_prefix))

    seq_len = legacy_cache_seq_len(cache)
    if budget <= 0 or seq_len <= budget:
        return cache, {
            "applied": False,
            "reason": "within_budget",
            "label": label,
            "seq_before": seq_len,
            "seq_after": seq_len,
            "budget": budget,
        }

    protected_prefix = min(protected_prefix, budget, seq_len)
    keep = set(range(protected_prefix))

    recent_budget = min(recent_size, budget - len(keep), seq_len - len(keep))
    recent_indices = list(range(max(protected_prefix, seq_len - recent_budget), seq_len))
    keep.update(recent_indices)

    heavy_indices = []
    remaining_budget = budget - len(keep)
    scores = _normalize_scores(heavy_scores, seq_len)
    if scores is not None and remaining_budget > 0 and heavy_size > 0:
        scores = scores.cpu()
        candidate_indices = [idx for idx in range(protected_prefix, seq_len) if idx not in keep]
        candidate_indices.sort(key=lambda idx: (float(scores[idx]), idx), reverse=True)
        heavy_indices = candidate_indices[: min(heavy_size, remaining_budget, len(candidate_indices))]
        keep.update(heavy_indices)

    remaining_budget = budget - len(keep)
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

    return tuple(trimmed_cache), {
        "applied": True,
        "label": label,
        "seq_before": seq_len,
        "seq_after": len(keep_indices),
        "budget": budget,
        "kept_recent": len(recent_indices),
        "kept_heavy": len(heavy_indices),
        "protected_prefix": protected_prefix,
        "keep_indices": keep_indices,
    }
