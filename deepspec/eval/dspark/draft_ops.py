from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import DynamicCache

from deepspec.eval.base_evaluator import DraftProposal
from deepspec.utils.device import device_type
from deepspec.utils.sampling import logits_to_probs
from deepspec.modeling.dspark.gemma4 import Gemma4DSparkModel
from deepspec.modeling.dspark.qwen3 import Qwen3DSparkModel


def _crop_cache(cache, max_length: int):
    """Crop a key-value cache, tolerating caches without a native crop()."""
    if cache is None:
        return
    if hasattr(cache, "crop"):
        cache.crop(max_length)


DSparkModel = Qwen3DSparkModel | Gemma4DSparkModel


@dataclass
class DSparkDraftProposal(DraftProposal):
    confidence_logits: torch.Tensor | None = None


def forward_dspark_draft_block(
    model: DSparkModel,
    *,
    draft_input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    past_key_values_draft: DynamicCache,
    target_hidden_states: torch.Tensor,
    start: int,
    block_size: int,
) -> torch.Tensor:
    # The draft attends to target_hidden_states (context) plus the noise block.
    # Pass absolute positions for both, matching DSpark training which uses
    # full_position_ids = context + draft. Without context positions the RoPE
    # applied to target_hidden K/V is wrong.
    ctx_len = target_hidden_states.shape[1]
    context_positions = torch.arange(
        start - ctx_len, start, device=draft_input_ids.device
    ).unsqueeze(0)
    draft_positions = position_ids[:, start : start + block_size]
    full_position_ids = torch.cat([context_positions, draft_positions], dim=1)

    block_hidden = model._forward_backbone(
        target_hidden_states=target_hidden_states,
        noise_embedding=model.embed_tokens(draft_input_ids),
        position_ids=full_position_ids,
        attention_mask=None,
        past_key_values=past_key_values_draft,
        use_cache=True,
        is_causal=False,
    )
    # Some NPU-compatible caches (e.g. Qwen3.5's hybrid cache) do not expose
    # crop(); only crop when the method exists.
    _crop_cache(past_key_values_draft, start)
    return block_hidden


def _empty_dspark_proposal(draft_input_ids: torch.Tensor) -> DSparkDraftProposal:
    return DSparkDraftProposal(
        draft_token_count=0,
        verify_input_ids=draft_input_ids[:, :1],
        draft_probs=None,
        confidence_logits=None,
    )


def _predict_confidence_logits(
    model: DSparkModel,
    *,
    proposal_hidden_states: torch.Tensor,
    draft_input_ids: torch.Tensor,
    sampled_tokens: torch.Tensor,
    block_size: int,
) -> torch.Tensor | None:
    prev_token_ids = torch.cat(
        [draft_input_ids[:, :1], sampled_tokens[:, :-1]],
        dim=1,
    )
    confidence_pred = model.predict_confidence_step(
        proposal_hidden_states,
        prev_token_ids=prev_token_ids,
    )
    if confidence_pred is None:
        return None
    # CUDA keeps confidence logits in fp32; NPU stays in active dtype to avoid
    # AICore record-task failures observed during eval.
    if device_type() != "npu":
        confidence_pred = confidence_pred.float()
    return confidence_pred.reshape(
        confidence_pred.shape[0],
        block_size,
        -1,
    )[:, :, 0]


def _confident_prefix_length(
    confidence_logits: torch.Tensor,
    *,
    block_size: int,
    threshold: float,
) -> int:
    if threshold <= 0.0:
        return int(block_size)
    below_threshold = confidence_logits.sigmoid() < threshold
    if not bool(below_threshold[0].any().item()):
        return int(block_size)
    return int(torch.nonzero(below_threshold[0], as_tuple=False)[0].item())


def build_dspark_proposal(
    model: DSparkModel,
    *,
    draft_input_ids: torch.Tensor,
    block_hidden: torch.Tensor,
    block_size: int,
    temperature: float,
    confidence_threshold: float,
) -> DSparkDraftProposal:
    assert draft_input_ids.size(0) == 1, "build_dspark_proposal requires batch_size=1"
    proposal_hidden_states = block_hidden[:, :block_size, :]
    base_draft_logits = model.compute_logits(proposal_hidden_states)
    sampled_tokens, draft_logits = model.sample_draft_tokens(
        base_draft_logits,
        first_prev_token_ids=draft_input_ids[:, 0],
        temperature=temperature,
        hidden_states=proposal_hidden_states,
    )

    proposal_draft_tokens = int(block_size)
    confidence_logits = None
    if model.confidence_head is not None:
        confidence_logits = _predict_confidence_logits(
            model,
            proposal_hidden_states=proposal_hidden_states,
            draft_input_ids=draft_input_ids,
            sampled_tokens=sampled_tokens,
            block_size=block_size,
        )
        if confidence_logits is None:
            return _empty_dspark_proposal(draft_input_ids)
        proposal_draft_tokens = _confident_prefix_length(
            confidence_logits,
            block_size=block_size,
            threshold=float(confidence_threshold),
        )

    if proposal_draft_tokens == 0:
        return _empty_dspark_proposal(draft_input_ids)

    verify_input_ids = torch.cat(
        [draft_input_ids[:, :1], sampled_tokens[:, :proposal_draft_tokens]],
        dim=1,
    )
    draft_probs = logits_to_probs(
        draft_logits[:, :proposal_draft_tokens, :],
        temperature,
    )
    return DSparkDraftProposal(
        draft_token_count=proposal_draft_tokens,
        verify_input_ids=verify_input_ids,
        draft_probs=draft_probs,
        confidence_logits=(
            confidence_logits[:, :proposal_draft_tokens]
            if confidence_logits is not None
            else None
        ),
    )
