import copy

from deepspec.modeling.dspark.common import validate_target_layer_ids
from deepspec.utils.device import get_device_type


TRAIN_ATTN_IMPLEMENTATION = "sdpa" if get_device_type() == "npu" else "flex_attention"


def _get_qwen3_text_config(target_config):
    """Return the text config used by Qwen3 / Qwen3.5 target models.

    Qwen3.5 uses a composite config where the language-model parameters live in
    ``text_config`` (similar to Gemma4).  For plain Qwen3 the target config
    itself is the text config.
    """
    model_type = str(getattr(target_config, "model_type", ""))
    if model_type in ("qwen3_5",):
        assert hasattr(target_config, "text_config"), (
            "Qwen3.5 target config must expose a text_config."
        )
        return copy.deepcopy(target_config.text_config)
    return copy.deepcopy(target_config)


def _validate_required_text_fields(text_config):
    required_fields = (
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "attention_bias",
        "attention_dropout",
        "hidden_act",
        "max_position_embeddings",
        "rms_norm_eps",
        "rope_theta",
        "head_dim",
        "tie_word_embeddings",
    )
    for field in required_fields:
        assert hasattr(text_config, field), (
            f"target text config.{field} must be provided."
        )


def build_draft_config(
    target_config,
    model_args,
):
    text_config = _get_qwen3_text_config(target_config)
    _validate_required_text_fields(text_config)

    num_target_layers = int(text_config.num_hidden_layers)
    num_draft_layers = int(model_args.num_draft_layers)
    layer_types = ["full_attention"] * num_draft_layers
    assert "target_layer_ids" in model_args, "target_layer_ids must be provided."
    target_layer_ids = validate_target_layer_ids(
        model_args.target_layer_ids,
        num_target_layers,
    )

    confidence_head_alpha = float(model_args.confidence_head_alpha)
    assert confidence_head_alpha >= 0.0
    enable_confidence_head = confidence_head_alpha > 0.0
    if enable_confidence_head:
        assert "confidence_head_with_markov" in model_args, (
            "confidence_head_with_markov must be provided when "
            "confidence_head_alpha > 0."
        )
    markov_rank = int(model_args.markov_rank)
    assert markov_rank >= 0, f"markov_rank must be >= 0, got {markov_rank}"
    if markov_rank > 0:
        assert "markov_head_type" in model_args, (
            "markov_head_type must be provided when markov_rank > 0."
        )

    draft_config = text_config
    draft_config.architectures = ["Qwen3DSparkModel"]
    draft_config.num_target_layers = num_target_layers
    draft_config.num_hidden_layers = num_draft_layers
    draft_config.block_size = int(model_args.block_size)
    draft_config.tie_word_embeddings = False
    draft_config.layer_types = layer_types
    draft_config._attn_implementation = TRAIN_ATTN_IMPLEMENTATION
    draft_config.mask_token_id = int(model_args.mask_token_id)
    draft_config.target_layer_ids = target_layer_ids
    draft_config.num_anchors = int(model_args.num_anchors)
    draft_config.enable_confidence_head = enable_confidence_head
    if enable_confidence_head:
        draft_config.confidence_head_with_markov = bool(
            model_args.confidence_head_with_markov
        )
    draft_config.markov_rank = markov_rank
    if markov_rank > 0:
        draft_config.markov_head_type = str(model_args.markov_head_type)

    # Force model_type to "qwen3" so that Qwen3PreTrainedModel / AutoConfig
    # route the draft checkpoint through the existing Qwen3 modeling path.
    # The underlying transformer architecture is identical.
    draft_config.model_type = "qwen3"

    return draft_config


__all__ = [
    "build_draft_config",
]
