import os
import gc
import re
import math
import torch
import transformer_engine.pytorch as te
import transformers

from transformers import GPT2Model, GPT2LMHeadModel, GPT2Config
from transformers.utils import WEIGHTS_INDEX_NAME
from transformers.modeling_utils import _add_variant, load_state_dict
from transformers.utils.hub import get_checkpoint_shard_files
from contextlib import contextmanager
from functools import partial


@contextmanager
def replace_gpt2_blocks(te_gpt2blk_cls):
    """Context manager to temporarily replace GPT2 blocks with TE GPT2 blocks."""
    original_blk_cls = transformers.models.gpt2.modeling_gpt2.GPT2Block
    transformers.models.gpt2.modeling_gpt2.GPT2Block = te_gpt2blk_cls
    try:
        yield
    finally:
        transformers.models.gpt2.modeling_gpt2.GPT2Block = original_blk_cls


class TEGPT2Block(te.TransformerLayer):
    def __init__(self, config: GPT2Config, layer_idx, *args, **kwargs):
        super().__init__(
            hidden_size=config.hidden_size,
            ffn_hidden_size=(
                config.n_inner if config.n_inner is not None else 4 * config.hidden_size
            ),
            num_attention_heads=config.num_attention_heads,
            bias=True,
            layernorm_epsilon=config.layer_norm_epsilon,
            fuse_qkv_params=True,
            hidden_dropout=config.resid_pdrop,
            attention_dropout=config.attn_pdrop,
            activation_function="gelu",
            attn_input_format="bshd",
            layer_number=layer_idx,
            init_method=partial(torch.nn.init.normal_, mean=0.0, std=0.02),
            output_layer_init_method=partial(
                torch.nn.init.normal_,
                mean=0.0,
                std=0.02 / math.sqrt(2 * config.num_hidden_layer),
            ),
            *args,
            **kwargs,
        )

    def forward(self, hidden_states, attention_mask, *args, **kwargs):
        return (
            super().forward(
                hidden_states, attention_mask=attention_mask, *args, **kwargs
            ),
        )


class TEGPT2LMHeadModel:
    def __init__(self, config: GPT2Config):
        with replace_gpt2_blocks(TEGPT2Block):
            self.transformer = GPT2Model(config)
        self.lm_head = te.LayerNormLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            init_method=partial(torch.nn.init.normal_, mean=0.0, std=0.02),
        )

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str, *args, config, **kwargs
    ):
        # Before loading the model, set the default dtype for torch
        torch.set_default_dtype(kwargs["torch_dtype"])

        # Load the vanilla model weights
        vanilla_model = cls(config)
        subfolder = ""
        variant = None
        if os.path.isfile(
            os.path.join(
                pretrained_model_name_or_path,
                subfolder,
                _add_variant("model.safetensors.index.json", variant),
            )
        ):
            # Load from a sharded PyTorch checkpoint
            archive_file = os.path.join(
                pretrained_model_name_or_path,
                subfolder,
                _add_variant("model.safetensors.index.json", variant),
            )
            is_sharded = True
        elif os.path.isfile(
            os.path.join(
                pretrained_model_name_or_path,
                subfolder,
                _add_variant(WEIGHTS_INDEX_NAME, variant),
            )
        ):
            # Load from a sharded PyTorch checkpoint
            archive_file = os.path.join(
                pretrained_model_name_or_path,
                subfolder,
                _add_variant(WEIGHTS_INDEX_NAME, variant),
            )
            is_sharded = True
        else:
            raise AssertionError(
                "Only sharded PyTorch ckpt format supported at the moment"
            )

        resolved_archive_file, _ = get_checkpoint_shard_files(
            pretrained_model_name_or_path,
            archive_file,
        )

        # If the checkpoint is not sharded, it's a trivial sharding case
        if not is_sharded:
            assert not isinstance(resolved_archive_file, list)
            resolved_archive_file = [resolved_archive_file]

        for shard_file in resolved_archive_file:
            state_dict = load_state_dict(shard_file)
            # replace_params copies parameters relevant only to TransformerEngine
            replace_params(state_dict, vanilla_model.state_dict(), config)
            # load_state_dict copies parameters other than those in TransformerEngine
            vanilla_model.load_state_dict(state_dict, strict=False)

            # Force mem release. Taken from huggingface code
            del state_dict
            gc.collect()

        return vanilla_model


# TODO: modify to support GPT2 model replacement
def replace_params(hf_state_dict, te_state_dict, config):
    # collect all layer prefixes to update
    all_layer_prefixes = set()
    for param_key in hf_state_dict.keys():
        layer_prefix_pat = "model.layers.\d+."
        m = re.match(layer_prefix_pat, param_key)
        if m is not None:
            all_layer_prefixes.add(m.group())

    for layer_prefix in all_layer_prefixes:
        # When loading weights into models with less number of layers, skip the
        # copy if the corresponding layer doesn't exist in HF model
        if layer_prefix + "input_layernorm.weight" in hf_state_dict:
            te_state_dict[
                layer_prefix + "self_attention.layernorm_qkv.layer_norm_weight"
            ].data[:] = hf_state_dict[layer_prefix + "input_layernorm.weight"].data[:]

        if layer_prefix + "self_attn.q_proj.weight" in hf_state_dict:
            te_state_dict[
                layer_prefix + "self_attention.layernorm_qkv.query_weight"
            ].data[:] = hf_state_dict[layer_prefix + "self_attn.q_proj.weight"].data[:]

        if layer_prefix + "self_attn.k_proj.weight" in hf_state_dict:
            te_state_dict[
                layer_prefix + "self_attention.layernorm_qkv.key_weight"
            ].data[:] = hf_state_dict[layer_prefix + "self_attn.k_proj.weight"].data[:]

        if layer_prefix + "self_attn.v_proj.weight" in hf_state_dict:
            te_state_dict[
                layer_prefix + "self_attention.layernorm_qkv.value_weight"
            ].data[:] = hf_state_dict[layer_prefix + "self_attn.v_proj.weight"].data[:]

        if layer_prefix + "self_attn.o_proj.weight" in hf_state_dict:
            te_state_dict[layer_prefix + "self_attention.proj.weight"].data[:] = (
                hf_state_dict[layer_prefix + "self_attn.o_proj.weight"].data[:]
            )

        if layer_prefix + "post_attention_layernorm.weight" in hf_state_dict:
            te_state_dict[layer_prefix + "layernorm_mlp.layer_norm_weight"].data[:] = (
                hf_state_dict[layer_prefix + "post_attention_layernorm.weight"].data[:]
            )

        # It may happen that gate_proj.weight and up_proj.weight will be in the different files, so we need to
        # load them separately.
        if layer_prefix + "mlp.gate_proj.weight" in hf_state_dict:
            te_state_dict[layer_prefix + "layernorm_mlp.fc1_weight"].data[
                : config.intermediate_size
            ] = hf_state_dict[layer_prefix + "mlp.gate_proj.weight"].data

        if layer_prefix + "mlp.up_proj.weight" in hf_state_dict:
            te_state_dict[layer_prefix + "layernorm_mlp.fc1_weight"].data[
                config.intermediate_size :
            ] = hf_state_dict[layer_prefix + "mlp.up_proj.weight"].data

        if layer_prefix + "mlp.down_proj.weight" in hf_state_dict:
            te_state_dict[layer_prefix + "layernorm_mlp.fc2_weight"].data[:] = (
                hf_state_dict[layer_prefix + "mlp.down_proj.weight"].data[:]
            )
    return all_layer_prefixes
