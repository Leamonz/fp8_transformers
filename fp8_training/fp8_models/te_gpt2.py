import os
import gc
import re
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformer_engine.pytorch as te
import transformers
import logging

from transformers import GPT2Config, GPT2PreTrainedModel
from transformers.utils import WEIGHTS_INDEX_NAME
from transformers.modeling_utils import _add_variant, load_state_dict
from transformers.modeling_outputs import (
    CausalLMOutputWithCrossAttentions,
    BaseModelOutputWithPastAndCrossAttentions,
)
from transformers.cache_utils import Cache, DynamicCache, EncoderDecoderCache
from transformers.masking_utils import create_causal_mask
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask_for_sdpa
from transformers.utils.hub import get_checkpoint_shard_files
from transformers.activations import ACT2FN
from functools import partial
from typing import Optional, Union

logger = logging.get_logger(__name__)


class TEDynamicLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        init_method=None,
        use_fp8=True,
        fp8_recipe=None,
    ):
        super().__init__()
        self.use_fp8 = use_fp8
        self.fp8_recipe = fp8_recipe
        if self.use_fp8 and self.fp8_recipe is None:
            raise ValueError("You have to specify fp8 recipe for fp8 training.")
        self.linear = te.Linear(
            in_features, out_features, bias=bias, init_method=init_method
        )

    def forward(self, x):
        with te.autocast(fp8_enabled=self.use_fp8, fp8_recipe=self.fp8_recipe):
            x = self.linear(x)
        return x


class TEGPT2FFN(nn.Module):
    def __init__(self, intermediate_size, config):
        super().__init__()
        self.c_fc = TEDynamicLinear(
            config.hidden_size,
            intermediate_size,
            init_method=partial(
                nn.init.normal_,
                mean=0.0,
                std=0.02 / math.sqrt(2 * config.num_hidden_layer),
            ),
        )
        self.c_proj = TEDynamicLinear(
            intermediate_size,
            config.hidden_size,
            init_method=partial(
                nn.init.normal_,
                mean=0.0,
                std=0.02 / math.sqrt(2 * config.num_hidden_layer),
            ),
        )
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TEGPT2Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.attn = te.MultiHeadAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            attention_dropout=config.attn_pdrop,
            layernorm_epsilon=config.layer_norm_epsilon,
            init_method=partial(torch.nn.init.normal_, mean=0.0, std=0.02),
            output_layer_init_method=partial(
                torch.nn.init.normal_,
                mean=0.0,
                std=0.02 / math.sqrt(2 * config.num_hidden_layer),
            ),
            layer_number=layer_idx,
            qkv_format="bshd",
            fuse_qkv_params=True,
        )
        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        self.mlp = TEGPT2FFN(
            config.n_inner if config.n_inner is not None else 4 * config.hidden_size,
            config,
        )

    def forward(
        self,
        hidden_states: Optional[tuple[torch.FloatTensor]],
        attention_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> Union[
        tuple[torch.Tensor],
        Optional[tuple[torch.Tensor, tuple[torch.FloatTensor, ...]]],
    ]:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output, self_attn_weights = self.attn(
            hidden_states,
            attention_mask=attention_mask,
            **kwargs,
        )
        # residual connection
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


class TEGPT2Model(GPT2PreTrainedModel):
    def __init__(self, config: GPT2Config, fp8_recipe: te.common.Recipe):
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList(
            [
                TEGPT2Block(config, layer_idx=i, fp8_recipe=fp8_recipe)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[tuple, BaseModelOutputWithPastAndCrossAttentions]:
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`):
            `input_ids_length` = `sequence_length` if `past_key_values` is `None` else
            `past_key_values.get_seq_length()` (`sequence_length` of input past key value states). Indices of input
            sequence tokens in the vocabulary.

            If `past_key_values` is used, only `input_ids` that do not have their past calculated should be passed as
            `input_ids`.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)

        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds.to(inputs_embeds.device)

        # Attention mask.
        # ._update_causal_mask() and ._prepare_4d_causal_attention_mask_with_cache_position() copied from LlamaModel
        if attention_mask is not None and attention_mask.ndim < 4:
            attention_mask = attention_mask.view(batch_size, -1)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        _use_sdpa = self._attn_implementation == "sdpa" and output_attentions is False
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = (
                encoder_hidden_states.size()
            )
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            if _use_sdpa:
                encoder_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                    mask=encoder_attention_mask,
                    dtype=inputs_embeds.dtype,
                    tgt_len=input_shape[-1],
                )
            elif self._attn_implementation != "flash_attention_2":
                encoder_attention_mask = self.invert_attention_mask(
                    encoder_attention_mask
                )
        else:
            encoder_attention_mask = None

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for i, block in enumerate(self.h):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            outputs = block(
                hidden_states,
                causal_mask,
                output_attentions=output_attentions,
                **kwargs,
            )

            hidden_states = outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[1],)

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    all_hidden_states,
                    all_self_attentions,
                ]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class TEGPT2LMHeadModel(GPT2PreTrainedModel):
    def __init__(
        self, config: GPT2Config, fp8_recipe: te.common.Recipe, last_linear_fp8: bool
    ):
        self.embed_dim = config.hidden_size
        self.transformer = TEGPT2Model(config, fp8_recipe)
        self.lm_head = TEDynamicLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            init_method=partial(torch.nn.init.normal_, mean=0.0, std=0.02),
            use_fp8=last_linear_fp8,
            fp8_recipe=fp8_recipe,
        )

        # weight tying
        self.transformer.wte.weight = self.lm_head.weight

        # weights initialization
        self.apply(self._init_weights)

    def num_parameters(
        self,
    ):
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Union[tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`):
            `input_ids_length` = `sequence_length` if `past_key_values` is `None` else
            `past_key_values.get_seq_length()` (`sequence_length` of input past key value states). Indices of input
            sequence tokens in the vocabulary.

            If `past_key_values` is used, only `input_ids` that do not have their past calculated should be passed as
            `input_ids`.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        labels (`torch.LongTensor` of shape `(batch_size, input_ids_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        hidden_states = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            # Flatten the tokens
            loss = self.loss_function(
                logits,
                labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        if not return_dict:
            return loss, logits

        return CausalLMOutputWithCrossAttentions(loss=loss, logits=logits)

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str, config: GPT2Config, *args, **kwargs
    ):
        # Set default dtype if provided
        torch_dtype = kwargs.get("torch_dtype", torch.float32)
        torch.set_default_dtype(torch_dtype)

        # Allow passing fp8 recipe and last_linear_fp8 flag through kwargs
        last_linear_fp8 = kwargs.get("last_linear_fp8", False)

        # Instantiate empty TE model (will load params into it)
        te_model = cls(config, last_linear_fp8)

        # Try to load a Hugging Face GPT2LMHeadModel from hub or local path and convert
        try:
            logger.info(
                f"Attempting to load GPT2LMHeadModel from {pretrained_model_name_or_path}"
            )
            hf_model = transformers.GPT2LMHeadModel.from_pretrained(
                pretrained_model_name_or_path,
                revision=kwargs.get("revision", None),
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=kwargs.get("low_cpu_mem_usage", False),
            )

            # Convert HF weights to TE model format using replace_params
            hf_state_dict = hf_model.state_dict()
            te_state_dict = te_model.state_dict()
            replace_params(hf_state_dict, te_state_dict, config)

            # Load converted TE state dict into te_model
            te_model.load_state_dict(te_state_dict, strict=False)

            # Free HF model memory
            del hf_model
            gc.collect()

            return te_model
        except Exception as e:
            logger.warning(
                f"Failed to load/convert HuggingFace GPT2LMHeadModel: {e}. Falling back to shard loader."
            )

        # Fallback: existing sharded checkpoint loader (expects sharded HF-style checkpoint files)
        subfolder = ""
        variant = None
        if os.path.isfile(
            os.path.join(
                pretrained_model_name_or_path,
                subfolder,
                _add_variant("model.safetensors.index.json", variant),
            )
        ):
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
            archive_file = os.path.join(
                pretrained_model_name_or_path,
                subfolder,
                _add_variant(WEIGHTS_INDEX_NAME, variant),
            )
            is_sharded = True
        else:
            raise AssertionError(
                "Only sharded PyTorch ckpt format supported at the moment and hub load failed"
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
            replace_params(state_dict, te_model.state_dict(), config)
            # load_state_dict copies parameters other than those in TransformerEngine
            te_model.load_state_dict(state_dict, strict=False)

            # Force mem release. Taken from huggingface code
            del state_dict
            gc.collect()

        return te_model


def replace_params(hf_state_dict, te_state_dict, config):
    # collect all layer prefixes to update
    all_layer_prefixes = set()
    for param_key in hf_state_dict.keys():
        layer_prefix_pat = "transformer.h.\d+."
        m = re.match(layer_prefix_pat, param_key)
        if m is not None:
            all_layer_prefixes.add(m.group())

    for layer_prefix in all_layer_prefixes:
        # When loading weights into models with less number of layers, skip the
        # copy if the corresponding layer doesn't exist in HF model
        # Attention weights replacement
        if layer_prefix + "attn.c_attn.weight" in hf_state_dict:
            te_state_dict[layer_prefix + "qkv.weight"] = hf_state_dict[
                layer_prefix + "attn.c_attn.weight"
            ].transpose(0, 1)

        if layer_prefix + "attn.c_attn.bias" in hf_state_dict:
            te_state_dict[layer_prefix + "qkv.bias"] = hf_state_dict[
                layer_prefix + "attn.c_attn.bias"
            ]

        # MLP weights replacement
        # transformers uses (in_feature, out_feature) weight shape, TE uses (out_feature, in_feature), so we need to transpose
        if layer_prefix + "mlp.c_fc.weight" in hf_state_dict:
            te_state_dict[layer_prefix + "mlp.c_fc.linear.weight"] = hf_state_dict[
                layer_prefix + "mlp.c_fc.weight"
            ].transpose(0, 1)

        if layer_prefix + "mlp.c_fc.bias" in hf_state_dict:
            te_state_dict[layer_prefix + "mlp.c_fc.linear.bias"] = hf_state_dict[
                layer_prefix + "mlp.c_fc.bias"
            ]

        if layer_prefix + "mlp.c_proj.weight" in hf_state_dict:
            te_state_dict[layer_prefix + "mlp.c_proj.linear.weight"] = hf_state_dict[
                layer_prefix + "mlp.c_proj.weight"
            ].transpose(0, 1)

        if layer_prefix + "mlp.c_proj.bias" in hf_state_dict:
            te_state_dict[layer_prefix + "mlp.c_proj.linear.bias"] = hf_state_dict[
                layer_prefix + "mlp.c_proj.bias"
            ]

    return all_layer_prefixes
