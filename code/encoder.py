import torch
import torch.nn as nn
from transformers import AutoConfig
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model
from typing import Optional, Tuple
from modelscope import snapshot_download

from .config import DEFAULT_PERCEPTION_MODEL_ID

class CustomQwen2ChronoModel(Qwen2Model):
    """
    Qwen2Model variant with:
    1. Causal-Topological Attention Mask
    2. Time-Aware Positional Bias
    """
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        token_type_ids=None,
        time_bias_matrix=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
    ):
        # Cache custom inputs for dynamic mask construction.
        self._current_token_type_ids = token_type_ids
        self._current_time_bias_matrix = time_bias_matrix

        if not getattr(self, "model_parallel_enabled", False):
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
            )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("Exactly one of `input_ids` or `inputs_embeds` must be provided.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids.to(self.input_device))
        else:
            inputs_embeds = inputs_embeds.to(self.input_device)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=self.input_device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)
        else:
            position_ids = position_ids.to(self.input_device)

        if not isinstance(attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
        else:
            causal_mask_mapping = attention_mask

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        decoder_kwargs = {
            "cache_position": cache_position,
        }

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            layer_device = self.layer_devices[i]
            hidden_states = hidden_states.to(layer_device)
            layer_position_ids = position_ids.to(layer_device)
            layer_attention_mask = self._move_to_device(
                causal_mask_mapping[self.config.layer_types[i]],
                layer_device
            )
            layer_position_embeddings = self._move_to_device(position_embeddings, layer_device)
            layer_decoder_kwargs = {
                key: self._move_to_device(value, layer_device)
                for key, value in decoder_kwargs.items()
                if value is not None
            }
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=layer_attention_mask,
                position_embeddings=layer_position_embeddings,
                position_ids=layer_position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **layer_decoder_kwargs,
            )

        hidden_states = hidden_states.to(self.norm_device)
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.to(self.output_device)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )

    @staticmethod
    def _move_to_device(value, device):
        if torch.is_tensor(value):
            return value.to(device)
        if isinstance(value, tuple):
            return tuple(CustomQwen2ChronoModel._move_to_device(item, device) for item in value)
        if isinstance(value, list):
            return [CustomQwen2ChronoModel._move_to_device(item, device) for item in value]
        if isinstance(value, dict):
            return {
                key: CustomQwen2ChronoModel._move_to_device(item, device)
                for key, item in value.items()
            }
        return value

    def configure_devices(self, devices):
        if not devices:
            raise ValueError("At least one device must be provided.")

        stage_devices = [torch.device(device) for device in devices]
        self.model_parallel_enabled = len(stage_devices) > 1
        self.input_device = stage_devices[0]
        self.output_device = stage_devices[0]
        self.norm_device = stage_devices[-1]
        self.embed_tokens = self.embed_tokens.to(self.input_device)
        self.rotary_emb = self.rotary_emb.to(self.input_device)

        num_layers = len(self.layers)
        base_layers = num_layers // len(stage_devices)
        extra_layers = num_layers % len(stage_devices)
        self.layer_devices = []
        start_index = 0

        for stage_index, stage_device in enumerate(stage_devices):
            layer_count = base_layers + (1 if stage_index < extra_layers else 0)
            end_index = start_index + layer_count
            for layer in self.layers[start_index:end_index]:
                layer.to(stage_device)
                self.layer_devices.append(stage_device)
            start_index = end_index

        self.norm = self.norm.to(self.norm_device)

    def _update_causal_mask(
        self,
        attention_mask,
        input_tensor,
        cache_position,
        past_key_values,
        output_attentions,
    ):
        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        batch_size, sequence_length = input_tensor.shape[0], input_tensor.shape[1]
        
        token_type_ids = self._current_token_type_ids
        intervention_node_id = getattr(self, "_current_intervention_node_id", None)
        
        # Build the Causal-Topological Attention Mask.
        causal_mask = self._create_chrono_4d_mask(
            sequence_length=sequence_length,
            dtype=dtype,
            device=device,
            batch_size=batch_size,
            token_type_ids=token_type_ids,
            intervention_node_id=intervention_node_id
        )
        
        # Apply the standard padding mask.
        if attention_mask is not None and attention_mask.dim() == 2:
            padding_mask = attention_mask[:, None, None, :].to(dtype=dtype)
            padding_mask = (1.0 - padding_mask) * min_dtype
            causal_mask = causal_mask + padding_mask

        # Inject the Time-Aware Positional Bias.
        if self._current_time_bias_matrix is not None:
            causal_mask = causal_mask + self._current_time_bias_matrix.to(dtype=dtype)
            
        return causal_mask

    def _create_chrono_4d_mask(
        self,
        sequence_length,
        dtype,
        device,
        batch_size,
        token_type_ids,
        intervention_node_id=None
    ):
        min_dtype = torch.finfo(dtype).min
        masks = []
        
        for b in range(batch_size):
            # Start from a fully masked attention matrix.
            mask = torch.full(
                (sequence_length, sequence_length),
                fill_value=min_dtype,
                dtype=dtype,
                device=device
            )
            
            type_ids = token_type_ids[b]
            
            v_positions = (type_ids == 0).nonzero(as_tuple=True)[0]
            q_positions = (type_ids == 1).nonzero(as_tuple=True)[0]
            
            if len(v_positions) > 0:
                mask[v_positions[:, None], v_positions] = 0.0
            
            for i, q_pos in enumerate(q_positions):
                if intervention_node_id is not None and i == intervention_node_id:
                    continue
                
                if len(v_positions) > 0:
                    mask[q_pos, v_positions] = 0.0
                
                for j, other_q_pos in enumerate(q_positions):
                    if intervention_node_id is not None and j == intervention_node_id:
                        continue
                    
                    if j <= i:
                        mask[q_pos, other_q_pos] = 0.0
            
            masks.append(mask)
        
        mask = torch.stack(masks, dim=0).unsqueeze(1)
        return mask


class TemporalCausalEncoder(nn.Module):
    """
    Temporal Causal Encoder.

    Extracts causal latent variables from fragmented text.
    """
    def __init__(
        self,
        model_id=DEFAULT_PERCEPTION_MODEL_ID,
        max_event_anchors: int = 128,
        attn_implementation: str = "sdpa"
    ):
        super().__init__()
        self.max_event_anchors = max_event_anchors
        
        print(f"Loading Temporal Causal Encoder backbone: {model_id}")
        model_dir = snapshot_download(model_id)
        
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        config._attn_implementation = attn_implementation
        
        self.hidden_dim = config.hidden_size
        
        self.encoder = CustomQwen2ChronoModel(config)
        
        self.word_embeddings = nn.Embedding(config.vocab_size, self.hidden_dim)
            
        # Event Anchors act as causal query tokens.
        self.event_anchors = nn.Embedding(max_event_anchors, self.hidden_dim)
        
        # Time Span Bias provides temporal priors for each event slot.
        self.time_span_bias = nn.Embedding(max_event_anchors, self.hidden_dim)

        self.time_bias_proj_q = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.time_bias_proj_v = nn.Linear(self.hidden_dim, self.hidden_dim)

        # Auxiliary Timestamp Prediction head.
        self.timestamp_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, 1)
        )
        self.compute_device = torch.device("cpu")

    def configure_devices(self, devices):
        if not devices:
            raise ValueError("At least one device must be provided.")

        stage_devices = [torch.device(device) for device in devices]
        self.compute_device = stage_devices[0]
        self.word_embeddings = self.word_embeddings.to(self.compute_device)
        self.event_anchors = self.event_anchors.to(self.compute_device)
        self.time_span_bias = self.time_span_bias.to(self.compute_device)
        self.time_bias_proj_q = self.time_bias_proj_q.to(self.compute_device)
        self.time_bias_proj_v = self.time_bias_proj_v.to(self.compute_device)
        self.timestamp_head = self.timestamp_head.to(self.compute_device)
        self.encoder.configure_devices(stage_devices)

    def forward(
        self, 
        text_chunks: list[list[str]], 
        intervention_node_id: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            text_chunks: List of text chunk sequences.
            intervention_node_id: Target event index for counterfactual intervention.

        Returns:
            z_causal: Causal latent variable sequence ``[batch_size, num_queries, hidden_dim]``.
            pred_timestamps: Predicted timestamps ``[batch_size, num_queries, 1]``.
            v_tokens: Text token features ``[batch_size, num_v_tokens, hidden_dim]``.
        """
        batch_size = len(text_chunks)
        
        joined_texts = ["\n".join(chunks) if chunks else "No content." for chunks in text_chunks]
        inputs = self.tokenizer(joined_texts, return_tensors="pt", padding=True, truncation=True)
        compute_device = self.word_embeddings.weight.device
        self.compute_device = compute_device
        input_ids = inputs["input_ids"].to(compute_device)
        v_tokens = self.word_embeddings(input_ids)
        num_v = v_tokens.shape[1]
        
        anchor_indices = torch.arange(self.max_event_anchors, device=v_tokens.device)
        q_tokens = self.event_anchors(anchor_indices) + self.time_span_bias(anchor_indices)
        q_tokens = q_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        
        q_proj = self.time_bias_proj_q(q_tokens)
        v_proj = self.time_bias_proj_v(v_tokens)
        raw_time_bias = torch.bmm(q_proj, v_proj.transpose(1, 2))
        
        seq_len = num_v + self.max_event_anchors
        time_bias_matrix = torch.zeros(batch_size, 1, seq_len, seq_len, device=v_tokens.device, dtype=v_tokens.dtype)
        time_bias_matrix[:, 0, num_v:, :num_v] = raw_time_bias
        
        combined_inputs = torch.cat([v_tokens, q_tokens], dim=1)
        
        type_v = torch.zeros(batch_size, num_v, dtype=torch.long, device=v_tokens.device)
        type_q = torch.ones(batch_size, self.max_event_anchors, dtype=torch.long, device=v_tokens.device)
        token_type_ids = torch.cat([type_v, type_q], dim=1)
            
        self.encoder._current_intervention_node_id = intervention_node_id
        encoder_outputs = self.encoder(
            inputs_embeds=combined_inputs,
            token_type_ids=token_type_ids,
            time_bias_matrix=time_bias_matrix
        )
        
        hidden_states = encoder_outputs[0]
        
        z_causal = hidden_states[:, num_v:, :]
        
        pred_timestamps = self.timestamp_head(z_causal)
        
        return z_causal, pred_timestamps, v_tokens
