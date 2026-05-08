import re
import torch
import torch.nn as nn
from typing import Optional, List

from .config import (
    DEFAULT_DECODER_MODEL_ID,
    DEFAULT_PATHS,
    DEFAULT_PERCEPTION_MODEL_ID,
    TCRFormerPaths,
    normalize_optional_path,
)
from .encoder import TemporalCausalEncoder
from .decoder import ReasoningDecoder


def parse_intervention_target(prompt: str, text_chunks: List[List[str]]) -> Optional[int]:
    """
    Resolve the target event index for a counterfactual intervention.

    Supported patterns:
      1. Ordinal references such as ``first`` or ``#1``.
      2. Event descriptions following ``the event ...``.
      3. Counterfactual clauses such as ``If X had not Y ...``.
    """
    if not prompt or not text_chunks or len(text_chunks) == 0:
        return None

    ordinal_map = {
        "first": 0, "second": 1, "third": 2, "fourth": 3, "fifth": 4,
        "sixth": 5, "seventh": 6, "eighth": 7, "ninth": 8, "tenth": 9,
        "1st": 0, "2nd": 1, "3rd": 2, "4th": 3, "5th": 4,
        "6th": 5, "7th": 6, "8th": 7, "9th": 8, "10th": 9,
    }
    hash_match = re.search(r'#\s*(\d+)', prompt, re.IGNORECASE)
    if hash_match:
        idx = int(hash_match.group(1)) - 1
        if 0 <= idx < len(text_chunks[0]):
            return idx

    ord_match = re.search(
        r'\b(the\s+)?(' + '|'.join(ordinal_map.keys()) + r')\s+(event|node|intervention)',
        prompt, re.IGNORECASE
    )
    if ord_match:
        key = ord_match.group(2).lower()
        idx = ordinal_map.get(key)
        if idx is not None and idx < len(text_chunks[0]):
            return idx

    index_match = re.search(r'event\s+(?:at\s+)?index\s+(\d+)', prompt, re.IGNORECASE)
    if index_match:
        idx = int(index_match.group(1)) - 1
        if 0 <= idx < len(text_chunks[0]):
            return idx

    description = None
    desc_match = re.search(
        r'the\s+event\s+(?:about|concerning|of|that)?\s*["“]?(.+?)["”]?(?:\s*,|\.|\s*$|\s*which|\s*is)',
        prompt, re.IGNORECASE
    )
    if desc_match:
        description = desc_match.group(1).strip()
        
    if not description:
        if_match = re.search(r'If\s+(.*?)\s+(?:had\s+not|did\s+not|were\s+not)\s+(.*?)(?:,|\s+given\s+that)', prompt, re.IGNORECASE)
        if if_match:
            description = f"{if_match.group(1)} {if_match.group(2)}".strip()

    if description:
        best_idx = None
        best_score = 0
        for i, chunk in enumerate(text_chunks[0]):
            first_sent = chunk.split('.')[0] if '.' in chunk else chunk
            desc_words = set(description.lower().split())
            chunk_words = set(first_sent.lower().split())
            common = desc_words.intersection(chunk_words)
            score = len(common)
            if score > best_score:
                best_score = score
                best_idx = i
        if best_score > 1:
            return best_idx

    return None


class TCRFormer(nn.Module):
    """
    TCR-Former with a Temporal Causal Encoder and a Reasoning Decoder.
    """
    def __init__(
        self,
        perception_model_id=DEFAULT_PERCEPTION_MODEL_ID,
        decoder_model_id=DEFAULT_DECODER_MODEL_ID,
        max_event_anchors: int = 128,
        encoder_ckpt: Optional[str] = None,
        projector_ckpt: Optional[str] = None,
        decoder_lora_dir: Optional[str] = None,
        paths: Optional[TCRFormerPaths] = None,
        **kwargs
    ):
        super().__init__()

        paths = paths or DEFAULT_PATHS
        encoder_ckpt = encoder_ckpt or paths.encoder_checkpoint
        projector_ckpt = projector_ckpt or paths.projector_checkpoint
        decoder_lora_dir = decoder_lora_dir or paths.decoder_lora_dir
        
        self.unified_encoder = TemporalCausalEncoder(
            model_id=perception_model_id,
            max_event_anchors=max_event_anchors
        )
        encoder_dim = self.unified_encoder.hidden_dim
        
        self.reasoning_layer = ReasoningDecoder(
            model_id=decoder_model_id,
            encoder_dim=encoder_dim,
            **kwargs
        )
        
        self._load_checkpoints(encoder_ckpt, projector_ckpt, decoder_lora_dir)

    def _load_checkpoints(self, encoder_ckpt, projector_ckpt, decoder_lora_dir):
        """Load optional checkpoints and LoRA weights from user-provided paths."""
        encoder_ckpt = normalize_optional_path(encoder_ckpt)
        projector_ckpt = normalize_optional_path(projector_ckpt)
        decoder_lora_dir = normalize_optional_path(decoder_lora_dir)

        if encoder_ckpt and encoder_ckpt.exists():
            print(f"Loading encoder checkpoint from: {encoder_ckpt}")
            state_dict1 = torch.load(encoder_ckpt, map_location="cpu")
            self.load_state_dict(state_dict1, strict=False)
            
        if projector_ckpt and projector_ckpt.exists():
            print(f"Loading projector checkpoint from: {projector_ckpt}")
            state_dict2 = torch.load(projector_ckpt, map_location="cpu")
            self.load_state_dict(state_dict2, strict=False)
            
        if decoder_lora_dir and decoder_lora_dir.exists():
            print(f"Loading decoder LoRA weights from: {decoder_lora_dir}")
            try:
                from peft import PeftModel
                self.reasoning_layer.llm = PeftModel.from_pretrained(
                    self.reasoning_layer.llm, 
                    str(decoder_lora_dir)
                )
            except ImportError:
                print("`peft` is required to load decoder LoRA weights. Install it with `pip install peft`.")
            except Exception as e:
                print(f"Failed to load decoder LoRA weights: {e}")

            
    def forward(
        self, 
        text_chunks: list[list[str]], 
        prompt_texts: list[str],
        intervention_node_id: Optional[int] = None
    ):
        """
        Forward pass.
        """
        z_causal, pred_timestamps, _ = self.unified_encoder(
            text_chunks, intervention_node_id=intervention_node_id
        )
        
        outputs = self.reasoning_layer(
            z_causal=z_causal, 
            prompt_texts=prompt_texts,
            text_chunks=text_chunks
        )
        
        return outputs, pred_timestamps

    def generate(
        self, 
        text_chunks: list[list[str]], 
        prompt_texts: list[str],
        intervention_node_id: Optional[int] = None,
        auto_parse: bool = True,
        **kwargs
    ):
        """
        Generation API with optional automatic counterfactual intervention parsing.
        """
        if auto_parse and intervention_node_id is None:
            parsed_id = parse_intervention_target(prompt_texts[0], text_chunks)
            if parsed_id is not None:
                print(f"[Auto Parse] Counterfactual intervention target index: {parsed_id}")
            intervention_node_id = parsed_id

        z_causal, pred_timestamps, _ = self.unified_encoder(
            text_chunks, intervention_node_id=intervention_node_id
        )
        
        generated_texts = self.reasoning_layer.generate(
            z_causal=z_causal, 
            prompt_texts=prompt_texts,
            text_chunks=text_chunks,
            **kwargs
        )
        
        return generated_texts, pred_timestamps

    def forward_phase1_training(self, text_chunks, intervention_node_id: int):
        """Two-route forward path used by the phase-1 training objective."""
        z_factual, pred_timestamps, v_tokens = self.unified_encoder(
            text_chunks, intervention_node_id=None
        )
        
        z_cf, _, _ = self.unified_encoder(
            text_chunks, intervention_node_id=intervention_node_id
        )
        
        return z_factual, z_cf, pred_timestamps, v_tokens
