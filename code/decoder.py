import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from modelscope import snapshot_download

from .config import DEFAULT_DECODER_MODEL_ID

class ReasoningDecoder(nn.Module):
    """
    Reasoning Decoder.

    Aligns encoder-side causal latent variables with user prompts and performs
    autoregressive generation through Fact-Logic Bypass Fusion.
    """
    def __init__(self, model_id=DEFAULT_DECODER_MODEL_ID, encoder_dim=896, max_prompt_tokens=256, max_text_tokens=256):
        super().__init__()
        self.max_prompt_tokens = max_prompt_tokens
        self.max_text_tokens = max_text_tokens
        print(f"Loading Reasoning Decoder backbone: {model_id}")
        try:
            model_dir = snapshot_download(model_id)
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
            device_map = self._build_device_map(num_hidden_layers=36)
            llm_compute_dtype = torch.float16
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True
            )
            self.llm = AutoModelForCausalLM.from_pretrained(
                model_dir, 
                trust_remote_code=True,
                dtype=llm_compute_dtype,
                quantization_config=quantization_config,
                device_map=device_map,
                low_cpu_mem_usage=True
            )
        except Exception as e:
            print(f"Failed to load the decoder backbone: {e}")
            raise e

        self.llm.gradient_checkpointing_enable()
        self.llm.config.use_cache = False
        if hasattr(self.llm, "enable_input_require_grads"):
            self.llm.enable_input_require_grads()
            
        self.decoder_dim = self.llm.config.hidden_size
        self.llm_input_device = self._get_input_device()
        self.llm_compute_dtype = llm_compute_dtype
        self.projector_dtype = torch.float32
        
        # Projector P aligns causal latent variables with the decoder space.
        self.projector = nn.Sequential(
            nn.Linear(encoder_dim, self.decoder_dim),
            nn.GELU(),
            nn.Linear(self.decoder_dim, self.decoder_dim)
        ).to(self.llm_input_device, dtype=self.projector_dtype)

    @staticmethod
    def _build_device_map(num_hidden_layers: int) -> dict:
        device_map = {
            "model.embed_tokens": 2,
            "model.rotary_emb": 2,
            "model.norm": 3,
            "lm_head": 3
        }

        for layer_idx in range(num_hidden_layers):
            if layer_idx < 8:
                device_map[f"model.layers.{layer_idx}"] = 2
            elif layer_idx < 16:
                device_map[f"model.layers.{layer_idx}"] = 3
            elif layer_idx < 24:
                device_map[f"model.layers.{layer_idx}"] = 1
            elif layer_idx < 30:
                device_map[f"model.layers.{layer_idx}"] = 0
            else:
                device_map[f"model.layers.{layer_idx}"] = 3

        return device_map

    def _get_input_device(self):
        return self.llm.get_input_embeddings().weight.device

    def _get_projector_device(self):
        return next(self.projector.parameters()).device

    def _get_projector_dtype(self):
        return next(self.projector.parameters()).dtype

    def _create_mixed_attention_mask(
        self, 
        Lv: int, Lz: int, Lp: int, 
        dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        """
        - ``V_text`` follows a causal mask.
        - ``Z_aligned`` is globally visible.
        - ``E_prompt`` follows autoregressive masking while attending to
          all ``V_text`` and ``Z_aligned`` tokens.
        """
        total = Lv + Lz + Lp
        min_dtype = torch.finfo(dtype).min
        mask = torch.full((total, total), fill_value=min_dtype, dtype=dtype, device=device)

        for i in range(Lv):
            mask[i, :i+1] = 0.0

        if Lz > 0:
            mask[Lv:Lv+Lz, :] = 0.0

        for i_idx in range(Lp):
            i = Lv + Lz + i_idx
            mask[i, :Lv+Lz] = 0.0
            for j_idx in range(i_idx + 1):
                j = Lv + Lz + j_idx
                mask[i, j] = 0.0

        return mask.unsqueeze(0).unsqueeze(0)

    def forward(
        self, 
        z_causal: torch.Tensor, 
        prompt_texts: list[str],
        text_chunks: list[list[str]],
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass for training or evaluation with the mixed attention mask.
        """
        batch_size = z_causal.shape[0]
        if batch_size > 1:
            raise NotImplementedError("forward with batch_size > 1 not implemented for mixed mask. "
                                      "Please batch manually or use generate().")

        projector_device = self._get_projector_device()
        projector_dtype = self._get_projector_dtype()
        z_causal = z_causal.to(projector_device, dtype=projector_dtype)

        z_aligned = self.projector(z_causal).to(self.llm_input_device, dtype=self.llm_compute_dtype)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        joined_texts = ["\n".join(chunks) for chunks in text_chunks]
        text_inputs = self.tokenizer(
            joined_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_text_tokens
        )
        text_input_ids = text_inputs["input_ids"].to(self.llm_input_device)
        text_embeds = self.llm.get_input_embeddings()(text_input_ids)
        Lv = text_embeds.shape[1]

        inputs = self.tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_prompt_tokens
        )
        input_ids = inputs["input_ids"].to(self.llm_input_device)
        prompt_embeds = self.llm.get_input_embeddings()(input_ids)
        Lp = prompt_embeds.shape[1]

        Lz = z_aligned.shape[1]

        inputs_embeds = torch.cat([text_embeds, z_aligned, prompt_embeds], dim=1)

        attn_mask_4d = self._create_mixed_attention_mask(
            Lv=Lv, Lz=Lz, Lp=Lp,
            dtype=self.llm_compute_dtype,
            device=self.llm_input_device
        )

        outputs = self.llm(inputs_embeds=inputs_embeds, attention_mask=attn_mask_4d)
        return outputs

    def generate(self, z_causal: torch.Tensor, prompt_texts: list[str], text_chunks: list[list[str]], **kwargs):
        """
        End-to-end generation with the mixed autoregressive attention scheme.
        """
        projector_device = self._get_projector_device()
        projector_dtype = self._get_projector_dtype()
        z_causal = z_causal.to(projector_device, dtype=projector_dtype)
        
        z_aligned = self.projector(z_causal).to(self.llm_input_device, dtype=self.llm_compute_dtype)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        joined_texts = ["\n".join(chunks) for chunks in text_chunks]
        text_inputs = self.tokenizer(
            joined_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_text_tokens
        )
        text_input_ids = text_inputs["input_ids"].to(self.llm_input_device)
        text_embeds = self.llm.get_input_embeddings()(text_input_ids)
        Lv = text_embeds.shape[1]

        inputs = self.tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_prompt_tokens
        )
        input_ids = inputs["input_ids"].to(self.llm_input_device)
        prompt_embeds = self.llm.get_input_embeddings()(input_ids)
        Lp = prompt_embeds.shape[1]

        Lz = z_aligned.shape[1]

        init_embeds = torch.cat([text_embeds, z_aligned, prompt_embeds], dim=1)

        max_new_tokens = kwargs.pop("max_new_tokens", 256)
        temperature = kwargs.pop("temperature", 1.0)
        do_sample = kwargs.pop("do_sample", temperature > 0)
        top_k = kwargs.pop("top_k", 0)
        top_p = kwargs.pop("top_p", 1.0)
        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is None:
            eos_token_id = self.tokenizer.pad_token_id

        generated_token_ids = []
        current_embeds = init_embeds

        for step in range(max_new_tokens):
            Lv, Lz, Lp_and_gen = Lv, Lz, Lp + len(generated_token_ids)

            attn_mask_4d = self._create_mixed_attention_mask(
                Lv=Lv, Lz=Lz, Lp=Lp_and_gen,
                dtype=self.llm_compute_dtype,
                device=self.llm_input_device
            )

            outputs = self.llm(inputs_embeds=current_embeds, attention_mask=attn_mask_4d)
            logits = outputs.logits[:, -1, :]

            if temperature != 1.0:
                logits = logits / temperature

            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k, dim=-1)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_mask = cumulative_probs > top_p
                sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
                sorted_mask[..., 0] = False
                indices_to_remove = sorted_mask.scatter(1, sorted_indices, sorted_mask)
                logits[indices_to_remove] = float('-inf')

            if do_sample:
                probs = torch.softmax(logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)
            else:
                next_token_id = torch.argmax(logits, dim=-1, keepdim=True)

            if next_token_id.item() == eos_token_id:
                break

            generated_token_ids.append(next_token_id.item())

            next_token_embed = self.llm.get_input_embeddings()(next_token_id)
            current_embeds = torch.cat([current_embeds, next_token_embed], dim=1)

        generated_ids = torch.tensor([generated_token_ids], device=self.llm_input_device, dtype=torch.long)
        generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return generated_texts
