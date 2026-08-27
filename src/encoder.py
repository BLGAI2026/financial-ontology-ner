"""
Wraps XLM-RoBERTa (Eq. 2, 7: h = XLM-R(x)) behind a single interface
used by every component in Section 4.

If `use_pretrained` is True (default), weights are downloaded from the
Hugging Face Hub the first time this runs (requires internet access,
e.g. via `huggingface_hub`) — this is what reproduces the paper's
numbers. If no internet access is available (e.g. this sandbox, or a
CI runner), set `use_pretrained=False` and the encoder falls back to a
randomly-initialized XLM-R-large *config* (same architecture, random
weights) so the full pipeline still runs end-to-end for testing.
"""
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer


class MultilingualEncoder(nn.Module):
    def __init__(self, model_name="xlm-roberta-large", use_pretrained=True, dropout=0.1):
        super().__init__()
        self.model_name = model_name
        try:
            if use_pretrained:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.backbone = AutoModel.from_pretrained(model_name)
            else:
                raise OSError("use_pretrained=False: skipping download")
        except Exception as e:  # offline / no network / model not cached
            print(f"[MultilingualEncoder] Falling back to a randomly-initialized "
                  f"'{model_name}' architecture ({e.__class__.__name__}: {e}). "
                  f"Set use_pretrained=True with network access to reproduce "
                  f"paper results.")
            config = AutoConfig.from_pretrained(model_name) if _config_cached(model_name) \
                else _fallback_config()
            self.tokenizer = _fallback_tokenizer()
            self.backbone = AutoModel.from_config(config)
        self.backbone.config.hidden_dropout_prob = dropout
        self.hidden_size = self.backbone.config.hidden_size

    def forward(self, texts, device="cpu", max_length=64):
        enc = self.tokenizer(
            texts, padding=True, truncation=True, max_length=max_length,
            return_tensors="pt",
        ).to(device)
        out = self.backbone(**enc)
        token_embeddings = out.last_hidden_state          # [B, L, H]
        mask = enc["attention_mask"].unsqueeze(-1).float()  # [B, L, 1]
        pooled = (token_embeddings * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        return {
            "token_embeddings": token_embeddings,
            "pooled": pooled,
            "attention_mask": enc["attention_mask"],
            "input_ids": enc["input_ids"],
        }


def _config_cached(model_name):
    try:
        AutoConfig.from_pretrained(model_name, local_files_only=True)
        return True
    except Exception:
        return False


def _fallback_config():
    """A tiny XLM-R-style config used only when the pretrained checkpoint
    cannot be downloaded (e.g. offline CI / this sandbox). It exists so
    the pipeline is runnable end-to-end without network access; it is
    NOT a substitute for the paper's XLM-RoBERTa-large encoder and will
    not reproduce paper-scale numbers. Kept deliberately small (vocab,
    layers, hidden size) so a full forward+backward pass fits in a few
    hundred MB of RAM."""
    from transformers import XLMRobertaConfig
    return XLMRobertaConfig(
        vocab_size=4000, hidden_size=256, num_hidden_layers=2,
        num_attention_heads=4, intermediate_size=512,
        max_position_embeddings=130, type_vocab_size=1,
    )


def _fallback_tokenizer():
    """A minimal whitespace/byte-level tokenizer used only when the
    real XLM-R tokenizer files cannot be fetched. Not used when the
    pretrained checkpoint loads successfully."""
    from transformers import PreTrainedTokenizerFast
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers

    tok = Tokenizer(models.WordLevel(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.WordLevelTrainer(
        special_tokens=["<s>", "<pad>", "</s>", "<unk>"], vocab_size=4000
    )
    # Train on a tiny generic corpus just so the tokenizer is usable;
    # real vocabulary coverage comes from the pretrained checkpoint.
    seed_corpus = ["<s> hello world </s>", "<s> financial ontology entity </s>"]
    tok.train_from_iterator(seed_corpus, trainer=trainer)
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok, unk_token="<unk>", pad_token="<pad>",
        cls_token="<s>", sep_token="</s>",
    )
    return fast
