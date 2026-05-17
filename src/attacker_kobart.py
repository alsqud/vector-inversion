from pathlib import Path

import torch
import torch.nn as nn
from tqdm.auto import tqdm
from transformers import AutoTokenizer, BartForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput


class EmbeddingToKoBART(nn.Module):
    """
    Embedding-to-text attacker based on KoBART.

    The input sentence embedding is projected into a continuous prefix.
    The prefix is then used as encoder outputs for KoBART generation.
    """

    def __init__(self, bart_model, embedding_dim, prefix_len=16, tokenizer=None):
        super().__init__()

        self.bart = bart_model
        self.prefix_len = int(prefix_len)
        self.d_model = bart_model.config.d_model
        self.tokenizer = tokenizer

        self.projector = nn.Sequential(
            nn.Linear(embedding_dim, self.d_model * 2),
            nn.GELU(),
            nn.LayerNorm(self.d_model * 2),
            nn.Linear(self.d_model * 2, self.prefix_len * self.d_model),
        )

    def make_encoder_outputs(self, sentence_embeddings):
        batch_size = sentence_embeddings.size(0)

        memory = self.projector(sentence_embeddings)
        memory = memory.view(batch_size, self.prefix_len, self.d_model)

        return memory

    def forward(self, sentence_embeddings, labels=None):
        memory = self.make_encoder_outputs(sentence_embeddings)

        encoder_outputs = BaseModelOutput(
            last_hidden_state=memory,
        )

        encoder_attention_mask = torch.ones(
            memory.size()[:2],
            dtype=torch.long,
            device=memory.device,
        )

        return self.bart(
            encoder_outputs=encoder_outputs,
            attention_mask=encoder_attention_mask,
            labels=labels,
        )

    @torch.no_grad()
    def generate_text(
        self,
        sentence_embeddings,
        max_len=96,
        num_beams=4,
        no_repeat_ngram_size=3,
        repetition_penalty=1.15,
        length_penalty=1.0,
        early_stopping=True,
    ):
        self.eval()

        memory = self.make_encoder_outputs(sentence_embeddings)

        encoder_outputs = BaseModelOutput(
            last_hidden_state=memory,
        )

        encoder_attention_mask = torch.ones(
            memory.size()[:2],
            dtype=torch.long,
            device=memory.device,
        )

        generated_ids = self.bart.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=encoder_attention_mask,
            max_length=int(max_len),
            num_beams=int(num_beams),
            no_repeat_ngram_size=int(no_repeat_ngram_size),
            repetition_penalty=float(repetition_penalty),
            length_penalty=float(length_penalty),
            early_stopping=bool(early_stopping),
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        return generated_ids


def load_attacker(checkpoint_path, device="cuda"):
    """
    Load the trained KoBART-based inversion attacker checkpoint.
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(
            str(checkpoint_path),
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(
            str(checkpoint_path),
            map_location=device,
        )

    config = checkpoint.get("config", {})

    generator_model_name = config.get("gen_model", "gogamza/kobart-base-v2")
    max_len = int(config.get("max_len", 96))
    prefix_len = int(config.get("prefix_len", 16))
    embedding_dim = int(config.get("embedding_dim", 768))

    tokenizer = AutoTokenizer.from_pretrained(generator_model_name)
    bart_model = BartForConditionalGeneration.from_pretrained(generator_model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bart_model.config.pad_token_id = tokenizer.pad_token_id

    if bart_model.config.decoder_start_token_id is None:
        bart_model.config.decoder_start_token_id = tokenizer.bos_token_id

    bart_model.config.eos_token_id = tokenizer.eos_token_id

    attacker = EmbeddingToKoBART(
        bart_model=bart_model,
        embedding_dim=embedding_dim,
        prefix_len=prefix_len,
        tokenizer=tokenizer,
    )

    attacker.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    attacker.to(device)
    attacker.eval()

    return attacker, tokenizer, max_len, prefix_len, config


@torch.no_grad()
def generate_predictions_from_embeddings(
    text_embeddings,
    attacker,
    tokenizer,
    device="cuda",
    max_len=96,
    batch_size=32,
):
    """
    Generate reconstructed text from input embeddings.
    """
    predictions = []
    num_items = len(text_embeddings)

    for start in tqdm(range(0, num_items, batch_size), desc="Generate inversion"):
        end = min(start + batch_size, num_items)

        batch_embeddings = torch.tensor(
            text_embeddings[start:end],
            dtype=torch.float32,
            device=device,
        )

        generated_ids = attacker.generate_text(
            sentence_embeddings=batch_embeddings,
            max_len=max_len,
            num_beams=4,
        )

        batch_predictions = tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        predictions.extend([text.strip() for text in batch_predictions])

    return predictions