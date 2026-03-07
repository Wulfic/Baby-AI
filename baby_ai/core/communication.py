"""
Communication head — utterance generation from core hidden state.

A lightweight autoregressive sequence generator that produces
discrete token sequences representing the agent's "speech".
This is not an LLM — it's a small RNN decoder that learns to
communicate through interaction.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CommunicationHead(nn.Module):
    """
    Lightweight autoregressive utterance generator.

    Uses a small GRU decoder to generate sequences from the agent's
    internal state. The vocabulary is compact and learned from scratch.

    Args:
        input_dim: Core hidden state dimension (used as initial decoder state).
        vocab_size: Size of the learned token vocabulary.
        embed_dim: Token embedding dimension.
        hidden_dim: Decoder GRU hidden dimension.
        max_len: Maximum utterance length.
    """

    # Special tokens
    BOS = 0
    EOS = 1
    PAD = 2
    VOCAB_OFFSET = 3  # actual tokens start here

    def __init__(
        self,
        input_dim: int = 256,
        vocab_size: int = 4096,
        embed_dim: int = 64,
        hidden_dim: int = 256,
        max_len: int = 32,
    ):
        super().__init__()
        self.vocab_size = vocab_size + self.VOCAB_OFFSET  # account for special tokens
        self.max_len = max_len
        self.hidden_dim = hidden_dim

        self.token_embed = nn.Embedding(self.vocab_size, embed_dim, padding_idx=self.PAD)
        self.state_proj = nn.Linear(input_dim, hidden_dim)
        self.decoder_gru = nn.GRUCell(embed_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, self.vocab_size)

    def forward(
        self,
        state: torch.Tensor,
        target: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            state: (B, input_dim) core hidden state.
            target: (B, T) target token ids for teacher forcing (training).
                    If None, runs autoregressive generation (inference).

        Returns:
            If target given: (B, T, vocab_size) logits.
            If target None: (B, max_len) generated token ids.
        """
        if target is not None:
            return self._forward_teacher_forcing(state, target)
        else:
            return self._forward_generate(state)

    def _forward_teacher_forcing(
        self,
        state: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Teacher-forced forward pass for training."""
        B, T = target.shape
        h = self.state_proj(state)  # (B, hidden_dim)

        logits_seq = []
        for t in range(T):
            token_emb = self.token_embed(target[:, t])  # (B, embed_dim)
            h = self.decoder_gru(token_emb, h)
            logits = self.output_proj(h)  # (B, vocab_size)
            logits_seq.append(logits)

        return torch.stack(logits_seq, dim=1)  # (B, T, vocab_size)

    @torch.no_grad()
    def _forward_generate(self, state: torch.Tensor) -> torch.Tensor:
        """Autoregressive generation for inference."""
        B = state.size(0)
        device = state.device
        h = self.state_proj(state)

        token = torch.full((B,), self.BOS, dtype=torch.long, device=device)
        output = torch.full((B, self.max_len), self.PAD, dtype=torch.long, device=device)

        for t in range(self.max_len):
            token_emb = self.token_embed(token)
            h = self.decoder_gru(token_emb, h)
            logits = self.output_proj(h)

            # Greedy or temperature-based sampling
            token = logits.argmax(dim=-1)
            output[:, t] = token

            # Stop if all sequences have generated EOS
            if (token == self.EOS).all():
                break

        return output

    def get_logits(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get first-step logits (useful for distillation).

        Returns: (B, vocab_size) logits for the first generated token.
        """
        h = self.state_proj(state)
        bos = torch.full(
            (state.size(0),), self.BOS,
            dtype=torch.long, device=state.device,
        )
        token_emb = self.token_embed(bos)
        h = self.decoder_gru(token_emb, h)
        return self.output_proj(h)
