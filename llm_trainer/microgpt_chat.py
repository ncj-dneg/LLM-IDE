from __future__ import annotations

from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any, Callable, Optional

import torch
import torch.nn.functional as F

from llm_trainer.config import ModelConfig
from llm_trainer.model import MicroGPT
from llm_trainer.tokenizer import EOS_TOKEN, load_tokenizer, token_id


class MicroGPTChatSession:
    """Persistent chat session backed by a native MicroGPT checkpoint."""

    def __init__(self, model_path: Path, device: str = "auto") -> None:
        """Load a native MicroGPT checkpoint for repeated prompts.

        Args:
            model_path: Model folder or checkpoint path.
            device: Device selector: auto, cuda, or cpu.

        Raises:
            FileNotFoundError: If checkpoint or tokenizer files are missing.
            ValueError: If the checkpoint is not a MicroGPT checkpoint.
        """

        self.model_path = _resolve_model_checkpoint(model_path)
        self.model_dir = self.model_path.parent
        tokenizer_path = self.model_dir / "tokenizer.json"
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer not found beside checkpoint: {tokenizer_path}")
        requested_device = device.lower().strip()
        self.device = "cuda" if requested_device == "auto" and torch.cuda.is_available() else requested_device
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
        if self.device not in {"cuda", "cpu"}:
            self.device = "cpu"

        checkpoint = torch.load(self.model_path, map_location=self.device)
        config_data = checkpoint.get("model_config")
        state_dict = checkpoint.get("model_state_dict")
        if not isinstance(config_data, dict) or not state_dict:
            raise ValueError("Checkpoint must contain model_config and model_state_dict.")
        self.config = ModelConfig(**config_data)
        self.model = MicroGPT(self.config).to(self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.tokenizer = load_tokenizer(tokenizer_path)
        self.eos_id = token_id(self.tokenizer, EOS_TOKEN)
        self._lock = Lock()
        self._messages: list[dict[str, str]] = []

    @property
    def runtime_summary(self) -> str:
        """Return a short runtime summary.

        Returns:
            Runtime summary text.
        """

        return (
            f"Runtime: native MicroGPT on {self.device.upper()} | "
            f"{self.config.layer_count} layers, {self.config.embedding_size} hidden, ctx {self.config.context_length}"
        )

    def reset(self) -> None:
        """Clear conversation history while keeping the model loaded."""

        with self._lock:
            self._messages = []

    def generate_stream(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repeat_penalty: float = 1.1,
        reasoning_effort: str = "Balanced",
        thinking_enabled: bool = True,
        progress: Optional[Callable[[Any], None]] = None,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> dict[str, Any]:
        """Stream one assistant reply and report timing metrics.

        Args:
            prompt: User message.
            system_prompt: Optional system instruction.
            max_tokens: Maximum new tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling value.
            repeat_penalty: Penalty for generated token repetition.
            reasoning_effort: Effort mode label.
            thinking_enabled: Whether to add reasoning guidance.
            progress: Optional progress callback.
            should_stop: Optional cancellation callback.

        Returns:
            Reply text and timing metrics.
        """

        started_at = perf_counter()
        reply_parts: list[str] = []
        generated_ids: list[int] = []
        with self._lock, torch.no_grad():
            prompt_text = self._render_prompt(prompt, system_prompt, reasoning_effort, thinking_enabled)
            input_ids = self.tokenizer.encode(prompt_text).ids[-self.config.context_length :]
            ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
            for _ in range(max_tokens):
                if should_stop and should_stop():
                    break
                idx_cond = ids[:, -self.config.context_length :]
                logits = self.model(idx_cond)[:, -1, :]
                logits = self._apply_repeat_penalty(logits, generated_ids, repeat_penalty)
                next_id = self._sample_next_token(logits, temperature, top_p)
                if next_id == self.eos_id and generated_ids:
                    break
                ids = torch.cat((ids, torch.tensor([[next_id]], dtype=torch.long, device=self.device)), dim=1)
                generated_ids.append(next_id)
                piece = self.tokenizer.decode([next_id], skip_special_tokens=True)
                if not piece:
                    continue
                reply_parts.append(piece)
                elapsed = max(perf_counter() - started_at, 0.001)
                if progress:
                    progress(
                        {
                            "type": "chat_delta",
                            "content": piece,
                            "elapsed_seconds": elapsed,
                            "token_count": len(generated_ids),
                            "tokens_per_second": len(generated_ids) / elapsed,
                        }
                    )

            reply = "".join(reply_parts).strip()
            if reply:
                self._messages.append({"role": "user", "content": prompt})
                self._messages.append({"role": "assistant", "content": reply})
        elapsed = max(perf_counter() - started_at, 0.001)
        return {
            "reply": reply,
            "elapsed_seconds": elapsed,
            "token_count": len(generated_ids),
            "tokens_per_second": len(generated_ids) / elapsed if generated_ids else 0.0,
            "stopped": bool(should_stop and should_stop()),
        }

    def _render_prompt(self, prompt: str, system_prompt: str, reasoning_effort: str, thinking_enabled: bool) -> str:
        """Render chat history into plain text for MicroGPT.

        Args:
            prompt: Latest user message.
            system_prompt: Optional system instruction.
            reasoning_effort: Effort mode label.
            thinking_enabled: Whether reasoning guidance is enabled.

        Returns:
            Prompt text.
        """

        instruction = self._effort_instruction(reasoning_effort) if thinking_enabled else self._plain_instruction()
        parts = []
        if system_prompt.strip() or instruction:
            parts.append(f"System: {' '.join(part for part in (system_prompt.strip(), instruction) if part)}")
        for message in self._messages[-12:]:
            role = "User" if message["role"] == "user" else "Assistant"
            parts.append(f"{role}: {message['content']}")
        parts.append(f"User: {prompt}")
        parts.append("Assistant:")
        return "\n".join(parts)

    def _apply_repeat_penalty(self, logits: torch.Tensor, generated_ids: list[int], repeat_penalty: float) -> torch.Tensor:
        """Apply a simple repeat penalty to recently generated tokens."""

        if repeat_penalty <= 1.0 or not generated_ids:
            return logits
        for token in set(generated_ids[-128:]):
            logits[:, token] = logits[:, token] / repeat_penalty
        return logits

    def _sample_next_token(self, logits: torch.Tensor, temperature: float, top_p: float) -> int:
        """Sample the next token from logits."""

        temperature = max(float(temperature), 1e-5)
        logits = logits / temperature
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            probs = F.softmax(sorted_logits, dim=-1)
            cumulative = torch.cumsum(probs, dim=-1)
            remove = cumulative > max(0.01, min(1.0, float(top_p)))
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = False
            sorted_logits = sorted_logits.masked_fill(remove, -float("inf"))
            filtered = torch.full_like(logits, -float("inf"))
            filtered.scatter_(1, sorted_indices, sorted_logits)
            logits = filtered
        probs = F.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item())

    @staticmethod
    def _effort_instruction(reasoning_effort: str) -> str:
        """Translate effort label into prompt guidance."""

        if reasoning_effort == "Fast":
            return "Answer concisely. Put code inside fenced Markdown code blocks with language labels."
        if reasoning_effort == "Deep":
            return "Think carefully and provide a detailed answer when useful. Put code inside fenced Markdown code blocks with language labels."
        return "Use balanced reasoning and answer clearly. Put code inside fenced Markdown code blocks with language labels."

    @staticmethod
    def _plain_instruction() -> str:
        """Return direct answer guidance."""

        return "Answer directly. Put code inside fenced Markdown code blocks with language labels."


def _resolve_model_checkpoint(path: Path) -> Path:
    """Resolve a model folder or checkpoint path to a checkpoint file."""

    path = Path(path)
    if path.is_dir():
        final_model = path / "final_model.pt"
        if final_model.exists():
            return final_model
        checkpoints = sorted((path / "checkpoints").glob("checkpoint_*.pt"), key=lambda item: item.stat().st_mtime, reverse=True)
        if checkpoints:
            return checkpoints[0]
    if path.exists() and path.suffix == ".pt":
        return path
    raise FileNotFoundError(f"MicroGPT checkpoint not found: {path}")


def load_microgpt_chat_session(model_path: Path, device: str = "auto") -> MicroGPTChatSession:
    """Load a native MicroGPT chat session.

    Args:
        model_path: Model folder or checkpoint path.
        device: Device selector.

    Returns:
        Loaded MicroGPT chat session.
    """

    return MicroGPTChatSession(model_path, device=device)


def stream_microgpt_chat_reply(
    session: MicroGPTChatSession,
    prompt: str,
    system_prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    repeat_penalty: float,
    reasoning_effort: str,
    thinking_enabled: bool = True,
    progress: Optional[Callable[[Any], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> dict[str, Any]:
    """Stream a reply from a native MicroGPT chat session."""

    return session.generate_stream(
        prompt,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        repeat_penalty=repeat_penalty,
        reasoning_effort=reasoning_effort,
        thinking_enabled=thinking_enabled,
        progress=progress,
        should_stop=should_stop,
    )
