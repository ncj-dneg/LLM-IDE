from __future__ import annotations

from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any, Callable, Optional


class LlamaChatSession:
    """Persistent GGUF chat session backed by llama-cpp-python."""

    def __init__(self, model_path: Path, n_ctx: int = 2048, n_threads: int = 4, n_gpu_layers: int = -1) -> None:
        """Load a GGUF model once for repeated chat prompts.

        Args:
            model_path: Path to a GGUF model file.
            n_ctx: Context window used by llama.cpp.
            n_threads: CPU thread count.
            n_gpu_layers: Number of layers to offload to GPU.

        Raises:
            ImportError: If llama-cpp-python is not installed.
            FileNotFoundError: If the model path does not exist.
        """

        if not Path(model_path).exists():
            raise FileNotFoundError(f"GGUF model not found: {model_path}")
        try:
            import llama_cpp
            from llama_cpp import Llama
        except ImportError as exc:
            raise ImportError("Install llama-cpp-python to load GGUF models.") from exc

        self.gpu_offload_supported = bool(
            getattr(llama_cpp, "llama_supports_gpu_offload", lambda: False)()
        )
        self.requested_gpu_layers = n_gpu_layers
        if n_gpu_layers != 0 and not self.gpu_offload_supported:
            raise RuntimeError(
                "This llama-cpp-python install is CPU-only. GPU layers were requested, "
                "but llama.cpp reports GPU offload support is unavailable. Reinstall "
                "llama-cpp-python with CUDA, Metal, Vulkan, or another GPU backend, "
                "or set GPU layers to 0 for CPU loading."
            )

        self.model_path = Path(model_path)
        self._lock = Lock()
        self._messages: list[dict[str, str]] = []
        try:
            self._llm = Llama(
                model_path=str(self.model_path),
                n_ctx=n_ctx,
                n_threads=n_threads,
                n_gpu_layers=n_gpu_layers,
                offload_kqv=n_gpu_layers != 0,
                verbose=False,
            )
        except ValueError as exc:
            file_size = self.model_path.stat().st_size if self.model_path.exists() else 0
            hints = []
            if file_size == 0:
                hints.append("The file is empty — the download may have failed.")
            elif file_size < 1_000_000:
                hints.append(f"The file is very small ({file_size:,} bytes) — it may be incomplete or corrupted.")
            if any(c > 127 for c in str(self.model_path).encode("utf-8", errors="replace")):
                hints.append("The path contains non-ASCII characters — try moving the model to a simple path.")
            detail = " ".join(hints) if hints else "Verify the file is a valid GGUF model and is fully downloaded."
            raise ValueError(
                f"Failed to load GGUF model: {self.model_path}\n{detail}"
            ) from exc

    @property
    def runtime_summary(self) -> str:
        """Return a short runtime summary.

        Returns:
            Runtime summary text.
        """

        if self.requested_gpu_layers == 0:
            return "Runtime: CPU"
        if self.gpu_offload_supported:
            return f"Runtime: GPU offload requested ({self.requested_gpu_layers} layers)"
        return "Runtime: CPU-only llama.cpp build"

    def reset(self) -> None:
        """Clear conversation history while keeping the model loaded."""

        with self._lock:
            self._messages = []

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repeat_penalty: float = 1.1,
        reasoning_effort: str = "Balanced",
        thinking_enabled: bool = True,
    ) -> str:
        """Generate one assistant reply.

        Args:
            prompt: User message.
            system_prompt: Optional system instruction.
            max_tokens: Maximum new tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling value.
            repeat_penalty: Repetition penalty.
            reasoning_effort: User-facing effort mode.
            thinking_enabled: Whether to add reasoning-style system guidance.

        Returns:
            Assistant reply text.
        """

        effort_instruction = self._effort_instruction(reasoning_effort) if thinking_enabled else self._plain_instruction()
        messages = []
        if system_prompt.strip() or effort_instruction:
            messages.append({"role": "system", "content": "\n".join(part for part in (system_prompt.strip(), effort_instruction) if part)})

        with self._lock:
            messages.extend(self._messages)
            messages.append({"role": "user", "content": prompt})
            response = self._llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repeat_penalty=repeat_penalty,
            )
            reply = response["choices"][0]["message"]["content"].strip()
            self._messages.append({"role": "user", "content": prompt})
            self._messages.append({"role": "assistant", "content": reply})
        return reply

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
            repeat_penalty: Repetition penalty.
            reasoning_effort: User-facing effort mode.
            thinking_enabled: Whether to add reasoning-style system guidance.
            progress: Optional callback receiving stream events.
            should_stop: Optional callback returning true when generation should stop.

        Returns:
            Reply text and generation metrics.
        """

        effort_instruction = self._effort_instruction(reasoning_effort) if thinking_enabled else self._plain_instruction()
        messages = []
        if system_prompt.strip() or effort_instruction:
            messages.append({"role": "system", "content": "\n".join(part for part in (system_prompt.strip(), effort_instruction) if part)})

        started_at = perf_counter()
        reply_parts: list[str] = []
        chunk_count = 0
        with self._lock:
            messages.extend(self._messages)
            messages.append({"role": "user", "content": prompt})
            stream = self._llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repeat_penalty=repeat_penalty,
                stream=True,
            )
            for chunk in stream:
                if should_stop and should_stop():
                    break
                delta = chunk["choices"][0].get("delta", {}).get("content", "")
                if not delta:
                    continue
                reply_parts.append(delta)
                chunk_count += 1
                elapsed = max(perf_counter() - started_at, 0.001)
                streamed_text = "".join(reply_parts).strip()
                token_count = self._count_tokens(streamed_text) if streamed_text else 0
                if progress:
                    progress(
                        {
                            "type": "chat_delta",
                            "content": delta,
                            "elapsed_seconds": elapsed,
                            "chunk_count": chunk_count,
                            "token_count": token_count,
                            "tokens_per_second": token_count / elapsed if token_count else 0.0,
                        }
                    )

            reply = "".join(reply_parts).strip()
            token_count = self._count_tokens(reply) if reply else 0
            elapsed = max(perf_counter() - started_at, 0.001)
            if reply:
                self._messages.append({"role": "user", "content": prompt})
                self._messages.append({"role": "assistant", "content": reply})

        return {
            "reply": reply,
            "elapsed_seconds": elapsed,
            "token_count": token_count,
            "tokens_per_second": token_count / elapsed if token_count else 0.0,
            "stopped": bool(should_stop and should_stop()),
        }

    def _count_tokens(self, text: str) -> int:
        """Count generated tokens using the loaded llama tokenizer.

        Args:
            text: Generated text.

        Returns:
            Token count.
        """

        try:
            return len(self._llm.tokenize(text.encode("utf-8")))
        except Exception:
            return max(1, len(text.split()))

    @staticmethod
    def _effort_instruction(reasoning_effort: str) -> str:
        """Translate a UI effort label into a system instruction.

        Args:
            reasoning_effort: Selected effort label.

        Returns:
            Instruction text.
        """

        if reasoning_effort == "Fast":
            return "Answer concisely and prioritize speed. Put code inside fenced Markdown code blocks with language labels."
        if reasoning_effort == "Deep":
            return "Think carefully, reason through the problem, and provide a detailed answer when useful. Put code inside fenced Markdown code blocks with language labels."
        return "Use balanced reasoning and answer clearly. Put code inside fenced Markdown code blocks with language labels."

    @staticmethod
    def _plain_instruction() -> str:
        """Return the non-thinking chat formatting instruction.

        Returns:
            Plain response instruction.
        """

        return "Answer directly. Put code inside fenced Markdown code blocks with language labels."


def load_llama_chat_session(model_path: Path, n_ctx: int, n_threads: int, n_gpu_layers: int) -> LlamaChatSession:
    """Load a GGUF chat session.

    Args:
        model_path: Path to a GGUF model file.
        n_ctx: Context window.
        n_threads: CPU thread count.
        n_gpu_layers: GPU offload layer count.

    Returns:
        Loaded chat session.
    """

    return LlamaChatSession(model_path, n_ctx=n_ctx, n_threads=n_threads, n_gpu_layers=n_gpu_layers)


def generate_chat_reply(
    session: LlamaChatSession,
    prompt: str,
    system_prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    repeat_penalty: float,
    reasoning_effort: str,
    thinking_enabled: bool = True,
) -> str:
    """Generate a reply from a loaded chat session.

    Args:
        session: Loaded llama.cpp chat session.
        prompt: User message.
        system_prompt: Optional system instruction.
        max_tokens: Maximum new tokens.
        temperature: Sampling temperature.
        top_p: Nucleus sampling value.
        repeat_penalty: Repetition penalty.
        reasoning_effort: Effort mode label.
        thinking_enabled: Whether reasoning-style guidance is enabled.

    Returns:
        Assistant reply.
    """

    return session.generate(
        prompt,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        repeat_penalty=repeat_penalty,
        reasoning_effort=reasoning_effort,
        thinking_enabled=thinking_enabled,
    )


def stream_chat_reply(
    session: LlamaChatSession,
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
    """Stream a reply from a loaded chat session.

    Args:
        session: Loaded llama.cpp chat session.
        prompt: User message.
        system_prompt: Optional system instruction.
        max_tokens: Maximum new tokens.
        temperature: Sampling temperature.
        top_p: Nucleus sampling value.
        repeat_penalty: Repetition penalty.
        reasoning_effort: Effort mode label.
        thinking_enabled: Whether reasoning-style guidance is enabled.
        progress: Optional stream event callback.
        should_stop: Optional cancellation callback.

    Returns:
        Reply text and timing metrics.
    """

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
