from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

from .config import ModelConfig, TrainingConfig, dataclass_to_jsonable
from .model import (
    MicroGPT,
    apply_lora_adapters,
    freeze_non_lora_parameters,
    load_lora_state_dict,
    lora_parameter_count,
    lora_state_dict,
    merge_lora_adapters,
)

try:
    import psutil
except ImportError:
    psutil = None


class Lion(torch.optim.Optimizer):
    """Lion optimizer with decoupled weight decay.

    The implementation follows the common Lion update rule and keeps the
    optimizer self-contained so the app does not require an extra dependency.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ) -> None:
        """Create a Lion optimizer.

        Args:
            params: Iterable of parameters to optimize.
            lr: Learning rate.
            betas: Momentum coefficients.
            weight_decay: Decoupled weight decay.
        """

        if lr <= 0.0:
            raise ValueError("lr must be greater than 0")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError("betas must be in [0, 1)")
        defaults = {"lr": lr, "betas": betas, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform one optimization step.

        Args:
            closure: Optional closure that reevaluates the model.

        Returns:
            Closure loss when a closure is provided.
        """

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                grad = parameter.grad
                if weight_decay:
                    parameter.mul_(1.0 - lr * weight_decay)
                state = self.state[parameter]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(parameter)
                exp_avg = state["exp_avg"]
                update = exp_avg.mul(beta1).add(grad, alpha=1.0 - beta1)
                parameter.add_(update.sign(), alpha=-lr)
                exp_avg.mul_(beta2).add_(grad, alpha=1.0 - beta2)
        return loss


class TokenDataset(Dataset):
    """Sliding-window token dataset for next-token prediction.

    Accepts a list of ints, a numpy array, or a numpy memmap.  When backed by a
    memmap the full token stream lives on disk — only individual windows are
    loaded into RAM on each ``__getitem__`` call, so datasets of any size can be
    used without exhausting system memory.
    """

    def __init__(self, tokens: Union[list[int], np.ndarray], context_length: int, stride: int = 1) -> None:
        """Create a token dataset.

        Args:
            tokens: Complete token stream (list, ndarray, or memmap).
            context_length: Number of input tokens per sample.
            stride: Token offset step between consecutive windows.

        Raises:
            ValueError: If there are not enough tokens.
        """

        if len(tokens) <= context_length:
            raise ValueError("Not enough tokens for the selected context length")
        if stride <= 0:
            raise ValueError("stride must be greater than 0")
        # Keep the backing store as-is (memmap stays on disk).
        if isinstance(tokens, np.ndarray):
            self._tokens_np: Optional[np.ndarray] = tokens
            self._tokens_tensor: Optional[torch.Tensor] = None
        else:
            self._tokens_np = None
            self._tokens_tensor = torch.tensor(tokens, dtype=torch.long)
        self.context_length = context_length
        self.stride = stride
        available_windows = len(tokens) - self.context_length
        self.sample_count = (available_windows + self.stride - 1) // self.stride

    def __len__(self) -> int:
        """Return the number of sliding windows available.

        Returns:
            Dataset length.
        """

        return self.sample_count

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one input/target token window.

        Args:
            index: Starting token index.

        Returns:
            Pair of input tokens and next-token targets.
        """

        start = index * self.stride
        end = start + self.context_length + 1
        if self._tokens_np is not None:
            # Read the slice from the numpy array / memmap and convert to tensor.
            chunk = torch.from_numpy(np.array(self._tokens_np[start:end], dtype=np.int64))
        else:
            chunk = self._tokens_tensor[start:end]  # type: ignore[index]
        return chunk[:-1], chunk[1:]


@dataclass
class TrainingResult:
    """Result returned after training.

    Attributes:
        checkpoint_path: Final model checkpoint path.
        summary_path: Training summary JSON path.
        final_train_loss: Final epoch training loss.
        final_val_loss: Final validation loss when available.
        stopped: Whether training was stopped by the user.
    """

    checkpoint_path: Path
    summary_path: Path
    final_train_loss: float
    final_val_loss: Optional[float]
    stopped: bool = False


@dataclass
class ResumeCompatibilityReport:
    """Compatibility result for a checkpoint resume attempt.

    Attributes:
        checkpoint_path: Checkpoint path that was inspected.
        errors: Blocking compatibility problems.
        warnings: Non-blocking but important differences.
        info: Informational compatibility details.
        can_load_optimizer_state: Whether optimizer state can be safely loaded.
        can_load_scheduler_state: Whether scheduler state can be safely loaded.
        can_load_scaler_state: Whether AMP scaler state can be safely loaded.
    """

    checkpoint_path: Path
    errors: list[str]
    warnings: list[str]
    info: list[str]
    can_load_optimizer_state: bool = True
    can_load_scheduler_state: bool = True
    can_load_scaler_state: bool = True


def emit_progress(
    progress: Optional[Callable[[Any], None]],
    message: str,
    percent: Optional[int] = None,
    **metrics: Any,
) -> None:
    """Emit training progress if a callback is available.

    Args:
        progress: Optional callback for progress dictionaries.
        message: Human-readable status message.
        percent: Optional progress percentage.
        **metrics: Optional structured metrics for UI dashboards.
    """

    if progress:
        progress({"message": message, "percent": percent, **metrics})


def set_seed(seed: int) -> None:
    """Set random seeds for repeatable training.

    Args:
        seed: Integer seed value.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_tokens(
    tokens: list[int],
    validation_split: float,
    chunk_size: int = 2048,
    seed: int = 1337,
) -> tuple[list[int], list[int]]:
    """Split tokens into train and validation streams.

    The corpus is written to disk as one big concatenation of source
    documents, then tokenized into a single flat stream. A plain positional
    split would make validation depend on whichever source happened to be at
    the tail of the corpus. This chunks and deterministically shuffles the
    stream first, so validation samples are drawn from across the corpus.

    Args:
        tokens: Full token stream.
        validation_split: Fraction reserved for validation.
        chunk_size: Number of tokens per shuffle unit.
        seed: Fixed seed for reproducible train/validation assignment.

    Returns:
        Pair of training tokens and validation tokens.
    """

    total = len(tokens)
    if total <= 1 or validation_split <= 0:
        return list(tokens), []
    if validation_split >= 1:
        return [], list(tokens)

    chunk_size = max(1, chunk_size)
    chunk_ranges = [(start, min(start + chunk_size, total)) for start in range(0, total, chunk_size)]
    if len(chunk_ranges) <= 1:
        split_at = int(total * (1.0 - validation_split))
        split_at = max(1, min(split_at, total - 1))
        return tokens[:split_at], tokens[split_at:]

    shuffled_indices = list(range(len(chunk_ranges)))
    random.Random(seed).shuffle(shuffled_indices)
    val_chunk_count = max(1, round(len(chunk_ranges) * validation_split))
    val_chunk_count = min(val_chunk_count, len(chunk_ranges) - 1)
    val_chunk_indices = set(shuffled_indices[:val_chunk_count])

    train_tokens: list[int] = []
    val_tokens: list[int] = []
    for chunk_index, (start, end) in enumerate(chunk_ranges):
        piece = tokens[start:end]
        if chunk_index in val_chunk_indices:
            val_tokens.extend(piece)
        else:
            train_tokens.extend(piece)
    return train_tokens, val_tokens


def make_optimizer(model: MicroGPT, training_config: TrainingConfig) -> torch.optim.Optimizer:
    """Create the configured optimizer.

    Args:
        model: Model whose parameters will be optimized.
        training_config: Training configuration.

    Returns:
        Configured optimizer.

    Raises:
        ValueError: If the optimizer is unsupported by the installed PyTorch.
    """

    name = training_config.optimizer_name
    common = {
        "lr": training_config.learning_rate,
        "weight_decay": training_config.weight_decay,
    }
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("No trainable parameters are available for optimization")
    if name == "adamw":
        return torch.optim.AdamW(parameters, betas=(0.9, 0.95), **common)
    if name == "adam":
        return torch.optim.Adam(parameters, betas=(0.9, 0.95), **common)
    if name == "lion":
        return Lion(parameters, betas=(0.9, 0.99), **common)
    if name == "adafactor":
        adafactor = getattr(torch.optim, "Adafactor", None)
        if adafactor is None:
            raise ValueError("Adafactor requires a newer PyTorch build that includes torch.optim.Adafactor")
        return adafactor(parameters, **common)
    raise ValueError(f"Unsupported optimizer: {name}")


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    training_config: TrainingConfig,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Create the configured learning-rate scheduler.

    Args:
        optimizer: Optimizer to schedule.
        total_steps: Total optimizer steps.
        training_config: Training configuration.

    Returns:
        Lambda learning-rate scheduler.
    """

    warmup_steps = training_config.warmup_steps
    warmup_steps = min(warmup_steps, max(total_steps - 1, 1))
    min_ratio = training_config.scheduler_min_lr_ratio
    schedule = training_config.scheduler_name

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return max(step, 1) / max(warmup_steps, 1)
        if schedule == "constant":
            return 1.0
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = max(0.0, min(progress, 1.0))
        if schedule == "cosine":
            value = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_ratio + (1.0 - min_ratio) * value
        if schedule == "polynomial":
            value = (1.0 - progress) ** training_config.polynomial_power
            return min_ratio + (1.0 - min_ratio) * value
        if schedule == "one_cycle":
            if progress < 0.3:
                return min_ratio + (1.0 - min_ratio) * (progress / 0.3)
            decay_progress = (progress - 0.3) / 0.7
            value = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
            return min_ratio + (1.0 - min_ratio) * value
        return max(min_ratio, 1.0 - progress)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def amp_settings(training_config: TrainingConfig) -> tuple[bool, bool, torch.dtype]:
    """Return autocast and scaler settings for the selected precision.

    Args:
        training_config: Training configuration.

    Returns:
        Tuple of ``use_autocast``, ``use_scaler``, and autocast dtype.
    """

    use_cuda_amp = training_config.use_amp and training_config.device == "cuda"
    if not use_cuda_amp or training_config.precision == "fp32":
        return False, False, torch.float32
    if training_config.precision == "bf16":
        return True, False, torch.bfloat16
    return True, True, torch.float16


def system_ram_percent() -> Optional[float]:
    """Return system RAM utilization when psutil is available.

    Returns:
        RAM utilization percentage, or None when unavailable.
    """

    if psutil is None:
        return None
    return float(psutil.virtual_memory().percent)


def system_cpu_percent() -> Optional[float]:
    """Return system CPU utilization when psutil is available.

    Returns:
        CPU utilization percentage, or None when unavailable.
    """

    if psutil is None:
        return None
    return float(psutil.cpu_percent(interval=None))


def evaluate(
    model: MicroGPT,
    loader: DataLoader,
    device: str,
    pad_token_id: int,
    max_batches: int = 50,
    progress: Optional[Callable[[Any], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    step: Optional[int] = None,
    total_steps: Optional[int] = None,
    percent: Optional[int] = None,
) -> float:
    """Evaluate validation loss.

    Args:
        model: Model to evaluate.
        loader: Validation data loader.
        device: Device used for evaluation.
        pad_token_id: Token ID ignored in loss.
        max_batches: Maximum validation batches to evaluate. Zero evaluates the full loader.
        progress: Optional progress callback.
        should_stop: Optional cancellation callback.
        step: Current optimizer step for progress metrics.
        total_steps: Total planned optimizer steps for progress metrics.
        percent: Current outer training progress percentage.

    Returns:
        Mean validation loss.
    """

    model.eval()
    losses: list[float] = []
    batch_limit = len(loader) if max_batches <= 0 else min(len(loader), max_batches)
    with torch.no_grad():
        for batch_index, (x, y) in enumerate(loader, start=1):
            if should_stop and should_stop():
                model.train()
                raise RuntimeError("Training stopped by user during validation.")
            if batch_index > batch_limit:
                break
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
                ignore_index=pad_token_id,
            )
            losses.append(float(loss.item()))
            if progress and (batch_index == 1 or batch_index == batch_limit or batch_index % 10 == 0):
                emit_progress(
                    progress,
                    f"Validation running: batch {batch_index}/{batch_limit}.",
                    percent,
                    step=step,
                    total_steps=total_steps,
                    system_cpu_percent=system_cpu_percent(),
                    system_ram_percent=system_ram_percent(),
                    validation_batch=batch_index,
                    validation_batches=batch_limit,
                )
    model.train()
    _release_cuda_cache()
    return sum(losses) / max(len(losses), 1)


def latest_checkpoint(checkpoints_dir: Path) -> Optional[Path]:
    """Find the newest checkpoint in a folder.

    Args:
        checkpoints_dir: Directory containing checkpoint files.

    Returns:
        Newest checkpoint path, or ``None``.
    """

    checkpoints = sorted(
        checkpoints_dir.glob("checkpoint_*.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return checkpoints[0] if checkpoints else None


def _config_value(data: dict[str, Any], key: str, default: Any) -> Any:
    """Return a saved config value with a default for old checkpoints.

    Args:
        data: Saved configuration dictionary.
        key: Configuration key.
        default: Default value when the key is missing.

    Returns:
        Saved or default value.
    """

    return data[key] if key in data else default


def _same_config_value(left: Any, right: Any) -> bool:
    """Compare config values with tolerance for numeric fields.

    Args:
        left: First value.
        right: Second value.

    Returns:
        True when values are effectively equal.
    """

    if isinstance(left, float) or isinstance(right, float):
        try:
            return abs(float(left) - float(right)) <= 1e-9
        except (TypeError, ValueError):
            return False
    return left == right


def _saved_model_default(key: str) -> Any:
    """Return ModelConfig defaults for legacy checkpoints.

    Args:
        key: ModelConfig field name.

    Returns:
        Default value used by current ModelConfig.
    """

    defaults = {
        "context_length": 128,
        "embedding_size": 256,
        "head_count": 4,
        "layer_count": 4,
        "dropout": 0.1,
        "bias": True,
        "norm_type": "layernorm",
        "position_encoding": "learned",
        "mlp_type": "gelu",
        "rope_theta": 10000.0,
        "attention_type": "mha",
        "kv_head_count": 0,
        "attention_backend": "sdpa",
        "attention_window": 0,
    }
    return defaults.get(key)


def _saved_training_default(key: str) -> Any:
    """Return TrainingConfig defaults for legacy checkpoints.

    Args:
        key: TrainingConfig field name.

    Returns:
        Default value used by current TrainingConfig.
    """

    defaults = {
        "optimizer_name": "adamw",
        "scheduler_name": "warmup_linear",
        "scheduler_min_lr_ratio": 0.1,
        "polynomial_power": 1.0,
        "learning_rate": 3e-4,
        "weight_decay": 0.1,
        "max_grad_norm": 1.0,
        "precision": "fp16",
        "use_amp": True,
        "training_mode": "pretrain",
        "fine_tune_from_checkpoint": None,
    }
    return defaults.get(key)


def check_resume_compatibility(
    checkpoint_path: Path,
    model_config: ModelConfig,
    training_config: TrainingConfig,
) -> ResumeCompatibilityReport:
    """Check whether a checkpoint can be safely resumed.

    Args:
        checkpoint_path: Checkpoint to inspect.
        model_config: Current model configuration.
        training_config: Current training configuration.

    Returns:
        Resume compatibility report.
    """

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    saved_model = checkpoint.get("model_config", {})
    saved_training = checkpoint.get("training_config", {})
    if not isinstance(saved_model, dict):
        saved_model = {}
    if not isinstance(saved_training, dict):
        saved_training = {}

    errors: list[str] = []
    warnings: list[str] = []
    info: list[str] = [f"Resume checkpoint: {checkpoint_path.name}."]

    critical_model_fields = (
        ("vocab_size", model_config.vocab_size, "Tokenizer vocabulary"),
        ("context_length", model_config.context_length, "Context length"),
        ("embedding_size", model_config.embedding_size, "n_embd"),
        ("head_count", model_config.head_count, "n_head"),
        ("layer_count", model_config.layer_count, "n_layer"),
        ("bias", model_config.bias, "Bias layout"),
        ("norm_type", model_config.norm_type, "Normalization"),
        ("position_encoding", model_config.position_encoding, "Position encoding"),
        ("mlp_type", model_config.mlp_type, "MLP type"),
        ("rope_theta", model_config.rope_theta, "RoPE theta"),
        ("attention_type", model_config.attention_type, "Attention type"),
    )
    for key, current_value, label in critical_model_fields:
        saved_value = _config_value(saved_model, key, _saved_model_default(key))
        if not _same_config_value(saved_value, current_value):
            errors.append(f"{label} changed: checkpoint={saved_value}, current={current_value}.")

    saved_attention_type = _config_value(saved_model, "attention_type", "mha")
    saved_kv_heads = _config_value(saved_model, "kv_head_count", 0)
    try:
        saved_kv_effective = ModelConfig(
            vocab_size=int(_config_value(saved_model, "vocab_size", model_config.vocab_size)),
            context_length=int(_config_value(saved_model, "context_length", _saved_model_default("context_length"))),
            embedding_size=int(_config_value(saved_model, "embedding_size", _saved_model_default("embedding_size"))),
            head_count=int(_config_value(saved_model, "head_count", _saved_model_default("head_count"))),
            layer_count=int(_config_value(saved_model, "layer_count", _saved_model_default("layer_count"))),
            attention_type=str(saved_attention_type),
            kv_head_count=int(saved_kv_heads),
        ).resolved_kv_head_count()
    except Exception:
        saved_kv_effective = saved_kv_heads
    current_kv_effective = model_config.resolved_kv_head_count()
    if saved_kv_effective != current_kv_effective:
        errors.append(f"Effective KV heads changed: checkpoint={saved_kv_effective}, current={current_kv_effective}.")

    warning_model_fields = (
        ("dropout", model_config.dropout, "Dropout"),
        ("attention_backend", model_config.attention_backend, "Attention backend"),
        ("attention_window", model_config.attention_window, "Sliding attention window"),
    )
    for key, current_value, label in warning_model_fields:
        saved_value = _config_value(saved_model, key, _saved_model_default(key))
        if not _same_config_value(saved_value, current_value):
            warnings.append(f"{label} changed: checkpoint={saved_value}, current={current_value}.")

    can_load_optimizer_state = True
    can_load_scheduler_state = True
    can_load_scaler_state = True
    if "optimizer_state_dict" in checkpoint:
        saved_optimizer = _config_value(saved_training, "optimizer_name", _saved_training_default("optimizer_name"))
        if saved_optimizer != training_config.optimizer_name:
            warnings.append(
                f"Optimizer changed: checkpoint={saved_optimizer}, current={training_config.optimizer_name}. "
                "Optimizer state will not be loaded."
            )
            can_load_optimizer_state = False
    if "scheduler_state_dict" in checkpoint:
        saved_scheduler = _config_value(saved_training, "scheduler_name", _saved_training_default("scheduler_name"))
        if saved_scheduler != training_config.scheduler_name:
            warnings.append(
                f"LR scheduler changed: checkpoint={saved_scheduler}, current={training_config.scheduler_name}. "
                "Scheduler state will not be loaded."
            )
            can_load_scheduler_state = False
        for key, current_value, label in (
            ("scheduler_min_lr_ratio", training_config.scheduler_min_lr_ratio, "Scheduler min LR ratio"),
            ("polynomial_power", training_config.polynomial_power, "Polynomial power"),
        ):
            saved_value = _config_value(saved_training, key, _saved_training_default(key))
            if not _same_config_value(saved_value, current_value):
                warnings.append(f"{label} changed: checkpoint={saved_value}, current={current_value}.")
    if "scaler_state_dict" in checkpoint:
        saved_precision = _config_value(saved_training, "precision", _saved_training_default("precision"))
        if saved_precision != training_config.precision:
            warnings.append(f"Precision changed: checkpoint={saved_precision}, current={training_config.precision}.")
            can_load_scaler_state = saved_precision == "fp16" and training_config.precision == "fp16"

    for key, current_value, label in (
        ("learning_rate", training_config.learning_rate, "Learning rate"),
        ("weight_decay", training_config.weight_decay, "Weight decay"),
        ("max_grad_norm", training_config.max_grad_norm, "Gradient clipping"),
    ):
        saved_value = _config_value(saved_training, key, _saved_training_default(key))
        if not _same_config_value(saved_value, current_value):
            warnings.append(f"{label} changed: checkpoint={saved_value}, current={current_value}.")

    if not errors:
        info.append("Checkpoint architecture and tokenizer are compatible.")
    return ResumeCompatibilityReport(
        checkpoint_path=checkpoint_path,
        errors=errors,
        warnings=warnings,
        info=info,
        can_load_optimizer_state=can_load_optimizer_state,
        can_load_scheduler_state=can_load_scheduler_state,
        can_load_scaler_state=can_load_scaler_state,
    )


def _configure_cuda_allocator() -> None:
    """Configure the CUDA caching allocator to reduce VRAM over-reservation.

    Sets ``expandable_segments:True`` so PyTorch grows GPU memory in smaller
    increments rather than grabbing large contiguous blocks up front.
    """

    key = "PYTORCH_CUDA_ALLOC_CONF"
    current = os.environ.get(key, "")
    if "expandable_segments" not in current:
        new_value = "expandable_segments:True"
        if current:
            new_value = current + "," + new_value
        os.environ[key] = new_value


def _release_cuda_cache() -> None:
    """Release unused cached VRAM back to the OS."""

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train_model(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    train_tokens: Union[list[int], np.ndarray],
    val_tokens: Union[list[int], np.ndarray],
    pad_token_id: int,
    progress: Optional[Callable[[Any], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    decode_preview: Optional[Callable[[list[int]], str]] = None,
) -> TrainingResult:
    """Train a MicroGPT model.

    Args:
        model_config: Architecture settings.
        training_config: Optimizer, device, and checkpoint settings.
        train_tokens: Training token stream.
        val_tokens: Validation token stream.
        pad_token_id: Token ID ignored by cross-entropy loss.
        progress: Optional callback receiving progress dictionaries.
        should_stop: Optional callback returning true when training should stop.
        decode_preview: Optional callback that decodes token IDs into a short text preview.

    Returns:
        Training result with checkpoint and summary paths.
    """

    model_config.validate()
    training_config.validate()
    set_seed(training_config.seed)

    # Reduce VRAM over-reservation by the CUDA caching allocator.
    if training_config.device.startswith("cuda") and torch.cuda.is_available():
        _configure_cuda_allocator()

    training_config.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = training_config.output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    emit_progress(progress, "Building model...", 2)
    model = MicroGPT(model_config).to(training_config.device)
    _release_cuda_cache()
    emit_progress(progress, "Preparing token batches...", 4)
    loader_workers = max(0, int(training_config.data_loader_workers))
    pin_memory = training_config.device.startswith("cuda") and torch.cuda.is_available()
    loader_kwargs = {
        "num_workers": loader_workers,
        "pin_memory": pin_memory,
        "persistent_workers": loader_workers > 0,
    }
    train_loader = DataLoader(
        TokenDataset(train_tokens, model_config.context_length, stride=training_config.sample_stride),
        batch_size=training_config.batch_size,
        shuffle=True,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = None
    if len(val_tokens) > model_config.context_length:
        val_loader = DataLoader(
            TokenDataset(val_tokens, model_config.context_length),
            batch_size=training_config.batch_size,
            shuffle=False,
            drop_last=False,
            **loader_kwargs,
        )

    global_step = 0
    start_epoch = 0
    final_train_loss = 0.0
    final_val_loss: Optional[float] = None
    best_val_loss: Optional[float] = None
    best_checkpoint_path: Optional[Path] = None
    early_stop_counter = 0
    early_stopped = False

    resume_path = training_config.resume_from_checkpoint if training_config.resume else None
    if resume_path is None and training_config.resume:
        resume_path = latest_checkpoint(checkpoints_dir)
    resume_checkpoint: Optional[dict[str, Any]] = None
    resume_compatibility: Optional[ResumeCompatibilityReport] = None
    if training_config.peft_method == "lora":
        base_path = training_config.fine_tune_from_checkpoint
        if resume_path and Path(resume_path).exists():
            resume_checkpoint = torch.load(resume_path, map_location=training_config.device)
            checkpoint_base = resume_checkpoint.get("fine_tune_base_checkpoint")
            if checkpoint_base:
                base_path = Path(checkpoint_base)
        if base_path is None:
            raise ValueError("LoRA fine-tuning requires a base checkpoint.")
        base_path = Path(base_path)
        if not base_path.exists():
            raise FileNotFoundError(f"LoRA base checkpoint not found: {base_path}")
        emit_progress(progress, f"Loading LoRA base checkpoint: {base_path}", 5)
        base_checkpoint = torch.load(base_path, map_location=training_config.device)
        model.load_state_dict(base_checkpoint["model_state_dict"])
        wrapped = apply_lora_adapters(
            model,
            training_config.lora_rank,
            training_config.lora_alpha,
            training_config.lora_dropout,
            training_config.lora_target_modules,
        )
        freeze_non_lora_parameters(model)
        emit_progress(
            progress,
            f"LoRA enabled: {wrapped} module(s), {lora_parameter_count(model):,} trainable adapter parameter(s).",
            6,
        )

    optimizer = make_optimizer(model, training_config)
    steps_per_epoch = max(math.ceil(len(train_loader) / training_config.gradient_accumulation), 1)
    total_steps = max(steps_per_epoch * training_config.epochs, 1)
    scheduler = make_scheduler(optimizer, total_steps, training_config)
    use_autocast, use_scaler, autocast_dtype = amp_settings(training_config)
    scaler = GradScaler("cuda", enabled=use_scaler)
    emit_progress(
        progress,
        "Optimizer: "
        f"{training_config.optimizer_name}, schedule: {training_config.scheduler_name}, "
        f"precision: {training_config.precision}.",
        5,
    )
    if resume_path and Path(resume_path).exists():
        emit_progress(progress, f"Resuming from checkpoint: {resume_path}", 6)
        compatibility = resume_compatibility or check_resume_compatibility(Path(resume_path), model_config, training_config)
        for line in compatibility.info:
            emit_progress(progress, line, 6)
        for line in compatibility.warnings:
            emit_progress(progress, f"[WARN] {line}", 6)
        strict_resume_errors = list(compatibility.errors)
        if training_config.require_compatible_resume:
            if not compatibility.can_load_optimizer_state:
                strict_resume_errors.append("Safe resume requires matching optimizer state.")
            if not compatibility.can_load_scheduler_state:
                strict_resume_errors.append("Safe resume requires matching scheduler state.")
            if not compatibility.can_load_scaler_state:
                strict_resume_errors.append("Safe resume requires matching AMP scaler state.")
        if strict_resume_errors:
            message = "Checkpoint is not compatible with the current training settings:\n" + "\n".join(
                f"- {line}" for line in strict_resume_errors
            )
            raise ValueError(message)
        checkpoint = resume_checkpoint or torch.load(resume_path, map_location=training_config.device)
        if training_config.peft_method == "lora" and "adapter_state_dict" in checkpoint:
            load_lora_state_dict(model, checkpoint["adapter_state_dict"])
        else:
            model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint and compatibility.can_load_optimizer_state:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scaler_state_dict" in checkpoint and use_scaler and compatibility.can_load_scaler_state:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        global_step = int(checkpoint.get("global_step", 0))
        start_epoch = min(int(checkpoint.get("epoch", 0)), training_config.epochs)
        final_train_loss = float(checkpoint.get("train_loss", 0.0))
        final_val_loss = checkpoint.get("val_loss")
        # Recompute total_steps to account for the resumed global_step so that
        # progress tracking, ETA, and the LR scheduler use a consistent target.
        remaining_epochs = max(training_config.epochs - start_epoch, 0)
        total_steps = global_step + steps_per_epoch * remaining_epochs
        # Recreate the scheduler against the corrected total_steps and reload
        # its state so the LR curve stays consistent with the checkpoint.
        scheduler = make_scheduler(optimizer, total_steps, training_config)
        if "scheduler_state_dict" in checkpoint and compatibility.can_load_scheduler_state:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        emit_progress(progress, f"Checkpoint loaded at step {global_step}.", 8)
        _release_cuda_cache()
    else:
        if (
            training_config.training_mode == "fine_tune"
            and training_config.fine_tune_from_checkpoint is not None
            and training_config.peft_method != "lora"
        ):
            base_path = Path(training_config.fine_tune_from_checkpoint)
            if not base_path.exists():
                raise FileNotFoundError(f"Fine-tune base checkpoint not found: {base_path}")
            emit_progress(progress, f"Fine-tuning from base checkpoint: {base_path}", 6)
            compatibility = check_resume_compatibility(base_path, model_config, training_config)
            for line in compatibility.info:
                emit_progress(progress, line, 6)
            for line in compatibility.warnings:
                emit_progress(progress, f"[WARN] {line}", 6)
            if compatibility.errors:
                message = "Fine-tune base checkpoint is not compatible with the current model settings:\n" + "\n".join(
                    f"- {line}" for line in compatibility.errors
                )
                raise ValueError(message)
            checkpoint = torch.load(base_path, map_location=training_config.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            emit_progress(progress, "Base model weights loaded. Starting fresh fine-tune optimizer state.", 8)
        else:
            emit_progress(progress, "Starting new training run.", 6)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    last_metric_time = perf_counter()
    step_time_window: list[float] = []
    for epoch in range(start_epoch, training_config.epochs):
        epoch_losses: list[float] = []
        epoch_batch_count = len(train_loader)
        for batch_index, (x, y) in enumerate(train_loader):
            if should_stop and should_stop():
                final_train_loss = sum(epoch_losses) / max(len(epoch_losses), 1) if epoch_losses else final_train_loss
                stopped_path = checkpoints_dir / f"checkpoint_stopped_step_{global_step}.pt"
                save_checkpoint(
                    stopped_path,
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    model_config,
                    training_config,
                    global_step,
                    epoch,
                    final_train_loss,
                    final_val_loss,
                )
                emit_progress(progress, f"Training stopped. Resume checkpoint saved: {stopped_path}", 100)
                summary_path = training_config.output_dir / "training_summary.json"
                summary = {
                    "model_config": dataclass_to_jsonable(model_config),
                    "training_config": dataclass_to_jsonable(training_config),
                    "final_train_loss": final_train_loss,
                    "final_val_loss": final_val_loss,
                    "total_steps": global_step,
                    "stopped": True,
                    "resume_checkpoint": str(stopped_path),
                    "parameters": sum(p.numel() for p in model.parameters()),
                }
                summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
                return TrainingResult(stopped_path, summary_path, final_train_loss, final_val_loss, stopped=True)
            x = x.to(training_config.device)
            y = y.to(training_config.device)
            with autocast("cuda", enabled=use_autocast, dtype=autocast_dtype):
                logits = model(x)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    y.reshape(-1),
                    ignore_index=pad_token_id,
                )
                loss = loss / training_config.gradient_accumulation

            scaler.scale(loss).backward()
            should_step = (
                (batch_index + 1) % training_config.gradient_accumulation == 0
                or (batch_index + 1) == epoch_batch_count
            )
            if should_step:
                scaler.unscale_(optimizer)
                grad_norm_tensor = torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.max_grad_norm)
                grad_norm = float(grad_norm_tensor.item() if hasattr(grad_norm_tensor, "item") else grad_norm_tensor)
                weight_norm = math.sqrt(
                    sum(float(parameter.detach().float().norm(2).item()) ** 2 for parameter in model.parameters())
                )
                learning_rate = float(scheduler.get_last_lr()[0])
                update_ratio = learning_rate * grad_norm / max(weight_norm, 1e-12)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
                now = perf_counter()
                step_seconds = max(now - last_metric_time, 1e-9)
                last_metric_time = now
                step_time_window.append(step_seconds)
                step_time_window = step_time_window[-50:]
                average_step_seconds = sum(step_time_window) / max(len(step_time_window), 1)
                remaining_steps = max(total_steps - global_step, 0)
                eta_seconds = remaining_steps * average_step_seconds
                samples_seen = training_config.batch_size * training_config.gradient_accumulation
                tokens_seen = samples_seen * model_config.context_length
                vram_allocated_gb = None
                vram_reserved_gb = None
                gpu_memory_percent = None
                if training_config.device.startswith("cuda") and torch.cuda.is_available():
                    device_index = torch.cuda.current_device()
                    vram_allocated_gb = torch.cuda.memory_allocated(device_index) / (1024 ** 3)
                    vram_reserved_gb = torch.cuda.memory_reserved(device_index) / (1024 ** 3)
                    free_vram, total_vram = torch.cuda.mem_get_info(device_index)
                    gpu_memory_percent = 100.0 * (1.0 - (free_vram / max(total_vram, 1)))
                sample_text = None
                if decode_preview is not None:
                    try:
                        sample_text = decode_preview(x[0].detach().cpu().tolist())
                    except Exception:
                        sample_text = None
                current_progress = 8 + int(86 * min(global_step, total_steps) / max(total_steps, 1))
                emit_progress(
                    progress,
                    f"Epoch {epoch + 1}/{training_config.epochs}, step {global_step}/{total_steps}, loss {float(loss.item() * training_config.gradient_accumulation):.4f}",
                    current_progress,
                    epoch=epoch + 1,
                    total_epochs=training_config.epochs,
                    step=global_step,
                    total_steps=total_steps,
                    train_loss=float(loss.item() * training_config.gradient_accumulation),
                    val_loss=final_val_loss,
                    learning_rate=learning_rate,
                    grad_norm=grad_norm,
                    weight_norm=weight_norm,
                    update_ratio=update_ratio,
                    tokens_per_second=tokens_seen / step_seconds,
                    samples_per_second=samples_seen / step_seconds,
                    step_seconds=step_seconds,
                    average_step_seconds=average_step_seconds,
                    eta_seconds=eta_seconds,
                    remaining_steps=remaining_steps,
                    vram_allocated_gb=vram_allocated_gb,
                    vram_reserved_gb=vram_reserved_gb,
                    gpu_memory_percent=gpu_memory_percent,
                    system_cpu_percent=system_cpu_percent(),
                    system_ram_percent=system_ram_percent(),
                    data_loader_workers=loader_workers,
                    sample_text=sample_text,
                )

                if (
                    val_loader is not None
                    and training_config.eval_interval > 0
                    and global_step % training_config.eval_interval == 0
                ):
                    emit_progress(
                        progress,
                        f"Running validation at step {global_step}...",
                        current_progress,
                        epoch=epoch + 1,
                        total_epochs=training_config.epochs,
                        step=global_step,
                        total_steps=total_steps,
                        train_loss=float(loss.item() * training_config.gradient_accumulation),
                        val_loss=final_val_loss,
                        system_cpu_percent=system_cpu_percent(),
                        system_ram_percent=system_ram_percent(),
                    )
                    final_val_loss = evaluate(
                        model,
                        val_loader,
                        training_config.device,
                        pad_token_id,
                        training_config.max_eval_batches,
                        progress,
                        should_stop,
                        global_step,
                        total_steps,
                        current_progress,
                    )
                    emit_progress(
                        progress,
                        f"Validation loss at step {global_step}: {final_val_loss:.4f}",
                        current_progress,
                        epoch=epoch + 1,
                        total_epochs=training_config.epochs,
                        step=global_step,
                        total_steps=total_steps,
                        train_loss=epoch_losses[-1] if epoch_losses else None,
                        val_loss=final_val_loss,
                        system_cpu_percent=system_cpu_percent(),
                        system_ram_percent=system_ram_percent(),
                    )
                    if best_val_loss is None or final_val_loss < best_val_loss:
                        best_val_loss = final_val_loss
                        best_checkpoint_path = checkpoints_dir / "checkpoint_best_val.pt"
                        save_checkpoint(
                            best_checkpoint_path,
                            model,
                            optimizer,
                            scheduler,
                            scaler,
                            model_config,
                            training_config,
                            global_step,
                            epoch + 1,
                            epoch_losses[-1] if epoch_losses else final_train_loss,
                            final_val_loss,
                        )
                        emit_progress(
                            progress,
                            f"New best validation checkpoint: {best_checkpoint_path.name} ({best_val_loss:.4f}).",
                            current_progress,
                            checkpoint_quality="best_validation",
                            best_val_loss=best_val_loss,
                            best_checkpoint_path=str(best_checkpoint_path),
                        )
                        early_stop_counter = 0
                    elif training_config.early_stopping and best_val_loss is not None:
                        early_stop_counter += 1
                        if early_stop_counter >= training_config.early_stopping_patience:
                            reason = (
                                f"Early stopping: validation loss has not improved for "
                                f"{early_stop_counter} consecutive evaluation(s). "
                                f"Best val loss: {best_val_loss:.4f}, current: {final_val_loss:.4f}. "
                                f"Best checkpoint: {best_checkpoint_path}."
                            )
                            emit_progress(progress, reason, current_progress)
                            early_stopped = True
                            break

                if training_config.save_interval > 0 and global_step % training_config.save_interval == 0:
                    save_checkpoint(
                        checkpoints_dir / f"checkpoint_{global_step}.pt",
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        model_config,
                        training_config,
                        global_step,
                        epoch + 1,
                        final_train_loss,
                        final_val_loss,
                    )
                    emit_progress(progress, f"Saved checkpoint at step {global_step}.", current_progress)

            epoch_losses.append(float(loss.item() * training_config.gradient_accumulation))

        if early_stopped:
            break
        final_train_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        if val_loader is not None:
            final_val_loss = evaluate(
                model,
                val_loader,
                training_config.device,
                pad_token_id,
                training_config.max_eval_batches,
                progress,
                should_stop,
                global_step,
                total_steps,
                8 + int(86 * (epoch + 1) / max(training_config.epochs, 1)),
            )
            if best_val_loss is None or final_val_loss < best_val_loss:
                best_val_loss = final_val_loss
                best_checkpoint_path = checkpoints_dir / "checkpoint_best_val.pt"
                save_checkpoint(
                    best_checkpoint_path,
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    model_config,
                    training_config,
                    global_step,
                    epoch + 1,
                    final_train_loss,
                    final_val_loss,
                )
                emit_progress(
                    progress,
                    f"New best validation checkpoint: {best_checkpoint_path.name} ({best_val_loss:.4f}).",
                    8 + int(86 * (epoch + 1) / max(training_config.epochs, 1)),
                    checkpoint_quality="best_validation",
                    best_val_loss=best_val_loss,
                    best_checkpoint_path=str(best_checkpoint_path),
                )
                early_stop_counter = 0
            elif training_config.early_stopping and best_val_loss is not None:
                early_stop_counter += 1
                if early_stop_counter >= training_config.early_stopping_patience:
                    reason = (
                        f"Early stopping at epoch {epoch + 1}: validation loss has not improved for "
                        f"{early_stop_counter} consecutive evaluation(s). "
                        f"Best val loss: {best_val_loss:.4f}, current: {final_val_loss:.4f}. "
                        f"Best checkpoint: {best_checkpoint_path}."
                    )
                    emit_progress(progress, reason, 8 + int(86 * (epoch + 1) / max(training_config.epochs, 1)))
                    early_stopped = True
        print(f"epoch {epoch + 1}/{training_config.epochs}: train_loss={final_train_loss:.4f}")
        save_checkpoint(
            checkpoints_dir / f"checkpoint_epoch_{epoch + 1}.pt",
            model,
            optimizer,
            scheduler,
            scaler,
            model_config,
            training_config,
            global_step,
            epoch + 1,
            final_train_loss,
            final_val_loss,
        )
        emit_progress(
            progress,
            f"Epoch {epoch + 1} complete. Checkpoint saved.",
            8 + int(86 * (epoch + 1) / max(training_config.epochs, 1)),
            epoch=epoch + 1,
            total_epochs=training_config.epochs,
            step=global_step,
            total_steps=total_steps,
            train_loss=final_train_loss,
            val_loss=final_val_loss,
            system_cpu_percent=system_cpu_percent(),
            system_ram_percent=system_ram_percent(),
        )
        if early_stopped:
            break

    if training_config.peft_method == "lora":
        adapter_path = training_config.output_dir / "final_adapter.pt"
        save_checkpoint(
            adapter_path,
            model,
            optimizer,
            scheduler,
            scaler,
            model_config,
            training_config,
            global_step,
            training_config.epochs,
            final_train_loss,
            final_val_loss,
        )
        merged_count = merge_lora_adapters(model)
        emit_progress(progress, f"Merged {merged_count} LoRA adapter module(s) into final model weights.", 96)
    checkpoint_path = training_config.output_dir / "final_model.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        scheduler,
        scaler,
        model_config,
        training_config,
        global_step,
        training_config.epochs,
        final_train_loss,
        final_val_loss,
    )
    summary_path = training_config.output_dir / "training_summary.json"
    summary = {
        "model_config": dataclass_to_jsonable(model_config),
        "training_config": dataclass_to_jsonable(training_config),
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "best_val_loss": best_val_loss,
        "best_checkpoint_path": str(best_checkpoint_path) if best_checkpoint_path else None,
        "recommended_checkpoint_path": str(best_checkpoint_path or checkpoint_path),
        "total_steps": global_step,
        "parameters": sum(p.numel() for p in model.parameters()),
        "adapter_checkpoint": str(training_config.output_dir / "final_adapter.pt") if training_config.peft_method == "lora" else None,
        "early_stopped": early_stopped,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    emit_progress(
        progress,
        "Training stopped early — validation loss converged." if early_stopped else "Training complete.",
        100,
        epoch=training_config.epochs,
        total_epochs=training_config.epochs,
        step=global_step,
        total_steps=total_steps,
        train_loss=final_train_loss,
        val_loss=final_val_loss,
    )
    return TrainingResult(checkpoint_path, summary_path, final_train_loss, final_val_loss)


def save_checkpoint(
    path: Path,
    model: MicroGPT,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    global_step: int,
    epoch: int,
    train_loss: float,
    val_loss: Optional[float],
) -> None:
    """Save a resumable training checkpoint.

    Args:
        path: Destination checkpoint path.
        model: Model being trained.
        optimizer: Optimizer state to save.
        scheduler: Learning-rate scheduler state to save.
        scaler: AMP scaler state to save.
        model_config: Model configuration.
        training_config: Training configuration.
        global_step: Current optimizer step.
        epoch: Current epoch number.
        train_loss: Most recent training loss.
        val_loss: Most recent validation loss.
    """

    payload = {
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "model_config": dataclass_to_jsonable(model_config),
        "training_config": dataclass_to_jsonable(training_config),
        "global_step": global_step,
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
    }
    if training_config.peft_method == "lora" and path.name != "final_model.pt":
        payload["adapter_state_dict"] = lora_state_dict(model)
        payload["fine_tune_base_checkpoint"] = (
            str(training_config.fine_tune_from_checkpoint)
            if training_config.fine_tune_from_checkpoint
            else None
        )
        payload["lora_config"] = {
            "rank": training_config.lora_rank,
            "alpha": training_config.lora_alpha,
            "dropout": training_config.lora_dropout,
            "target_modules": training_config.lora_target_modules,
        }
    else:
        payload["model_state_dict"] = model.state_dict()
    torch.save(payload, path)
