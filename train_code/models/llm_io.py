"""Resolve local LLM checkpoints and load their own tokenizers.

A Hub cache repository directory is not itself a ``from_pretrained``
checkpoint: the files live in ``snapshots/<revision>``. All LLM consumers
must use the returned directory for both model and tokenizer loading.
"""
from pathlib import Path
import json
import warnings


def resolve_model_path(model_path, *, required_file="config.json") -> str:
    """Return a local checkpoint, resolving an unambiguous Hub cache root.

    ``refs/main`` takes precedence when present. Without it, exactly one
    snapshot must contain ``required_file``; never select by date or name.
    """
    path = Path(model_path).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(
            f"Local model directory does not exist: {path}. "
            "Check MODEL_PATHS in config/path.py."
        )
    if (path / required_file).is_file():
        return str(path)

    snapshots = path / "snapshots"
    if not snapshots.is_dir():
        raise FileNotFoundError(
            f"{path} contains neither {required_file} nor snapshots/. "
            "Point MODEL_PATHS to the downloaded checkpoint directory."
        )

    main_ref = path / "refs" / "main"
    if main_ref.is_file():
        revision = main_ref.read_text(encoding="utf-8").strip()
        if not revision or revision in {".", ".."} or "/" in revision or "\\" in revision:
            raise ValueError(f"Invalid revision in {main_ref}: {revision!r}")
        checkpoint = snapshots / revision
        if not (checkpoint / required_file).is_file():
            raise FileNotFoundError(
                f"{main_ref} selects {checkpoint}, but {required_file} is missing "
                "or its symlink is broken. Complete that snapshot or set "
                "MODEL_PATHS to an explicit complete checkpoint."
            )
        return str(checkpoint)

    candidates = sorted(
        child for child in snapshots.iterdir()
        if child.is_dir() and (child / required_file).is_file()
    )
    if len(candidates) == 1:
        return str(candidates[0])
    if not candidates:
        raise FileNotFoundError(
            f"No snapshot containing {required_file} exists under {snapshots}. "
            "Check the download and any symlink targets; model files must "
            "be available locally."
        )
    choices = ", ".join(child.name for child in candidates)
    raise ValueError(
        f"Multiple snapshots under {snapshots} contain {required_file} "
        f"({choices}), and refs/main is absent. Set MODEL_PATHS to the "
        "specific snapshots/<revision> directory you intend to use."
    )


def get_llm_hidden_dim(resolved_path) -> int:
    """Read the decoder width from a local config without loading weights."""
    with (Path(resolved_path) / "config.json").open(encoding="utf-8") as handle:
        config = json.load(handle)
    hidden_dim = config.get("hidden_size")
    if isinstance(hidden_dim, bool) or not isinstance(hidden_dim, int) or hidden_dim <= 0:
        raise ValueError(f"No valid text hidden_size in {resolved_path}/config.json")
    return hidden_dim


def load_llm_tokenizer(resolved_path):
    """Load the checkpoint's own tokenizer (fast when supplied), locally only."""
    path = Path(resolved_path)
    has_assets = (
        (path / "tokenizer.json").is_file()
        or ((path / "vocab.json").is_file() and (path / "merges.txt").is_file())
        or (path / "tokenizer.model").is_file()
        or (path / "spiece.model").is_file()
    )
    if not has_assets:
        raise FileNotFoundError(
            f"No usable tokenizer assets found in {path}. Expected "
            "tokenizer.json, vocab.json + merges.txt, or a SentencePiece "
            "tokenizer.model/spiece.model from this same checkpoint. "
            "Check that the download includes tokenizer files and symlinks "
            "are intact; tokenizer_config.json alone is insufficient."
        )

    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(path), trust_remote_code=True, local_files_only=True, use_fast=True
        )
    except (OSError, ValueError, ImportError) as exc:
        raise RuntimeError(
            f"Cannot load the tokenizer from the resolved checkpoint {path}. "
            "Check tokenizer file completeness and the installed transformers/"
            "tokenizers versions. The original backend error is preserved below; "
            "a conversion-dependency hint alone does not establish that "
            "SentencePiece or tiktoken is needed."
        ) from exc
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError(f"Tokenizer at {path} has neither a pad token nor an EOS token.")
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _flash_attention_fallback(reason, *, cause=None):
    """Honor the fallback setting and retain the original dependency error."""
    from config.path import LLM_ALLOW_SDPA_FALLBACK

    if not LLM_ALLOW_SDPA_FALLBACK:
        raise RuntimeError(
            f"Cannot enable flash_attention_2: {reason}. Install a compatible "
            "flash-attn build/use supported hardware, or set "
            "LLM_ALLOW_SDPA_FALLBACK=True in config/path.py."
        ) from cause
    warnings.warn(
        f"Cannot enable flash_attention_2: {reason}. Using PyTorch SDPA instead.",
        RuntimeWarning,
        stacklevel=3,
    )
    return "sdpa"


def _select_attention_backend(torch_dtype):
    """Check prerequisites and import the native extension before loading weights."""
    import torch
    from config.path import LLM_ATTENTION_IMPLEMENTATION

    requested = LLM_ATTENTION_IMPLEMENTATION
    if requested not in {"flash_attention_2", "sdpa", "eager"}:
        raise ValueError(
            "LLM_ATTENTION_IMPLEMENTATION must be flash_attention_2, sdpa, "
            f"or eager; got {requested!r}."
        )
    if requested != "flash_attention_2":
        return requested

    from transformers.utils import is_flash_attn_2_available

    reason = None
    if torch_dtype not in (torch.float16, torch.bfloat16):
        reason = "FlashAttention-2 requires float16 or bfloat16 model weights"
    elif not torch.cuda.is_available():
        reason = "no CUDA/ROCm GPU is available to PyTorch"
    elif not getattr(torch.version, "hip", None) and torch.cuda.get_device_capability()[0] < 8:
        reason = "the current NVIDIA GPU predates Ampere (compute capability < 8.0)"
    elif not is_flash_attn_2_available():
        reason = "Transformers cannot find a supported flash-attn installation"

    if reason is not None:
        return _flash_attention_fallback(reason)

    try:
        # Package metadata alone does not detect missing CUDA libraries or ABI
        # mismatches. Importing these entry points loads the native extension.
        from flash_attn import flash_attn_func, flash_attn_varlen_func
    except (ImportError, OSError) as exc:
        return _flash_attention_fallback(
            f"flash-attn import failed ({type(exc).__name__}: {exc})", cause=exc
        )
    return requested


def _is_flash_attention_import_error(exc):
    """Recognize lazy import failures, including exceptions wrapped by Transformers."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, (ImportError, OSError)):
            if isinstance(exc, ImportError) and any(
                name in str(exc).lower()
                for name in ("flash_attn", "flash-attn", "flashattention", "flash attention")
            ):
                return True
            traceback = exc.__traceback__
            while traceback is not None:
                module = traceback.tb_frame.f_globals.get("__name__", "")
                if module.startswith("flash_attn") or module == "transformers.modeling_flash_attention_utils":
                    return True
                traceback = traceback.tb_next
        exc = exc.__cause__ if exc.__cause__ is not None else exc.__context__
    return False


def load_llm_model(resolved_path, torch_dtype):
    """Load local causal-LM weights with the configured attention backend.

    Device placement remains the caller's responsibility. FlashAttention import
    failures may select SDPA; unrelated load errors and SDPA failures propagate.
    """
    from packaging.version import Version
    from transformers import AutoModelForCausalLM, __version__ as transformers_version

    backend = _select_attention_backend(torch_dtype)
    # Transformers 4.56 renamed this argument; keep older supported versions usable.
    dtype_key = "dtype" if Version(transformers_version) >= Version("4.56.0") else "torch_dtype"
    load_kwargs = {"trust_remote_code": True, "local_files_only": True, dtype_key: torch_dtype}
    retry_with_sdpa = False
    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(resolved_path), attn_implementation=backend, **load_kwargs
        )
    except (ImportError, OSError, RuntimeError) as exc:
        if backend != "flash_attention_2" or not _is_flash_attention_import_error(exc):
            raise
        backend = _flash_attention_fallback(
            f"FlashAttention import during model loading failed ({type(exc).__name__}: {exc})",
            cause=exc,
        )
        retry_with_sdpa = True

    # Retry only once, outside the exception handler so the failed load's
    # traceback (and any partially initialized model) can be released first.
    if retry_with_sdpa:
        model = AutoModelForCausalLM.from_pretrained(
            str(resolved_path), attn_implementation=backend, **load_kwargs
        )
    actual_backend = getattr(model.config, "_attn_implementation", backend)
    print(f"[LLM] Attention backend: {actual_backend}; checkpoint: {resolved_path}")
    return model
