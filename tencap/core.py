from __future__ import annotations

import contextvars
import dataclasses
import datetime as _dt
import json
import os
import re
import tempfile
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

_LOGGER_NAME = "tencap"


@dataclasses.dataclass
class _Config:
    root_dir: Path | None = None
    enabled: bool = True
    interval: int = 1
    max_captures: int | None = None


@dataclasses.dataclass
class _Scope:
    tag: str
    enabled: bool | None = None
    interval: int | None = None
    max_captures: int | None = None


@dataclasses.dataclass
class _Runtime:
    capture_count: int = 0
    active_step: int | None = None
    active_dir: Path | None = None


@dataclasses.dataclass
class _State:
    config: _Config = dataclasses.field(default_factory=_Config)
    step: int = 0
    pid: int = dataclasses.field(default_factory=os.getpid)
    runtimes: dict[tuple[Path, str], _Runtime] = dataclasses.field(default_factory=dict)
    lock: threading.RLock = dataclasses.field(default_factory=threading.RLock)


_STATE = _State()

_SCOPE_STACK: contextvars.ContextVar[list[_Scope]] = contextvars.ContextVar(
    "tencap_scope_stack",
    default=[],
)


_INVALID_PART_CHARS_RE = re.compile(r'[\x00-\x1f<>:"/\\|?*]+')
_WHITESPACE_RE = re.compile(r"\s+")
_DUP_UNDERSCORE_RE = re.compile(r"_+")


_UNSET = object()


def setup(
    root_dir: str | os.PathLike[str] | object | None = _UNSET,
    *,
    enabled: bool | object = _UNSET,
    interval: int | object = _UNSET,
    max_captures: int | None | object = _UNSET,
    mkdir: bool = True,
) -> None:
    """
    Global initialization. Can be called multiple times to update config.

    - `interval`: capture every N steps (N>=1).
    - `max_captures`: per (root_dir, tag_path, pid) maximum capture rounds.
    """
    with _STATE.lock:
        _refresh_pid_locked()
        if root_dir is not _UNSET:
            if root_dir is None:
                raise ValueError("root_dir cannot be None")
            _STATE.config.root_dir = Path(root_dir).expanduser().resolve()
        if enabled is not _UNSET:
            _STATE.config.enabled = bool(enabled)
        if interval is not _UNSET:
            if interval is None:  # type: ignore[redundant-expr]
                raise ValueError("interval cannot be None")
            if interval < 1:  # type: ignore[operator]
                raise ValueError(f"interval must be >= 1, got {interval}")
            _STATE.config.interval = int(interval)  # type: ignore[arg-type]
        if max_captures is not _UNSET:
            if max_captures is None:
                _STATE.config.max_captures = None
            else:
                if max_captures < 0:  # type: ignore[operator]
                    raise ValueError(f"max_captures must be >= 0, got {max_captures}")
                _STATE.config.max_captures = int(max_captures)  # type: ignore[arg-type]

        if _STATE.config.root_dir is None:
            raise ValueError("root_dir is required on first setup() call")

        if mkdir:
            _STATE.config.root_dir.mkdir(parents=True, exist_ok=True)


@contextmanager
def scope(
    tag: str,
    *,
    enabled: bool | None = None,
    interval: int | None = None,
    max_captures: int | None = None,
) -> Iterator[None]:
    """
    Define a capture scope. Scopes can be nested; tag_path is built by joining tags.
    """
    if not isinstance(tag, str) or not tag.strip():
        raise ValueError("tag must be a non-empty string")
    if not any(_sanitize_part(p) for p in _split_pathlike(tag)):
        raise ValueError(f"tag resolves to empty path after sanitization: {tag!r}")
    if interval is not None and interval < 1:
        raise ValueError(f"interval must be >= 1, got {interval}")
    if max_captures is not None and max_captures < 0:
        raise ValueError(f"max_captures must be >= 0, got {max_captures}")

    token = _SCOPE_STACK.set(
        [*_SCOPE_STACK.get(), _Scope(tag=tag, enabled=enabled, interval=interval, max_captures=max_captures)]
    )
    try:
        yield
    finally:
        _SCOPE_STACK.reset(token)


def step(n: int = 1) -> int:
    """
    End of a step/tick. Increments the global step counter and finalizes captures for this step.
    Returns the new step value.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    with _STATE.lock:
        for _ in range(n):
            _finalize_step_locked()
            _STATE.step += 1
        return _STATE.step


def dump_np(obj: Any, name: str, *, allow_pickle: bool = False) -> Path | None:
    """
    Dump a numpy ndarray (or array-like) as .npy.
    """
    try:
        import numpy as np  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError("numpy is required for dump_np()") from exc

    if not isinstance(name, str) or not name.strip():
        raise ValueError("name must be a non-empty string")

    with _STATE.lock:
        capture_dir = _get_or_start_capture_dir_locked()
        if capture_dir is None:
            return None

        path = _make_output_path_locked(capture_dir, name=name, ext=".npy")

    arr = np.asarray(obj)

    def _writer(tmp_path: Path) -> None:
        np.save(str(tmp_path), arr, allow_pickle=allow_pickle)

    _atomic_write(path, _writer)
    return path


def dump_torch(obj: Any, name: str) -> Path | None:
    """
    Dump a torch.Tensor as .pt (saved on CPU).
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("name must be a non-empty string")

    try:
        import torch  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError("torch is required for dump_torch() (pip install tencap[torch])") from exc

    if not isinstance(obj, torch.Tensor):  # type: ignore[attr-defined]
        raise TypeError(f"dump_torch() expects torch.Tensor, got {type(obj)!r}")

    tensor = obj.detach().cpu()

    with _STATE.lock:
        capture_dir = _get_or_start_capture_dir_locked()
        if capture_dir is None:
            return None

        path = _make_output_path_locked(capture_dir, name=name, ext=".pt")

    def _writer(tmp_path: Path) -> None:
        torch.save(tensor, str(tmp_path))

    _atomic_write(path, _writer)
    return path


def _finalize_step_locked() -> None:
    current_step = _STATE.step
    for runtime in _STATE.runtimes.values():
        if runtime.active_step == current_step:
            runtime.capture_count += 1
            runtime.active_step = None
            runtime.active_dir = None


def _get_or_start_capture_dir_locked() -> Path | None:
    _refresh_pid_locked()
    config = _STATE.config
    if config.root_dir is None:
        return None

    scope_stack = _SCOPE_STACK.get()
    if not scope_stack:
        return None

    effective_enabled = config.enabled
    effective_interval = config.interval
    effective_max_captures = config.max_captures

    for scope in scope_stack:
        if scope.enabled is not None:
            effective_enabled = scope.enabled
        if scope.interval is not None:
            effective_interval = scope.interval
        if scope.max_captures is not None:
            effective_max_captures = scope.max_captures

    if not effective_enabled:
        return None

    step_idx = _STATE.step
    if effective_interval < 1:
        return None
    if (step_idx % effective_interval) != 0:
        return None

    tag_path = _sanitize_rel_dir(_join_tag_path(scope_stack))
    root_dir = config.root_dir
    runtime = _STATE.runtimes.setdefault((root_dir, tag_path), _Runtime())

    if effective_max_captures is not None and runtime.capture_count >= effective_max_captures:
        return None

    if runtime.active_step == step_idx and runtime.active_dir is not None:
        return runtime.active_dir

    pid_part = f"pid_{_STATE.pid}"
    cap_part = f"cap_{runtime.capture_count:06d}"

    capture_dir = root_dir
    if tag_path:
        capture_dir = capture_dir / tag_path
    capture_dir = capture_dir / pid_part / cap_part
    capture_dir.mkdir(parents=True, exist_ok=True)

    _write_meta_json_once(capture_dir, tag_path=tag_path, pid=_STATE.pid, step=step_idx, capture_count=runtime.capture_count)

    runtime.active_step = step_idx
    runtime.active_dir = capture_dir
    return capture_dir


def _refresh_pid_locked() -> None:
    current_pid = os.getpid()
    if _STATE.pid != current_pid:
        _STATE.pid = current_pid
        _STATE.runtimes.clear()


def _write_meta_json_once(
    capture_dir: Path,
    *,
    tag_path: str,
    pid: int,
    step: int,
    capture_count: int,
) -> None:
    meta_path = capture_dir / "meta.json"
    if meta_path.exists():
        return

    payload = {
        "tag_path": tag_path,
        "pid": pid,
        "step": step,
        "capture_count": capture_count,
        "created_at": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
    }
    meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _make_output_path_locked(capture_dir: Path, *, name: str, ext: str) -> Path:
    config = _STATE.config
    if config.root_dir is None:
        raise RuntimeError("TenCap is not initialized; call setup() first")

    name_path = _sanitize_rel_path(name)
    parent = capture_dir / name_path.parent
    parent.mkdir(parents=True, exist_ok=True)

    base = _sanitize_part(name_path.name)
    if not base:
        raise ValueError(f"invalid name: {name!r}")

    candidate = parent / f"{base}{ext}"
    candidate = _ensure_under_root(config.root_dir, candidate)
    if not candidate.exists():
        return candidate

    for i in range(1, 10_000):
        alt = parent / f"{base}__{i}{ext}"
        alt = _ensure_under_root(config.root_dir, alt)
        if not alt.exists():
            return alt

    raise RuntimeError(f"too many name collisions for {candidate}")


def _atomic_write(path: Path, writer: Callable[[Path], None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=str(path.parent),
        prefix=f".tmp_{path.stem}_",
        suffix=path.suffix,
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)

    try:
        writer(tmp_path)
        os.replace(tmp_path, path)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except FileNotFoundError:
            pass


def _join_tag_path(scope_stack: list[_Scope]) -> str:
    parts: list[str] = []
    for scope in scope_stack:
        parts.extend(_split_pathlike(scope.tag))
    return "/".join(parts)


def _split_pathlike(value: str) -> list[str]:
    value = value.strip()
    if not value:
        return []
    value = value.replace("\\", "/")
    return [p for p in value.split("/") if p]


def _sanitize_part(part: str) -> str:
    part = part.strip()
    part = _INVALID_PART_CHARS_RE.sub("_", part)
    part = _WHITESPACE_RE.sub("_", part)
    part = _DUP_UNDERSCORE_RE.sub("_", part)
    part = part.strip(" .")
    if part in {"", ".", ".."}:
        return ""
    return part


def _sanitize_rel_dir(value: str) -> str:
    if not value:
        return ""
    parts = [_sanitize_part(p) for p in _split_pathlike(value)]
    parts = [p for p in parts if p]
    return "/".join(parts)


def _sanitize_rel_path(value: str) -> Path:
    parts = [_sanitize_part(p) for p in _split_pathlike(value)]
    parts = [p for p in parts if p]
    if not parts:
        raise ValueError(f"invalid path-like value: {value!r}")
    return Path(*parts)


def _ensure_under_root(root_dir: Path, path: Path) -> Path:
    try:
        root = root_dir.resolve()
        resolved = path.resolve()
    except FileNotFoundError:
        root = root_dir.resolve()
        resolved = path
    if hasattr(resolved, "is_relative_to"):
        if not resolved.is_relative_to(root):
            raise ValueError(f"refusing to write outside root_dir: {resolved} (root={root})")
    else:  # pragma: no cover
        if str(resolved).startswith(str(root) + os.sep):
            return path
        raise ValueError(f"refusing to write outside root_dir: {resolved} (root={root})")
    return path
