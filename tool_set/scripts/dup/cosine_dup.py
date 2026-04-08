#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import sys
import threading
import time
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
from tqdm import tqdm

# Try faiss; fall back to NumPy if unavailable
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None

# UTF-8 stdout
sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
from uni_toolcall.secrets import get_openai_compatible_key


# ==== Constants (paths relative to repo root) ====
WORKSPACE_ROOT = _REPO_ROOT
APIS_DIR = _REPO_ROOT / "tool_set" / "apis" / "apis_nonull"
OUTPUT_DIR = _REPO_ROOT / "tool_set" / "apis" / "apis_cosdup"
EMBED_MODEL = "Qwen/Qwen3-Embedding-8B"
EMBED_URL = "https://api.siliconflow.cn/v1/embeddings"
EMBED_SERVER_URL = "http://127.0.0.1:8025/v1/embeddings"
PROTECTED_FILES = {"train_set_tool_dedup.json", "test_set_tool_dedup.json"}
SKIP_FILES = {"test_set_tool.json", "toolret_tool.json"}
SIM_THRESHOLD = 0.9
MAX_RETRIES = 5
RETRY_BACKOFF = 2.0
CHUNK_REBUILD_INTERVAL = 256  # Rebuild matrix cache every N new rows
REMOVED_CASES_MD = "removed_tools_cases.md"
STATS_JSON = "dedup_stats.json"
MAPPING_JSON = "dedup_mapping.json"
REQUEST_SLEEP_SECONDS = 0.05
CACHE_SAVE_INTERVAL = 10
EMBED_CACHE_DIR = Path("/home/yijuan_liang/10.12Tool_Set/tool_set/tool_embedding_cache")
CACHE_SUFFIX = ".embeddings.json"
MAX_WORKERS = 20  # Default worker threads
REQUEST_TIMEOUT_SECONDS = 60
BATCH_SIZE = 64  # Default batch size (vLLM often 32-128)


@dataclass
class ToolRecord:
    file: str
    key: str
    name: str
    description: str
    combined_text: str
    is_protected: bool
    raw_data: Any
    embedding: Optional[np.ndarray] = None
    container_kind: str = "root"
    container_key: Optional[str] = None
    container_index: Optional[int] = None


@dataclass
class FileContext:
    structure_type: str  # "dict" or "list"
    record_indices: List[int] = field(default_factory=list)
    dict_keys_order: List[str] = field(default_factory=list)
    dict_value_types: Dict[str, str] = field(default_factory=dict)
    dict_single_index: Dict[str, int] = field(default_factory=dict)
    dict_list_indices: Dict[str, List[int]] = field(default_factory=dict)


@dataclass(frozen=True)
class RequestConfig:
    mode: str  # "api" or "server"
    url: str
    headers: Dict[str, str]
    timeout: int = REQUEST_TIMEOUT_SECONDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deduplicate tool sets using embeddings"
    )
    parser.add_argument(
        "--apis-dir",
        type=Path,
        default=APIS_DIR,
        help="Directory of JSON files to deduplicate (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory (default: %(default)s)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=SIM_THRESHOLD,
        help="Cosine similarity threshold in (0, 1] (default: %(default)s)",
    )
    parser.add_argument(
        "--rebuild-interval",
        type=int,
        default=CHUNK_REBUILD_INTERVAL,
        help="Rebuild matrix cache every N new vectors (default: %(default)s)",
    )
    parser.add_argument(
        "--sleep-interval",
        type=float,
        default=REQUEST_SLEEP_SECONDS,
        help="Sleep seconds between API calls on cache miss (default: %(default)s)",
    )
    parser.add_argument(
        "--cache-save-interval",
        type=int,
        default=CACHE_SAVE_INTERVAL,
        help="Flush cache after this many new embeddings (<=0: only at end)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=EMBED_CACHE_DIR,
        help="Embedding cache directory (default: %(default)s)",
    )
    parser.add_argument(
        "--mode",
        choices=("api", "server"),
        default="api",
        help="Embedding mode: api or server (default: %(default)s)",
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default=EMBED_SERVER_URL,
        help="OpenAI-compatible URL for local model (default: %(default)s)",
    )
    parser.add_argument(
        "--server-api-key",
        type=str,
        default="EMPTY",
        help="API key for local server (default: %(default)s; match vLLM --api-key)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=MAX_WORKERS,
        help="Thread pool size (default: %(default)s; suggest 10-50)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only write stats/mapping; do not write *_cosdup.json"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size for requests, suggest 32-128 (default: %(default)s)",
    )
    parser.add_argument(
        "--enable-batch",
        action="store_true",
        default=True,
        help="Use batch mode (faster; default on)",
    )
    parser.add_argument(
        "--disable-batch",
        action="store_false",
        dest="enable_batch",
        help="Disable batch mode (use threaded single requests)",
    )
    return parser.parse_args()


# ==== Helpers ====
def ensure_output_dir(path: Path) -> None:
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)


def load_embedding_caches(
    cache_dir: Path, file_names: List[str]
) -> Tuple[Dict[str, List[float]], Dict[str, Dict[str, List[float]]], Dict[str, bool]]:
    """Load per-file caches and merge into a global view."""
    global_cache: Dict[str, List[float]] = {}
    per_file_cache: Dict[str, Dict[str, List[float]]] = {}
    dirty_flags: Dict[str, bool] = {}

    for file_name in file_names:
        cache_dict: Dict[str, List[float]] = {}
        cache_path = cache_dir / f"{file_name}{CACHE_SUFFIX}"
        temp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        
        # If .tmp exists, previous save may be incomplete; recover
        if temp_path.exists() and not cache_path.exists():
            print(f"Found incomplete cache file {temp_path.name}; attempting recovery...")
            try:
                temp_path.replace(cache_path)
            except Exception as exc:  # noqa: BLE001
                print(f"Warning: Failed to recover temp cache: {exc}")
        
        if cache_path.exists():
            try:
                with cache_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    for key, value in data.items():
                        if (
                            isinstance(key, str)
                            and isinstance(value, list)
                            and all(isinstance(x, (float, int)) for x in value)
                        ):
                            cache_dict[key] = [float(x) for x in value]
                    if len(cache_dict) > 0:
                        print(f"Loaded from cache {file_name}: {len(cache_dict)} embeddings")
                    else:
                        print(f"Cache file {file_name} exists but empty; will regenerate")
            except Exception as exc:  # noqa: BLE001
                print(f"Warning: Read cache {cache_path.name} failed; will regenerate. Reason: {exc}")
        else:
            # Skip per-file message when missing; summary at end
            pass
        per_file_cache[file_name] = cache_dict
        dirty_flags[file_name] = False
        global_cache.update(cache_dict)

    total_cached = len(global_cache)
    # Files with/without cache
    files_with_cache = [f for f, cache in per_file_cache.items() if len(cache) > 0]
    files_without_cache = [f for f in file_names if f not in files_with_cache]
    
    if total_cached > 0:
        print(f"Loaded total {total_cached} cached embeddings")
    if files_without_cache:
        print(f"Files without cache ({len(files_without_cache)}): {', '.join(files_without_cache[:5])}" + 
              (f" ..." if len(files_without_cache) > 5 else ""))
    print()
    
    return global_cache, per_file_cache, dirty_flags


def save_embedding_caches(
    cache_dir: Path,
    per_file_cache: Dict[str, Dict[str, List[float]]],
    dirty_flags: Dict[str, bool],
    force: bool = False,
) -> None:
    """Write caches to disk (thread-safe; saves all dirty files)."""
    ensure_output_dir(cache_dir)
    saved_count = 0
    for file_name, cache_dict in per_file_cache.items():
        if not force and not dirty_flags.get(file_name):
            continue
        cache_path = cache_dir / f"{file_name}{CACHE_SUFFIX}"
        try:
            # Temp file then atomic rename
            temp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
            with temp_path.open("w", encoding="utf-8") as f:
                json.dump(cache_dict, f, ensure_ascii=False)
            # Atomic replace
            temp_path.replace(cache_path)
            dirty_flags[file_name] = False
            saved_count += 1
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: failed to save cache {cache_path.name}: {exc}")
    if saved_count > 0:
        print(f"Saved caches for {saved_count} file(s)")


def request_embedding(text: str, config: RequestConfig) -> List[float]:
    """Single-text embedding request (legacy compat)."""
    results = request_embeddings_batch([text], config)
    return results[0]


def request_embeddings_batch(texts: List[str], config: RequestConfig) -> List[List[float]]:
    """
    Batch embedding requests (vLLM OpenAI API supports batch input)
    
    Args:
        texts: list of strings
        config: request config
    
    Returns:
        Embeddings in same order as texts
    """
    if not texts:
        return []
    
    headers = dict(config.headers)
    # vLLM OpenAI API: input may be a list of strings
    payload = {"model": EMBED_MODEL, "input": texts}

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                config.url, headers=headers, json=payload, timeout=config.timeout
            )
            response.raise_for_status()
            data = response.json()
            
            # Response: data["data"] is a list of embedding items
            if "data" not in data or not isinstance(data["data"], list):
                raise ValueError("Invalid response: missing data or not a list")
            
            embeddings = []
            for item in data["data"]:
                if "embedding" not in item:
                    raise ValueError("Invalid response: missing embedding field")
                embedding = item["embedding"]
                if not isinstance(embedding, list):
                    raise ValueError("embedding field is not a list")
                embeddings.append([float(x) for x in embedding])
            
            # Verify count matches
            if len(embeddings) != len(texts):
                raise ValueError(
                    f"Embedding count mismatch: got {len(embeddings)}, expected {len(texts)}"
                )
            
            return embeddings
        except Exception as exc:  # noqa: BLE001
            wait_time = RETRY_BACKOFF** (attempt - 1)
            print(
                f"Warning: batch embedding failed (attempt {attempt}): {exc}. "
                f"{'stopping' if attempt == MAX_RETRIES else f'retry in {wait_time:.1f}s'}"
            )
            if attempt == MAX_RETRIES:
                raise
            time.sleep(wait_time)
    raise RuntimeError("Failed to obtain embeddings")


def get_embedding(
    text: str,
    file_name: str,
    global_cache: Dict[str, List[float]],
    per_file_cache: Dict[str, Dict[str, List[float]]],
    dirty_flags: Dict[str, bool],
    cache_lock: Optional[threading.Lock] = None,
    request_config: Optional[RequestConfig] = None,
) -> Tuple[np.ndarray, bool]:
    """Return normalized embedding and cache-hit flag (thread-safe).

    Per-file caches; key is text content.
    """
    if request_config is None:
        raise ValueError("request_config is required for embeddings")
    
    # Text key; per-file caches
    cache_key = text
    
    # Check this file's cache first
    file_cache = per_file_cache.get(file_name, {})
    if cache_key in file_cache:
        vec = np.asarray(file_cache[cache_key], dtype=np.float32)
        return normalize_vector(vec), True
    
    # Then global cache
    if cache_key in global_cache:
        vec = np.asarray(global_cache[cache_key], dtype=np.float32)
        # Lock when updating per-file cache
        if cache_lock:
            with cache_lock:
                file_cache = per_file_cache.setdefault(file_name, {})
                if cache_key not in file_cache:
                    file_cache[cache_key] = global_cache[cache_key]
                    dirty_flags[file_name] = True
        else:
            file_cache = per_file_cache.setdefault(file_name, {})
            if cache_key not in file_cache:
                file_cache[cache_key] = global_cache[cache_key]
                dirty_flags[file_name] = True
        return normalize_vector(vec), True

    # API path: double-check after lock
    if cache_lock:
        with cache_lock:
            # Recheck file cache
            file_cache = per_file_cache.get(file_name, {})
            if cache_key in file_cache:
                vec = np.asarray(file_cache[cache_key], dtype=np.float32)
                return normalize_vector(vec), True
            # Recheck global cache
            if cache_key in global_cache:
                vec = np.asarray(global_cache[cache_key], dtype=np.float32)
                file_cache = per_file_cache.setdefault(file_name, {})
                if cache_key not in file_cache:
                    file_cache[cache_key] = global_cache[cache_key]
                    dirty_flags[file_name] = True
                return normalize_vector(vec), True

    # API call outside lock
    if request_config is None:
        raise ValueError("request_config is required for embeddings")

    raw_embedding = request_embedding(text, request_config)
    
    # Lock; update per-file cache
    if cache_lock:
        with cache_lock:
            # Double-check to avoid duplicate fetch
            file_cache = per_file_cache.get(file_name, {})
            if cache_key not in file_cache:
                # Update file cache first
                file_cache = per_file_cache.setdefault(file_name, {})
                file_cache[cache_key] = raw_embedding
                # Then global cache
                global_cache[cache_key] = raw_embedding
                dirty_flags[file_name] = True
            else:
                # Another thread filled cache
                raw_embedding = file_cache[cache_key]
    else:
        file_cache = per_file_cache.setdefault(file_name, {})
        file_cache[cache_key] = raw_embedding
        global_cache[cache_key] = raw_embedding
        dirty_flags[file_name] = True
    
    vec = np.asarray(raw_embedding, dtype=np.float32)
    return normalize_vector(vec), False


def normalize_vector(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return vec
    return vec / norm


def format_tool_text(name: str, description: str) -> str:
    """Format tool text for consistency (used as cache key)."""
    name = name.strip() if isinstance(name, str) else ""
    description = description.strip() if isinstance(description, str) else ""
    # Unified format: name + newline + description, then strip ends
    result = f"{name}\n{description}".strip()
    # Normalize newlines: collapse consecutive newlines to one
    import re
    result = re.sub(r'\n+', '\n', result)
    return result.strip()


def load_tools_from_file(
    file_path: Path,
) -> Any:
    with file_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def rebuild_matrix_cache(embeddings: List[np.ndarray]) -> np.ndarray:
    """Rebuild matrix cache (NumPy fallback)."""
    if not embeddings:
        return np.empty((0, 0), dtype=np.float32)
    return np.vstack(embeddings).astype(np.float32, copy=False)


def build_faiss_index(embeddings: List[np.ndarray]) -> Optional[Any]:
    """Build a Faiss index if available."""
    if not FAISS_AVAILABLE or not embeddings:
        return None
    
    # Stack all embeddings into a matrix
    embeddings_matrix = np.vstack(embeddings).astype(np.float32)
    dim = embeddings_matrix.shape[1]
    
    # Inner-product index for cosine similarity (vectors are normalized)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings_matrix)
    
    return index




def escape_markdown(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", "\\n")


def process_embeddings_batch(
    records: List[ToolRecord],
    global_cache: Dict[str, List[float]],
    per_file_cache: Dict[str, Dict[str, List[float]]],
    dirty_flags: Dict[str, bool],
    request_config: RequestConfig,
    batch_size: int,
    cache_lock: Optional[threading.Lock] = None,
) -> Tuple[int, int]:
    """
    Batch-process embeddings (efficient mode).

    Args:
        records: Tool records
        global_cache: Global cache
        per_file_cache: Per-file cache
        dirty_flags: Dirty flags for cache files
        request_config: Request config
        batch_size: Batch size
        cache_lock: Optional cache lock

    Returns:
        (cache_hits, newly_computed)
    """
    # Step 1: check cache and collect texts to request
    texts_to_request: List[Tuple[int, str, str]] = []  # (record_idx, text, file_name)
    cache_hits = 0
    
    for idx, record in enumerate(records):
        cache_key = record.combined_text
        file_cache = per_file_cache.get(record.file, {})
        
        # Check cache
        if cache_key in file_cache:
            vec = np.asarray(file_cache[cache_key], dtype=np.float32)
            record.embedding = normalize_vector(vec)
            cache_hits += 1
        elif cache_key in global_cache:
            vec = np.asarray(global_cache[cache_key], dtype=np.float32)
            record.embedding = normalize_vector(vec)
            # Update per_file_cache
            if cache_lock:
                with cache_lock:
                    file_cache = per_file_cache.setdefault(record.file, {})
                    if cache_key not in file_cache:
                        file_cache[cache_key] = global_cache[cache_key]
                        dirty_flags[record.file] = True
            else:
                file_cache = per_file_cache.setdefault(record.file, {})
                if cache_key not in file_cache:
                    file_cache[cache_key] = global_cache[cache_key]
                    dirty_flags[record.file] = True
            cache_hits += 1
        else:
            # Need request — check for similar key (format mismatch)
            normalized_key = cache_key.strip()
            found_similar = False
            
            # Check file cache for similar key first
            if len(file_cache) > 0:
                for cached_key in file_cache.keys():
                    if cached_key.strip() == normalized_key:
                        vec = np.asarray(file_cache[cached_key], dtype=np.float32)
                        record.embedding = normalize_vector(vec)
                        file_cache[cache_key] = file_cache[cached_key]
                        cache_hits += 1
                        found_similar = True
                        break
            
            if not found_similar and len(global_cache) > 0:
                for cached_key in global_cache.keys():
                    if cached_key.strip() == normalized_key:
                        vec = np.asarray(global_cache[cached_key], dtype=np.float32)
                        record.embedding = normalize_vector(vec)
                        file_cache = per_file_cache.setdefault(record.file, {})
                        file_cache[cache_key] = global_cache[cached_key]
                        global_cache[cache_key] = global_cache[cached_key]
                        dirty_flags[record.file] = True
                        cache_hits += 1
                        found_similar = True
                        break
            
            if found_similar:
                continue
            
            texts_to_request.append((idx, cache_key, record.file))
    
    if not texts_to_request:
        return cache_hits, 0
    
    # Step 2: batch requests
    newly_computed = 0
    total_batches = (len(texts_to_request) + batch_size - 1) // batch_size
    
    # Debug: per-file cache stats
    file_cache_stats: Dict[str, Dict[str, int]] = {}
    for idx, cache_key, file_name in texts_to_request[:10]:  # first 10 only
        if file_name not in file_cache_stats:
            file_cache_stats[file_name] = {"total": 0, "cached": 0}
        file_cache_stats[file_name]["total"] += 1
        file_cache = per_file_cache.get(file_name, {})
        if len(file_cache) > 0:
            file_cache_stats[file_name]["cached"] = len(file_cache)
    
    if file_cache_stats:
        print(f"\nDebug — cache check:")
        for file_name, stats in file_cache_stats.items():
            print(f"  File {file_name}: need {stats['total']} requests, cache has {stats['cached']} embeddings")
        if len(texts_to_request) > 10:
            print(f"  ... {len(texts_to_request) - 10} more to request")
    
    print(f"\nRequesting {len(texts_to_request)} embeddings in {total_batches} batches (batch size: {batch_size})")
    
    for batch_idx in range(0, len(texts_to_request), batch_size):
        batch_items = texts_to_request[batch_idx : batch_idx + batch_size]
        batch_texts = [item[1] for item in batch_items]
        
        try:
            batch_embeddings = request_embeddings_batch(batch_texts, request_config)
        except Exception as exc:
            print(f"\nError: batch request failed (batch {batch_idx // batch_size + 1}/{total_batches}): {exc}")
            raise
        
        # Step 3: update cache and records
        if cache_lock:
            with cache_lock:
                for (record_idx, cache_key, file_name), embedding in zip(batch_items, batch_embeddings):
                    global_cache[cache_key] = embedding
                    file_cache = per_file_cache.setdefault(file_name, {})
                    file_cache[cache_key] = embedding
                    dirty_flags[file_name] = True
                    vec = np.asarray(embedding, dtype=np.float32)
                    records[record_idx].embedding = normalize_vector(vec)
                    newly_computed += 1
        else:
            for (record_idx, cache_key, file_name), embedding in zip(batch_items, batch_embeddings):
                global_cache[cache_key] = embedding
                file_cache = per_file_cache.setdefault(file_name, {})
                file_cache[cache_key] = embedding
                dirty_flags[file_name] = True
                vec = np.asarray(embedding, dtype=np.float32)
                records[record_idx].embedding = normalize_vector(vec)
                newly_computed += 1
        
        if (batch_idx // batch_size + 1) % 10 == 0:
            print(f"  Processed {batch_idx + len(batch_items)}/{len(texts_to_request)} embeddings")
    
    return cache_hits, newly_computed


# ==== Main ====
def main() -> None:
    args = parse_args()

    apis_dir = Path(args.apis_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    sim_threshold = float(args.threshold)
    chunk_rebuild_interval = int(args.rebuild_interval)
    sleep_interval = float(args.sleep_interval)
    cache_save_interval = int(args.cache_save_interval)
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    mode = str(args.mode).lower()
    server_url = str(args.server_url).strip()
    server_api_key = str(args.server_api_key).strip()
    max_workers = int(args.max_workers)
    dry_run = bool(args.dry_run)
    batch_size = int(args.batch_size)
    enable_batch = getattr(args, 'enable_batch', True)

    if not 0.0 < sim_threshold <= 1.0:
        raise ValueError("Cosine similarity threshold must be in (0, 1]")
    if chunk_rebuild_interval <= 0:
        raise ValueError("Cache rebuild interval must be a positive integer")
    if sleep_interval < 0:
        raise ValueError("Request sleep interval cannot be negative")
    if cache_save_interval < 0:
        raise ValueError("Cache save interval cannot be negative")
    if mode not in {"api", "server"}:
        raise ValueError("mode must be api or server")
    if mode == "server" and not server_url:
        raise ValueError("server mode requires --server-url")
    if max_workers <= 0:
        raise ValueError("Thread pool size must be a positive integer")
    if batch_size <= 0:
        raise ValueError("Batch size must be a positive integer")
    if cache_dir.exists() and not cache_dir.is_dir():
        raise NotADirectoryError(f"Cache path is not a directory: {cache_dir}")
    ensure_output_dir(cache_dir)
    if not apis_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {apis_dir}")
    if not apis_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {apis_dir}")

    api_key = get_openai_compatible_key()
    if mode == "api" and not api_key:
        raise ValueError(
            "api mode requires SILICONFLOW_API_KEY or OPENAI_API_KEY"
        )
    # API config
    api_headers = {
        "Authorization": f"Bearer {api_key or ''}",
        "Content-Type": "application/json",
    }
    api_config = RequestConfig(
        mode="api",
        url=EMBED_URL,
        headers=api_headers,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    
    # Server config
    server_headers = {
        "Authorization": f"Bearer {server_api_key}",
        "Content-Type": "application/json",
    }
    server_config = RequestConfig(
        mode="server",
        url=server_url or EMBED_SERVER_URL,
        headers=server_headers,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    
    # Select config by mode
    if mode == "api":
        request_config = api_config
    else:  # server
        request_config = server_config

    ensure_output_dir(output_dir)

    json_files = sorted(
        [
            p
            for p in apis_dir.glob("*.json")
            if p.is_file() and p.name not in SKIP_FILES
        ],
        key=lambda p: (p.name not in PROTECTED_FILES, p.name.lower()),
    )

    if not json_files:
        print(f"No JSON files found under {apis_dir}; exiting.")
        return

    file_names = [p.name for p in json_files]
    embedding_cache, per_file_cache, cache_dirty = load_embedding_caches(
        cache_dir, file_names
    )
    
    print(f"Cache load complete:")
    print(f"  Global cache: {len(embedding_cache)} embeddings")
    for file_name, file_cache in per_file_cache.items():
        if len(file_cache) > 0:
            print(f"  {file_name}: {len(file_cache)} embeddings")
    print()

    print(f"Found {len(json_files)} JSON files; loading tools...\n")

    records: List[ToolRecord] = []
    file_contexts: Dict[str, FileContext] = {}
    file_order: List[str] = []
    stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "kept": 0, "removed": 0})

    # Phase 1: load tools and prepare text
    for file_path in json_files:
        file_name = file_path.name
        is_protected = file_name in PROTECTED_FILES
        data = load_tools_from_file(file_path)
 
        file_order.append(file_name)
        context = FileContext(structure_type="dict" if isinstance(data, dict) else "list")
        file_contexts[file_name] = context
        stats[file_name]["total"] = 0

        def register_record(
            key: str,
            tool_obj: Any,
            container_kind: str,
            container_key: Optional[str] = None,
            container_index: Optional[int] = None,
        ) -> int:
            if isinstance(tool_obj, dict):
                name = tool_obj.get("name", "")
                description = tool_obj.get("description", "")
            else:
                name = ""
                description = ""

            combined_text = format_tool_text(name, description)
            if not combined_text:
                combined_text = name or description or f"{file_name}:{key}"

            record = ToolRecord(
                file=file_name,
                key=str(key),
                name=name,
                description=description,
                combined_text=combined_text,
                is_protected=is_protected,
                raw_data=tool_obj,
                container_kind=container_kind,
                container_key=container_key,
                container_index=container_index,
            )
            records.append(record)
            record_index = len(records) - 1
            context.record_indices.append(record_index)
            stats[file_name]["total"] += 1
            return record_index

        if isinstance(data, dict):
            for top_key, value in data.items():
                context.dict_keys_order.append(top_key)
                if isinstance(value, dict):
                    context.dict_value_types[top_key] = "dict"
                    idx = register_record(top_key, value, container_kind="dict", container_key=top_key)
                    context.dict_single_index[top_key] = idx
                elif isinstance(value, list):
                    context.dict_value_types[top_key] = "list"
                    list_indices: List[int] = []
                    for pos, element in enumerate(value):
                        idx = register_record(
                            f"{top_key}[{pos}]",
                            element,
                            container_kind="dict_list",
                            container_key=top_key,
                            container_index=pos,
                        )
                        list_indices.append(idx)
                    context.dict_list_indices[top_key] = list_indices
                else:
                    context.dict_value_types[top_key] = "raw"
                    idx = register_record(top_key, value, container_kind="raw", container_key=top_key)
                    context.dict_single_index[top_key] = idx
        elif isinstance(data, list):
            for pos, element in enumerate(data):
                register_record(str(pos), element, container_kind="list", container_index=pos)
        else:
            raise TypeError(f"Unsupported JSON structure: {type(data)} in {file_path}")

        print(
            f"Loaded file: {file_name} (tools: {stats[file_name]['total']})"
            f"{' [protected]' if is_protected else ''}"
        )

    total_tools = len(records)
    if total_tools == 0:
        print("No tools loaded; exiting.")
        return

    print(f"\nLoaded {total_tools} tools. Generating/loading embeddings...\n")
    
    if enable_batch:
        print(f"Batch mode (batch size: {batch_size})\n")
        print(f"Batch mode (mode={request_config.mode}, url={request_config.url})\n")
        cache_hits, newly_computed = process_embeddings_batch(
            records,
            embedding_cache,
            per_file_cache,
            cache_dirty,
            request_config,
            batch_size,
            cache_lock=None,
        )
        total_hits = cache_hits
        
        if newly_computed > 0:
            print(f"\nSaving cache...")
            save_embedding_caches(cache_dir, per_file_cache, cache_dirty, force=True)
        
        print(
            f"\nEmbeddings done: cache hits {total_hits}, newly computed {newly_computed}.\n"
        )
    else:
        print(
            f"Using {max_workers} worker threads (mode={request_config.mode}, url={request_config.url})\n"
        )

        cache_lock = threading.Lock()
        newly_computed_lock = threading.Lock()
        save_lock = threading.Lock()
        
        newly_computed = 0
        newly_since_last_save = 0
        
        def process_record(idx_and_record: Tuple[int, ToolRecord]) -> Tuple[int, bool, Optional[Exception]]:
            """Process one record's embedding; returns (record_index, from_cache, error)."""
            idx, record = idx_and_record
            try:
                embedding, from_cache = get_embedding(
                    record.combined_text,
                    record.file,
                    embedding_cache,
                    per_file_cache,
                    cache_dirty,
                    cache_lock,
                    request_config,
                )
                record.embedding = embedding
                
                if not from_cache:
                    should_save = False
                    with newly_computed_lock:
                        nonlocal newly_computed, newly_since_last_save
                        newly_computed += 1
                        newly_since_last_save += 1
                        # Decide whether to flush cache to disk
                        if cache_save_interval > 0 and newly_since_last_save >= cache_save_interval:
                            should_save = True
                            newly_since_last_save = 0
                    
                    # Dedicated save lock so only one thread writes at a time
                    # When saving, flush all dirty cache files
                    if should_save:
                        with save_lock:
                            # Re-acquire cache lock inside save lock so the flush includes all embeddings computed so far
                            with cache_lock:
                                # Persist all files marked dirty
                                # Note: this saves all dirty files at this moment, possibly including embeddings just computed by other threads
                                save_embedding_caches(cache_dir, per_file_cache, cache_dirty, force=False)
                    
                    if sleep_interval > 0:
                        time.sleep(sleep_interval)
                
                return idx, from_cache, None
            except Exception as exc:  # noqa: BLE001
                return idx, False, exc

        # Process with a thread pool
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks with index and record
            future_to_record = {
                executor.submit(process_record, (idx, record)): (idx, record)
                for idx, record in enumerate(records)
            }
            
            # tqdm progress bar
            with tqdm(total=len(records), desc="Embedding", unit="tool") as pbar:
                for future in as_completed(future_to_record):
                    idx, record = future_to_record[future]
                    try:
                        record_idx, from_cache, error = future.result()
                        if error is not None:
                            print(
                                f"\nError: failed to get embedding for tool {record.file}#{record.key}: {error}"
                            )
                            raise error
                    except Exception as exc:  # noqa: BLE001
                        print(f"\nError: exception while processing tool {record.file}#{record.key}: {exc}")
                        raise
                    pbar.update(1)

        print(
            f"\nEmbeddings done: cache hits {total_tools - newly_computed}, newly computed {newly_computed}.\n"
        )

    # Phase 2: global cosine deduplication
    print("Starting global cosine-similarity deduplication...\n")
    if FAISS_AVAILABLE:
        print("Using Faiss for similarity (faiss installed)\n")
    else:
        print("Using NumPy matmul for similarity (faiss not installed; install faiss for better performance)\n")

    kept_records: List[int] = []
    kept_embeddings: List[np.ndarray] = []
    faiss_index = None
    matrix_cache = np.empty((0, 0), dtype=np.float32)
    last_rebuild_size = 0

    duplicate_of: Dict[int, Dict[str, Any]] = {}
    duplicate_groups: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    removed_records: List[Dict[str, Any]] = []

    for idx, record in enumerate(tqdm(records, desc="Dedup", unit="tool")):
        if record.embedding is None:
            raise RuntimeError(f"Record {record.file}#{record.key} is missing embedding.")

        embedding = record.embedding
        best_sim = -1.0
        best_rep_position = -1
        best_rep_index = -1

        if kept_embeddings:
            # Rebuild index/cache if needed
            needs_rebuild = (
                len(kept_embeddings) - last_rebuild_size >= chunk_rebuild_interval
                or (FAISS_AVAILABLE and faiss_index is None)
                or (not FAISS_AVAILABLE and (matrix_cache.size == 0 or matrix_cache.shape[0] != len(kept_embeddings)))
            )
            
            if needs_rebuild:
                if FAISS_AVAILABLE:
                    # Faiss in memory only; index not persisted
                    faiss_index = build_faiss_index(kept_embeddings)
                    last_rebuild_size = len(kept_embeddings)
                else:
                    # NumPy fallback
                    matrix_cache = rebuild_matrix_cache(kept_embeddings)
                    last_rebuild_size = len(kept_embeddings)

            # Compute similarity
            if FAISS_AVAILABLE and faiss_index is not None:
                # Faiss top-1 search
                query_vector = embedding.reshape(1, -1).astype(np.float32)
                scores, indices = faiss_index.search(query_vector, 1)
                if scores.size > 0 and scores[0][0] > 0:
                    best_rep_position = int(indices[0][0])
                    best_sim = float(scores[0][0])
                    best_rep_index = kept_records[best_rep_position]
            elif matrix_cache.size > 0:
                # NumPy matmul
                sims = matrix_cache @ embedding
                best_rep_position = int(np.argmax(sims))
                best_sim = float(sims[best_rep_position])
                best_rep_index = kept_records[best_rep_position]

        # Dedup rules:
        # - Tools in protected files (train_set_tool_dedup.json, test_set_tool_dedup.json):
        #   1. Participate in similarity (added to kept_embeddings for later comparisons)
        #   2. Never removed by dedup (even if very similar to another tool)
        # - Tools in non-protected files: removed if similarity >= threshold
        if not record.is_protected and best_sim >= sim_threshold:
            # Non-protected tool above threshold: remove
            representative = records[best_rep_index]
            stats[record.file]["removed"] += 1
            duplicate_info = {
                "index": idx,
                "file": record.file,
                "key": record.key,
                "name": record.name,
                "description": record.description,
                "similarity": best_sim,
                "representative_index": best_rep_index,
            }
            duplicate_groups[best_rep_index].append(duplicate_info)
            duplicate_of[idx] = {
                "representative_index": best_rep_index,
                "similarity": best_sim,
            }
            removed_records.append(
                {
                    "removed_file": record.file,
                    "removed_key": record.key,
                    "removed_name": record.name,
                    "reference_file": representative.file,
                    "reference_key": representative.key,
                    "reference_name": representative.name,
                    "similarity": best_sim,
                    "reason": (
                        f"Similarity {best_sim:.4f} to {representative.file}#{representative.key}"
                    ),
                }
            )
        else:
            # Keep tool (protected tools, or non-protected below threshold)
            # Protected tools are still added to kept_embeddings for later similarity checks
            stats[record.file]["kept"] += 1
            kept_records.append(idx)
            kept_embeddings.append(embedding)
            duplicate_groups.setdefault(idx, [])

    total_removed = sum(file_stat["removed"] for file_stat in stats.values())
    total_retained = total_tools - total_removed

    # Faiss index is not persisted; in-memory similarity only

    kept_set = set(kept_records)

    # Phase 3: write deduplicated JSON
    for file_name in file_order:
        context = file_contexts[file_name]
        source_path = apis_dir / file_name
        output_file = output_dir / f"{source_path.stem}_cosdup.json"
 
        if dry_run:
            print(
                f"Done {file_name}: kept {stats[file_name]['kept']} removed {stats[file_name]['removed']}"
                " [dry-run: no files written]"
            )
            continue

        if context.structure_type == "dict":
            ordered_dict = OrderedDict()
            for top_key in context.dict_keys_order:
                value_type = context.dict_value_types.get(top_key)
                if value_type == "dict":
                    idx = context.dict_single_index.get(top_key)
                    if idx is not None and idx in kept_set:
                        ordered_dict[top_key] = records[idx].raw_data
                elif value_type == "list":
                    indices = context.dict_list_indices.get(top_key, [])
                    filtered = [records[idx].raw_data for idx in indices if idx in kept_set]
                    ordered_dict[top_key] = filtered
                else:  # raw or other
                    idx = context.dict_single_index.get(top_key)
                    if idx is not None and idx in kept_set:
                        ordered_dict[top_key] = records[idx].raw_data
            with output_file.open("w", encoding="utf-8") as f:
                json.dump(ordered_dict, f, ensure_ascii=False, indent=2)
        else:
            filtered_list = [
                records[idx].raw_data for idx in context.record_indices if idx in kept_set
            ]
            with output_file.open("w", encoding="utf-8") as f:
                json.dump(filtered_list, f, ensure_ascii=False, indent=2)
 
        print(
            f"Done {file_name}: kept {stats[file_name]['kept']} removed {stats[file_name]['removed']}"
        )

    # Persist embedding caches
    save_embedding_caches(cache_dir, per_file_cache, cache_dirty, force=True)

    # Phase 4: stats / mapping report
    stats_output = {
        "total_files": len(json_files),
        "total_tools": total_tools,
        "total_removed": total_removed,
        "total_retained": total_retained,
        "embedding_newly_computed": newly_computed,
        "embedding_cache_hits": total_tools - newly_computed,
        "threshold": sim_threshold,
        "rebuild_interval": chunk_rebuild_interval,
        "sleep_interval": sleep_interval,
        "cache_save_interval": cache_save_interval,
        "mode": mode,
        "max_workers": max_workers,
        "dry_run": dry_run,
        "input_dir": str(apis_dir),
        "output_dir": str(output_dir),
        "cache_dir": str(cache_dir),
        "per_file": {file: dict(data) for file, data in stats.items()},
    }
    
    # Request URL for this run
    stats_output["request_url"] = request_config.url
    stats_output["use_faiss"] = FAISS_AVAILABLE

    stats_path = output_dir / STATS_JSON
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats_output, f, ensure_ascii=False, indent=2)

    # dedup_mapping.json is not generated

    removed_path = output_dir / REMOVED_CASES_MD
    with removed_path.open("w", encoding="utf-8") as f:
        f.write("# Deduplication removals\n\n")
        f.write(f"- Total tools: {total_tools}\n")
        f.write(f"- Removed: {total_removed}\n")
        f.write(f"- Kept: {total_retained}\n")
        f.write(f"- Threshold: cosine similarity >= {sim_threshold}\n\n")

        if not removed_records:
            f.write("No tools were removed.\n")
        else:
            f.write("| Removed file | Removed key | Removed name | Reference file | Reference key | Reference name | Similarity |\n")
            f.write("| --- | --- | --- | --- | --- | --- | --- |\n")
            for record in removed_records:
                f.write(
                    "| {removed_file} | {removed_key} | {removed_name} | {reference_file} | {reference_key} | {reference_name} | {similarity:.4f} |\n".format(
                        removed_file=escape_markdown(record["removed_file"]),
                        removed_key=escape_markdown(str(record["removed_key"])),
                        removed_name=escape_markdown(record["removed_name"]),
                        reference_file=escape_markdown(record["reference_file"]),
                        reference_key=escape_markdown(str(record["reference_key"])),
                        reference_name=escape_markdown(record["reference_name"]),
                        similarity=record["similarity"],
                    )
                )

    print("=" * 70)
    print("Deduplication summary:")
    print(f"  Run mode: {'Dry-run' if dry_run else 'Write output'}")
    print(f"  Embedding mode: {request_config.mode} ({request_config.url})")
    print(f"  Input directory: {apis_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"  Total tools: {total_tools}")
    print(f"  Removed: {total_removed}")
    print(f"  Kept: {total_retained}")
    if total_tools > 0:
        print(f"  Retention rate: {total_retained / total_tools * 100:.2f}%")
    print(f"\n  Stats: {stats_path}")
    print(f"  Removal cases: {removed_path}")
    print(f"  Embedding cache directory: {cache_dir}")
    
    # Per-file stats
    print("\nPer-file stats:")
    print("-" * 70)
    for file_name in file_order:
        file_stat = stats[file_name]
        print(f"  {file_name}:")
        print(f"    Original: {file_stat['total']}")
        print(f"    Kept: {file_stat['kept']}")
        print(f"    Removed: {file_stat['removed']}")
        if file_stat['total'] > 0:
            print(f"    Retention rate: {file_stat['kept'] / file_stat['total'] * 100:.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()


