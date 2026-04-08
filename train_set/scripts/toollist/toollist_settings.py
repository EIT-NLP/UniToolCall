#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import requests
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
from uni_toolcall.secrets import get_openai_compatible_key

# Try importing faiss; fall back to NumPy if not installed
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None

# Force UTF-8 stdout
sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

# ==== Constants (paths relative to UniToolCall repo root) ====
WORKSPACE_ROOT = _REPO_ROOT
TOOL_POOL_DIR = WORKSPACE_ROOT / "tool_set" / "apis" / "apis_cosdup"
DATA_NOTIME_DIR = WORKSPACE_ROOT / "train_set" / "data" / "data_nonull"
OUTPUT_DIR1 = WORKSPACE_ROOT / "train_set" / "data" / "data_toollist"
EMBED_CACHE_DIR = WORKSPACE_ROOT / "tool_set" / "tool_embedding_cache"
QUERY_EMBEDDING_CACHE_DIR = WORKSPACE_ROOT / "train_set" / "query_embedding_cache"
TOOL_EMBEDDING_PROGRESS_FILE = QUERY_EMBEDDING_CACHE_DIR / "tool_embedding_progress.json"  # tool embedding progress
GT_CACHE_FILE = QUERY_EMBEDDING_CACHE_DIR / "gt_tool_embeddings.json"  # GT tool embedding cache
EMBED_MODEL = "Qwen/Qwen3-Embedding-8B"
EMBED_URL = "https://api.siliconflow.cn/v1/embeddings"
EMBED_SERVER_URL = "http://127.0.0.1:8025/v1/embeddings"
def _api_key() -> str:
    k = get_openai_compatible_key()
    if not k:
        raise RuntimeError(
            "Please set environment variable SILICONFLOW_API_KEY or OPENAI_API_KEY"
        )
    return k


API_KEY = ""  # Runtime uses get_openai_compatible_key(); name kept for legacy ts.API_KEY assignment
CACHE_SUFFIX = ".embeddings.json"
FAISS_INDEX_SUFFIX = ".faiss.index"
FAISS_META_SUFFIX = ".faiss.meta.json"
MAX_R = 5
MAX_RETRIES = 10  # Max retries
RETRY_BACKOFF = 2.0
REQUEST_TIMEOUT_SECONDS = 60
BATCH_SIZE = 512  # Larger batch size
MAX_WORKERS = 2 # Thread pool size
SAVE_BATCH_SIZE = 50  # Save every N items for resume support


def normalize_vector(vec: np.ndarray) -> np.ndarray:
    """L2-normalize a vector."""
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return vec
    return vec / norm


def format_tool_text(name: str, description: str) -> str:
    """Format tool text for stable cache keys (matches cosine_dup.format_tool_text)."""
    name = name.strip() if isinstance(name, str) else ""
    description = description.strip() if isinstance(description, str) else ""
    # Format: name, newline, description; trim
    result = f"{name}\n{description}".strip()
    # Collapse repeated newlines
    import re
    result = re.sub(r'\n+', '\n', result)  # collapse newlines
    return result.strip()


def request_embedding(text: str, mode: str = "api", server_url: Optional[str] = None) -> List[float]:
    """Request embedding for one text."""
    results = request_embeddings_batch([text], mode, server_url)
    return results[0]


def request_embeddings_batch(texts: List[str], mode: str = "api", server_url: Optional[str] = None) -> List[List[float]]:
    """Batch embedding requests.

    Some servers (e.g. vLLM) may not support batches and fall back to single calls.
    HTTP 500 on large batches triggers split-and-retry.
    """
    if not texts:
        return []
    
    if mode == "server":
        url = server_url if server_url else EMBED_SERVER_URL
        headers = {"Authorization": "Bearer EMPTY", "Content-Type": "application/json"}
    else:  # mode == "api"
        url = EMBED_URL
        key = API_KEY.strip() if API_KEY else _api_key()
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    
    # Batch: input may be a list of strings
    payload = {"model": EMBED_MODEL, "input": texts}
    
    # Server mode may fall back to single requests on failure
    use_fallback = (mode == "server")
    
    last_status_code = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
            last_status_code = response.status_code
            
            # Check HTTP status
            if response.status_code != 200:
                print(f"Debug: HTTP status: {response.status_code}")
                print(f"Debug: response body: {response.text[:500]}")
            
            # On HTTP 500 with large batch, split
            if response.status_code == 500 and len(texts) > 1 and attempt <= 3:
                # First 3 attempts: split large batches
                if len(texts) > 32:
                    print(f"Warning: HTTP 500; splitting batch from {len(texts)} into smaller chunks...")
                    # Recursive split
                    mid = len(texts) // 2
                    left_embeddings = request_embeddings_batch(texts[:mid], mode, server_url)
                    right_embeddings = request_embeddings_batch(texts[mid:], mode, server_url)
                    return left_embeddings + right_embeddings
                else:
                    # Small batch: retry
                    response.raise_for_status()
            else:
                response.raise_for_status()
            
            data = response.json()
            
            # Check API error field
            if "error" in data:
                error_msg = data.get("error", {})
                print(f"Debug: API error: {error_msg}")
                raise ValueError(f"API error: {error_msg}")
            
            # Validate response shape
            if "data" not in data:
                # Log full response for debugging
                print(f"Error: response missing 'data' field")
                print(f"Body: {json.dumps(data, ensure_ascii=False, indent=2)[:2000]}")
                raise ValueError("Invalid response: missing data field")
            
            if not isinstance(data["data"], list):
                print(f"Error: data is not a list, type: {type(data['data'])}")
                print(f"Body: {json.dumps(data, ensure_ascii=False, indent=2)[:2000]}")
                raise ValueError("Invalid response: data is not a list")
            
            embeddings = []
            if len(data["data"]) == 0:
                # Empty data[]: log response
                print(f"Error: data['data'] is empty")
                print(f"Requested texts count: {len(texts)}")
                print(f"Request payload: {json.dumps(payload, ensure_ascii=False)[:500]}")
                print(f"Full response: {json.dumps(data, ensure_ascii=False, indent=2)}")
                raise ValueError("Invalid response: data['data'] is empty")
            
            for item in data["data"]:
                if "embedding" not in item:
                    print(f"Debug: item keys: {list(item.keys()) if isinstance(item, dict) else type(item)}")
                    raise ValueError("Invalid response: missing embedding field")
                embedding = item["embedding"]
                if not isinstance(embedding, list):
                    raise ValueError("embedding field is not a list")
                embeddings.append([float(x) for x in embedding])
            
            # Check embedding count matches
            if len(embeddings) != len(texts):
                # Debug mismatch
                print(f"Debug: requested {len(texts)}, got {len(embeddings)}")
                print(f"Debug: len(data['data']): {len(data['data']) if 'data' in data else 'N/A'}")
                print(f"Debug: response (first 1000 chars): {json.dumps(data, ensure_ascii=False)[:1000]}")
                raise ValueError(
                    f"Embedding count mismatch: got {len(embeddings)}, expected {len(texts)}"
                )
            
            return embeddings
        except requests.exceptions.HTTPError as exc:
            # HTTP errors (incl. 500)
            wait_time = RETRY_BACKOFF ** (attempt - 1)
            status_info = f" (HTTP {last_status_code})" if last_status_code else ""
            print(f"Warning: batch embedding failed (attempt {attempt}{status_info}): {exc}. "
                  f"{'stopping' if attempt == MAX_RETRIES else f'retry in {wait_time:.1f}s'}")
            if attempt == MAX_RETRIES:
                raise
            import time
            time.sleep(wait_time)
        except Exception as exc:
            wait_time = RETRY_BACKOFF ** (attempt - 1)
            print(f"Warning: batch embedding failed (attempt {attempt}): {exc}. "
                  f"{'stopping' if attempt == MAX_RETRIES else f'retry in {wait_time:.1f}s'}")
            if attempt == MAX_RETRIES:
                raise
            import time
            time.sleep(wait_time)
    raise RuntimeError("Failed to obtain embeddings")


def load_embedding_cache(cache_file: Path) -> Dict[str, List[float]]:
    """Load embedding cache (supports large files via ijson)."""
    cache = {}
    if cache_file.exists():
        try:
            # Try loading whole file
            with cache_file.open("r", encoding="utf-8") as f:
                cache = json.load(f)
            print(f"Loaded {len(cache)} embeddings from cache")
        except MemoryError:
            # MemoryError: stream with ijson
            print(f"File large; streaming parse: {cache_file.name}")
            try:
                import ijson
                cache = {}
                with cache_file.open("rb") as f:
                    # ijson parse
                    parser = ijson.items(f, "")
                    for key, value in parser:
                        if isinstance(key, str) and isinstance(value, list):
                            cache[key] = value
                print(f"Loaded {len(cache)} embeddings (streamed)")
            except ImportError:
                print(f"Warning: install ijson for large files: pip install ijson")
                print(f"Skipping {cache_file.name}; may need API for embeddings")
            except Exception as exc:
                print(f"Warning: stream parse failed for {cache_file.name}: {exc}")
        except Exception as exc:
            print(f"Warning: failed to load cache {cache_file.name}: {exc}")
    return cache


def get_embedding(text: str, cache: Dict[str, List[float]], mode: str = "api", server_url: Optional[str] = None) -> np.ndarray:
    """Get embedding; prefer cache."""
    if text in cache:
        vec = np.asarray(cache[text], dtype=np.float32)
    else:
        # Rare: fetch via API if missing
        vec = np.asarray(request_embedding(text, mode, server_url), dtype=np.float32)
        cache[text] = vec.tolist()
    return normalize_vector(vec)


def save_query_cache(query_cache: Dict[str, List[float]], cache_file: Path) -> None:
    """Atomically save query embedding cache."""
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        # Temp file then rename
        temp_path = cache_file.with_suffix(cache_file.suffix + ".tmp")
        with temp_path.open("w", encoding="utf-8") as f:
            json.dump(query_cache, f, ensure_ascii=False, indent=2)
        # Atomic replace
        temp_path.replace(cache_file)
    except Exception as exc:
        print(f"Warning: failed to save query cache: {exc}")


def batch_fetch_missing_embeddings(
    missing_texts: List[str],
    cache: Dict[str, List[float]],
    progress_file: Path,
    mode: str = "api",
    server_url: Optional[str] = None,
    batch_size: int = BATCH_SIZE,
    gt_cache: Optional[Dict[str, List[float]]] = None,
    gt_tool_keys: Optional[Set[str]] = None,
    gt_cache_file: Optional[Path] = None
) -> None:
    """
    Fetch missing embeddings concurrently; supports resume.

    Args:
        missing_texts: texts needing embeddings
        cache: embedding cache (updated in place)
        progress_file: progress path for resume
        mode: "api" or "server"
        server_url: server URL (server mode only)
        batch_size: batch size
        gt_cache: optional GT tool cache (updated when GT embeddings are fetched)
        gt_tool_keys: optional set of GT tool keys
        gt_cache_file: optional path to persist GT cache
    """
    if not missing_texts:
        return
    
    # Load progress file if present
    progress = {}
    if progress_file.exists():
        try:
            with progress_file.open("r", encoding="utf-8") as f:
                progress = json.load(f)
            # Restore finished embeddings from progress
            for text, embedding in progress.items():
                if text not in cache:
                    cache[text] = embedding
                    # Update GT cache for GT tools
                    if gt_cache is not None and gt_tool_keys is not None and text in gt_tool_keys:
                        gt_cache[text] = embedding
            print(f"Restored {len(progress)} embeddings from progress file")
        except Exception as exc:
            print(f"Warning: failed to load progress: {exc}")
    
    # Skip texts already done
    remaining_texts = [text for text in missing_texts if text not in cache]
    
    if not remaining_texts:
        print("All embeddings fetched")
        # Remove progress file
        if progress_file.exists():
            try:
                progress_file.unlink()
            except:
                pass
        return
    
    print(f"Batch-fetching {len(remaining_texts)} missing embeddings (batch_size={batch_size}, workers={MAX_WORKERS})...")
    
    total_batches = (len(remaining_texts) + batch_size - 1) // batch_size
    
    # Locks for shared state
    progress_lock = Lock()
    cache_lock = Lock()
    gt_cache_updated = False  # GT cache dirty flag
    
    # process_batch helper
    def process_batch(batch_idx: int, batch_texts: List[str]) -> Tuple[int, List[Tuple[str, List[float]]]]:
        """Process one batch.

        Args:
            batch_idx: batch index
            batch_texts: texts in batch
        Returns:
            (batch_idx, [(text, embedding), ...])
        """
        try:
            # Batch API call
            batch_embeddings = request_embeddings_batch(batch_texts, mode, server_url)
            
            # Build results
            batch_results = []
            for text, embedding in zip(batch_texts, batch_embeddings):
                batch_results.append((text, embedding))
            
            return (batch_idx, batch_results)
        except Exception as exc:
            print(f"Error: batch embedding failed (batch {batch_idx + 1}/{total_batches}): {exc}")
            raise
    
    # Thread pool
    completed_count = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit batches
        future_to_batch = {}
        for batch_idx in range(0, len(remaining_texts), batch_size):
            batch_texts = remaining_texts[batch_idx:batch_idx + batch_size]
            future = executor.submit(process_batch, batch_idx // batch_size, batch_texts)
            future_to_batch[future] = batch_idx // batch_size
        
        # tqdm
        with tqdm(total=total_batches, desc="Fetch embeddings", unit="batch") as pbar:
            # Collect futures
            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                try:
                    _, batch_results = future.result()
                    
                    # Update cache + progress (locked)
                    with cache_lock:
                        for text, embedding in batch_results:
                            cache[text] = embedding
                            with progress_lock:
                                progress[text] = embedding
                            
                            # Update GT cache for GT tools
                            if gt_cache is not None and gt_tool_keys is not None and text in gt_tool_keys:
                                gt_cache[text] = embedding
                                gt_cache_updated = True
                    
                    # Save progress each batch
                    try:
                        with progress_lock:
                            progress_file.parent.mkdir(parents=True, exist_ok=True)
                            temp_progress_path = progress_file.with_suffix(progress_file.suffix + ".tmp")
                            with temp_progress_path.open("w", encoding="utf-8") as f:
                                json.dump(progress, f, ensure_ascii=False, indent=2)
                            temp_progress_path.replace(progress_file)
                    except Exception as exc:
                        print(f"Warning: failed to save progress: {exc}")
                    
                    # Periodically save GT cache
                    if gt_cache_updated and gt_cache is not None and gt_cache_file is not None:
                        try:
                            with cache_lock:
                                save_query_cache(gt_cache, gt_cache_file)
                                gt_cache_updated = False  # reset dirty flag
                        except Exception as exc:
                            print(f"Warning: failed to save GT cache: {exc}")
                    
                    completed_count += len(batch_results)
                    pbar.update(1)
                    
                except Exception as exc:
                    print(f"Error: batch embedding failed (batch {batch_idx + 1}/{total_batches}): {exc}")
                    print(f"Progress saved; rerun to continue")
                    raise
    
    print(f"Done: fetched all {len(remaining_texts)} embeddings")
    
    # Final GT cache save
    if gt_cache_updated and gt_cache is not None and gt_cache_file is not None:
        try:
            save_query_cache(gt_cache, gt_cache_file)
        except Exception as exc:
            print(f"Warning: failed to save GT cache: {exc}")
    
    # Remove progress file
    if progress_file.exists():
        try:
            progress_file.unlink()
        except:
            pass


def build_faiss_index_from_tool_pool(
    tool_pool: Dict[str, Dict[str, Any]],
    cache: Dict[str, List[float]],
    tool_embeddings: Dict[str, np.ndarray],
    mode: str = "api",
    server_url: Optional[str] = None,
    gt_cache: Optional[Dict[str, List[float]]] = None,
    gt_tool_keys: Optional[Set[str]] = None,
    gt_cache_file: Optional[Path] = None
) -> Tuple[Optional[Any], Dict[int, str]]:
    """
    Build a Faiss index (CPU) from the tool pool.

    Returns:
        (faiss_index, index_to_tool_key) or (None, {}) if Faiss is unavailable.
    """
    if not FAISS_AVAILABLE:
        return None, {}
    
    try:
        # Cache hit stats
        total_tools = len(tool_pool)
        missing_tools = []
        matched_by_normalization = 0
        
        for tool_key in tool_pool.keys():
            # Exact key match
            if tool_key in cache or tool_key in tool_embeddings:
                continue
            
            # Normalize newlines/spaces
            # Normalize whitespace
            normalized_key = tool_key.strip()
            import re
            normalized_key = re.sub(r'\n+', '\n', normalized_key).strip()
            
            # Match normalized keys
            found = False
            for cached_key in cache.keys():
                normalized_cached = cached_key.strip()
                normalized_cached = re.sub(r'\n+', '\n', normalized_cached).strip()
                if normalized_key == normalized_cached:
                    # Alias cache entry
                    cache[tool_key] = cache[cached_key]
                    matched_by_normalization += 1
                    found = True
                    break
            
            if not found:
                missing_tools.append(tool_key)
        
        if matched_by_normalization > 0:
            print(f"Matched {matched_by_normalization} embeddings via normalization (same content, different formatting)")
        
        # Fetch missing embeddings first
        if missing_tools:
            print(f"{len(missing_tools)} tools missing from embedding cache")
            print(f"Cache hit rate: {(total_tools - len(missing_tools)) / total_tools * 100:.1f}%")
            print("Fetching missing embeddings...\n")
            
            batch_fetch_missing_embeddings(
                missing_tools,
                cache,
                TOOL_EMBEDDING_PROGRESS_FILE,
                mode,
                server_url,
                batch_size=BATCH_SIZE,
                gt_cache=gt_cache,
                gt_tool_keys=gt_tool_keys,
                gt_cache_file=gt_cache_file
            )
            print()
        
        # Collect vectors
        embeddings_list = []
        index_to_tool_key = {}
        tool_key_to_index = {}
        
        print(f"Building Faiss index ({total_tools} tools)...")
        
        # tqdm over tools
        for idx, (tool_key, tool_obj) in enumerate(tqdm(tool_pool.items(), desc="Build index", unit="tool", total=total_tools)):
            # Embedding should exist
            if tool_key not in tool_embeddings:
                if tool_key in cache:
                    tool_embedding = normalize_vector(np.asarray(cache[tool_key], dtype=np.float32))
                    tool_embeddings[tool_key] = tool_embedding
                else:
                    # Should not happen after batch fetch
                    print(f"Warning: missing embedding for {tool_key}; skip")
                    continue
            else:
                tool_embedding = tool_embeddings[tool_key]
            
            embeddings_list.append(tool_embedding)
            index_to_tool_key[idx] = tool_key
            tool_key_to_index[tool_key] = idx
        
        if not embeddings_list:
            return None, {}
        
        # Build Faiss CPU index
        print("Building Faiss matrix...")
        embeddings_matrix = np.vstack(embeddings_list).astype(np.float32)
        dim = embeddings_matrix.shape[1]
        
        # Inner product = cosine on L2-normalized vectors
        print("Adding vectors to index...")
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings_matrix)
        
        print(f"Built CPU Faiss index: {len(embeddings_list)} tool vectors")
        
        return index, index_to_tool_key
        
    except Exception as exc:
        print(f"Warning: Faiss build failed: {exc}; using NumPy fallback")
        return None, {}


def load_tool_pool() -> Dict[str, Dict[str, Any]]:
    """Load tool pool (excludes details/).

    Keys are name+description (see format_tool_text) because IDs may collide across files.
    """
    tool_pool = {}
    
    # JSON files in pool dir
    json_files = [f for f in TOOL_POOL_DIR.glob("*.json") if f.is_file()]
    
    for json_file in json_files:
        try:
            with json_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
            
            file_count = 0
            # Dict: tool id -> ...
            if isinstance(data, dict):
                for tool_id, tool_obj in data.items():
                    # Dict value is list
                    if isinstance(tool_obj, list):
                        for item in tool_obj:
                            if isinstance(item, dict) and "name" in item:
                                tool_name = item.get("name", "")
                                tool_description = item.get("description", "")
                                if tool_name:
                                    # Key = format_tool_text(name, description)
                                    tool_key = format_tool_text(tool_name, tool_description)
                                    tool_pool[tool_key] = item
                                    file_count += 1
                    # Dict value is object
                    elif isinstance(tool_obj, dict) and "name" in tool_obj:
                        tool_name = tool_obj.get("name", "")
                        tool_description = tool_obj.get("description", "")
                        if tool_name:
                            # Key = format_tool_text(name, description)
                            tool_key = format_tool_text(tool_name, tool_description)
                            tool_pool[tool_key] = tool_obj
                            file_count += 1
            
            # List of tools
            elif isinstance(data, list):
                for idx, tool_obj in enumerate(data):
                    if isinstance(tool_obj, dict) and "name" in tool_obj:
                        tool_name = tool_obj.get("name", "")
                        tool_description = tool_obj.get("description", "")
                        if tool_name:
                            # Key = format_tool_text(name, description)
                            tool_key = format_tool_text(tool_name, tool_description)
                            tool_pool[tool_key] = tool_obj
                            file_count += 1
            
            print(f"  {json_file.name}: {file_count} tools")
        
        except Exception as exc:
            print(f"Warning: failed to load {json_file.name}: {exc}")
            continue
    
    print(f"Tool pool: {len(tool_pool)} unique tools")
    return tool_pool


def extract_first_function_call(conversations: List[Dict[str, str]]) -> Optional[str]:
    """Return first function_call value (JSON string) or None."""
    for conv in conversations:
        if conv.get("from") == "function_call":
            return conv.get("value", "")
    return None


def generate_item_id(conversations: List[Dict[str, str]]) -> str:
    """Stable item id from conversations (MD5 of canonical JSON)."""
    import hashlib
    # Canonical JSON for hashing
    conv_str = json.dumps(conversations, sort_keys=True, ensure_ascii=False)
    # MD5
    conv_hash = hashlib.md5(conv_str.encode('utf-8')).hexdigest()
    return conv_hash


def extract_gt_tools(conversations: List[Dict[str, str]]) -> Set[str]:
    """Extract ground-truth tool names from conversations."""
    gt_tools = set()
    for conv in conversations:
        if conv.get("from") == "function_call":
            try:
                value = json.loads(conv.get("value", "{}"))
                tool_name = value.get("name")
                if tool_name:
                    gt_tools.add(tool_name)
            except:
                pass
    return gt_tools


def find_tool_by_name(tool_pool: Dict[str, Dict[str, Any]], tool_name: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Find tool by name; pool keys are "name\\ndescription" (first match)."""
    for tool_key, tool_obj in tool_pool.items():
        if tool_obj.get("name") == tool_name:
            return tool_key, tool_obj
    return None


def compute_similarities(
    query_embedding: np.ndarray,
    tool_pool: Dict[str, Dict[str, Any]],
    tool_embeddings: Dict[str, np.ndarray],
    cache: Dict[str, List[float]],
    mode: str = "api",
    server_url: Optional[str] = None,
    faiss_index: Optional[Any] = None,
    index_to_tool_key: Optional[Dict[int, str]] = None,
    top_k: Optional[int] = None
) -> List[Tuple[str, float]]:
    """Cosine similarity vs all tools; optional Faiss top-k."""
    # Faiss path
    if FAISS_AVAILABLE and faiss_index is not None and index_to_tool_key is not None:
        try:
            # Faiss top-k
            k = top_k if top_k is not None else len(tool_pool)
            k = min(k, len(tool_pool))
            
            query_vector = query_embedding.reshape(1, -1).astype(np.float32)
            scores, indices = faiss_index.search(query_vector, k)
            
            similarities = []
            for i in range(min(k, len(indices[0]))):
                idx = int(indices[0][i])
                score = float(scores[0][i])
                if idx in index_to_tool_key:
                    tool_key = index_to_tool_key[idx]
                    similarities.append((tool_key, score))
            
            # Faiss returns sorted
            return similarities
            
        except Exception as exc:
            print(f"Warning: Faiss search failed: {exc}; using NumPy")
            # fall through
    
        # NumPy fallback
    similarities = []
    
    for tool_key, tool_obj in tool_pool.items():
        # tool_key is formatted text
        if tool_key not in tool_embeddings:
            # tool_key is cache key
            tool_embedding = get_embedding(tool_key, cache, mode, server_url)
            tool_embeddings[tool_key] = tool_embedding
        else:
            tool_embedding = tool_embeddings[tool_key]
        
        # dot product
        similarity = float(np.dot(query_embedding, tool_embedding))
        similarities.append((tool_key, similarity))
    
    # Sort by score
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Truncate top_k
    if top_k is not None:
        return similarities[:top_k]
    
    return similarities


def build_toollist_setting(
    gt_tools: Set[str],
    tool_pool: Dict[str, Dict[str, Any]],
    tool_embeddings: Dict[str, np.ndarray],
    cache: Dict[str, List[float]],
    gt_cache: Dict[str, List[float]],
    mode: str = "api",
    server_url: Optional[str] = None,
    faiss_index: Optional[Any] = None,
    index_to_tool_key: Optional[Dict[int, str]] = None
) -> List[Dict[str, Any]]:
    """
    Pick up to 15 tools by cosine similarity to GT, then add 5 random tools (20 total).
    All GT tools are included.

    Rules:
    - If |GT| < 15: keep all GT, fill to 15 by similarity, then add 5 random.
    - If |GT| == 15: keep all GT, add 5 random.
    """
    selected_tools = []
    selected_tool_ids = set()
    
    # 1) Include all GT tools
    gt_tool_keys = []
    gt_tool_embeddings = {}
    gt_cache_updated = False  # GT cache dirty
    for gt_tool_name in gt_tools:
        result = find_tool_by_name(tool_pool, gt_tool_name)
        if result:
            tool_key, tool_obj = result
            if tool_key not in selected_tool_ids:
                selected_tools.append(tool_obj)
                selected_tool_ids.add(tool_key)
                gt_tool_keys.append(tool_key)
                
                # GT embedding from cache
                if tool_key in gt_cache:
                    gt_embedding = normalize_vector(np.asarray(gt_cache[tool_key], dtype=np.float32))
                    gt_tool_embeddings[tool_key] = gt_embedding
                elif tool_key not in tool_embeddings:
                    # Fall back to pool cache
                    if tool_key in cache:
                        gt_embedding = normalize_vector(np.asarray(cache[tool_key], dtype=np.float32))
                        tool_embeddings[tool_key] = gt_embedding
                        gt_tool_embeddings[tool_key] = gt_embedding
                        # Save GT cache
                        gt_cache[tool_key] = cache[tool_key]
                        gt_cache_updated = True
                    else:
                        # API as last resort
                        gt_embedding = get_embedding(tool_key, cache, mode, server_url)
                        tool_embeddings[tool_key] = gt_embedding
                        gt_tool_embeddings[tool_key] = gt_embedding
                        # Save GT cache
                        gt_cache[tool_key] = cache[tool_key]
                        gt_cache_updated = True
    
    # Persist GT cache
    if gt_cache_updated:
        save_query_cache(gt_cache, GT_CACHE_FILE)
    
    gt_count = len(selected_tools)
    
    # 2) Similar tools
    if gt_count < 15:
        # Pad to 15
        need_count = 15 - gt_count
        
        # Rank by similarity
        all_similarities = []
        for gt_tool_key in gt_tool_keys:
            gt_embedding = gt_tool_embeddings.get(gt_tool_key)
            if gt_embedding is None:
                continue
            
            # compute_similarities
            # Index covers full pool
            similarities = compute_similarities(
                gt_embedding,
                tool_pool,  # full pool
                tool_embeddings,
                cache,
                mode,
                server_url,
                faiss_index,
                index_to_tool_key,
                top_k=None  # need full ranking
            )
            
            # Exclude selected
            filtered_similarities = [
                (tool_key, sim) for tool_key, sim in similarities 
                if tool_key not in selected_tool_ids
            ]
            
            all_similarities.extend(filtered_similarities)
        
        # Sort by score; take top need_count (dedup)
        all_similarities.sort(key=lambda x: x[1], reverse=True)
        seen_tool_keys = set()
        for tool_key, _ in all_similarities:
            if tool_key not in selected_tool_ids and tool_key not in seen_tool_keys:
                selected_tools.append(tool_pool[tool_key])
                selected_tool_ids.add(tool_key)
                seen_tool_keys.add(tool_key)
                if len(selected_tools) >= 15:
                    break
    
    # 3) Random fill to 20
    remaining_tools = [
        (tool_key, tool_obj) for tool_key, tool_obj in tool_pool.items()
        if tool_key not in selected_tool_ids
    ]
    
    if len(selected_tools) < 20 and remaining_tools:
        random.shuffle(remaining_tools)
        for tool_key, tool_obj in remaining_tools:
            if len(selected_tools) >= 20:
                break
            selected_tools.append(tool_obj)
            selected_tool_ids.add(tool_key)
    
    return selected_tools[:20]


build_toollist_setting1 = build_toollist_setting  # legacy alias


def process_file_setting1(
    input_file: Path,
    output_file: Path,
    tool_pool: Dict[str, Dict[str, Any]],
    tool_embeddings: Dict[str, np.ndarray],
    cache: Dict[str, List[float]],
    gt_cache: Dict[str, List[float]],
    mode: str = "api",
    server_url: Optional[str] = None,
    faiss_index: Optional[Any] = None,
    index_to_tool_key: Optional[Dict[int, str]] = None
) -> None:
    """Process one file (setting 1); supports resume."""
    with input_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        print(f"Warning: {input_file.name} is not a list; skip")
        return
    
    # Resume from partial output
    processed_data = []
    processed_item_ids = set()
    if output_file.exists():
        try:
            with output_file.open("r", encoding="utf-8") as f:
                existing_data = json.load(f)
            if isinstance(existing_data, list):
                # Collect item ids
                for item in existing_data:
                    conversations = item.get("conversations", [])
                    item_id = generate_item_id(conversations)
                    if item_id:
                        processed_item_ids.add(item_id)
                processed_data = existing_data
                print(f"  Loaded {len(processed_data)} items from existing output (by conversation id)")
        except Exception as exc:
            print(f"Warning: could not load existing output: {exc}; reprocessing")
            processed_data = []
            processed_item_ids = set()
    
    # Count remaining items
    total_to_process = len(data)
    if processed_item_ids:
        # Unprocessed count
        unprocessed_count = 0
        for item in data:
            conversations = item.get("conversations", [])
            item_id = generate_item_id(conversations)
            if not item_id or item_id not in processed_item_ids:
                unprocessed_count += 1
        total_to_process = unprocessed_count
        print(f"  To process: {total_to_process} (skipped {len(data) - total_to_process})")
    
    # Main loop
    processed_count = 0
    pbar = tqdm(data, desc=f"Process {input_file.name}", total=len(data))
    for idx, item in enumerate(pbar):
        # Skip if item_id seen
        conversations = item.get("conversations", [])
        item_id = generate_item_id(conversations)
        if item_id and item_id in processed_item_ids:
            # Skip processed
            continue
        
        # GT tools
        gt_tools = extract_gt_tools(conversations)
        
        if not gt_tools:
            # No GT: keep item
            processed_data.append(item)
        elif len(gt_tools) == 20:
            # 20 GT: keep tools
            processed_data.append(item)
        else:
            # build_toollist_setting
            toollist = build_toollist_setting(
                gt_tools, tool_pool, tool_embeddings, cache, gt_cache, 
                mode, server_url, faiss_index, index_to_tool_key
            )
            
            # Replace tools JSON
            new_item = item.copy()
            new_item["tools"] = json.dumps(toollist, ensure_ascii=False)
            processed_data.append(new_item)
        
        # Track item_id
        if item_id:
            processed_item_ids.add(item_id)
        
        # Update tqdm desc
        processed_count += 1
        if total_to_process > 0:
            pbar.set_description(f"Process {input_file.name} ({processed_count}/{total_to_process})")
        
        # Periodic save
        if len(processed_data) % SAVE_BATCH_SIZE == 0:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            temp_output_file = output_file.with_suffix(output_file.suffix + ".tmp")
            with temp_output_file.open("w", encoding="utf-8") as f:
                json.dump(processed_data, f, ensure_ascii=False, indent=2)
            temp_output_file.replace(output_file)
    
    # Final write
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build toollist and replace tools field")
    parser.add_argument("--mode", choices=["api", "server"], default="api", 
                       help="Embedding mode: api or server (default: api)")
    parser.add_argument("--server-url", type=str, default=EMBED_SERVER_URL, help="Local server URL (only when mode=server)")
    args = parser.parse_args()
    
    mode = args.mode
    server_url = args.server_url if args.server_url else EMBED_SERVER_URL
    
    # Load tool pool
    print("Loading tool pool...")
    tool_pool = load_tool_pool()
    
    # Load pool embedding caches
    print("Loading tool embedding caches...")
    cache = {}
    # Glob *.embeddings.json
    cache_files = list(EMBED_CACHE_DIR.glob("*.embeddings.json"))
    
    if not cache_files:
        print(f"Warning: no .embeddings.json under {EMBED_CACHE_DIR}")
    else:
        print(f"Found {len(cache_files)} embedding cache files")
    
    for cache_file in cache_files:
        if cache_file.exists():
            file_cache = load_embedding_cache(cache_file)
            cache.update(file_cache)
            print(f"  Loaded {cache_file.name}: {len(file_cache)} embeddings")
    
    print(f"Total pooled embeddings: {len(cache)}\n")
    
    # Load GT cache
    print("Loading GT tool embeddings...")
    gt_cache = load_embedding_cache(GT_CACHE_FILE)
    if not gt_cache:
        # Fallback glob
        # e.g. *train_set*.embeddings.json
        potential_files = list(EMBED_CACHE_DIR.glob("*train_set*.embeddings.json"))
        for gt_cache_file in potential_files:
            file_cache = load_embedding_cache(gt_cache_file)
            if file_cache:
                gt_cache.update(file_cache)
                print(f"  Loaded {len(file_cache)} GT embeddings from {gt_cache_file.name}")
    print(f"GT embedding cache: {len(gt_cache)}\n")
    
    # In-memory tool embeddings
    tool_embeddings = {}
    
    # List input JSON
    print("Collecting input files...")
    # rglob json
    json_files = list(DATA_NOTIME_DIR.rglob("*.json"))
    print(f"Found {len(json_files)} files\n")
    
    # Collect GT tool keys
    print("Collecting GT tool keys...")
    all_gt_tool_names = set()
    for input_file in json_files:
        try:
            with input_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    conversations = item.get("conversations", [])
                    gt_tools = extract_gt_tools(conversations)
                    all_gt_tool_names.update(gt_tools)
        except Exception as exc:
            print(f"Warning: failed to read {input_file.name}: {exc}")
    
    # Name -> tool_key
    gt_tool_keys = set()
    for gt_tool_name in all_gt_tool_names:
        result = find_tool_by_name(tool_pool, gt_tool_name)
        if result:
            tool_key, _ = result
            gt_tool_keys.add(tool_key)
    
    print(f"Collected {len(gt_tool_keys)} GT tool keys\n")
    
    # Ephemeral Faiss index
    # build_faiss fetches missing
    print("Building Faiss index (CPU) for similarity...")
    faiss_index, index_to_tool_key = build_faiss_index_from_tool_pool(
        tool_pool, cache, tool_embeddings, mode, server_url,
        gt_cache=gt_cache, gt_tool_keys=gt_tool_keys, gt_cache_file=GT_CACHE_FILE
    )
    if faiss_index is not None:
        print(f"Faiss index OK: {len(index_to_tool_key)} vectors\n")
    else:
        print("Faiss unavailable; using NumPy for similarity\n")
    
    print("=" * 70)
    print("Processing data (GT-similarity toollist)...")
    print("=" * 70)
    for input_file in json_files:
        # Preserve relative paths
        output_file = OUTPUT_DIR1 / input_file.relative_to(DATA_NOTIME_DIR)
        try:
            process_file_setting1(
                input_file, output_file, tool_pool, tool_embeddings, cache, gt_cache, 
                mode, server_url, faiss_index, index_to_tool_key
            )
            print(f"Done: {input_file.name} -> {output_file}")
        except Exception as exc:
            print(f"Error: failed on {input_file.name}: {exc}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("All done.")
    print("=" * 70)


if __name__ == "__main__":
    main()

