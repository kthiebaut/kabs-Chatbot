
import os
import re
import uuid
import time
import hashlib
import tempfile
import json
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
import concurrent.futures
from threading import Lock
from tenacity import retry, stop_after_attempt, wait_exponential

# Flask imports
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Document processing
import pandas as pd
from pypdf import PdfReader
import numpy as np

# Vector DB - Pinecone
from pinecone import Pinecone, ServerlessSpec

# AWS S3 (optional)
try:
    import boto3
    from botocore.exceptions import NoCredentialsError
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False

# OpenAI
import tiktoken
from openai import OpenAI

# Fuzzy matching for better item code matching
try:
    from fuzzywuzzy import fuzz, process
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False
    print("⚠️ fuzzywuzzy not installed. Install with: pip install fuzzywuzzy python-Levenshtein")

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------
# Configuration
# -----------------------------
CHUNK_TOKENS = 800
OVERLAP_TOKENS = 150
MAX_CONTEXT_LENGTH = 8000
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
PINECONE_INDEX_NAME = "kb-scout-documents"
MAX_FILE_SIZE = 50 * 1024 * 1024       # Increased to 50MB to handle larger files
PINECONE_DIMENSION = 1536
BATCH_SIZE = 20     # Batch size for Pinecone uploads
MAX_CHUNKS_PER_FILE = None    # NO LIMIT - Process ALL data
ALLOWED_EXTENSIONS = {'pdf', 'csv', 'xlsx', 'xls', 'txt'}
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
CACHE_SIZE = 10000
EMBEDDING_BATCH_SIZE = 100
MAX_EMBEDDING_INPUT = 8191
MAX_PAGES_FOR_CHUNKING = None  # NO LIMIT - Process ALL pages
CSV_ROWS_PER_CHUNK = 15
MAX_METADATA_LENGTH = 500
PDF_CHUNK_SIZE = 800
PDF_OVERLAP = 150
# CRITICAL: No sampling limits - index EVERYTHING
INDEX_ALL_DATA = True  # Flag to ensure all data is indexed

# -----------------------------
# Global Caches
# -----------------------------
_cache = {
    'openai_client': None,
    'pinecone_client': None,
    'pinecone_index': None,
    's3_client': None,
    'tokenizer': None,
    'embeddings': {},
    'uploaded_files': {},
    'csv_lookup_cache': {},
    'full_documents': {}  # Cache for full document content
}
_embedding_cache = {}
_cache_lock = Lock()

# Enhanced document store for persistence with file tracking
document_store = {
    'files': {},  # filename -> file metadata
    'full_content': {},  # Store complete document content
    'file_vectors': {},  # Track which vectors belong to which files
    'lock': Lock()
}

# -----------------------------
# Client Initialization
# -----------------------------
@lru_cache(maxsize=1)
def get_openai_client() -> Optional[OpenAI]:
    """Get or create OpenAI client"""
    if not _cache['openai_client']:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not found in environment")
            return None
        try:
            _cache['openai_client'] = OpenAI(api_key=api_key, timeout=30)
            logger.info("✅ OpenAI client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI: {e}")
            return None
    return _cache['openai_client']

@lru_cache(maxsize=1)
def get_pinecone_client():
    """Get or create Pinecone client"""
    if not _cache['pinecone_client']:
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            logger.warning("PINECONE_API_KEY not found in environment")
            return None
        try:
            _cache['pinecone_client'] = Pinecone(api_key=api_key)
            logger.info("✅ Pinecone client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            return None
    return _cache['pinecone_client']

def get_pinecone_index():
    """Get or create Pinecone index"""
    if not _cache['pinecone_index']:
        pc = get_pinecone_client()
        if not pc:
            return None
        
        try:
            # Check if index exists
            existing_indexes = pc.list_indexes().names()
            
            if PINECONE_INDEX_NAME not in existing_indexes:
                logger.info(f"Creating Pinecone index: {PINECONE_INDEX_NAME}")
                pc.create_index(
                    name=PINECONE_INDEX_NAME,
                    dimension=PINECONE_DIMENSION,
                    metric='cosine',
                    spec=ServerlessSpec(
                        cloud='aws',
                        region=os.getenv('PINECONE_REGION', 'us-east-1')
                    )
                )
                time.sleep(5)  # Wait for index creation
            
            _cache['pinecone_index'] = pc.Index(PINECONE_INDEX_NAME)
            
            # Test the index
            stats = _cache['pinecone_index'].describe_index_stats()
            logger.info(f"✅ Pinecone index ready. Vectors: {stats.get('total_vector_count', 0)}")
            
        except Exception as e:
            logger.error(f"Failed to setup Pinecone index: {e}")
            return None
    
    return _cache['pinecone_index']

@lru_cache(maxsize=1)
def get_s3_client():
    """Get or create S3 client"""
    if not S3_AVAILABLE:
        return None
        
    if not _cache['s3_client']:
        access_key = os.getenv("AWS_ACCESS_KEY_ID")
        secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        if not access_key or not secret_key:
            return None  # S3 is optional
        _cache['s3_client'] = boto3.client(
            's3',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=os.getenv("AWS_REGION")
        )
    return _cache['s3_client']

def get_tokenizer():
    """Get or create tokenizer"""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = tiktoken.get_encoding("cl100k_base")
    return _tokenizer

# -----------------------------
# S3 Storage Functions
# -----------------------------
def upload_to_s3(file_path: str, s3_key: str) -> bool:
    """Upload file to S3"""
    s3_client = get_s3_client()
    if not s3_client:
        logger.info("S3 not configured, skipping upload")
        return True  # Continue without S3
    
    try:
        s3_client.upload_file(file_path, S3_BUCKET_NAME, s3_key)
        logger.info(f"✅ Uploaded {s3_key} to S3")
        return True
    except Exception as e:
        logger.error(f"S3 upload failed: {e}")
        return False

def delete_from_s3(s3_key: str) -> bool:
    """Delete file from S3"""
    s3_client = get_s3_client()
    if not s3_client:
        return True  # If S3 not configured, consider it successful
    
    try:
        s3_client.delete_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
        logger.info(f"✅ Deleted {s3_key} from S3")
        return True
    except Exception as e:
        logger.error(f"S3 delete failed: {e}")
        return False

def download_from_s3(s3_key: str, local_path: str) -> bool:
    """Download file from S3"""
    s3_client = get_s3_client()
    if not s3_client:
        return False
    
    try:
        s3_client.download_file(S3_BUCKET_NAME, s3_key, local_path)
        return True
    except Exception as e:
        logger.error(f"S3 download failed: {e}")
        return False

# -----------------------------
# Text Processing Functions
# -----------------------------
def clean_text(text: str) -> str:
    """Clean text for processing"""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\.\,\!\?\-\(\)\$\%\/]', '', text)
    return text.strip()

def chunk_text(text: str, max_tokens: int = CHUNK_TOKENS) -> List[str]:
    """Simple fast chunking"""
    # Character-based chunking for speed
    chunk_size = max_tokens * 4  # Approximate chars per token
    chunks = []
    
    for i in range(0, len(text), chunk_size - 200):  # Overlap
        chunk = text[i:i + chunk_size]
        if len(chunk.strip()) > 100:
            chunks.append(chunk)
            if len(chunks) >= MAX_CHUNKS_PER_FILE:
                break
    
    return chunks

# -----------------------------
# Embedding Functions
# -----------------------------

_tokenizer = None



def count_tokens(text: str) -> int:
    """Accurately count tokens in text"""
    try:
        tokenizer = get_tokenizer()
        return len(tokenizer.encode(text))
    except Exception as e:
        logger.warning(f"Token counting failed: {e}")
        return len(text) // 4

def truncate_text_to_tokens(text: str, max_tokens: int = MAX_EMBEDDING_INPUT) -> str:
    """Truncate text to fit within token limit"""
    try:
        tokenizer = get_tokenizer()
        tokens = tokenizer.encode(text)
        
        if len(tokens) <= max_tokens:
            return text
        
        truncated_tokens = tokens[:max_tokens]
        return tokenizer.decode(truncated_tokens)
    except Exception as e:
        logger.warning(f"Text truncation failed: {e}")
        return text[:max_tokens * 4]

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True
)








def _generate_embedding_with_retry(client: OpenAI, text: str) -> List[float]:
    """Generate embedding with automatic retry logic"""
    truncated_text = truncate_text_to_tokens(text, MAX_EMBEDDING_INPUT)
    
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=truncated_text,
        encoding_format="float"
    )
    
    return response.data[0].embedding


def generate_embedding(
    text: str,
    use_cache: bool = True,
    normalize: bool = True
) -> Optional[List[float]]:
    """
    Generate embedding for text
    
    Args:
        text: Input text to embed
        use_cache: Whether to use cache (default True)
        normalize: Whether to normalize embedding vector (default True)
    
    Returns:
        List of floats representing the embedding, or None on failure
    """
    # Input validation
    if not text or not text.strip():
        logger.warning("Empty text provided for embedding")
        return None
    
    text = text.strip()
    
    # Check cache first
    if use_cache:
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        cached_embedding = _get_cached_embedding(text_hash)
        if cached_embedding is not None:
            return cached_embedding
    
    # Get OpenAI client
    client = get_openai_client()
    if not client:
        logger.error("OpenAI client not available")
        return None
    
    try:
        # Generate embedding with retry logic
        embedding = _generate_embedding_with_retry(client, text)
        
        # Normalize if requested
        if normalize:
            embedding = normalize_embedding(embedding)
        
        # Cache the result
        if use_cache:
            _cache_embedding(text_hash, embedding)
        
        return embedding
        
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        return None

###################################


def generate_embeddings_batch(
    texts: List[str],
    preserve_order: bool = True,
    max_workers: int = 3,
    timeout: int = 30
) -> List[Optional[List[float]]]:
    """
    Generate embeddings for multiple texts with proper error handling
    
    IMPORTANT: This uses threading but OpenAI's batch API is more efficient.
    Consider using the batch approach instead for large volumes.
    
    Args:
        texts: List of texts to embed
        preserve_order: Maintain original order (default True)
        max_workers: Number of concurrent threads (default 3, reduced from 5)
        timeout: Timeout per embedding in seconds
    
    Returns:
        List of embeddings in same order as input (None for failures)
    """
    if not texts:
        return []
    
    # Clean texts
    cleaned_texts = [text.strip() if text else "" for text in texts]
    
    embeddings_dict = {}
    failed_indices = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit with index tracking
        future_to_index = {
            executor.submit(generate_embedding, text): idx 
            for idx, text in enumerate(cleaned_texts) if text
        }
        
        # Process completed futures
        for future in concurrent.futures.as_completed(future_to_index):
            idx = future_to_index[future]
            
            try:
                embedding = future.result(timeout=timeout)
                
                if embedding is not None and validate_embedding(embedding):
                    embeddings_dict[idx] = embedding
                else:
                    logger.warning(f"Invalid embedding at index {idx}")
                    failed_indices.append(idx)
                    embeddings_dict[idx] = None
                    
            except concurrent.futures.TimeoutError:
                logger.error(f"Timeout generating embedding at index {idx}")
                failed_indices.append(idx)
                embeddings_dict[idx] = None
                
            except Exception as e:
                logger.error(f"Error generating embedding at index {idx}: {e}")
                failed_indices.append(idx)
                embeddings_dict[idx] = None
    
    # Reconstruct in original order
    if preserve_order:
        embeddings = [embeddings_dict.get(i) for i in range(len(cleaned_texts))]
    else:
        embeddings = list(embeddings_dict.values())
    
    # Log statistics
    success_count = sum(1 for e in embeddings if e is not None)
    logger.info(f"Batch embedding: {success_count}/{len(texts)} successful")
    
    if failed_indices:
        logger.warning(f"Failed indices: {failed_indices[:10]}{'...' if len(failed_indices) > 10 else ''}")
    
    return embeddings


def generate_embeddings_batch_optimized(
    texts: List[str],
    batch_size: int = 100,
    use_cache: bool = True,
    show_progress: bool = False
) -> List[Optional[List[float]]]:
    """
    Generate embeddings using OpenAI's native batching (RECOMMENDED)
    
    Args:
        texts: List of texts to embed
        batch_size: Number of texts per API call (max 100)
        use_cache: Whether to use caching
        show_progress: Show progress for large batches
    
    Returns:
        List of embeddings in same order as input (None for failures)
    """
    if not texts:
        return []
    
    client = get_openai_client()
    if not client:
        logger.error("OpenAI client not available")
        return [None] * len(texts)
    
    # Clean and prepare texts
    cleaned_texts = [text.strip() if text else "" for text in texts]
    embeddings = [None] * len(texts)
    
    # Separate cached and non-cached
    texts_to_process = []
    indices_to_process = []
    
    for idx, text in enumerate(cleaned_texts):
        if not text:
            continue
        
        # Check cache
        if use_cache:
            text_hash = hashlib.sha256(text.encode()).hexdigest()
            cached = _get_cached_embedding(text_hash)
            if cached is not None:
                embeddings[idx] = cached
                continue
        
        # Truncate to token limit
        truncated = truncate_text_to_tokens(text, MAX_EMBEDDING_INPUT)
        texts_to_process.append((idx, text, truncated))
        indices_to_process.append(idx)
    
    if not texts_to_process:
        logger.info("All embeddings retrieved from cache")
        return embeddings
    
    # Process in batches
    total_batches = (len(texts_to_process) + batch_size - 1) // batch_size
    
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(texts_to_process))
        
        batch_items = texts_to_process[start_idx:end_idx]
        batch_texts = [item[2] for item in batch_items]  # Get truncated texts
        
        try:
            # Single API call for entire batch
            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=batch_texts,
                encoding_format="float"
            )
            
            # Process results
            for i, embedding_data in enumerate(response.data):
                original_idx, original_text, _ = batch_items[i]
                embedding = embedding_data.embedding
                
                # Normalize
                embedding = normalize_embedding(embedding)
                
                # Validate
                if validate_embedding(embedding):
                    embeddings[original_idx] = embedding
                    
                    # Cache
                    if use_cache:
                        text_hash = hashlib.sha256(original_text.encode()).hexdigest()
                        _cache_embedding(text_hash, embedding)
                else:
                    logger.warning(f"Invalid embedding at index {original_idx}")
                    embeddings[original_idx] = None
            
            if show_progress:
                logger.info(f"Batch {batch_num + 1}/{total_batches} completed")
            
            # Rate limiting between batches
            if batch_num < total_batches - 1:
                time.sleep(0.2)
                
        except Exception as e:
            logger.error(f"Batch {batch_num + 1} failed: {e}")
            # Mark batch as failed
            for item in batch_items:
                embeddings[item[0]] = None
    
    # Statistics
    success_count = sum(1 for e in embeddings if e is not None)
    logger.info(f"Generated {success_count}/{len(texts)} embeddings successfully")
    
    return embeddings


def generate_embeddings_with_retry(
    texts: List[str],
    max_retries: int = 2,
    retry_delay: float = 1.0
) -> List[Optional[List[float]]]:
    """
    Generate embeddings with automatic retry for failed items
    """
    embeddings = generate_embeddings_batch_optimized(texts)
    
    # Identify failures
    failed_indices = [i for i, e in enumerate(embeddings) if e is None and texts[i].strip()]
    
    # Retry failed embeddings
    for retry in range(max_retries):
        if not failed_indices:
            break
        
        logger.info(f"Retry {retry + 1}/{max_retries} for {len(failed_indices)} failed embeddings")
        time.sleep(retry_delay)
        
        # Retry only failed items
        retry_texts = [texts[i] for i in failed_indices]
        retry_embeddings = generate_embeddings_batch_optimized(
            retry_texts,
            use_cache=False
        )
        
        # Update results
        new_failed = []
        for i, idx in enumerate(failed_indices):
            if retry_embeddings[i] is not None:
                embeddings[idx] = retry_embeddings[i]
            else:
                new_failed.append(idx)
        
        failed_indices = new_failed
        retry_delay *= 2
    
    if failed_indices:
        logger.error(f"Permanently failed embeddings: {len(failed_indices)} items")
    
    return embeddings

def generate_embeddings_safe(
    texts: List[str],
    fallback_to_zero: bool = False
) -> List[List[float]]:
    """
    Generate embeddings with guaranteed non-None return
    
    WARNING: Use only when you MUST have an embedding for every text.
    Fallback embeddings will hurt retrieval quality!
    
    Args:
        texts: List of texts to embed
        fallback_to_zero: Use zero vector (False) or skip embedding (True)
    
    Returns:
        List of embeddings (zero vectors for failures if fallback_to_zero=True)
    """
    embeddings = generate_embeddings_with_retry(texts)
    
    if fallback_to_zero:
        for i, emb in enumerate(embeddings):
            if emb is None:
                logger.warning(f"Using zero vector fallback for index {i}")
                embeddings[i] = [0.0] * PINECONE_DIMENSION
    
    return [e if e is not None else [0.0] * PINECONE_DIMENSION for e in embeddings]


# Comparison function for testing
def compare_batch_methods(texts: List[str]):
    """Compare threading vs native batching performance"""
    import time
    
    # Method 1: Threading (old approach)
    start = time.time()
    embeddings1 = generate_embeddings_batch(texts, max_workers=5)
    time1 = time.time() - start
    success1 = sum(1 for e in embeddings1 if e is not None)
    
    # Method 2: Native batching (recommended)
    start = time.time()
    embeddings2 = generate_embeddings_batch_optimized(texts, batch_size=100)
    time2 = time.time() - start
    success2 = sum(1 for e in embeddings2 if e is not None)
    
    print(f"\nThreading approach: {time1:.2f}s, {success1}/{len(texts)} successful")
    print(f"Batching approach: {time2:.2f}s, {success2}/{len(texts)} successful")
    print(f"Speed improvement: {((time1 - time2) / time1 * 100):.1f}%")






####################################




def normalize_embedding(embedding: List[float]) -> List[float]:
    """Normalize embedding to unit length"""
    embedding_array = np.array(embedding)
    norm = np.linalg.norm(embedding_array)
    
    if norm == 0:
        logger.warning("Zero-norm embedding detected")
        return embedding
    
    normalized = (embedding_array / norm).tolist()
    return normalized

@lru_cache(maxsize=CACHE_SIZE)
def _get_cached_embedding(text_hash: str) -> Optional[List[float]]:
    """Get embedding from cache"""
    with _cache_lock:
        return _embedding_cache.get(text_hash)

def _cache_embedding(text_hash: str, embedding: List[float]) -> None:
    """Store embedding in cache"""
    with _cache_lock:
        # Limit cache size
        if len(_embedding_cache) > CACHE_SIZE:
            # Remove oldest 20% of cache
            items_to_remove = len(_embedding_cache) // 5
            for key in list(_embedding_cache.keys())[:items_to_remove]:
                del _embedding_cache[key]
        
        _embedding_cache[text_hash] = embedding

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True
)

def get_embedding_with_fallback(
    text: str,
    use_cache: bool = True
) -> List[float]:
    """
    Generate embedding with guaranteed return (uses zero vector as fallback)
    Use this ONLY when you absolutely need an embedding
    
    Warning: Returning a fallback embedding can lead to poor retrieval results
    """
    embedding = generate_embedding(text, use_cache=use_cache)
    
    if embedding is None:
        logger.warning(f"Using zero vector fallback for text: {text[:50]}...")
        # Return zero vector instead of random (more predictable behavior)
        return [0.0] * PINECONE_DIMENSION
    
    return embedding

def validate_embedding(embedding: Optional[List[float]]) -> bool:
    """Validate embedding quality"""
    if not embedding:
        return False
    
    if len(embedding) != PINECONE_DIMENSION:
        logger.error(f"Invalid embedding dimension: {len(embedding)}")
        return False
    
    if all(x == 0 for x in embedding):
        logger.warning("Zero vector detected")
        return False
    
    embedding_array = np.array(embedding)
    if np.isnan(embedding_array).any() or np.isinf(embedding_array).any():
        logger.error("Invalid values (NaN/Inf) in embedding")
        return False
    
    return True


# Hybrid search helper
def create_hybrid_embedding(
    text: str,
    boost_terms: Optional[List[str]] = None
) -> Optional[List[float]]:
    """Create embedding with optional keyword boosting"""
    base_embedding = generate_embedding(text)
    if base_embedding is None:
        return None
    
    if not boost_terms:
        return base_embedding
    
    boosted_text = text
    for term in boost_terms:
        if term.lower() in text.lower():
            boosted_text += f" {term} " * 2
    
    boosted_embedding = generate_embedding(boosted_text, use_cache=False)
    if boosted_embedding is None:
        return base_embedding
    
    blended = [
        0.7 * base + 0.3 * boost
        for base, boost in zip(base_embedding, boosted_embedding)
    ]
    
    return normalize_embedding(blended)







# -----------------------------
# Document Processing
# -----------------------------


# def process_pdf(file_path: str, filename: str) -> Dict:
#     """
#     Enhanced PDF processing with better chunking and metadata
#     """
#     try:
#         logger.info(f"Processing PDF: {filename}")
#         reader = PdfReader(file_path)
#         total_pages = len(reader.pages)
        
#         # Extract all pages with structure
#         all_pages_content = []
#         page_texts = []
        
#         for i in range(total_pages):
#             try:
#                 page_text = reader.pages[i].extract_text()
#                 if page_text and page_text.strip():
#                     cleaned_page = clean_text(page_text)
#                     all_pages_content.append(f"[Page {i+1}]\n{page_text}")
#                     page_texts.append({
#                         'page': i + 1,
#                         'text': cleaned_page,
#                         'raw_text': page_text,  # Keep raw for better context
#                         'char_count': len(cleaned_page)
#                     })
#             except Exception as e:
#                 logger.warning(f"Failed to extract page {i+1}: {e}")
#                 continue
        
#         if not page_texts:
#             return {'success': False, 'error': 'No text could be extracted from PDF'}
        
#         # Store complete document with better formatting
#         complete_document = "\n\n".join(all_pages_content)
#         with document_store['lock']:
#             document_store['full_content'][filename] = {
#                 'content': complete_document,
#                 'pages': total_pages,
#                 'type': 'PDF',
#                 'page_breakdown': page_texts
#             }
#             _cache['full_documents'][filename] = complete_document
        
#         # Generate document ID
#         doc_id = hashlib.md5(f"{filename}{total_pages}".encode()).hexdigest()[:8]
        
#         # Strategy 1: Page-level vectors (PRIMARY for PDFs)
#         page_vectors = create_enhanced_page_vectors(page_texts, filename, doc_id)
        
#         # Strategy 2: Semantic chunks with better boundaries
#         semantic_chunks = create_semantic_pdf_chunks(page_texts, filename, doc_id)
        
#         # Strategy 3: Dense overlapping chunks for comprehensive coverage
#         dense_chunks = create_dense_pdf_chunks(page_texts, filename, doc_id)
        
#         # Combine all strategies
#         all_chunks = page_vectors + semantic_chunks + dense_chunks
        
#         logger.info(f"Created {len(all_chunks)} total chunks ({len(page_vectors)} pages, {len(semantic_chunks)} semantic, {len(dense_chunks)} dense)")
        
#         # Generate embeddings in batch
#         chunk_texts = [c['text'] for c in all_chunks]
#         logger.info(f"Generating embeddings for {len(chunk_texts)} chunks...")
        
#         embeddings = generate_embeddings_batch_optimized(
#             chunk_texts,
#             show_progress=True
#         )
        
#         # Prepare vectors for Pinecone
#         vectors = []
#         vector_ids = []
#         successful_embeddings = 0
        
#         for i, (chunk_data, embedding) in enumerate(zip(all_chunks, embeddings)):
#             if embedding is None:
#                 logger.warning(f"Skipping chunk {i} - embedding failed")
#                 continue
            
#             vector_id = f"{doc_id}_{chunk_data['chunk_type']}_{chunk_data.get('page_start', i)}__{i}"
#             vector_ids.append(vector_id)
            
#             # Enhanced metadata for better retrieval
#             metadata = {
#                 'filename': filename,
#                 'doc_id': doc_id,
#                 'chunk_index': i,
#                 'chunk_type': chunk_data['chunk_type'],
#                 'content': chunk_data['text'][:MAX_METADATA_LENGTH],
#                 'full_content': chunk_data['text'][:1000],  # Store more for context
#                 'file_type': 'pdf',
#                 'page_start': chunk_data.get('page_start'),
#                 'page_end': chunk_data.get('page_end'),
#                 'total_pages': total_pages,
#                 'section': chunk_data.get('section', ''),
#                 'has_numbers': chunk_data.get('has_numbers', False),
#                 'has_dates': chunk_data.get('has_dates', False),
#                 'has_financial': chunk_data.get('has_financial', False),
#                 'token_count': chunk_data.get('token_count', 0),
#                 'importance_score': chunk_data.get('importance_score', 0.5)
#             }
            
#             vectors.append({
#                 'id': vector_id,
#                 'values': embedding,
#                 'metadata': metadata
#             })
#             successful_embeddings += 1
        
#         logger.info(f"Successfully created {successful_embeddings}/{len(all_chunks)} vectors")
        
#         return {
#             'success': True,
#             'doc_id': doc_id,
#             'filename': filename,
#             'vectors': vectors,
#             'vector_ids': vector_ids,
#             'chunks': len(vectors),
#             'total_pages': total_pages,
#             'type': 'PDF Document',
#             'pages_processed': len(page_texts)
#         }
        
#     except Exception as e:
#         logger.error(f"Error processing PDF {filename}: {e}", exc_info=True)
#         return {'success': False, 'error': str(e)}


# def process_pdf(file_path: str, filename: str) -> Dict:
#     """
#     Fixed PDF processing - creates multiple indexing strategies
#     """
#     try:
#         logger.info(f"Processing PDF: {filename}")
#         reader = PdfReader(file_path)
#         total_pages = len(reader.pages)
        
#         # Extract ALL pages
#         all_pages_content = []
#         page_texts = []
        
#         for i in range(total_pages):
#             try:
#                 page_text = reader.pages[i].extract_text()
#                 if page_text and page_text.strip():
#                     all_pages_content.append(f"[Page {i+1}]\n{page_text}")
#                     page_texts.append({
#                         'page': i + 1,
#                         'text': clean_text(page_text),
#                         'raw': page_text,
#                         'char_count': len(page_text)
#                     })
#             except Exception as e:
#                 logger.warning(f"Failed to extract page {i+1}: {e}")
#                 continue
        
#         if not page_texts:
#             return {'success': False, 'error': 'No text extracted from PDF'}
        
#         # Store complete document
#         complete_document = "\n\n".join(all_pages_content)
#         with document_store['lock']:
#             document_store['full_content'][filename] = {
#                 'content': complete_document,
#                 'pages': total_pages,
#                 'type': 'PDF',
#                 'page_breakdown': page_texts
#             }
#             _cache['full_documents'][filename] = complete_document
        
#         doc_id = hashlib.md5(f"{filename}{total_pages}".encode()).hexdigest()[:8]
        
#         # Create ALL chunks
#         all_chunks = []
        
#         # Strategy 1: FULL PAGE vectors (most important!)
#         for page_data in page_texts:
#             page_num = page_data['page']
#             page_text = page_data['raw']  # Use raw, unprocessed text
            
#             # Create comprehensive page chunk with MORE content
#             page_chunk = f"""DOCUMENT: {filename}
# PAGE: {page_num} of {total_pages}

# FULL PAGE CONTENT:
# {page_text}

# This is page {page_num} from {filename}. Total pages: {total_pages}."""
            
#             all_chunks.append({
#                 'text': page_chunk,
#                 'chunk_type': 'page',
#                 'page_start': page_num,
#                 'page_end': page_num
#             })
        
#         # Strategy 2: OVERLAPPING chunks across pages
#         full_text = "\n\n".join([p['raw'] for p in page_texts])
        
#         # Create MANY overlapping chunks (more chunks = better search)
#         chunk_size = 2000  # chars
#         overlap = 1000     # 50% overlap!
        
#         for i in range(0, len(full_text), chunk_size - overlap):
#             chunk_text = full_text[i:i + chunk_size]
            
#             if len(chunk_text.strip()) < 100:
#                 continue
            
#             # Find page range
#             page_start = 1
#             cumulative = 0
#             for p in page_texts:
#                 if cumulative > i:
#                     page_start = p['page']
#                     break
#                 cumulative += p['char_count']
            
#             chunk_with_context = f"""DOCUMENT: {filename}
# PAGES: Around page {page_start}

# CONTENT:
# {chunk_text}"""
            
#             all_chunks.append({
#                 'text': chunk_with_context,
#                 'chunk_type': 'overlap',
#                 'page_start': page_start,
#                 'page_end': min(page_start + 1, total_pages)
#             })
        
#         logger.info(f"Created {len(all_chunks)} chunks for {filename}")
        
#         # Generate embeddings
#         chunk_texts = [c['text'] for c in all_chunks]
#         embeddings = generate_embeddings_batch_optimized(chunk_texts, show_progress=True)
        
#         # Create vectors
#         vectors = []
#         vector_ids = []
        
#         for i, (chunk_data, embedding) in enumerate(zip(all_chunks, embeddings)):
#             if embedding is None:
#                 continue
            
#             vector_id = f"{doc_id}_{chunk_data['chunk_type']}_p{chunk_data['page_start']}_{i}"
#             vector_ids.append(vector_id)
            
#             vectors.append({
#                 'id': vector_id,
#                 'values': embedding,
#                 'metadata': {
#                     'filename': filename,
#                     'doc_id': doc_id,
#                     'chunk_index': i,
#                     'chunk_type': chunk_data['chunk_type'],
#                     'content': chunk_data['text'][:500],  # Store first 500 chars
#                     'full_text': chunk_data['text'][:2000],  # Store more!
#                     'file_type': 'pdf',
#                     'page_start': chunk_data['page_start'],
#                     'page_end': chunk_data['page_end'],
#                     'total_pages': total_pages
#                 }
#             })
        
#         logger.info(f"Successfully created {len(vectors)} vectors")
        
#         return {
#             'success': True,
#             'doc_id': doc_id,
#             'filename': filename,
#             'vectors': vectors,
#             'vector_ids': vector_ids,
#             'chunks': len(vectors),
#             'total_pages': total_pages,
#             'type': 'PDF Document',
#             'pages_processed': len(page_texts)
#         }
        
#     except Exception as e:
#         logger.error(f"Error processing PDF {filename}: {e}", exc_info=True)
#         return {'success': False, 'error': str(e)}


def process_pdf(file_path: str, filename: str) -> Dict:
    """
    FIXED PDF processing with better content preservation
    """
    try:
        logger.info(f"Processing PDF: {filename}")
        reader = PdfReader(file_path)
        total_pages = len(reader.pages)
        
        if total_pages == 0:
            return {'success': False, 'error': 'PDF has no pages'}
        
        # Extract ALL pages with full content
        all_pages_content = []
        page_texts = []
        
        for i in range(total_pages):
            try:
                page_text = reader.pages[i].extract_text()
                if page_text and page_text.strip():
                    # Keep raw text for better searching
                    all_pages_content.append(f"[Page {i+1}]\n{page_text}")
                    page_texts.append({
                        'page': i + 1,
                        'text': clean_text(page_text),
                        'raw': page_text,
                        'char_count': len(page_text)
                    })
            except Exception as e:
                logger.warning(f"Failed to extract page {i+1}: {e}")
                continue
        
        if not page_texts:
            return {'success': False, 'error': 'No text extracted from PDF'}
        
        # Store complete document
        complete_document = "\n\n".join(all_pages_content)
        with document_store['lock']:
            document_store['full_content'][filename] = {
                'content': complete_document,
                'pages': total_pages,
                'type': 'PDF',
                'page_breakdown': page_texts
            }
            _cache['full_documents'][filename] = complete_document
        
        doc_id = hashlib.md5(f"{filename}{total_pages}".encode()).hexdigest()[:8]
        
        # Strategy 1: Full page vectors (MOST IMPORTANT)
        all_chunks = []
        for page_data in page_texts:
            page_num = page_data['page']
            page_text = page_data['raw']
            
            # Create rich page representation with MORE context
            page_chunk = f"""Document: {filename}
Page: {page_num} of {total_pages}

FULL PAGE CONTENT:
{page_text}

[This is page {page_num} from {filename}]"""
            
            all_chunks.append({
                'text': page_chunk,
                'chunk_type': 'page',
                'page_start': page_num,
                'page_end': page_num,
                'importance': 1.0  # Highest priority
            })
        
        # Strategy 2: Overlapping chunks (for comprehensive coverage)
        full_text = "\n\n".join([p['raw'] for p in page_texts])
        
        # More aggressive chunking with overlap
        chunk_size = 1500  # chars
        overlap = 750      # 50% overlap for better coverage
        
        for i in range(0, len(full_text), chunk_size - overlap):
            chunk_text = full_text[i:i + chunk_size]
            
            if len(chunk_text.strip()) < 100:
                continue
            
            # Find approximate page
            page_start = 1
            cumulative = 0
            for p in page_texts:
                if cumulative <= i < cumulative + p['char_count']:
                    page_start = p['page']
                    break
                cumulative += p['char_count']
            
            overlap_chunk = f"""Document: {filename}
Location: Around page {page_start}

CONTENT:
{chunk_text}"""
            
            all_chunks.append({
                'text': overlap_chunk,
                'chunk_type': 'overlap',
                'page_start': page_start,
                'page_end': min(page_start + 1, total_pages),
                'importance': 0.8
            })
        
        logger.info(f"Created {len(all_chunks)} chunks for {filename}")

        # Generate embeddings with RETRY logic to ensure NO data is missed
        chunk_texts = [c['text'] for c in all_chunks]

        logger.info(f"Generating embeddings for {len(chunk_texts)} chunks with retry logic...")

        try:
            # Use generate_embeddings_with_retry for maximum reliability
            embeddings = generate_embeddings_with_retry(
                chunk_texts,
                max_retries=3,
                retry_delay=2.0
            )
        except Exception as e:
            logger.error(f"Embedding generation failed after retries: {e}")
            return {'success': False, 'error': f'Embedding generation failed: {str(e)}'}

        # Create vectors with MANDATORY retry for failed embeddings
        vectors = []
        vector_ids = []
        successful = 0
        failed_indices = []

        for i, (chunk_data, embedding) in enumerate(zip(all_chunks, embeddings)):
            if embedding is None:
                logger.warning(f"⚠️ Chunk {i} has no embedding - will retry")
                failed_indices.append(i)
                continue

            vector_id = f"{doc_id}_{chunk_data['chunk_type']}_p{chunk_data['page_start']}_{i}"
            vector_ids.append(vector_id)

            # Store MORE content in metadata
            vectors.append({
                'id': vector_id,
                'values': embedding,
                'metadata': {
                    'filename': filename,
                    'doc_id': doc_id,
                    'chunk_index': i,
                    'chunk_type': chunk_data['chunk_type'],
                    'content': chunk_data['text'][:500],
                    'full_text': chunk_data['text'][:2000],  # Store more!
                    'file_type': 'pdf',
                    'page_start': chunk_data['page_start'],
                    'page_end': chunk_data['page_end'],
                    'total_pages': total_pages,
                    'importance': chunk_data['importance']
                }
            })
            successful += 1

        # CRITICAL: Retry failed embeddings individually
        if failed_indices:
            logger.warning(f"⚠️ Retrying {len(failed_indices)} failed embeddings individually...")
            for idx in failed_indices:
                chunk_data = all_chunks[idx]
                retry_count = 0
                max_individual_retries = 5

                while retry_count < max_individual_retries:
                    try:
                        time.sleep(1)  # Rate limiting
                        embedding = generate_embedding(chunk_texts[idx], use_cache=False)

                        if embedding is not None:
                            vector_id = f"{doc_id}_{chunk_data['chunk_type']}_p{chunk_data['page_start']}_{idx}"
                            vector_ids.append(vector_id)

                            vectors.append({
                                'id': vector_id,
                                'values': embedding,
                                'metadata': {
                                    'filename': filename,
                                    'doc_id': doc_id,
                                    'chunk_index': idx,
                                    'chunk_type': chunk_data['chunk_type'],
                                    'content': chunk_data['text'][:500],
                                    'full_text': chunk_data['text'][:2000],
                                    'file_type': 'pdf',
                                    'page_start': chunk_data['page_start'],
                                    'page_end': chunk_data['page_end'],
                                    'total_pages': total_pages,
                                    'importance': chunk_data['importance']
                                }
                            })
                            successful += 1
                            logger.info(f"✅ Successfully retried chunk {idx}")
                            break
                    except Exception as e:
                        retry_count += 1
                        logger.warning(f"Retry {retry_count}/{max_individual_retries} failed for chunk {idx}: {e}")
                        time.sleep(retry_count * 2)

                if retry_count >= max_individual_retries:
                    logger.error(f"❌ FAILED to embed chunk {idx} after {max_individual_retries} retries - DATA LOST")

        # Report final stats
        coverage_percent = (successful / len(all_chunks)) * 100
        logger.info(f"📊 Final stats: {successful}/{len(all_chunks)} vectors created ({coverage_percent:.1f}% coverage)")

        if successful == 0:
            return {'success': False, 'error': 'All embeddings failed'}

        # Warn if not 100% coverage
        if successful < len(all_chunks):
            logger.warning(f"⚠️ WARNING: Only {successful}/{len(all_chunks)} chunks embedded. {len(all_chunks) - successful} chunks were lost!")
        
        logger.info(f"Successfully created {successful}/{len(all_chunks)} vectors")
        
        return {
            'success': True,
            'doc_id': doc_id,
            'filename': filename,
            'vectors': vectors,
            'vector_ids': vector_ids,
            'chunks': len(vectors),
            'total_pages': total_pages,
            'type': 'PDF Document',
            'pages_processed': len(page_texts)
        }
        
    except Exception as e:
        logger.error(f"Error processing PDF {filename}: {e}", exc_info=True)
        return {'success': False, 'error': str(e)}


def create_enhanced_page_vectors(
    page_texts: List[Dict],
    filename: str,
    doc_id: str
) -> List[Dict]:
    """
    Create enhanced page-level vectors with more context
    These are PRIMARY for PDF retrieval
    """
    page_vectors = []
    
    for i, page_data in enumerate(page_texts):
        page_num = page_data['page']
        page_text = page_data['text']
        
        # Detect content characteristics
        has_numbers = bool(re.search(r'\d+', page_text))
        has_dates = bool(re.search(r'\d{4}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', page_text))
        has_financial = bool(re.search(r'\$|revenue|profit|cost|expense|sales', page_text.lower()))
        
        # Add context from adjacent pages
        context_text = page_text
        if i > 0:
            prev_text = page_texts[i-1]['text'][:200]
            context_text = f"[Previous page context: {prev_text}...]\n\n{context_text}"
        if i < len(page_texts) - 1:
            next_text = page_texts[i+1]['text'][:200]
            context_text = f"{context_text}\n\n[Next page context: {next_text}...]"
        
        # Create rich page representation
        page_representation = f"""Document: {filename}
Page Number: {page_num} of {len(page_texts)}
Page Type: {'Data/Numbers' if has_numbers else 'Text'}

Full Page Content:
{context_text}"""
        
        page_vectors.append({
            'text': page_representation,
            'chunk_type': 'page',
            'page_start': page_num,
            'page_end': page_num,
            'section': f'Page {page_num}',
            'has_numbers': has_numbers,
            'has_dates': has_dates,
            'has_financial': has_financial,
            'token_count': count_tokens(page_text),
            'importance_score': 0.9  # High importance for page vectors
        })
    
    return page_vectors


def create_semantic_pdf_chunks(
    page_texts: List[Dict],
    filename: str,
    doc_id: str
) -> List[Dict]:
    """
    Create semantic chunks that respect paragraph/section boundaries
    """
    chunks = []
    
    for page_data in page_texts:
        page_num = page_data['page']
        page_text = page_data['raw_text']  # Use raw text for better paragraph detection
        
        # Split by paragraphs (double newlines or significant breaks)
        paragraphs = re.split(r'\n\s*\n', page_text)
        paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > 50]
        
        # Combine paragraphs into semantic chunks
        current_chunk = []
        current_length = 0
        target_length = 600  # tokens
        
        for para in paragraphs:
            para_tokens = count_tokens(para)
            
            if current_length + para_tokens > target_length and current_chunk:
                # Save current chunk
                chunk_text = '\n\n'.join(current_chunk)
                
                contextualized_text = f"""Document: {filename}
Page: {page_num}
Section: Semantic Chunk

{chunk_text}"""
                
                chunks.append({
                    'text': contextualized_text,
                    'chunk_type': 'semantic',
                    'page_start': page_num,
                    'page_end': page_num,
                    'section': f'Page {page_num} - Semantic',
                    'has_numbers': bool(re.search(r'\d+', chunk_text)),
                    'has_dates': bool(re.search(r'\d{4}', chunk_text)),
                    'has_financial': bool(re.search(r'\$|revenue', chunk_text.lower())),
                    'token_count': current_length,
                    'importance_score': 0.7
                })
                
                # Start new chunk with overlap (keep last paragraph)
                current_chunk = [current_chunk[-1], para] if current_chunk else [para]
                current_length = count_tokens(current_chunk[-1]) + para_tokens
            else:
                current_chunk.append(para)
                current_length += para_tokens
        
        # Add remaining chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            contextualized_text = f"""Document: {filename}
Page: {page_num}
Section: Semantic Chunk

{chunk_text}"""
            
            chunks.append({
                'text': contextualized_text,
                'chunk_type': 'semantic',
                'page_start': page_num,
                'page_end': page_num,
                'section': f'Page {page_num} - Semantic',
                'has_numbers': bool(re.search(r'\d+', chunk_text)),
                'has_dates': bool(re.search(r'\d{4}', chunk_text)),
                'has_financial': bool(re.search(r'\$|revenue', chunk_text.lower())),
                'token_count': current_length,
                'importance_score': 0.7
            })
    
    return chunks


def create_dense_pdf_chunks(
    page_texts: List[Dict],
    filename: str,
    doc_id: str
) -> List[Dict]:
    """
    Create dense overlapping chunks for comprehensive coverage
    High overlap ensures no information is lost
    """
    chunks = []
    
    # Combine all text
    full_text = " ".join([p['text'] for p in page_texts])
    
    # Character-based chunking with high overlap
    chunk_size = 3000  # characters
    overlap = 1500     # 50% overlap
    
    for i in range(0, len(full_text), chunk_size - overlap):
        chunk_text = full_text[i:i + chunk_size]
        
        if len(chunk_text.strip()) < 200:
            continue
        
        # Find which page(s) this chunk spans
        char_position = i
        page_start, page_end = find_page_range_from_chars(char_position, page_texts)
        
        contextualized_text = f"""Document: {filename}
Pages: {page_start} to {page_end}
Dense Chunk

{chunk_text}"""
        
        chunks.append({
            'text': contextualized_text,
            'chunk_type': 'dense',
            'page_start': page_start,
            'page_end': page_end,
            'section': f'Pages {page_start}-{page_end}',
            'has_numbers': bool(re.search(r'\d+', chunk_text)),
            'has_dates': bool(re.search(r'\d{4}', chunk_text)),
            'has_financial': bool(re.search(r'\$|revenue', chunk_text.lower())),
            'token_count': count_tokens(chunk_text),
            'importance_score': 0.6
        })
    
    return chunks


def find_page_range_from_chars(char_position: int, page_texts: List[Dict]) -> tuple:
    """Find which pages a character position spans"""
    cumulative = 0
    start_page = 1
    
    for page in page_texts:
        cumulative += page['char_count']
        if cumulative > char_position:
            start_page = page['page']
            break
    
    # Estimate end page (assume chunk spans ~2 pages)
    end_page = min(start_page + 1, page_texts[-1]['page'])
    return start_page, end_page


def create_pdf_chunks_optimized(
    page_texts: List[Dict],
    filename: str,
    doc_id: str
) -> List[Dict]:
    """
    Create overlapping chunks with context preservation
    """
    chunks = []
    
    # Combine all text
    full_text = " ".join([p['text'] for p in page_texts])
    
    # Detect potential sections (basic heading detection)
    current_section = "Introduction"
    
    # Token-aware chunking
    tokens = full_text.split()
    
    for i in range(0, len(tokens), PDF_CHUNK_SIZE - PDF_OVERLAP):
        chunk_tokens = tokens[i:i + PDF_CHUNK_SIZE]
        chunk_text = " ".join(chunk_tokens)
        
        # Find which pages this chunk spans
        char_position = len(" ".join(tokens[:i]))
        page_start, page_end = find_page_range(char_position, page_texts)
        
        # Detect if chunk contains numbers/data
        has_numbers = bool(re.search(r'\d+', chunk_text))
        
        # Add context header
        contextualized_text = f"""Document: {filename}
Pages: {page_start} to {page_end}
Section: {current_section}

{chunk_text}"""
        
        chunks.append({
            'text': contextualized_text,
            'chunk_type': 'content',
            'page_start': page_start,
            'page_end': page_end,
            'section': current_section,
            'has_numbers': has_numbers,
            'token_count': len(chunk_tokens)
        })
    
    return chunks


def create_page_level_vectors(
    page_texts: List[Dict],
    filename: str,
    doc_id: str
) -> List[Dict]:
    """
    Create one vector per page for quick page-specific retrieval
    """
    page_vectors = []
    
    for page_data in page_texts:
        page_num = page_data['page']
        page_text = page_data['text']
        
        # Add page context
        contextualized = f"""Document: {filename}
Page: {page_num}

{page_text[:1500]}"""  # Limit to avoid token overflow
        
        page_vectors.append({
            'text': contextualized,
            'chunk_type': 'page',
            'page_start': page_num,
            'page_end': page_num,
            'section': f'Page {page_num}',
            'has_numbers': bool(re.search(r'\d+', page_text)),
            'token_count': count_tokens(page_text)
        })
    
    return page_vectors


def find_page_range(char_position: int, page_texts: List[Dict]) -> tuple:
    """Find which pages a character position spans"""
    cumulative = 0
    start_page = 1
    
    for page in page_texts:
        cumulative += page['char_count']
        if cumulative > char_position:
            start_page = page['page']
            break
    
    end_page = min(start_page + 1, page_texts[-1]['page'])
    return start_page, end_page



def process_csv(file_path: str, filename: str) -> Dict:
    """
    Optimized CSV/Excel processing with enhanced error handling
    """
    try:
        logger.info(f"Processing CSV/Excel: {filename}")
        
        # Read file with robust error handling
        df = read_file_with_encoding(file_path, filename)
        
        if df is None:
            return {
                'success': False,
                'error': 'Could not read file. Please check the file format and ensure required libraries are installed (pip install openpyxl xlrd)'
            }
        
        if df.empty:
            return {
                'success': False,
                'error': 'File is empty or contains no readable data'
            }
        
        logger.info(f"Read {len(df)} rows and {len(df.columns)} columns")
        
        # Clean column names
        df.columns = [str(col).strip() for col in df.columns]
        
        # Remove completely empty rows
        df = df.dropna(how='all')
        
        if df.empty:
            return {
                'success': False,
                'error': 'File contains only empty rows'
            }
        
        logger.info(f"After cleaning: {len(df)} rows")
        
        # Detect column types with error handling
        try:
            column_info = analyze_columns(df)
        except Exception as e:
            logger.error(f"Column analysis failed: {e}")
            column_info = {}
        
        # Store full DataFrame - COMPLETE DATA, not sampled
        full_content = df.to_string(max_rows=None)  # No limit!
        with document_store['lock']:
            document_store['full_content'][filename] = {
                'content': full_content,
                'rows': len(df),
                'columns': list(df.columns),
                'column_info': column_info,
                'type': 'CSV/Excel',
                'dataframe': df.to_dict('records')  # Store ALL rows, not just 100
            }
            _cache['full_documents'][filename] = full_content
            _cache['csv_lookup_cache'][filename] = df  # Complete dataframe
        
        doc_id = hashlib.md5(f"{filename}{len(df)}".encode()).hexdigest()[:8]
        
        # Prepare all chunks for batch embedding
        all_chunks = []
        
        # Strategy 1: Row-level indexing
        logger.info(f"Creating row-level vectors...")
        try:
            row_chunks = create_row_vectors(df, filename, doc_id, column_info)
            all_chunks.extend(row_chunks)
            logger.info(f"Created {len(row_chunks)} row vectors")
        except Exception as e:
            logger.error(f"Row vector creation failed: {e}")
        
        # Strategy 2: Column summaries
        logger.info(f"Creating column summary vectors...")
        try:
            column_chunks = create_column_vectors(df, filename, doc_id, column_info)
            all_chunks.extend(column_chunks)
            logger.info(f"Created {len(column_chunks)} column vectors")
        except Exception as e:
            logger.error(f"Column vector creation failed: {e}")
        
        # Strategy 3: Table chunks
        logger.info(f"Creating table chunk vectors...")
        try:
            table_chunks = create_table_chunks(df, filename, doc_id, column_info)
            all_chunks.extend(table_chunks)
            logger.info(f"Created {len(table_chunks)} table chunks")
        except Exception as e:
            logger.error(f"Table chunk creation failed: {e}")
        
        if not all_chunks:
            return {
                'success': False,
                'error': 'Failed to create any chunks from the file'
            }
        
        logger.info(f"Total chunks created: {len(all_chunks)}")

        # Generate embeddings with RETRY logic
        chunk_texts = [c['text'] for c in all_chunks]
        logger.info(f"Generating embeddings for {len(chunk_texts)} chunks with retry logic...")

        try:
            embeddings = generate_embeddings_with_retry(
                chunk_texts,
                max_retries=3,
                retry_delay=2.0
            )
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return {
                'success': False,
                'error': f'Embedding generation failed: {str(e)}'
            }

        # Prepare vectors with MANDATORY retry for failures
        vectors = []
        vector_ids = []
        successful = 0
        failed_indices = []

        for i, (chunk_data, embedding) in enumerate(zip(all_chunks, embeddings)):
            if embedding is None:
                logger.warning(f"⚠️ Chunk {i} has no embedding - will retry")
                failed_indices.append(i)
                continue

            vector_id = f"{doc_id}_{chunk_data['chunk_type']}_{i}"
            vector_ids.append(vector_id)

            metadata = {
                'filename': filename,
                'doc_id': doc_id,
                'chunk_index': i,
                'type': chunk_data['chunk_type'],
                'chunk_type': chunk_data['chunk_type'],  # Add this for compatibility
                'content': chunk_data['text'][:MAX_METADATA_LENGTH],
                'full_text': chunk_data['text'][:2000],  # Store more content
                'file_type': 'csv'
            }

            metadata.update(chunk_data.get('metadata', {}))

            vectors.append({
                'id': vector_id,
                'values': embedding,
                'metadata': metadata
            })
            successful += 1

        # CRITICAL: Retry failed embeddings individually
        if failed_indices:
            logger.warning(f"⚠️ Retrying {len(failed_indices)} failed embeddings individually...")
            for idx in failed_indices:
                chunk_data = all_chunks[idx]
                retry_count = 0
                max_individual_retries = 5

                while retry_count < max_individual_retries:
                    try:
                        time.sleep(1)  # Rate limiting
                        embedding = generate_embedding(chunk_texts[idx], use_cache=False)

                        if embedding is not None:
                            vector_id = f"{doc_id}_{chunk_data['chunk_type']}_{idx}"
                            vector_ids.append(vector_id)

                            metadata = {
                                'filename': filename,
                                'doc_id': doc_id,
                                'chunk_index': idx,
                                'type': chunk_data['chunk_type'],
                                'chunk_type': chunk_data['chunk_type'],
                                'content': chunk_data['text'][:MAX_METADATA_LENGTH],
                                'full_text': chunk_data['text'][:2000],
                                'file_type': 'csv'
                            }
                            metadata.update(chunk_data.get('metadata', {}))

                            vectors.append({
                                'id': vector_id,
                                'values': embedding,
                                'metadata': metadata
                            })
                            successful += 1
                            logger.info(f"✅ Successfully retried chunk {idx}")
                            break
                    except Exception as e:
                        retry_count += 1
                        logger.warning(f"Retry {retry_count}/{max_individual_retries} failed for chunk {idx}: {e}")
                        time.sleep(retry_count * 2)

                if retry_count >= max_individual_retries:
                    logger.error(f"❌ FAILED to embed chunk {idx} after {max_individual_retries} retries - DATA LOST")

        # Report final stats
        coverage_percent = (successful / len(all_chunks)) * 100
        logger.info(f"📊 Final stats: {successful}/{len(all_chunks)} vectors created ({coverage_percent:.1f}% coverage)")

        if successful == 0:
            return {
                'success': False,
                'error': 'All embeddings failed - please check OpenAI API'
            }

        # Warn if not 100% coverage
        if successful < len(all_chunks):
            logger.warning(f"⚠️ WARNING: Only {successful}/{len(all_chunks)} chunks embedded. {len(all_chunks) - successful} chunks were lost!")
        
        logger.info(f"Successfully created {successful}/{len(all_chunks)} vectors")
        
        return {
            'success': True,
            'doc_id': doc_id,
            'filename': filename,
            'vectors': vectors,
            'vector_ids': vector_ids,
            'chunks': len(vectors),
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'column_info': column_info,
            'type': 'CSV/Excel Spreadsheet'
        }
        
    except Exception as e:
        logger.error(f"Error processing CSV/Excel {filename}: {e}", exc_info=True)
        return {
            'success': False,
            'error': f'Processing error: {str(e)}'
        }




# def read_file_with_encoding(file_path: str, filename: str) -> Optional[pd.DataFrame]:
#     """Robust file reading with encoding detection and multiple fallbacks"""
    
#     filename_lower = filename.lower()
    
#     # ============================================
#     # CSV FILES
#     # ============================================
#     if filename_lower.endswith('.csv'):
#         encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252', 'iso-8859-1']
        
#         for encoding in encodings:
#             try:
#                 df = pd.read_csv(file_path, encoding=encoding)
#                 logger.info(f"Successfully read CSV with {encoding}")
#                 return df
#             except (UnicodeDecodeError, Exception) as e:
#                 logger.debug(f"Failed with {encoding}: {e}")
#                 continue
        
#         # Try with chardet
#         try:
#             import chardet
#             with open(file_path, 'rb') as f:
#                 raw_data = f.read(100000)  # Read first 100KB for detection
#             detected = chardet.detect(raw_data)
#             encoding = detected.get('encoding', 'utf-8')
#             logger.info(f"Detected encoding: {encoding} (confidence: {detected.get('confidence', 0)})")
#             return pd.read_csv(file_path, encoding=encoding)
#         except ImportError:
#             logger.warning("chardet not available for encoding detection")
#         except Exception as e:
#             logger.debug(f"chardet detection failed: {e}")
        
#         # Last resort
#         try:
#             return pd.read_csv(file_path, encoding='utf-8', errors='replace')
#         except Exception as e:
#             logger.error(f"All CSV read attempts failed: {e}")
#             return None
    
#     # ============================================
#     # EXCEL FILES (.xlsx, .xls)
#     # ============================================
#     else:
#         # Try different approaches for Excel files
        
#         # Approach 1: Try openpyxl for .xlsx (most common)
#         if filename_lower.endswith('.xlsx'):
#             try:
#                 logger.info("Attempting to read .xlsx with openpyxl engine...")
#                 df = pd.read_excel(file_path, engine='openpyxl')
#                 logger.info(f"Successfully read Excel file with openpyxl: {len(df)} rows")
#                 return df
#             except ImportError as e:
#                 logger.error("openpyxl not installed. Install with: pip install openpyxl")
#                 return None
#             except Exception as e:
#                 logger.warning(f"openpyxl failed: {e}")
        
#         # Approach 2: Try xlrd for .xls (older Excel format)
#         if filename_lower.endswith('.xls'):
#             try:
#                 logger.info("Attempting to read .xls with xlrd engine...")
#                 df = pd.read_excel(file_path, engine='xlrd')
#                 logger.info(f"Successfully read Excel file with xlrd: {len(df)} rows")
#                 return df
#             except ImportError as e:
#                 logger.error("xlrd not installed. Install with: pip install xlrd")
#                 return None
#             except Exception as e:
#                 logger.warning(f"xlrd failed: {e}")
        
#         # Approach 3: Try default engine (auto-detect)
#         try:
#             logger.info("Attempting to read Excel with default engine...")
#             df = pd.read_excel(file_path)
#             logger.info(f"Successfully read Excel file with default engine: {len(df)} rows")
#             return df
#         except Exception as e:
#             logger.warning(f"Default engine failed: {e}")
        
#         # Approach 4: Try calamine (fast Excel reader)
#         try:
#             logger.info("Attempting to read Excel with calamine engine...")
#             df = pd.read_excel(file_path, engine='calamine')
#             logger.info(f"Successfully read Excel file with calamine: {len(df)} rows")
#             return df
#         except Exception as e:
#             logger.warning(f"calamine failed: {e}")
        
#         # Approach 5: Convert to CSV first (using external library)
#         try:
#             logger.info("Attempting to convert Excel to CSV first...")
#             import subprocess
#             import tempfile
            
#             # Create temp CSV
#             temp_csv = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
#             temp_csv_path = temp_csv.name
#             temp_csv.close()
            
#             # Try using ssconvert if available (from gnumeric)
#             try:
#                 subprocess.run(
#                     ['ssconvert', file_path, temp_csv_path],
#                     check=True,
#                     capture_output=True,
#                     timeout=30
#                 )
#                 df = pd.read_csv(temp_csv_path)
#                 os.unlink(temp_csv_path)
#                 logger.info(f"Successfully converted Excel to CSV: {len(df)} rows")
#                 return df
#             except (subprocess.CalledProcessError, FileNotFoundError):
#                 if os.path.exists(temp_csv_path):
#                     os.unlink(temp_csv_path)
#         except Exception as e:
#             logger.debug(f"Excel to CSV conversion failed: {e}")
        
#         # Approach 6: Try reading as binary and parsing manually
#         try:
#             logger.info("Attempting manual Excel parsing...")
#             import openpyxl
            
#             wb = openpyxl.load_workbook(file_path, data_only=True)
#             sheet = wb.active
            
#             # Extract data
#             data = []
#             for row in sheet.iter_rows(values_only=True):
#                 if any(cell is not None for cell in row):  # Skip empty rows
#                     data.append(row)
            
#             if not data:
#                 logger.error("No data found in Excel file")
#                 return None
            
#             # Create DataFrame
#             df = pd.DataFrame(data[1:], columns=data[0])
#             logger.info(f"Successfully parsed Excel manually: {len(df)} rows")
#             return df
            
#         except ImportError:
#             logger.error("openpyxl not available for manual parsing")
#         except Exception as e:
#             logger.warning(f"Manual Excel parsing failed: {e}")
        
#         # Final error
#         logger.error(f"Failed to read Excel file with any method. File: {filename}")
#         logger.error("Please ensure you have the required libraries installed:")
#         logger.error("  For .xlsx: pip install openpyxl")
#         logger.error("  For .xls: pip install xlrd")
#         logger.error("  Alternative: pip install python-calamine")
        
#         return None


def read_file_with_encoding(file_path: str, filename: str) -> Optional[pd.DataFrame]:
    """
    UPDATED: Enhanced file reading with multi-sheet Excel support
    """
    filename_lower = filename.lower()
    
    # CSV FILES
    if filename_lower.endswith('.csv'):
        encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                logger.info(f"Successfully read CSV with {encoding}")
                return df
            except Exception:
                continue
        
        # Try with chardet
        try:
            import chardet
            with open(file_path, 'rb') as f:
                raw_data = f.read(100000)
            detected = chardet.detect(raw_data)
            encoding = detected.get('encoding', 'utf-8')
            return pd.read_csv(file_path, encoding=encoding)
        except:
            pass
        
        # Last resort
        try:
            return pd.read_csv(file_path, encoding='utf-8', errors='replace')
        except Exception as e:
            logger.error(f"All CSV read attempts failed: {e}")
            return None
    
    # EXCEL FILES - Multi-sheet support
    else:
        # Try multi-sheet reading first
        df = read_excel_all_sheets(file_path, filename)
        if df is not None:
            return df
        
        # Fallback to single sheet
        if filename_lower.endswith('.xlsx'):
            try:
                df = pd.read_excel(file_path, engine='openpyxl')
                logger.info(f"Read single sheet Excel: {len(df)} rows")
                return df
            except ImportError:
                logger.error("openpyxl not installed. Install: pip install openpyxl")
                return None
            except Exception as e:
                logger.warning(f"Single sheet read failed: {e}")
        
        if filename_lower.endswith('.xls'):
            try:
                df = pd.read_excel(file_path, engine='xlrd')
                logger.info(f"Read .xls file: {len(df)} rows")
                return df
            except ImportError:
                logger.error("xlrd not installed. Install: pip install xlrd")
                return None
            except Exception as e:
                logger.warning(f".xls read failed: {e}")
        
        # Try default
        try:
            df = pd.read_excel(file_path)
            logger.info(f"Read with default engine: {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"All Excel read attempts failed: {e}")
            return None

def analyze_columns(df: pd.DataFrame) -> Dict[str, Dict]:
    """Analyze column types and statistics with error handling"""
    column_info = {}
    
    for col in df.columns:
        try:
            col_data = df[col].dropna()
            
            if len(col_data) == 0:
                column_info[col] = {
                    'dtype': 'empty',
                    'null_count': len(df[col]),
                    'unique_count': 0,
                    'is_numeric': False
                }
                continue
            
            info = {
                'dtype': str(col_data.dtype),
                'null_count': int(df[col].isnull().sum()),
                'unique_count': int(col_data.nunique()),
                'is_numeric': pd.api.types.is_numeric_dtype(col_data)
            }
            
            if info['is_numeric']:
                try:
                    info.update({
                        'min': float(col_data.min()),
                        'max': float(col_data.max()),
                        'mean': float(col_data.mean()),
                        'median': float(col_data.median())
                    })
                except Exception as e:
                    logger.warning(f"Failed to calculate numeric stats for {col}: {e}")
            else:
                try:
                    top_values = col_data.value_counts().head(5).to_dict()
                    info['top_values'] = {str(k): int(v) for k, v in top_values.items()}
                except Exception as e:
                    logger.warning(f"Failed to calculate top values for {col}: {e}")
                    info['top_values'] = {}
            
            column_info[col] = info
            
        except Exception as e:
            logger.error(f"Error analyzing column {col}: {e}")
            column_info[col] = {
                'dtype': 'error',
                'null_count': 0,
                'unique_count': 0,
                'is_numeric': False
            }
    
    return column_info




def create_row_vectors(
    df: pd.DataFrame,
    filename: str,
    doc_id: str,
    column_info: Dict
) -> List[Dict]:
    """Create one vector per row with rich context - INDEX ALL ROWS"""
    chunks = []

    # INDEX ALL ROWS - No sampling! Every row gets embedded
    df_sample = df
    logger.info(f"🔥 Indexing ALL {len(df)} rows from {filename}")

    # If file is HUGE (>10,000 rows), warn but still index everything
    if len(df) > 10000:
        logger.warning(f"⚠️ Large file with {len(df)} rows - this may take several minutes but ALL data will be indexed!")
    
    for idx, row in df_sample.iterrows():
        row_dict = {}
        row_parts = []
        
        for col in df.columns:
            value = row[col]
            if pd.notna(value):
                row_dict[col] = str(value)
                row_parts.append(f"{col}: {value}")
        
        # Create searchable text with context
        row_text = f"""File: {filename}
Row: {idx + 1}
Data: {'. '.join(row_parts)}"""
        
        # Detect if row contains important keywords
        row_str = ' '.join(row_parts).lower()
        has_financial = any(term in row_str for term in ['revenue', 'profit', 'sales', 'cost', 'price', '$'])
        has_date = bool(re.search(r'\d{4}|\d{1,2}/\d{1,2}', row_str))
        
        chunks.append({
            'text': row_text,
            'chunk_type': 'csv_row',
            'metadata': {
                'row_index': int(idx),
                'row_data': json.dumps(row_dict)[:MAX_METADATA_LENGTH],
                'has_financial': has_financial,
                'has_date': has_date,
                **{f"col_{col}": str(row[col])[:100] if pd.notna(row[col]) else "" 
                   for col in list(df.columns)[:5]}  # First 5 columns
            }
        })
    
    return chunks


def create_column_vectors(
    df: pd.DataFrame,
    filename: str,
    doc_id: str,
    column_info: Dict
) -> List[Dict]:
    """Create vectors for column summaries"""
    chunks = []
    
    for col, info in column_info.items():
        col_data = df[col].dropna()
        
        if len(col_data) == 0:
            continue
        
        # Create rich column description
        if info['is_numeric']:
            summary = f"""File: {filename}
Column: {col}
Type: Numeric
Statistics:
- Count: {len(col_data)}
- Min: {info['min']:.2f}
- Max: {info['max']:.2f}
- Mean: {info['mean']:.2f}
- Median: {info['median']:.2f}
- Unique values: {info['unique_count']}
Sample values: {col_data.head(10).tolist()}"""
        else:
            top_vals = info.get('top_values', {})
            summary = f"""File: {filename}
Column: {col}
Type: Text/Categorical
Statistics:
- Total values: {len(col_data)}
- Unique values: {info['unique_count']}
- Top values: {', '.join([f'{k} ({v})' for k, v in list(top_vals.items())[:5]])}
Sample values: {col_data.head(10).tolist()}"""
        
        chunks.append({
            'text': summary,
            'chunk_type': 'csv_column',
            'metadata': {
                'column_name': col,
                'data_type': info['dtype'],
                'is_numeric': info['is_numeric'],
                'unique_count': info['unique_count'],
                'null_count': info['null_count']
            }
        })
    
    return chunks


def create_table_chunks(
    df: pd.DataFrame,
    filename: str,
    doc_id: str,
    column_info: Dict
) -> List[Dict]:
    """Create table chunk vectors with overlap"""
    chunks = []
    chunk_size = CSV_ROWS_PER_CHUNK
    overlap = 3  # Row overlap
    
    for i in range(0, len(df), chunk_size - overlap):
        chunk_df = df.iloc[i:i + chunk_size]
        
        # Create markdown-style table
        chunk_text = f"""File: {filename}
Rows: {i + 1} to {min(i + chunk_size, len(df))} of {len(df)}
Columns: {', '.join(df.columns)}

{chunk_df.to_markdown(index=False)}"""
        
        chunks.append({
            'text': chunk_text,
            'chunk_type': 'csv_chunk',
            'metadata': {
                'chunk_index': i // chunk_size,
                'start_row': i + 1,
                'end_row': min(i + chunk_size, len(df)),
                'row_count': len(chunk_df)
            }
        })
    
    return chunks



# -----------------------------
# Upload to Pinecone with VERIFICATION
# -----------------------------
def upload_to_pinecone(vectors: List[Dict], verify: bool = True) -> bool:
    """
    Upload vectors to Pinecone in batches with VERIFICATION

    Args:
        vectors: List of vector dictionaries to upload
        verify: Whether to verify upload succeeded (default True)

    Returns:
        True if all vectors uploaded successfully, False otherwise
    """
    index = get_pinecone_index()
    if not index:
        logger.warning("Pinecone not available, skipping upload")
        return True  # Return True to not block

    if not vectors:
        logger.warning("No vectors to upload")
        return True

    total_vectors = len(vectors)
    logger.info(f"📤 Starting upload of {total_vectors} vectors to Pinecone...")

    try:
        uploaded_count = 0
        failed_batches = []

        # Upload in batches with retry
        for batch_idx in range(0, len(vectors), BATCH_SIZE):
            batch = vectors[batch_idx:batch_idx + BATCH_SIZE]
            batch_num = batch_idx // BATCH_SIZE + 1
            total_batches = (len(vectors) + BATCH_SIZE - 1) // BATCH_SIZE

            retry_count = 0
            max_retries = 3

            while retry_count < max_retries:
                try:
                    index.upsert(vectors=batch)
                    uploaded_count += len(batch)
                    logger.info(f"✅ Uploaded batch {batch_num}/{total_batches} ({uploaded_count}/{total_vectors} vectors)")
                    break
                except Exception as e:
                    retry_count += 1
                    if retry_count >= max_retries:
                        logger.error(f"❌ Batch {batch_num} FAILED after {max_retries} retries: {e}")
                        failed_batches.append(batch_num)
                    else:
                        logger.warning(f"⚠️ Batch {batch_num} failed, retry {retry_count}/{max_retries}: {e}")
                        time.sleep(retry_count * 2)

        # Verification step
        if verify and failed_batches:
            logger.error(f"❌ Upload incomplete! Failed batches: {failed_batches}")
            return False

        if verify:
            # Wait a moment for Pinecone to index
            time.sleep(2)

            # Verify by checking index stats
            try:
                stats = index.describe_index_stats()
                current_total = stats.get('total_vector_count', 0)
                logger.info(f"📊 Verification: Pinecone now has {current_total} total vectors")

                # Check if our vectors are there by sampling a few IDs
                sample_size = min(5, len(vectors))
                sample_ids = [vectors[i]['id'] for i in range(0, len(vectors), len(vectors) // sample_size)][:sample_size]

                fetch_result = index.fetch(ids=sample_ids)
                found_count = len(fetch_result.get('vectors', {}))

                if found_count == len(sample_ids):
                    logger.info(f"✅ Verification PASSED: All {sample_size} sampled vectors found in Pinecone")
                else:
                    logger.warning(f"⚠️ Verification WARNING: Only {found_count}/{len(sample_ids)} sampled vectors found")

            except Exception as e:
                logger.warning(f"⚠️ Verification check failed: {e}")

        success_rate = (uploaded_count / total_vectors) * 100
        logger.info(f"📊 Upload complete: {uploaded_count}/{total_vectors} vectors ({success_rate:.1f}% success rate)")

        return uploaded_count == total_vectors

    except Exception as e:
        logger.error(f"Pinecone upload failed: {e}")
        return False

def delete_vectors_from_pinecone(vector_ids: List[str]) -> bool:
    """Delete specific vectors from Pinecone"""
    index = get_pinecone_index()
    if not index:
        logger.warning("Pinecone not available")
        return True
    
    try:
        # Delete in batches
        for i in range(0, len(vector_ids), 100):  # Pinecone delete batch limit
            batch = vector_ids[i:i + 100]
            index.delete(ids=batch)
            logger.info(f"Deleted vector batch {i//100 + 1}")
        
        return True
    except Exception as e:
        logger.error(f"Pinecone delete failed: {e}")
        return False

# -----------------------------
# Search Functions
# -----------------------------


# Here's the highly optimized search function with advanced retrieval techniques:
# python
import re
from typing import List, Dict, Optional, Tuple, Set
from collections import defaultdict
import numpy as np

# def search_documents(
#     query: str,
#     top_k: int = 5,
#     filters: Optional[Dict] = None,
#     search_strategy: str = "hybrid",
#     rerank: bool = True,
#     include_context: bool = True
# ) -> List[Dict]:
#     """
#     Advanced document search with multiple strategies
    
#     Args:
#         query: Search query
#         top_k: Number of final results to return (default 5)
#         filters: Optional metadata filters (e.g., {'file_type': 'pdf'})
#         search_strategy: 'semantic', 'keyword', 'hybrid' (default 'hybrid')
#         rerank: Apply reranking for better relevance (default True)
#         include_context: Fetch adjacent chunks for context (default True)
    
#     Returns:
#         List of search results with scores and metadata
#     """
#     index = get_pinecone_index()
#     if not index:
#         logger.error("Pinecone index not available")
#         return []
    
#     try:
#         # Clean and analyze query
#         cleaned_query = query.strip()
#         if not cleaned_query:
#             return []
        
#         query_analysis = analyze_query(cleaned_query)
#         logger.info(f"Query analysis: {query_analysis}")
        
#         # Choose search strategy based on query type
#         if search_strategy == "auto":
#             search_strategy = select_search_strategy(query_analysis)
        
#         # Execute search based on strategy
#         if search_strategy == "hybrid":
#             results = hybrid_search(
#                 index, 
#                 cleaned_query, 
#                 query_analysis,
#                 top_k=top_k * 3,  # Fetch more for reranking
#                 filters=filters
#             )
#         elif search_strategy == "keyword":
#             results = keyword_boosted_search(
#                 index,
#                 cleaned_query,
#                 query_analysis,
#                 top_k=top_k * 3,
#                 filters=filters
#             )
#         else:  # semantic
#             results = semantic_search(
#                 index,
#                 cleaned_query,
#                 top_k=top_k * 3,
#                 filters=filters
#             )
        
#         if not results:
#             logger.warning(f"No results found for query: {cleaned_query}")
#             return []
        
#         # Apply reranking
#         if rerank and len(results) > top_k:
#             results = rerank_results(results, cleaned_query, query_analysis, top_k * 2)
        
#         # Deduplicate by document chunks
#         results = deduplicate_results(results)
        
#         # Add context from adjacent chunks
#         if include_context:
#             results = enrich_with_context(results, index)
        
#         # Apply query-specific boosting
#         results = apply_query_boosting(results, query_analysis)
        
#         # Sort by final score and limit
#         results.sort(key=lambda x: x.get('score', 0), reverse=True)
#         final_results = results[:top_k]
        
#         # Format results
#         formatted_results = format_search_results(final_results)
        
#         logger.info(f"Returning {len(formatted_results)} results for query: {cleaned_query[:50]}...")
#         return formatted_results
        
#     except Exception as e:
#         logger.error(f"Search error: {e}", exc_info=True)
#         return []

def read_excel_all_sheets(file_path: str, filename: str) -> Optional[pd.DataFrame]:
    """
    Read Excel file with multiple sheets and combine them
    """
    try:
        # Read all sheets
        if filename.lower().endswith('.xlsx'):
            excel_file = pd.ExcelFile(file_path, engine='openpyxl')
        elif filename.lower().endswith('.xls'):
            excel_file = pd.ExcelFile(file_path, engine='xlrd')
        else:
            excel_file = pd.ExcelFile(file_path)
        
        sheet_names = excel_file.sheet_names
        logger.info(f"Found {len(sheet_names)} sheets: {sheet_names}")
        
        # Combine all sheets
        all_dfs = []
        for sheet_name in sheet_names:
            try:
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
                if not df.empty:
                    # Add sheet name as column
                    df['_source_sheet'] = sheet_name
                    all_dfs.append(df)
                    logger.info(f"  Sheet '{sheet_name}': {len(df)} rows, {len(df.columns)} columns")
            except Exception as e:
                logger.warning(f"  Failed to read sheet '{sheet_name}': {e}")
                continue
        
        if not all_dfs:
            logger.error("No sheets could be read")
            return None
        
        # Combine all sheets
        combined_df = pd.concat(all_dfs, ignore_index=True, sort=False)
        logger.info(f"Combined into {len(combined_df)} total rows")
        
        return combined_df
        
    except Exception as e:
        logger.error(f"Error reading multi-sheet Excel: {e}")
        return None

# def search_documents(
#     query: str,
#     top_k: int = 5,
#     filters: Optional[Dict] = None,
#     search_strategy: str = "hybrid",
#     rerank: bool = True,
#     include_context: bool = True
# ) -> List[Dict]:
#     """
#     FIXED search that works properly with PDFs
#     """
#     index = get_pinecone_index()
#     if not index:
#         logger.error("Pinecone index not available")
#         return []
    
#     try:
#         cleaned_query = query.strip()
#         if not cleaned_query:
#             return []
        
#         logger.info(f"Searching for: {cleaned_query}")
        
#         # Generate query embedding
#         query_embedding = generate_embedding(cleaned_query)
#         if query_embedding is None:
#             logger.error("Failed to generate query embedding")
#             return []
        
#         # Search with MORE results initially
#         try:
#             results = index.query(
#                 vector=query_embedding,
#                 top_k=min(50, top_k * 10),  # Get MORE results
#                 include_metadata=True,
#                 filter=filters
#             )
            
#             matches = results.get('matches', [])
#             logger.info(f"Found {len(matches)} initial matches")
            
#             if not matches:
#                 return []
            
#             # Convert to dict format
#             formatted_matches = []
#             for match in matches:
#                 metadata = match.get('metadata', {})
                
#                 formatted_matches.append({
#                     'id': match['id'],
#                     'score': match['score'],
#                     'filename': metadata.get('filename', 'Unknown'),
#                     'content': metadata.get('full_text', metadata.get('content', '')),
#                     'chunk_type': metadata.get('chunk_type', 'unknown'),
#                     'metadata': metadata
#                 })
            
#             # BOOST page-level results
#             for match in formatted_matches:
#                 if match['metadata'].get('chunk_type') == 'page':
#                     match['score'] = min(match['score'] * 1.3, 1.0)  # Boost pages!
            
#             # Re-sort by score
#             formatted_matches.sort(key=lambda x: x['score'], reverse=True)
            
#             # Apply keyword boosting
#             query_lower = cleaned_query.lower()
#             for match in formatted_matches:
#                 content = match['content'].lower()
                
#                 # Exact phrase match - HUGE boost
#                 if query_lower in content:
#                     match['score'] = min(match['score'] + 0.2, 1.0)
                
#                 # Keyword matches
#                 keywords = query_lower.split()
#                 keyword_count = sum(1 for kw in keywords if kw in content)
#                 if keyword_count > 0:
#                     boost = keyword_count * 0.05
#                     match['score'] = min(match['score'] + boost, 1.0)
            
#             # Re-sort after boosting
#             formatted_matches.sort(key=lambda x: x['score'], reverse=True)
            
#             # Return top results
#             final_results = formatted_matches[:top_k]
            
#             logger.info(f"Returning {len(final_results)} results")
#             for i, r in enumerate(final_results[:3], 1):
#                 logger.info(f"  Result {i}: {r['filename']} (page {r['metadata'].get('page_start')}) - score: {r['score']:.3f}")
            
#             return final_results
            
#         except Exception as e:
#             logger.error(f"Pinecone query failed: {e}")
#             return []
        
#     except Exception as e:
#         logger.error(f"Search error: {e}", exc_info=True)
#         return []


def search_documents(
    query: str,
    top_k: int = 20,  # Increased to get more results
    filters: Optional[Dict] = None,
    search_strategy: str = "hybrid",
    rerank: bool = True,
    include_context: bool = True
) -> List[Dict]:
    """
    ENHANCED search with better retrieval and full content access
    """
    index = get_pinecone_index()
    if not index:
        logger.error("Pinecone index not available")
        return []

    try:
        cleaned_query = query.strip()
        if not cleaned_query:
            return []

        logger.info(f"Searching: {cleaned_query}")

        # Generate query embedding
        query_embedding = generate_embedding(cleaned_query)
        if query_embedding is None:
            logger.error("Failed to generate query embedding")
            return []

        # Search with MORE results
        try:
            results = index.query(
                vector=query_embedding,
                top_k=min(150, top_k * 10),  # Get many more results
                include_metadata=True,
                filter=filters
            )

            matches = results.get('matches', [])
            logger.info(f"Found {len(matches)} matches")

            if not matches:
                return []

            # Format matches with FULL content retrieval
            formatted_matches = []
            for match in matches:
                metadata = match.get('metadata', {})
                filename = metadata.get('filename', 'Unknown')

                # Try to get FULL content from document store first
                full_content = ''
                try:
                    with document_store['lock']:
                        if filename in document_store.get('full_content', {}):
                            doc_info = document_store['full_content'][filename]
                            doc_type = doc_info.get('type', '')

                            # For PDFs, try to get specific page
                            if doc_type == 'PDF':
                                page_start = metadata.get('page_start')
                                if page_start and 'page_breakdown' in doc_info:
                                    # Find the specific page content
                                    for page_data in doc_info['page_breakdown']:
                                        if page_data['page'] == page_start:
                                            full_content = page_data.get('raw', page_data.get('text', ''))
                                            break

                                # If not found, use stored content
                                if not full_content:
                                    full_content = doc_info.get('content', '')[:5000]

                            # For CSV/Excel, get relevant rows
                            elif 'CSV' in doc_type or 'Excel' in doc_type:
                                row_index = metadata.get('row_index')
                                if row_index is not None and 'dataframe' in doc_info:
                                    # Get the specific row plus context
                                    df_data = doc_info['dataframe']
                                    if row_index < len(df_data):
                                        # Get 3 rows: previous, current, next
                                        start_idx = max(0, row_index - 1)
                                        end_idx = min(len(df_data), row_index + 2)
                                        context_rows = df_data[start_idx:end_idx]

                                        # Format as readable text
                                        full_content = "Data rows:\n"
                                        for idx, row_data in enumerate(context_rows, start=start_idx):
                                            full_content += f"\nRow {idx + 1}:\n"
                                            for key, value in row_data.items():
                                                full_content += f"  {key}: {value}\n"

                                # Fallback to stored content
                                if not full_content:
                                    full_content = doc_info.get('content', '')[:5000]

                            # For text files
                            else:
                                full_content = doc_info.get('content', '')[:5000]

                except Exception as e:
                    logger.warning(f"Could not retrieve full content for {filename}: {e}")

                # Fallback to metadata content
                if not full_content:
                    full_content = metadata.get('full_text', metadata.get('content', ''))

                formatted_matches.append({
                    'id': match['id'],
                    'score': float(match['score']),
                    'filename': filename,
                    'content': full_content,  # Now contains much more content
                    'chunk_type': metadata.get('chunk_type', 'unknown'),
                    'metadata': metadata
                })

            # ENHANCED SCORING with better boosting
            query_lower = cleaned_query.lower()
            query_words = [w.lower() for w in cleaned_query.split() if len(w) > 2]

            for match in formatted_matches:
                base_score = match['score']
                boost = 0.0

                # Page chunks are most important for PDFs
                if match['metadata'].get('chunk_type') == 'page':
                    boost += 0.15

                # CSV rows are important for data queries
                if match['metadata'].get('chunk_type') == 'csv_row':
                    boost += 0.12

                # Boost for high importance
                importance = match['metadata'].get('importance', 0.5)
                boost += importance * 0.1

                # Keyword matching with better scoring
                content_lower = match['content'].lower()

                # Exact phrase match - HUGE boost
                if query_lower in content_lower:
                    boost += 0.30

                # Individual keywords - count matches
                keyword_matches = sum(1 for kw in query_words if kw in content_lower)
                if keyword_matches > 0:
                    # More keywords = better match
                    keyword_ratio = keyword_matches / max(len(query_words), 1)
                    boost += keyword_ratio * 0.25

                # Word proximity - check if query words appear near each other
                if len(query_words) > 1:
                    first_word_pos = content_lower.find(query_words[0])
                    if first_word_pos != -1:
                        # Check if other words appear within 200 characters
                        window = content_lower[first_word_pos:first_word_pos + 200]
                        proximity_matches = sum(1 for w in query_words[1:] if w in window)
                        if proximity_matches > 0:
                            boost += proximity_matches * 0.08

                # Apply boost
                match['score'] = min(base_score + boost, 1.0)

            # Sort by final score
            formatted_matches.sort(key=lambda x: x['score'], reverse=True)

            # Return top results
            final_results = formatted_matches[:top_k]

            logger.info(f"Returning {len(final_results)} results")
            for i, r in enumerate(final_results[:5], 1):
                logger.info(f"  {i}. {r['filename']} - {r['score']:.3f} - {len(r['content'])} chars")

            return final_results

        except Exception as e:
            logger.error(f"Query failed: {e}")
            return []

    except Exception as e:
        logger.error(f"Search error: {e}", exc_info=True)
        return []



def detect_multi_item_query(query: str) -> bool:
    """
    Detect if user is asking about multiple items at once
    """
    query_lower = query.lower()

    # Patterns for multi-item queries
    multi_patterns = [
        r'\band\b.*\band\b',  # "item A and item B and item C"
        r',\s*\w+\s*,',  # "item1, item2, item3"
        r'\ball\b.*\b(items?|products?|rows?|entries?)\b',  # "all items"
        r'\blist\b.*\b(all|multiple)\b',  # "list all"
        r'\b(multiple|several|few)\b.*\b(items?|products?|rows?)\b',  # "multiple items"
        r'\bevery\b',  # "every item"
        r'\beach\b.*\b(item|product|row)\b'  # "each item"
    ]

    for pattern in multi_patterns:
        if re.search(pattern, query_lower):
            return True

    # Check for comma-separated list
    comma_count = query.count(',')
    if comma_count >= 2:
        return True

    # Check for multiple item codes/IDs
    item_codes = re.findall(r'\b[A-Z]{2,4}\d{3,5}\b', query)
    if len(item_codes) >= 2:
        return True

    return False


def extract_items_from_query(query: str) -> List[str]:
    """
    Extract individual items from a multi-item query

    Examples:
    - "get elite price of SB39 SBMAT, BSS33 WD, DDR1824" -> ["SB39 SBMAT", "BSS33 WD", "DDR1824"]
    - "what is the price for item A and item B" -> ["item A", "item B"]
    """
    items = []

    # Try to split by common separators
    # Pattern 1: Comma-separated items
    if ',' in query:
        # Split by comma and clean
        potential_items = [item.strip() for item in query.split(',')]

        # Remove the question part (usually in the first segment)
        # Look for patterns like "get elite price of", "what is the", etc.
        first_item = potential_items[0]

        # Find where the actual item starts (after prepositions like "of", "for", etc.)
        for prep in [' of ', ' for ', ' on ', ': ']:
            if prep in first_item.lower():
                parts = first_item.split(prep, 1)
                if len(parts) > 1:
                    first_item = parts[1].strip()
                break

        items.append(first_item)
        items.extend(potential_items[1:])

    # Pattern 2: "and" separated items
    elif ' and ' in query.lower():
        parts = re.split(r'\s+and\s+', query, flags=re.IGNORECASE)

        # Clean first part (remove question prefix)
        first_part = parts[0]
        for prep in [' of ', ' for ', ' on ', ': ']:
            if prep in first_part.lower():
                segments = first_part.split(prep, 1)
                if len(segments) > 1:
                    first_part = segments[1].strip()
                break

        items.append(first_part)
        items.extend([p.strip() for p in parts[1:]])

    # Pattern 3: Extract item codes (fallback)
    else:
        # Look for item codes like SB39, BSS33, DDR1824, etc.
        item_codes = re.findall(r'\b[A-Z]{2,6}\d{2,6}(?:\s+[A-Z]{2,10})?\b', query)
        if item_codes:
            items = item_codes

    # Clean and deduplicate
    items = [item.strip() for item in items if item.strip()]
    items = list(dict.fromkeys(items))  # Remove duplicates while preserving order

    logger.info(f"Extracted {len(items)} items from query: {items}")
    return items


def search_multi_items(query: str, top_k_per_item: int = 10) -> List[Dict]:
    """
    Search for multiple items by splitting the query and searching each item separately
    This ensures better retrieval when asking about multiple items at once
    """
    logger.info(f"🔍 Multi-item search for: {query}")

    # Extract individual items
    items = extract_items_from_query(query)

    if not items:
        # Fallback to regular search if extraction fails
        logger.warning("Could not extract items, falling back to regular search")
        return search_documents(query, top_k=top_k_per_item * 3)

    logger.info(f"Searching for {len(items)} items individually: {items}")

    all_results = []
    seen_ids = set()

    # Search for each item separately
    for item in items:
        logger.info(f"  → Searching for: {item}")

        # Try multiple search variations for better matching
        search_variations = [
            item,  # Original
            item.replace(' ', ''),  # No spaces: "BSS33WD"
            item.replace(' ', '-'),  # Dashes: "BSS33-WD"
            item.replace('-', ' '),  # Spaces instead of dashes
            item.replace('_', ' '),  # Spaces instead of underscores
        ]

        # Remove duplicates while preserving order
        search_variations = list(dict.fromkeys(search_variations))

        item_results = []

        # Try each variation until we get good results
        for variation in search_variations[:3]:  # Try first 3 variations
            if variation == item:
                logger.info(f"    Trying: {variation}")
            else:
                logger.info(f"    Trying variation: {variation}")

            variation_results = search_documents(
                variation,
                top_k=top_k_per_item,
                search_strategy="auto",
                rerank=True,
                include_context=True
            )

            # If we found good results (high similarity), use them
            if variation_results and len(variation_results) > 0:
                # Check if first result has decent similarity (>0.6 for item codes, >0.7 for phrases)
                has_numbers = any(char.isdigit() for char in item)
                similarity_threshold = 0.6 if has_numbers else 0.7

                if variation_results[0].get('similarity', 0) > similarity_threshold:
                    item_results = variation_results
                    logger.info(f"    ✓ Found {len(variation_results)} good results with variation '{variation}' (similarity: {variation_results[0].get('similarity', 0):.3f})")
                    break
                elif not item_results:
                    # Keep results if we haven't found anything better
                    item_results = variation_results
                    logger.info(f"    → Keeping {len(variation_results)} results from '{variation}' (similarity: {variation_results[0].get('similarity', 0):.3f})")

        if not item_results:
            logger.warning(f"    ⚠️ No results found for '{item}' with any variation")

        # Add results, avoiding duplicates
        for result in item_results:
            result_id = result.get('id', str(result))
            if result_id not in seen_ids:
                seen_ids.add(result_id)
                all_results.append(result)

        logger.info(f"    Total results for '{item}': {len(item_results)}")

    logger.info(f"✅ Multi-item search complete: {len(all_results)} unique results from {len(items)} items")
    return all_results


def analyze_query(query: str) -> Dict:
    """Analyze query to determine type and extract key information"""
    query_lower = query.lower()
    
    analysis = {
        'original': query,
        'length': len(query.split()),
        'has_numbers': bool(re.search(r'\d+', query)),
        'has_financial': any(term in query_lower for term in 
                            ['revenue', 'profit', 'sales', 'cost', 'price', 
                             'expense', 'income', 'margin', 'quarter', 'q1', 'q2', 'q3', 'q4']),
        'has_date': bool(re.search(r'\d{4}|january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec', query_lower)),
        'is_question': query.strip().endswith('?'),
        'is_comparison': any(term in query_lower for term in ['compare', 'vs', 'versus', 'difference', 'between']),
        'is_aggregation': any(term in query_lower for term in ['total', 'sum', 'average', 'count', 'how many', 'list all']),
        'file_mentioned': extract_filename_from_query(query),
        'keywords': extract_keywords(query),
        'query_type': None
    }
    
    # Determine query type
    if analysis['is_aggregation']:
        analysis['query_type'] = 'aggregation'
    elif analysis['is_comparison']:
        analysis['query_type'] = 'comparison'
    elif analysis['has_financial'] and analysis['has_numbers']:
        analysis['query_type'] = 'data_lookup'
    elif analysis['has_date']:
        analysis['query_type'] = 'temporal'
    elif analysis['is_question']:
        analysis['query_type'] = 'question'
    else:
        analysis['query_type'] = 'general'
    
    return analysis


def extract_keywords(query: str) -> List[str]:
    """Extract important keywords from query"""
    # Remove stop words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                  'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
                  'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                  'would', 'should', 'could', 'what', 'when', 'where', 'who', 'which',
                  'how', 'why', 'this', 'that', 'these', 'those'}
    
    words = re.findall(r'\b\w+\b', query.lower())
    keywords = [w for w in words if w not in stop_words and len(w) > 2]
    
    return keywords


def extract_filename_from_query(query: str) -> Optional[str]:
    """Extract filename if mentioned in query"""
    # Look for common file patterns
    patterns = [
        r'in\s+([^\s]+\.(?:pdf|csv|xlsx|xls|txt))',
        r'from\s+([^\s]+\.(?:pdf|csv|xlsx|xls|txt))',
        r'file\s+([^\s]+\.(?:pdf|csv|xlsx|xls|txt))',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, query.lower())
        if match:
            return match.group(1)
    
    return None


def select_search_strategy(query_analysis: Dict) -> str:
    """Select optimal search strategy based on query analysis"""
    if query_analysis['query_type'] == 'data_lookup':
        return 'keyword'
    elif query_analysis['has_numbers'] or query_analysis['has_financial']:
        return 'hybrid'
    else:
        return 'semantic'


def semantic_search(
    index,
    query: str,
    top_k: int = 15,
    filters: Optional[Dict] = None
) -> List[Dict]:
    """Pure semantic/vector search"""
    query_embedding = generate_embedding(query)
    
    if query_embedding is None:
        logger.error("Failed to generate query embedding")
        return []
    
    try:
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filters
        )
        
        return results.get('matches', [])
    except Exception as e:
        logger.error(f"Semantic search failed: {e}")
        return []


def keyword_boosted_search(
    index,
    query: str,
    query_analysis: Dict,
    top_k: int = 15,
    filters: Optional[Dict] = None
) -> List[Dict]:
    """Search with keyword-boosted embeddings"""
    keywords = query_analysis['keywords']
    
    # Create boosted embedding
    boosted_embedding = create_hybrid_embedding(query, keywords[:5])
    
    if boosted_embedding is None:
        # Fallback to regular semantic search
        return semantic_search(index, query, top_k, filters)
    
    try:
        results = index.query(
            vector=boosted_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filters
        )
        
        return results.get('matches', [])
    except Exception as e:
        logger.error(f"Keyword boosted search failed: {e}")
        return semantic_search(index, query, top_k, filters)


def hybrid_search(
    index,
    query: str,
    query_analysis: Dict,
    top_k: int = 15,
    filters: Optional[Dict] = None,
    alpha: float = 0.6
) -> List[Dict]:
    """
    Hybrid search combining semantic + keyword matching
    
    Args:
        alpha: Weight for semantic search (0-1). 1 = pure semantic, 0 = pure keyword
    """
    # 1. Semantic search
    semantic_results = semantic_search(index, query, top_k, filters)
    
    # 2. Keyword boosted search
    keyword_results = keyword_boosted_search(index, query, query_analysis, top_k, filters)
    
    # 3. Merge and reweight results
    merged_results = {}
    
    # Add semantic results
    for i, result in enumerate(semantic_results):
        doc_id = result['id']
        semantic_score = result['score']
        # Decay by position
        position_weight = 1.0 - (i / len(semantic_results)) * 0.3
        
        merged_results[doc_id] = {
            'result': result,
            'semantic_score': semantic_score * position_weight,
            'keyword_score': 0
        }
    
    # Add keyword results
    for i, result in enumerate(keyword_results):
        doc_id = result['id']
        keyword_score = result['score']
        position_weight = 1.0 - (i / len(keyword_results)) * 0.3
        
        if doc_id in merged_results:
            merged_results[doc_id]['keyword_score'] = keyword_score * position_weight
        else:
            merged_results[doc_id] = {
                'result': result,
                'semantic_score': 0,
                'keyword_score': keyword_score * position_weight
            }
    
    # Calculate hybrid scores
    for doc_id, data in merged_results.items():
        # Combine scores
        hybrid_score = (alpha * data['semantic_score'] + 
                       (1 - alpha) * data['keyword_score'])
        
        # Boost if both methods found it (consensus)
        if data['semantic_score'] > 0 and data['keyword_score'] > 0:
            hybrid_score *= 1.2
        
        data['result']['score'] = min(hybrid_score, 1.0)
    
    # Sort by hybrid score
    final_results = [data['result'] for data in merged_results.values()]
    final_results.sort(key=lambda x: x['score'], reverse=True)
    
    return final_results[:top_k]


def rerank_results(
    results: List[Dict],
    query: str,
    query_analysis: Dict,
    top_k: int = 10
) -> List[Dict]:
    """
    Rerank results using additional signals
    """
    query_lower = query.lower()
    keywords = set(query_analysis['keywords'])
    
    for result in results:
        metadata = result.get('metadata', {})
        content = metadata.get('content', '').lower()
        
        base_score = result.get('score', 0)
        boost = 0
        
        # Boost for exact keyword matches
        keyword_matches = sum(1 for kw in keywords if kw in content)
        boost += keyword_matches * 0.05
        
        # Boost for query terms in content
        query_terms = query_lower.split()
        term_matches = sum(1 for term in query_terms if term in content)
        boost += term_matches * 0.03
        
        # Boost for specific content types based on query
        if query_analysis['has_financial'] and metadata.get('has_financial'):
            boost += 0.1
        
        if query_analysis['has_date'] and metadata.get('has_date'):
            boost += 0.1
        
        # Boost for content chunks (they have more context)
        if metadata.get('chunk_type') in ['content', 'csv_chunk']:
            boost += 0.05
        
        # Boost for page/row vectors if query is specific
        if query_analysis['has_numbers'] and metadata.get('type') in ['csv_row', 'page']:
            boost += 0.08
        
        # Boost for mentioned filename
        if query_analysis['file_mentioned']:
            if query_analysis['file_mentioned'] in metadata.get('filename', '').lower():
                boost += 0.15
        
        # Apply boost
        result['rerank_score'] = min(base_score + boost, 1.0)
        result['boost_applied'] = boost
    
    # Sort by reranked score
    results.sort(key=lambda x: x.get('rerank_score', x.get('score', 0)), reverse=True)
    
    return results[:top_k]


def deduplicate_results(results: List[Dict], similarity_threshold: float = 0.95) -> List[Dict]:
    """Remove near-duplicate results from same document"""
    if not results:
        return []
    
    deduplicated = []
    seen_content = set()
    doc_chunk_count = defaultdict(int)
    
    for result in results:
        metadata = result.get('metadata', {})
        content = metadata.get('content', '')
        doc_id = metadata.get('doc_id', '')
        chunk_index = metadata.get('chunk_index', -1)
        
        # Create a hash of content
        content_hash = hashlib.md5(content.encode()).hexdigest()
        
        # Skip if we've seen very similar content
        if content_hash in seen_content:
            continue
        
        # Limit chunks per document (keep best ones)
        if doc_chunk_count[doc_id] >= 5:  # Max 5 chunks per document
            continue
        
        deduplicated.append(result)
        seen_content.add(content_hash)
        doc_chunk_count[doc_id] += 1
    
    return deduplicated


def enrich_with_context(results: List[Dict], index) -> List[Dict]:
    """
    Fetch adjacent chunks to provide better context
    """
    enriched_results = []
    
    for result in results:
        metadata = result.get('metadata', {})
        doc_id = metadata.get('doc_id')
        chunk_index = metadata.get('chunk_index')
        chunk_type = metadata.get('type', metadata.get('chunk_type'))
        
        # Only add context for content chunks
        if chunk_type not in ['content', 'csv_chunk']:
            enriched_results.append(result)
            continue
        
        # Try to fetch adjacent chunks
        adjacent_chunks = []
        
        if chunk_index is not None:
            for offset in [-1, 1]:  # Previous and next chunk
                adjacent_id = f"{doc_id}_{chunk_type}_{chunk_index + offset}"
                try:
                    adjacent_result = index.fetch(ids=[adjacent_id])
                    if adjacent_result and 'vectors' in adjacent_result:
                        vector_data = adjacent_result['vectors'].get(adjacent_id)
                        if vector_data and 'metadata' in vector_data:
                            adjacent_chunks.append({
                                'position': 'previous' if offset == -1 else 'next',
                                'content': vector_data['metadata'].get('content', '')[:300]
                            })
                except Exception as e:
                    logger.debug(f"Could not fetch adjacent chunk: {e}")
        
        # Add context to metadata
        if adjacent_chunks:
            result['adjacent_context'] = adjacent_chunks
        
        enriched_results.append(result)
    
    return enriched_results


def apply_query_boosting(results: List[Dict], query_analysis: Dict) -> List[Dict]:
    """Apply final boosting based on query-specific logic"""
    
    for result in results:
        metadata = result.get('metadata', {})
        score = result.get('rerank_score', result.get('score', 0))
        
        # Boost for data lookup queries
        if query_analysis['query_type'] == 'data_lookup':
            if metadata.get('type') == 'csv_row':
                score *= 1.15
        
        # Boost for aggregation queries
        elif query_analysis['query_type'] == 'aggregation':
            if metadata.get('type') in ['csv_column', 'csv_chunk']:
                score *= 1.1
        
        # Boost for temporal queries
        elif query_analysis['query_type'] == 'temporal':
            if metadata.get('has_date'):
                score *= 1.1
        
        result['final_score'] = min(score, 1.0)
    
    return results


def format_search_results(results: List[Dict]) -> List[Dict]:
    """Format results for display"""
    formatted = []
    
    for i, result in enumerate(results):
        metadata = result.get('metadata', {})
        
        formatted_result = {
            'rank': i + 1,
            'score': round(result.get('final_score', result.get('score', 0)), 4),
            'filename': metadata.get('filename', 'Unknown'),
            'content': metadata.get('content', ''),
            'file_type': metadata.get('file_type', 'unknown'),
            'chunk_type': metadata.get('type', metadata.get('chunk_type', 'unknown')),
            'metadata': {
                'doc_id': metadata.get('doc_id'),
                'chunk_index': metadata.get('chunk_index'),
                'page_start': metadata.get('page_start'),
                'page_end': metadata.get('page_end'),
                'row_index': metadata.get('row_index'),
                'column_name': metadata.get('column_name')
            }
        }
        
        # Add adjacent context if available
        if 'adjacent_context' in result:
            formatted_result['context'] = result['adjacent_context']
        
        # Add row data for CSV results
        if 'row_data' in metadata:
            try:
                formatted_result['row_data'] = json.loads(metadata['row_data'])
            except:
                pass
        
        formatted.append(formatted_result)
    
    return formatted


def search_with_fallback(query: str, top_k: int = 5) -> List[Dict]:
    """
    ENHANCED fallback search using multiple strategies when primary search fails
    Includes keyword matching, fuzzy search, and relaxed vector search
    """
    logger.warning(f"🔄 No results from primary search. Trying fallback strategies for: {query}")

    all_results = []

    # Strategy 1: Keyword-based search in full documents
    logger.info("🔍 Strategy 1: Keyword search in full documents")
    keyword_results = keyword_search_in_documents(query, top_k=top_k * 2)
    if keyword_results:
        logger.info(f"✅ Keyword search found {len(keyword_results)} results")
        all_results.extend(keyword_results)

    # Strategy 2: Relaxed vector search with NO threshold
    logger.info("🔍 Strategy 2: Relaxed vector search (no threshold)")
    try:
        index = get_pinecone_index()
        if index:
            query_embedding = generate_embedding(query)
            if query_embedding:
                results = index.query(
                    vector=query_embedding,
                    top_k=100,  # Get many results
                    include_metadata=True
                )

                for match in results.get('matches', []):
                    metadata = match.get('metadata', {})
                    content = metadata.get('full_text', metadata.get('content', ''))

                    all_results.append({
                        'id': match['id'],
                        'score': float(match['score']) * 0.7,  # Reduce fallback scores slightly
                        'filename': metadata.get('filename', 'Unknown'),
                        'content': content,
                        'chunk_type': metadata.get('chunk_type', 'unknown'),
                        'metadata': metadata
                    })

                logger.info(f"✅ Relaxed vector search found {len(results.get('matches', []))} results")
    except Exception as e:
        logger.error(f"❌ Relaxed vector search failed: {e}")

    # Strategy 3: Direct term matching in document_store
    logger.info("🔍 Strategy 3: Direct term matching")
    query_terms = [term.lower() for term in query.split() if len(term) > 2]

    with document_store['lock']:
        for filename, doc_info in document_store.get('full_content', {}).items():
            content = doc_info.get('content', '')
            if not content:
                continue

            content_lower = content.lower()

            # Count term matches
            match_count = sum(1 for term in query_terms if term in content_lower)

            if match_count > 0:
                match_score = min(match_count / max(len(query_terms), 1), 1.0) * 0.5

                # Extract relevant snippet
                first_term = next((t for t in query_terms if t in content_lower), None)
                if first_term:
                    pos = content_lower.find(first_term)
                    start = max(0, pos - 500)
                    end = min(len(content), pos + 1500)
                    snippet = content[start:end]

                    all_results.append({
                        'id': f"fallback_{filename}_{match_count}",
                        'score': match_score,
                        'filename': filename,
                        'content': snippet,
                        'chunk_type': 'term_match',
                        'metadata': {
                            'filename': filename,
                            'match_count': match_count,
                            'total_terms': len(query_terms)
                        }
                    })

    if all_results:
        logger.info(f"✅ Total fallback results: {len(all_results)}")
    else:
        logger.error("❌ ALL fallback strategies failed!")

    # Remove duplicates and sort
    seen_content = set()
    unique_results = []

    for result in sorted(all_results, key=lambda x: x['score'], reverse=True):
        content_hash = hash(result['content'][:200])
        if content_hash not in seen_content:
            seen_content.add(content_hash)
            unique_results.append(result)

    logger.info(f"📊 Returning {len(unique_results[:top_k])} unique fallback results")

    return unique_results[:top_k]


def keyword_search_in_documents(query: str, top_k: int = 30) -> List[Dict]:
    """
    Search documents using keyword matching
    Returns chunks with matching keywords, scored by relevance
    """
    results = []
    query_lower = query.lower()
    query_words = [w.lower() for w in query.split() if len(w) > 2]

    if not query_words:
        return []

    logger.info(f"🔎 Keyword searching for: {query_words}")

    with document_store['lock']:
        for filename, doc_info in document_store.get('full_content', {}).items():
            content = doc_info.get('content', '')
            if not content:
                continue

            content_lower = content.lower()

            # Check for exact phrase match
            if query_lower in content_lower:
                logger.info(f"✅ Exact phrase found in {filename}")

                # Find all occurrences
                pos = 0
                occurrence_count = 0
                while True:
                    pos = content_lower.find(query_lower, pos)
                    if pos == -1:
                        break

                    occurrence_count += 1

                    # Extract context
                    start = max(0, pos - 300)
                    end = min(len(content), pos + len(query) + 1200)
                    context = content[start:end]

                    results.append({
                        'id': f"keyword_exact_{filename}_{pos}",
                        'score': 0.95,  # Very high score for exact match
                        'filename': filename,
                        'content': context,
                        'chunk_type': 'keyword_exact',
                        'metadata': {
                            'filename': filename,
                            'match_type': 'exact_phrase'
                        }
                    })

                    pos += len(query)
                    if occurrence_count >= 3:  # Limit to 3 occurrences per file
                        break

            # Check for individual keyword matches
            else:
                keyword_positions = []
                for word in query_words:
                    if word in content_lower:
                        keyword_positions.append(content_lower.find(word))

                if keyword_positions:
                    match_ratio = len(keyword_positions) / len(query_words)
                    score = match_ratio * 0.65  # Max 0.65 for partial

                    first_pos = min(keyword_positions)
                    start = max(0, first_pos - 400)
                    end = min(len(content), first_pos + 1600)
                    context = content[start:end]

                    results.append({
                        'id': f"keyword_partial_{filename}",
                        'score': score,
                        'filename': filename,
                        'content': context,
                        'chunk_type': 'keyword_partial',
                        'metadata': {
                            'filename': filename,
                            'matched_keywords': len(keyword_positions),
                            'total_keywords': len(query_words)
                        }
                    })

    results.sort(key=lambda x: x['score'], reverse=True)
    logger.info(f"📊 Keyword search: {len(results)} results")

    return results[:top_k]


# Example usage
def example_searches():
    """Example of different search patterns"""
    
    # Financial data query
    results1 = search_documents(
        "What was the Q4 revenue?",
        top_k=5,
        search_strategy="hybrid"
    )
    
    # Specific file query
    results2 = search_documents(
        "Find employee data in HR_2024.csv",
        top_k=5,
        filters={'file_type': 'csv'}
    )
    
    # Comparison query
    results3 = search_documents(
        "Compare sales between Q1 and Q2",
        top_k=10,
        search_strategy="keyword"
    )
    
    # With fallback
    results4 = search_with_fallback(
        "information about product launches",
        top_k=5
    )
    
    return results1, results2, results3, results4




# -----------------------------
# File Management Functions
# -----------------------------
def get_all_uploaded_files() -> Dict:
    """Get all uploaded files from both local store and query Pinecone for verification"""
    with document_store['lock']:
        files = document_store['files'].copy()
    
    # Enhance with additional stats from Pinecone if available
    index = get_pinecone_index()
    if index:
        try:
            stats = index.describe_index_stats()
            total_vectors = stats.get('total_vector_count', 0)
            
            # Add global stats
            return {
                'files': files,
                'total_files': len(files),
                'total_vectors_in_db': total_vectors,
                'last_updated': datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting Pinecone stats: {e}")
    
    return {
        'files': files,
        'total_files': len(files),
        'total_vectors_in_db': 0,
        'last_updated': datetime.utcnow().isoformat()
    }

def delete_file_completely(filename: str) -> Dict:
    """Delete file from both Pinecone and S3, and local storage"""
    try:
        with document_store['lock']:
            if filename not in document_store['files']:
                return {'success': False, 'message': f'File {filename} not found'}
            
            file_info = document_store['files'][filename]
            doc_id = file_info['doc_id']
        
        # Get vector IDs to delete from Pinecone
        vector_ids = document_store.get('file_vectors', {}).get(filename, [])
        
        if not vector_ids:
            # Try to find vectors by doc_id if vector_ids not stored
            index = get_pinecone_index()
            if index:
                try:
                    # Query for vectors with this doc_id
                    # Note: This is a workaround since Pinecone doesn't support metadata-only queries
                    # In production, you should maintain better vector ID tracking
                    logger.warning(f"Vector IDs not found for {filename}, attempting cleanup by doc_id")
                    
                    # Generate some potential vector IDs based on common patterns
                    potential_ids = []
                    for i in range(100):  # Check up to 100 potential chunks
                        potential_ids.extend([
                            f"{doc_id}_{i}",
                            f"{doc_id}_row_{i}",
                            f"{doc_id}_chunk_{i}",
                            f"{doc_id}_col_{i}"
                        ])
                    
                    # Try to delete these IDs (Pinecone ignores non-existent IDs)
                    vector_ids = potential_ids[:1000]  # Limit to reasonable number
                    
                except Exception as e:
                    logger.error(f"Error finding vectors for {filename}: {e}")
        
        # Delete from Pinecone
        pinecone_success = True
        if vector_ids:
            pinecone_success = delete_vectors_from_pinecone(vector_ids)
            # Wait for Pinecone to propagate the deletion
            if pinecone_success:
                time.sleep(1)  # Give Pinecone time to update stats
        
        # Delete from S3
        s3_key = f"documents/{filename}"
        s3_success = delete_from_s3(s3_key)
        
        # Remove from local storage
        with document_store['lock']:
            if filename in document_store['files']:
                del document_store['files'][filename]
            if filename in document_store['full_content']:
                del document_store['full_content'][filename]
            if filename in document_store.get('file_vectors', {}):
                del document_store['file_vectors'][filename]
            
            # Clear caches
            if filename in _cache.get('csv_lookup_cache', {}):
                del _cache['csv_lookup_cache'][filename]
            if filename in _cache.get('full_documents', {}):
                del _cache['full_documents'][filename]
        
        logger.info(f"✅ Deleted {filename} - Pinecone: {pinecone_success}, S3: {s3_success}")
        
        return {
            'success': True,
            'message': f'Successfully deleted {filename}',
            'pinecone_success': pinecone_success,
            's3_success': s3_success,
            'vectors_deleted': len(vector_ids) if vector_ids else 0
        }
        
    except Exception as e:
        logger.error(f"Error deleting file {filename}: {e}")
        return {'success': False, 'message': f'Error deleting file: {str(e)}'}

# -----------------------------
# Flask Application
# -----------------------------
app = Flask(__name__, 
            static_folder='static',
            template_folder='templates')
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/static/<path:path>')
def send_static(path):
    """Serve static files"""
    return send_from_directory('static', path)

@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle file uploads"""
    start_time = time.time()
    
    if 'files' not in request.files:
        return jsonify({'success': False, 'message': 'No files provided'}), 400
    
    files = request.files.getlist('files')
    results = []
    total_chunks = 0
    
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            temp_path = os.path.join(tempfile.gettempdir(), filename)
            
            try:
                # Save file
                file.save(temp_path)
                
                # Check if already processed
                with document_store['lock']:
                    if filename in document_store['files']:
                        logger.info(f"File {filename} already processed")
                        results.append(f"{filename} (already indexed)")
                        os.remove(temp_path)
                        continue
                
                # Upload to S3 first
                s3_key = f"documents/{filename}"
                s3_success = upload_to_s3(temp_path, s3_key)
                
                # Process based on file type
                extension = filename.rsplit('.', 1)[1].lower()
                
                if extension == 'pdf':
                    doc_result = process_pdf(temp_path, filename)
                elif extension in ['csv', 'xlsx', 'xls']:
                    doc_result = process_csv(temp_path, filename)
                else:
                    # Text file
                    with open(temp_path, 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read()
                    
                    # Store full content
                    with document_store['lock']:
                        document_store['full_content'][filename] = {
                            'content': text,
                            'size': len(text),
                            'type': 'Text'
                        }
                        _cache['full_documents'][filename] = text
                    
                    chunks = chunk_text(text[:10000])  # Chunk limited portion
                    embeddings = generate_embeddings_batch(chunks)
                    
                    doc_id = str(uuid.uuid4())[:8]
                    vectors = []
                    vector_ids = []
                    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                        vector_id = f"{doc_id}_{i}"
                        vector_ids.append(vector_id)
                        vectors.append({
                            'id': vector_id,
                            'values': embedding,
                            'metadata': {
                                'filename': filename,
                                'doc_id': doc_id,
                                'chunk_index': i,
                                'content': chunk[:1000],
                                'file_type': 'text'
                            }
                        })
                    
                    doc_result = {
                        'success': True,
                        'doc_id': doc_id,
                        'filename': filename,
                        'vectors': vectors,
                        'vector_ids': vector_ids,
                        'chunks': len(chunks),
                        'type': 'Text Document'
                    }
                
                # Clean up temp file
                os.remove(temp_path)
                
                if doc_result.get('success'):
                    # Upload to Pinecone
                    upload_success = upload_to_pinecone(doc_result['vectors'])
                    
                    if upload_success:
                        # Store document info and vector IDs for future deletion
                        with document_store['lock']:
                            document_store['files'][filename] = {
                                'doc_id': doc_result['doc_id'],
                                'chunks': doc_result['chunks'],
                                'type': doc_result['type'],
                                'uploaded_at': datetime.utcnow().isoformat(),
                                's3_uploaded': s3_success,
                                's3_key': s3_key if s3_success else None
                            }
                            
                            # Store vector IDs for deletion tracking
                            if 'file_vectors' not in document_store:
                                document_store['file_vectors'] = {}
                            document_store['file_vectors'][filename] = doc_result.get('vector_ids', [])
                            
                            _cache['uploaded_files'] = document_store['files'].copy()
                        
                        results.append(f"{filename} ({doc_result['chunks']} chunks)")
                        total_chunks += doc_result['chunks']
                        logger.info(f"✅ Successfully processed {filename}")
                    else:
                        results.append(f"{filename} (upload failed)")
                else:
                    results.append(f"{filename} (processing failed)")
                    
            except Exception as e:
                logger.error(f"Failed to process {filename}: {e}")
                results.append(f"{filename} (error)")
                if os.path.exists(temp_path):
                    os.remove(temp_path)
    
    processing_time = time.time() - start_time
    
    if results:
        return jsonify({
            'success': True,
            'message': f'Processed {len(results)} file(s) with {total_chunks} chunks in {processing_time:.1f}s',
            'files': results
        })
    else:
        return jsonify({
            'success': False,
            'message': 'No valid files to process'
        }), 400

@app.route('/files', methods=['GET'])
def get_files():
    """Get list of ALL uploaded files with enhanced information"""
    try:
        files_info = get_all_uploaded_files()
        return jsonify({
            'success': True,
            **files_info
        })
    except Exception as e:
        logger.error(f"Error getting files: {e}")
        return jsonify({
            'success': False,
            'message': 'Error retrieving files'
        }), 500

@app.route('/files/<filename>', methods=['DELETE'])
def delete_file(filename):
    """Delete a specific file from everywhere (Pinecone, S3, local storage)"""
    try:
        # Decode filename
        filename_decoded = filename.replace('%20', ' ')
        
        result = delete_file_completely(filename_decoded)
        
        if result['success']:
            return jsonify(result)
        else:
            return jsonify(result), 404
            
    except Exception as e:
        logger.error(f"Error in delete endpoint: {e}")
        return jsonify({
            'success': False,
            'message': f'Error deleting file: {str(e)}'
        }), 500

@app.route('/files/bulk-delete', methods=['POST'])
def bulk_delete_files():
    """Delete multiple files at once"""
    try:
        data = request.json
        filenames = data.get('filenames', [])
        
        if not filenames:
            return jsonify({
                'success': False,
                'message': 'No filenames provided'
            }), 400
        
        results = {}
        overall_success = True
        
        for filename in filenames:
            result = delete_file_completely(filename)
            results[filename] = result
            if not result['success']:
                overall_success = False
        
        return jsonify({
            'success': overall_success,
            'message': f'Processed {len(filenames)} files',
            'results': results
        })
        
    except Exception as e:
        logger.error(f"Error in bulk delete: {e}")
        return jsonify({
            'success': False,
            'message': f'Error in bulk delete: {str(e)}'
        }), 500

@app.route('/status', methods=['GET'])
def get_status():
    """Get enhanced system status"""
    try:
        # Initialize defaults
        vector_count = 0
        connected = False
        index_stats = {}
        
        # Safely check Pinecone connection
        try:
            index = get_pinecone_index()
            if index is not None:
                stats = index.describe_index_stats()
                vector_count = stats.get('total_vector_count', 0)
                connected = True
                index_stats = {
                    'total_vectors': vector_count,
                    'namespaces': stats.get('namespaces', {}),
                    'dimension': stats.get('dimension', PINECONE_DIMENSION)
                }
                logger.info(f"✅ Pinecone connected: {vector_count} vectors")
            else:
                logger.warning("❌ Pinecone index is None - check API key and configuration")
                connected = False
        except Exception as e:
            logger.error(f"❌ Error getting Pinecone stats: {e}")
            connected = False
        
        # Safely get local file info
        try:
            with document_store['lock']:
                file_count = len(document_store.get('files', {}))
                total_chunks = sum(info.get('chunks', 0) for info in document_store.get('files', {}).values())
        except Exception as e:
            logger.error(f"Error getting local file stats: {e}")
            file_count = 0
            total_chunks = 0
        
        # Safely check S3 status
        s3_connected = False
        try:
            s3_client = get_s3_client()
            s3_connected = s3_client is not None
        except Exception as e:
            logger.error(f"Error checking S3 status: {e}")
            s3_connected = False
        
        # Safely get cache sizes
        cache_info = {
            'embeddings': 0,
            'csv_cache': 0,
            'full_documents': 0
        }
        try:
            cache_info = {
                'embeddings': len(_cache.get('embeddings', {})),
                'csv_cache': len(_cache.get('csv_lookup_cache', {})),
                'full_documents': len(_cache.get('full_documents', {}))
            }
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
        
        # Format the response
        response_data = {
            'connected': connected,
            'pinecone_stats': {
                'total_vectors': vector_count,
                'namespaces': index_stats.get('namespaces', {}),
                'dimension': index_stats.get('dimension', PINECONE_DIMENSION)
            } if connected else {},
            's3_connected': s3_connected,
            'local_files': file_count,
            'total_chunks_tracked': total_chunks,
            'cache_size': cache_info,
            'system_health': {
                'openai_available': get_openai_client() is not None,
                'pinecone_available': connected,
                's3_available': s3_connected
            }
        }

        logger.info(f"Status check: Pinecone={connected}, Vectors={vector_count}, Files={file_count}")

        return jsonify(response_data)
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        return jsonify({
            'connected': False,
            'pinecone_stats': {},
            's3_connected': False,
            'local_files': 0,
            'total_chunks_tracked': 0,
            'cache_size': {'embeddings': 0, 'csv_cache': 0, 'full_documents': 0},
            'error': str(e),
            'system_health': {
                'openai_available': False,
                'pinecone_available': False,
                's3_available': False
            }
        })

#######################chat api



@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify server is running"""
    return jsonify({
        'success': True,
        'status': 'healthy',
        'message': 'Server is running',
        'openai_configured': bool(OPENAI_API_KEY),
        'pinecone_configured': bool(PINECONE_API_KEY),
        'documents_loaded': len(document_store.get('files', {}))
    })


@app.route('/chat', methods=['POST'])
def chat():
    """
    Advanced chat endpoint with intelligent retrieval and response generation

    Features:
    - Smart query routing
    - Multi-strategy search
    - Context optimization
    - Source attribution
    - Fallback handling
    """
    try:
        # Log request received
        logger.info("=" * 80)
        logger.info("📨 CHAT REQUEST RECEIVED")

        data = request.json
        if not data:
            logger.error("❌ No JSON data in request")
            return jsonify({
                'success': False,
                'message': 'No data provided in request'
            }), 400

        message = data.get('message', '').strip()
        conversation_history = data.get('history', [])  # Optional conversation context

        logger.info(f"📝 Message: {message[:200]}")
        logger.info(f"📜 History items: {len(conversation_history)}")

        if not message:
            logger.warning("⚠️ Empty message received")
            return jsonify({'success': False, 'message': 'No message provided'}), 400
    except Exception as e:
        logger.error(f"❌ Error parsing request: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'message': 'Invalid request format',
            'error': str(e)
        }), 400

    try:
        # Step 1: Analyze query intent
        logger.info(f"🔍 Analyzing query intent...")
        query_intent = analyze_query_intent(message)
        logger.info(f"✅ Query intent: {query_intent['intent_type']}")
        
        # Step 2: Route based on intent
        if query_intent['intent_type'] == 'full_document_request':
            return handle_full_document_request(message, query_intent)
        
        elif query_intent['intent_type'] == 'specific_page_request':
            return handle_page_request(message, query_intent)
        
        elif query_intent['intent_type'] == 'document_summary':
            return handle_summary_request(message, query_intent)
        
        elif query_intent['intent_type'] == 'list_documents':
            return handle_list_documents()
        
        elif query_intent['intent_type'] == 'direct_data_lookup':
            return handle_direct_lookup(message, query_intent)
        
        else:  # RAG query
            return handle_rag_query(message, query_intent, conversation_history)
        
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)

        # Return detailed error for debugging
        error_details = {
            'success': False,
            'message': 'An error occurred while processing your request. Please check the logs or try again.',
            'error': str(e),
            'error_type': type(e).__name__
        }

        # Add more context in development
        if app.debug:
            import traceback
            error_details['traceback'] = traceback.format_exc()

        return jsonify(error_details), 500


def analyze_query_intent(message: str) -> Dict:
    """
    Analyze user query to determine intent and routing
    """
    message_lower = message.lower()
    
    intent = {
        'intent_type': 'rag_query',  # default
        'filename': None,
        'page_number': None,
        'confidence': 0.0,
        'query_analysis': analyze_query(message)
    }
    
    # Full document request
    full_doc_patterns = [
        r'(full|complete|entire|whole)\s+(document|file|pdf|content)',
        r'(show|give|display)\s+me\s+(everything|all)',
        r'all\s+pages',
        r'entire\s+content'
    ]
    if any(re.search(pattern, message_lower) for pattern in full_doc_patterns):
        intent['intent_type'] = 'full_document_request'
        intent['filename'] = extract_filename_from_message(message)
        intent['confidence'] = 0.9
        return intent
    
    # Specific page request
    page_match = re.search(r'page\s*(\d+)', message_lower)
    if page_match and any(word in message_lower for word in ['show', 'get', 'display', 'see']):
        intent['intent_type'] = 'specific_page_request'
        intent['page_number'] = int(page_match.group(1))
        intent['filename'] = extract_filename_from_message(message)
        intent['confidence'] = 0.95
        return intent
    
    # Summary request
    summary_patterns = [r'summar(y|ize)', r'overview\s+of', r'what\s+is\s+in', r'what\'s\s+in']
    if any(re.search(pattern, message_lower) for pattern in summary_patterns):
        intent['intent_type'] = 'document_summary'
        intent['filename'] = extract_filename_from_message(message)
        intent['confidence'] = 0.85
        return intent
    
    # List documents
    list_patterns = [r'list\s+(all\s+)?(documents|files)', r'what\s+files', r'show\s+me\s+(all\s+)?documents']
    if any(re.search(pattern, message_lower) for pattern in list_patterns):
        intent['intent_type'] = 'list_documents'
        intent['confidence'] = 0.95
        return intent
    
    # Direct data lookup (specific values/rows)
    if intent['query_analysis']['query_type'] == 'data_lookup':
        intent['intent_type'] = 'direct_data_lookup'
        intent['confidence'] = 0.7
    
    return intent


def extract_filename_from_message(message: str) -> Optional[str]:
    """Extract filename from user message"""
    message_lower = message.lower()
    
    # Check against uploaded files
    with document_store['lock']:
        available_files = list(document_store['files'].keys())
    
    for filename in available_files:
        filename_lower = filename.lower()
        basename = os.path.splitext(filename)[0].lower()
        
        # Direct match
        if filename_lower in message_lower:
            return filename
        
        # Match without extension
        if basename in message_lower:
            return filename
        
        # Fuzzy match if available
        if FUZZY_AVAILABLE:
            from fuzzywuzzy import fuzz
            score = fuzz.partial_ratio(basename, message_lower)
            if score > 80:
                return filename
    
    return None


def handle_full_document_request(message: str, query_intent: Dict) -> tuple:
    """Handle full document retrieval requests"""
    filename = query_intent['filename']
    
    if not filename:
        # List available documents
        with document_store['lock']:
            available_docs = list(document_store['files'].keys())
        
        if not available_docs:
            return jsonify({
                'success': True,
                'response': "No documents have been uploaded yet. Please upload a document first.",
                'intent': 'full_document_request'
            })
        
        response = "📚 **Available Documents:**\n\n"
        for doc in available_docs:
            doc_info = document_store['files'][doc]
            response += f"• **{doc}**\n"
            response += f"  - Type: {doc_info['type']}\n"
            response += f"  - Chunks: {doc_info['chunks']}\n"
            response += f"  - Uploaded: {doc_info['uploaded_at']}\n\n"
        
        response += "To view a full document, ask: *'Show me the full document for [filename]'*"
        
        return jsonify({
            'success': True,
            'response': response,
            'intent': 'full_document_request'
        })
    
    # Retrieve full document
    if filename not in document_store.get('full_content', {}):
        return jsonify({
            'success': False,
            'response': f"Document '{filename}' not found or not fully processed.",
            'intent': 'full_document_request'
        }), 404
    
    full_doc = document_store['full_content'][filename]
    doc_content = full_doc['content']
    
    # Handle pagination
    MAX_RESPONSE_SIZE = 30000
    page_match = re.search(r'part\s*(\d+)|chunk\s*(\d+)|section\s*(\d+)', message.lower())
    page_num = int(page_match.group(1) or page_match.group(2) or page_match.group(3) or 1) if page_match else 1
    
    response_data = format_document_response(
        filename, 
        full_doc, 
        doc_content,
        page_num,
        MAX_RESPONSE_SIZE
    )
    
    return jsonify(response_data)


def format_document_response(
    filename: str,
    full_doc: Dict,
    doc_content: str,
    page_num: int,
    max_size: int
) -> Dict:
    """Format full document response with pagination"""
    
    response = f"📄 **Full Document: {filename}**\n\n"
    response += f"**Type:** {full_doc['type']}\n"
    
    if full_doc['type'] == 'PDF':
        response += f"**Pages:** {full_doc.get('pages', 'Unknown')}\n"
    elif full_doc['type'] in ['CSV/Excel', 'CSV']:
        response += f"**Rows:** {full_doc.get('rows', 'Unknown')}\n"
        response += f"**Columns:** {', '.join(full_doc.get('columns', []))[:200]}\n"
    
    response += f"**Size:** {len(doc_content):,} characters\n"
    response += "=" * 60 + "\n\n"
    
    if len(doc_content) > max_size:
        # Paginate
        chunk_size = max_size - 1000
        total_chunks = (len(doc_content) + chunk_size - 1) // chunk_size
        chunk_idx = max(0, min(page_num - 1, total_chunks - 1))
        
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, len(doc_content))
        
        response += f"**Part {chunk_idx + 1} of {total_chunks}**\n"
        response += f"*(Characters {start_idx:,} to {end_idx:,})*\n\n"
        response += doc_content[start_idx:end_idx]
        
        response += f"\n\n{'=' * 60}\n"
        response += f"📖 **Navigation:**\n"
        if chunk_idx < total_chunks - 1:
            response += f"• Next: *'Show part {chunk_idx + 2} of {filename}'*\n"
        if chunk_idx > 0:
            response += f"• Previous: *'Show part {chunk_idx} of {filename}'*\n"
    else:
        response += "**Complete Document:**\n\n"
        response += doc_content
    
    return {
        'success': True,
        'response': response,
        'intent': 'full_document_request',
        'document_info': {
            'filename': filename,
            'type': full_doc['type'],
            'total_length': len(doc_content),
            'page_shown': page_num
        }
    }


def handle_page_request(message: str, query_intent: Dict) -> tuple:
    """Handle specific page requests"""
    filename = query_intent['filename']
    page_num = query_intent['page_number']
    
    if not filename:
        return jsonify({
            'success': False,
            'response': "Please specify which document you want to see. Available documents can be listed with 'list all documents'.",
            'intent': 'specific_page_request'
        }), 400
    
    if filename not in document_store.get('full_content', {}):
        return jsonify({
            'success': False,
            'response': f"Document '{filename}' not found.",
            'intent': 'specific_page_request'
        }), 404
    
    full_doc = document_store['full_content'][filename]
    
    if full_doc['type'] != 'PDF':
        return jsonify({
            'success': False,
            'response': f"'{filename}' is not a PDF. Page requests only work for PDF documents.",
            'intent': 'specific_page_request'
        }), 400
    
    # Extract specific page
    doc_content = full_doc['content']
    pages = doc_content.split('[Page ')
    
    if page_num < 1 or page_num >= len(pages):
        return jsonify({
            'success': False,
            'response': f"Page {page_num} not found. Document has {len(pages) - 1} pages.",
            'intent': 'specific_page_request'
        }), 404
    
    page_content = f"[Page {pages[page_num]}"
    response = f"📄 **{filename} - Page {page_num}**\n\n{page_content}"
    
    return jsonify({
        'success': True,
        'response': response,
        'intent': 'specific_page_request',
        'page_info': {
            'filename': filename,
            'page_number': page_num,
            'total_pages': len(pages) - 1
        }
    })


def handle_summary_request(message: str, query_intent: Dict) -> tuple:
    """Handle document summary requests"""
    filename = query_intent['filename']
    
    if not filename:
        return jsonify({
            'success': False,
            'response': "Please specify which document to summarize.",
            'intent': 'document_summary'
        }), 400
    
    if filename not in document_store.get('full_content', {}):
        return jsonify({
            'success': False,
            'response': f"Document '{filename}' not found.",
            'intent': 'document_summary'
        }), 404
    
    full_doc = document_store['full_content'][filename]
    doc_info = document_store['files'][filename]
    
    response = f"📊 **Document Summary: {filename}**\n\n"
    response += f"**General Information:**\n"
    response += f"• Type: {doc_info['type']}\n"
    response += f"• Uploaded: {doc_info['uploaded_at']}\n"
    response += f"• Indexed Chunks: {doc_info['chunks']}\n"
    response += f"• Size: {len(full_doc['content']):,} characters\n\n"
    
    if full_doc['type'] == 'PDF':
        response += f"**PDF Details:**\n"
        response += f"• Total Pages: {full_doc.get('pages', 'Unknown')}\n"
    elif full_doc['type'] in ['CSV/Excel', 'CSV']:
        response += f"**Spreadsheet Details:**\n"
        response += f"• Total Rows: {full_doc.get('rows', 'Unknown')}\n"
        response += f"• Columns ({len(full_doc.get('columns', []))}): {', '.join(full_doc.get('columns', []))}\n"
        
        # Add column statistics if available
        if 'column_info' in full_doc:
            response += f"\n**Column Statistics:**\n"
            for col, info in list(full_doc['column_info'].items())[:5]:
                response += f"• {col}: {info['dtype']}"
                if info.get('is_numeric'):
                    response += f" (range: {info.get('min', 0):.2f} - {info.get('max', 0):.2f})"
                response += f", {info.get('unique_count', 0)} unique values\n"
    
    response += f"\n**Content Preview:**\n```\n{full_doc['content'][:1000]}\n```\n...\n\n"
    response += f"💡 **Next Steps:**\n"
    response += f"• Full document: *'Show me the full document for {filename}'*\n"
    response += f"• Ask questions: *'What information is in {filename}?'*\n"
    
    return jsonify({
        'success': True,
        'response': response,
        'intent': 'document_summary',
        'summary': {
            'filename': filename,
            'type': doc_info['type'],
            'size': len(full_doc['content'])
        }
    })


def handle_list_documents() -> tuple:
    """List all uploaded documents"""
    with document_store['lock']:
        if not document_store['files']:
            return jsonify({
                'success': True,
                'response': "No documents uploaded yet. Upload documents to get started!",
                'intent': 'list_documents',
                'documents': []
            })
        
        response = "📚 **Uploaded Documents:**\n\n"
        documents = []
        
        for filename, doc_info in document_store['files'].items():
            response += f"**{filename}**\n"
            response += f"  • Type: {doc_info['type']}\n"
            response += f"  • Chunks: {doc_info['chunks']}\n"
            response += f"  • Uploaded: {doc_info['uploaded_at']}\n"
            
            if filename in document_store.get('full_content', {}):
                full_doc = document_store['full_content'][filename]
                if full_doc['type'] == 'PDF':
                    response += f"  • Pages: {full_doc.get('pages', 'N/A')}\n"
                elif full_doc['type'] in ['CSV/Excel', 'CSV']:
                    response += f"  • Rows: {full_doc.get('rows', 'N/A')}\n"
            
            response += "\n"
            
            documents.append({
                'filename': filename,
                'type': doc_info['type'],
                'chunks': doc_info['chunks'],
                'uploaded_at': doc_info['uploaded_at']
            })
        
        response += f"**Total:** {len(documents)} document(s)\n\n"
        response += "💡 Ask questions about any document or request summaries!"
    
    return jsonify({
        'success': True,
        'response': response,
        'intent': 'list_documents',
        'documents': documents
    })


def handle_direct_lookup(message: str, query_intent: Dict) -> tuple:
    """
    Handle direct data lookups in CSV files
    Fast path for exact value searches
    """
    if not FUZZY_AVAILABLE or not _cache['csv_lookup_cache']:
        # Fall back to RAG if direct lookup not available
        return handle_rag_query(message, query_intent, [])
    
    query_analysis = query_intent['query_analysis']
    keywords = query_analysis['keywords']
    
    matches = []
    
    for filename, df in _cache['csv_lookup_cache'].items():
        # Search in all columns
        for col in df.columns:
            col_lower = str(col).lower()
            
            # Check if any keyword matches column name
            if any(kw in col_lower for kw in keywords):
                # Search in this column
                for idx, value in df[col].items():
                    value_str = str(value).lower()
                    
                    # Check for keyword matches in value
                    if any(kw in value_str for kw in keywords):
                        row_data = df.iloc[idx].to_dict()
                        matches.append({
                            'filename': filename,
                            'row_index': int(idx),
                            'column': col,
                            'matched_value': value,
                            'full_row': row_data
                        })
    
    if matches:
        response = f"🎯 **Found {len(matches)} exact match(es):**\n\n"
        
        for i, match in enumerate(matches[:5], 1):
            response += f"**Match {i}** (from {match['filename']}, row {match['row_index'] + 1}):\n"
            for key, val in list(match['full_row'].items())[:8]:
                response += f"  • {key}: {val}\n"
            response += "\n"
        
        if len(matches) > 5:
            response += f"*... and {len(matches) - 5} more matches*\n"
        
        return jsonify({
            'success': True,
            'response': response,
            'intent': 'direct_data_lookup',
            'matches': matches[:10]
        })
    
    # No direct matches, fall back to RAG
    return handle_rag_query(message, query_intent, [])


def handle_rag_query(message: str, query_intent: Dict, conversation_history: List) -> tuple:
    """
    Handle RAG queries with advanced retrieval and generation
    ENHANCED: Better handling of multiple items and batch queries
    """
    query_analysis = query_intent['query_analysis']

    # Detect if this is a multi-item query
    is_multi_item = detect_multi_item_query(message)

    # Step 1: Search with optimal strategy
    if is_multi_item:
        # ENHANCED: Search each item separately for better retrieval
        # But limit to reasonable number to avoid timeouts
        items = extract_items_from_query(message)
        items_count = len(items)

        # Adjust results per item based on total items (avoid overwhelming context)
        if items_count <= 3:
            top_k = 20
        elif items_count <= 5:
            top_k = 15
        else:
            top_k = 10

        search_results = search_multi_items(message, top_k_per_item=top_k)
        logger.info(f"Multi-item search for {items_count} items returned {len(search_results)} total results")
    else:
        # Single item search
        search_results = search_documents(
            message,
            top_k=15,
            search_strategy="auto",
            rerank=True,
            include_context=True
        )

    if not search_results:
        # Try fallback strategies
        search_top_k = 30 if is_multi_item else 15
        search_results = search_with_fallback(message, top_k=search_top_k)

    if not search_results:
        return jsonify({
            'success': True,
            'response': "I couldn't find relevant information in the uploaded documents. Please try:\n• Rephrasing your question\n• Being more specific\n• Checking which documents are uploaded with 'list all documents'",
            'intent': 'rag_query',
            'search_metadata': {
                'results_found': 0,
                'files_searched': 0
            }
        })

    # Step 2: Build optimized context (scale based on number of items)
    if is_multi_item:
        items_count = len(extract_items_from_query(message))
        # Scale context based on number of items - prevent timeouts
        if items_count <= 3:
            max_tokens = 6000
        elif items_count <= 5:
            max_tokens = 5000
        else:
            max_tokens = 4000
    else:
        max_tokens = 3000

    context_data = build_optimized_context(
        search_results,
        query_analysis,
        max_context_tokens=max_tokens
    )

    if not context_data['context']:
        return jsonify({
            'success': True,
            'response': f"Found {len(search_results)} results but they don't seem highly relevant. Try being more specific or check available documents.",
            'intent': 'rag_query',
            'search_metadata': {
                'results_found': len(search_results),
                'relevant_results': 0
            }
        })

    # Step 3: Generate response with LLM
    client = get_openai_client()

    if not client:
        # Return formatted context without LLM
        return jsonify({
            'success': True,
            'response': format_context_without_llm(context_data, message),
            'intent': 'rag_query',
            'search_metadata': context_data['metadata']
        })

    # Build prompt with context
    prompt = build_rag_prompt(message, context_data, query_analysis, conversation_history, is_multi_item)

    try:
        # Use more tokens for multi-item queries but cap it to prevent timeouts
        if is_multi_item:
            max_response_tokens = min(2000, 300 * len(extract_items_from_query(message)))  # ~300 tokens per item
        else:
            max_response_tokens = 1200

        logger.info(f"Calling LLM with max_tokens={max_response_tokens}, context_tokens={context_data['metadata']['context_tokens']}")

        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": get_system_prompt(query_analysis, is_multi_item)
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.1 if query_analysis['query_type'] == 'data_lookup' else 0.2,
            max_tokens=max_response_tokens,
            timeout=30  # 30 second timeout to prevent hanging
        )

        ai_response = response.choices[0].message.content

        # Add source attribution
        ai_response += format_source_attribution(context_data)

        return jsonify({
            'success': True,
            'response': ai_response,
            'intent': 'rag_query',
            'search_metadata': context_data['metadata'],
            'sources': context_data['sources']
        })

    except Exception as e:
        logger.error(f"LLM generation error: {e}", exc_info=True)

        # Check if it's a timeout error
        if 'timeout' in str(e).lower() or 'timed out' in str(e).lower():
            error_msg = "The request took too long to process. Try asking about fewer items at once, or ask about them individually."
        else:
            error_msg = f"Error generating response: {str(e)}"

        # If LLM fails, return context without LLM
        try:
            fallback_response = format_context_without_llm(context_data, message)
            return jsonify({
                'success': True,
                'response': fallback_response,
                'intent': 'rag_query',
                'search_metadata': context_data['metadata'],
                'warning': f'LLM unavailable ({str(e)}), showing direct context'
            })
        except Exception as fallback_error:
            logger.error(f"Fallback formatting failed: {fallback_error}", exc_info=True)
            # Last resort: return raw context
            return jsonify({
                'success': True,
                'response': f"Found relevant information but formatting failed. Here's the raw context:\n\n{context_data.get('context', 'No context available')[:2000]}",
                'intent': 'rag_query',
                'error': f"Both LLM and fallback failed: {str(e)}, {str(fallback_error)}"
            })


def build_optimized_context(
    search_results: List[Dict],
    query_analysis: Dict,
    max_context_tokens: int = 3000
) -> Dict:
    """Build optimized context from search results with configurable size"""

    context_parts = []
    files_found = set()
    sources = []
    total_tokens = 0

    # CRITICAL: Very low threshold for multi-item queries - include almost everything!
    MIN_SCORE_THRESHOLD = 0.1 if max_context_tokens > 5000 else 0.15

    for rank, result in enumerate(search_results, 1):
        score = result.get('score', 0)

        # Only skip VERY low scores
        if score < MIN_SCORE_THRESHOLD:
            logger.debug(f"Skipping result {rank} with score {score:.3f} (below {MIN_SCORE_THRESHOLD})")
            continue

        filename = result.get('filename', 'Unknown')
        content = result.get('content', '')
        chunk_type = result.get('chunk_type', 'unknown')
        metadata = result.get('metadata', {})

        if not content or len(content.strip()) < 10:  # Even shorter content is OK
            logger.debug(f"Skipping result {rank} - content too short")
            continue

        # Estimate tokens (rough: 1 token ≈ 4 chars)
        content_tokens = len(content) // 4

        if total_tokens + content_tokens > max_context_tokens:
            # Truncate content to fit
            available_tokens = max_context_tokens - total_tokens
            if available_tokens > 100:
                content = content[:available_tokens * 4]
                logger.info(f"Truncating result {rank} to fit context window")
            else:
                logger.info(f"Context window full, stopping at {rank-1} results")
                break

        # Format context based on type
        context_piece = format_context_piece(result, rank, query_analysis)

        context_parts.append(context_piece)
        files_found.add(filename)
        total_tokens += min(content_tokens, len(content) // 4)

        # Track sources
        source_info = {
            'filename': filename,
            'chunk_type': chunk_type,
            'score': round(score, 3),
            'page': metadata.get('page_start'),
            'row': metadata.get('row_index')
        }
        sources.append(source_info)

        logger.debug(f"Added result {rank}: {filename} (score: {score:.3f}, tokens: {content_tokens})")

    logger.info(f"Built context from {len(context_parts)} chunks ({total_tokens} tokens, {len(files_found)} files)")

    context = "\n\n".join(context_parts)

    return {
        'context': context,
        'files_found': files_found,
        'sources': sources,
        'metadata': {
            'files_searched': len(files_found),
            'total_results': len(search_results),
            'relevant_results': len(context_parts),
            'context_tokens': total_tokens,
            'min_score': min([s['score'] for s in sources]) if sources else 0,
            'max_score': max([s['score'] for s in sources]) if sources else 0
        }
    }


def build_multi_item_context(
    query: str,
    search_results: List[Dict],
    query_analysis: Dict,
    max_context_tokens: int = 6000
) -> Dict:
    """
    Build optimized context for multi-item queries
    Organizes results by item to ensure each item gets representation
    """
    items = extract_items_from_query(query)
    logger.info(f"Building multi-item context for {len(items)} items")

    # Group results by which item they match
    items_with_results = {}
    unmatched_results = []

    for result in search_results:
        content = result.get('content', '').lower()
        matched_item = None

        # Try to match result to an item
        for item in items:
            # Check various formats
            item_variations = [
                item.lower(),
                item.replace(' ', '').lower(),
                item.replace(' ', '-').lower(),
            ]

            if any(variation in content for variation in item_variations):
                matched_item = item
                break

        if matched_item:
            if matched_item not in items_with_results:
                items_with_results[matched_item] = []
            items_with_results[matched_item].append(result)
        else:
            unmatched_results.append(result)

    logger.info(f"Matched results: {len(items_with_results)} items have matches")
    for item, results in items_with_results.items():
        logger.info(f"  - {item}: {len(results)} results")

    # Build context organized by item
    context_parts = []
    files_found = set()
    sources = []
    total_tokens = 0

    # Add results for each item (balanced approach)
    max_results_per_item = max(3, min(10, max_context_tokens // (len(items) * 200)))

    for item in items:
        item_results = items_with_results.get(item, [])

        if item_results:
            context_parts.append(f"\n**=== Results for: {item} ===**\n")

            # Add top results for this item
            for rank, result in enumerate(item_results[:max_results_per_item], 1):
                content = result.get('content', '')
                filename = result.get('filename', 'Unknown')
                score = result.get('score', 0)

                content_tokens = len(content) // 4

                if total_tokens + content_tokens > max_context_tokens:
                    logger.info(f"Context window full, stopping at item '{item}'")
                    break

                context_piece = format_context_piece(result, rank, query_analysis)
                context_parts.append(context_piece)
                files_found.add(filename)
                total_tokens += content_tokens

                sources.append({
                    'filename': filename,
                    'item': item,
                    'score': round(score, 3),
                    'page': result.get('metadata', {}).get('page_start'),
                    'row': result.get('metadata', {}).get('row_index')
                })

    # Add any high-scoring unmatched results (might contain multiple items)
    if unmatched_results and total_tokens < max_context_tokens * 0.8:
        context_parts.append(f"\n**=== Additional Relevant Information ===**\n")

        for result in unmatched_results[:5]:  # Max 5 unmatched
            if result.get('score', 0) > 0.5:  # Only high-scoring
                content_tokens = len(result.get('content', '')) // 4

                if total_tokens + content_tokens > max_context_tokens:
                    break

                context_piece = format_context_piece(result, 0, query_analysis)
                context_parts.append(context_piece)
                files_found.add(result.get('filename', 'Unknown'))
                total_tokens += content_tokens

    logger.info(f"Built multi-item context: {len(context_parts)} chunks, {total_tokens} tokens, {len(files_found)} files")

    context = "\n\n".join(context_parts)

    return {
        'context': context,
        'files_found': files_found,
        'sources': sources,
        'metadata': {
            'files_searched': len(files_found),
            'total_results': len(search_results),
            'relevant_results': len(context_parts),
            'context_tokens': total_tokens,
            'items_with_matches': len(items_with_results),
            'items_requested': len(items)
        }
    }


def format_context_piece(result: Dict, rank: int, query_analysis: Dict) -> str:
    """Format a single context piece"""
    filename = result.get('filename', 'Unknown')
    content = result.get('content', '')
    chunk_type = result.get('chunk_type', 'unknown')
    metadata = result.get('metadata', {})
    
    # Add location context
    location = []
    if metadata.get('page_start'):
        if metadata.get('page_end') and metadata['page_end'] != metadata['page_start']:
            location.append(f"Pages {metadata['page_start']}-{metadata['page_end']}")
        else:
            location.append(f"Page {metadata['page_start']}")
    
    if metadata.get('row_index') is not None:
        location.append(f"Row {metadata['row_index'] + 1}")
    
    location_str = ", ".join(location) if location else chunk_type
    
    # Format with source header
    formatted = f"[Source {rank}: {filename} - {location_str}]\n{content}"
    
    # Add row data for CSV results
    if metadata.get('row_data'):
        try:
            row_data = json.loads(metadata['row_data'])
            formatted += f"\n**Data:** {json.dumps(row_data, indent=2)}"
        except:
            pass
    
    return formatted


def build_rag_prompt(
    message: str,
    context_data: Dict,
    query_analysis: Dict,
    conversation_history: List,
    is_multi_item: bool = False
) -> str:
    """Build optimized prompt for LLM with multi-item support"""

    files_list = ", ".join(context_data['files_found'])

    # Base prompt
    prompt = f"""Based on the following context from uploaded documents, answer the user's question.

**Context from {len(context_data['files_found'])} file(s): {files_list}**

{context_data['context']}

---

**User Question:** {message}

**Instructions:**
- Provide a clear, accurate answer based ONLY on the context above
- Cite specific sources (e.g., "According to [filename]...")
- If the context doesn't contain enough information, say so
- For data queries, preserve exact numbers and formatting
- Format financial data with $ symbols
- Use tables or lists for structured data
- Be concise but comprehensive"""

    # Add query-specific instructions
    if is_multi_item:
        # Extract items for explicit listing
        items = extract_items_from_query(message)
        items_list = "\n".join([f"  {i+1}. {item}" for i, item in enumerate(items)])

        prompt += f"""

**IMPORTANT - Multi-Item Query:**
The user asked about the following {len(items)} items:
{items_list}

You MUST:
- Provide information for ALL {len(items)} items listed above
- Use a clear table format with columns for Item and the requested information
- For any item not found in the context, explicitly state "Not Found"
- Present items in the SAME ORDER as listed above
- DO NOT skip or omit any items from your response"""

    if query_analysis['query_type'] == 'data_lookup':
        prompt += "\n- This is a data lookup query - provide exact values from the context"
    elif query_analysis['query_type'] == 'comparison':
        prompt += "\n- This is a comparison query - clearly contrast the relevant items"
    elif query_analysis['query_type'] == 'aggregation':
        prompt += "\n- This is an aggregation query - summarize or calculate as needed"

    return prompt


def get_system_prompt(query_analysis: Dict, is_multi_item: bool = False) -> str:
    """Get optimized system prompt based on query type"""

    base_prompt = "You are K&B Scout AI, an intelligent document assistant. You help users find and understand information from their uploaded documents."

    if is_multi_item:
        base_prompt += """

IMPORTANT - Multi-Item Query Instructions:
- The user asked about MULTIPLE items at once
- You will receive search results for each individual item
- You MUST respond with information for ALL items requested
- Use a clear table format to present all items
- If any item is not found in the context, explicitly state "Not Found" for that item
- DO NOT omit any requested items from your response
- Present results in the SAME ORDER as the user requested them"""

    if query_analysis['query_type'] == 'data_lookup':
        return base_prompt + "\n\nFocus on providing exact, accurate data values. Always cite the source document. For multi-item queries, use tables to present data clearly."
    elif query_analysis['has_financial']:
        return base_prompt + "\n\nFormat financial data clearly with $ symbols and proper number formatting. Use tables for multiple price comparisons."
    elif query_analysis['is_comparison']:
        return base_prompt + "\n\nProvide clear, side-by-side comparisons. Highlight key differences using tables or structured format."
    else:
        return base_prompt + "\n\nProvide comprehensive, well-structured answers with proper source attribution."


def format_source_attribution(context_data: Dict) -> str:
    """Format source attribution footer"""
    
    attribution = f"\n\n{'─' * 60}\n"
    attribution += f"📚 **Sources:** Found information from {len(context_data['files_found'])} document(s)\n"
    
    # Group sources by file
    sources_by_file = {}
    for source in context_data['sources']:
        filename = source['filename']
        if filename not in sources_by_file:
            sources_by_file[filename] = []
        sources_by_file[filename].append(source)
    
    for filename, sources in sources_by_file.items():
        attribution += f"• **{filename}**: {len(sources)} section(s)"
        
        # Add location details
        locations = []
        for s in sources[:3]:  # Show first 3
            if s.get('page'):
                locations.append(f"p.{s['page']}")
            elif s.get('row') is not None:
                locations.append(f"row {s['row'] + 1}")
        
        if locations:
            attribution += f" ({', '.join(locations)})"
        attribution += "\n"
    
    attribution += f"\n💡 *Tip: Ask for full documents with 'show me the full document for [filename]'*"
    
    return attribution


def format_context_without_llm(context_data: Dict, message: str) -> str:
    """Format context when LLM is not available"""
    
    files_list = ", ".join(context_data['files_found'])
    
    response = f"🔍 **Search Results** (OpenAI unavailable)\n\n"
    response += f"**Query:** {message}\n"
    response += f"**Sources:** {len(context_data['files_found'])} file(s) - {files_list}\n"
    response += f"**Results:** {context_data['metadata']['relevant_results']} relevant sections found\n\n"
    response += "=" * 60 + "\n\n"
    response += context_data['context']
    
    return response




####################################
    

# Add these functions after the existing S3 functions (around line 150)
# and BEFORE the "Text Processing Functions" section

def list_s3_files() -> List[Dict]:
    """List all files in S3 bucket"""
    s3_client = get_s3_client()
    if not s3_client or not S3_BUCKET_NAME:
        return []
    
    try:
        response = s3_client.list_objects_v2(
            Bucket=S3_BUCKET_NAME,
            Prefix='documents/'
        )
        
        files = []
        for obj in response.get('Contents', []):
            # Extract filename from S3 key
            filename = obj['Key'].replace('documents/', '')
            if filename:  # Skip empty names
                files.append({
                    'filename': filename,
                    'size': obj['Size'],
                    'last_modified': obj['LastModified'],
                    's3_key': obj['Key']
                })
        
        return files
    except Exception as e:
        logger.error(f"Error listing S3 files: {e}")
        return []

def sync_with_s3() -> Dict:
    """Sync local document store with S3 bucket and Pinecone"""
    logger.info("Syncing with S3 bucket and Pinecone...")

    s3_files = list_s3_files()
    if not s3_files:
        logger.info("No files found in S3 or S3 not configured")
        return {'synced_files': 0, 'missing_files': 0}

    synced_count = 0
    missing_count = 0
    needs_reprocessing = []

    # Get Pinecone stats to see what's indexed
    index = get_pinecone_index()
    pinecone_vector_count = 0
    if index:
        try:
            stats = index.describe_index_stats()
            pinecone_vector_count = stats.get('total_vector_count', 0)
            logger.info(f"📊 Pinecone currently has {pinecone_vector_count} vectors")
        except Exception as e:
            logger.warning(f"Could not get Pinecone stats: {e}")

    with document_store['lock']:
        for s3_file in s3_files:
            filename = s3_file['filename']

            # Check if file exists in local store
            if filename not in document_store['files']:
                # File exists in S3 but not in local store
                # Add placeholder entry with S3 info
                document_store['files'][filename] = {
                    'doc_id': 'unknown',  # Will be regenerated if reprocessed
                    'chunks': 0,  # Unknown until reprocessed
                    'type': determine_file_type(filename),
                    'uploaded_at': s3_file['last_modified'].isoformat(),
                    's3_uploaded': True,
                    's3_key': s3_file['s3_key'],
                    'size': s3_file['size'],
                    'status': 'in_s3_only'  # Flag to indicate needs reprocessing
                }
                synced_count += 1
                needs_reprocessing.append(filename)
                logger.info(f"✅ Found S3 file not in local store: {filename}")
            else:
                # File exists in local store, check if it's in Pinecone
                file_info = document_store['files'][filename]
                if file_info.get('chunks', 0) == 0 or file_info.get('status') == 'in_s3_only':
                    needs_reprocessing.append(filename)
                    logger.info(f"⚠️ File {filename} exists but has 0 chunks - needs reprocessing")

        # Check for files in local store but not in S3
        for filename in list(document_store['files'].keys()):
            if not any(s3_file['filename'] == filename for s3_file in s3_files):
                file_info = document_store['files'][filename]
                if file_info.get('s3_uploaded', False):
                    # File should be in S3 but isn't
                    file_info['s3_missing'] = True
                    missing_count += 1
                    logger.warning(f"⚠️ File in local store but missing from S3: {filename}")

    logger.info(f"📊 S3 sync complete. Synced: {synced_count}, Missing: {missing_count}")

    # If there are files needing reprocessing and Pinecone is available
    if needs_reprocessing and index:
        logger.info(f"🔄 Found {len(needs_reprocessing)} files needing Pinecone indexing")
        logger.info(f"💡 Files: {', '.join(needs_reprocessing[:5])}{'...' if len(needs_reprocessing) > 5 else ''}")

    return {
        'synced_files': synced_count,
        'missing_files': missing_count,
        'needs_reprocessing': len(needs_reprocessing),
        'files_needing_reprocessing': needs_reprocessing
    }

def determine_file_type(filename: str) -> str:
    """Determine file type from extension"""
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else 'unknown'
    type_map = {
        'pdf': 'PDF Document',
        'csv': 'CSV Spreadsheet',
        'xlsx': 'Excel Spreadsheet',
        'xls': 'Excel Spreadsheet',
        'txt': 'Text Document'
    }
    return type_map.get(ext, 'Unknown')

def reprocess_s3_file(filename: str) -> Dict:
    """Download and reprocess a file from S3"""
    try:
        with document_store['lock']:
            if filename not in document_store['files']:
                return {'success': False, 'error': 'File not found in store'}
            
            file_info = document_store['files'][filename]
            s3_key = file_info.get('s3_key')
            
            if not s3_key:
                return {'success': False, 'error': 'No S3 key found for file'}
        
        # Download from S3 to temp file
        temp_path = os.path.join(tempfile.gettempdir(), filename)
        
        if not download_from_s3(s3_key, temp_path):
            return {'success': False, 'error': 'Failed to download from S3'}
        
        try:
            # Process the file based on extension
            extension = filename.rsplit('.', 1)[1].lower()
            
            if extension == 'pdf':
                doc_result = process_pdf(temp_path, filename)
            elif extension in ['csv', 'xlsx', 'xls']:
                doc_result = process_csv(temp_path, filename)
            else:
                # Text file
                with open(temp_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                
                # Store full content
                with document_store['lock']:
                    document_store['full_content'][filename] = {
                        'content': text,
                        'size': len(text),
                        'type': 'Text'
                    }
                    _cache['full_documents'][filename] = text
                
                chunks = chunk_text(text[:10000])
                embeddings = generate_embeddings_batch(chunks)
                
                doc_id = str(uuid.uuid4())[:8]
                vectors = []
                vector_ids = []
                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                    vector_id = f"{doc_id}_{i}"
                    vector_ids.append(vector_id)
                    vectors.append({
                        'id': vector_id,
                        'values': embedding,
                        'metadata': {
                            'filename': filename,
                            'doc_id': doc_id,
                            'chunk_index': i,
                            'content': chunk[:1000],
                            'file_type': 'text'
                        }
                    })
                
                doc_result = {
                    'success': True,
                    'doc_id': doc_id,
                    'filename': filename,
                    'vectors': vectors,
                    'vector_ids': vector_ids,
                    'chunks': len(chunks),
                    'type': 'Text Document'
                }
            
            # Clean up temp file
            os.remove(temp_path)
            
            if doc_result.get('success'):
                # Upload to Pinecone
                upload_success = upload_to_pinecone(doc_result['vectors'])
                
                if upload_success:
                    # Update document info
                    with document_store['lock']:
                        document_store['files'][filename].update({
                            'doc_id': doc_result['doc_id'],
                            'chunks': doc_result['chunks'],
                            'type': doc_result['type'],
                            'status': 'processed',
                            'reprocessed_at': datetime.utcnow().isoformat()
                        })
                        
                        # Store vector IDs
                        if 'file_vectors' not in document_store:
                            document_store['file_vectors'] = {}
                        document_store['file_vectors'][filename] = doc_result.get('vector_ids', [])
                    
                    return {
                        'success': True,
                        'message': f'Successfully reprocessed {filename}',
                        'chunks': doc_result['chunks']
                    }
                else:
                    return {'success': False, 'error': 'Failed to upload to Pinecone'}
            else:
                return {'success': False, 'error': f'Processing failed: {doc_result.get("error", "Unknown error")}'}
                
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return {'success': False, 'error': f'Processing error: {str(e)}'}
            
    except Exception as e:
        logger.error(f"Error reprocessing S3 file {filename}: {e}")
        return {'success': False, 'error': f'Reprocessing error: {str(e)}'}


# Add these routes AFTER the existing Flask routes (around line 800, after @app.route('/search', methods=['POST']))

@app.route('/sync-s3', methods=['POST'])
def sync_s3():
    """Sync with S3 bucket to find files not in local store"""
    try:
        result = sync_with_s3()

        # Build detailed message
        message_parts = []
        if result['synced_files'] > 0:
            message_parts.append(f"Found {result['synced_files']} new files in S3")
        if result['needs_reprocessing'] > 0:
            message_parts.append(f"{result['needs_reprocessing']} files need processing")
        if result['missing_files'] > 0:
            message_parts.append(f"{result['missing_files']} files missing from S3")

        if not message_parts:
            message_parts.append("All files are in sync")

        message = '. '.join(message_parts) + '.'

        return jsonify({
            'success': True,
            'message': message,
            **result
        })
    except Exception as e:
        logger.error(f"S3 sync error: {e}")
        return jsonify({
            'success': False,
            'message': f'Sync failed: {str(e)}'
        }), 500

@app.route('/reprocess/<filename>', methods=['POST'])
def reprocess_file_route(filename):
    """Reprocess a file from S3"""
    try:
        filename_decoded = filename.replace('%20', ' ')
        
        result = reprocess_s3_file(filename_decoded)
        
        if result['success']:
            return jsonify(result)
        else:
            return jsonify(result), 400
            
    except Exception as e:
        logger.error(f"Error in reprocess endpoint: {e}")
        return jsonify({
            'success': False,
            'message': f'Error reprocessing file: {str(e)}'
        }), 500


# REPLACE the existing initialize_system function with this updated version:

def initialize_system():
    """Initialize system on startup"""
    logger.info("=" * 50)
    logger.info("Initializing K&B Scout AI System...")
    logger.info("=" * 50)
    
    # Test OpenAI
    client = get_openai_client()
    if client:
        try:
            test = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input="test"
            )
            logger.info("✅ OpenAI connection successful")
        except Exception as e:
            logger.error(f"❌ OpenAI test failed: {e}")
    
    # Test Pinecone
    index = get_pinecone_index()
    if index:
        logger.info("✅ Pinecone connection successful")
        try:
            stats = index.describe_index_stats()
            vector_count = stats.get('total_vector_count', 0)
            logger.info(f"📊 Total vectors in database: {vector_count}")
        except Exception as e:
            logger.warning(f"Could not get Pinecone stats: {e}")
    else:
        logger.warning("⚠️ Pinecone not available - will work in limited mode")
    
    # Test S3 (optional) and ALWAYS sync on startup
    s3 = get_s3_client()
    if s3:
        logger.info("✅ S3 client available")
        # ALWAYS sync with S3 on startup
        try:
            logger.info("=" * 50)
            logger.info("🔄 SYNCING WITH S3 ON STARTUP...")
            logger.info("=" * 50)
            sync_result = sync_with_s3()
            logger.info(f"✅ S3 Sync complete: {sync_result['synced_files']} new files, {sync_result['missing_files']} missing")

            # Show what files we found
            with document_store['lock']:
                total_files = len(document_store['files'])
                logger.info(f"📁 Total files after sync: {total_files}")
                if total_files > 0:
                    logger.info(f"📋 Files in store: {list(document_store['files'].keys())}")

            if sync_result.get('needs_reprocessing', 0) > 0:
                logger.warning(f"⚠️ {sync_result['needs_reprocessing']} files need Pinecone indexing!")
                logger.warning(f"   Files: {', '.join(sync_result.get('files_needing_reprocessing', [])[:5])}")

        except Exception as e:
            logger.error(f"❌ S3 sync failed during startup: {e}", exc_info=True)
    else:
        logger.info("ℹ️ S3 not configured (optional)")

    logger.info("=" * 50)
    logger.info("System ready! All searches will query the entire document database.")
    logger.info("=" * 50)

@app.route('/document/<filename>', methods=['GET'])
def get_full_document(filename):
    """API endpoint to retrieve full document content - can be used for downloading"""
    try:
        # Support both with and without extension in URL
        filename_decoded = filename.replace('%20', ' ')
        
        with document_store['lock']:
            # Try exact match first
            if filename_decoded in document_store.get('full_content', {}):
                full_doc = document_store['full_content'][filename_decoded]
            else:
                # Try to find by partial match
                found = None
                for doc_name in document_store.get('full_content', {}).keys():
                    if filename_decoded in doc_name or doc_name.startswith(filename_decoded):
                        found = doc_name
                        break
                
                if found:
                    full_doc = document_store['full_content'][found]
                    filename_decoded = found
                else:
                    return jsonify({
                        'success': False,
                        'message': f'Document {filename} not found'
                    }), 404
            
            # Return as plain text for large documents
            from flask import Response
            
            if len(full_doc['content']) > 100000:  # If larger than 100KB
                # Return as downloadable text file
                return Response(
                    full_doc['content'],
                    mimetype='text/plain',
                    headers={
                        'Content-Disposition': f'attachment; filename="{filename_decoded}.txt"',
                        'Content-Type': 'text/plain; charset=utf-8'
                    }
                )
            else:
                # Return as JSON for smaller documents
                return jsonify({
                    'success': True,
                    'filename': filename_decoded,
                    'type': full_doc['type'],
                    'content': full_doc['content'],
                    'metadata': {
                        'pages': full_doc.get('pages'),
                        'rows': full_doc.get('rows'),
                        'columns': full_doc.get('columns'),
                        'size': len(full_doc['content'])
                    }
                })
                
    except Exception as e:
        logger.error(f"Error retrieving document: {e}")
        return jsonify({
            'success': False,
            'message': 'Error retrieving document'
        }), 500
    
##################################Search API    

@app.route('/search', methods=['POST'])
def search_endpoint():
    """Direct search endpoint that searches across ALL files"""
    data = request.json
    query = data.get('query', '')
    
    if not query:
        return jsonify({'success': False, 'message': 'No query provided'}), 400
    
    try:
        # Search across entire Pinecone database
        results = search_documents(query, top_k=50)
        
        formatted_results = []
        files_found = set()
        
        for match in results:
            if match['score'] > 0.2:  # Lower threshold for search endpoint
                filename = match['metadata'].get('filename', 'Unknown')
                files_found.add(filename)
                formatted_results.append({
                    'content': match['metadata'].get('content', ''),
                    'filename': filename,
                    'score': match['score'],
                    'doc_id': match['metadata'].get('doc_id'),
                    'chunk_index': match['metadata'].get('chunk_index'),
                    'file_type': match['metadata'].get('file_type', 'unknown')
                })
        
        return jsonify({
            'success': True,
            'results': formatted_results,
            'total_results': len(formatted_results),
            'files_found': list(files_found),
            'query': query
        })
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

# -----------------------------
# Setup and Initialization
# -----------------------------
def setup_directories():
    """Create necessary directories"""
    dirs = ['static', 'static/css', 'static/js', 'templates']
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

def initialize_system():
    """Initialize system on startup"""
    logger.info("=" * 50)
    logger.info("Initializing K&B Scout AI System...")
    logger.info("=" * 50)
    
    # Test OpenAI
    client = get_openai_client()
    if client:
        try:
            test = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input="test"
            )
            logger.info("✅ OpenAI connection successful")
        except Exception as e:
            logger.error(f"❌ OpenAI test failed: {e}")
    
    # Test Pinecone
    index = get_pinecone_index()
    if index:
        logger.info("✅ Pinecone connection successful")
        try:
            stats = index.describe_index_stats()
            vector_count = stats.get('total_vector_count', 0)
            logger.info(f"📊 Total vectors in database: {vector_count}")
        except Exception as e:
            logger.warning(f"Could not get Pinecone stats: {e}")
    else:
        logger.warning("⚠️ Pinecone not available - will work in limited mode")
    
    # Test S3 (optional)
    s3 = get_s3_client()
    if s3:
        logger.info("✅ S3 client available")
    else:
        logger.info("ℹ️ S3 not configured (optional)")
    
    logger.info("=" * 50)
    logger.info("System ready! All searches will query the entire document database.")
    logger.info("=" * 50)

# -----------------------------
# Main Entry Point
# -----------------------------
if __name__ == '__main__':
    setup_directories()
    initialize_system()
    
    logger.info("📍 Access the application at: http://localhost:5000")
    logger.info("🔍 Searches will query across ALL uploaded files in the database")
    logger.info("🗂️ File management: GET /files, DELETE /files/<filename>")
    
    app.run(debug=False, port=5000,host='0.0.0.0', threaded=True)
