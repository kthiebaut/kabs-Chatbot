import os
import re
import uuid
import time
import hashlib
import tempfile
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
import concurrent.futures
from decimal import Decimal, InvalidOperation
import json

# Flask imports
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

# Document processing
import pandas as pd
from pypdf import PdfReader

# Vector DB - Pinecone
from pinecone import Pinecone, ServerlessSpec

# AWS S3
import boto3
from botocore.exceptions import NoCredentialsError

# OpenAI
import tiktoken
from openai import OpenAI

# Fuzzy matching for better item code matching
try:
    from fuzzywuzzy import fuzz, process
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False
    print("Warning: fuzzywuzzy not installed. Install with: pip install fuzzywuzzy python-Levenshtein")

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# -----------------------------
# Configuration
# -----------------------------
CHUNK_TOKENS = 500
OVERLAP_TOKENS = 50
MAX_CONTEXT_LENGTH = 12000
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
PINECONE_INDEX_NAME = "kb-scout-documents"
MAX_FILE_SIZE = 32 * 1024 * 1024  # 32MB

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
    'csv_lookup_cache': {},  # Cache for direct CSV lookups
    'pdf_item_cache': {},    # Cache for PDF item extraction
    'flexible_patterns': {}  # Cache for flexible query patterns
}

# -----------------------------
# Client Initialization
# -----------------------------
@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
    """Get or create OpenAI client"""
    if not _cache['openai_client']:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        _cache['openai_client'] = OpenAI(api_key=api_key)
    return _cache['openai_client']

@lru_cache(maxsize=1)
def get_pinecone_client():
    """Get or create Pinecone client"""
    if not _cache['pinecone_client']:
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY not found in environment")
        _cache['pinecone_client'] = Pinecone(api_key=api_key)
    return _cache['pinecone_client']

@lru_cache(maxsize=1)
def get_s3_client():
    """Get or create S3 client"""
    if not _cache['s3_client']:
        access_key = os.getenv("AWS_ACCESS_KEY_ID")
        secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        if not access_key or not secret_key:
            return None  # S3 is optional
        _cache['s3_client'] = boto3.client(
            's3',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=os.getenv("AWS_REGION", "us-east-1")
        )
    return _cache['s3_client']

def get_tokenizer():
    """Get or create tokenizer"""
    if not _cache['tokenizer']:
        _cache['tokenizer'] = tiktoken.get_encoding("cl100k_base")
    return _cache['tokenizer']

# -----------------------------
# Enhanced Item Code Extraction
# -----------------------------
def extract_item_codes_from_query(query: str) -> List[str]:
    """Enhanced item code extraction with flexible parsing"""
    items = []
    query_upper = query.upper()
    
    print(f"Processing query: '{query}'")
    
    # Step 1: Handle comma/separator-delimited lists
    separators = [',', ';', ' AND ', ' & ', ' OR ']
    potential_items = [query_upper]
    
    for sep in separators:
        if sep in query_upper:
            potential_items = []
            parts = query_upper.split(sep)
            for part in parts:
                cleaned = part.strip()
                # Remove common prefixes/suffixes
                cleaned = re.sub(r'^(WHAT IS THE |WHAT IS |THE |PRICE OF |COST OF |HOW MUCH IS |FIND |GET |SHOW ME )', '', cleaned)
                cleaned = re.sub(r'(\?|PRICE|COST|TOTAL)$', '', cleaned).strip()
                if cleaned and len(cleaned) > 2:
                    potential_items.append(cleaned)
            break
    
    # Step 2: Extract item codes from each potential item
    for potential_item in potential_items:
        potential_item = potential_item.strip()
        
        # Skip pure question words
        if potential_item.lower() in ['what', 'how', 'when', 'where', 'why', 'the', 'is', 'of', 'price', 'are', 'cost', 'much', 'total']:
            continue
        
        # Enhanced patterns for different item code formats
        patterns = [
            r'^([A-Z]{2,4}\d+[A-Z]*)$',                    # BS2460, BS3096
            r'^([A-Z]+\s+\d+[A-Z]*[A-Z])$',               # KNOB 3563MB
            r'^([A-Z]+\d+\s+[A-Z\s]+[A-Z])$',             # BSS33 SHELF RAS
            r'^([A-Z]+\d+\s+[A-Z\s]*[A-Z]+)$',            # Variations with spaces
            r'([A-Z]{2,4}\d+[A-Z]*)',                     # Find within text
            r'([A-Z]+\s+\d+[A-Z]+)',                      # Find KNOB 3563MB within text
            r'([A-Z]+\d+\s+[A-Z\s]+[A-Z]+)',              # Find complex codes
            r'([A-Z]+[-_]\d+[-_][A-Z]+)',                 # Codes with separators
            r'([A-Z]+\d+[-_][A-Z]+)',                     # Simple separator codes
        ]
        
        found_match = False
        for pattern in patterns:
            matches = re.findall(pattern, potential_item)
            if matches:
                for match in matches:
                    if isinstance(match, tuple):
                        match = match[0] if match[0] else match[1]
                    
                    match = match.strip()
                    if len(match) > 2 and match not in items:
                        items.append(match)
                        found_match = True
                        print(f"Pattern matched: '{match}' from '{potential_item}'")
        
        # Fallback: if it looks like an item code, add it
        if not found_match and len(potential_item) > 2:
            if re.search(r'[A-Z]', potential_item) and re.search(r'\d', potential_item):
                # Additional validation - must have reasonable structure
                if not re.search(r'^(WHAT|HOW|THE|PRICE|COST)', potential_item):
                    items.append(potential_item)
                    print(f"Added as potential item: '{potential_item}'")
    
    # Remove duplicates while preserving order
    unique_items = []
    for item in items:
        if item not in unique_items:
            unique_items.append(item)
    
    print(f"Final extracted item codes: {unique_items}")
    return unique_items

def extract_items_from_catalog(catalog_name: str) -> List[str]:
    """Extract all item codes from a catalog/CSV file"""
    if catalog_name in _cache['pdf_item_cache']:
        return _cache['pdf_item_cache'][catalog_name]
    
    items = []
    
    # Search through cached CSV data
    if catalog_name in _cache['csv_lookup_cache']:
        df = _cache['csv_lookup_cache'][catalog_name]
        if df is not None:
            # Look for SKU columns
            sku_columns = [col for col in df.columns if any(term in col.lower() for term in ['sku', 'item', 'product', 'code', 'part'])]
            if sku_columns:
                sku_col = sku_columns[0]
                
                # Extract all non-null SKUs
                for idx, row in df.iterrows():
                    sku_val = str(row[sku_col]) if not pd.isna(row[sku_col]) else ""
                    if sku_val and sku_val.strip() and len(sku_val.strip()) > 2:
                        sku_clean = sku_val.strip()
                        # Filter out obvious headers/categories
                        if not re.search(r'^(CATEGORY|SECTION|TYPE|WALL|BASE|DOOR)', sku_clean.upper()):
                            if re.search(r'[A-Z]', sku_clean) and re.search(r'\d', sku_clean):
                                items.append(sku_clean)
    
    # Cache the result
    _cache['pdf_item_cache'][catalog_name] = items
    print(f"Extracted {len(items)} items from catalog: {catalog_name}")
    
    return items

# -----------------------------
# Flexible Query Processing
# -----------------------------
def normalize_query(query: str) -> str:
    """Normalize query for flexible matching"""
    normalized = query.lower().strip()
    
    # Common variations mapping
    replacements = {
        r'\bwhat\'s\b': 'what is',
        r'\bhow much is\b': 'price of',
        r'\bhow much are\b': 'price of',
        r'\bcost of\b': 'price of',
        r'\bshow me\b': 'find',
        r'\btell me\b': 'what is',
        r'\bget me\b': 'find',
        r'\blook up\b': 'find',
        r'\bfind the price\b': 'price of',
        r'\bget pricing\b': 'price of',
        r'\btotal cost\b': 'total price',
        r'\ball items\b': 'every item',
        r'\bevery single\b': 'every',
        r'\bfull list\b': 'all items',
        r'\bcomplete list\b': 'all items',
    }
    
    for pattern, replacement in replacements.items():
        normalized = re.sub(pattern, replacement, normalized)
    
    return normalized

def detect_query_intent(query: str) -> Dict[str, any]:
    """Detect the intent of the query"""
    normalized = normalize_query(query)
    
    intent = {
        'type': 'unknown',
        'wants_pricing': False,
        'wants_total': False,
        'wants_all_items': False,
        'wants_catalog_pricing': False,
        'specific_items': [],
        'confidence': 0.0
    }
    
    # Pricing indicators
    pricing_patterns = [
        r'\bprice\b', r'\bcost\b', r'\bhow much\b', r'\bdollar\b', r'\$',
        r'\bpricing\b', r'\bexpense\b', r'\btotal\b'
    ]
    
    if any(re.search(pattern, normalized) for pattern in pricing_patterns):
        intent['wants_pricing'] = True
        intent['confidence'] += 0.3
    
    # Total/sum indicators
    total_patterns = [
        r'\btotal\b', r'\bsum\b', r'\ball together\b', r'\bcombined\b',
        r'\bgrand total\b', r'\boverall\b', r'\bentire\b'
    ]
    
    if any(re.search(pattern, normalized) for pattern in total_patterns):
        intent['wants_total'] = True
        intent['confidence'] += 0.2
    
    # All items indicators
    all_items_patterns = [
        r'\ball items\b', r'\bevery item\b', r'\bfull list\b', r'\bcomplete list\b',
        r'\beverything\b', r'\bentire (pdf|catalog|document)\b', r'\bfull (pdf|catalog|document)\b', 
        r'\bwhole document\b', r'\bcatalog pricing\b'
    ]
    
    if any(re.search(pattern, normalized) for pattern in all_items_patterns):
        intent['wants_all_items'] = True
        intent['wants_catalog_pricing'] = True
        intent['confidence'] += 0.4
    
    # Document-specific indicators
    doc_patterns = [
        r'\b(pdf|document|file|quote|estimate|proposal|order|catalog)\b'
    ]
    
    if any(re.search(pattern, normalized) for pattern in doc_patterns):
        intent['wants_catalog_pricing'] = True
        intent['confidence'] += 0.2
    
    # Extract specific items
    intent['specific_items'] = extract_item_codes_from_query(query)
    if intent['specific_items']:
        intent['confidence'] += 0.3
    
    # Determine primary intent type
    if intent['wants_all_items'] or intent['wants_catalog_pricing']:
        intent['type'] = 'catalog_pricing'
    elif intent['specific_items'] and intent['wants_pricing']:
        intent['type'] = 'specific_pricing'
    elif intent['wants_pricing']:
        intent['type'] = 'general_pricing'
    else:
        intent['type'] = 'general_query'
    
    print(f"Query intent: {intent}")
    return intent

# -----------------------------
# Enhanced Direct CSV Lookup
# -----------------------------
def enhanced_direct_csv_lookup(item_codes: List[str]) -> Dict[str, Dict]:
    """Enhanced direct lookup with fuzzy matching and flexible patterns"""
    results = {}
    
    for filename, df in _cache['csv_lookup_cache'].items():
        if df is None:
            continue
            
        try:
            # Find relevant columns
            sku_columns = [col for col in df.columns if any(term in col.lower() for term in ['sku', 'item', 'product', 'code', 'part'])]
            price_columns = [col for col in df.columns if any(term in col.lower() for term in ['price', 'cost', 'amount', 'value', '$'])]
            
            if not sku_columns:
                continue
                
            sku_col = sku_columns[0]
            price_col = price_columns[0] if price_columns else None
            
            for item_code in item_codes:
                if item_code in results:
                    continue
                    
                best_match = None
                best_score = 0
                
                for idx, row in df.iterrows():
                    sku_val = str(row[sku_col]) if not pd.isna(row[sku_col]) else ""
                    
                    if not sku_val.strip():
                        continue
                    
                    sku_clean = sku_val.strip().upper()
                    item_clean = item_code.strip().upper()
                    
                    # Multiple matching strategies
                    score = 0
                    match_type = "no_match"
                    
                    # Exact match (highest priority)
                    if sku_clean == item_clean:
                        score = 100
                        match_type = "exact"
                    # Contains match (high priority)
                    elif item_clean in sku_clean or sku_clean in item_clean:
                        score = 90
                        match_type = "contains"
                    # Word boundary match
                    elif re.search(rf'\b{re.escape(item_clean)}\b', sku_clean):
                        score = 85
                        match_type = "word_boundary"
                    # Fuzzy match
                    elif FUZZY_AVAILABLE:
                        fuzzy_score = fuzz.ratio(item_clean, sku_clean)
                        if fuzzy_score > 80:
                            score = fuzzy_score
                            match_type = "fuzzy"
                        # Token sort ratio for items with different word orders
                        token_score = fuzz.token_sort_ratio(item_clean, sku_clean)
                        if token_score > score and token_score > 85:
                            score = token_score
                            match_type = "token_fuzzy"
                    
                    # Update best match
                    if score > best_score:
                        price_val = row[price_col] if price_col and not pd.isna(row[price_col]) else None
                        best_match = {
                            'sku': sku_val.strip(),
                            'price': price_val,
                            'source': filename,
                            'row': idx,
                            'match_type': match_type,
                            'score': score,
                            'raw_row': dict(row)
                        }
                        best_score = score
                
                # Add best match if good enough
                if best_match and best_score >= 75:  # Lower threshold for flexibility
                    results[item_code] = best_match
                    print(f"Match found - {item_code}: {best_match['sku']} = ${best_match['price']} (score: {best_score}, type: {best_match['match_type']})")
        
        except Exception as e:
            print(f"Error in enhanced lookup for {filename}: {e}")
            continue
    
    return results

def calculate_total_price(pricing_results: Dict[str, Dict]) -> Tuple[Decimal, List[str]]:
    """Calculate total price from pricing results"""
    total = Decimal('0.00')
    errors = []
    
    for item_code, match_data in pricing_results.items():
        try:
            price = match_data.get('price')
            if price is not None:
                if isinstance(price, (int, float)):
                    total += Decimal(str(price))
                elif isinstance(price, str):
                    # Clean price string
                    price_clean = re.sub(r'[^\d.,]', '', str(price))
                    if price_clean:
                        total += Decimal(price_clean)
            else:
                errors.append(f"{item_code}: No price available")
        except (InvalidOperation, TypeError) as e:
            errors.append(f"{item_code}: Invalid price format")
            print(f"Price calculation error for {item_code}: {e}")
    
    return total, errors

# -----------------------------
# Catalog Pricing Handler
# -----------------------------
def handle_catalog_pricing_query(query: str, intent: Dict) -> Dict:
    """Handle catalog pricing queries with full item lists and totals"""
    print(f"Handling catalog pricing query with intent: {intent}")
    
    # Get all available CSV files (representing different catalogs)
    available_files = list(_cache['csv_lookup_cache'].keys())
    
    if not available_files:
        return {
            'success': False,
            'response': 'No pricing catalogs are currently loaded. Please upload a CSV pricing file first.'
        }
    
    # For now, use the first available file - in future could let user specify
    target_file = available_files[0]
    print(f"Using pricing file: {target_file}")
    
    # Extract all items from the file
    all_items = extract_items_from_catalog(target_file)
    
    if not all_items:
        return {
            'success': False,
            'response': f'No items found in {target_file}. Please check the file format.'
        }
    
    print(f"Found {len(all_items)} total items in catalog")
    
    # Limit to reasonable number for processing (prevent timeouts)
    max_items = 50  # Reduced for better performance
    if len(all_items) > max_items:
        items_to_process = all_items[:max_items]
        print(f"Limited processing to first {max_items} items")
    else:
        items_to_process = all_items
    
    # Get pricing for all items
    pricing_results = enhanced_direct_csv_lookup(items_to_process)
    
    # Calculate totals
    total_price, price_errors = calculate_total_price(pricing_results)
    
    # Build response
    response_parts = []
    
    if intent.get('wants_all_items', False):
        response_parts.append("**COMPLETE PRICING LIST:**\n")
        
        # Group by found/not found
        found_items = []
        not_found_items = []
        
        for item_code in items_to_process:
            if item_code in pricing_results:
                match_data = pricing_results[item_code]
                price = match_data.get('price')
                if price is not None:
                    found_items.append(f"**{item_code}**: ${price}")
                else:
                    not_found_items.append(item_code)
            else:
                not_found_items.append(item_code)
        
        # Add found items
        if found_items:
            response_parts.append("\n".join(found_items))
        
        # Add not found items summary
        if not_found_items:
            response_parts.append(f"\n**ITEMS WITHOUT PRICING:** {len(not_found_items)} items")
            if len(not_found_items) <= 10:
                response_parts.append(", ".join(not_found_items))
    
    # Always add summary
    response_parts.append(f"\n**SUMMARY:**")
    response_parts.append(f"- Total items processed: {len(items_to_process)}")
    response_parts.append(f"- Items with pricing: {len(pricing_results)}")
    response_parts.append(f"- Items without pricing: {len(items_to_process) - len(pricing_results)}")
    response_parts.append(f"- **TOTAL PRICE: ${total_price}**")
    
    if len(all_items) > max_items:
        response_parts.append(f"\n*Note: Processing limited to first {max_items} items. Full catalog contains {len(all_items)} items.*")
    
    response = "\n".join(response_parts)
    
    return {
        'success': True,
        'response': response,
        'metadata': {
            'total_items_found': len(all_items),
            'items_processed': len(items_to_process),
            'items_with_pricing': len(pricing_results),
            'total_price': str(total_price),
            'source_file': target_file,
            'query_type': 'catalog_pricing'
        }
    }

# -----------------------------
# Enhanced Multi-Item Handler
# -----------------------------
def handle_specific_pricing_query(query: str, item_codes: List[str], intent: Dict) -> Dict:
    """Handle specific item pricing queries with totals"""
    print(f"Handling specific pricing query for items: {item_codes}")
    
    # Get direct matches
    direct_matches = enhanced_direct_csv_lookup(item_codes)
    print(f"Direct matches found: {len(direct_matches)}")
    
    # Build individual item responses
    answer_parts = []
    successful_matches = {}
    
    for item_code in item_codes:
        if item_code in direct_matches:
            match_data = direct_matches[item_code]
            price = match_data.get('price')
            if price is not None:
                answer_parts.append(f"**{item_code}**: ${price}")
                successful_matches[item_code] = match_data
            else:
                answer_parts.append(f"**{item_code}**: Price not available")
        else:
            # Try fallback search
            fallback_results = search_pinecone(f"SKU {item_code} price", top_k=5)
            
            found_in_fallback = False
            for text, metadata, score in fallback_results:
                if item_code.upper() in text.upper():
                    price_match = re.search(rf'{re.escape(item_code)}[^$]*\$?(\d+\.?\d*)', text, re.IGNORECASE)
                    if price_match:
                        answer_parts.append(f"**{item_code}**: ${price_match.group(1)}")
                        found_in_fallback = True
                        break
            
            if not found_in_fallback:
                answer_parts.append(f"**{item_code}**: Price not available in the context")
    
    # Add total if requested or if multiple items
    if intent.get('wants_total', False) or len(item_codes) > 1:
        total_price, price_errors = calculate_total_price(successful_matches)
        if successful_matches:
            answer_parts.append(f"\n**TOTAL PRICE: ${total_price}**")
            if price_errors:
                answer_parts.append(f"*Note: {len(price_errors)} items excluded from total due to missing prices*")
    
    answer = "\n".join(answer_parts)
    
    # Add sources
    sources = set()
    for match_data in successful_matches.values():
        sources.add(match_data['source'])
    
    if sources:
        answer += f"\n\n📚 Sources: {', '.join(sources)}"
    
    return {
        'success': True,
        'response': answer,
        'metadata': {
            'item_codes_requested': item_codes,
            'items_found': len(successful_matches),
            'total_price': str(calculate_total_price(successful_matches)[0]) if successful_matches else '0.00',
            'query_type': 'specific_pricing'
        }
    }

# -----------------------------
# Core RAG Functions
# -----------------------------
def parse_multiple_questions(message: str) -> List[str]:
    """Parse message into multiple questions/inputs"""
    message = re.sub(r'\s+', ' ', message.strip())
    
    question_patterns = [
        r'\?\s*(?=[A-Z])',  
        r'\?\s*\d+\)',      
        r'\?\s*[-•]',       
        r';\s*(?=[A-Z])',   
        r'\n\s*\d+\.',      
        r'\n\s*[-•]',       
        r'(?:also|additionally|furthermore|moreover|and)\s+(?:what|how|when|where|why|can|could|should|would|tell me|explain)',
    ]
    
    questions = [message]
    
    for pattern in question_patterns:
        new_questions = []
        for q in questions:
            parts = re.split(pattern, q, flags=re.IGNORECASE)
            if len(parts) > 1:
                for i, part in enumerate(parts):
                    if i == 0:
                        new_questions.append(part.strip())
                    else:
                        new_questions.append(part.strip())
            else:
                new_questions.append(q)
        questions = new_questions
        break
    
    clean_questions = []
    for q in questions:
        q = q.strip()
        if q and len(q) > 5:
            if not q.endswith(('?', '.', '!', ':')):
                q += '?'
            clean_questions.append(q)
    
    if len(clean_questions) <= 1:
        keywords = ['and', 'also', 'additionally', 'furthermore', 'plus', 'as well as']
        for keyword in keywords:
            if keyword in message.lower():
                parts = re.split(rf'\s+{keyword}\s+', message, flags=re.IGNORECASE)
                if len(parts) > 1:
                    clean_questions = [part.strip() + '?' for part in parts if part.strip()]
                    break
    
    return clean_questions if clean_questions else [message]

def get_or_create_index(index_name: str = PINECONE_INDEX_NAME):
    """Get or create Pinecone index"""
    if _cache['pinecone_index']:
        return _cache['pinecone_index']
    
    pc = get_pinecone_client()
    existing_indexes = [idx.name for idx in pc.list_indexes().indexes]
    
    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        time.sleep(5)
    
    _cache['pinecone_index'] = pc.Index(index_name)
    return _cache['pinecone_index']

def chunk_text(text: str, chunk_size: int = CHUNK_TOKENS, overlap: int = OVERLAP_TOKENS) -> List[str]:
    """Split text into chunks"""
    if not text or not text.strip():
        return []
    
    tokenizer = get_tokenizer()
    text = re.sub(r'\s+', ' ', text).strip()
    
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    if not paragraphs:
        paragraphs = [text]
    
    chunks = []
    current_chunk = ""
    current_tokens = 0
    
    for para in paragraphs:
        para_tokens = len(tokenizer.encode(para))
        
        if current_tokens + para_tokens > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = para
            current_tokens = para_tokens
        else:
            current_chunk += ("\n\n" + para if current_chunk else para)
            current_tokens += para_tokens
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks

def create_embeddings(texts: List[str]) -> List[List[float]]:
    """Create embeddings for texts"""
    if not texts:
        return []
    
    client = get_openai_client()
    
    # Check cache
    embeddings = []
    texts_to_embed = []
    cache_indices = []
    
    for i, text in enumerate(texts):
        cache_key = hashlib.md5(text.encode()).hexdigest()
        if cache_key in _cache['embeddings']:
            embeddings.append(_cache['embeddings'][cache_key])
        else:
            texts_to_embed.append(text)
            cache_indices.append(i)
            embeddings.append(None)
    
    # Embed uncached texts
    if texts_to_embed:
        batch_size = 50
        new_embeddings = []
        
        for i in range(0, len(texts_to_embed), batch_size):
            batch = texts_to_embed[i:i + batch_size]
            response = client.embeddings.create(input=batch, model=EMBEDDING_MODEL)
            new_embeddings.extend([d.embedding for d in response.data])
        
        # Update cache and results
        for text, embedding, idx in zip(texts_to_embed, new_embeddings, cache_indices):
            cache_key = hashlib.md5(text.encode()).hexdigest()
            _cache['embeddings'][cache_key] = embedding
            embeddings[idx] = embedding
    
    return embeddings

# -----------------------------
# Enhanced Document Processing
# -----------------------------
def process_pdf(file_path: str) -> List[Tuple[str, Dict]]:
    """Process PDF file"""
    reader = PdfReader(file_path)
    filename = os.path.basename(file_path)
    results = []
    
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = re.sub(r'\s+', ' ', text).strip()
        if text and len(text) > 20:
            results.append((text, {
                "source": filename,
                "type": "pdf",
                "page": i + 1
            }))
    
    return results

def enhanced_process_csv(file_path: str) -> List[Tuple[str, Dict]]:
    """Enhanced CSV processing with better item matching"""
    filename = os.path.basename(file_path)
    results = []
    
    # Try different encodings
    encodings = ['utf-8', 'iso-8859-1', 'windows-1252', 'latin-1', 'cp1252', 'utf-16']
    df = None
    
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding, on_bad_lines='skip')
            print(f"Successfully read CSV with {encoding} encoding")
            break
        except (UnicodeDecodeError, LookupError) as e:
            continue
        except Exception as e:
            if encoding == encodings[-1]:
                print(f"Error processing CSV after all encoding attempts: {e}")
                return []
    
    if df is None:
        print(f"Could not read CSV file {filename} with any encoding")
        return []
    
    try:
        # Clean column names - remove extra whitespace and normalize
        df.columns = df.columns.str.strip()
        
        # Cache the CSV data for direct lookups
        _cache['csv_lookup_cache'][filename] = df
        
        # Add header info with all columns for better context
        header = f"CSV: {filename} | Columns: {', '.join(df.columns.astype(str))}"
        results.append((header, {"source": filename, "type": "csv", "row": 0}))
        
        # Enhanced processing for price/catalog data
        # Look for common price-related columns
        sku_columns = [col for col in df.columns if any(term in col.lower() for term in ['sku', 'item', 'product', 'code', 'part'])]
        price_columns = [col for col in df.columns if any(term in col.lower() for term in ['price', 'cost', 'amount', 'value', '$'])]
        
        # Create lookup chunks for better searching
        if sku_columns and price_columns:
            sku_col = sku_columns[0]
            price_col = price_columns[0]
            
            # Create grouped chunks by product categories or sections
            category_chunk = ""
            chunk_rows = []
            
            for idx, row in df.head(5000).iterrows():
                # Skip completely empty rows
                if row.isna().all():
                    continue
                
                # Check if this is a category header (SKU exists but price is missing)
                sku_val = str(row[sku_col]) if not pd.isna(row[sku_col]) else ""
                price_val = row[price_col] if not pd.isna(row[price_col]) else None
                
                # If SKU exists but no price, might be a category header
                if sku_val and sku_val.strip() and price_val is None:
                    # Save previous chunk if it has content
                    if chunk_rows:
                        chunk_text = f"Category: {category_chunk}\n" + "\n".join(chunk_rows)
                        results.append((chunk_text, {
                            "source": filename,
                            "type": "csv_section",
                            "category": category_chunk,
                            "row_start": len(results)
                        }))
                        chunk_rows = []
                    
                    category_chunk = sku_val.strip()
                    continue
                
                # Regular data row
                if sku_val and sku_val.strip():
                    # Create individual item entry
                    row_parts = []
                    for col, val in row.items():
                        if pd.notna(val) and str(val).strip():
                            row_parts.append(f"{col}: {val}")
                    
                    if row_parts:
                        individual_row = f"SKU: {sku_val} | " + " | ".join(row_parts)
                        
                        # Add to current chunk
                        chunk_rows.append(individual_row)
                        
                        # Also create individual item entry for exact matching
                        results.append((individual_row, {
                            "source": filename,
                            "type": "csv",
                            "row": idx + 1,
                            "sku": sku_val.strip(),
                            "category": category_chunk,
                            "price": price_val
                        }))
                        
                        # If chunk gets too large, save it
                        if len(chunk_rows) >= 20:
                            chunk_text = f"Category: {category_chunk}\n" + "\n".join(chunk_rows)
                            results.append((chunk_text, {
                                "source": filename,
                                "type": "csv_section",
                                "category": category_chunk,
                                "row_count": len(chunk_rows)
                            }))
                            chunk_rows = []
            
            # Save final chunk
            if chunk_rows:
                chunk_text = f"Category: {category_chunk}\n" + "\n".join(chunk_rows)
                results.append((chunk_text, {
                    "source": filename,
                    "type": "csv_section", 
                    "category": category_chunk,
                    "row_count": len(chunk_rows)
                }))
        
        else:
            # Fallback to original processing if no clear SKU/price columns
            for idx, row in df.head(5000).iterrows():
                row_text = " | ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
                if row_text:
                    results.append((row_text, {
                        "source": filename,
                        "type": "csv",
                        "row": idx + 1
                    }))
    
    except Exception as e:
        print(f"Error processing CSV data: {e}")
    
    return results

def process_csv(file_path: str) -> List[Tuple[str, Dict]]:
    """Wrapper to use enhanced CSV processing"""
    return enhanced_process_csv(file_path)

def process_excel(file_path: str) -> List[Tuple[str, Dict]]:
    """Process Excel file"""
    filename = os.path.basename(file_path)
    results = []
    
    try:
        xl_file = pd.ExcelFile(file_path)
        
        for sheet_name in xl_file.sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            
            # Cache the Excel data for direct lookups
            cache_key = f"{filename}#{sheet_name}"
            _cache['csv_lookup_cache'][cache_key] = df
            
            # Add header info
            header = f"Excel: {filename} | Sheet: {sheet_name} | Columns: {', '.join(df.columns)}"
            results.append((header, {
                "source": f"{filename}#{sheet_name}",
                "type": "xlsx",
                "row": 0
            }))
            
            # Process rows
            for idx, row in df.head(1000).iterrows():  # Limit per sheet
                row_text = " | ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
                if row_text:
                    results.append((row_text, {
                        "source": f"{filename}#{sheet_name}",
                        "type": "xlsx",
                        "row": idx + 1,
                        "sheet": sheet_name
                    }))
    except Exception as e:
        print(f"Error processing Excel: {e}")
    
    return results

def process_text(file_path: str) -> List[Tuple[str, Dict]]:
    """Process text file"""
    filename = os.path.basename(file_path)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return [(content, {"source": filename, "type": "txt", "page": 1})]
    except Exception as e:
        print(f"Error processing text file: {e}")
        return []

# -----------------------------
# S3 Operations
# -----------------------------
def upload_to_s3(file_path: str, bucket: str) -> Optional[str]:
    """Upload file to S3"""
    s3 = get_s3_client()
    if not s3 or not bucket:
        return None
    
    try:
        filename = os.path.basename(file_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        s3_key = f"documents/{timestamp}_{filename}"
        
        s3.upload_file(file_path, bucket, s3_key)
        return s3_key
    except Exception as e:
        print(f"S3 upload error: {e}")
        return None

def get_s3_url(bucket: str, key: str) -> str:
    """Generate S3 URL from bucket and key"""
    return f"https://{bucket}.s3.amazonaws.com/{key}"

def save_file_metadata_to_s3(metadata: dict, bucket: str):
    """Save file metadata to S3"""
    s3 = get_s3_client()
    if not s3 or not bucket:
        return False
    
    try:
        import json
        metadata_key = f"metadata/files_metadata.json"
        
        # Get existing metadata
        existing = get_file_metadata_from_s3(bucket)
        existing.update(metadata)
        
        # Upload updated metadata
        s3.put_object(
            Bucket=bucket,
            Key=metadata_key,
            Body=json.dumps(existing, default=str),
            ContentType='application/json'
        )
        return True
    except Exception as e:
        print(f"Error saving metadata to S3: {e}")
        return False

def get_file_metadata_from_s3(bucket: str) -> dict:
    """Get file metadata from S3"""
    s3 = get_s3_client()
    if not s3 or not bucket:
        return {}
    
    try:
        import json
        response = s3.get_object(Bucket=bucket, Key='metadata/files_metadata.json')
        return json.loads(response['Body'].read())
    except s3.exceptions.NoSuchKey:
        return {}
    except Exception as e:
        print(f"Error getting metadata from S3: {e}")
        return {}

def get_all_uploaded_files() -> dict:
    """Get all uploaded files from S3 or local cache"""
    bucket = os.getenv('S3_BUCKET_NAME')
    
    if bucket:
        try:
            metadata = get_file_metadata_from_s3(bucket)
            for filename, info in metadata.items():
                if 's3_key' in info and bucket:
                    info['s3_url'] = get_s3_url(bucket, info['s3_key'])
            return metadata
        except Exception as e:
            print(f"Error getting files from S3: {e}")
            return _cache.get('uploaded_files', {})
    else:
        return _cache.get('uploaded_files', {})

# -----------------------------
# Enhanced RAG Operations
# -----------------------------
@dataclass
class RAGChunk:
    id: str
    text: str
    metadata: Dict

def add_to_pinecone(chunks: List[RAGChunk]) -> bool:
    """Add chunks to Pinecone index"""
    if not chunks:
        return False
    
    index = get_or_create_index()
    
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        
        # Create embeddings
        texts = [c.text for c in batch]
        embeddings = create_embeddings(texts)
        
        # Prepare vectors
        vectors = []
        for chunk, embedding in zip(batch, embeddings):
            metadata = chunk.metadata.copy()
            metadata['text'] = chunk.text[:1000]  # Store preview
            
            vectors.append({
                'id': chunk.id,
                'values': embedding,
                'metadata': metadata
            })
        
        # Upsert to Pinecone
        index.upsert(vectors=vectors)
    
    return True

def search_pinecone(query: str, top_k: int = 10) -> List[Tuple[str, Dict, float]]:
    """Search Pinecone index"""
    index = get_or_create_index()
    
    # Get query embedding
    query_embedding = create_embeddings([query])[0]
    
    # Search
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    
    output = []
    for match in results['matches']:
        text = match['metadata'].get('text', '')
        metadata = {k: v for k, v in match['metadata'].items() if k != 'text'}
        score = match['score']
        output.append((text, metadata, 1 - score))  # Convert to distance
    
    return output

def search_pinecone_with_fallback(query: str, item_codes: List[str] = None, top_k: int = 20) -> List[Tuple[str, Dict, float]]:
    """Enhanced search with fallback for exact item matching"""
    # First, try direct CSV lookup for exact matches
    exact_matches = {}
    if item_codes:
        exact_matches = enhanced_direct_csv_lookup(item_codes)
        print(f"Direct CSV lookup found: {list(exact_matches.keys())}")
    
    # Then do regular semantic search
    results = search_pinecone(query, top_k=top_k)
    
    # If we have specific item codes, also do targeted searches
    if item_codes:
        additional_results = []
        for code in item_codes:
            # Search for each item code specifically
            code_results = search_pinecone(f"SKU {code} price", top_k=5)
            additional_results.extend(code_results)
            
            # Also try variations
            code_results2 = search_pinecone(f"{code} item product", top_k=3)
            additional_results.extend(code_results2)
        
        # Combine and deduplicate results
        all_results = results + additional_results
        seen_texts = set()
        unique_results = []
        
        for text, metadata, score in all_results:
            if text not in seen_texts:
                seen_texts.add(text)
                unique_results.append((text, metadata, score))
        
        # Sort by relevance score
        unique_results.sort(key=lambda x: x[2])
        results = unique_results[:top_k]
    
    # Enhance results with direct lookup data
    if exact_matches:
        enhanced_results = []
        for text, metadata, score in results:
            enhanced_results.append((text, metadata, score))
        
        # Add direct matches as high-priority results
        for item_code, match_data in exact_matches.items():
            if match_data.get('price') is not None:
                direct_text = f"SKU: {match_data['sku']} | Price: ${match_data['price']}"
                direct_metadata = {
                    'source': match_data['source'],
                    'type': 'direct_lookup',
                    'sku': match_data['sku'],
                    'price': match_data['price']
                }
                enhanced_results.insert(0, (direct_text, direct_metadata, 0.0))  # Highest priority
        
        results = enhanced_results[:top_k]
    
    return results

def search_multiple_queries(questions: List[str], top_k_per_question: int = 5) -> Dict[str, List[Tuple[str, Dict, float]]]:
    """Search for multiple questions and return aggregated results"""
    results = {}
    
    for question in questions:
        # Extract item codes for this specific question
        item_codes = extract_item_codes_from_query(question)
        question_results = search_pinecone_with_fallback(question, item_codes, top_k=top_k_per_question)
        results[question] = question_results
    
    return results

def enhanced_generate_answer(question: str, context: str, item_codes: List[str] = None) -> str:
    """Enhanced answer generation with focus on specific items"""
    client = get_openai_client()
    
    system_prompt = """You are K&B Scout AI, a helpful assistant that answers questions about product pricing and specifications.

Instructions:
1. Focus on providing specific, accurate information for each requested item
2. When asked about multiple items, address each one individually with clear formatting
3. For pricing questions, always provide the exact price if available in the format: **ITEM_CODE**: $XX.XX
4. If an item is not found, clearly state "**ITEM_CODE**: Price not available in the context"
5. Use bold formatting for item names/codes to make them easily identifiable
6. Only use information from the provided context
7. Be precise and avoid generic statements
8. If you find exact matches, prioritize those over partial matches
9. Format your response clearly with each item on a separate line or bullet point"""

    # Enhanced context formatting if we have item codes
    enhanced_context = context
    if item_codes:
        enhanced_context = f"REQUESTED ITEMS: {', '.join(item_codes)}\n\nCONTEXT:\n{context}"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{enhanced_context}\n\nQuestion: {question}"}
    ]
    
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.1,
        max_tokens=1500
    )
    
    return response.choices[0].message.content

def generate_multi_answer(questions: List[str], context_per_question: Dict[str, str]) -> str:
    """Generate comprehensive answer for multiple questions"""
    client = get_openai_client()
    
    system_prompt = """You are K&B Scout AI, a helpful assistant that answers multiple questions based on provided context.

Instructions:
1. Address each question clearly and completely
2. Use only information from the provided context
3. Number your responses (1., 2., etc.) for multiple questions
4. If a question cannot be answered from the context, state this clearly
5. Provide comprehensive but concise answers
6. Cross-reference information between questions when relevant
7. For pricing questions, use bold formatting: **ITEM**: $XX.XX
8. Be specific about item codes and prices when available"""
    
    # Format the questions and context
    formatted_content = "QUESTIONS TO ANSWER:\n"
    for i, question in enumerate(questions, 1):
        formatted_content += f"{i}. {question}\n"
    
    formatted_content += "\nCONTEXT FOR EACH QUESTION:\n"
    for i, (question, context) in enumerate(context_per_question.items(), 1):
        formatted_content += f"\nContext for Question {i}:\n{context}\n"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": formatted_content}
    ]
    
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.1,
        max_tokens=2000
    )
    
    return response.choices[0].message.content

# -----------------------------
# Flask App
# -----------------------------
app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE
app.config['S3_BUCKET'] = os.getenv('S3_BUCKET_NAME')

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    """Upload and process files"""
    if 'files' not in request.files:
        return jsonify({'success': False, 'message': 'No files provided'})
    
    files = request.files.getlist('files')
    processed_files = []
    all_chunks = []
    
    for file in files:
        if not file.filename:
            continue
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Upload to S3 (optional)
        s3_url = upload_to_s3(filepath, app.config['S3_BUCKET'])
        
        # Process file based on type
        ext = filename.lower().split('.')[-1]
        
        if ext == 'pdf':
            units = process_pdf(filepath)
        elif ext == 'csv':
            units = process_csv(filepath)  # Uses enhanced_process_csv
        elif ext in ['xlsx', 'xls']:
            units = process_excel(filepath)
        elif ext == 'txt':
            units = process_text(filepath)
        else:
            os.remove(filepath)
            continue
        
        # Create chunks
        file_chunks = []
        
        for text, metadata in units:
            if not text or len(text.strip()) < 10:
                continue
            
            chunks = chunk_text(text)
            for chunk in chunks:
                if chunk.strip():
                    metadata_copy = metadata.copy()
                    if s3_url:
                        metadata_copy['s3_url'] = s3_url
                    
                    chunk_obj = RAGChunk(
                        id=str(uuid.uuid4()),
                        text=chunk,
                        metadata=metadata_copy
                    )
                    file_chunks.append(chunk_obj)
                    all_chunks.append(chunk_obj)
        
        # Track file
        if file_chunks:
            file_info = {
                'chunks': len(file_chunks),
                'type': ext,
                's3_key': s3_url,
                'uploaded_at': datetime.now().isoformat()
            }
            
            # Save to S3 metadata or local cache
            if app.config['S3_BUCKET']:
                metadata = {filename: file_info}
                save_file_metadata_to_s3(metadata, app.config['S3_BUCKET'])
            else:
                _cache['uploaded_files'][filename] = file_info
            
            processed_files.append(filename)
        
        os.remove(filepath)
    
    # Add to Pinecone
    if all_chunks:
        success = add_to_pinecone(all_chunks)
        if success:
            return jsonify({
                'success': True,
                'message': f'Processed {len(processed_files)} files with {len(all_chunks)} chunks',
                'files': processed_files
            })
    
    return jsonify({'success': False, 'message': 'No valid content found'})

@app.route('/chat', methods=['POST'])
def chat():
    """Enhanced chat endpoint with exact matching priority"""
    data = request.get_json()
    message = data.get('message', '').strip()
    
    if not message:
        return jsonify({'success': False, 'message': 'No message provided'})
    
    # Extract item codes
    item_codes = extract_item_codes_from_query(message)
    print(f"Extracted item codes: {item_codes}")
    
    # Parse multiple questions
    questions = parse_multiple_questions(message)
    print(f"Parsed {len(questions)} questions: {questions}")
    
    if len(questions) == 1:
        # Single question - use enhanced exact matching
        # Do enhanced direct lookup first
        direct_matches = enhanced_direct_csv_lookup(item_codes) if item_codes else {}
        print(f"Direct matches found: {len(direct_matches)}")
        
        # Also do vector search as backup
        vector_results = search_pinecone_with_fallback(message, item_codes, top_k=15)
        
        if not vector_results and not direct_matches:
            return jsonify({
                'success': True,
                'response': "I couldn't find relevant information in the uploaded documents for your question."
            })
        
        # Build context prioritizing direct matches
        context_parts = []
        
        # Add direct match information first
        if direct_matches:
            for item_code, match_data in direct_matches.items():
                context_parts.append(f"EXACT MATCH - SKU: {match_data['sku']} | Price: ${match_data['price']}")
        
        # Add vector search results
        for text, metadata, score in vector_results[:10]:
            if text not in context_parts:  # Avoid duplicates
                context_parts.append(text)
        
        context = "\n\n".join(context_parts)
        
        # Generate answer with exact matching priority
        if item_codes and direct_matches:
            answer = generate_exact_price_answer(message, item_codes, direct_matches, context)
        else:
            answer = enhanced_generate_answer(message, context, item_codes)
        
        # Add sources
        sources = set()
        if direct_matches:
            for match_data in direct_matches.values():
                sources.add(match_data['source'])
        for _, metadata, _ in vector_results[:3]:
            source = metadata.get('source', 'unknown')
            sources.add(source)
        
        if sources:
            answer += f"\n\n📚 Sources: {', '.join(sources)}"
        
        return jsonify({
            'success': True, 
            'response': answer,
            'item_codes_found': item_codes,
            'direct_matches': len(direct_matches),
            'vector_results': len(vector_results)
        })
    
    else:
        # Multiple questions - use enhanced logic
        all_results = search_multiple_queries(questions, top_k_per_question=8)
        
        if not any(all_results.values()):
            return jsonify({
                'success': True,
                'response': "I couldn't find relevant information in the uploaded documents for your questions."
            })
        
        # Prepare context for each question
        context_per_question = {}
        all_sources = set()
        
        for question, results in all_results.items():
            if results:
                # Separate direct matches from semantic matches
                direct_matches = []
                semantic_matches = []
                
                for doc, metadata, score in results:
                    if metadata.get('type') == 'direct_lookup':
                        direct_matches.append(doc)
                    else:
                        semantic_matches.append(doc)
                
                # Combine with priority to direct matches
                context_docs = direct_matches + semantic_matches
                context = "\n\n".join(context_docs[:5])
                context_per_question[question] = context
                
                # Collect sources
                for _, metadata, _ in results[:3]:
                    source = metadata.get('source', 'unknown')
                    all_sources.add(source)
            else:
                context_per_question[question] = "No relevant information found."
        
        # Generate comprehensive answer
        answer = generate_multi_answer(questions, context_per_question)
        
        # Add sources
        if all_sources:
            answer += f"\n\n📚 Sources: {', '.join(sorted(all_sources))}"
        
        return jsonify({
            'success': True, 
            'response': answer,
            'questions_parsed': len(questions),
            'total_item_codes': len(item_codes)
        })

@app.route('/chat-exact', methods=['POST'])
def chat_exact():
    """Chat endpoint with enhanced exact matching"""
    data = request.get_json()
    message = data.get('message', '').strip()
    
    if not message:
        return jsonify({'success': False, 'message': 'No message provided'})
    
    # Extract item codes
    item_codes = extract_item_codes_from_query(message)
    print(f"Extracted item codes: {item_codes}")
    
    if not item_codes:
        return jsonify({'success': False, 'message': 'No item codes found in query'})
    
    # Do enhanced direct lookup first
    direct_matches = enhanced_direct_csv_lookup(item_codes)
    print(f"Direct matches found: {len(direct_matches)}")
    
    # Also do vector search as backup
    vector_results = search_pinecone(message, top_k=20)
    
    # Build context prioritizing direct matches
    context_parts = []
    
    # Add direct match information
    if direct_matches:
        for item_code, match_data in direct_matches.items():
            context_parts.append(f"EXACT MATCH - SKU: {match_data['sku']} | Price: {match_data['price']}")
    
    # Add vector search results
    for text, metadata, score in vector_results[:10]:
        if text not in context_parts:  # Avoid duplicates
            context_parts.append(text)
    
    context = "\n\n".join(context_parts)
    
    # Generate answer with exact matching priority
    answer = generate_exact_price_answer(message, item_codes, direct_matches, context)
    
    # Add debug information
    debug_info = {
        'item_codes_extracted': item_codes,
        'direct_matches_count': len(direct_matches),
        'direct_matches': {k: {'sku': v['sku'], 'price': v['price'], 'score': v['score']} for k, v in direct_matches.items()},
        'vector_results_count': len(vector_results)
    }
    
    return jsonify({
        'success': True, 
        'response': answer,
        'debug': debug_info
    })

@app.route('/test-lookup', methods=['POST'])
def test_lookup():
    """Test endpoint to verify item lookup"""
    test_items = ['BS2460', 'BS3096', 'BSS33 SHELF RAS', 'KNOB 3563MB']
    
    results = enhanced_direct_csv_lookup(test_items)
    
    formatted_results = {}
    for item, data in results.items():
        formatted_results[item] = {
            'found': True,
            'sku': data['sku'],
            'price': data['price'],
            'match_type': data['match_type'],
            'score': data['score']
        }
    
    # Check for missing items
    for item in test_items:
        if item not in formatted_results:
            formatted_results[item] = {
                'found': False,
                'reason': 'Not found in any cached CSV'
            }
    
    return jsonify({
        'test_items': test_items,
        'results': formatted_results,
        'cached_files': list(_cache['csv_lookup_cache'].keys()),
        'total_found': len([r for r in formatted_results.values() if r.get('found')])
    })

@app.route('/status', methods=['GET'])
def status():
    """Get system status"""
    try:
        index = get_or_create_index()
        stats = index.describe_index_stats()
        
        uploaded_files = get_all_uploaded_files()
        
        return jsonify({
            'success': True,
            'connected': True,
            'documents': stats.get('total_vector_count', 0),
            'count': stats.get('total_vector_count', 0),
            'files': len(uploaded_files),
            'uploaded_files': uploaded_files,
            'csv_cache_count': len(_cache['csv_lookup_cache'])
        })
    except Exception as e:
        print(f"Status error: {e}")
        return jsonify({
            'success': False,
            'connected': False,
            'documents': 0,
            'count': 0,
            'files': 0,
            'message': str(e)
        })

@app.route('/files', methods=['GET'])
def list_files():
    """List all uploaded files"""
    try:
        uploaded_files = get_all_uploaded_files()
        
        files_dict = {}
        for filename, info in uploaded_files.items():
            files_dict[filename] = {
                'type': info.get('type', 'unknown'),
                'chunks': info.get('chunks', 0),
                'uploaded_at': info.get('uploaded_at', ''),
                's3_url': info.get('s3_url', ''),
                's3_key': info.get('s3_key', '')
            }
        
        return jsonify({
            'success': True,
            'files': files_dict,
            'total_files': len(files_dict),
            'cached_csvs': list(_cache['csv_lookup_cache'].keys())
        })
    except Exception as e:
        print(f"Error listing files: {e}")
        return jsonify({
            'success': False,
            'files': {},
            'total_files': 0,
            'message': str(e)
        })

@app.route('/clear', methods=['POST'])
def clear():
    """Clear all data"""
    try:
        pc = get_pinecone_client()
        
        # Delete index
        try:
            pc.delete_index(PINECONE_INDEX_NAME)
        except:
            pass
        
        # Clear S3 metadata
        if app.config['S3_BUCKET']:
            s3 = get_s3_client()
            if s3:
                try:
                    import json
                    s3.put_object(
                        Bucket=app.config['S3_BUCKET'],
                        Key='metadata/files_metadata.json',
                        Body=json.dumps({}),
                        ContentType='application/json'
                    )
                except:
                    pass
        
        # Clear caches
        _cache['pinecone_index'] = None
        _cache['embeddings'].clear()
        _cache['uploaded_files'].clear()
        _cache['csv_lookup_cache'].clear()
        
        # Recreate index
        get_or_create_index()
        
        return jsonify({'success': True, 'message': 'All data cleared'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

# -----------------------------
# Additional API Endpoints for Multi-Input Support
# -----------------------------

@app.route('/parse-questions', methods=['POST'])
def parse_questions():
    """API endpoint to test question parsing"""
    data = request.get_json()
    message = data.get('message', '').strip()
    
    if not message:
        return jsonify({'success': False, 'message': 'No message provided'})
    
    questions = parse_multiple_questions(message)
    question_contexts = extract_question_context(questions)
    item_codes = extract_item_codes_from_query(message)
    
    return jsonify({
        'success': True,
        'original_message': message,
        'parsed_questions': questions,
        'question_count': len(questions),
        'extracted_item_codes': item_codes,
        'question_contexts': [
            {'question': q, 'keywords': kw} 
            for q, kw in question_contexts
        ]
    })

@app.route('/direct-lookup', methods=['POST'])
def direct_lookup():
    """API endpoint to test direct CSV lookup"""
    data = request.get_json()
    item_codes = data.get('item_codes', [])
    
    if not item_codes:
        return jsonify({'success': False, 'message': 'No item codes provided'})
    
    results = enhanced_direct_csv_lookup(item_codes)
    
    return jsonify({
        'success': True,
        'item_codes': item_codes,
        'results': results,
        'found_count': len(results),
        'cached_files': list(_cache['csv_lookup_cache'].keys())
    })

@app.route('/search-multiple', methods=['POST'])
def search_multiple():
    """API endpoint to search for multiple questions separately"""
    data = request.get_json()
    message = data.get('message', '').strip()
    
    if not message:
        return jsonify({'success': False, 'message': 'No message provided'})
    
    questions = parse_multiple_questions(message)
    results = search_multiple_queries(questions, top_k_per_question=5)
    
    # Format results for response
    formatted_results = {}
    for question, search_results in results.items():
        formatted_results[question] = [
            {
                'text': text[:200] + '...' if len(text) > 200 else text,
                'metadata': metadata,
                'score': score
            }
            for text, metadata, score in search_results
        ]
    
    return jsonify({
        'success': True,
        'questions': questions,
        'results': formatted_results,
        'total_results': sum(len(r) for r in results.values())
    })

if __name__ == '__main__':
    print("🚀 Starting K&B Scout AI with Enhanced Multi-Input Support")
    
    # Validate setup
    try:
        get_openai_client()
        print("✅ OpenAI ready")
    except Exception as e:
        print(f"❌ OpenAI error: {e}")
    
    try:
        get_or_create_index()
        print("✅ Pinecone ready")
    except Exception as e:
        print(f"❌ Pinecone error: {e}")
    
    if get_s3_client():
        print("✅ S3 ready")
        try:
            bucket = os.getenv('S3_BUCKET_NAME')
            if bucket:
                files = get_file_metadata_from_s3(bucket)
                print(f"📁 Loaded {len(files)} files from S3")
                _cache['uploaded_files'] = files
        except Exception as e:
            print(f"⚠️ Could not load files from S3: {e}")
    else:
        print("ℹ️ S3 not configured (optional)")
    
    print("\n🔥 Enhanced Multi-Input Processing Features:")
    print("  - Automatic question parsing and splitting")
    print("  - Direct CSV lookup for exact item matching")
    print("  - Enhanced item code extraction with regex patterns")
    print("  - Fuzzy matching support (install fuzzywuzzy for better results)")
    print("  - Individual context retrieval per question")
    print("  - Comprehensive multi-question answering")
    print("  - Hybrid search: semantic + exact matching")
    print("  - Priority ranking for direct matches")
    print("  - Enhanced exact matching with scoring system")
    
    if not FUZZY_AVAILABLE:
        print("\n💡 Tip: Install fuzzywuzzy for better item matching:")
        print("   pip install fuzzywuzzy python-Levenshtein")
    
    print("\n🔧 API Endpoints:")
    print("  - /chat (main chat with enhanced matching)")
    print("  - /chat-exact (strict exact matching for pricing)")
    print("  - /test-lookup (test item lookup functionality)")
    print("  - /parse-questions (test question parsing)")
    print("  - /direct-lookup (test direct CSV lookup)")
    
    app.run(debug=False, host='0.0.0.0', port=5000)
