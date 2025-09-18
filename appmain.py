import sys
import importlib
import os
import uuid
import json
import re
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import tempfile
import shutil

try:
    import pysqlite3  # installed via pysqlite3-binary
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    pass

try:
    import sqlite3
    print("SQLite version:", sqlite3.sqlite_version)
except Exception:
    pass

# Flask imports
from flask import Flask, render_template_string, request, jsonify, session
from werkzeug.utils import secure_filename

# Document processing imports
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import pandas as pd
from pypdf import PdfReader

# Vector DB
import chromadb
from chromadb.config import Settings

# Tokenization & chunking
import tiktoken

# OpenAI SDK v1
from openai import OpenAI

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# -----------------------------
# ---------- Utils ------------
# -----------------------------

def get_openai_client() -> OpenAI:
    load_dotenv(override=True)
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("No OpenAI API key found. Please set OPENAI_API_KEY in your .env file.")
    return OpenAI(api_key=key)

def new_uuid() -> str:
    return str(uuid.uuid4())

def make_tokenizer():
    return tiktoken.get_encoding("cl100k_base")

def parse_multi_item_query(query: str) -> List[str]:
    """Extract individual product codes or items from a query."""
    # Pattern to match product codes like WRH3027, RH4224, etc.
    product_pattern = r'\b[A-Z]{2,4}\d{3,5}(?:\s+[A-Z]{1,3})?\b'
    products = re.findall(product_pattern, query, re.IGNORECASE)
    
    if len(products) > 1:
        return [p.strip() for p in products]
    
    # Fallback: split by common separators for multi-item queries
    separators = [',', ';', ' and ', '&', '\n']
    items = [query.strip()]
    
    for sep in separators:
        new_items = []
        for item in items:
            if sep in item:
                parts = [part.strip() for part in item.split(sep)]
                new_items.extend(parts)
            else:
                new_items.append(item)
        items = new_items
    
    # Only return multiple items if we actually split something
    if len(items) > 1:
        return [item for item in items if item and len(item.strip()) > 2]
    
    return [query.strip()]

def chunk_text(
    text: str,
    tokenizer,
    chunk_tokens: int = 800,
    overlap_tokens: int = 150
) -> List[str]:
    if not text or not text.strip():
        return []
    tokens = tokenizer.encode(text)
    if len(tokens) == 0:
        return []
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_tokens, len(tokens))
        chunk = tokenizer.decode(tokens[start:end])
        if chunk.strip():
            chunks.append(chunk.strip())
        if end == len(tokens):
            break
        start = end - overlap_tokens
        if start < 0:
            start = 0
    return chunks

def read_pdf(file_path: str) -> List[Tuple[str, Dict]]:
    """Extract text from a PDF with OCR fallback."""
    reader = PdfReader(file_path)
    pages = []
    filename = os.path.basename(file_path)
    
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((text, {"source": filename, "type": "pdf", "page": i+1}))
        else:
            # OCR fallback
            try:
                images = convert_from_path(file_path, first_page=i+1, last_page=i+1, dpi=300)
                ocr_text = ""
                for img in images:
                    ocr_text += pytesseract.image_to_string(img)
                pages.append((ocr_text, {"source": filename, "type": "pdf", "page": i+1}))
            except:
                pages.append(("", {"source": filename, "type": "pdf", "page": i+1}))
    return pages

def read_csv(file_path: str) -> List[Tuple[str, Dict]]:
    """Returns (row_text, metadata) per row."""
    encodings_to_try = ['utf-8', 'cp1252', 'latin-1', 'iso-8859-1']
    
    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(file_path, encoding=encoding, low_memory=False)
            filename = os.path.basename(file_path)
            rows = []
            
            # Also add header information as a searchable row
            header_info = f"CSV Headers: {' | '.join(df.columns.tolist())}"
            rows.append((header_info, {"source": filename, "type": "csv", "row": 0}))
            
            for idx, row in df.iterrows():
                row_values = []
                for col in df.columns:
                    val = row[col]
                    if pd.notna(val) and str(val).strip():
                        # Clean the value and make it more searchable
                        clean_val = str(val).strip()
                        row_values.append(f"{col}: {clean_val}")
                
                if row_values:
                    row_text = " | ".join(row_values)
                    # Add additional searchable content
                    searchable_text = f"{row_text} | Row {idx + 1}"
                    rows.append((searchable_text, {"source": filename, "type": "csv", "row": int(idx) + 1}))
            
            print(f"Successfully read CSV {filename} with encoding {encoding}, found {len(rows)} rows")
            return rows
            
        except (UnicodeDecodeError, pd.errors.EmptyDataError) as e:
            print(f"Failed to read CSV with encoding {encoding}: {e}")
            continue
        except Exception as e:
            print(f"Error reading CSV {file_path} with encoding {encoding}: {e}")
            continue
    
    print(f"Failed to read CSV {file_path} with any encoding")
    return []

def read_xlsx(file_path: str) -> List[Tuple[str, Dict]]:
    encodings_to_try = ['utf-8', 'cp1252', 'latin-1', 'iso-8859-1']
    
    for encoding in encodings_to_try:
        try:
            df = pd.read_excel(file_path)
            filename = os.path.basename(file_path)
            rows = []
            
            # Also add header information as a searchable row
            header_info = f"Excel Headers: {' | '.join(df.columns.tolist())}"
            rows.append((header_info, {"source": filename, "type": "xlsx", "row": 0}))
            
            for idx, row in df.iterrows():
                row_values = []
                for col in df.columns:
                    val = row[col]
                    if pd.notna(val) and str(val).strip():
                        # Clean the value and make it more searchable
                        clean_val = str(val).strip()
                        row_values.append(f"{col}: {clean_val}")
                
                if row_values:
                    row_text = " | ".join(row_values)
                    # Add additional searchable content
                    searchable_text = f"{row_text} | Row {idx + 1}"
                    rows.append((searchable_text, {"source": filename, "type": "xlsx", "row": int(idx) + 1}))
            
            print(f"Successfully read Excel {filename}, found {len(rows)} rows")
            return rows
            
        except Exception as e:
            print(f"Error reading Excel {file_path}: {e}")
            continue
    
    print(f"Failed to read Excel {file_path}")
    return []

def read_text_file(file_path: str) -> List[Tuple[str, Dict]]:
    """Read text files (txt, doc, docx)."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        filename = os.path.basename(file_path)
        return [(content, {"source": filename, "type": "txt", "page": 1})]
    except Exception as e:
        print(f"Error reading text file {file_path}: {e}")
        return []

def safe_clean(s: str) -> str:
    if not s:
        return ""
    cleaned = s.replace("\x00", " ").replace("\r", " ").replace("\n", " ")
    cleaned = " ".join(cleaned.split())
    return cleaned.strip()

@dataclass
class RAGChunk:
    id: str
    text: str
    metadata: Dict

# -----------------------------
# ------ Vector Store ---------
# -----------------------------

def get_chroma_client():
    """Creates a persistent ChromaDB client."""
    persist_dir = "./chromadb_storage"
    try:
        os.makedirs(persist_dir, exist_ok=True)
        client = chromadb.PersistentClient(path=persist_dir)
        return client
    except Exception as e:
        print(f"Could not create persistent client: {e}")
        return None

def get_or_create_collection(chroma_client, collection_name: str = "kb_scout_documents"):
    """Get existing collection or create new one."""
    try:
        collection = chroma_client.get_collection(name=collection_name)
        return collection
    except:
        try:
            return chroma_client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            print(f"Error creating collection: {e}")
            return None

def embed_texts(client: OpenAI, texts: List[str], model: str = "text-embedding-3-small", batch_size: int = 100) -> List[List[float]]:
    """Batches embeddings to avoid hitting request-size limits."""
    if not texts:
        return []
    
    all_embeddings: List[List[float]] = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            resp = client.embeddings.create(input=batch, model=model)
            batch_embeddings = [d.embedding for d in resp.data]
            all_embeddings.extend(batch_embeddings)
        except Exception as e:
            print(f"Error creating embeddings: {e}")
            return []
    
    return all_embeddings

def add_chunks_to_collection(collection, client: OpenAI, rag_chunks: List[RAGChunk]):
    """Add chunks to persistent collection."""
    if not rag_chunks or not collection:
        return False
    
    valid_chunks = [c for c in rag_chunks if c.text and c.text.strip()]
    if not valid_chunks:
        return False
    
    documents = [c.text for c in valid_chunks]
    metadatas = [c.metadata for c in valid_chunks]
    ids = [c.id for c in valid_chunks]

    embeddings = embed_texts(client, documents)
    
    if not embeddings or len(embeddings) != len(documents):
        return False

    try:
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings
        )
        return True
    except Exception as e:
        print(f"Error adding to collection: {e}")
        return False

def retrieve(collection, client: OpenAI, query: str, top_k: int = 6) -> List[Tuple[str, Dict, float]]:
    if not collection:
        return []
    
    count = collection.count()
    if count == 0:
        return []
    
    try:
        q_emb = embed_texts(client, [query])[0]
        res = collection.query(
            query_embeddings=[q_emb],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"]
        )
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        
        scored = list(zip(docs, metas, dists))
        scored.sort(key=lambda x: x[2])
        return scored
    except Exception as e:
        print(f"Error during retrieval: {e}")
        return []

def retrieve_multi_query(collection, client: OpenAI, queries: List[str], top_k_per_query: int = 4) -> List[Tuple[str, Dict, float]]:
    """Retrieve documents for multiple queries and combine results."""
    if not collection or not queries:
        return []
    
    all_results = []
    seen_docs = set()
    
    # Search for each individual query
    for query in queries:
        if query.strip():
            results = retrieve(collection, client, query.strip(), top_k_per_query)
            for doc, meta, dist in results:
                # Create a unique key for deduplication
                doc_key = f"{doc[:100]}_{meta.get('source', '')}_{meta.get('page', '')}_{meta.get('row', '')}"
                if doc_key not in seen_docs:
                    seen_docs.add(doc_key)
                    all_results.append((doc, meta, dist))
    
    # Sort by relevance (distance) and limit total results
    all_results.sort(key=lambda x: x[2])
    return all_results[:15]  # Increased limit for multi-queries

def format_context(snippets: List[Tuple[str, Dict, float]]) -> str:
    """Format context without numbered brackets - just clean context"""
    blocks = []
    for doc, meta, dist in snippets:
        src = meta.get("source", "unknown")
        if meta.get("type") == "pdf":
            loc = f"page {meta.get('page', 'unknown')}"
        elif meta.get("type") in ["csv", "xlsx"]:
            loc = f"row {meta.get('row', 'unknown')}"
        else:
            loc = f"section {meta.get('page', 'unknown')}"
        blocks.append(f"From {src} ({meta.get('type','')}, {loc}):\n{doc}")
    return "\n\n".join(blocks)

def format_sources(snippets: List[Tuple[str, Dict, float]]) -> str:
    """Format source information separately with all relevant pages/rows"""
    source_info = {}
    
    for doc, meta, dist in snippets:
        src = meta.get("source", "unknown")
        file_type = meta.get("type", "")
        
        if src not in source_info:
            source_info[src] = {
                'type': file_type,
                'locations': set()
            }
        
        if meta.get("type") == "pdf":
            loc = meta.get('page', 'unknown')
            source_info[src]['locations'].add(f"page {loc}")
        elif meta.get("type") in ["csv", "xlsx"]:
            loc = meta.get('row', 'unknown')
            source_info[src]['locations'].add(f"row {loc}")
        else:
            loc = meta.get('page', 'unknown')
            source_info[src]['locations'].add(f"section {loc}")
    
    sources = []
    for src, info in source_info.items():
        locations = sorted(list(info['locations']))
        if len(locations) == 1:
            loc_str = locations[0]
        elif len(locations) <= 3:
            loc_str = ", ".join(locations)
        else:
            loc_str = f"{locations[0]}, {locations[1]}, ... +{len(locations)-2} more"
        
        sources.append(f"• {src} ({info['type']}, {loc_str})")
    
    if sources:
        return "\n\n**Sources:**\n" + "\n".join(sources)
    return ""

def get_uploaded_files_from_collection(collection):
    """Get list of unique files that have been uploaded to the collection."""
    if not collection:
        return []
    
    try:
        all_data = collection.get(include=["metadatas"])
        metadatas = all_data.get("metadatas", [])
        
        files = set()
        for meta in metadatas:
            if "source" in meta and "type" in meta:
                files.add((meta["source"], meta["type"]))
        
        return list(files)
    except:
        return []

# Updated system prompt for better multi-item handling
SYSTEM_PROMPT = """You are K&B Scout AI, a helpful enterprise document assistant.
Follow these rules:
- Use only the information provided in the context to answer questions.
- If the answer cannot be found in the context, clearly state that you do not have enough information.
- Be concise, accurate, and professional in your responses.
- When asked about multiple items (like multiple product codes), provide information for each item separately if available.
- If some items have information available and others don't, clearly indicate which ones you found information for.
- Organize multi-item responses in a clear, structured format (use bullet points or numbered lists when appropriate).
- Be friendly and helpful while staying focused on the available information.
- If multiple sources contain similar information, synthesize it naturally.
- For product queries, include relevant details like prices, specifications, availability, etc.
"""

def answer_with_rag(client: OpenAI, question: str, context_text: str, source_info: str = ""):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Context information:\n{context_text}\n\nQuestion: {question}\nAnswer:"
        }
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.0
        )
        answer = response.choices[0].message.content
        
        # Add source information at the end if provided
        if source_info:
            answer += source_info
            
        return answer
    except Exception as e:
        print(f"Error generating response: {e}")
        return "I apologize, but I encountered an error while processing your question."

# -----------------------------
# --------- Flask App ---------
# -----------------------------

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize global objects
try:
    client = get_openai_client()
    ch_client = get_chroma_client()
    collection = get_or_create_collection(ch_client, "kb_scout_documents") if ch_client else None
except Exception as e:
    print(f"Error initializing: {e}")
    client = None
    ch_client = None
    collection = None

# HTML Template
HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>K&B Scout AI - Enterprise Document Assistant</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.1);
            overflow: hidden;
            min-height: calc(100vh - 40px);
            display: flex;
            flex-direction: column;
        }

        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            display: flex;
            align-items: center;
            gap: 20px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        }

        .header-icon {
            font-size: 40px;
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.05); }
        }

        .header-text h1 {
            font-size: 32px;
            font-weight: 700;
            margin-bottom: 5px;
        }

        .header-text p {
            font-size: 16px;
            opacity: 0.9;
            font-weight: 300;
        }

        .main-content {
            display: flex;
            flex: 1;
        }

        .upload-section {
            flex: 1;
            background: #f8f9fa;
            padding: 40px;
            border-right: 2px solid #e9ecef;
            display: flex;
            flex-direction: column;
        }

        .chat-section {
            flex: 1.2;
            padding: 40px;
            display: flex;
            flex-direction: column;
        }

        .section-title {
            font-size: 24px;
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .upload-area {
            border: 3px dashed #dee2e6;
            border-radius: 15px;
            padding: 50px 30px;
            text-align: center;
            background: white;
            transition: all 0.3s ease;
            cursor: pointer;
            margin-bottom: 30px;
        }

        .upload-area:hover {
            border-color: #667eea;
            background: linear-gradient(135deg, #f8f9ff 0%, #e8edff 100%);
            transform: translateY(-2px);
        }

        .upload-area.dragover {
            border-color: #667eea;
            background: linear-gradient(135deg, #f0f4ff 0%, #e0e7ff 100%);
            transform: scale(1.02);
        }

        .upload-icon {
            font-size: 60px;
            color: #667eea;
            margin-bottom: 20px;
        }

        .upload-text {
            font-size: 18px;
            color: #495057;
            margin-bottom: 10px;
            font-weight: 500;
        }

        .upload-subtitle {
            color: #6c757d;
            font-size: 14px;
        }

        .file-input {
            display: none;
        }

        .selected-files {
            margin-bottom: 20px;
        }

        .file-item {
            background: white;
            border: 1px solid #e9ecef;
            border-radius: 10px;
            padding: 15px 20px;
            margin: 10px 0;
            display: flex;
            align-items: center;
            gap: 15px;
            transition: all 0.3s ease;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        }

        .file-item:hover {
            transform: translateX(5px);
            box-shadow: 0 5px 20px rgba(0, 0, 0, 0.1);
        }

        .file-icon {
            font-size: 20px;
            width: 40px;
            height: 40px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
        }

        .file-icon.pdf { background: linear-gradient(135deg, #ff6b6b, #ee5a52); }
        .file-icon.csv, .file-icon.xlsx { background: linear-gradient(135deg, #51cf66, #40c057); }
        .file-icon.txt, .file-icon.doc { background: linear-gradient(135deg, #339af0, #228be6); }

        .file-info {
            flex: 1;
        }

        .file-name {
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 2px;
        }

        .file-size {
            font-size: 12px;
            color: #6c757d;
        }

        .remove-file {
            color: #dc3545;
            cursor: pointer;
            padding: 5px;
            border-radius: 50%;
            transition: all 0.3s ease;
        }

        .remove-file:hover {
            background: #f8d7da;
        }

        .process-btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 12px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            margin-bottom: 30px;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }

        .process-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        }

        .process-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }

        .uploaded-files {
            background: white;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        }

        .status-indicator {
            display: inline-flex;
            align-items: center;
            gap: 10px;
            padding: 10px 20px;
            border-radius: 25px;
            font-size: 14px;
            font-weight: 600;
            margin-bottom: 30px;
        }

        .status-ready {
            background: linear-gradient(135deg, #d4edda, #c3e6cb);
            color: #155724;
            border: 1px solid #c3e6cb;
        }

        .status-waiting {
            background: linear-gradient(135deg, #fff3cd, #ffeaa7);
            color: #856404;
            border: 1px solid #ffeaa7;
        }

        .chat-container {
            flex: 1;
            background: #f8f9fa;
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
            overflow-y: auto;
            max-height: 400px;
            box-shadow: inset 0 2px 10px rgba(0, 0, 0, 0.05);
        }

        .message {
            margin-bottom: 20px;
            display: flex;
            gap: 15px;
            animation: fadeIn 0.5s ease;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .message.user {
            flex-direction: row-reverse;
        }

        .message-avatar {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
            color: white;
            flex-shrink: 0;
        }

        .message.assistant .message-avatar {
            background: linear-gradient(135deg, #667eea, #764ba2);
        }

        .message.user .message-avatar {
            background: linear-gradient(135deg, #51cf66, #40c057);
        }

        .message-content {
            background: white;
            padding: 15px 20px;
            border-radius: 18px;
            max-width: 70%;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            line-height: 1.6;
        }

        .message.user .message-content {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
        }

        .chat-input-container {
            display: flex;
            gap: 15px;
            margin-bottom: 20px;
        }

        .chat-input {
            flex: 1;
            padding: 15px 20px;
            border: 2px solid #e9ecef;
            border-radius: 25px;
            font-size: 16px;
            outline: none;
            transition: all 0.3s ease;
        }

        .chat-input:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }

        .send-btn {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border: none;
            width: 50px;
            height: 50px;
            border-radius: 50%;
            font-size: 18px;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }

        .send-btn:hover {
            transform: scale(1.1);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        }

        .chat-controls {
            display: flex;
            gap: 15px;
        }

        .control-btn {
            flex: 1;
            padding: 12px 20px;
            border: 2px solid #e9ecef;
            background: white;
            border-radius: 10px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .control-btn.clear {
            border-color: #ffc107;
            color: #856404;
        }

        .control-btn.clear:hover {
            background: #fff3cd;
        }

        .control-btn.delete {
            border-color: #dc3545;
            color: #721c24;
        }

        .control-btn.delete:hover {
            background: #f8d7da;
        }

        .spinner {
            width: 20px;
            height: 20px;
            border: 2px solid #f3f3f3;
            border-top: 2px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .loading {
            display: flex;
            align-items: center;
            gap: 10px;
            color: #667eea;
            font-weight: 500;
        }

        @media (max-width: 768px) {
            .main-content {
                flex-direction: column;
            }
            
            .upload-section {
                border-right: none;
                border-bottom: 2px solid #e9ecef;
            }
            
            .message-content {
                max-width: 85%;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="header-icon">🤖</div>
            <div class="header-text">
                <h1>K&B Scout AI</h1>
                <p>Enterprise Document Assistant - Enhanced Multi-Query Support</p>
            </div>
        </div>

        <div class="main-content">
            <!-- Upload Section -->
            <div class="upload-section">
                <h2 class="section-title">
                    <i class="fas fa-cloud-upload-alt"></i>
                    Upload Documents
                </h2>

                <div class="upload-area" id="uploadArea">
                    <div class="upload-icon">
                        <i class="fas fa-file-upload"></i>
                    </div>
                    <div class="upload-text">Drag & drop files here</div>
                    <div class="upload-subtitle">or click to browse</div>
                    <input type="file" id="fileInput" class="file-input" multiple 
                           accept=".pdf,.csv,.xlsx,.xls,.txt,.doc,.docx">
                </div>

                <div class="selected-files" id="selectedFiles" style="display: none;">
                    <h3 style="margin-bottom: 15px; color: #2c3e50;">Selected Files</h3>
                    <div id="fileList"></div>
                </div>

                <button class="process-btn" id="processBtn" style="display: none;">
                    <i class="fas fa-rocket"></i> Process Files
                </button>

                <div class="uploaded-files">
                    <h3 style="margin-bottom: 15px; color: #2c3e50;">
                        <i class="fas fa-database"></i> Database Files
                    </h3>
                    <div id="uploadedFilesList">
                        <div style="color: #6c757d; text-align: center; padding: 20px;">
                            Loading...
                        </div>
                    </div>
                </div>
            </div>

            <!-- Chat Section -->
            <div class="chat-section">
                <h2 class="section-title">
                    <i class="fas fa-comments"></i>
                    Chat with K&B Scout AI
                </h2>

                <div class="status-indicator status-waiting" id="statusIndicator">
                    <i class="fas fa-circle"></i>
                    Loading...
                </div>

                <div class="chat-container" id="chatContainer">
                    <div class="message assistant">
                        <div class="message-avatar">🤖</div>
                        <div class="message-content">
                            <strong>Hello! I'm K&B Scout AI</strong>, your enhanced enterprise document assistant.<br><br>
                           I can help you find information from your uploaded files. What would you like to know?<br><br>
                          
                        </div>
                    </div>
                </div>

                <div class="chat-input-container">
                    <input type="text" class="chat-input" id="chatInput" 
                           placeholder="Ask about single or multiple items from your documents..." 
                           disabled>
                    <button class="send-btn" id="sendBtn" disabled>
                        <i class="fas fa-paper-plane"></i>
                    </button>
                </div>

                <div class="chat-controls">
                    <button class="control-btn clear" id="clearChatBtn">
                        <i class="fas fa-refresh"></i> Clear Chat
                    </button>
                    <button class="control-btn delete" id="clearDataBtn">
                        <i class="fas fa-trash"></i> Clear All Data
                    </button>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Global state
        let selectedFiles = [];
        let chatHistory = [];

        // DOM elements
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const selectedFilesDiv = document.getElementById('selectedFiles');
        const fileList = document.getElementById('fileList');
        const processBtn = document.getElementById('processBtn');
        const uploadedFilesList = document.getElementById('uploadedFilesList');
        const statusIndicator = document.getElementById('statusIndicator');
        const chatContainer = document.getElementById('chatContainer');
        const chatInput = document.getElementById('chatInput');
        const sendBtn = document.getElementById('sendBtn');
        const clearChatBtn = document.getElementById('clearChatBtn');
        const clearDataBtn = document.getElementById('clearDataBtn');

        // File type icons
        const fileIcons = {
            pdf: 'fas fa-file-pdf',
            csv: 'fas fa-file-csv',
            xlsx: 'fas fa-file-excel',
            xls: 'fas fa-file-excel',
            txt: 'fas fa-file-alt',
            doc: 'fas fa-file-word',
            docx: 'fas fa-file-word'
        };

        // Initialize
        function init() {
            setupEventListeners();
            loadUploadedFiles();
            updateStatus();
        }

        function setupEventListeners() {
            // Upload area events
            uploadArea.addEventListener('click', () => fileInput.click());
            uploadArea.addEventListener('dragover', handleDragOver);
            uploadArea.addEventListener('dragleave', handleDragLeave);
            uploadArea.addEventListener('drop', handleDrop);

            // File input
            fileInput.addEventListener('change', handleFileSelect);

            // Process button
            processBtn.addEventListener('click', processFiles);

            // Chat input
            chatInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') sendMessage();
            });
            sendBtn.addEventListener('click', sendMessage);

            // Control buttons
            clearChatBtn.addEventListener('click', clearChat);
            clearDataBtn.addEventListener('click', clearAllData);
        }

        function handleDragOver(e) {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        }

        function handleDragLeave(e) {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
        }

        function handleDrop(e) {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const files = Array.from(e.dataTransfer.files);
            addFilesToSelection(files);
        }

        function handleFileSelect(e) {
            const files = Array.from(e.target.files);
            addFilesToSelection(files);
        }

        function addFilesToSelection(files) {
            files.forEach(file => {
                if (!selectedFiles.find(f => f.name === file.name)) {
                    selectedFiles.push(file);
                }
            });
            updateSelectedFiles();
        }

        function updateSelectedFiles() {
            if (selectedFiles.length === 0) {
                selectedFilesDiv.style.display = 'none';
                processBtn.style.display = 'none';
                return;
            }

            selectedFilesDiv.style.display = 'block';
            processBtn.style.display = 'block';

            fileList.innerHTML = selectedFiles.map((file, index) => {
                const ext = file.name.split('.').pop().toLowerCase();
                const icon = fileIcons[ext] || 'fas fa-file';
                const size = formatFileSize(file.size);

                return `
                    <div class="file-item">
                        <div class="file-icon ${ext}">
                            <i class="${icon}"></i>
                        </div>
                        <div class="file-info">
                            <div class="file-name">${file.name}</div>
                            <div class="file-size">${size}</div>
                        </div>
                        <div class="remove-file" onclick="removeFile(${index})">
                            <i class="fas fa-times"></i>
                        </div>
                    </div>
                `;
            }).join('');
        }

        function removeFile(index) {
            selectedFiles.splice(index, 1);
            updateSelectedFiles();
        }

        function formatFileSize(bytes) {
            if (bytes === 0) return '0 Bytes';
            const k = 1024;
            const sizes = ['Bytes', 'KB', 'MB', 'GB'];
            const i = Math.floor(Math.log(bytes) / Math.log(k));
            return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
        }

        async function processFiles() {
            if (selectedFiles.length === 0) return;

            processBtn.disabled = true;
            processBtn.innerHTML = '<div class="spinner"></div> Processing...';

            const formData = new FormData();
            selectedFiles.forEach(file => {
                formData.append('files', file);
            });

            try {
                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });

                const result = await response.json();

                if (result.success) {
                    // Clear selection
                    selectedFiles = [];
                    fileInput.value = '';
                    updateSelectedFiles();
                    loadUploadedFiles();
                    updateStatus();

                    addMessage('assistant', `✅ ${result.message}`);
                } else {
                    addMessage('assistant', `❌ Error: ${result.message}`);
                }
            } catch (error) {
                console.error('Upload error:', error);
                addMessage('assistant', '❌ Failed to upload files. Please try again.');
            }

            processBtn.disabled = false;
            processBtn.innerHTML = '<i class="fas fa-rocket"></i> Process Files';
        }

        async function loadUploadedFiles() {
            try {
                const response = await fetch('/files');
                const files = await response.json();

                if (files.length === 0) {
                    uploadedFilesList.innerHTML = `
                        <div style="color: #6c757d; text-align: center; padding: 20px;">
                            No files uploaded yet
                        </div>
                    `;
                    return;
                }

                uploadedFilesList.innerHTML = files.map(file => {
                    const icon = fileIcons[file.type] || 'fas fa-file';

                    return `
                        <div class="file-item">
                            <div class="file-icon ${file.type}">
                                <i class="${icon}"></i>
                            </div>
                            <div class="file-info">
                                <div class="file-name">${file.name}</div>
                                <div class="file-size">${file.type.toUpperCase()}</div>
                            </div>
                        </div>
                    `;
                }).join('');
            } catch (error) {
                console.error('Error loading files:', error);
                uploadedFilesList.innerHTML = `
                    <div style="color: #dc3545; text-align: center; padding: 20px;">
                        Error loading files
                    </div>
                `;
            }
        }

        async function updateStatus() {
            try {
                const response = await fetch('/status');
                const status = await response.json();

                if (status.count > 0) {
                    statusIndicator.className = 'status-indicator status-ready';
                    statusIndicator.innerHTML = `
                        <i class="fas fa-check-circle"></i>
                        Ready • ${status.count} documents indexed
                    `;
                    chatInput.disabled = false;
                    sendBtn.disabled = false;
                } else {
                    statusIndicator.className = 'status-indicator status-waiting';
                    statusIndicator.innerHTML = `
                        <i class="fas fa-clock"></i>
                        Upload files to get started
                    `;
                    chatInput.disabled = true;
                    sendBtn.disabled = true;
                }
            } catch (error) {
                console.error('Error updating status:', error);
                statusIndicator.className = 'status-indicator status-waiting';
                statusIndicator.innerHTML = `
                    <i class="fas fa-exclamation-triangle"></i>
                    Error connecting to server
                `;
            }
        }

        async function sendMessage() {
            const message = chatInput.value.trim();
            if (!message) return;

            addMessage('user', message);
            chatInput.value = '';

            // Show typing indicator
            const typingDiv = document.createElement('div');
            typingDiv.className = 'message assistant';
            typingDiv.innerHTML = `
                <div class="message-avatar">🤖</div>
                <div class="message-content">
                    <div class="loading">
                        <div class="spinner"></div>
                        Analyzing your query for multiple items...
                    </div>
                </div>
            `;
            chatContainer.appendChild(typingDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;

            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ message: message })
                });

                const result = await response.json();

                // Remove typing indicator
                chatContainer.removeChild(typingDiv);

                if (result.success) {
                    addMessage('assistant', result.response);
                } else {
                    addMessage('assistant', `❌ Error: ${result.message}`);
                }
            } catch (error) {
                console.error('Chat error:', error);
                // Remove typing indicator
                chatContainer.removeChild(typingDiv);
                addMessage('assistant', '❌ Failed to get response. Please try again.');
            }
        }

        function addMessage(role, content) {
            const message = { role, content, timestamp: new Date() };
            chatHistory.push(message);

            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${role}`;
            
            const avatar = role === 'assistant' ? '🤖' : '👤';
            
            messageDiv.innerHTML = `
                <div class="message-avatar">${avatar}</div>
                <div class="message-content">${content}</div>
            `;

            chatContainer.appendChild(messageDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }

        function clearChat() {
            chatHistory = [];
            chatContainer.innerHTML = `
                <div class="message assistant">
                    <div class="message-avatar">🤖</div>
                    <div class="message-content">
                        <strong>Hello! I'm K&B Scout AI</strong>, your enhanced enterprise document assistant.<br><br>
                      I can help you find information from your uploaded files. What would you like to know?<br><br>
                        <em>Example: "What is the price of WRH3027 SV, WRH3624 DM, RH4224 RP?"</em>
                    </div>
                </div>
            `;
        }

        async function clearAllData() {
            if (confirm('Are you sure you want to clear all data? This will remove all uploaded files and chat history.')) {
                try {
                    const response = await fetch('/clear', {
                        method: 'POST'
                    });

                    const result = await response.json();

                    if (result.success) {
                        selectedFiles = [];
                        chatHistory = [];
                        fileInput.value = '';
                        
                        updateSelectedFiles();
                        loadUploadedFiles();
                        updateStatus();
                        clearChat();
                        
                        addMessage('assistant', '🗑️ All data has been cleared successfully!');
                    } else {
                        addMessage('assistant', `❌ Error clearing data: ${result.message}`);
                    }
                } catch (error) {
                    console.error('Clear data error:', error);
                    addMessage('assistant', '❌ Failed to clear data. Please try again.');
                }
            }
        }

        // Initialize the application
        init();
    </script>
</body>
</html>'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/upload', methods=['POST'])
def upload_files():
    if not client or not collection:
        return jsonify({'success': False, 'message': 'Server not properly initialized'})

    if 'files' not in request.files:
        return jsonify({'success': False, 'message': 'No files provided'})

    files = request.files.getlist('files')
    if not files:
        return jsonify({'success': False, 'message': 'No files selected'})

    tokenizer = make_tokenizer()
    rag_chunks: List[RAGChunk] = []
    processed_count = 0

    for file in files:
        if file.filename == '':
            continue

        try:
            # Check if file already exists
            existing_files = get_uploaded_files_from_collection(collection)
            if any(existing_file[0] == file.filename for existing_file in existing_files):
                continue

            # Save file temporarily
            filename = secure_filename(file.filename)
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(temp_path)

            # Process based on file type
            ext = filename.lower().split('.')[-1]
            
            if ext == 'pdf':
                units = read_pdf(temp_path)
            elif ext == 'csv':
                units = read_csv(temp_path)
            elif ext in ['xlsx', 'xls']:
                units = read_xlsx(temp_path)
            elif ext in ['txt', 'doc', 'docx']:
                units = read_text_file(temp_path)
            else:
                os.remove(temp_path)
                continue

            # Process units into chunks
            for unit_text, meta in units:
                unit_text = safe_clean(unit_text)
                if not unit_text:
                    continue
                
                chunks = chunk_text(unit_text, tokenizer)
                
                for chunk_idx, chunk in enumerate(chunks):
                    if chunk.strip():
                        chunk_meta = meta.copy()
                        chunk_meta["chunk_id"] = chunk_idx + 1
                        rag_chunks.append(RAGChunk(id=new_uuid(), text=chunk, metadata=chunk_meta))

            # Clean up temp file
            os.remove(temp_path)
            processed_count += 1

        except Exception as e:
            print(f"Error processing file {file.filename}: {e}")
            continue

    # Add chunks to collection
    if rag_chunks:
        success = add_chunks_to_collection(collection, client, rag_chunks)
        if success:
            return jsonify({
                'success': True, 
                'message': f'Successfully processed {processed_count} files with {len(rag_chunks)} chunks'
            })
        else:
            return jsonify({'success': False, 'message': 'Failed to add documents to database'})
    else:
        return jsonify({'success': False, 'message': 'No valid content found in uploaded files'})

@app.route('/files')
def get_files():
    if not collection:
        return jsonify([])
    
    try:
        uploaded_files = get_uploaded_files_from_collection(collection)
        return jsonify([{'name': name, 'type': file_type} for name, file_type in uploaded_files])
    except Exception as e:
        print(f"Error getting files: {e}")
        return jsonify([])

@app.route('/status')
def get_status():
    if not collection:
        return jsonify({'count': 0})
    
    try:
        count = collection.count()
        return jsonify({'count': count})
    except Exception as e:
        print(f"Error getting status: {e}")
        return jsonify({'count': 0})

@app.route('/chat', methods=['POST'])
def chat():
    if not client or not collection:
        return jsonify({'success': False, 'message': 'Server not properly initialized'})

    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'success': False, 'message': 'No message provided'})

    message = data['message'].strip()
    if not message:
        return jsonify({'success': False, 'message': 'Empty message'})

    try:
        doc_count = collection.count()
        if doc_count == 0:
            return jsonify({
                'success': True, 
                'response': "I'd be happy to help, but I don't have any documents to search through yet. Please upload some files first!"
            })

        # Parse query for multiple items
        individual_queries = parse_multi_item_query(message)
        
        print(f"Original query: {message}")
        print(f"Parsed queries: {individual_queries}")
        
        if len(individual_queries) > 1:
            # Multi-query retrieval
            print(f"Using multi-query retrieval for {len(individual_queries)} items")
            retrieved = retrieve_multi_query(collection, client, individual_queries)
            
            # Also include the original query for context
            original_results = retrieve(collection, client, message, 4)
            
            # Combine and deduplicate
            all_results = list(retrieved) + list(original_results)
            seen_docs = set()
            unique_results = []
            
            for doc, meta, dist in all_results:
                doc_key = f"{doc[:100]}_{meta.get('source', '')}_{meta.get('page', '')}_{meta.get('row', '')}"
                if doc_key not in seen_docs:
                    seen_docs.add(doc_key)
                    unique_results.append((doc, meta, dist))
            
            # Sort by distance and limit
            unique_results.sort(key=lambda x: x[2])
            retrieved = unique_results[:12]  # Increased for multi-queries
            
        else:
            # Single query retrieval
            print("Using single query retrieval")
            retrieved = retrieve(collection, client, message, 8)
        
        print(f"Retrieved {len(retrieved)} documents")
        
        if not retrieved:
            return jsonify({
                'success': True, 
                'response': "I couldn't find any relevant information in your uploaded documents for this question."
            })

        # Generate response
        context_text = format_context(retrieved)
        source_info = format_sources(retrieved)
        response = answer_with_rag(client, message, context_text, source_info)
        
        return jsonify({'success': True, 'response': response})

    except Exception as e:
        print(f"Error in chat: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Error processing question: {str(e)}'})

@app.route('/clear', methods=['POST'])
def clear_data():
    global collection, ch_client
    
    if not ch_client:
        return jsonify({'success': False, 'message': 'Database not initialized'})

    try:
        # Delete and recreate collection
        ch_client.delete_collection(name="kb_scout_documents")
        collection = get_or_create_collection(ch_client, "kb_scout_documents")
        
        # Clear upload folder
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")

        return jsonify({'success': True, 'message': 'All data cleared successfully'})

    except Exception as e:
        print(f"Error clearing data: {e}")
        return jsonify({'success': False, 'message': f'Error clearing data: {str(e)}'})

@app.route('/debug/<query>')
def debug_search(query):
    """Debug endpoint to see what documents are found for a specific query"""
    if not client or not collection:
        return jsonify({'error': 'Server not properly initialized'})
    
    try:
        # Get all documents that contain the query term
        all_data = collection.get(include=["documents", "metadatas"])
        documents = all_data.get("documents", [])
        metadatas = all_data.get("metadatas", [])
        
        matching_docs = []
        for i, doc in enumerate(documents):
            if query.upper() in doc.upper():
                matching_docs.append({
                    'content': doc[:500],  # First 500 chars
                    'metadata': metadatas[i],
                    'full_content': doc
                })
        
        return jsonify({
            'query': query,
            'total_documents': len(documents),
            'matching_documents': len(matching_docs),
            'matches': matching_docs
        })
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    print("Starting Enhanced K&B Scout AI Enterprise Assistant...")
    print("Enhanced features:")
    print("- Multi-item query support")
    print("- Improved product code recognition")
    print("- Better context retrieval for complex queries")
    print("Make sure you have set your OPENAI_API_KEY in your .env file")
    
    if not client:
        print("ERROR: OpenAI client not initialized. Check your API key.")
    if not collection:
        print("ERROR: ChromaDB not initialized.")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
