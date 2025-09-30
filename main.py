# import os
# import re
# import uuid
# import time
# import hashlib
# import tempfile
# import json
# import logging
# from typing import List, Dict, Tuple, Optional
# from dataclasses import dataclass
# from datetime import datetime
# from functools import lru_cache
# import concurrent.futures
# from threading import Lock

# # Flask imports
# from flask import Flask, render_template, request, jsonify, send_from_directory
# from flask_cors import CORS
# from werkzeug.utils import secure_filename

# # Document processing
# import pandas as pd
# from pypdf import PdfReader
# import numpy as np

# # Vector DB - Pinecone
# from pinecone import Pinecone, ServerlessSpec

# # AWS S3 (optional)
# try:
#     import boto3
#     from botocore.exceptions import NoCredentialsError
#     S3_AVAILABLE = True
# except ImportError:
#     S3_AVAILABLE = False

# # OpenAI
# import tiktoken
# from openai import OpenAI

# # Fuzzy matching for better item code matching
# try:
#     from fuzzywuzzy import fuzz, process
#     FUZZY_AVAILABLE = True
# except ImportError:
#     FUZZY_AVAILABLE = False
#     print("⚠️ fuzzywuzzy not installed. Install with: pip install fuzzywuzzy python-Levenshtein")

# # Load environment variables
# from dotenv import load_dotenv
# load_dotenv()

# # Setup logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # -----------------------------
# # Configuration
# # -----------------------------
# CHUNK_TOKENS = 400
# OVERLAP_TOKENS = 50
# MAX_CONTEXT_LENGTH = 8000
# EMBEDDING_MODEL = "text-embedding-3-small"
# CHAT_MODEL = "gpt-4o-mini"
# PINECONE_INDEX_NAME = "kb-scout-documents"
# MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB for faster processing
# PINECONE_DIMENSION = 1536
# BATCH_SIZE = 20  # Smaller batch size for reliability
# MAX_CHUNKS_PER_FILE = 30  # Limit chunks to speed up processing
# ALLOWED_EXTENSIONS = {'pdf', 'csv', 'xlsx', 'xls', 'txt'}

# # -----------------------------
# # Global Caches
# # -----------------------------
# _cache = {
#     'openai_client': None,
#     'pinecone_client': None,
#     'pinecone_index': None,
#     's3_client': None,
#     'tokenizer': None,
#     'embeddings': {},
#     'uploaded_files': {},
#     'csv_lookup_cache': {},
#     'full_documents': {}  # Cache for full document content
# }

# # Document store for persistence
# document_store = {
#     'files': {},
#     'full_content': {},  # Store complete document content
#     'lock': Lock()
# }

# # -----------------------------
# # Client Initialization
# # -----------------------------
# @lru_cache(maxsize=1)
# def get_openai_client() -> Optional[OpenAI]:
#     """Get or create OpenAI client"""
#     if not _cache['openai_client']:
#         api_key = os.getenv("OPENAI_API_KEY")
#         if not api_key:
#             logger.warning("OPENAI_API_KEY not found in environment")
#             return None
#         try:
#             _cache['openai_client'] = OpenAI(api_key=api_key, timeout=30)
#             logger.info("✅ OpenAI client initialized")
#         except Exception as e:
#             logger.error(f"Failed to initialize OpenAI: {e}")
#             return None
#     return _cache['openai_client']

# @lru_cache(maxsize=1)
# def get_pinecone_client():
#     """Get or create Pinecone client"""
#     if not _cache['pinecone_client']:
#         api_key = os.getenv("PINECONE_API_KEY")
#         if not api_key:
#             logger.warning("PINECONE_API_KEY not found in environment")
#             return None
#         try:
#             _cache['pinecone_client'] = Pinecone(api_key=api_key)
#             logger.info("✅ Pinecone client initialized")
#         except Exception as e:
#             logger.error(f"Failed to initialize Pinecone: {e}")
#             return None
#     return _cache['pinecone_client']

# def get_pinecone_index():
#     """Get or create Pinecone index"""
#     if not _cache['pinecone_index']:
#         pc = get_pinecone_client()
#         if not pc:
#             return None
        
#         try:
#             # Check if index exists
#             existing_indexes = pc.list_indexes().names()
            
#             if PINECONE_INDEX_NAME not in existing_indexes:
#                 logger.info(f"Creating Pinecone index: {PINECONE_INDEX_NAME}")
#                 pc.create_index(
#                     name=PINECONE_INDEX_NAME,
#                     dimension=PINECONE_DIMENSION,
#                     metric='cosine',
#                     spec=ServerlessSpec(
#                         cloud='aws',
#                         region=os.getenv('PINECONE_REGION', 'us-east-1')
#                     )
#                 )
#                 time.sleep(5)  # Wait for index creation
            
#             _cache['pinecone_index'] = pc.Index(PINECONE_INDEX_NAME)
            
#             # Test the index
#             stats = _cache['pinecone_index'].describe_index_stats()
#             logger.info(f"✅ Pinecone index ready. Vectors: {stats.get('total_vector_count', 0)}")
            
#         except Exception as e:
#             logger.error(f"Failed to setup Pinecone index: {e}")
#             return None
    
#     return _cache['pinecone_index']

# @lru_cache(maxsize=1)
# def get_s3_client():
#     """Get or create S3 client"""
#     if not S3_AVAILABLE:
#         return None
        
#     if not _cache['s3_client']:
#         access_key = os.getenv("AWS_ACCESS_KEY_ID")
#         secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
#         if not access_key or not secret_key:
#             return None  # S3 is optional
#         _cache['s3_client'] = boto3.client(
#             's3',
#             aws_access_key_id=access_key,
#             aws_secret_access_key=secret_key,
#             region_name=os.getenv("AWS_REGION", "us-east-1")
#         )
#     return _cache['s3_client']

# def get_tokenizer():
#     """Get or create tokenizer"""
#     if not _cache['tokenizer']:
#         _cache['tokenizer'] = tiktoken.get_encoding("cl100k_base")
#     return _cache['tokenizer']

# # -----------------------------
# # Text Processing Functions
# # -----------------------------
# def clean_text(text: str) -> str:
#     """Clean text for processing"""
#     text = re.sub(r'\s+', ' ', text)
#     text = re.sub(r'[^\w\s\.\,\!\?\-\(\)\$\%\/]', '', text)
#     return text.strip()

# def chunk_text(text: str, max_tokens: int = CHUNK_TOKENS) -> List[str]:
#     """Simple fast chunking"""
#     # Character-based chunking for speed
#     chunk_size = max_tokens * 4  # Approximate chars per token
#     chunks = []
    
#     for i in range(0, len(text), chunk_size - 200):  # Overlap
#         chunk = text[i:i + chunk_size]
#         if len(chunk.strip()) > 100:
#             chunks.append(chunk)
#             if len(chunks) >= MAX_CHUNKS_PER_FILE:
#                 break
    
#     return chunks

# # -----------------------------
# # Embedding Functions
# # -----------------------------
# def generate_embedding(text: str) -> List[float]:
#     """Generate embedding for text with fallback"""
#     # Check cache first
#     text_hash = hashlib.md5(text.encode()).hexdigest()
#     if text_hash in _cache['embeddings']:
#         return _cache['embeddings'][text_hash]
    
#     client = get_openai_client()
#     if not client:
#         # Return random embedding as fallback
#         return np.random.randn(PINECONE_DIMENSION).tolist()
    
#     try:
#         response = client.embeddings.create(
#             model=EMBEDDING_MODEL,
#             input=text[:8000]  # Limit input size
#         )
#         embedding = response.data[0].embedding
#         _cache['embeddings'][text_hash] = embedding
#         return embedding
#     except Exception as e:
#         logger.warning(f"Embedding generation failed: {e}")
#         # Return random embedding as fallback
#         return np.random.randn(PINECONE_DIMENSION).tolist()

# def generate_embeddings_batch(texts: List[str]) -> List[List[float]]:
#     """Generate embeddings for multiple texts"""
#     embeddings = []
    
#     with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
#         futures = [executor.submit(generate_embedding, text) for text in texts]
#         for future in concurrent.futures.as_completed(futures):
#             try:
#                 embedding = future.result(timeout=10)
#                 embeddings.append(embedding)
#             except Exception as e:
#                 logger.warning(f"Failed to get embedding: {e}")
#                 embeddings.append(np.random.randn(PINECONE_DIMENSION).tolist())
    
#     return embeddings

# # -----------------------------
# # Document Processing
# # -----------------------------
# def process_pdf(file_path: str, filename: str) -> Dict:
#     """Process PDF file quickly and store full content"""
#     try:
#         logger.info(f"Processing PDF: {filename}")
#         reader = PdfReader(file_path)
        
#         # Store full document content
#         full_text = ""
#         all_pages_content = []
        
#         # Process all pages but chunk only limited pages
#         total_pages = len(reader.pages)
#         for i in range(total_pages):
#             try:
#                 page_text = reader.pages[i].extract_text()
#                 if page_text:
#                     all_pages_content.append(f"[Page {i+1}]\n{page_text}")
#                     if i < 20:  # Only first 20 pages for chunking
#                         full_text += f" {page_text}"
#             except Exception as e:
#                 logger.warning(f"Failed to extract page {i}: {e}")
#                 continue
        
#         # Store complete document
#         complete_document = "\n".join(all_pages_content)
#         with document_store['lock']:
#             document_store['full_content'][filename] = {
#                 'content': complete_document,
#                 'pages': total_pages,
#                 'type': 'PDF'
#             }
#             _cache['full_documents'][filename] = complete_document
        
#         # Clean and chunk for vector search
#         cleaned_text = clean_text(full_text)
#         chunks = chunk_text(cleaned_text)
        
#         doc_id = str(uuid.uuid4())[:8]
        
#         # Generate embeddings
#         logger.info(f"Generating embeddings for {len(chunks)} chunks...")
#         embeddings = generate_embeddings_batch(chunks)
        
#         # Prepare for Pinecone
#         vectors = []
#         for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
#             vectors.append({
#                 'id': f"{doc_id}_{i}",
#                 'values': embedding,
#                 'metadata': {
#                     'filename': filename,
#                     'doc_id': doc_id,
#                     'chunk_index': i,
#                     'content': chunk[:1000]  # Store truncated content
#                 }
#             })
        
#         return {
#             'success': True,
#             'doc_id': doc_id,
#             'filename': filename,
#             'vectors': vectors,
#             'chunks': len(chunks),
#             'total_pages': total_pages,
#             'type': 'PDF Document'
#         }
        
#     except Exception as e:
#         logger.error(f"Error processing PDF: {e}")
#         return {'success': False, 'error': str(e)}

# def process_csv(file_path: str, filename: str) -> Dict:
#     """Process CSV/Excel file with row-by-row indexing for exact Q&A"""
#     try:
#         logger.info(f"Processing CSV/Excel: {filename}")
        
#         # Read file based on extension
#         if filename.lower().endswith('.csv'):
#             df = pd.read_csv(file_path)
#         else:  # Excel files
#             df = pd.read_excel(file_path)
        
#         # Clean column names
#         df.columns = df.columns.str.strip()
        
#         # Store full DataFrame content
#         full_content = df.to_string()
#         with document_store['lock']:
#             document_store['full_content'][filename] = {
#                 'content': full_content,
#                 'rows': len(df),
#                 'columns': list(df.columns),
#                 'type': 'CSV/Excel',
#                 'dataframe': df.to_dict('records')  # Store as dict for easy access
#             }
#             _cache['full_documents'][filename] = full_content
        
#         # Cache DataFrame for direct lookups
#         _cache['csv_lookup_cache'][filename] = df
        
#         doc_id = str(uuid.uuid4())[:8]
#         vectors = []
        
#         # Strategy 1: Index each row as a separate vector
#         logger.info(f"Indexing {len(df)} rows individually...")
        
#         for idx, row in df.iterrows():
#             # Create searchable text from row
#             row_text = ""
#             row_dict = {}
            
#             for col in df.columns:
#                 value = row[col]
#                 # Handle different data types
#                 if pd.notna(value):
#                     row_dict[col] = str(value)
#                     row_text += f"{col}: {value}. "
            
#             # Create embedding for this row
#             embedding = generate_embedding(row_text)
            
#             # Store as vector with rich metadata
#             vectors.append({
#                 'id': f"{doc_id}_row_{idx}",
#                 'values': embedding,
#                 'metadata': {
#                     'filename': filename,
#                     'doc_id': doc_id,
#                     'row_index': int(idx),
#                     'type': 'csv_row',
#                     'content': row_text[:1000],  # Truncated for storage
#                     'row_data': json.dumps(row_dict)[:1000],  # Store row data as JSON
#                     **{f"col_{col}": str(row[col])[:100] if pd.notna(row[col]) else "" 
#                        for col in df.columns[:5]}  # Store first 5 columns as metadata
#                 }
#             })
        
#         # Strategy 2: Index column summaries for aggregate queries
#         for col in df.columns:
#             col_data = df[col].dropna()
#             if len(col_data) > 0:
#                 # Create column summary
#                 col_summary = f"Column {col} contains: "
                
#                 if col_data.dtype in ['int64', 'float64']:
#                     # Numeric column
#                     col_summary += f"min={col_data.min()}, max={col_data.max()}, "
#                     col_summary += f"mean={col_data.mean():.2f}, count={len(col_data)}"
                    
#                     # Sample values
#                     samples = col_data.head(10).tolist()
#                     col_summary += f". Sample values: {samples}"
#                 else:
#                     # Text column
#                     unique_values = col_data.unique()[:20]  # First 20 unique values
#                     col_summary += f"{len(unique_values)} unique values: {', '.join(map(str, unique_values))}"
                
#                 # Create embedding for column summary
#                 embedding = generate_embedding(col_summary)
                
#                 vectors.append({
#                     'id': f"{doc_id}_col_{col}",
#                     'values': embedding,
#                     'metadata': {
#                         'filename': filename,
#                         'doc_id': doc_id,
#                         'type': 'csv_column',
#                         'column_name': col,
#                         'content': col_summary[:1000],
#                         'data_type': str(col_data.dtype),
#                         'unique_count': int(col_data.nunique())
#                     }
#                 })
        
#         # Strategy 3: Create searchable chunks of the table
#         chunk_size = 10  # Rows per chunk
#         for i in range(0, len(df), chunk_size):
#             chunk_df = df.iloc[i:i+chunk_size]
#             chunk_text = f"Rows {i+1} to {min(i+chunk_size, len(df))} of {filename}:\n"
#             chunk_text += chunk_df.to_string()
            
#             # Create embedding
#             embedding = generate_embedding(chunk_text)
            
#             vectors.append({
#                 'id': f"{doc_id}_chunk_{i//chunk_size}",
#                 'values': embedding,
#                 'metadata': {
#                     'filename': filename,
#                     'doc_id': doc_id,
#                     'type': 'csv_chunk',
#                     'chunk_index': i//chunk_size,
#                     'start_row': i,
#                     'end_row': min(i+chunk_size, len(df)),
#                     'content': chunk_text[:1000]
#                 }
#             })
        
#         logger.info(f"Created {len(vectors)} vectors for {filename}")
        
#         return {
#             'success': True,
#             'doc_id': doc_id,
#             'filename': filename,
#             'vectors': vectors,
#             'chunks': len(vectors),
#             'total_rows': len(df),
#             'total_columns': len(df.columns),
#             'type': 'CSV/Excel Spreadsheet'
#         }
        
#     except Exception as e:
#         logger.error(f"Error processing CSV/Excel: {e}")
#         return {'success': False, 'error': str(e)}

# # -----------------------------
# # Upload to Pinecone
# # -----------------------------
# def upload_to_pinecone(vectors: List[Dict]) -> bool:
#     """Upload vectors to Pinecone in batches"""
#     index = get_pinecone_index()
#     if not index:
#         logger.warning("Pinecone not available, skipping upload")
#         return True  # Return True to not block
    
#     try:
#         # Upload in batches
#         for i in range(0, len(vectors), BATCH_SIZE):
#             batch = vectors[i:i + BATCH_SIZE]
#             index.upsert(vectors=batch)
#             logger.info(f"Uploaded batch {i//BATCH_SIZE + 1}")
        
#         return True
#     except Exception as e:
#         logger.error(f"Pinecone upload failed: {e}")
#         return False

# # -----------------------------
# # Search Functions
# # -----------------------------
# def search_documents(query: str, top_k: int = 5) -> List[Dict]:
#     """Search documents"""
#     index = get_pinecone_index()
#     if not index:
#         return []
    
#     try:
#         # Generate query embedding
#         query_embedding = generate_embedding(query)
        
#         # Search
#         results = index.query(
#             vector=query_embedding,
#             top_k=top_k,
#             include_metadata=True
#         )
        
#         return results['matches']
#     except Exception as e:
#         logger.error(f"Search error: {e}")
#         return []

# # -----------------------------
# # Flask Application
# # -----------------------------
# app = Flask(__name__, 
#             static_folder='static',
#             template_folder='templates')
# CORS(app)
# app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# def allowed_file(filename):
#     return '.' in filename and \
#            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# @app.route('/')
# def index():
#     """Serve the main page"""
#     return render_template('index.html')

# @app.route('/static/<path:path>')
# def send_static(path):
#     """Serve static files"""
#     return send_from_directory('static', path)

# @app.route('/upload', methods=['POST'])
# def upload_files():
#     """Handle file uploads"""
#     start_time = time.time()
    
#     if 'files' not in request.files:
#         return jsonify({'success': False, 'message': 'No files provided'}), 400
    
#     files = request.files.getlist('files')
#     results = []
#     total_chunks = 0
    
#     for file in files:
#         if file and allowed_file(file.filename):
#             filename = secure_filename(file.filename)
#             temp_path = os.path.join(tempfile.gettempdir(), filename)
            
#             try:
#                 # Save file
#                 file.save(temp_path)
                
#                 # Check if already processed
#                 with document_store['lock']:
#                     if filename in document_store['files']:
#                         logger.info(f"File {filename} already processed")
#                         results.append(f"{filename} (already indexed)")
#                         os.remove(temp_path)
#                         continue
                
#                 # Process based on file type
#                 extension = filename.rsplit('.', 1)[1].lower()
                
#                 if extension == 'pdf':
#                     doc_result = process_pdf(temp_path, filename)
#                 elif extension in ['csv', 'xlsx', 'xls']:
#                     doc_result = process_csv(temp_path, filename)
#                 else:
#                     # Text file
#                     with open(temp_path, 'r', encoding='utf-8', errors='ignore') as f:
#                         text = f.read()
                    
#                     # Store full content
#                     with document_store['lock']:
#                         document_store['full_content'][filename] = {
#                             'content': text,
#                             'size': len(text),
#                             'type': 'Text'
#                         }
#                         _cache['full_documents'][filename] = text
                    
#                     chunks = chunk_text(text[:10000])  # Chunk limited portion
#                     embeddings = generate_embeddings_batch(chunks)
                    
#                     doc_id = str(uuid.uuid4())[:8]
#                     vectors = []
#                     for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
#                         vectors.append({
#                             'id': f"{doc_id}_{i}",
#                             'values': embedding,
#                             'metadata': {
#                                 'filename': filename,
#                                 'doc_id': doc_id,
#                                 'chunk_index': i,
#                                 'content': chunk[:1000]
#                             }
#                         })
                    
#                     doc_result = {
#                         'success': True,
#                         'doc_id': doc_id,
#                         'filename': filename,
#                         'vectors': vectors,
#                         'chunks': len(chunks),
#                         'type': 'Text Document'
#                     }
                
#                 # Clean up temp file
#                 os.remove(temp_path)
                
#                 if doc_result.get('success'):
#                     # Upload to Pinecone
#                     upload_success = upload_to_pinecone(doc_result['vectors'])
                    
#                     if upload_success:
#                         # Store document info
#                         with document_store['lock']:
#                             document_store['files'][filename] = {
#                                 'doc_id': doc_result['doc_id'],
#                                 'chunks': doc_result['chunks'],
#                                 'type': doc_result['type'],
#                                 'uploaded_at': datetime.utcnow().isoformat()
#                             }
#                             _cache['uploaded_files'] = document_store['files'].copy()
                        
#                         results.append(f"{filename} ({doc_result['chunks']} chunks)")
#                         total_chunks += doc_result['chunks']
#                         logger.info(f"✅ Successfully processed {filename}")
#                     else:
#                         results.append(f"{filename} (upload failed)")
#                 else:
#                     results.append(f"{filename} (processing failed)")
                    
#             except Exception as e:
#                 logger.error(f"Failed to process {filename}: {e}")
#                 results.append(f"{filename} (error)")
#                 if os.path.exists(temp_path):
#                     os.remove(temp_path)
    
#     processing_time = time.time() - start_time
    
#     if results:
#         return jsonify({
#             'success': True,
#             'message': f'Processed {len(results)} file(s) with {total_chunks} chunks in {processing_time:.1f}s'
#         })
#     else:
#         return jsonify({
#             'success': False,
#             'message': 'No valid files to process'
#         }), 400

# @app.route('/files', methods=['GET'])
# def get_files():
#     """Get list of uploaded files"""
#     with document_store['lock']:
#         files = document_store['files'].copy()
    
#     return jsonify({
#         'success': True,
#         'files': files
#     })

# @app.route('/status', methods=['GET'])
# def get_status():
#     """Get system status"""
#     try:
#         index = get_pinecone_index()
#         vector_count = 0
#         connected = False
        
#         if index:
#             try:
#                 stats = index.describe_index_stats()
#                 vector_count = stats.get('total_vector_count', 0)
#                 connected = True
#             except:
#                 pass
        
#         with document_store['lock']:
#             file_count = len(document_store['files'])
        
#         return jsonify({
#             'connected': connected,
#             'count': vector_count,
#             'files': file_count
#         })
#     except Exception as e:
#         logger.error(f"Status check failed: {e}")
#         return jsonify({
#             'connected': False,
#             'count': 0,
#             'files': 0
#         })

# @app.route('/chat', methods=['POST'])
# def chat():
#     """Handle chat queries with full document retrieval"""
#     data = request.json
#     message = data.get('message', '')
    
#     if not message:
#         return jsonify({'success': False, 'message': 'No message provided'}), 400
    
#     try:
#         # Check if user wants full document
#         full_doc_keywords = ['full document', 'complete document', 'entire document', 
#                            'whole document', 'full pdf', 'complete pdf', 'entire pdf',
#                            'all pages', 'complete file', 'full content', 'everything in',
#                            'show me all', 'give me everything', 'entire content', 'full text']
        
#         wants_full_doc = any(keyword in message.lower() for keyword in full_doc_keywords)
        
#         if wants_full_doc:
#             # Extract filename from message
#             filename_found = None
#             with document_store['lock']:
#                 for filename in document_store['files'].keys():
#                     if filename.lower() in message.lower() or \
#                        filename.split('.')[0].lower() in message.lower():
#                         filename_found = filename
#                         break
            
#             if filename_found and filename_found in document_store.get('full_content', {}):
#                 # Retrieve full document
#                 full_doc = document_store['full_content'][filename_found]
#                 doc_content = full_doc['content']
                
#                 # For very large documents, provide in segments
#                 MAX_RESPONSE_SIZE = 30000  # Increased limit for full content
                
#                 response = f"📄 **Full Document: {filename_found}**\n\n"
#                 response += f"Document Type: {full_doc['type']}\n"
                
#                 if full_doc['type'] == 'PDF':
#                     response += f"Total Pages: {full_doc.get('pages', 'Unknown')}\n"
#                 elif full_doc['type'] == 'CSV':
#                     response += f"Total Rows: {full_doc.get('rows', 'Unknown')}\n"
#                     response += f"Columns: {', '.join(full_doc.get('columns', []))[:200]}\n"
                
#                 response += f"Total Length: {len(doc_content)} characters\n"
#                 response += "=" * 50 + "\n\n"
                
#                 # Check if we need to paginate
#                 page_requested = None
#                 if 'page' in message.lower():
#                     # Extract page number if specified
#                     import re
#                     page_match = re.search(r'page\s*(\d+)', message.lower())
#                     if page_match:
#                         page_requested = int(page_match.group(1))
                
#                 if len(doc_content) > MAX_RESPONSE_SIZE:
#                     # Provide document in chunks
#                     chunk_size = MAX_RESPONSE_SIZE - 1000  # Leave room for metadata
#                     total_chunks = (len(doc_content) + chunk_size - 1) // chunk_size
                    
#                     # Determine which chunk to show
#                     chunk_num = (page_requested - 1) if page_requested else 0
#                     chunk_num = max(0, min(chunk_num, total_chunks - 1))
                    
#                     start_idx = chunk_num * chunk_size
#                     end_idx = min(start_idx + chunk_size, len(doc_content))
                    
#                     response += f"**Document Part {chunk_num + 1} of {total_chunks}**\n"
#                     response += f"(Showing characters {start_idx:,} to {end_idx:,})\n\n"
#                     response += doc_content[start_idx:end_idx]
                    
#                     response += f"\n\n{'=' * 50}\n"
#                     response += f"**This is part {chunk_num + 1} of {total_chunks}**\n"
                    
#                     if chunk_num < total_chunks - 1:
#                         response += f"To see the next part, ask: 'Show me page {chunk_num + 2} of {filename_found}'\n"
#                     if chunk_num > 0:
#                         response += f"To see the previous part, ask: 'Show me page {chunk_num} of {filename_found}'\n"
                    
#                     response += f"\nAlternatively, you can download the full document using the endpoint:\n"
#                     response += f"GET /document/{filename_found}"
#                 else:
#                     # Document is small enough to show in full
#                     response += "**Complete Document Content:**\n\n"
#                     response += doc_content
                
#                 return jsonify({
#                     'success': True,
#                     'response': response,
#                     'document_info': {
#                         'filename': filename_found,
#                         'total_length': len(doc_content),
#                         'type': full_doc['type']
#                     }
#                 })
#             elif not filename_found:
#                 # List available documents
#                 with document_store['lock']:
#                     available_docs = list(document_store['files'].keys())
                
#                 if available_docs:
#                     response = "I couldn't identify which document you want. Available documents:\n\n"
#                     for doc in available_docs:
#                         doc_info = document_store['files'][doc]
#                         response += f"• **{doc}** - {doc_info['type']} ({doc_info['chunks']} chunks)\n"
#                     response += "\nPlease specify the filename in your request. For example:\n"
#                     response += '"Show me the full document for example.pdf"'
#                 else:
#                     response = "No documents have been uploaded yet. Please upload a document first."
                
#                 return jsonify({
#                     'success': True,
#                     'response': response
#                 })
#             else:
#                 return jsonify({
#                     'success': True,
#                     'response': f"Document '{filename_found}' was found but full content is not available. It may not have been fully processed during upload."
#                 })
        
#         # Check for specific page request
#         if 'page' in message.lower() and any(word in message.lower() for word in ['show', 'get', 'display', 'give']):
#             # Extract page number and filename
#             import re
#             page_match = re.search(r'page\s*(\d+)', message.lower())
            
#             if page_match:
#                 page_num = int(page_match.group(1))
                
#                 # Find filename
#                 filename_found = None
#                 with document_store['lock']:
#                     for filename in document_store['files'].keys():
#                         if filename.lower() in message.lower() or \
#                            filename.split('.')[0].lower() in message.lower():
#                             filename_found = filename
#                             break
                
#                 if filename_found and filename_found in document_store.get('full_content', {}):
#                     full_doc = document_store['full_content'][filename_found]
                    
#                     # Extract specific page if PDF
#                     if full_doc['type'] == 'PDF' and '[Page' in full_doc['content']:
#                         # Split by page markers
#                         pages = full_doc['content'].split('[Page ')
#                         if 0 < page_num <= len(pages) - 1:
#                             page_content = f"[Page {pages[page_num]}"
#                             response = f"📄 **{filename_found} - Page {page_num}**\n\n"
#                             response += page_content
#                             return jsonify({
#                                 'success': True,
#                                 'response': response
#                             })
        
#         # Check for specific document summary request
#         summary_keywords = ['summary of', 'summarize', 'overview of', 'what is in', 'what\'s in']
#         wants_summary = any(keyword in message.lower() for keyword in summary_keywords)
        
#         if wants_summary:
#             # Extract filename
#             filename_found = None
#             with document_store['lock']:
#                 for filename in document_store['files'].keys():
#                     if filename.lower() in message.lower() or \
#                        filename.split('.')[0].lower() in message.lower():
#                         filename_found = filename
#                         break
            
#             if filename_found and filename_found in document_store.get('full_content', {}):
#                 full_doc = document_store['full_content'][filename_found]
#                 doc_info = document_store['files'][filename_found]
                
#                 # Create summary
#                 response = f"📊 **Document Summary: {filename_found}**\n\n"
#                 response += f"• **Type:** {doc_info['type']}\n"
#                 response += f"• **Uploaded:** {doc_info['uploaded_at']}\n"
#                 response += f"• **Chunks:** {doc_info['chunks']}\n"
                
#                 if full_doc['type'] == 'PDF':
#                     response += f"• **Pages:** {full_doc.get('pages', 'Unknown')}\n"
#                 elif full_doc['type'] == 'CSV':
#                     response += f"• **Rows:** {full_doc.get('rows', 'Unknown')}\n"
#                     response += f"• **Columns:** {', '.join(full_doc.get('columns', []))}\n"
                
#                 response += f"• **Total Size:** {len(full_doc['content'])} characters\n"
                
#                 # Add content preview
#                 response += f"\n**Preview (first 1000 characters):**\n"
#                 response += "```\n"
#                 response += full_doc['content'][:1000]
#                 response += "\n```\n..."
#                 response += f"\n\nTo see the full document, ask: 'Show me the full document for {filename_found}'"
                
#                 return jsonify({
#                     'success': True,
#                     'response': response
#                 })
        
#         # Check for direct CSV lookups
#         if FUZZY_AVAILABLE:
#             for filename, df in _cache['csv_lookup_cache'].items():
#                 # Try to find product codes or specific items
#                 if any(keyword in message.lower() for keyword in ['price', 'cost', 'item', 'product']):
#                     # Simple search in DataFrame
#                     search_results = []
#                     for col in df.columns:
#                         for idx, value in df[col].items():
#                             if str(value).lower() in message.lower() or message.lower() in str(value).lower():
#                                 search_results.append(df.iloc[idx].to_dict())
                    
#                     if search_results:
#                         response = f"Found {len(search_results)} matching items:\n\n"
#                         for item in search_results[:3]:
#                             response += f"• {item}\n"
#                         return jsonify({
#                             'success': True,
#                             'response': response
#                         })
        
#         # Regular semantic search
#         search_results = search_documents(message)
        
#         if not search_results:
#             return jsonify({
#                 'success': True,
#                 'response': "I couldn't find any relevant information. Please make sure you've uploaded the documents."
#             })
        
#         # Build context
#         context = "\n\n".join([
#             f"[{r['metadata'].get('filename', 'Unknown')}]: {r['metadata'].get('content', '')}"
#             for r in search_results[:3]
#             if r['score'] > 0.3
#         ])
        
#         if not context:
#             return jsonify({
#                 'success': True,
#                 'response': "Found some documents but they don't seem relevant to your question."
#             })
        
#         # Generate response
#         client = get_openai_client()
#         if not client:
#             # Return context directly if OpenAI not available
#             return jsonify({
#                 'success': True,
#                 'response': f"Found relevant information:\n\n{context[:1000]}"
#             })
        
#         prompt = f"""You are K&B Scout AI. Answer based on this context:

# Context: {context[:3000]}

# Question: {message}

# Provide a clear, concise answer. Format prices with $ and highlight product codes.
# If the user might benefit from seeing the full document, mention they can ask for it."""
        
#         response = client.chat.completions.create(
#             model=CHAT_MODEL,
#             messages=[
#                 {"role": "system", "content": "You are K&B Scout AI, a helpful assistant."},
#                 {"role": "user", "content": prompt}
#             ],
#             temperature=0.3,
#             max_tokens=500
#         )
        
#         return jsonify({
#             'success': True,
#             'response': response.choices[0].message.content
#         })
        
#     except Exception as e:
#         logger.error(f"Chat error: {e}")
#         return jsonify({
#             'success': False,
#             'message': 'Error processing your request'
#         }), 500

# @app.route('/document/<filename>', methods=['GET'])
# def get_full_document(filename):
#     """API endpoint to retrieve full document content - can be used for downloading"""
#     try:
#         # Support both with and without extension in URL
#         filename_decoded = filename.replace('%20', ' ')
        
#         with document_store['lock']:
#             # Try exact match first
#             if filename_decoded in document_store.get('full_content', {}):
#                 full_doc = document_store['full_content'][filename_decoded]
#             else:
#                 # Try to find by partial match
#                 found = None
#                 for doc_name in document_store.get('full_content', {}).keys():
#                     if filename_decoded in doc_name or doc_name.startswith(filename_decoded):
#                         found = doc_name
#                         break
                
#                 if found:
#                     full_doc = document_store['full_content'][found]
#                     filename_decoded = found
#                 else:
#                     return jsonify({
#                         'success': False,
#                         'message': f'Document {filename} not found'
#                     }), 404
            
#             # Return as plain text for large documents
#             from flask import Response
            
#             if len(full_doc['content']) > 100000:  # If larger than 100KB
#                 # Return as downloadable text file
#                 return Response(
#                     full_doc['content'],
#                     mimetype='text/plain',
#                     headers={
#                         'Content-Disposition': f'attachment; filename="{filename_decoded}.txt"',
#                         'Content-Type': 'text/plain; charset=utf-8'
#                     }
#                 )
#             else:
#                 # Return as JSON for smaller documents
#                 return jsonify({
#                     'success': True,
#                     'filename': filename_decoded,
#                     'type': full_doc['type'],
#                     'content': full_doc['content'],
#                     'metadata': {
#                         'pages': full_doc.get('pages'),
#                         'rows': full_doc.get('rows'),
#                         'columns': full_doc.get('columns'),
#                         'size': len(full_doc['content'])
#                     }
#                 })
                
#     except Exception as e:
#         logger.error(f"Error retrieving document: {e}")
#         return jsonify({
#             'success': False,
#             'message': 'Error retrieving document'
#         }), 500

# # @app.route('/documents/list', methods=['GET'])
# # def list_documents():
# #     """List all available documents with details"""
# #     try:
# #         with document_store['lock']:
# #             documents = []
# #             for filename, info in document_store['files'].items():
# #                 doc_detail = {
# #                     'filename': filename,
# #                     'type': info['type'],
# #                     'chunks': info['chunks'],
# #                     'uploaded_at': info['uploaded_at'],
# #                     'has_full_content': filename in document_store.get('full_content', {})
# #                 }
                
# #                 if filename in document_store.get('full_content', {}):
# #                     full_doc = document_store['full_content'][filename]
# #                     doc_detail['pages'] = full_doc.get('pages')
# #                     doc_detail['rows'] = full_doc.get('rows')
# #                     doc_detail['size'] = len(full_doc['content'])
                
# #                 documents.append(doc_detail)
        
# #         return jsonify({
# #             'success': True,
# #             'documents': documents,
# #             'total': len(documents)
# #         })
# #     except Exception as e:
# #         logger.error(f"Error listing documents: {e}")
# #         return jsonify({
# #             'success': False,
# #             'message': 'Error listing documents'
# #         }), 500
#     """Direct search endpoint"""
#     data = request.json
#     query = data.get('query', '')
    
#     if not query:
#         return jsonify({'success': False, 'message': 'No query provided'}), 400
    
#     try:
#         results = search_documents(query, top_k=10)
        
#         formatted_results = []
#         for match in results:
#             if match['score'] > 0.3:
#                 formatted_results.append({
#                     'content': match['metadata'].get('content', ''),
#                     'filename': match['metadata'].get('filename', 'Unknown'),
#                     'score': match['score']
#                 })
        
#         return jsonify({
#             'success': True,
#             'results': formatted_results,
#             'count': len(formatted_results)
#         })
        
#     except Exception as e:
#         logger.error(f"Search error: {e}")
#         return jsonify({'success': False, 'message': str(e)}), 500

# # -----------------------------
# # Setup and Initialization
# # -----------------------------
# def setup_directories():
#     """Create necessary directories"""
#     dirs = ['static', 'static/css', 'static/js', 'templates']
#     for dir_path in dirs:
#         os.makedirs(dir_path, exist_ok=True)

# def initialize_system():
#     """Initialize system on startup"""
#     logger.info("=" * 50)
#     logger.info("Initializing K&B Scout AI System...")
#     logger.info("=" * 50)
    
#     # Test OpenAI
#     client = get_openai_client()
#     if client:
#         try:
#             test = client.embeddings.create(
#                 model=EMBEDDING_MODEL,
#                 input="test"
#             )
#             logger.info("✅ OpenAI connection successful")
#         except Exception as e:
#             logger.error(f"❌ OpenAI test failed: {e}")
    
#     # Test Pinecone
#     index = get_pinecone_index()
#     if index:
#         logger.info("✅ Pinecone connection successful")
#     else:
#         logger.warning("⚠️ Pinecone not available - will work in limited mode")
    
#     # Test S3 (optional)
#     s3 = get_s3_client()
#     if s3:
#         logger.info("✅ S3 client available")
#     else:
#         logger.info("ℹ️ S3 not configured (optional)")
    
#     logger.info("=" * 50)
#     logger.info("System ready!")
#     logger.info("=" * 50)

# # -----------------------------
# # Main Entry Point
# # -----------------------------
# if __name__ == '__main__':
#     setup_directories()
#     initialize_system()
    
#     logger.info("Access the application at: http://localhost:5000")
    
#     app.run(debug=False, port=5000, threaded=True)



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
CHUNK_TOKENS = 400
OVERLAP_TOKENS = 50
MAX_CONTEXT_LENGTH = 8000
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
PINECONE_INDEX_NAME = "kb-scout-documents"
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB for faster processing
PINECONE_DIMENSION = 1536
BATCH_SIZE = 20  # Smaller batch size for reliability
MAX_CHUNKS_PER_FILE = 30  # Limit chunks to speed up processing
ALLOWED_EXTENSIONS = {'pdf', 'csv', 'xlsx', 'xls', 'txt'}
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

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
    if not _cache['tokenizer']:
        _cache['tokenizer'] = tiktoken.get_encoding("cl100k_base")
    return _cache['tokenizer']

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
def generate_embedding(text: str) -> List[float]:
    """Generate embedding for text with fallback"""
    # Check cache first
    text_hash = hashlib.md5(text.encode()).hexdigest()
    if text_hash in _cache['embeddings']:
        return _cache['embeddings'][text_hash]
    
    client = get_openai_client()
    if not client:
        # Return random embedding as fallback
        return np.random.randn(PINECONE_DIMENSION).tolist()
    
    try:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text[:8000]  # Limit input size
        )
        embedding = response.data[0].embedding
        _cache['embeddings'][text_hash] = embedding
        return embedding
    except Exception as e:
        logger.warning(f"Embedding generation failed: {e}")
        # Return random embedding as fallback
        return np.random.randn(PINECONE_DIMENSION).tolist()

def generate_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for multiple texts"""
    embeddings = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(generate_embedding, text) for text in texts]
        for future in concurrent.futures.as_completed(futures):
            try:
                embedding = future.result(timeout=10)
                embeddings.append(embedding)
            except Exception as e:
                logger.warning(f"Failed to get embedding: {e}")
                embeddings.append(np.random.randn(PINECONE_DIMENSION).tolist())
    
    return embeddings








# -----------------------------
# Document Processing
# -----------------------------
def process_pdf(file_path: str, filename: str) -> Dict:
    """Process PDF file quickly and store full content"""
    try:
        logger.info(f"Processing PDF: {filename}")
        reader = PdfReader(file_path)
        
        # Store full document content
        full_text = ""
        all_pages_content = []
        
        # Process all pages but chunk only limited pages
        total_pages = len(reader.pages)
        for i in range(total_pages):
            try:
                page_text = reader.pages[i].extract_text()
                if page_text:
                    all_pages_content.append(f"[Page {i+1}]\n{page_text}")
                    if i < 20:  # Only first 20 pages for chunking
                        full_text += f" {page_text}"
            except Exception as e:
                logger.warning(f"Failed to extract page {i}: {e}")
                continue
        
        # Store complete document
        complete_document = "\n".join(all_pages_content)
        with document_store['lock']:
            document_store['full_content'][filename] = {
                'content': complete_document,
                'pages': total_pages,
                'type': 'PDF'
            }
            _cache['full_documents'][filename] = complete_document
        
        # Clean and chunk for vector search
        cleaned_text = clean_text(full_text)
        chunks = chunk_text(cleaned_text)
        
        doc_id = str(uuid.uuid4())[:8]
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(chunks)} chunks...")
        embeddings = generate_embeddings_batch(chunks)
        
        # Prepare for Pinecone
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
                    'content': chunk[:1000],  # Store truncated content
                    'file_type': 'pdf'
                }
            })
        
        return {
            'success': True,
            'doc_id': doc_id,
            'filename': filename,
            'vectors': vectors,
            'vector_ids': vector_ids,
            'chunks': len(chunks),
            'total_pages': total_pages,
            'type': 'PDF Document'
        }
        
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        return {'success': False, 'error': str(e)}

def process_csv(file_path: str, filename: str) -> Dict:
    """Process CSV/Excel file with row-by-row indexing for exact Q&A"""
    try:
        logger.info(f"Processing CSV/Excel: {filename}")
        
        # Read file based on extension with encoding handling
        if filename.lower().endswith('.csv'):
            # Try multiple encodings for CSV files
            encodings_to_try = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252', 'iso-8859-1']
            df = None
            
            for encoding in encodings_to_try:
                try:
                    logger.info(f"Trying encoding: {encoding}")
                    df = pd.read_csv(file_path, encoding=encoding)
                    logger.info(f"Successfully read CSV with {encoding} encoding")
                    break
                except UnicodeDecodeError as e:
                    logger.warning(f"Failed with {encoding}: {e}")
                    continue
                except Exception as e:
                    logger.warning(f"Other error with {encoding}: {e}")
                    continue
            
            if df is None:
                # If all encodings fail, try with error handling
                try:
                    df = pd.read_csv(file_path, encoding='utf-8', errors='ignore')
                    logger.warning("Read CSV with 'ignore' errors - some characters may be missing")
                except Exception as e:
                    # Last resort: try reading as binary and converting
                    try:
                        with open(file_path, 'rb') as f:
                            raw_data = f.read()
                        
                        # Try to detect encoding
                        import chardet
                        detected = chardet.detect(raw_data)
                        detected_encoding = detected.get('encoding', 'utf-8')
                        logger.info(f"Detected encoding: {detected_encoding}")
                        
                        df = pd.read_csv(file_path, encoding=detected_encoding)
                    except ImportError:
                        # chardet not available, use errors='replace'
                        df = pd.read_csv(file_path, encoding='utf-8', errors='replace')
                        logger.warning("Read CSV with 'replace' errors - some characters replaced with �")
                    except Exception as final_error:
                        raise Exception(f"Could not read CSV file with any encoding method: {final_error}")
        else:  # Excel files
            try:
                df = pd.read_excel(file_path)
            except Exception as e:
                # Try with different engines for Excel
                try:
                    df = pd.read_excel(file_path, engine='openpyxl')
                except Exception as e2:
                    try:
                        df = pd.read_excel(file_path, engine='xlrd')
                    except Exception as e3:
                        raise Exception(f"Could not read Excel file: {e}, {e2}, {e3}")
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Store full DataFrame content
        full_content = df.to_string()
        with document_store['lock']:
            document_store['full_content'][filename] = {
                'content': full_content,
                'rows': len(df),
                'columns': list(df.columns),
                'type': 'CSV/Excel',
                'dataframe': df.to_dict('records')  # Store as dict for easy access
            }
            _cache['full_documents'][filename] = full_content
        
        # Cache DataFrame for direct lookups
        _cache['csv_lookup_cache'][filename] = df
        
        doc_id = str(uuid.uuid4())[:8]
        vectors = []
        vector_ids = []
        
        # Strategy 1: Index each row as a separate vector
        logger.info(f"Indexing {len(df)} rows individually...")
        
        for idx, row in df.iterrows():
            # Create searchable text from row
            row_text = ""
            row_dict = {}
            
            for col in df.columns:
                value = row[col]
                # Handle different data types
                if pd.notna(value):
                    row_dict[col] = str(value)
                    row_text += f"{col}: {value}. "
            
            # Create embedding for this row
            embedding = generate_embedding(row_text)
            
            # Store as vector with rich metadata
            vector_id = f"{doc_id}_row_{idx}"
            vector_ids.append(vector_id)
            vectors.append({
                'id': vector_id,
                'values': embedding,
                'metadata': {
                    'filename': filename,
                    'doc_id': doc_id,
                    'row_index': int(idx),
                    'type': 'csv_row',
                    'content': row_text[:1000],  # Truncated for storage
                    'row_data': json.dumps(row_dict)[:1000],  # Store row data as JSON
                    'file_type': 'csv',
                    **{f"col_{col}": str(row[col])[:100] if pd.notna(row[col]) else "" 
                       for col in df.columns[:5]}  # Store first 5 columns as metadata
                }
            })
        
        # Strategy 2: Index column summaries for aggregate queries
        for col in df.columns:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                # Create column summary
                col_summary = f"Column {col} contains: "
                
                if col_data.dtype in ['int64', 'float64']:
                    # Numeric column
                    col_summary += f"min={col_data.min()}, max={col_data.max()}, "
                    col_summary += f"mean={col_data.mean():.2f}, count={len(col_data)}"
                    
                    # Sample values
                    samples = col_data.head(10).tolist()
                    col_summary += f". Sample values: {samples}"
                else:
                    # Text column
                    unique_values = col_data.unique()[:20]  # First 20 unique values
                    col_summary += f"{len(unique_values)} unique values: {', '.join(map(str, unique_values))}"
                
                # Create embedding for column summary
                embedding = generate_embedding(col_summary)
                
                vector_id = f"{doc_id}_col_{col}"
                vector_ids.append(vector_id)
                vectors.append({
                    'id': vector_id,
                    'values': embedding,
                    'metadata': {
                        'filename': filename,
                        'doc_id': doc_id,
                        'type': 'csv_column',
                        'column_name': col,
                        'content': col_summary[:1000],
                        'data_type': str(col_data.dtype),
                        'unique_count': int(col_data.nunique()),
                        'file_type': 'csv'
                    }
                })
        
        # Strategy 3: Create searchable chunks of the table
        chunk_size = 10  # Rows per chunk
        for i in range(0, len(df), chunk_size):
            chunk_df = df.iloc[i:i+chunk_size]
            chunk_text = f"Rows {i+1} to {min(i+chunk_size, len(df))} of {filename}:\n"
            chunk_text += chunk_df.to_string()
            
            # Create embedding
            embedding = generate_embedding(chunk_text)
            
            vector_id = f"{doc_id}_chunk_{i//chunk_size}"
            vector_ids.append(vector_id)
            vectors.append({
                'id': vector_id,
                'values': embedding,
                'metadata': {
                    'filename': filename,
                    'doc_id': doc_id,
                    'type': 'csv_chunk',
                    'chunk_index': i//chunk_size,
                    'start_row': i,
                    'end_row': min(i+chunk_size, len(df)),
                    'content': chunk_text[:1000],
                    'file_type': 'csv'
                }
            })
        
        logger.info(f"Created {len(vectors)} vectors for {filename}")
        
        return {
            'success': True,
            'doc_id': doc_id,
            'filename': filename,
            'vectors': vectors,
            'vector_ids': vector_ids,
            'chunks': len(vectors),
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'type': 'CSV/Excel Spreadsheet'
        }
        
    except Exception as e:
        logger.error(f"Error processing CSV/Excel: {e}")
        return {'success': False, 'error': str(e)}

# -----------------------------
# Upload to Pinecone
# -----------------------------
def upload_to_pinecone(vectors: List[Dict]) -> bool:
    """Upload vectors to Pinecone in batches"""
    index = get_pinecone_index()
    if not index:
        logger.warning("Pinecone not available, skipping upload")
        return True  # Return True to not block
    
    try:
        # Upload in batches
        for i in range(0, len(vectors), BATCH_SIZE):
            batch = vectors[i:i + BATCH_SIZE]
            index.upsert(vectors=batch)
            logger.info(f"Uploaded batch {i//BATCH_SIZE + 1}")
        
        return True
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
def search_documents(query: str, top_k: int = 50) -> List[Dict]:
    """Search documents across ALL uploaded files in Pinecone"""
    index = get_pinecone_index()
    if not index:
        return []
    
    try:
        # Generate query embedding
        query_embedding = generate_embedding(query)
        
        # Search across ALL vectors in Pinecone (no filename filtering)
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        return results['matches']
    except Exception as e:
        logger.error(f"Search error: {e}")
        return []

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
            else:
                logger.warning("Pinecone index is None")
        except Exception as e:
            logger.error(f"Error getting Pinecone stats: {e}")
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
        
        return jsonify({
            'connected': connected,
            'pinecone_stats': index_stats,
            's3_connected': s3_connected,
            'local_files': file_count,
            'total_chunks_tracked': total_chunks,
            'cache_size': cache_info,
            'system_health': {
                'openai_available': get_openai_client() is not None,
                'pinecone_available': connected,
                's3_available': s3_connected
            }
        })
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






@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat queries with full document retrieval and search across ALL files"""
    data = request.json
    message = data.get('message', '')
    
    if not message:
        return jsonify({'success': False, 'message': 'No message provided'}), 400
    
    try:
        # Check if user wants full document
        full_doc_keywords = ['full document', 'complete document', 'entire document','content', 
                           'whole document', 'full pdf', 'complete pdf', 'entire pdf',
                           'all pages', 'complete file', 'full content', 'everything in',
                           'show me all', 'give me everything', 'entire content', 'full text']
        
        wants_full_doc = any(keyword in message.lower() for keyword in full_doc_keywords)
        
        if wants_full_doc:
            # [Keep existing full document handling code as-is]
            # Extract filename from message
            filename_found = None
            with document_store['lock']:
                for filename in document_store['files'].keys():
                    if filename.lower() in message.lower() or \
                       filename.split('.')[0].lower() in message.lower():
                        filename_found = filename
                        break
            
            if filename_found and filename_found in document_store.get('full_content', {}):
                full_doc = document_store['full_content'][filename_found]
                doc_content = full_doc['content']
                MAX_RESPONSE_SIZE = 30000
                
                response = f"📄 **Full Document: {filename_found}**\n\n"
                response += f"Document Type: {full_doc['type']}\n"
                
                if full_doc['type'] == 'PDF':
                    response += f"Total Pages: {full_doc.get('pages', 'Unknown')}\n"
                elif full_doc['type'] == 'CSV':
                    response += f"Total Rows: {full_doc.get('rows', 'Unknown')}\n"
                    response += f"Columns: {', '.join(full_doc.get('columns', []))[:200]}\n"
                
                response += f"Total Length: {len(doc_content)} characters\n"
                response += "=" * 50 + "\n\n"
                
                page_requested = None
                if 'page' in message.lower():
                    import re
                    page_match = re.search(r'page\s*(\d+)', message.lower())
                    if page_match:
                        page_requested = int(page_match.group(1))
                
                if len(doc_content) > MAX_RESPONSE_SIZE:
                    chunk_size = MAX_RESPONSE_SIZE - 1000
                    total_chunks = (len(doc_content) + chunk_size - 1) // chunk_size
                    chunk_num = (page_requested - 1) if page_requested else 0
                    chunk_num = max(0, min(chunk_num, total_chunks - 1))
                    start_idx = chunk_num * chunk_size
                    end_idx = min(start_idx + chunk_size, len(doc_content))
                    
                    response += f"**Document Part {chunk_num + 1} of {total_chunks}**\n"
                    response += f"(Showing characters {start_idx:,} to {end_idx:,})\n\n"
                    response += doc_content[start_idx:end_idx]
                    response += f"\n\n{'=' * 50}\n"
                    response += f"**This is part {chunk_num + 1} of {total_chunks}**\n"
                    
                    if chunk_num < total_chunks - 1:
                        response += f"To see the next part, ask: 'Show me page {chunk_num + 2} of {filename_found}'\n"
                    if chunk_num > 0:
                        response += f"To see the previous part, ask: 'Show me page {chunk_num} of {filename_found}'\n"
                    
                    response += f"\nAlternatively, you can download the full document using the endpoint:\n"
                    response += f"GET /document/{filename_found}"
                else:
                    response += "**Complete Document Content:**\n\n"
                    response += doc_content
                
                return jsonify({
                    'success': True,
                    'response': response,
                    'document_info': {
                        'filename': filename_found,
                        'total_length': len(doc_content),
                        'type': full_doc['type']
                    }
                })
            elif not filename_found:
                with document_store['lock']:
                    available_docs = list(document_store['files'].keys())
                
                if available_docs:
                    response = "I couldn't identify which document you want. Available documents:\n\n"
                    for doc in available_docs:
                        doc_info = document_store['files'][doc]
                        response += f"• **{doc}** - {doc_info['type']} ({doc_info['chunks']} chunks)\n"
                    response += "\nPlease specify the filename in your request."
                else:
                    response = "No documents have been uploaded yet."
                
                return jsonify({'success': True, 'response': response})
            else:
                return jsonify({
                    'success': True,
                    'response': f"Document '{filename_found}' was found but full content is not available."
                })
        
        # [Keep existing page and summary handling code]
        if 'page' in message.lower() and any(word in message.lower() for word in ['show', 'get', 'display', 'give']):
            import re
            page_match = re.search(r'page\s*(\d+)', message.lower())
            if page_match:
                page_num = int(page_match.group(1))
                filename_found = None
                with document_store['lock']:
                    for filename in document_store['files'].keys():
                        if filename.lower() in message.lower() or \
                           filename.split('.')[0].lower() in message.lower():
                            filename_found = filename
                            break
                
                if filename_found and filename_found in document_store.get('full_content', {}):
                    full_doc = document_store['full_content'][filename_found]
                    if full_doc['type'] == 'PDF' and '[Page' in full_doc['content']:
                        pages = full_doc['content'].split('[Page ')
                        if 0 < page_num <= len(pages) - 1:
                            page_content = f"[Page {pages[page_num]}"
                            response = f"📄 **{filename_found} - Page {page_num}**\n\n{page_content}"
                            return jsonify({'success': True, 'response': response})
        
        summary_keywords = ['summary of', 'summarize', 'overview of', 'what is in', 'what\'s in']
        wants_summary = any(keyword in message.lower() for keyword in summary_keywords)
        
        if wants_summary:
            filename_found = None
            with document_store['lock']:
                for filename in document_store['files'].keys():
                    if filename.lower() in message.lower() or \
                       filename.split('.')[0].lower() in message.lower():
                        filename_found = filename
                        break
            
            if filename_found and filename_found in document_store.get('full_content', {}):
                full_doc = document_store['full_content'][filename_found]
                doc_info = document_store['files'][filename_found]
                
                response = f"📊 **Document Summary: {filename_found}**\n\n"
                response += f"• **Type:** {doc_info['type']}\n"
                response += f"• **Uploaded:** {doc_info['uploaded_at']}\n"
                response += f"• **Chunks:** {doc_info['chunks']}\n"
                
                if full_doc['type'] == 'PDF':
                    response += f"• **Pages:** {full_doc.get('pages', 'Unknown')}\n"
                elif full_doc['type'] == 'CSV':
                    response += f"• **Rows:** {full_doc.get('rows', 'Unknown')}\n"
                    response += f"• **Columns:** {', '.join(full_doc.get('columns', []))}\n"
                
                response += f"• **Total Size:** {len(full_doc['content'])} characters\n"
                response += f"\n**Preview (first 1000 characters):**\n```\n"
                response += full_doc['content'][:1000]
                response += "\n```\n..."
                response += f"\n\nTo see the full document, ask: 'Show me the full document for {filename_found}'"
                
                return jsonify({'success': True, 'response': response})
        
        # NEW: Extract multiple items from the query
        items = extract_multiple_items(message)
        
        if len(items) > 1:
            # Multi-item query detected
            return handle_multi_item_query(message, items)
        
        # Single item or general query - use existing search
        search_results = search_documents(message, top_k=50)
        
        if not search_results:
            return jsonify({
                'success': True,
                'response': "I couldn't find any relevant information in the document database. Please make sure your question relates to the uploaded documents."
            })
        
        # Build context from search results
        context_parts = []
        files_found = set()
        
        for r in search_results:
            if r['score'] > 0.2:
                filename = r['metadata'].get('filename', 'Unknown')
                content = r['metadata'].get('content', '')
                files_found.add(filename)
                context_parts.append(f"[{filename}]: {content}")
        
        context = "\n\n".join(context_parts[:5])
        
        if not context:
            return jsonify({
                'success': True,
                'response': f"Found {len(search_results)} documents but they don't seem highly relevant. Try rephrasing your query."
            })
        
        # Generate response
        client = get_openai_client()
        if not client:
            files_list = ", ".join(files_found)
            return jsonify({
                'success': True,
                'response': f"Found relevant information from {len(files_found)} file(s): {files_list}\n\n{context[:2000]}...\n\n(OpenAI not available)"
            })
        
        files_list = ", ".join(files_found)
        prompt = f"""You are K&B Scout AI. Answer based on this context from the document database:

Context from files ({files_list}):
{context[:4000]}

Question: {message}

Provide a clear, comprehensive answer. Mention which files the information comes from.
Format prices with $ and highlight product codes if relevant."""
        
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": "You are K&B Scout AI, a helpful assistant that searches across all uploaded documents."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=800
        )
        
        ai_response = response.choices[0].message.content
        ai_response += f"\n\n---\n*Search found relevant information from {len(files_found)} file(s): {files_list}*"
        
        return jsonify({
            'success': True,
            'response': ai_response,
            'search_metadata': {
                'files_searched': list(files_found),
                'total_results': len(search_results),
                'relevant_results': len([r for r in search_results if r['score'] > 0.2])
            }
        })
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({
            'success': False,
            'message': 'Error processing your request'
        }), 500


def extract_multiple_items(message):
    """Extract multiple item codes or product names from a query"""
    import re
    
    items = []
    
    # Pattern 1: Comma-separated items (e.g., "price of item1, item2, and item3")
    if ',' in message or ' and ' in message.lower():
        # Split by common delimiters
        parts = re.split(r',|\sand\s|\sor\s', message.lower())
        
        # Extract potential item codes/names from each part
        for part in parts:
            # Remove common words/phrases
            part = re.sub(r'\b(price|cost|details|of|for|the|about|what|is|are)\b', '', part, flags=re.IGNORECASE)
            part = part.strip()
            
            # Look for product codes (alphanumeric patterns with 3+ chars)
            codes = re.findall(r'\b[A-Z0-9]{3,}[-_]?[A-Z0-9]*\b', part.upper())
            items.extend(codes)
            
            # Look for quoted items
            quoted = re.findall(r'["\']([^"\']+)["\']', part)
            items.extend([q.upper() for q in quoted])
            
            # If no codes found but part has substance, include it
            if not codes and not quoted and len(part) >= 3:
                # Extract the most substantial alphanumeric sequence
                substantial = re.findall(r'\b[A-Z0-9][\w\-]{2,}\b', part.upper())
                if substantial:
                    items.extend(substantial[:1])  # Take only the first one per part
    
    # Pattern 2: Listed items with numbers (e.g., "1. item1 2. item2")
    numbered = re.findall(r'\d+[\.)]\s*([A-Z0-9\-_]{3,})', message.upper())
    if numbered:
        items.extend(numbered)
    
    # Pattern 3: Multiple product codes in sequence (only if 2-5 codes found)
    if not items:
        codes = re.findall(r'\b[A-Z0-9]{3,}[-_]?[A-Z0-9]{2,}\b', message.upper())
        if 2 <= len(codes) <= 5:  # Reasonable number of items
            items = codes
    
    # Clean and deduplicate while preserving order
    seen = set()
    cleaned_items = []
    for item in items:
        item = item.strip().upper()
        # Filter out common words that might be mistakenly captured
        if (len(item) >= 3 and 
            item not in seen and 
            item not in ['THE', 'AND', 'FOR', 'WHAT', 'PRICE', 'COST', 'ITEM', 'ABOUT', 'DETAILS']):
            seen.add(item)
            cleaned_items.append(item)
    
    return cleaned_items[:10]  # Limit to max 10 items to prevent abuse


def handle_multi_item_query(message, items):
    """Handle queries with multiple items by searching for each separately"""
    
    client = get_openai_client()
    all_results = {}
    files_found = set()
    
    # Search for each item individually
    for item in items:
        # Create a focused query for this specific item
        item_query = f"{item} price cost details specifications"
        search_results = search_documents(item_query, top_k=10)
        
        # Collect relevant results for this item
        item_context = []
        for r in search_results:
            if r['score'] > 0.15:  # Lower threshold for multi-item
                filename = r['metadata'].get('filename', 'Unknown')
                content = r['metadata'].get('content', '')
                files_found.add(filename)
                
                # Check if this result actually mentions the item
                if item.lower() in content.lower():
                    item_context.append(f"[{filename}]: {content}")
        
        all_results[item] = {
            'context': "\n".join(item_context[:3]),  # Top 3 results per item
            'found': len(item_context) > 0
        }
    
    # Build comprehensive response
    if not client:
        # Fallback without OpenAI
        response = f"Found information for {len([k for k,v in all_results.items() if v['found']])} out of {len(items)} items:\n\n"
        for item, data in all_results.items():
            if data['found']:
                response += f"**{item}:**\n{data['context'][:500]}...\n\n"
            else:
                response += f"**{item}:** No information found\n\n"
        return jsonify({'success': True, 'response': response})
    
    # Use OpenAI to format results nicely
    files_list = ", ".join(files_found)
    
    # Build context for all items
    context_for_prompt = ""
    for item, data in all_results.items():
        if data['found']:
            context_for_prompt += f"\n\n--- Information for {item} ---\n{data['context']}"
        else:
            context_for_prompt += f"\n\n--- Information for {item} ---\nNo specific information found."
    
    # Build list of items that were actually found
    found_items = [item for item, data in all_results.items() if data['found']]
    not_found_items = [item for item, data in all_results.items() if not data['found']]
    
    prompt = f"""You are K&B Scout AI. The user asked about these specific items: {', '.join(items)}

Context from files ({files_list}):
{context_for_prompt[:6000]}

Original question: {message}

CRITICAL INSTRUCTIONS:
1. ONLY respond about the {len(items)} items the user asked about: {', '.join(items)}
2. DO NOT make up or invent additional items
3. For each item found, extract ALL available information from the context (price, specifications, descriptions, etc.)
4. Use dynamic labels based on what data is actually available - NOT just "Price" and "Details"
5. Only show information that exists in the context - do not use placeholder text

Items found: {', '.join(found_items) if found_items else 'None'}
Items not found: {', '.join(not_found_items) if not_found_items else 'None'}

Format your response like this:

**[Item Code/Name]**
- [Key1]: [Value1]  (only include if data exists)
- [Key2]: [Value2]  (only include if data exists)
- Source: [filename]

For items not found, simply state: "**[Item Code]** - No information found in database"

Do NOT include any items beyond these {len(items)} items: {', '.join(items)}"""
    
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "You are K&B Scout AI. Extract and present ONLY the information available in the context. Do not use fixed templates or placeholder text. Present data naturally based on what fields actually exist. NEVER invent or add items beyond what the user specifically requested."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,  # Lower temperature for more precise responses
        max_tokens=1500
    )
    
    ai_response = response.choices[0].message.content
    
    # Add metadata
    found_count = len([v for v in all_results.values() if v['found']])
    ai_response += f"\n\n---\n*Searched for {len(items)} items. Found information for {found_count} items across {len(files_found)} file(s): {files_list}*"
    
    return jsonify({
        'success': True,
        'response': ai_response,
        'search_metadata': {
            'items_requested': items,
            'items_found': found_count,
            'files_searched': list(files_found)
        }
    })


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
    """Sync local document store with S3 bucket"""
    logger.info("Syncing with S3 bucket...")
    
    s3_files = list_s3_files()
    if not s3_files:
        logger.info("No files found in S3 or S3 not configured")
        return {'synced_files': 0, 'missing_files': 0}
    
    synced_count = 0
    missing_count = 0
    
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
                    'type': 'S3 File (needs reprocessing)',
                    'uploaded_at': s3_file['last_modified'].isoformat(),
                    's3_uploaded': True,
                    's3_key': s3_file['s3_key'],
                    'size': s3_file['size'],
                    'status': 'in_s3_only'  # Flag to indicate needs reprocessing
                }
                synced_count += 1
                logger.info(f"Found S3 file not in local store: {filename}")
        
        # Check for files in local store but not in S3
        for filename in list(document_store['files'].keys()):
            if not any(s3_file['filename'] == filename for s3_file in s3_files):
                file_info = document_store['files'][filename]
                if file_info.get('s3_uploaded', False):
                    # File should be in S3 but isn't
                    file_info['s3_missing'] = True
                    missing_count += 1
                    logger.warning(f"File in local store but missing from S3: {filename}")
    
    logger.info(f"S3 sync complete. Synced: {synced_count}, Missing: {missing_count}")
    return {'synced_files': synced_count, 'missing_files': missing_count}

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
        return jsonify({
            'success': True,
            'message': f'Sync complete. Found {result["synced_files"]} new files, {result["missing_files"]} missing files.',
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
    
    # Test S3 (optional)
    s3 = get_s3_client()
    if s3:
        logger.info("✅ S3 client available")
        # Sync with S3 on startup
        try:
            sync_result = sync_with_s3()
            if sync_result['synced_files'] > 0:
                logger.info(f"🔄 Found {sync_result['synced_files']} files in S3 not in local store")
            if sync_result['missing_files'] > 0:
                logger.warning(f"⚠️ {sync_result['missing_files']} local files missing from S3")
        except Exception as e:
            logger.error(f"S3 sync failed during startup: {e}")
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
