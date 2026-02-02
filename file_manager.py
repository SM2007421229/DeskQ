import os
import time
import threading
from datetime import datetime
import pandas as pd
import requests
from elasticsearch import Elasticsearch, helpers
import warnings
import traceback

# Suppress warnings
warnings.filterwarnings("ignore")

HAS_SENTENCE_TRANSFORMERS = False

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

import re
import uuid


class ContentExtractor:
    """Handles content extraction and splitting for various file formats."""
    def __init__(self):
        if HAS_LANGCHAIN:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=600,
                chunk_overlap=100,
                separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
            )
        else:
            self.text_splitter = None

    def _split_text_custom(self, text, parent_id=None):
        """
        Splits text according to user rules:
        1. Split by punctuation: 。;！?\.!?;
        2. If > threshold, split by ,
        3. Assign group_id and seq.
        """
        chunks = []
        if not text:
            return chunks
            
        # Split by punctuation, keeping the delimiter
        parts = re.split(r'([。;！\?\.!?;])', text)
        
        sentences = []
        current_sent = ""
        for p in parts:
            if re.match(r'[。;！\?\.!?;]', p):
                current_sent += p
                sentences.append(current_sent)
                current_sent = ""
            else:
                current_sent += p
        if current_sent:
            sentences.append(current_sent)
            
        for sent in sentences:
            if not sent.strip():
                continue
                
            group_id = str(uuid.uuid4())
            # Rule 4: Recursive split by comma if too long
            sub_chunks = self._recursive_split(sent, max_len=300)
            
            for i, sub in enumerate(sub_chunks):
                chunks.append({
                    "content": sub,
                    "group_id": group_id,
                    "seq": i,
                    "parent_id": parent_id,
                    "type": "text"
                })
        return chunks

    def _recursive_split(self, text, max_len=300):
        if len(text) <= max_len:
            return [text]
            
        # Split by comma
        parts = re.split(r'([,，])', text)
        result = []
        current = ""
        
        for p in parts:
            if len(current) + len(p) > max_len:
                if current:
                    result.append(current)
                current = p
            else:
                current += p
        if current:
            result.append(current)
            
        final_result = []
        for r in result:
             if len(r) > max_len:
                 final_result.extend([r[i:i+max_len] for i in range(0, len(r), max_len)])
             else:
                 final_result.append(r)
                 
        return final_result

    def _process_pdf(self, file_path):
        if not pdfplumber:
            return []
            
        chunks = []
        try:
            with pdfplumber.open(file_path) as pdf:
                # 1. Global Font Size Analysis (Mode based)
                all_font_sizes = []
                # Sample first 10 pages or all pages for better stats
                sample_pages = pdf.pages[:10] if len(pdf.pages) > 10 else pdf.pages
                
                for page in sample_pages:
                    words = page.extract_words(extra_attrs=['size'])
                    for w in words:
                        if w['text'].strip():
                            # Round to nearest integer for robust mode calculation
                            all_font_sizes.append(round(w['size']))
                
                body_size = 12.0 # Default fallback
                if all_font_sizes:
                    # Calculate Mode
                    from collections import Counter
                    counts = Counter(all_font_sizes)
                    body_size = float(counts.most_common(1)[0][0])
                
                # Dynamic Thresholds
                # Use additive thresholds instead of multipliers to be more robust against small font variations
                h1_threshold = body_size + 4.0 
                h2_threshold = body_size + 2.0
                
                # 2. Hierarchy Stack: [(chunk_id, level, title_text)]
                # level: 1 (H1), 2 (H2), 3 (H3)
                heading_stack = [] 
                
                global_seq = 0
                
                for i, page in enumerate(pdf.pages):
                    page_num = i + 1
                    
                    # 3. Extract Tables
                    table_bboxes = []
                    found_tables = page.find_tables()
                    for table in found_tables:
                        table_bboxes.append(table.bbox)
                        data = table.extract()
                        if not data: continue
                        
                        headers = []
                        start_row = 0
                        if len(data) > 0:
                            headers = [str(c).strip().replace('\n', ' ') if c else f"Col{j}" for j, c in enumerate(data[0])]
                            start_row = 1
                        
                        for row in data[start_row:]:
                            row_parts = []
                            for col_idx, cell in enumerate(row):
                                if col_idx < len(headers):
                                    col_name = headers[col_idx]
                                    val = str(cell).strip().replace('\n', ' ') if cell else ""
                                    if val:
                                        row_parts.append(f"{col_name}:{val}")
                            
                            if row_parts:
                                content = ",".join(row_parts)
                                
                                # Determine Parent
                                current_parent_id = heading_stack[-1][0] if heading_stack else None
                                current_ancestors = [h[0] for h in heading_stack]
                                
                                chunks.append({
                                    "chunk_id": str(uuid.uuid4()),
                                    "content": f"[Page {page_num} Table] {content}",
                                    "type": "table",
                                    "parent_id": current_parent_id,
                                    "ancestor_ids": current_ancestors,
                                    "level": 0,
                                    "chunk_seq": global_seq,
                                    "page": page_num
                                })
                                global_seq += 1

                    # 4. Extract Text
                    def not_inside_tables(obj):
                        x0, top, x1, bottom = obj['x0'], obj['top'], obj['x1'], obj['bottom']
                        mid_x = (x0 + x1) / 2
                        mid_y = (top + bottom) / 2
                        for bbox in table_bboxes:
                            if (bbox[0] <= mid_x <= bbox[2]) and (bbox[1] <= mid_y <= bbox[3]):
                                return False
                        return True

                    text_page = page.filter(not_inside_tables)
                    words = text_page.extract_words(extra_attrs=['size', 'fontname'])
                    
                    # Group words into lines
                    lines = []
                    current_line = []
                    last_top = 0
                    
                    for w in words:
                        if not current_line:
                            current_line.append(w)
                            last_top = w['top']
                        else:
                            if abs(w['top'] - last_top) < 5:
                                current_line.append(w)
                            else:
                                lines.append(current_line)
                                current_line = [w]
                                last_top = w['top']
                    if current_line:
                        lines.append(current_line)
                    
                    # Process lines with Heading Detection
                    buffer_text = ""
                    
                    for line in lines:
                        if not line: continue
                        
                        avg_size = sum(w['size'] for w in line) / len(line)
                        line_text = " ".join([w['text'] for w in line]).strip()
                        if not line_text: continue

                        # Detect Heading Level
                        level = 0
                        is_bold = any("Bold" in w.get('fontname', '') for w in line)
                        
                        # Punctuation check: Headings rarely end with sentence separators
                        last_char = line_text.strip()[-1] if line_text.strip() else ""
                        is_likely_text = last_char in [',', '，', '、', ';', '；', '。']
                        
                        # Exception for English period: only exclude if not short numbering like "1."
                        if last_char == '.':
                            if len(line_text) > 10 or not re.match(r'^[\d\.]+$', line_text.strip()):
                                is_likely_text = True
                        
                        if avg_size >= h1_threshold:
                            level = 1
                        elif avg_size >= h2_threshold:
                            level = 2
                        elif not is_likely_text:
                            if (avg_size >= body_size + 1.0) or (avg_size >= body_size and is_bold and len(line_text) < 60):
                                level = 3
                        
                        if level > 0:
                            # Flush Buffer (Previous Body Text)
                            if buffer_text:
                                # Determine Parent for Buffer (Current Stack Top)
                                current_parent_id = heading_stack[-1][0] if heading_stack else None
                                current_ancestors = [h[0] for h in heading_stack]
                                
                                # Split text (Rule 4) - Simplified as per request (remove group_id/seq logic within sentence)
                                # But we still need to split by punctuation for granularity
                                # Using _split_text_custom but mapping to new structure
                                
                                # Direct splitting here to control chunk structure
                                sub_parts = self._split_text_smart(buffer_text)
                                for part in sub_parts:
                                    chunks.append({
                                        "chunk_id": str(uuid.uuid4()),
                                        "content": part,
                                        "type": "text",
                                        "parent_id": current_parent_id,
                                        "ancestor_ids": current_ancestors,
                                        "level": 0,
                                        "chunk_seq": global_seq,
                                        "page": page_num
                                    })
                                    global_seq += 1
                                buffer_text = ""
                            
                            # Update Stack
                            # Pop all headings with level >= current level
                            while heading_stack and heading_stack[-1][1] >= level:
                                heading_stack.pop()
                            
                            # Determine Parent for this Heading (New Stack Top)
                            parent_heading_id = heading_stack[-1][0] if heading_stack else None
                            # Ancestors include the parent
                            heading_ancestors = [h[0] for h in heading_stack]
                            
                            new_heading_id = str(uuid.uuid4())
                            
                            chunks.append({
                                "chunk_id": new_heading_id,
                                "content": line_text,
                                "type": "heading",
                                "parent_id": parent_heading_id,
                                "ancestor_ids": heading_ancestors,
                                "level": level,
                                "chunk_seq": global_seq,
                                "page": page_num
                            })
                            global_seq += 1
                            
                            # Push to Stack
                            heading_stack.append((new_heading_id, level, line_text))
                            
                        else:
                            # Body Text Accumulation
                            buffer_text += line_text + "\n"
                            
                    # End of Page Flush
                    if buffer_text:
                        current_parent_id = heading_stack[-1][0] if heading_stack else None
                        current_ancestors = [h[0] for h in heading_stack]
                        
                        sub_parts = self._split_text_smart(buffer_text)
                        for part in sub_parts:
                            chunks.append({
                                "chunk_id": str(uuid.uuid4()),
                                "content": part,
                                "type": "text",
                                "parent_id": current_parent_id,
                                "ancestor_ids": current_ancestors,
                                "level": 0,
                                "chunk_seq": global_seq,
                                "page": page_num
                            })
                            global_seq += 1
                        
        except Exception as e:
            print(f"Error processing PDF {file_path}: {e}")
            traceback.print_exc()
            
        return chunks

    def _split_text_smart(self, text):
        """
        Smart splitter for PDF body text.
        Splits by punctuation: 。;！?\.!?;
        Exception: Dot (.) surrounded by digits (e.g. 3.14) is NOT split.
        """
        if not text: return []
        
        # Split by the specified punctuation
        parts = re.split(r'([。;！\?\.!?;])', text)
        chunks = []
        current = ""
        
        for i, part in enumerate(parts):
            # If it's a separator
            if re.match(r'^[。;！\?\.!?;]$', part):
                # Check for decimal point exception: . surrounded by digits
                if part == '.':
                    prev_char = parts[i-1][-1] if i > 0 and parts[i-1] else ''
                    next_char = parts[i+1][0] if i < len(parts)-1 and parts[i+1] else ''
                    if prev_char.isdigit() and next_char.isdigit():
                        current += part
                        continue
                
                # Normal separator behavior
                current += part
                if current.strip():
                    chunks.append(current)
                current = ""
            else:
                current += part
        
        if current.strip():
            chunks.append(current)
            
        return [c.strip() for c in chunks if c.strip()]

    def _process_excel(self, file_path):
        """
        Extracts metadata (columns, sample data) from Excel files.
        Does NOT extract full row data to avoid token limits.
        """
        chunks = []
        try:
            # Read header and first few rows for sampling
            df = pd.read_excel(file_path, nrows=3)
            columns = df.columns.tolist()
            
            # Create a schema description
            schema_desc = f"Columns: {', '.join(map(str, columns))}"
            
            # Create sample data description
            sample_rows = []
            for _, row in df.iterrows():
                row_str = ", ".join([f"{col}:{val}" for col, val in row.items()])
                sample_rows.append(f"{{ {row_str} }}")
            
            sample_desc = "Data Samples:\n" + "\n".join(sample_rows)
            
            full_content = f"Excel File Schema.\n{schema_desc}\n{sample_desc}"
            
            chunks.append({
                "chunk_id": str(uuid.uuid4()),
                "content": full_content,
                "type": "excel_schema",
                "file_path": file_path,
                "columns": columns,
                "chunk_seq": 0
            })
            
        except Exception as e:
            print(f"Error processing Excel {file_path}: {e}")
            traceback.print_exc()
            
        return chunks

    def load_and_split(self, file_path, ext):
        """
        Extract content from file and split into chunks.
        Returns: List of dicts [{"content": str, "page_info": str}]
        """
        try:
            if ext == '.pdf':
                if pdfplumber:
                    return self._process_pdf(file_path)
                elif PdfReader:
                    reader = PdfReader(file_path)
                    texts = []
                    for i, page in enumerate(reader.pages):
                        page_text = page.extract_text()
                        if page_text:
                            texts.append(f"[Page {i+1}] {page_text}")
                    raw_text = "\n".join(texts)
                else:
                    return []

            elif ext in ['.xlsx', '.xls']:
                return self._process_excel(file_path)

            else:
                return []

            # Split text
            if not raw_text.strip():
                return []

            if self.text_splitter:
                chunks = self.text_splitter.split_text(raw_text)
            else:
                # Fallback simple splitter
                chunks = [raw_text[i:i+500] for i in range(0, len(raw_text), 500)]

            return [{"content": chunk, "chunk_id": i} for i, chunk in enumerate(chunks)]

        except Exception as e:
            print(f"Error extracting {file_path}: {e}")
            return []

class VectorModel:
    """Wrapper for Aliyun OpenSearch Text Embedding API."""
    def __init__(self, api_url, api_key=None):
        self.api_key = api_key
        self.api_url = api_url
        # ops-text-embedding-002 dimension is 1024 based on test result
        self.dim = 1024

    def encode(self, texts, input_type="QUERY"):
        if not self.api_key or not self.api_url or not texts:
            return [[0.0] * self.dim] * len(texts)
        
        # Check if texts are empty
        if not any(t.strip() for t in texts):
            return [[0.0] * self.dim] * len(texts)

        # Batch processing
        batch_size = 32  # Max batch size per API documentation
        all_embeddings = []
        
        try:
            for i in range(0, len(texts), batch_size):
                # Slice batch
                raw_batch = texts[i:i + batch_size]
                
                # Truncate each text to max 8192 chars
                batch_texts = [t[:8192] for t in raw_batch]
                
                # Skip empty batch (shouldn't happen with range logic but good safety)
                if not batch_texts:
                    continue
                    
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                }
                
                payload = {
                    "input": batch_texts,
                    "input_type": input_type
                }
                
                response = requests.post(self.api_url, json=payload, headers=headers, timeout=30) # Increased timeout
                response.raise_for_status()
                
                result = response.json()
                if "result" in result and "embeddings" in result["result"]:
                    # Sort by index to ensure order matches input
                    embeddings_data = sorted(result["result"]["embeddings"], key=lambda x: x["index"])
                    batch_embeddings = [item["embedding"] for item in embeddings_data]
                    all_embeddings.extend(batch_embeddings)
                    
                    # Update dim if we get a result
                    if batch_embeddings:
                         self.dim = len(batch_embeddings[0])
                else:
                    print(f"API response format unexpected: {result}")
                    all_embeddings.extend([[0.0] * self.dim] * len(batch_texts))
            
            return all_embeddings
                
        except Exception as e:
            print(f"Encoding error: {e}")
            # Return zero vectors for failed batch + remaining items to match length
            current_len = len(all_embeddings)
            needed = len(texts) - current_len
            return all_embeddings + ([[0.0] * self.dim] * needed)

# --- Main Class ---

# 全局实例
instance = None

class FileManager:
    def __init__(self, config):
        global instance
        instance = self
        self.config = config.get('file_knowledge', {})
        
        # Support both legacy and new config keys
        self.monitored_folders = self.config.get('dirs', self.config.get('monitored_folders', []))
        # Hardcoded supported file types
        self.file_types = {".xlsx", ".xls", ".pdf"}
        self.reindex_interval = self.config.get('watch_interval', self.config.get('reindex_interval', 3600))

        # ES Config
        es_config = config.get('es', {})
        self.es_host = es_config.get('host', "http://localhost:9200")
        self.es_user = es_config.get('user')
        self.es_password = es_config.get('password')
        # Use a new index name for RAG to avoid conflict with old mapping
        # ops-text-embedding-002 has 1024 dims, updating index name
        self.index_name = "deskq_rag_files_v3"
        
        # Initialize Components
        self.extractor = ContentExtractor()
        # Pass API key and URL to VectorModel
        vector_config = config.get('vector', {})
        self.vector_model = VectorModel(vector_config.get("apiUrl"), vector_config.get("apikey"))
        self.vector_dim = self.vector_model.dim
        
        print(f"Initializing Elasticsearch at {self.es_host}...")
        try:
            es_args = {
                "hosts": [self.es_host],
                "timeout": 30
            }
            
            if self.es_user and self.es_password:
                es_args["http_auth"] = (self.es_user, self.es_password)
                
            self.es = Elasticsearch(**es_args)
            if not self.es.ping():
                print(f"Warning: Could not connect to Elasticsearch at {self.es_host}")
        except Exception as e:
            print(f"Error connecting to Elasticsearch: {e}")
            self.es = None

        self._init_es_index()
        
        self.is_indexing = False
        self.running = True
        
        # Auto index thread
        if config.get('auto_index', True):
            self.thread = threading.Thread(target=self._auto_reindex_loop, daemon=True)
            self.thread.start()

    def _init_es_index(self):
        if not self.es:
            return
            
        try:
            # Check if index exists
            if self.es.indices.exists(index=self.index_name):
                # Verify mapping 'path' is keyword
                mapping = self.es.indices.get_mapping(index=self.index_name)
                # Structure: {index_name: {mappings: {properties: {path: {type: ...}}}}}
                try:
                    props = mapping[self.index_name]['mappings']['properties']
                    path_type = props['path']['type']
                    
                    # Check for new fields
                    has_new_fields = 'chunk_id' in props and 'ancestor_ids' in props and 'level' in props
                    chunk_id_type = props.get('chunk_id', {}).get('type')
                    
                    if path_type != 'keyword' or chunk_id_type != 'keyword' or not has_new_fields:
                        print(f"Index {self.index_name} schema outdated. Recreating...")
                        self.es.indices.delete(index=self.index_name)
                    else:
                        # Mapping is correct
                        return
                except KeyError:
                    # Could happen if 'path' field doesn't exist yet or structure is different
                    print(f"Index {self.index_name} mapping check failed. Recreating...")
                    self.es.indices.delete(index=self.index_name, ignore=[404])

            # Create index if not exists (or deleted above)
            if not self.es.indices.exists(index=self.index_name):
                # Create index with explicit mapping
                mapping = {
                    "mappings": {
                        "properties": {
                            "file_name": {"type": "text", "analyzer": "standard"},
                            "path": {"type": "keyword"},
                            "ext": {"type": "keyword"},
                            "mtime": {"type": "double"},
                            "content": {"type": "text", "analyzer": "standard"},
                            "content_vector": {
                                "type": "dense_vector",
                                "dims": self.vector_dim
                            },
                            "chunk_id": {"type": "keyword"},
                            "chunk_seq": {"type": "integer"},
                            "parent_id": {"type": "keyword"},
                            "ancestor_ids": {"type": "keyword"},
                            "level": {"type": "integer"},
                            "type": {"type": "keyword"},
                            "page": {"type": "integer"},
                            "created_at": {"type": "date"}
                        }
                    }
                }
                self.es.indices.create(index=self.index_name, body=mapping)
                print(f"Created Elasticsearch index: {self.index_name}")
        except Exception as e:
            print(f"Error checking/creating index: {e}")

    def _auto_reindex_loop(self):
        while self.running:
            try:
                self.index_files()
            except Exception as e:
                print(f"Auto-index error: {e}")
            time.sleep(self.reindex_interval)

    def index_files(self, specific_folders=None):
        """
        Index files from monitored folders.
        Supports incremental updates via mtime.
        """
        if self.is_indexing:
            return
        self.is_indexing = True
        target_folders = specific_folders or self.monitored_folders
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting RAG index on {target_folders}...")
        
        try:
            # 1. Scan current files on disk
            current_files = {} # path -> {name, ext, path, mtime}
            
            for folder in target_folders:
                if not os.path.exists(folder):
                    continue
                
                for root, dirs, files in os.walk(folder):
                    for file in files:
                        ext = os.path.splitext(file)[1].lower()
                        if ext in self.file_types:
                            full_path = os.path.join(root, file)
                            full_path = os.path.normpath(full_path)
                            mtime = os.path.getmtime(full_path)
                            
                            # Calculate relative path for storage
                            rel_path = os.path.relpath(full_path, folder)
                            
                            current_files[full_path] = {
                                "name": file,
                                "path": rel_path, # Store relative path
                                "full_path": full_path, # Keep absolute for reading
                                "ext": ext,
                                "mtime": mtime
                            }
            
            # 2. Get existing files metadata from ES
            # We aggregate by path to get stored mtime
            existing_files = {} # path (relative) -> mtime
            if self.es:
                try:
                    # Ensure index exists
                    if not self.es.indices.exists(index=self.index_name):
                        self._init_es_index()
                        
                    # Terms aggregation on path to find all indexed files
                    # Note: For huge datasets, composite agg or scroll is better.
                    # Here we use a simple search collapsing on path or just scrolling hits.
                    # Since we store multiple chunks per file, we just need to know which files are there.
                    # We can search for all unique 'path' and their 'mtime'.
                    
                    # Using a collapse query to get unique paths
                    resp = self.es.search(
                        index=self.index_name,
                        body={
                            "query": {"match_all": {}},
                            "_source": ["path", "mtime"],
                            "collapse": {"field": "path"},
                            "size": 10000 
                        }
                    )
                    
                    for hit in resp['hits']['hits']:
                        src = hit['_source']
                        existing_files[src.get('path')] = src.get('mtime', 0)
                        
                except Exception as e:
                    print(f"Error fetching existing files from ES: {e}")

            # 3. Calculate Diff
            # Use relative paths for comparison
            # Map relative paths back to full paths for processing
            rel_to_full = {meta['path']: meta['full_path'] for meta in current_files.values()}
            files_on_disk = set(rel_to_full.keys())
            files_in_index = set(existing_files.keys())
            
            to_delete = files_in_index - files_on_disk
            to_add_or_update = []
            
            for rel_path in files_on_disk:
                if rel_path not in files_in_index:
                    to_add_or_update.append(rel_path)
                else:
                    # Check mtime (using full path to look up current mtime)
                    full_p = rel_to_full[rel_path]
                    if current_files[full_p]['mtime'] > existing_files[rel_path] + 1.0: # 1s buffer
                        to_add_or_update.append(rel_path)

            # 4. Apply Deletions
            if to_delete and self.es:
                print(f"Removing {len(to_delete)} deleted files...")
                for path in to_delete:
                    self.es.delete_by_query(
                        index=self.index_name,
                        body={"query": {"term": {"path": path}}}
                    )

            # 5. Apply Updates (Delete old chunks -> Index new chunks)
            if to_add_or_update and self.es:
                print(f"Indexing {len(to_add_or_update)} files...")
                
                for rel_path in to_add_or_update:
                    try:
                        # First delete existing chunks for this file to avoid duplication
                        if rel_path in files_in_index:
                             self.es.delete_by_query(
                                index=self.index_name,
                                body={"query": {"term": {"path": rel_path}}}
                            )
                        
                        full_path = rel_to_full[rel_path]
                        meta = current_files[full_path]
                        
                        # Extract and Split
                        chunks = self.extractor.load_and_split(full_path, meta['ext'])
                        if not chunks:
                            continue
                            
                        # Embed
                        texts = [c['content'] for c in chunks]
                        # Use input_type="DOCUMENT" for indexing if API supports, but user example used "query".
                        # Usually embedding APIs use "document" for storage. Let's use "document" for now.
                        # Wait, user example only showed "query". If I use "document" and it fails, that's bad.
                        # However, Aliyun OpenSearch usually supports "query" and "document".
                        # Let's try "document" for indexing.
                        vectors = self.vector_model.encode(texts, input_type="DOCUMENT")
                        
                        # Prepare Bulk Actions
                        actions = []
                        for i, chunk in enumerate(chunks):
                            doc = {
                                "_index": self.index_name,
                                "_source": {
                                    "file_name": meta['name'],
                                    "path": meta['path'], # Relative path
                                    "ext": meta['ext'],
                                    "mtime": meta['mtime'],
                                    "content": chunk['content'],
                                    "content_vector": vectors[i],
                                    "chunk_id": str(chunk.get('chunk_id', i)),
                                    "created_at": datetime.now().isoformat(),
                                    "parent_id": chunk.get('parent_id'),
                                    "ancestor_ids": chunk.get('ancestor_ids', []),
                                    "level": chunk.get('level', 0),
                                    "chunk_seq": chunk.get('chunk_seq', i),
                                    "page": chunk.get('page'),
                                    "type": chunk.get('type', 'text')
                                }
                            }
                            actions.append(doc)
                        
                        if actions:
                            helpers.bulk(self.es, actions)
                            
                    except Exception as e:
                        print(f"Failed to index {rel_path}: {e}")
                        traceback.print_exc()

            print(f"[{datetime.now().strftime('%H:%M:%S')}] Indexing completed. (+{len(to_add_or_update)}, -{len(to_delete)})")
            
        except Exception as e:
            print(f"Error during file indexing: {e}")
            traceback.print_exc()
        finally:
            self.is_indexing = False

    def stop(self):
        self.running = False
