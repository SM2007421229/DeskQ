import os
import time
import threading
from datetime import datetime
import pandas as pd
from elasticsearch import Elasticsearch

try:
    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.preprocessing import normalize
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Optional imports for file reading
try:
    from docx import Document
except ImportError:
    Document = None

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

class FileManager:
    def __init__(self, config):
        self.config = config.get('file_knowledge', {})
        
        # Support both legacy and new config keys
        self.monitored_folders = self.config.get('dirs', self.config.get('monitored_folders', []))
        self.file_types = set(self.config.get('types', self.config.get('file_types', [])))
        self.reindex_interval = self.config.get('watch_interval', self.config.get('reindex_interval', 3600))

        # ES Config
        vector_config = config.get('vector', {})
        self.es_host = vector_config.get('host', "http://localhost:9200")
        self.es_user = vector_config.get('user')
        self.es_password = vector_config.get('password')
        self.index_name = "deskq_files"
        self.vector_dim = 1024
        
        print(f"Initializing Elasticsearch at {self.es_host}...")
        try:
            es_args = {
                "hosts": [self.es_host],
                "timeout": 30
            }
            
            if self.es_user and self.es_password:
                print(f"Using Basic Auth with user: {self.es_user}")
                # For elasticsearch<8.0.0, use http_auth
                es_args["http_auth"] = (self.es_user, self.es_password)
                
            self.es = Elasticsearch(**es_args)
            if not self.es.ping():
                print(f"Warning: Could not connect to Elasticsearch at {self.es_host}")
        except Exception as e:
            print(f"Error connecting to Elasticsearch: {e}")
            self.es = None

        if HAS_SKLEARN:
            # Use HashingVectorizer to create dense vectors via hashing + normalization
            # Note: This is a lightweight way to get embeddings without a heavy ML model
            self.vectorizer = HashingVectorizer(
                n_features=self.vector_dim,
                analyzer='char_wb',
                ngram_range=(2, 4),
                norm=None,
                alternate_sign=False
            )
        else:
            print("Warning: scikit-learn not found. Vector search will fail.")
            self.vectorizer = None

        self._init_es_index()
        
        self.is_indexing = False
        self.running = True
        
        # 启动自动索引线程 (可以通过配置禁用，方便测试)
        # Check root config for auto_index
        if config.get('auto_index', True):
            self.thread = threading.Thread(target=self._auto_reindex_loop, daemon=True)
            self.thread.start()

    def _init_es_index(self):
        if not self.es:
            return
            
        try:
            if not self.es.indices.exists(index=self.index_name):
                mapping = {
                    "mappings": {
                        "properties": {
                            "name": {"type": "text"},
                            "path": {"type": "keyword"},
                            "ext": {"type": "keyword"},
                            "embedding": {
                                "type": "dense_vector",
                                "dims": self.vector_dim
                            },
                            "created_at": {"type": "date"}
                        }
                    }
                }
                self.es.indices.create(index=self.index_name, body=mapping)
                print(f"Created Elasticsearch index: {self.index_name}")
        except Exception as e:
            print(f"Error creating index: {e}")

    def _vectorize(self, text):
        if not self.vectorizer:
            return [0.0] * self.vector_dim
        X = self.vectorizer.transform([text])
        # L2 normalization to ensure cosine similarity works as expected
        X = normalize(X.toarray(), norm='l2')
        return X[0].tolist()

    def _auto_reindex_loop(self):
        while self.running:
            self.index_files()
            time.sleep(self.reindex_interval)

    def index_files(self):
        if self.is_indexing:
            return
        self.is_indexing = True
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting file index on {self.monitored_folders}...")
        
        try:
            # 1. Scan current files on disk
            current_files = {} # path -> {name, ext, path}
            
            for folder in self.monitored_folders:
                if not os.path.exists(folder):
                    continue
                
                for root, dirs, files in os.walk(folder):
                    for file in files:
                        ext = os.path.splitext(file)[1].lower()
                        if ext in self.file_types:
                            full_path = os.path.join(root, file)
                            # Normalize path separators to avoid duplicates
                            full_path = os.path.normpath(full_path)
                            current_files[full_path] = {
                                "name": file,
                                "path": full_path,
                                "ext": ext
                            }
            
            # 2. Get existing files in Vector Store
            existing_ids = set()
            if self.es:
                try:
                    # Scan all documents to get current IDs
                    # Note: For very large indices, use helpers.scan
                    resp = self.es.search(
                        index=self.index_name,
                        body={
                            "query": {"match_all": {}},
                            "_source": ["path"],
                            "size": 10000 
                        }
                    )
                    for hit in resp['hits']['hits']:
                         # Use path as ID or retrieve from source
                         path = hit['_source'].get('path')
                         if path:
                             existing_ids.add(path)
                except Exception as e:
                    print(f"Error fetching existing files from ES: {e}")

            # 3. Calculate Diff
            files_on_disk_ids = set(current_files.keys())
            
            # Files to add (present on disk but not in DB)
            to_add_ids = list(files_on_disk_ids - existing_ids)
            
            # Files to delete (present in DB but not on disk)
            to_delete_ids = list(existing_ids - files_on_disk_ids)
            
            # 4. Apply Updates
            if to_delete_ids and self.es:
                print(f"Removing {len(to_delete_ids)} deleted files from index...")
                # Delete by query is safer or bulk delete
                # Since we use path as ID logic (but maybe not _id), let's delete by term
                # Ideally we should use path as _id to make this easier.
                # For now, let's assume we search and delete.
                # Actually, using delete_by_query is good.
                try:
                    # Batch delete might be tricky with terms query limit (65k).
                    # Loop chunks if needed.
                    chunk_size = 1000
                    for i in range(0, len(to_delete_ids), chunk_size):
                        chunk = to_delete_ids[i:i+chunk_size]
                        self.es.delete_by_query(
                            index=self.index_name,
                            body={"query": {"terms": {"path": chunk}}}
                        )
                except Exception as e:
                     print(f"Error deleting files: {e}")
                
            if to_add_ids and self.es:
                print(f"Adding {len(to_add_ids)} new files to index...")
                from elasticsearch import helpers
                
                actions = []
                for file_path in to_add_ids:
                    meta = current_files[file_path]
                    vector = self._vectorize(meta['name'])
                    
                    doc = {
                        "_index": self.index_name,
                        # Use path as _id to ensure uniqueness and easy access
                        # However, path might contain chars invalid for some ID contexts, but usually fine.
                        # Base64 encoding path as ID is safer, but raw path is readable.
                        # Let's rely on ES autogen ID or just search by path.
                        # Using path as ID allows upserts easily.
                        # Let's NOT force _id=path to avoid issues, just rely on path field uniqueness via logic.
                        # But wait, to check existing_ids efficiently, _id is best.
                        # Let's stick to the logic: we query all paths, so we know what to add/delete.
                        "_source": {
                            "name": meta['name'],
                            "path": meta['path'],
                            "ext": meta['ext'],
                            "embedding": vector,
                            "created_at": datetime.now().isoformat()
                        }
                    }
                    actions.append(doc)
                
                if actions:
                    try:
                        helpers.bulk(self.es, actions)
                    except Exception as e:
                         print(f"Error bulk indexing: {e}")
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] File index completed. Total files: {len(files_on_disk_ids)} (+{len(to_add_ids)}, -{len(to_delete_ids)})")
            
        except Exception as e:
            print(f"Error during file indexing: {e}")
        finally:
            self.is_indexing = False

    def query_files(self, query):
        """
        根据查询词语义搜索文件
        :param query: 查询关键词
        :return: 文件列表字符串描述
        """
        try:
            if isinstance(query, dict):
                query = query.get('query', '') or query.get('text', '') or ''
            
            if not query or (isinstance(query, str) and not query.strip()):
                if not self.es:
                    return "Elasticsearch 未连接。"
                
                # Return all files (or top 50 to avoid overflow)
                try:
                    resp = self.es.search(
                        index=self.index_name,
                        body={
                            "query": {"match_all": {}},
                            "size": 50,
                            "_source": ["name", "path"]
                        }
                    )
                    hits = resp['hits']['hits']
                    if not hits:
                        return "知识库为空。"
                    
                    output = ["知识库文件列表 (前 50 个):"]
                    for i, hit in enumerate(hits):
                        source = hit['_source']
                        output.append(f"- {source['name']} (Path: {source['path']})")
                    
                    total = resp['hits']['total']['value']
                    if total > 50:
                        output.append(f"... 等 {total - 50} 个更多文件")
                    return "\n".join(output)
                except Exception as e:
                    return f"查询出错: {e}"

            if not self.es:
                return "Elasticsearch 未连接。"

            # Vector Search
            query_vector = self._vectorize(query)
            
            # ES 7.x script_score query
            # Note: cosineSimilarity requires doc['embedding']
            script_query = {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, doc['embedding']) + 1.0",
                        "params": {"query_vector": query_vector}
                    }
                }
            }
            
            resp = self.es.search(
                index=self.index_name,
                body={
                    "query": script_query,
                    "size": 20,
                    "_source": ["name", "path"]
                }
            )
            
            hits = resp['hits']['hits']
            if not hits:
                return "未找到匹配的文件。"
            
            output = []
            for hit in hits:
                source = hit['_source']
                score = hit['_score']
                # Score is cosine+1, so range 0-2. Higher is better.
                output.append(f"{source['name']} (Path: {source['path']})")
                
            return "\n".join(output)
            
        except Exception as e:
            return f"搜索出错: {str(e)}"

    def _find_best_match(self, query):
        if not self.es:
            return None
            
        try:
            query_vector = self._vectorize(query)
            
            script_query = {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, doc['embedding']) + 1.0",
                        "params": {"query_vector": query_vector}
                    }
                }
            }
            
            resp = self.es.search(
                index=self.index_name,
                body={
                    "query": script_query,
                    "size": 1,
                    "_source": ["name", "path", "ext"]
                }
            )
            
            hits = resp['hits']['hits']
            if not hits:
                return None
            
            return hits[0]['_source']
            
        except Exception as e:
            print(f"Error finding best match: {e}")
            return None

    def open_file(self, file_name_query):
        try:
            target_meta = self._find_best_match(file_name_query)
            
            if not target_meta:
                return f"未找到名为 '{file_name_query}' 的文件。"
            
            target = target_meta['path']
            os.startfile(target)
            return f"已打开文件: {target}"
            
        except Exception as e:
            return f"打开文件失败: {str(e)}"

    def read_file_content(self, file_name_query):
        try:
            target_meta = self._find_best_match(file_name_query)
            
            if not target_meta:
                return f"未找到名为 '{file_name_query}' 的文件。"
            
            target_path = target_meta['path']
            ext = target_meta['ext']
            
            content = ""
            if ext == '.docx':
                if not Document:
                    return "错误: 未安装 python-docx 库，无法读取 .docx 文件。"
                doc = Document(target_path)
                content = "\n".join([p.text for p in doc.paragraphs])
            
            elif ext == '.pdf':
                if not PdfReader:
                    return "错误: 未安装 pypdf 库，无法读取 .pdf 文件。"
                reader = PdfReader(target_path)
                for page in reader.pages[:10]: # 限制前10页以防过大
                    text = page.extract_text()
                    if text:
                        content += text + "\n"
            
            elif ext in ['.xlsx', '.xls']:
                if not pd:
                    return "错误: 未安装 pandas/openpyxl 库，无法读取 Excel 文件。"
                
                # 尝试根据扩展名自动推断引擎，如果失败则尝试显式指定
                try:
                    dfs = pd.read_excel(target_path, sheet_name=None)
                except Exception:
                    # 如果自动检测失败（无论是ValueError还是BadZipFile），尝试遍历所有可能的引擎
                    try:
                        # 先尝试 xlrd (兼容 .xls 即使后缀是 .xlsx)
                        dfs = pd.read_excel(target_path, sheet_name=None, engine='xlrd')
                    except Exception:
                        try:
                            # 再尝试 openpyxl
                            dfs = pd.read_excel(target_path, sheet_name=None, engine='openpyxl')
                        except Exception as e:
                            # 如果都失败，抛出最后的异常
                            raise e
                
                parts = []
                for sheet_name, df in dfs.items():
                    parts.append(f"Sheet: {sheet_name}")
                    # 转换为 Markdown 表格格式，限制行数以防过长
                    parts.append(df.head(50).to_markdown(index=False))
                    parts.append("\n")
                content = "\n".join(parts)
            
            elif ext == '.doc':
                return "提示: 暂不支持直接读取 .doc (旧版Word) 格式，请另存为 .docx 格式后再试。"
            
            elif ext in ['.txt', '.md', '.py', '.json', '.log', '.xml', '.ini']:
                with open(target_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
            
            else:
                return f"文件类型 {ext} 不支持内容读取。"
                
            # 截断过长内容
            if len(content) > 3000:
                content = content[:3000] + "\n...(内容过长已截断)"
                
            return f"文件 '{target_path}' 的内容:\n{content}"
                
        except Exception as e:
            return f"读取文件出错: {str(e)}"

    def stop(self):
        self.running = False
