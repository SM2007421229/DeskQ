import os
import time
import threading
from datetime import datetime
import pandas as pd
from vector_store import SimpleVectorDB

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

        print("Initializing Vector Database (SimpleVectorDB)...")
        self.collection = SimpleVectorDB(collection_name="file_index")
        
        self.is_indexing = False
        self.running = True
        
        # 启动自动索引线程 (可以通过配置禁用，方便测试)
        # Check root config for auto_index
        if config.get('auto_index', True):
            self.thread = threading.Thread(target=self._auto_reindex_loop, daemon=True)
            self.thread.start()

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
            existing_data = self.collection.get()
            existing_ids = set(existing_data["ids"])
            
            # 3. Calculate Diff
            files_on_disk_ids = set(current_files.keys())
            
            # Files to add (present on disk but not in DB)
            to_add_ids = list(files_on_disk_ids - existing_ids)
            
            # Files to delete (present in DB but not on disk)
            to_delete_ids = list(existing_ids - files_on_disk_ids)
            
            # 4. Apply Updates
            if to_delete_ids:
                print(f"Removing {len(to_delete_ids)} deleted files from index...")
                self.collection.delete(ids=to_delete_ids)
                
            if to_add_ids:
                print(f"Adding {len(to_add_ids)} new files to index...")
                # SimpleVectorDB handles batching internally if needed, but we pass all at once here
                batch_ids = to_add_ids
                batch_documents = [current_files[id]["name"] for id in batch_ids]
                batch_metadatas = [current_files[id] for id in batch_ids]
                
                self.collection.add(
                    documents=batch_documents,
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )
            
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
                # Return all files (or top 50 to avoid overflow)
                all_data = self.collection.get()
                if not all_data['ids']:
                     return "知识库为空。"
                
                output = ["知识库文件列表 (前 50 个):"]
                for i, meta in enumerate(all_data['metadatas']):
                    if i >= 50:
                        output.append(f"... 等 {len(all_data['ids']) - 50} 个更多文件")
                        break
                    output.append(f"- {meta['name']} (Path: {meta['path']})")
                return "\n".join(output)

            results = self.collection.query(
                query_texts=[query],
                n_results=20
            )
            
            if not results['ids'] or not results['ids'][0]:
                return "未找到匹配的文件。"
            
            output = []
            ids = results['ids'][0]
            metadatas = results['metadatas'][0]
            distances = results['distances'][0]
            
            for i, (id, meta, dist) in enumerate(zip(ids, metadatas, distances)):
                # dist is cosine distance (1 - similarity). Lower is better.
                output.append(f"{meta['name']} (Path: {meta['path']})")
                
            return "\n".join(output)
            
        except Exception as e:
            return f"搜索出错: {str(e)}"

    def _find_best_match(self, query):
        results = self.collection.query(
            query_texts=[query],
            n_results=1
        )
        
        if not results['ids'] or not results['ids'][0]:
            return None
            
        return results['metadatas'][0][0]

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
