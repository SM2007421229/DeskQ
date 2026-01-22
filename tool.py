import os
import traceback
from langchain_core.tools import StructuredTool
import file_manager

def get_tools():
    """
    Generate tools.
    Dependencies are handled internally by importing file_manager.
    """

    def query_files(query: str = "") -> str:
        """
        Hybrid Retrieval: Vector Search + BM25 + File Name Match
        With Re-ranking logic.
        """
        fm = file_manager.instance
        if not fm:
            return "File Manager is not initialized."

        try:
            query_text = query
            if isinstance(query, dict):
                query_text = query.get('query', '') or query.get('text', '') or ''
            
            if not query_text or not query_text.strip():
                return "请输入有效的查询内容。"
                
            if not fm.es:
                return "Elasticsearch 未连接。"

            # 1. Vector Search (Semantic)
            # Use input_type="QUERY" for searching
            query_vector = fm.vector_model.encode([query_text], input_type="QUERY")[0]
            
            # Check for zero vector (model failure or empty input)
            is_zero_vector = all(abs(v) < 1e-6 for v in query_vector)
            
            vec_resp = None
            if not is_zero_vector and fm.es.indices.exists(index=fm.index_name):
                try:
                    vector_query = {
                        "script_score": {
                            "query": {"match_all": {}},
                            "script": {
                                "source": "cosineSimilarity(params.query_vector, doc['content_vector']) + 1.0",
                                "params": {"query_vector": query_vector}
                            }
                        }
                    }
                    
                    # Path A: Vector Candidates
                    vec_resp = fm.es.search(
                        index=fm.index_name,
                        body={"query": vector_query, "size": 50, "_source": ["file_name", "path", "content"]}
                    )
                except Exception as e:
                    print(f"Vector search failed: {e}")
                    vec_resp = None
            else:
                print("Warning: Generated query vector is all zeros. Skipping vector search.")

            # 2. BM25 Search (Keyword) - Content and Filename
            keyword_query = {
                "bool": {
                    "should": [
                        {"match": {"content": {"query": query_text, "boost": 1.0}}},
                        {"match": {"file_name": {"query": query_text, "boost": 2.0}}} # Filename match has higher boost
                    ]
                }
            }
            
            # Path B: Keyword Candidates
            kw_resp = fm.es.search(
                index=fm.index_name,
                body={"query": keyword_query, "size": 50, "_source": ["file_name", "path", "content"]}
            )
            
            # Fuse Results (Weighted Sum of normalized scores)
            # This is a simplified re-ranking
            hits_map = {} # path_chunk_id -> {score, hit}
            
            # Helper to normalize scores (simple min-max per result set)
            def normalize_hits(hits, weight):
                if not hits: return
                max_score = max(h['_score'] for h in hits) if hits else 1.0
                if max_score == 0: max_score = 1.0
                
                for h in hits:
                    key = h['_id']
                    score = (h['_score'] / max_score) * weight
                    if key not in hits_map:
                        hits_map[key] = {"score": 0, "source": h['_source']}
                    hits_map[key]["score"] += score

            if vec_resp and 'hits' in vec_resp and 'hits' in vec_resp['hits']:
                normalize_hits(vec_resp['hits']['hits'], weight=0.6) # Vector weight
            
            if kw_resp and 'hits' in kw_resp and 'hits' in kw_resp['hits']:
                normalize_hits(kw_resp['hits']['hits'], weight=0.4) # Keyword weight
            
            # Sort by fused score
            sorted_results = sorted(hits_map.values(), key=lambda x: x['score'], reverse=True)
            top_results = sorted_results[:50] # Increase limit for statistical tasks
            
            if not top_results:
                return "未找到相关文件。"
                
            # Format Output
            # Return full content for LLM analysis without file-level deduplication
            output = []
            output.append(f"共找到 {len(top_results)} 条相关数据：\n")
            
            for i, res in enumerate(top_results):
                src = res['source']
                path = src['path']
                name = src['file_name']
                content = src['content'] # Return full content
                
                # Format: [Index] File: ... Content: ...
                output.append(f"[{i+1}] 文件: {name}\n内容: {content}\n")
            
            return "\n".join(output)

        except Exception as e:
            traceback.print_exc()
            return f"搜索出错: {str(e)}"

    def open_file(file_name_query: str) -> str:
        """
        Open a specific file by name.
        """
        fm = file_manager.instance
        if not fm:
            return "File Manager is not initialized."
            
        # Use simple keyword search for open file
        if not fm.es: return "ES 未连接"
        try:
            resp = fm.es.search(
                index=fm.index_name,
                body={
                    "query": {"match": {"file_name": file_name_query}},
                    "size": 1,
                    "collapse": {"field": "path"} 
                }
            )
            hits = resp['hits']['hits']
            if hits:
                path = hits[0]['_source']['path']
                os.startfile(path)
                return f"已打开: {path}"
            return "未找到文件"
        except Exception as e:
            return str(e)

    return [
        StructuredTool.from_function(
            func=query_files,
            name="query_files",
            description="Search for files in the knowledge base. Useful when user asks 'what files do I have' or looks for a document. If query is empty, lists all files."
        ),
        StructuredTool.from_function(
            func=open_file,
            name="open_file",
            description="Open a specific file by name. Useful when user asks to 'open' a file."
        )
    ]
