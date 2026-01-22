import os
import traceback
import json
import math
import statistics
from langchain_core.tools import StructuredTool
import file_manager

# --- Calculation Tools ---

def calc_sum_desc(numerical_sequence: list) -> str:
    """
    接收单参数“数值序列”，对给定序列求和并返回结果与精度说明
    """
    try:
        nums = [float(x) for x in numerical_sequence if x is not None]
        if not nums:
            return json.dumps({"sum": 0, "precision": 0})
        
        total = sum(nums)
        
        # Calculate precision (max decimal places)
        max_prec = 0
        for x in nums:
            s = str(x)
            if '.' in s:
                p = len(s) - s.index('.') - 1
                max_prec = max(max_prec, p)
        
        # Round total to avoid floating point artifacts
        if max_prec > 0:
            total = round(total, max_prec)
        
        return json.dumps({"sum": total, "precision": max_prec})
    except Exception as e:
        return f"Error: {str(e)}"

def calc_mean_desc(numerical_sequence: list) -> str:
    """
    同样只需一个“数值序列”，自动过滤空值后算算术平均
    """
    try:
        nums = [float(x) for x in numerical_sequence if x is not None]
        if not nums:
            return json.dumps({"mean": 0.0})
        mean_val = statistics.mean(nums)
        return json.dumps({"mean": mean_val})
    except Exception as e:
        return f"Error: {str(e)}"

def calc_percentile_desc(numerical_sequence: list, percentile: float) -> str:
    """
    需两个参数，先给“数值序列”再给出 0–100 的百分位
    """
    try:
        nums = sorted([float(x) for x in numerical_sequence if x is not None])
        if not nums:
            return json.dumps({"percentile": 0.0})
        
        # Linear interpolation
        k = (len(nums) - 1) * (float(percentile) / 100.0)
        f = math.floor(k)
        c = math.ceil(k)
        
        if f == c:
            result = nums[int(k)]
        else:
            d0 = nums[int(f)] * (c - k)
            d1 = nums[int(c)] * (k - f)
            result = d0 + d1
            
        return json.dumps({"percentile": result})
    except Exception as e:
        return f"Error: {str(e)}"

def calc_growth_rate_desc(previous_value: float, current_value: float) -> str:
    """
    接收“上期值”和“本期值”两个数值，输出同比增长率
    """
    try:
        prev = float(previous_value)
        curr = float(current_value)
        if prev == 0:
            return json.dumps({"growth_rate": 0.0}) # Handle divide by zero
        rate = ((curr - prev) / prev) * 100
        return json.dumps({"growth_rate": rate})
    except Exception as e:
        return f"Error: {str(e)}"

def calc_ratio_desc(numerator_sequence: list, denominator_sequence: list) -> str:
    """
    也取两个参数，即“分子序列”与“分母序列”，逐元素算比例
    """
    try:
        nums = [float(x) if x is not None else 0 for x in numerator_sequence]
        dens = [float(x) if x is not None else 1 for x in denominator_sequence]
        
        ratios = []
        for n, d in zip(nums, dens):
            if d == 0:
                ratios.append(None)
            else:
                ratios.append(n / d)
                
        return json.dumps({"ratio": ratios})
    except Exception as e:
        return f"Error: {str(e)}"

def calc_round_desc(value: float, decimals: int) -> str:
    """
    需要“数值”和“保留小数位”两个参数，按指定位数四舍五入并以字符串形式返回
    """
    try:
        val = float(value)
        dec = int(decimals)
        # Use format to get string representation
        fmt = f"{{:.{dec}f}}"
        rounded_str = fmt.format(val)
        return json.dumps({"rounded": rounded_str})
    except Exception as e:
        return f"Error: {str(e)}"

def calc_cagr_desc(start_value: float, end_value: float, years: float) -> str:
    """
    要求三个参数，依次为“起始值、结束值、年数”，计算复合年均增长率
    """
    try:
        start = float(start_value)
        end = float(end_value)
        yrs = float(years)
        
        if start == 0 or yrs == 0:
             return json.dumps({"cagr": 0.0})
        
        cagr = ((end / start) ** (1 / yrs) - 1) * 100
        return json.dumps({"cagr": cagr})
    except Exception as e:
        return f"Error: {str(e)}"

def calc_std_desc(numerical_sequence: list) -> str:
    """
    仅接收一个“数值序列”，计算样本标准差并保留四位小数
    """
    try:
        nums = [float(x) for x in numerical_sequence if x is not None]
        if len(nums) < 2:
            return json.dumps({"std": 0.0})
            
        std_val = statistics.stdev(nums)
        return json.dumps({"std": round(std_val, 4)})
    except Exception as e:
        return f"Error: {str(e)}"

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
            description="Search for files in the knowledge base. MUST be used when user asks for specific data, reports, 'what files do I have', or when a calculation requires external data. If query is empty, lists all files."
        ),
        StructuredTool.from_function(
            func=open_file,
            name="open_file",
            description="Open a specific file by name. Useful when user asks to 'open' a file."
        ),
        StructuredTool.from_function(
            func=calc_sum_desc,
            name="calc_sum_desc",
            description="接收单参数“数值序列”，对给定序列求和并返回结果与精度说明"
        ),
        StructuredTool.from_function(
            func=calc_mean_desc,
            name="calc_mean_desc",
            description="同样只需一个“数值序列”，自动过滤空值后算算术平均"
        ),
        StructuredTool.from_function(
            func=calc_percentile_desc,
            name="calc_percentile_desc",
            description="需两个参数，先给“数值序列”再给出 0–100 的百分位"
        ),
        StructuredTool.from_function(
            func=calc_growth_rate_desc,
            name="calc_growth_rate_desc",
            description="接收“上期值”和“本期值”两个数值，输出同比增长率"
        ),
        StructuredTool.from_function(
            func=calc_ratio_desc,
            name="calc_ratio_desc",
            description="也取两个参数，即“分子序列”与“分母序列”，逐元素算比例"
        ),
        StructuredTool.from_function(
            func=calc_round_desc,
            name="calc_round_desc",
            description="需要“数值”和“保留小数位”两个参数，按指定位数四舍五入并以字符串形式返回"
        ),
        StructuredTool.from_function(
            func=calc_cagr_desc,
            name="calc_cagr_desc",
            description="计算复合年均增长率 (CAGR)。当用户问及“年均增长率”、“复合增长率”或“CAGR”时，必须调用此工具。参数：start_value(起始值), end_value(结束值), years(年数，如果是日期差请先转换为年，如9个月=0.75年)。"
        ),
        StructuredTool.from_function(
            func=calc_std_desc,
            name="calc_std_desc",
            description="计算样本标准差。当需要分析数据波动、稳定性或标准差时调用。参数：numerical_sequence(数值序列)。"
        )
    ]
