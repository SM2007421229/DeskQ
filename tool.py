import os
import traceback
import json
import math
import statistics
from langchain_core.tools import StructuredTool
import file_manager

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import uuid

# Set matplotlib backend to Agg for headless environments
plt.switch_backend('Agg')
# Support Chinese characters in plots
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS'] 
plt.rcParams['axes.unicode_minus'] = False

# --- Calculation Tools ---

def _resolve_absolute_path(rel_path: str) -> str:
    """Resolve relative path to absolute using monitored folders from config.json"""
    if os.path.isabs(rel_path) and os.path.exists(rel_path):
        return rel_path

    try:
        # config.json is in the same directory as this script
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            monitored_folders = config.get('file_knowledge', {}).get('monitored_folders', [])
            
            for folder in monitored_folders:
                abs_path_candidate = os.path.join(folder, rel_path)
                if os.path.exists(abs_path_candidate):
                    return os.path.normpath(abs_path_candidate)
            
            return rel_path # Fallback
    except Exception as e:
        print(f"Error resolving path: {e}")
        return rel_path

def _read_excel_column(file_path: str, column_name: str, filter_conditions: str = None) -> list:
    """
    Helper to read a specific column from Excel with optional filtering.
    filter_conditions: JSON string, e.g. '{"Region": "East China", "Year": 2025}'
    """
    try:
        # Resolve path
        file_path = _resolve_absolute_path(file_path)
        
        if not os.path.exists(file_path):
            return []
        
        df = pd.read_excel(file_path)
        
        # Normalize column names for easier matching
        # Map lowercased/stripped name to actual name
        col_map = {str(c).strip().lower(): c for c in df.columns}
        
        # Apply filters if present
        if filter_conditions:
            try:
                filters = json.loads(filter_conditions)
                for key, value in filters.items():
                    # Find the actual column name for the filter key
                    key_norm = str(key).strip().lower()
                    if key_norm in col_map:
                        actual_key = col_map[key_norm]
                        # Apply filter
                        # Handle type conversion loosely?
                        # For now, strict equality. 
                        # Note: value from JSON might be int/str, df content might be different.
                        # We convert df column to string for comparison if needed?
                        # Or just let pandas handle it.
                        # Let's try direct comparison first, but be aware of type mismatches.
                        
                        # Better approach for mixed types: convert both to string for comparison if direct fails?
                        # But numerical comparison is important (2025 vs "2025").
                        # Let's try standard pandas filtering.
                        df = df[df[actual_key] == value]
                    else:
                        print(f"Warning: Filter column '{key}' not found in Excel.")
            except json.JSONDecodeError:
                print(f"Error parsing filter_conditions: {filter_conditions}")
            except Exception as e:
                print(f"Error applying filters: {e}")

        # Find target column
        target_col = None
        target_norm = str(column_name).strip().lower()
        if target_norm in col_map:
            target_col = col_map[target_norm]
        
        if target_col:
            return df[target_col].tolist()
        else:
            return []
    except Exception as e:
        print(f"Error reading excel column {column_name}: {e}")
        return []

def get_column_values(file_path: str, column_name: str) -> str:
    """
    获取 Excel 文件中指定列的所有去重值（用于后续的批量统计）。
    返回去重后的值列表。
    """
    try:
        file_path = _resolve_absolute_path(file_path)
        if not os.path.exists(file_path):
            return "File not found"
        
        df = pd.read_excel(file_path)
        col_map = {str(c).strip().lower(): c for c in df.columns}
        target_norm = str(column_name).strip().lower()
        
        if target_norm in col_map:
            target_col = col_map[target_norm]
            # Get unique values, drop NA
            values = df[target_col].dropna().unique().tolist()
            # Convert to appropriate types (str/int/float)
            return json.dumps(values, default=str)
        else:
            return "Column not found"
    except Exception as e:
        return f"Error: {str(e)}"

def calc_sum_desc(file_path: str, column_name: str, filter_conditions: str = None) -> str:
    """
    计算 Excel 文件中指定列的数值总和。
    参数 filter_conditions (可选): JSON 字符串格式的筛选条件。
    **新特性**：支持在筛选条件中传入列表（如 `{"大区": ["华东", "华北"]}`），工具将批量计算并返回每个值的统计结果字典。
    """
    try:
        filters = json.loads(filter_conditions) if filter_conditions else {}
        
        # Check for list values in filters (Batch Mode)
        list_keys = [k for k, v in filters.items() if isinstance(v, list)]
        
        if list_keys:
            # Batch Mode Implementation (Optimized: Read once)
            file_path = _resolve_absolute_path(file_path)
            if not os.path.exists(file_path):
                return json.dumps({})
            
            df = pd.read_excel(file_path)
            col_map = {str(c).strip().lower(): c for c in df.columns}
            
            # Apply fixed filters (non-list)
            for k, v in filters.items():
                if k not in list_keys:
                    k_norm = str(k).strip().lower()
                    if k_norm in col_map:
                        df = df[df[col_map[k_norm]] == v]
            
            # Identify the batch dimension (use the first list key found)
            target_key = list_keys[0]
            target_vals = filters[target_key]
            target_col_name = col_map.get(str(target_key).strip().lower())
            
            value_col_name = col_map.get(str(column_name).strip().lower())
            
            if not value_col_name:
                return "Target column not found"
            
            results = {}
            if target_col_name:
                for val in target_vals:
                    # Filter for specific value
                    sub_df = df[df[target_col_name] == val]
                    
                    # Extract numbers safely
                    nums = pd.to_numeric(sub_df[value_col_name], errors='coerce').dropna().tolist()
                    
                    if not nums:
                        results[val] = 0
                    else:
                        total = sum(nums)
                        # Simple rounding for display
                        results[val] = round(total, 2)
            
            return json.dumps(results)

        # Original Single Mode Logic
        numerical_sequence = _read_excel_column(file_path, column_name, filter_conditions)
        nums = [float(x) for x in numerical_sequence if x is not None and str(x).replace('.','',1).isdigit()]
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

def calc_mean_desc(file_path: str, column_name: str, filter_conditions: str = None) -> str:
    """
    计算 Excel 文件中指定列的算术平均值。
    参数 filter_conditions (可选): JSON 字符串格式的筛选条件。
    **新特性**：支持在筛选条件中传入列表，工具将批量计算并返回每个值的统计结果字典。
    """
    try:
        filters = json.loads(filter_conditions) if filter_conditions else {}
        
        # Check for list values in filters (Batch Mode)
        list_keys = [k for k, v in filters.items() if isinstance(v, list)]
        
        if list_keys:
            # Batch Mode Implementation
            file_path = _resolve_absolute_path(file_path)
            if not os.path.exists(file_path):
                return json.dumps({})
            
            df = pd.read_excel(file_path)
            col_map = {str(c).strip().lower(): c for c in df.columns}
            
            # Apply fixed filters
            for k, v in filters.items():
                if k not in list_keys:
                    k_norm = str(k).strip().lower()
                    if k_norm in col_map:
                        df = df[df[col_map[k_norm]] == v]
            
            target_key = list_keys[0]
            target_vals = filters[target_key]
            target_col_name = col_map.get(str(target_key).strip().lower())
            value_col_name = col_map.get(str(column_name).strip().lower())
            
            if not value_col_name:
                return "Target column not found"
            
            results = {}
            if target_col_name:
                for val in target_vals:
                    sub_df = df[df[target_col_name] == val]
                    nums = pd.to_numeric(sub_df[value_col_name], errors='coerce').dropna().tolist()
                    if not nums:
                        results[val] = 0.0
                    else:
                        mean_val = statistics.mean(nums)
                        results[val] = round(mean_val, 2)
            return json.dumps(results)

        # Original Single Mode Logic
        numerical_sequence = _read_excel_column(file_path, column_name, filter_conditions)
        nums = [float(x) for x in numerical_sequence if x is not None and str(x).replace('.','',1).isdigit()]
        if not nums:
            return json.dumps({"mean": 0.0})
        mean_val = statistics.mean(nums)
        return json.dumps({"mean": mean_val})
    except Exception as e:
        return f"Error: {str(e)}"

def calc_percentile_desc(file_path: str, column_name: str, percentile: float, filter_conditions: str = None) -> str:
    """
    计算 Excel 文件中指定列的百分位数（如 95%）。
    参数 filter_conditions (可选): JSON 字符串格式的筛选条件，例如 '{"大区": "华东", "季度": "2025Q1"}'
    """
    try:
        numerical_sequence = _read_excel_column(file_path, column_name, filter_conditions)
        nums = sorted([float(x) for x in numerical_sequence if x is not None and str(x).replace('.','',1).isdigit()])
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

def calc_growth_rate_desc(data: str) -> str:
    """
    计算一组数据的逐期增长率。
    参数 data: JSON 字符串格式的数值列表，例如 '[100, 120, 150]'。
    """
    try:
        values = json.loads(data) if isinstance(data, str) else data
        
        if not isinstance(values, list):
            return "Error: Input must be a list"
            
        if len(values) < 2:
            return json.dumps([])

        rates = []
        for i in range(1, len(values)):
            try:
                prev = float(values[i-1])
                curr = float(values[i])
                
                if prev == 0:
                    rates.append(0.0)
                else:
                    rate = ((curr - prev) / prev) * 100
                    rates.append(round(rate, 2))
            except (ValueError, TypeError):
                rates.append(0.0)
                
        return json.dumps(rates)
    except Exception as e:
        return f"Error: {str(e)}"

def calc_ratio_desc(file_path: str, numerator_col: str, denominator_col: str, filter_conditions: str = None) -> str:
    """
    计算 Excel 中两列数据的逐行比例（分子列 / 分母列）。
    参数 filter_conditions (可选): JSON 字符串格式的筛选条件，例如 '{"大区": "华东", "季度": "2025Q1"}'
    """
    try:
        nums = _read_excel_column(file_path, numerator_col, filter_conditions)
        dens = _read_excel_column(file_path, denominator_col, filter_conditions)
        
        # Align lengths
        min_len = min(len(nums), len(dens))
        nums = nums[:min_len]
        dens = dens[:min_len]
        
        ratios = []
        for n, d in zip(nums, dens):
            try:
                n_val = float(n) if n is not None else 0
                d_val = float(d) if d is not None else 0
                if d_val == 0:
                    ratios.append(None)
                else:
                    ratios.append(n_val / d_val)
            except:
                ratios.append(None)
                
        return json.dumps({"ratio": ratios})
    except Exception as e:
        return f"Error: {str(e)}"

def calc_round_desc(value: float, decimals: int) -> str:
    """
    对数值进行四舍五入。
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
    计算复合年均增长率 (CAGR)。
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

def calc_std_desc(file_path: str, column_name: str, filter_conditions: str = None) -> str:
    """
    计算 Excel 文件中指定列的标准差。
    参数 filter_conditions (可选): JSON 字符串格式的筛选条件，例如 '{"大区": "华东", "季度": "2025Q1"}'
    """
    try:
        numerical_sequence = _read_excel_column(file_path, column_name, filter_conditions)
        nums = [float(x) for x in numerical_sequence if x is not None and str(x).replace('.','',1).isdigit()]
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

    def query_files(query: str = "", filter_key: str = None) -> str:
        """
        Retrieves relevant document content or file schemas.
        - For PDF/Word/Text: Returns relevant text snippets for QA.
        - For Excel: Returns the FILE PATH, TABLE SCHEMA (columns), and SAMPLE DATA only.
          DO NOT use this tool to get full Excel data rows.
          Use calculation tools (calc_*) with the file path to perform data analysis.
        """
        fm = file_manager.instance
        if not fm:
            return "File Manager is not initialized."

        try:
            limit = 50 # Default limit for text snippets

            query_text = query
            if isinstance(query, dict):
                query_text = query.get('query', '') or query.get('text', '') or ''
            
            if not query_text or not query_text.strip():
                return "请输入有效的查询内容。"
                
            if not fm.es:
                return "Elasticsearch 未连接。"

            # Construct Filter Conditions (applied to both Vector and Keyword search)
            filter_conditions = []
            if filter_key:
                keywords = filter_key.split()
                for k in keywords:
                    # Each keyword must match either content or file_name
                    filter_conditions.append({
                        "multi_match": {
                            "query": k,
                            "fields": ["content", "file_name"],
                            # "type": "phrase" # Relaxed to default (best_fields) for better recall
                        }
                    })

            # 1. Vector Search (Semantic)
            # Use input_type="QUERY" for searching
            query_vector = fm.vector_model.encode([query_text], input_type="QUERY")[0]
            
            # Check for zero vector (model failure or empty input)
            is_zero_vector = all(abs(v) < 1e-6 for v in query_vector)
            
            vec_resp = None
            if not is_zero_vector and fm.es.indices.exists(index=fm.index_name):
                try:
                    # Base vector query (match_all if no filter)
                    vector_base_query = {"match_all": {}}
                    if filter_conditions:
                        vector_base_query = {"bool": {"must": filter_conditions}}

                    vector_query = {
                        "script_score": {
                            "query": vector_base_query,
                            "script": {
                                "source": "cosineSimilarity(params.query_vector, doc['content_vector']) + 1.0",
                                "params": {"query_vector": query_vector}
                            }
                        }
                    }
                    
                    # Path A: Vector Candidates
                    vec_resp = fm.es.search(
                        index=fm.index_name,
                        body={"query": vector_query, "size": limit, "_source": ["file_name", "path", "content", "chunk_id", "chunk_seq", "parent_id", "ancestor_ids", "level", "type"]}
                    )
                except Exception as e:
                    print(f"Vector search failed: {e}")
                    vec_resp = None
            else:
                print("Warning: Generated query vector is all zeros. Skipping vector search.")

            # 2. BM25 Search (Keyword) - Content and Filename
            bool_query = {
                "should": [
                    {"match": {"content": {"query": query_text, "boost": 1.0}}},
                    {"match": {"file_name": {"query": query_text, "boost": 2.0}}} # Filename match has higher boost
                ]
            }
            
            if filter_conditions:
                bool_query["must"] = filter_conditions

            keyword_query = {"bool": bool_query}
            
            # Path B: Keyword Candidates
            kw_resp = fm.es.search(
                index=fm.index_name,
                body={"query": keyword_query, "size": limit, "_source": ["file_name", "path", "content", "chunk_id", "chunk_seq", "parent_id", "ancestor_ids", "level", "type"]}
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
            top_results = sorted_results[:limit]
            
            if not top_results:
                return "未找到相关文件。"
                
            # Format Output
            output = []
            
            output.append(f"共找到 {len(top_results)} 条相关数据：\n")
            
            for i, res in enumerate(top_results):
                src = res['source']
                name = src['file_name']
                path = src.get('path', name)
                content = src['content'] # Return full content
                
                chunk_id = src.get('chunk_id')
                
                # Format: [Index] ID: ... Path: ... Content: ...
                output.append(f"[{i+1}] ID: {chunk_id} 路径: {path}\n内容: {content}\n")

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
            # First try exact match on path if query looks like path
            if os.path.isabs(file_name_query) and os.path.exists(file_name_query):
                os.startfile(file_name_query)
                return f"已打开: {file_name_query}"

            # Search by filename
            resp = fm.es.search(
                index=fm.index_name,
                body={
                    "query": {"match": {"file_name": file_name_query}},
                    "size": 1,
                    # Remove collapse to avoid errors if field type is issue
                    "_source": ["path"]
                }
            )
            hits = resp['hits']['hits']
            if hits:
                path = hits[0]['_source']['path']
                # Resolve relative path to absolute
                abs_path = _resolve_absolute_path(path)
                
                if os.path.exists(abs_path):
                    os.startfile(abs_path)
                    return f"已打开: {abs_path}"
                else:
                     return f"文件不存在: {abs_path}"
            return "未找到文件"
        except Exception as e:
            traceback.print_exc()
            return f"打开文件失败: {str(e)}"

    def fetch_section_content(chunk_ids: list) -> str:
        """
        Fetch full content for specific sections/headings by their chunk IDs.
        Retrieves all descendant text (children, sub-sections) for the given chunks.
        """
        fm = file_manager.instance
        if not fm or not fm.es:
            return "ES not initialized."
            
        try:
            if not chunk_ids:
                return "No chunk IDs provided."
                
            # Search for chunks where ancestor_ids contain any of the provided chunk_ids
            # Also include the chunks themselves
            # User request: "query all parent_id=chunk_id text" -> we interpret this as subtree retrieval
            
            should_clauses = []
            for cid in chunk_ids:
                # Match chunks that have this ID as ancestor
                should_clauses.append({"term": {"ancestor_ids": cid}})
                # Match chunks that have this ID as parent (redundant if ancestor_ids logic is correct but safe)
                should_clauses.append({"term": {"parent_id": cid}})
                # Match the chunk itself
                should_clauses.append({"term": {"chunk_id": cid}})
            
            query = {
                "bool": {
                    "should": should_clauses,
                    "minimum_should_match": 1
                }
            }
            
            resp = fm.es.search(
                index=fm.index_name,
                body={
                    "query": query,
                    "size": 1000, # Large size for full section
                    "sort": [
                        {"path": "asc"}, # Group by file
                        {"chunk_seq": "asc"} # Then by sequence
                    ],
                    "_source": ["content", "chunk_id", "type", "file_name"]
                }
            )
            
            hits = resp['hits']['hits']
            if not hits:
                return "No content found for provided IDs."
            
            # Assemble content
            output = []
            current_file = ""
            
            for h in hits:
                src = h['_source']
                fname = src.get('file_name', 'Unknown')
                content = src.get('content', '')
                
                if fname != current_file:
                    output.append(f"\n--- File: {fname} ---")
                    current_file = fname
                    
                output.append(content)
            
            return "\n".join(output)
            
        except Exception as e:
            traceback.print_exc()
            return f"Error fetching section content: {str(e)}"

    def plot_chart(chart_type: str, data: str, x_label: str = "", y_label: str = "", title: str = "") -> str:
        """
        生成 ECharts 图表配置。
        
        :param chart_type: 图表类型，支持 'bar' (柱状图), 'line' (折线图), 'pie' (饼图)
        :param data: JSON 字符串格式的数据。
                     对于 bar/line/pie: {"Category1": Value1, "Category2": Value2, ...}
                     或者 {"labels": ["A", "B"], "values": [10, 20]}
        :param x_label: X轴标签
        :param y_label: Y轴标签
        :param title: 图表标题
        :return: 包含 ECharts 配置的 JSON 字符串
        """
        try:
            data_dict = json.loads(data)
            
            labels = []
            values = []

            # Normalize data format to labels and values
            if isinstance(data_dict, list):
                # Handle list input e.g. [{"name": "A", "value": 10}, ...] or [{"A": 10}, {"B": 20}]
                if not data_dict:
                     return "Error: Empty data list."
                
                first_item = data_dict[0]
                if isinstance(first_item, dict):
                    keys = list(first_item.keys())
                    # Heuristic: look for common value keys
                    val_key = next((k for k in keys if str(k).lower() in ['value', 'values', 'count', 'amount', 'num', 'number', 'sales', 'total', 'score', 'price']), None)
                    # Heuristic: look for common label keys
                    lbl_key = next((k for k in keys if str(k).lower() in ['name', 'label', 'category', 'type', 'x', 'date', 'year', 'quarter', 'month', 'day', 'region', 'city']), None)
                    
                    if val_key and lbl_key:
                        for item in data_dict:
                            labels.append(item.get(lbl_key, ''))
                            values.append(item.get(val_key, 0))
                    elif len(keys) == 1: 
                        # Assume [{"A": 10}, {"B": 20}] format
                         for item in data_dict:
                             k = list(item.keys())[0]
                             labels.append(k)
                             values.append(item[k])
                    else:
                        # Fallback: Use first key as label, last key as value (often ordered dicts)
                        # Or if we have 2 keys, assume 0 is label, 1 is value
                        if len(keys) >= 2:
                            lbl_key = keys[0]
                            val_key = keys[-1] # Assume last is value (e.g. Name, ..., Value)
                            for item in data_dict:
                                labels.append(item.get(lbl_key, ''))
                                values.append(item.get(val_key, 0))
                        else:
                             return f"Error: Could not infer labels and values from data list. Keys found: {keys}"
                else:
                    # List of values
                    labels = [str(i+1) for i in range(len(data_dict))]
                    values = data_dict
            
            elif isinstance(data_dict, dict):
                if "labels" in data_dict and "values" in data_dict:
                    labels = data_dict["labels"]
                    values = data_dict["values"]
                else:
                    labels = list(data_dict.keys())
                    values = list(data_dict.values())
            else:
                return "Error: Data must be a JSON object or list."
            
            # Ensure values are numeric
            # values = [float(v) for v in values] # ECharts handles numbers/strings usually
            
            option = {
                "title": {
                    "text": title,
                    "left": "center",
                    "textStyle": {"color": "#f1f5f9"} # Default to light color for dark mode
                },
                "tooltip": {"trigger": "item" if chart_type == 'pie' else "axis"},
                "legend": {
                    "orient": "vertical",
                    "left": "left",
                    "textStyle": {"color": "#94a3b8"}
                },
                "series": []
            }
            
            if chart_type in ['bar', 'line']:
                option["xAxis"] = {
                    "type": "category",
                    "data": labels,
                    "name": x_label,
                    "axisLabel": {"color": "#94a3b8"},
                    "axisLine": {"lineStyle": {"color": "#475569"}},
                    "nameTextStyle": {"color": "#94a3b8"}
                }
                option["yAxis"] = {
                    "type": "value",
                    "name": y_label,
                    "axisLabel": {"color": "#94a3b8"},
                    "splitLine": {"lineStyle": {"color": "#334155"}},
                    "nameTextStyle": {"color": "#94a3b8"}
                }
                option["series"].append({
                    "data": values,
                    "type": chart_type,
                    "itemStyle": {"color": "#6366f1"} # Indigo 500
                })
            elif chart_type == 'pie':
                pie_data = [{"name": str(l), "value": v} for l, v in zip(labels, values)]
                option["series"].append({
                    "type": "pie",
                    "radius": "50%",
                    "data": pie_data,
                    "emphasis": {
                        "itemStyle": {
                            "shadowBlur": 10,
                            "shadowOffsetX": 0,
                            "shadowColor": "rgba(0, 0, 0, 0.5)"
                        }
                    },
                    "label": {"color": "#f1f5f9"}
                })
            else:
                return "Error: Unsupported chart type. Use 'bar', 'line', or 'pie'."
            
            # Return the option wrapped in markdown
            json_str = json.dumps(option, ensure_ascii=False, indent=2)
            return f"```echarts\n{json_str}\n```"
            
        except Exception as e:
            traceback.print_exc()
            return f"Error generating chart: {str(e)}"

    return [
        StructuredTool.from_function(
            func=query_files,
            name="query_files",
            description="Search for files in the knowledge base. MUST be used when user asks for specific data, reports, 'what files do I have', or when a calculation requires external data. If query is empty, lists all files. param 'limit' (default 50): When the user's question involves a large scope of data statistics (e.g., 'all data', 'annual total', 'summary of 2025'), set this parameter to a larger value (e.g., 500 or 1000, max 2000) to ensure data completeness. The tool output will indicate if there are more matches than the limit; if so, you MUST re-query with a larger limit for accurate aggregation. param 'filter_key' (optional): Use this to enforce strict filtering. If the user specifies explicit conditions (e.g., 'East Region', '2025', 'Product A'), put these keywords here (space-separated). Only files containing ALL these keywords will be retrieved. This avoids irrelevant data (e.g., 'South Region' data appearing in 'East Region' query) from polluting the results."
        ),
        StructuredTool.from_function(
            func=fetch_section_content,
            name="fetch_section_content",
            description="Fetch full content for specific sections/headings (PDF/Doc) by their chunk_ids. Use this when you need to read the details of a specific section returned by query_files."
        ),
        StructuredTool.from_function(
            func=open_file,
            name="open_file",
            description="Open a specific file in the local OS (e.g. launch PDF viewer). DO NOT use this tool to read/analyze content. Only use when user explicitly asks to 'open' or 'launch' a file."
        ),
        StructuredTool.from_function(
            func=get_column_values,
            name="get_column_values",
            description="获取Excel文件某一列的所有去重值。参数：file_path(文件路径), column_name(列名)。使用场景：当用户询问'有哪些大区'、'统计各产品销售额'时，先调用此工具获取所有类别（如所有大区名），然后将这些类别列表作为filter_conditions的值传给统计工具进行批量计算。"
        ),
        StructuredTool.from_function(
            func=calc_sum_desc,
            name="calc_sum_desc",
            description="计算Excel文件中指定列的数值总和。参数：file_path(文件路径), column_name(列名), filter_conditions(可选，JSON格式筛选条件)。新特性：支持在筛选条件中传入列表（如 {'大区': ['华东', '华北']}），工具将批量计算并返回每个值的统计结果字典。"
        ),
        StructuredTool.from_function(
            func=calc_mean_desc,
            name="calc_mean_desc",
            description="计算Excel文件中指定列的算术平均值。参数：file_path(文件路径), column_name(列名), filter_conditions(可选，JSON格式筛选条件)。新特性：支持在筛选条件中传入列表，工具将批量计算并返回每个值的统计结果字典。"
        ),
        StructuredTool.from_function(
            func=calc_percentile_desc,
            name="calc_percentile_desc",
            description="计算Excel文件中指定列的百分位数。参数：file_path(文件路径), column_name(列名), percentile(0-100), filter_conditions(可选，JSON格式筛选条件)。"
        ),
        StructuredTool.from_function(
            func=calc_growth_rate_desc,
            name="calc_growth_rate_desc",
            description="计算一组数据的逐期增长率。参数 data: JSON 字符串格式的数值列表（如 '[100, 120, 150]'）。返回对应的增长率列表（如 '[20.0, 25.0]'）。"
        ),
        StructuredTool.from_function(
            func=calc_ratio_desc,
            name="calc_ratio_desc",
            description="计算Excel中两列数据的逐行比例（分子列/分母列）。参数：file_path(文件路径), numerator_col(分子列名), denominator_col(分母列名), filter_conditions(可选，JSON格式筛选条件)。"
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
            description="计算Excel文件中指定列的标准差。参数：file_path(文件路径), column_name(列名), filter_conditions(可选，JSON格式筛选条件)。"
        ),
        StructuredTool.from_function(
            func=plot_chart,
            name="plot_chart",
            description="生成ECharts图表配置。参数：chart_type('bar'|'line'|'pie'), data(JSON字符串), x_label, y_label, title。当用户要求'画图'、'可视化'时使用此工具。注意：工具会返回一段JSON配置代码，你必须将这段代码完整包含在你的最终回答中（包裹在```echarts ... ```代码块内）。严禁在代码块后添加任何解释性文字（如“注：...”）。"
        )
    ]
