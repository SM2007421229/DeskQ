import requests
import json


# 构建问题
def build_messages(prompt, question, history=None):
    system_prompt = f"{prompt}"
    messages = [
        {"role": "system", "content": system_prompt}
    ]
    if history:
        for h in history:
            messages.append({"role": "user", "content": h['question']})
            messages.append({"role": "assistant", "content": h['answer']})
    messages.append({"role": "user", "content": question})
    return messages


# 流式请求
def ask_stream(prompt, question, history=None, apikey=None, apiUrl=None):
    messages = build_messages(prompt, question, history)
    payload = {
        "model": "deepseek-v3-250324",
        "messages": messages,
        "stream": True
    }
    headers = {
        "Authorization": f"Bearer {apikey}",
        "Content-Type": "application/json"
    }
    with requests.post(apiUrl, json=payload, headers=headers, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            if line.startswith(b"data: "):
                line = line[6:]
            try:
                data = json.loads(line)
                # 兼容不同流式格式
                if "choices" in data and data["choices"]:
                    delta = data["choices"][0].get("delta", {}).get("content")
                    if delta:
                        yield delta
            except Exception:
                continue