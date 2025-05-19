import os
import json5
import faiss
import numpy as np
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from config import DATA_DIR, ARK_API_KEY, ARK_BASE_URL

client = OpenAI(api_key=ARK_API_KEY, base_url=ARK_BASE_URL)

def load_json(file_name):
    try:
        file_path = os.path.join(DATA_DIR, file_name)
        with open(file_path, "r", encoding="utf-8") as f:
            return json5.load(f)
    except Exception as e:
        print(f"❌ 加载失败: {file_name} 错误: {e}")
        return None

def load_schema():
    schema_path = os.path.join(DATA_DIR, "data_schema.json5")
    return load_json(schema_path)

def get_relevant_files(user_question, schema):
    prompt = "\n".join([f"{k}: {v['description']}" for k, v in schema.items()])
    messages = [
        {"role": "system", "content": "你是一个智能助手，请根据用户的问题和数据文件的用途描述，判断需要读取哪些数据文件。只返回 JSON 数组，例如：[\"recent_remarks.json\", \"bigFive.json\"]"},
        {"role": "user", "content": f"用户问题：{user_question}\n\n数据文件描述如下：\n{prompt}"}
    ]
    try:
        response = client.chat.completions.create(
            model="doubao-pro-32k-241215",
            messages=messages
        )
        raw_content = response.choices[0].message.content.strip()
        print(f"📁 模型原始返回内容：{raw_content}")
        # 解析为JSON数组
        files = json5.loads(raw_content)
        print(f"📁 模型判断相关文件：{files}")
        return files
    except Exception as e:
        print(f"❌ 文件选择失败：{e}")
        return []



def entry_to_text(entry, file_name):
    try:
        if file_name == "recent_remarks.json":
            return f"{entry.get('person', '')} 的言论: {entry.get('remark', '')}"
        elif file_name == "recent_news.json":
            return f"{entry.get('person', '')} 的新闻: {entry.get('title_cn', '')}"
        elif file_name == "bigFive.json":
            scores = ", ".join([f"{k}:{v}" for k, v in entry.items()])
            return f"人物大五人格得分: {scores}"
        elif file_name == "values.json":
            values = ", ".join([f"{k}:{v}" for k, v in entry.items()])
            return f"人物价值观特征: {values}"
        else:
            return json5.dumps(entry, ensure_ascii=False,) # 确保中文不被转义
    except Exception:
        return str(entry)

class MultiFaissRetriever:
    def __init__(self):
        self.texts = []
        self.metadata = []
        self.vectorizer = None
        self.index = None

    def load_files(self, file_list):
        self.texts = []
        self.metadata = []
        for file_name in file_list:
            file_path = os.path.join(DATA_DIR, file_name)
            data = load_json(file_path)
            if not data:
                continue
            if isinstance(data, dict):
                data = [data]
            for i, item in enumerate(data):
                text = entry_to_text(item, file_name)
                self.texts.append(text)
                self.metadata.append({"source": file_name, "index": i, "text": text})

        if self.texts:
            self.vectorizer = TfidfVectorizer(stop_words="english")
            self.matrix = self.vectorizer.fit_transform(self.texts).toarray().astype(np.float32)
            faiss.normalize_L2(self.matrix)
            self.index = faiss.IndexFlatIP(self.matrix.shape[1])
            self.index.add(self.matrix)
            print(f"✅ 检索器加载完成，共 {len(self.metadata)} 条数据")
        else:
            print("⚠️ 没有加载到任何数据")
            self.vectorizer = None
            self.index = None

    def search(self, question, top_k=6):
        if not self.index or not self.vectorizer:
            return []
        q_vec = self.vectorizer.transform([question]).toarray().astype(np.float32)
        faiss.normalize_L2(q_vec)
        D, I = self.index.search(q_vec, top_k)
        return [self.metadata[i] for i in I[0] if i < len(self.metadata)]

def ask_model(user_question, evidence):
    context = "\n".join([f"{e['text']}（来自 {e['source']} 第 {e['index']} 条）" for e in evidence])
    system_prompt = (
        "你是一个知识型 AI 助手，用户将提问，你将获得若干与问题相关的数据内容。"
        "请根据这些内容生成详细且有逻辑的回答，如信息不足可进行合理补充推理。"
    )
    user_prompt = f"问题：{user_question}\n\n相关数据内容：\n{context}"

    try:
        response = client.chat.completions.create(
            model="doubao-pro-32k-241215",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ 模型调用失败：{e}"

def answer_question(question: str, top_k: int = 6, debug: bool = False):
    schema = load_schema()
    if not schema:
        return {"error": "无法加载 schema 文件"}

    relevant_files = get_relevant_files(question, schema)
    if not relevant_files:
        return {"error": "未能识别相关文件"}

    retriever = MultiFaissRetriever()
    retriever.load_files(relevant_files)
    top_evidence = retriever.search(question, top_k=top_k)
    answer = ask_model(question, top_evidence)

    result = {
        "question": question,
        "answer": answer,
        "used_files": relevant_files,
    }
    if debug:
        result["evidence"] = top_evidence
    return result