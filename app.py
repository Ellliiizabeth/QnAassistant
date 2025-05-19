from flask import Flask, request, jsonify
from question_engine import answer_question
from config import ARK_API_KEY

app = Flask(__name__)

@app.route("/ask", methods=["POST"])
def ask():
    if not ARK_API_KEY:
        return jsonify({"error": "未配置 ARK_API_KEY"}), 500

    if not request.is_json:
        return jsonify({"error": "请求体必须是 JSON 格式"}), 400

    data = request.get_json()
    print("收到请求数据：", data)  # 调试用
    question = data.get("question")
    print("解析的问题是：", question)  # 调试用
    top_k = data.get("top_k", 6)
    debug = data.get("debug", False)

    if not question:
        return jsonify({"error": "缺少 'question' 字段"}), 400

    if not isinstance(question, str):
        return jsonify({"error": "'question' 字段必须是字符串"}), 400

    if top_k is not None and not isinstance(top_k, int):
        return jsonify({"error": "'top_k' 字段必须是整数"}), 400

    if debug is not None and not isinstance(debug, bool):
        return jsonify({"error": "'debug' 字段必须是布尔值"}), 400

    try:
        result = answer_question(question, top_k=top_k, debug=debug)
        return jsonify(result)
    except Exception as e:
        print(f"❌ 处理请求时发生错误: {e}")
        return jsonify({"error": f"处理请求时发生错误: {e}"}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not Found"}), 404

@app.errorhandler(500)
def internal_server_error(error):
    return jsonify({"error": "Internal Server Error"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8888)