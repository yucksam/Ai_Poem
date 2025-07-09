from flask import Flask, render_template, request, jsonify
from generator import generate_poem

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/generate_poem', methods=['POST'])
def generate():
    try:
        data = request.get_json()
        start_words = data.get("start", "").strip()
        length = int(data.get("length", 20))
        model_type = data.get("model", "rnn")
        strategy = data.get("sampling", "temperature").lower()

        # 可选参数
        temperature = float(data.get("temperature", 1.0))
        top_k = int(data.get("top_k", 10))
        top_p = float(data.get("top_p", 0.9))

        if not start_words:
            return jsonify({"error": "起始词不能为空"}), 400

        poem = generate_poem(
            theme=start_words,
            length=length,
            model_type=model_type,
            strategy=strategy,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )

        return jsonify({"poem": poem})

    except Exception as e:
        print("生成失败：", e)
        return jsonify({"error": str(e)}), 500

@app.route('/health')
def health_check():
    return jsonify({"status": "ok"}), 200

if __name__ == "__main__":
    # 开发阶段使用（生产建议用 gunicorn 启动）
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
