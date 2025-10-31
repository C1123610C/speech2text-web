# app.py - 为 Render 优化：延迟加载 tiny 模型，兼容 gunicorn
import os
import tempfile
from flask import Flask, request, jsonify, render_template
import threading

app = Flask(__name__)

# 全局模型变量（延迟加载）
model = None
model_lock = threading.Lock()  # 防止并发首次加载冲突

def load_model_once():
    global model
    with model_lock:
        if model is None:
            # 加载 tiny 模型（内存小、速度快）
            import whisper
            model = whisper.load_model("tiny")
    return model

@app.route("/", methods=["GET"])
def index():
    # 简单首页
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    # 如果没有文件，返回错误
    if 'file' not in request.files:
        return jsonify({"error": "未上传文件"}), 400

    audio = request.files['file']
    if audio.filename == '':
        return jsonify({"error": "未选择文件"}), 400

    # 延迟加载模型（第一次调用时加载）
    load_model_once()

    # 将文件保存到临时位置
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=os.path.splitext(audio.filename)[1])
    os.close(tmp_fd)
    audio.save(tmp_path)

    try:
        # 使用 whisper 模型进行识别
        result = model.transcribe(tmp_path, language="vi")
        text = result.get("text", "")
    except Exception as e:
        return jsonify({"error": "识别失败: " + str(e)}), 500
    finally:
        try:
            os.remove(tmp_path)
        except:
            pass

    return jsonify({"text": text})

if __name__ == "__main__":
    # 本地启动时使用 PORT 环境变量（Render 会提供）
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
