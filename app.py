from flask import Flask, request, render_template, jsonify
import whisper
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# 创建上传文件夹
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 加载模型
print("🎧 正在加载 Whisper 模型，请稍等...")
model = whisper.load_model("base")
print("✅ 模型加载完成！")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files.get('file')
    if not file:
        return jsonify({"error": "未收到音频文件"})

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    print(f"📁 已收到文件：{filepath}")
    result = model.transcribe(filepath, language='vi')
    text = result["text"]

    return jsonify({"text": text})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
