import os
from flask import Flask, request,jsonify
from werkzeug.utils import secure_filename
import numpy as np
import shutil
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask_cors import CORS


# ——— Konfigurasi dasar Flask ———
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024    # batasi upload maksimal 5MB
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your_secret_key')

# ——— Pastikan folder uploads ada ———
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ——— Load model Keras ———
MODEL_PATH = os.path.join(app.root_path, 'Model', 'Model_CNN_256px.keras')
model = load_model(MODEL_PATH)

# ——— Label kelas ———
class_labels = {
    0: 'Bercak',
    1: 'Hawar',
    2: 'Karat',
    3: 'Sehat'
}

# ——— Fungsi bantu untuk prediksi ———
def predict_image(model, img_path):
    img = image.load_img(img_path, target_size=(256, 256))               # resize sesuai input model :contentReference[oaicite:0]{index=0}
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0                                                   # normalisasi

    preds = model.predict(img_array)
    idx = np.argmax(preds)
    return idx, preds[0]

CORS(app)
# ——— Route utama ———
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('file')
    if not file or file.filename == '':
        return jsonify({'error': 'No file uploaded'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    idx, probs = predict_image(model, filepath)
    label = class_labels.get(idx, "Tidak diketahui")
    confidence = probs[idx] if probs is not None and idx < len(probs) else None
    probs_dict = {class_labels[i]: float(p * 100) for i, p in enumerate(probs)} if probs is not None else {}

    # Simpan hasil gambar dengan label di nama file (atau bisa diberi watermark/overlay jika mau)
    output_folder = os.path.join(app.root_path, 'static', 'outputs')
    os.makedirs(output_folder, exist_ok=True)
    output_filename = f"{label}_{filename}"
    output_path = os.path.join(output_folder, output_filename)
    shutil.copy(filepath, output_path)  # Jika hanya copy, atau tambahkan proses overlay label jika perlu

    if confidence is not None:
        confidence = round(float(confidence * 100), 2)
    else:
        confidence = 0.0

    return jsonify({
        'predicted_label': label,
        'confidence': confidence,
        'probabilities': probs_dict,
        'result_image': f"outputs/{output_filename}"  # relative path dari static
    })

# ——— Jalankan secara lokal (debug) ———
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
