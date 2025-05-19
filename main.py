# main.py
import os
import traceback
import uuid
import zipfile
from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename
from solver import truss_solver

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

def save_text_input(content, filepath):
    with open(filepath, 'w') as f:
        f.write(content)

@app.route('/solve', methods=['POST'])
def solve():
    uid = str(uuid.uuid4())[:8]
    session_dir = os.path.join(app.config['UPLOAD_FOLDER'], uid)
    os.makedirs(session_dir, exist_ok=True)

    files = {}
    for key in ['nodes', 'elements', 'ubc', 'fbc']:
        f = request.files.get(key)
        text_data = request.form.get(f"{key}_text")
        if f and f.filename:
            filename = secure_filename(f.filename)
            path = os.path.join(session_dir, filename)
            f.save(path)
            files[key] = path
        elif text_data.strip():
            path = os.path.join(session_dir, f"{key}.csv")
            save_text_input(text_data, path)
            files[key] = path
        else:
            return f"<h2>Missing input for: {key}</h2>", 400

    result_dir = os.path.join(RESULT_FOLDER, uid)
    os.makedirs(result_dir, exist_ok=True)

    try:
        output_csv, undeformed_img, deformed_img, inter_img = truss_solver(
            files['nodes'], files['elements'], files['ubc'], files['fbc'], result_dir)
    except Exception:
        return f"<h2>Solver Error:</h2><pre>{traceback.format_exc()}</pre>", 500

    zip_path = os.path.join(result_dir, 'results.zip')
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for file in [output_csv, undeformed_img, deformed_img, inter_img]:
            zipf.write(file, arcname=os.path.basename(file))

    return render_template('results.html',
        result_csv=os.path.basename(output_csv),
        undeformed_img=os.path.basename(undeformed_img),
        deformed_img=os.path.basename(deformed_img),
        interactive_img=os.path.basename(inter_img),
        zip_file=os.path.basename(zip_path),
        folder=uid
    )

@app.route('/download/<folder>/<filename>')
def download_file(folder, filename):
    return send_file(os.path.join(RESULT_FOLDER, folder, filename), as_attachment=True)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
