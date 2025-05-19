import os
import uuid
import zipfile
import traceback
from flask import Flask, render_template, request, send_from_directory, redirect, url_for
from werkzeug.utils import secure_filename
from solver import truss_solver

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/solve', methods=['POST'])
def solve():
    uid = str(uuid.uuid4())[:8]
    session_dir = os.path.join(app.config['UPLOAD_FOLDER'], uid)
    os.makedirs(session_dir, exist_ok=True)

    files = {}
    for key in ['nodes', 'elements', 'ubc', 'fbc']:
        file = request.files.get(key)
        manual_text = request.form.get(f"{key}_manual")

        if file and file.filename:
            filename = secure_filename(file.filename)
            path = os.path.join(session_dir, filename)
            file.save(path)
        elif manual_text:
            filename = f"{key}.csv"
            path = os.path.join(session_dir, filename)
            with open(path, 'w') as f:
                f.write(manual_text.strip())
        else:
            return f"<h2>Error: Missing input for {key.title()}</h2>", 400

        files[key] = path

    result_dir = os.path.join(RESULT_FOLDER, uid)
    os.makedirs(result_dir, exist_ok=True)

    try:
        output_csv, undeformed_img, deformed_img = truss_solver(
            files['nodes'], files['elements'], files['ubc'], files['fbc'], result_dir)
    except Exception as e:
        return f"<h2>Solver Error:</h2><pre>{traceback.format_exc()}</pre>", 500

    zip_path = os.path.join(result_dir, 'results.zip')
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for file in [output_csv, undeformed_img, deformed_img]:
            zipf.write(file, arcname=os.path.basename(file))

    return render_template('results.html',
        result_csv=os.path.basename(output_csv),
        undeformed_img=os.path.basename(undeformed_img),
        deformed_img=os.path.basename(deformed_img),
        zip_file=os.path.basename(zip_path),
        folder=uid
    )

@app.route('/download/<folder>/<filename>')
def download_file(folder, filename):
    directory = os.path.join(RESULT_FOLDER, folder)
    return send_from_directory(directory, filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
