import os
import traceback
from flask import Flask, render_template, request, send_file
from solver import truss_solver
from werkzeug.utils import secure_filename
import zipfile
import uuid

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'static'
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
    print("Received files:", request.files)

    for key in ['nodes', 'elements', 'ubc', 'fbc']:
        if key not in request.files:
            return f"<h2>Missing file input:</h2> <pre>{key}</pre>", 400
        f = request.files[key]
        filename = secure_filename(f.filename)
        path = os.path.join(session_dir, filename)
        f.save(path)
        files[key] = path
        print(f"Saved {key} to {path}")

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
    return send_file(os.path.join(RESULT_FOLDER, folder, filename), as_attachment=True)

@app.errorhandler(500)
def internal_error(e):
    return f"<h1>Internal Server Error</h1><pre>{traceback.format_exc()}</pre>", 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
