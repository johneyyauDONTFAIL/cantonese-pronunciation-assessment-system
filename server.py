from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess
import os
from werkzeug.utils import secure_filename
import uuid
import json

app = Flask(__name__)
CORS(app, resources={r"/evaluate": {"origins": "http://localhost:8000"}})

# Allowed audio extensions
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/evaluate', methods=['POST'])
def evaluate():
    print("Received request to /evaluate")
    if 'audio' not in request.files or 'expected_text' not in request.form:
        print("Missing audio file or expected text")
        return jsonify({"error": "Missing audio file or expected text"}), 400
    
    file = request.files['audio']
    expected_text = request.form['expected_text']
    
    if file.filename == '' or not allowed_file(file.filename):
        print("Invalid or no audio file selected")
        return jsonify({"error": "Invalid or no audio file selected"}), 400
    
    file_extension = os.path.splitext(secure_filename(file.filename))[1]
    unique_filename = f"audio_{uuid.uuid4().hex}{file_extension}"
    audio_path = os.path.join(os.path.dirname(__file__), "audio", unique_filename)
    
    file.save(audio_path)
    print(f"Saved audio file: {audio_path}")
    
    try:
        result = subprocess.check_output(['python', 'inference.py', audio_path, expected_text], text=True, stderr=subprocess.STDOUT)
        lines = [line.strip() for line in result.splitlines() if line.strip()]
        for line in lines:
            if line.startswith('{') and line.endswith('}'):
                json_data = json.loads(line)

        if not lines:
            print("Inference script returned no output")
            return jsonify({"error": "Inference script returned no output"}), 500
        
        return jsonify(json_data), 200, {'Content-Type': 'application/json'}
    except subprocess.CalledProcessError as e:
        print(f"Subprocess error: {e.output}")
        return jsonify({"error": f"Script execution failed: {e.output}"}), 500
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)
            print(f"Deleted audio file: {audio_path}")
        pass

if __name__ == '__main__':
    app.run(debug=True, port=5000)