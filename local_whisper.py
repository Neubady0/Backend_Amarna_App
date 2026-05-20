from flask import Flask, request, jsonify
from flask_cors import CORS
from faster_whisper import WhisperModel
import os
import tempfile

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load model - "small" is a good balance for CPU
# compute_type="int8" is more efficient on CPU
print("Loading Whisper model 'small'...")
model = WhisperModel("small", device="cpu", compute_type="int8")
print("Model loaded.")

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save to a temporary file
    # unique filename to avoid collisions
    temp_filename = f"temp_{file.filename}"
    file.save(temp_filename)

    try:
        segments, info = model.transcribe(temp_filename, beam_size=5)
        
        # Combine all segments into one text
        transcribed_text = " ".join([segment.text for segment in segments])
        
        return jsonify({"text": transcribed_text.strip()})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
        
    finally:
        # Clean up
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

if __name__ == '__main__':
    print("Starting server on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
