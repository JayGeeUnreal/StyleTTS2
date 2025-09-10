import os
import sys
import io
import random
import configparser

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import soundfile as sf
import numpy as np
import torch
import phonemizer

# --- Core StyleTTS2 Imports ---
import styletts2importable
from txtsplit import txtsplit

# --- Load Configuration from settings.ini ---
config = configparser.ConfigParser()
config.read('settings.ini')

# Configuration loading
DIFFUSION_STEPS = config.getint('TTS', 'diffusion_steps')
EMBEDDING_SCALE = config.getfloat('TTS', 'embedding_scale')
ALPHA = config.getfloat('TTS', 'alpha')
BETA = config.getfloat('TTS', 'beta')
SAMPLE_RATE = config.getint('TTS', 'sample_rate')
SEED = config.getint('TTS', 'seed')
REFERENCE_VOICE = config.get('TTS', 'reference_voice')
SERVER_HOST = config.get('Server', 'host')
SERVER_PORT = config.getint('Server', 'port')
SERVER_DEBUG = config.getboolean('Server', 'debug')

# --- Helper Function for Reproducibility ---
def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- Flask App Initialization ---
app = Flask(__name__)
CORS(app)

# --- StyleTTS 2 Phonemizer Initialization ---
print("Initializing StyleTTS 2 phonemizer with stress...")
global_phonemizer = phonemizer.backend.EspeakBackend(
    language='en-us', preserve_punctuation=True, with_stress=True
)
print("Phonemizer initialized.")

# --- Global variable for our pre-computed voice style ---
global_target_style = None

# --- Initialization Function for StyleTTS 2 ---
def initialize_styletts2(reference_voice_path):
    global global_target_style
    print("--- Initializing StyleTTS 2 ---")
    try:
        if not os.path.exists(reference_voice_path):
            raise FileNotFoundError(f"Reference voice file not found at '{reference_voice_path}'")
        print(f"Computing StyleTTS 2 voice style from '{reference_voice_path}'...")
        global_target_style = styletts2importable.compute_style(reference_voice_path)
        print("StyleTTS 2 instance initialized successfully.")
    except Exception as e:
        print(f"FATAL ERROR: Failed to initialize StyleTTS 2: {e}")
        sys.exit(1)

# --- The TTS API Endpoint ---
@app.route('/tts', methods=['POST', 'PUT'])
def tts_endpoint():
    set_seed(SEED)
    print("\n--- New TTS Request Received ---")
   
    data = request.get_json()
    text_to_speak = data.get('chatmessage')

    if not text_to_speak: return jsonify({"error": "Missing 'chatmessage' field"}), 400

    print(f"Synthesizing: '{text_to_speak[:100]}...'")

    try:
        texts = txtsplit(text_to_speak)
        audios = [styletts2importable.inference(t, global_target_style, alpha=ALPHA, beta=BETA, diffusion_steps=DIFFUSION_STEPS, embedding_scale=EMBEDDING_SCALE) for t in texts]
        full_audio = np.concatenate(audios)
        print("Synthesis complete.")

        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_filepath = os.path.join(script_dir, "server_output.wav")
        
        sf.write(output_filepath, full_audio, SAMPLE_RATE)
        print(f"Saved StyleTTS 2 audio to: {output_filepath}")

        # Send the audio back in the response
        buffer = io.BytesIO()
        sf.write(buffer, full_audio, SAMPLE_RATE, format='WAV')
        buffer.seek(0)
        return Response(buffer, mimetype='audio/wav')

    except Exception as e:
        print(f"Error during synthesis: {e}"); import traceback; traceback.print_exc()
        return jsonify({"error": "Internal server error during generation"}), 500

# --- Main Function to Start the Server ---
if __name__ == "__main__":
    
    initialize_styletts2(reference_voice_path=REFERENCE_VOICE)
    
    print(f"\n--- Starting StyleTTS 2 Server (Silent Mode) on http://{SERVER_HOST}:{SERVER_PORT} ---")
    print(f" -> Endpoint: /tts (expects JSON {{'chatmessage': '...'}})")
    
    app.run(host=SERVER_HOST, port=SERVER_PORT, debug=SERVER_DEBUG)