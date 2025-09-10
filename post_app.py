import os
import sys
import argparse
import io
import random
import configparser # MODIFICATION: Import the configuration library

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import soundfile as sf
import numpy as np
import torch
import pygame
import phonemizer

# --- Core StyleTTS2 Imports ---
import styletts2importable
from txtsplit import txtsplit

# --- START MODIFICATION: Load Configuration from settings.ini ---
config = configparser.ConfigParser()
config.read('settings.ini')

# Load TTS parameters from the file
DIFFUSION_STEPS = config.getint('TTS', 'diffusion_steps')
EMBEDDING_SCALE = config.getfloat('TTS', 'embedding_scale')
ALPHA = config.getfloat('TTS', 'alpha')
BETA = config.getfloat('TTS', 'beta')
SAMPLE_RATE = config.getint('TTS', 'sample_rate')
SEED = config.getint('TTS', 'seed')
REFERENCE_VOICE = config.get('TTS', 'reference_voice')

# Load Server parameters from the file
SERVER_HOST = config.get('Server', 'host')
SERVER_PORT = config.getint('Server', 'port')
SERVER_DEBUG = config.getboolean('Server', 'debug')
# --- END MODIFICATION ---


# --- Helper Function for Reproducibility ---
def set_seed(seed):
    """Sets the random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- Flask App Initialization ---
app = Flask(__name__)
CORS(app)

print("Initializing phonemizer with stress...")
global_phonemizer = phonemizer.backend.EspeakBackend(
    language='en-us',
    preserve_punctuation=True,
    with_stress=True
)
print("Phonemizer initialized.")

# --- Global variable to hold our pre-computed voice style ---
global_target_style = None

# --- Initialization Function for StyleTTS2 ---
def initialize_styletts2(reference_voice_path):
    global global_target_style
    print("--- Initializing StyleTTS 2 ---")
    if not os.path.exists(reference_voice_path):
        print(f"FATAL ERROR: Reference voice file not found at '{reference_voice_path}'")
        sys.exit(1)
    try:
        print(f"Computing voice style from '{reference_voice_path}'...")
        global_target_style = styletts2importable.compute_style(reference_voice_path)
        print("StyleTTS 2 instance initialized successfully.")
    except Exception as e:
        print(f"FATAL ERROR: Failed to initialize StyleTTS 2: {e}")
        sys.exit(1)

# --- The TTS API Endpoint ---
@app.route('/tts', methods=['POST'])
def tts_endpoint():
    set_seed(SEED) # Use the seed value loaded from the settings file
    print("\n--- New TTS Request Received ---")
    
    try:
        if pygame.mixer.get_init():
            pygame.mixer.music.stop()
            pygame.mixer.music.unload()
    except Exception as e:
        print(f"Warning: Could not unload previous audio. This might be the first run. Error: {e}")

    data = request.get_json()
    print(f"Parsed JSON data: {data}")

    if global_target_style is None:
        return jsonify({"error": "TTS model is not initialized"}), 503

    text_to_speak = data.get('chatmessage')

    if not text_to_speak:
        return jsonify({"error": "Missing 'chatmessage' field in JSON payload"}), 400

    print(f"Successfully extracted text to synthesize: '{text_to_speak[:100]}...'")

    try:
        texts = txtsplit(text_to_speak)
        audios = []
        for t in texts:
            audio_chunk = styletts2importable.inference(
                t, global_target_style, alpha=ALPHA, beta=BETA,
                diffusion_steps=DIFFUSION_STEPS, embedding_scale=EMBEDDING_SCALE
            )
            audios.append(audio_chunk)

        full_audio = np.concatenate(audios)
        print("Synthesis complete.")
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_filepath = os.path.join(script_dir, "server_output.wav")
        
        print(f"Saving a copy of the audio to: {output_filepath}")
        sf.write(output_filepath, full_audio, SAMPLE_RATE)

        try:
            if pygame.mixer.get_init():
                pygame.mixer.music.load(output_filepath)
                pygame.mixer.music.play()
        except Exception as e:
            print(f"Warning: Could not play audio. Pygame error: {e}")

        buffer = io.BytesIO()
        sf.write(buffer, full_audio, SAMPLE_RATE, format='WAV')
        buffer.seek(0)
        return Response(buffer, mimetype='audio/wav')

    except Exception as e:
        print(f"Error processing TTS request: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Internal server error during audio generation"}), 500

# --- Main Function to Start the Server ---
if __name__ == "__main__":
    try:
        pygame.mixer.init()
        print("Pygame mixer initialized successfully.")
    except Exception as e:
        print(f"Warning: Could not initialize Pygame mixer: {e}. Audio playback will be disabled.")

    # The reference voice path now comes from the settings file
    initialize_styletts2(reference_voice_path=REFERENCE_VOICE)
    
    print(f"\nStarting Flask server on http://{SERVER_HOST}:{SERVER_PORT}")
    print("Send a POST request to /tts with JSON {'chatmessage': 'your text here'}")
    
    # The host, port, and debug mode all come from the settings file
    app.run(host=SERVER_HOST, port=SERVER_PORT, debug=SERVER_DEBUG)