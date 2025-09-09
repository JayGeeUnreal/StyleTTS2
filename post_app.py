import os
import sys
import argparse
import io

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

# --- Configuration ---
DIFFUSION_STEPS = 20
EMBEDDING_SCALE = 1
ALPHA = 0.4
BETA = 0.7
SAMPLE_RATE = 24000

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
    print("\n--- New TTS Request Received ---")
    
    try:
        if pygame.mixer.get_init():
            print("Releasing previous audio file lock...")
            pygame.mixer.music.stop()
            pygame.mixer.music.unload()
            print("File lock released.")
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
                print("Playing new audio track...")
                pygame.mixer.music.load(output_filepath)
                pygame.mixer.music.play()
        except Exception as e:
            print(f"Warning: Could not play audio. Pygame error: {e}")

        # Prepare the audio to send back as a response
        buffer = io.BytesIO()
        sf.write(buffer, full_audio, SAMPLE_RATE, format='WAV')
        buffer.seek(0)
        return Response(buffer, mimetype='audio/wav')

    except Exception as e:
        print(f"Error processing TTS request: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Internal server error during audio generation"}), 500

# --- Argument Parsing for the Flask Server ---
def parse_args():
    parser = argparse.ArgumentParser(description="StyleTTS 2 Flask Server")
    parser.add_argument("--reference_voice", type=str, default="voices/f-us-1.wav", help="Path to the reference voice audio file (.wav) for cloning.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address for the Flask server.")
    parser.add_argument("--port", type=int, default=13000, help="Port for the Flask server.")
    # MODIFICATION: Corrected the typo from add__argument to add_argument
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode.")
    return parser.parse_args()

# --- Main Function to Start the Server ---
if __name__ == "__main__":
    args = parse_args()

    try:
        pygame.mixer.init()
        print("Pygame mixer initialized successfully.")
    except Exception as e:
        print(f"Warning: Could not initialize Pygame mixer: {e}. Audio playback will be disabled.")

    initialize_styletts2(reference_voice_path=args.reference_voice)
    
    print(f"\nStarting Flask server on http://{args.host}:{args.port}")
    print("Send a POST request to /tts with JSON {'chatmessage': 'your text here'}")
    
    app.run(host=args.host, port=args.port, debug=args.debug)