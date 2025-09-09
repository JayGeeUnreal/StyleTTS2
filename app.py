import torch
import os
import soundfile as sf
import numpy as np
from tqdm import tqdm  # For a nice progress bar in the console

# --- Core StyleTTS2 Imports ---
import styletts2importable
from txtsplit import txtsplit

# --- Configuration ---
# All your settings are here at the top for easy access.

# 1. Input Files
PROMPT_FILE = "my_startup_prompt.txt"
REFERENCE_VOICE_PATH = "voices/earn_lucky_pitch_minus_one_samplerate_24000_short.wav"

# 2. Output File
OUTPUT_FILENAME = "output_voice.wav"

# 3. Synthesis Parameters (taken from the Gradio defaults)
DIFFUSION_STEPS = 20
EMBEDDING_SCALE = 1.0
ALPHA = 0.3
BETA = 0.7
SAMPLE_RATE = 24000

# -------------------------------------------------------------------

def run_synthesis():
    """
    Main function to run the text-to-speech synthesis from the command line.
    """
    print("--- Starting Command-Line TTS Synthesis ---")

    # --- Step 1: Load the text prompt ---
    print(f"Attempting to load text from '{PROMPT_FILE}'...")
    try:
        with open(PROMPT_FILE, "r", encoding="utf-8") as f:
            text_to_speak = f.read()
        if not text_to_speak.strip():
            print("Error: The prompt file is empty.")
            return # Exit the function
        print("Text loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Prompt file not found at '{PROMPT_FILE}'. Please create it.")
        return # Exit the function

    # --- Step 2: Check for the reference voice file ---
    print(f"Checking for reference voice at '{REFERENCE_VOICE_PATH}'...")
    if not os.path.exists(REFERENCE_VOICE_PATH):
        print(f"Error: Reference voice file not found at '{REFERENCE_VOICE_PATH}'.")
        return # Exit the function
    print("Reference voice found.")

    # --- Step 3: Run the TTS Model ---
    print("\nSynthesizing speech... (This may take a moment)")
    
    # Split text into manageable chunks
    texts = txtsplit(text_to_speak)
    
    # Pre-calculate the voice style embedding
    print("Computing voice style...")
    target_style = styletts2importable.compute_style(REFERENCE_VOICE_PATH)
    
    audios = []
    # Process each text chunk with a progress bar
    for t in tqdm(texts, desc="Processing Chunks"):
        # The core inference call from the original script
        audio_chunk = styletts2importable.inference(
            t,
            target_style,
            alpha=ALPHA,
            beta=BETA,
            diffusion_steps=DIFFUSION_STEPS,
            embedding_scale=EMBEDDING_SCALE
        )
        audios.append(audio_chunk)

    # Combine all the audio chunks into one array
    full_audio = np.concatenate(audios)
    
    print("Synthesis complete.")

    # --- Step 4: Save the output file ---
    print(f"Saving audio to '{OUTPUT_FILENAME}'...")
    sf.write(OUTPUT_FILENAME, full_audio, SAMPLE_RATE)
    
    print("\n--- Success! ---")
    print(f"Generated speech has been saved to '{os.path.abspath(OUTPUT_FILENAME)}'")


# This is the standard entry point for a Python script.
if __name__ == "__main__":
    run_synthesis()