import subprocess
import torch
import gc
import os
from faster_whisper import WhisperModel

def run_telejournalism_pipeline(video_path):
    print(f"--- Starting Pipeline for {video_path} ---")
    
    # STEP 1: VideoLogger -> FFmpeg (Extract Audio)
    # Automatically saves the audio in the same folder as the video file
    audio_path = video_path.rsplit('.', 1)[0] + ".wav"
    
    print("[Phase 1] Extracting audio via FFmpeg...")
    
    # Call the FFmpeg engine we installed to rip the audio track instantly
    subprocess.run([
        r"C:\ffmpeg\bin\ffmpeg.exe", "-y", "-i", video_path, 
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    if not os.path.exists(audio_path):
        print("CRITICAL ERROR: FFmpeg failed to create the audio file. Check paths.")
        return

    print(f"[Phase 1] Audio extracted successfully to: {audio_path}\n")

    # STEP 2: Whisper (Transcription)
    print("[Phase 2] Loading Whisper model into RTX 3060...")
    print("[System] Enforcing INT8 quantization to protect 6GB VRAM limits...")
    
    # 'int8_float16' drastically reduces VRAM usage to fit your card safely.
    model = WhisperModel("small", device="cuda", compute_type="int8_float16")
    
    print("[Phase 2] Transcribing audio...")
    segments, info = model.transcribe(audio_path, beam_size=5, language="pt")
    
    print(f"\nDetected language: {info.language} (Probability: {info.language_probability:.2f})")
    print("--- Transcription ---")
    for segment in segments:
        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
        
    # STEP 3: VRAM Cleanup (Crucial for Local Processing)
    print("\n[System] Flushing VRAM to prepare for next tasks (Demucs/Llama)...")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    print("[System] Pipeline test complete!")

if __name__ == "__main__":
    # Sanity check to ensure PyTorch sees your RTX 3060
    if torch.cuda.is_available():
        print(f"CUDA active. GPU detected: {torch.cuda.get_device_name(0)}\n")
        
        # Pointing the pipeline to your media folder and the specific video
        run_telejournalism_pipeline("media/bnes.mp4")
    else:
        print("CRITICAL ERROR: CUDA is not available. PyTorch is running on CPU.")