"""
Audio transcription module using OpenAI Whisper.
"""
import whisper
import numpy as np
from collections import deque
import threading
import time


class AudioTranscriber:
    def __init__(self, model_size="base"):
        """
        Initialize Whisper model for transcription.
        model_size: 'tiny', 'base', 'small', 'medium', 'large'
        """
        print(f"[AudioTranscriber] Loading Whisper model: {model_size}")
        self.model = whisper.load_model(model_size)
        print(f"[AudioTranscriber] Model loaded successfully")
        self.audio_buffer = deque(maxlen=16000 * 5)  # 5 seconds at 16kHz
        self.transcription = ""
        self.is_transcribing = False
        self.total_chunks = 0
        self.last_transcribe_time = 0
        
    def add_audio_chunk(self, audio_chunk):
        """Add audio data to buffer"""
        self.audio_buffer.extend(audio_chunk)
        self.total_chunks += 1
        
        # Print debug info every 100 chunks
        if self.total_chunks % 100 == 0:
            buffer_size = len(self.audio_buffer)
            buffer_seconds = buffer_size / 16000
            print(f"[AudioTranscriber] Received {self.total_chunks} chunks, buffer: {buffer_seconds:.2f}s")
        
    def transcribe(self):
        """Transcribe the current audio buffer"""
        buffer_size = len(self.audio_buffer)
        buffer_seconds = buffer_size / 16000
        
        print(f"[AudioTranscriber] Transcribe requested, buffer size: {buffer_seconds:.2f}s")
        
        if buffer_size < 16000:  # Need at least 1 second
            print(f"[AudioTranscriber] Not enough audio (need 1s, have {buffer_seconds:.2f}s)")
            return ""
            
        if self.is_transcribing:
            print(f"[AudioTranscriber] Already transcribing, skipping...")
            return self.transcription
            
        self.is_transcribing = True
        print(f"[AudioTranscriber] Starting transcription...")
        start_time = time.time()
        
        # Convert buffer to numpy array
        audio_array = np.array(list(self.audio_buffer), dtype=np.float32)
        
        # Transcribe
        try:
            result = self.model.transcribe(audio_array, fp16=False)
            self.transcription = result["text"]
            elapsed = time.time() - start_time
            print(f"\n{'='*60}")
            print(f"[TRANSCRIPTION] ({elapsed:.2f}s to process)")
            print(f"{self.transcription}")
            print(f"{'='*60}\n")
        except Exception as e:
            self.transcription = f"Error: {str(e)}"
            print(f"[AudioTranscriber] ERROR: {str(e)}")
        finally:
            self.is_transcribing = False
            self.last_transcribe_time = time.time()
            
        return self.transcription
    
    def clear_buffer(self):
        """Clear audio buffer and transcription"""
        self.audio_buffer.clear()
        self.transcription = ""
        print(f"[AudioTranscriber] Buffer cleared")
