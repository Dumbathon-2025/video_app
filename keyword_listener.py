"""
Real-time keyword detection using Vosk speech recognition.
Runs in a background thread and detects specified keywords.
"""
import queue
import json
import vosk
import sounddevice as sd
import threading

class KeywordListener:
    def __init__(self, keywords, model_path="vosk-model-small-en-us-0.15"):
        """
        Initialize keyword listener.
        
        Args:
            keywords: List of keywords to detect (case-insensitive)
            model_path: Path to Vosk model directory
        """
        self.keywords = [k.lower() for k in keywords]
        self.model = vosk.Model(model_path)
        self.q = queue.Queue()
        self.last_detected = None
        self.running = False
        self.thread = None
        
    def callback(self, indata, frames, time, status):
        """Audio callback for sounddevice"""
        if status:
            print(f"Audio status: {status}")
        self.q.put(bytes(indata))
    
    def detect_keywords(self, text):
        """Check if text contains any keywords"""
        text_lower = text.lower()
        for keyword in self.keywords:
            if keyword in text_lower:
                return keyword
        return None
    
    def listen_loop(self):
        """Main listening loop (runs in background thread)"""
        rec = vosk.KaldiRecognizer(self.model, 16000)
        
        with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                               channels=1, callback=self.callback):
            print("🎤 Keyword listener started...")
            
            while self.running:
                data = self.q.get()
                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    text = result.get('text', '')
                    if text:
                        print(f"Heard: {text}")
                        keyword = self.detect_keywords(text)
                        if keyword:
                            self.last_detected = keyword
                            print(f"✅ KEYWORD DETECTED: {keyword.upper()}")
                else:
                    # Partial result
                    partial = json.loads(rec.PartialResult())
                    text = partial.get('partial', '')
                    if text:
                        keyword = self.detect_keywords(text)
                        if keyword:
                            self.last_detected = keyword
                            print(f"✅ KEYWORD DETECTED: {keyword.upper()}")
    
    def start(self):
        """Start listening in background thread"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self.listen_loop, daemon=True)
            self.thread.start()
    
    def stop(self):
        """Stop listening"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
    
    def get_last_keyword(self):
        """Get the last detected keyword and clear it"""
        keyword = self.last_detected
        self.last_detected = None
        return keyword
    
    def has_keyword(self):
        """Check if a keyword was detected"""
        return self.last_detected is not None
