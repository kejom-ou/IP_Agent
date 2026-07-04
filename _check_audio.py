import librosa
w, sr = librosa.load(r"e:\OwenSpace\IP_Agent\local_models\test_downloads\tts_output.wav", sr=None)
print(f"TTS时长: {len(w)/sr:.2f}s, {sr}Hz")
