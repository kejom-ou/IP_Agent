"""最终合成：lipsync 视频 + TTS 音频 + SRT 字幕"""
import subprocess
import os

OUT_DIR = r"e:\OwenSpace\IP_Agent\local_models\test_downloads"
LIPSYNC_VIDEO = os.path.join(OUT_DIR, "lipsync_full.mp4")
TTS_AUDIO = os.path.join(OUT_DIR, "tts_output.wav")
SRT_FILE = os.path.join(OUT_DIR, "tts_timeline.srt")
OUTPUT = os.path.join(OUT_DIR, "final_output.mp4")

# ffmpeg subtitles 滤镜需要路径转义
srt_path = SRT_FILE.replace("\\", "/").replace(":", "\\:")
style = "FontSize=22,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,Outline=2,Alignment=2,MarginV=40"
filter_complex = f"[0:v]subtitles='{srt_path}':force_style='{style}'[v]"

cmd = [
    "ffmpeg", "-y",
    "-i", LIPSYNC_VIDEO,
    "-i", TTS_AUDIO,
    "-filter_complex", filter_complex,
    "-map", "[v]",
    "-map", "1:a",
    "-c:v", "libx264",
    "-preset", "fast",
    "-crf", "23",
    "-c:a", "aac",
    "-b:a", "128k",
    "-shortest",
    OUTPUT
]
print("RUN:", " ".join(cmd))
subprocess.run(cmd, check=True)
size = os.path.getsize(OUTPUT) / 1024 / 1024
print(f"OK: {OUTPUT}, {size:.1f} MB")
