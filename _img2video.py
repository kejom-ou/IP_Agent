"""
图片 → 动态视频（简化的缓慢推近）
按 TTS 音频时长生成
"""
import sys, os, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT_DIR = r"e:\OwenSpace\IP_Agent\local_models\test_downloads"
IMG_PATH = r"e:\OwenSpace\IP_Agent\local_models\屏幕截图 2026-07-04 180458.png"
TTS_AUDIO = os.path.join(OUT_DIR, "tts_output.wav")
OUT_VIDEO = os.path.join(OUT_DIR, "img_animated.mp4")

import librosa
wav, sr = librosa.load(TTS_AUDIO, sr=None)
duration_s = len(wav) / sr + 0.5
print(f"TTS 时长: {duration_s:.1f}s")

# 简化方案：scale 到 2160x3840（高分辨率源图），
# 用 zoompan 固定公式 z=1+0.0015*on（on=输出帧序号）
# 每帧稍微放大一点，循环结束
fps = 25
total_frames = int(duration_s * fps)
print(f"总帧数: {total_frames}")

cmd = [
    "ffmpeg", "-y",
    "-loop", "1",
    "-i", IMG_PATH,
    "-vf",
    # 缩放到源分辨率（zoompan 起点）+ zoompan
    f"scale=2160:-1,"
    f"zoompan=z='min(zoom+0.0005,1.3)':d={total_frames}:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=720x1280:fps={fps}",
    "-t", str(duration_s),
    "-c:v", "libx264",
    "-preset", "fast",
    "-crf", "23",
    "-pix_fmt", "yuv420p",
    OUT_VIDEO
]
print(f"生成动态视频: {OUT_VIDEO}")
subprocess.run(cmd, check=True)
print(f"✅ 完成: {OUT_VIDEO}, {os.path.getsize(OUT_VIDEO)/1024/1024:.1f} MB")
