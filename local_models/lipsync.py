"""
本地口型合成引擎（Wav2Lip，从本地模型加载）
轻量替代 MuseTalk — 无需 MMLab，显存 ~2GB
"""
# 禁用 torch.compile（face_alignment SFD 编译耗时 18s+ 且无 Triton）
import os
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import gc
import logging
import math
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
torch.backends.cudnn.enabled = False  # RTX 5060 Blackwell cuDNN 兼容
import torch.nn as nn
import torch.nn.functional as F
import cv2

from local_models.config import LIPSYNC_CONFIG

logger = logging.getLogger(__name__)

# ============================================================
# Wav2Lip 模型架构（从 Rudrabha/Wav2Lip 内联，去掉判别器）
# ============================================================

class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size, stride, padding),
            nn.BatchNorm2d(cout)
        )
        self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)


class Conv2dTranspose(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, output_padding=0):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ConvTranspose2d(cin, cout, kernel_size, stride, padding, output_padding),
            nn.BatchNorm2d(cout)
        )
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.conv_block(x))


class Wav2Lip(nn.Module):
    """Wav2Lip 生成器 — 音频驱动口型同步"""

    def __init__(self):
        super().__init__()

        self.face_encoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(6, 16, kernel_size=7, stride=1, padding=3)),   # 96,96

            nn.Sequential(Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 48,48
                          Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 24,24
                          Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 12,12
                          Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 6,6
                          Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # 3,3
                          Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(512, 512, kernel_size=3, stride=1, padding=0),  # 1,1
                          Conv2d(512, 512, kernel_size=1, stride=1, padding=0)),
        ])

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
        )

        self.face_decoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(512, 512, kernel_size=1, stride=1, padding=0)),

            nn.Sequential(Conv2dTranspose(1024, 512, kernel_size=3, stride=1, padding=0),
                          Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2dTranspose(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
                          Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2dTranspose(768, 384, kernel_size=3, stride=2, padding=1, output_padding=1),
                          Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2dTranspose(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
                          Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2dTranspose(320, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
                          Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2dTranspose(160, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                          Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True)),
        ])

        self.output_block = nn.Sequential(
            Conv2d(80, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, audio_sequences, face_sequences):
        B = audio_sequences.size(0)
        input_dim_size = len(face_sequences.size())

        if input_dim_size > 4:
            audio_sequences = torch.cat(
                [audio_sequences[:, i] for i in range(audio_sequences.size(1))], dim=0
            )
            face_sequences = torch.cat(
                [face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0
            )

        audio_embedding = self.audio_encoder(audio_sequences)  # B, 512, 1, 1

        feats = []
        x = face_sequences
        for f in self.face_encoder_blocks:
            x = f(x)
            feats.append(x)

        x = audio_embedding
        for f in self.face_decoder_blocks:
            x = f(x)
            try:
                x = torch.cat((x, feats[-1]), dim=1)
            except Exception as e:
                logger.error(f"尺寸不匹配: x={x.size()}, feat={feats[-1].size()}")
                raise e
            feats.pop()

        x = self.output_block(x)

        if input_dim_size > 4:
            x = torch.split(x, B, dim=0)
            outputs = torch.stack(x, dim=2)
        else:
            outputs = x

        return outputs


# ============================================================
# Mel 频谱提取（简化自 Wav2Lip audio.py，现代 librosa API）
# ============================================================

class AudioProcessor:
    """梅尔频谱提取器 — 与 Wav2Lip 训练时参数一致"""
    def __init__(self):
        self.sample_rate = 16000
        self.n_fft = 800
        self.hop_size = 200
        self.win_size = 800
        self.num_mels = 80
        self.fmin = 55
        self.fmax = 7600
        self.preemphasis = 0.97
        self.ref_level_db = 20
        self.min_level_db = -100
        self.max_abs_value = 4.0
        self.symmetric_mels = True
        self.signal_normalization = True
        self.allow_clipping_in_normalization = True
        self._mel_basis = None

    def _build_mel_basis(self):
        import librosa
        return librosa.filters.mel(
            sr=self.sample_rate, n_fft=self.n_fft,
            n_mels=self.num_mels, fmin=self.fmin, fmax=self.fmax
        )

    def get_mel(self, wav):
        """输入 wav (np.ndarray, sr=16000)，返回 (T, 80) mel 频谱"""
        import librosa
        from scipy import signal

        if self._mel_basis is None:
            self._mel_basis = self._build_mel_basis()

        # 预加重
        wav = signal.lfilter([1, -self.preemphasis], [1], wav)

        # STFT
        D = librosa.stft(
            wav, n_fft=self.n_fft, hop_length=self.hop_size,
            win_length=self.win_size
        )

        # 幅度 → dB
        S = np.abs(D)
        S = np.dot(self._mel_basis, S)
        min_level = np.exp(self.min_level_db / 20 * np.log(10))
        S = 20 * np.log10(np.maximum(min_level, S))
        S = S - self.ref_level_db

        # 归一化
        if self.signal_normalization:
            if self.allow_clipping_in_normalization:
                if self.symmetric_mels:
                    S = np.clip(
                        2 * self.max_abs_value * ((S - self.min_level_db) / (-self.min_level_db))
                        - self.max_abs_value,
                        -self.max_abs_value, self.max_abs_value
                    )
            else:
                if self.symmetric_mels:
                    S = 2 * self.max_abs_value * ((S - self.min_level_db) / (-self.min_level_db)) \
                        - self.max_abs_value

        return S.T  # (T, 80)


# ============================================================
# Wav2Lip 口型合成引擎
# ============================================================

class Wav2LipEngine:
    """Wav2Lip 口型合成 — 轻量，~2GB 显存，无需 MMLab"""

    IMG_SIZE = 96
    MEL_STEP_SIZE = 16
    BATCH_SIZE = 128

    def __init__(self, model_path: str = None):
        self.model_path = model_path or os.path.join(
            LIPSYNC_CONFIG["local_path"], "wav2lip_gan.pth"
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model: Optional[Wav2Lip] = None
        self.audio_proc = AudioProcessor()
        self.face_detector = None
        self._is_torchscript = False

    def load(self) -> bool:
        """加载 Wav2Lip 模型 + 人脸检测器"""
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 1. 加载 Wav2Lip 生成器
            logger.info(f"加载 Wav2Lip 模型从: {self.model_path}")
            if not os.path.exists(self.model_path):
                logger.error(f"模型文件不存在: {self.model_path}")
                logger.error("请下载 wav2lip_gan.pth 到 pretrained_models/Wav2Lip/")
                return False

            # 尝试加载（优先 state_dict 避免 TorchScript cuDNN 固化问题）
            try:
                # 模型文件可能是 TorchScript 或普通 checkpoint
                # 策略：先 torch.jit.load，再提取 state_dict 灌入 Python nn.Module
                script_model = torch.jit.load(self.model_path, map_location=self.device)
                self.model = Wav2Lip().to(self.device)
                self.model.load_state_dict(script_model.state_dict(), strict=True)
                self._is_torchscript = False
                del script_model
                logger.info("以 state_dict 格式加载 Wav2Lip（从 TorchScript 提取）")
            except Exception:
                try:
                    # 回退 2：普通 checkpoint
                    checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
                    self.model = Wav2Lip().to(self.device)
                    state_dict = checkpoint.get("state_dict", checkpoint)
                    new_state = {}
                    for k, v in state_dict.items():
                        new_state[k.replace("module.", "")] = v
                    self.model.load_state_dict(new_state, strict=True)
                    self._is_torchscript = False
                    logger.info("以 state_dict 格式加载 Wav2Lip")
                except Exception:
                    # 回退 3：纯 TorchScript
                    self.model = torch.jit.load(self.model_path, map_location=self.device)
                    self._is_torchscript = True
                    logger.info("以 TorchScript 格式加载 Wav2Lip")

            self.model.eval()
            logger.info("Wav2Lip 模型加载完成")

            # 2. 加载人脸检测器
            try:
                import face_alignment
                self.face_detector = face_alignment.FaceAlignment(
                    face_alignment.LandmarksType.TWO_D,
                    device=self.device,
                    face_detector='sfd'
                )
                logger.info("人脸检测器 (face-alignment/SFD) 加载完成")

                # 预热 face_alignment（首次调用 torch 编译耗时 ~19s，预热后 ~0.25s/帧）
                logger.info("正在预热人脸检测器（首次编译，约 15-20s，请耐心等待）...")
                t_warm = time.time()
                dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
                self.face_detector.get_landmarks(dummy)
                logger.info(f"人脸检测器预热完成，耗时 {time.time() - t_warm:.1f}s")

            except ImportError:
                logger.warning("face-alignment 未安装，尝试使用 OpenCV DNN")
                self._init_opencv_face_detector()

            logger.info(f"Wav2Lip 加载完毕，设备: {self.device}")
            return True

        except Exception as e:
            logger.error(f"Wav2Lip 加载失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _init_opencv_face_detector(self):
        """备用人脸检测 — OpenCV DNN"""
        try:
            model_file = os.path.join(
                LIPSYNC_CONFIG["local_path"],
                "opencv_face_detector_uint8.pb"
            )
            config_file = os.path.join(
                LIPSYNC_CONFIG["local_path"],
                "opencv_face_detector.pbtxt"
            )
            if os.path.exists(model_file) and os.path.exists(config_file):
                self.face_detector = cv2.dnn.readNetFromTensorflow(model_file, config_file)
                logger.info("OpenCV DNN 人脸检测器加载完成")
        except Exception:
            self.face_detector = None

    def _detect_face(self, frame: np.ndarray):
        """检测单帧人脸的 bbox (x1, y1, x2, y2) 或 None"""
        if hasattr(self.face_detector, 'get_landmarks'):
            # face-alignment
            try:
                preds = self.face_detector.get_landmarks(frame)
                if preds is None:
                    return None
                pred = preds[0]
                x1 = int(np.min(pred[:, 0]))
                y1 = int(np.min(pred[:, 1]))
                x2 = int(np.max(pred[:, 0]))
                y2 = int(np.max(pred[:, 1]))
                return (x1, y1, x2, y2)
            except Exception:
                return None
        elif isinstance(self.face_detector, cv2.dnn.Net):
            # OpenCV DNN
            h, w = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123])
            self.face_detector.setInput(blob)
            detections = self.face_detector.forward()
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:
                    x1 = int(detections[0, 0, i, 3] * w)
                    y1 = int(detections[0, 0, i, 4] * h)
                    x2 = int(detections[0, 0, i, 5] * w)
                    y2 = int(detections[0, 0, i, 6] * h)
                    return (x1, y1, x2, y2)
            return None
        return None

    def _get_smoothened_boxes(self, boxes, window=5):
        """人脸框时域平滑"""
        if len(boxes) == 0 or boxes[0] is None:
            return boxes
        for i in range(len(boxes)):
            if boxes[i] is None:
                for j in range(i, len(boxes)):
                    if boxes[j] is not None:
                        boxes[i] = boxes[j]
                        break

        boxes_arr = np.array(boxes, dtype=np.float32)
        smoothed = []
        for i in range(len(boxes)):
            start = max(0, i - window // 2)
            end = min(len(boxes), i + window // 2 + 1)
            avg = np.mean(boxes_arr[start:end], axis=0).astype(int)
            smoothed.append(tuple(avg))
        return smoothed

    def _crop_face(self, frame: np.ndarray, bbox):
        """裁剪人脸并缩放到 96x96，返回 (face_96, bbox_original, mask)"""
        if bbox is None:
            return None, None, None

        x1, y1, x2, y2 = bbox
        # Padding 30%
        w, h = x2 - x1, y2 - y1
        pad_w, pad_h = int(w * 0.3), int(h * 0.3)

        img_h, img_w = frame.shape[:2]
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(img_w, x2 + pad_w)
        y2 = min(img_h, y2 + pad_h)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None, None, None

        crop_96 = cv2.resize(crop, (self.IMG_SIZE, self.IMG_SIZE))

        # Mask: 下半脸
        mask = np.zeros((self.IMG_SIZE, self.IMG_SIZE), dtype=np.float32)
        mask[self.IMG_SIZE // 2:, :] = 1.0

        return crop_96, (x1, y1, x2, y2), mask

    def generate(
        self,
        video_path: str,
        audio_path: str,
        output_path: Optional[str] = None,
        fps: int = 25,
    ) -> Optional[str]:
        """
        生成口型同步视频

        Args:
            video_path: 输入视频路径
            audio_path: 输入音频路径（16kHz wav）
            output_path: 输出视频路径（默认: 同目录 + _lipsync 后缀）
            fps: 输出帧率
        """
        if self.model is None:
            logger.error("模型未加载，请先调用 load()")
            return None

        if output_path is None:
            suffix = Path(video_path).suffix or ".mp4"
            output_path = str(
                Path(video_path).parent / f"{Path(video_path).stem}_lipsync{suffix}"
            )

        try:
            # 1. 读视频帧
            logger.info("正在读取视频帧...")
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"无法打开视频: {video_path}")
                return None

            orig_frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                orig_frames.append(frame)
            cap.release()
            orig_h, orig_w = orig_frames[0].shape[:2]
            logger.info(f"读取 {len(orig_frames)} 帧, 分辨率 {orig_w}x{orig_h}")

            # 2. 读音频 + mel
            logger.info("正在处理音频...")
            import librosa
            wav, sr = librosa.load(audio_path, sr=16000)
            mel = self.audio_proc.get_mel(wav)  # (T_mel, 80)
            logger.info(f"Mel 形状: {mel.shape}")

            # 3. 人脸检测
            logger.info("正在检测人脸...")
            t_det = time.time()
            raw_boxes = []
            for i, frame in enumerate(orig_frames):
                bbox = self._detect_face(frame)
                raw_boxes.append(bbox)
                if (i + 1) % 30 == 0 or i == 0:
                    logger.info(f"  人脸检测进度: {i+1}/{len(orig_frames)} ({(time.time()-t_det)/(i+1)*1000:.0f}ms/帧)")
            logger.info(f"人脸检测完成，共 {len(orig_frames)} 帧，耗时 {time.time()-t_det:.1f}s")

            # 平滑
            boxes = self._get_smoothened_boxes(raw_boxes)

            # 找最稳定的人脸框（检测置信度最高）
            valid_indices = [i for i, b in enumerate(boxes) if b is not None]
            if not valid_indices:
                logger.error("未检测到人脸")
                return None
            logger.info(f"检测到人脸帧: {len(valid_indices)}/{len(orig_frames)}")

            # 4. 准备 crop
            face_crops = []  # (N, 96, 96, 3)
            crop_boxes = []  # bbox in original frame
            face_masks = []  # (96, 96) mask

            for i, frame in enumerate(orig_frames):
                bbox = boxes[i] if boxes[i] is not None else boxes[valid_indices[0]]
                crop, cbox, mask = self._crop_face(frame, bbox)
                if crop is not None:
                    face_crops.append(crop)
                    crop_boxes.append(cbox)
                    face_masks.append(mask)
                else:
                    face_crops.append(np.zeros((self.IMG_SIZE, self.IMG_SIZE, 3), dtype=np.uint8))
                    crop_boxes.append((0, 0, self.IMG_SIZE, self.IMG_SIZE))
                    face_masks.append(np.zeros((self.IMG_SIZE, self.IMG_SIZE), dtype=np.float32))

            # 5. 准备输入数据
            # face: (N, 6, 96, 96) — 6通道 = 当前帧 + 下半脸mask
            faces = []
            for i in range(len(face_crops)):
                f = face_crops[i].astype(np.float32) / 255.0  # (96, 96, 3)
                mask = face_masks[i]  # (96, 96)
                # 6 channels: lower_half * mask
                lower_half = f.copy()
                lower_half[:self.IMG_SIZE // 2, :, :] = 0.0
                # Stack: [rgb, masked_lowerhalf]
                face_input = np.concatenate([f, lower_half * mask[:, :, np.newaxis]], axis=2)
                faces.append(face_input.transpose(2, 0, 1))  # (6, 96, 96)

            faces = np.stack(faces)  # (N, 6, 96, 96)
            faces_tensor = torch.FloatTensor(faces).to(self.device)

            # mel → audio chunks: each chunk corresponds to 1 frame
            # Mel has hop_size=200 samples at 16000Hz → 12.5ms per mel frame
            # Video at 25fps → 40ms per frame
            # So each video frame needs 40/12.5 = 3.2 mel frames → we use 16-step chunks
            mel_chunks = []
            mel_idx_multiplier = 80.0 / float(fps)  # magic number from Wav2Lip
            mel_step_size = self.MEL_STEP_SIZE

            for i in range(len(face_crops)):
                start_idx = int(i * mel_idx_multiplier)
                if start_idx + mel_step_size > len(mel):
                    start_idx = len(mel) - mel_step_size
                chunk = mel[start_idx:start_idx + mel_step_size]  # (16, 80)
                mel_chunks.append(chunk)

            mel_chunks = np.array(mel_chunks)  # (N, 16, 80)
            mel_tensor = torch.FloatTensor(mel_chunks).unsqueeze(2).to(self.device)  # (N, 16, 1, 80)
            # Reshape to (N, 1, 80, 16) for the model
            mel_tensor = mel_tensor.permute(0, 2, 3, 1)  # (N, 1, 80, 16)

            # 6. Wav2Lip 推理 (batch)
            logger.info("正在进行 Wav2Lip 推理...")
            gen_faces = []
            self.model.eval()
            with torch.no_grad():
                n_batches = (len(faces) + self.BATCH_SIZE - 1) // self.BATCH_SIZE
                for bi, start in enumerate(range(0, len(faces), self.BATCH_SIZE)):
                    end = min(start + self.BATCH_SIZE, len(faces))
                    m_batch = mel_tensor[start:end]  # (B, 1, 80, 16)

                    # 统一使用 4D 输入 (B, C, H, W)，forward 内部自动处理
                    f_batch = faces_tensor[start:end]  # (B, 6, 96, 96)
                    pred = self.model(m_batch, f_batch)  # (B, 3, 96, 96)
                    gen = pred.permute(0, 2, 3, 1).cpu().numpy()  # (B, 96, 96, 3)
                    gen_faces.append(gen)
                    if (bi + 1) % max(1, n_batches // 5) == 0:
                        logger.info(f"  Wav2Lip 推理进度: {bi+1}/{n_batches} batches")

            gen_faces = np.concatenate(gen_faces, axis=0)  # (N, 96, 96, 3)
            gen_faces = (gen_faces * 255).clip(0, 255).astype(np.uint8)
            logger.info("Wav2Lip 推理完成")

            # 7. 贴回原始帧
            logger.info("正在合成帧...")
            out_frames = []
            for i in range(len(orig_frames)):
                frame = orig_frames[i].copy()
                cbox = crop_boxes[i]
                gen = cv2.resize(gen_faces[i], (cbox[2] - cbox[0], cbox[3] - cbox[1]))
                frame[cbox[1]:cbox[3], cbox[0]:cbox[2]] = gen
                out_frames.append(frame)

            # 8. 写出视频
            logger.info(f"正在写出视频: {output_path}")
            temp_video = output_path + ".temp.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_video, fourcc, fps, (orig_w, orig_h))
            for f in out_frames:
                out.write(f)
            out.release()

            # 9. 添加音频
            import subprocess
            cmd = [
                "ffmpeg", "-y",
                "-i", temp_video,
                "-i", audio_path,
                "-c:v", "libx264",
                "-c:a", "aac",
                "-shortest",
                "-loglevel", "error",
                output_path
            ]
            subprocess.run(cmd, check=True)

            # 清理
            if os.path.exists(temp_video):
                os.remove(temp_video)

            logger.info(f"口型合成完成: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Wav2Lip 生成失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def unload(self):
        """卸载模型释放显存"""
        if self.model is not None:
            del self.model
            self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
