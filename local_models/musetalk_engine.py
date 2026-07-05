"""
MuseTalk 口型合成引擎封装
基于 TMElyralab/MuseTalk v1.5，用 face_alignment 替代 mmpose 依赖
"""
import os
import gc
import sys
import time
import math
import logging
import subprocess
from pathlib import Path
from typing import Optional, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# 确保 MuseTalk 源码可导入
_MUSETALK_ROOT = Path(__file__).resolve().parent.parent / "MuseTalk_repo"
if str(_MUSETALK_ROOT) not in sys.path:
    sys.path.insert(0, str(_MUSETALK_ROOT))


class MuseTalkEngine:
    """MuseTalk 口型合成引擎 — 高质量，~6-8GB 显存"""

    def __init__(self, models_dir: str = None):
        if models_dir is None:
            models_dir = str(Path(__file__).resolve().parent.parent / "pretrained_models" / "MuseTalk")
        self.models_dir = models_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.weight_dtype = torch.float16  # fp16 省显存

        # 模型组件
        self.vae = None
        self.unet = None
        self.pe = None
        self.whisper = None
        self.face_parser = None
        self.face_detector = None

        # 状态
        self._loaded = False

    # ------------------------------------------------------------------
    # 模型加载
    # ------------------------------------------------------------------

    def load(self) -> bool:
        """加载所有模型组件"""
        try:
            t0 = time.time()
            logger.info(f"加载 MuseTalk 模型，设备: {self.device}")

            # 1. VAE (SD VAE FT-MSE)
            logger.info("[MuseTalk] 加载 VAE...")
            from musetalk.models.vae import VAE
            vae_path = os.path.join(self.models_dir, "sd-vae-ft-mse")
            self.vae = VAE(model_path=vae_path, use_float16=True)
            self.vae.vae = self.vae.vae.to(self.device)
            logger.info(f"[MuseTalk] VAE 加载完成")

            # 2. UNet (优先 v1.5，回退 v1.0)
            from musetalk.models.unet import UNet, PositionalEncoding
            v15_config = os.path.join(self.models_dir, "musetalkV15", "musetalk.json")
            v15_model = os.path.join(self.models_dir, "musetalkV15", "unet.pth")
            v10_config = os.path.join(self.models_dir, "musetalk", "musetalk.json")
            v10_model = os.path.join(self.models_dir, "musetalk", "pytorch_model.bin")

            if os.path.exists(v15_model):
                self._version = "v1.5"
                unet_config, unet_model = v15_config, v15_model
            elif os.path.exists(v10_model):
                self._version = "v1.0"
                unet_config, unet_model = v10_config, v10_model
            else:
                raise FileNotFoundError(
                    f"UNet 模型缺失！请下载到 {self.models_dir}/musetalkV15/ 或 {self.models_dir}/musetalk/"
                )

            logger.info(f"[MuseTalk] 加载 UNet {self._version}...")
            self.unet = UNet(unet_config=unet_config, model_path=unet_model, use_float16=True, device=self.device)
            self.pe = PositionalEncoding(d_model=384).to(self.device, dtype=self.weight_dtype)
            logger.info(f"[MuseTalk] UNet {self._version} 加载完成")

            # 3. Whisper (音频特征提取)
            logger.info("[MuseTalk] 加载 Whisper...")
            whisper_dir = os.path.join(self.models_dir, "whisper")
            if not os.path.exists(os.path.join(whisper_dir, "pytorch_model.bin")):
                logger.info("[MuseTalk] Whisper 模型缺失，尝试自动下载...")
                try:
                    from huggingface_hub import snapshot_download
                    os.makedirs(whisper_dir, exist_ok=True)
                    snapshot_download(
                        'openai/whisper-tiny',
                        local_dir=whisper_dir,
                        allow_patterns=['pytorch_model.bin', 'config.json', 'preprocessor_config.json'],
                    )
                    logger.info("[MuseTalk] Whisper 下载完成")
                except Exception as e:
                    logger.warning(f"[MuseTalk] Whisper 自动下载失败: {e}")
                    logger.warning("[MuseTalk] 尝试从 modelscope 下载...")
                    try:
                        from modelscope import snapshot_download as ms_download
                        ms_download('AI-ModelScope/whisper-tiny', local_dir=whisper_dir)
                        logger.info("[MuseTalk] Whisper 从 modelscope 下载完成")
                    except Exception as e2:
                        raise FileNotFoundError(
                            f"无法下载 Whisper 模型: {e2}\n"
                            f"请手动下载到 {whisper_dir}"
                        )

            from transformers import WhisperModel
            self.whisper = WhisperModel.from_pretrained(whisper_dir)
            self.whisper = self.whisper.to(device=self.device, dtype=self.weight_dtype).eval()
            self.whisper.requires_grad_(False)
            logger.info(f"[MuseTalk] Whisper 加载完成")

            # 4. Face Parsing (BiSeNet — 面部语义分割用于融合)
            logger.info("[MuseTalk] 加载 FaceParsing...")
            # FaceParsing 默认从 ./models/face-parse-bisent/ 加载
            # 创建目录 junction 指向 pretrained_models/MuseTalk
            fp_model_dir = os.path.join(self.models_dir, "face-parse-bisent")
            models_link = os.path.join(str(_MUSETALK_ROOT.parent), "models")
            if not os.path.exists(models_link):
                try:
                    os.symlink(self.models_dir, models_link, target_is_directory=True)
                except OSError:
                    # Windows 需要管理员权限或开发者模式，fallback 到 junction
                    import subprocess as _sp
                    _sp.run(["mklink", "/J", models_link, self.models_dir], shell=True, check=False)

            from musetalk.utils.face_parsing import FaceParsing
            # 显式传绝对路径（避免上游默认的 ./models/... 相对 cwd 找不到）
            resnet_abs = os.path.join(fp_model_dir, "resnet18-5c106cde.pth")
            bisenet_abs = os.path.join(fp_model_dir, "79999_iter.pth")
            if not os.path.exists(resnet_abs):
                logger.error(f"[MuseTalk] FaceParsing resnet18 缺失: {resnet_abs}")
                raise FileNotFoundError(resnet_abs)
            if not os.path.exists(bisenet_abs):
                logger.error(f"[MuseTalk] FaceParsing BiSeNet 缺失: {bisenet_abs}")
                raise FileNotFoundError(bisenet_abs)
            # Monkey-patch 默认参数为绝对路径，再实例化
            # （上游 model_init 默认走 './models/face-parse-bisent/...' 相对 cwd，
            #  在 Gradio 进程下 cwd 不固定，必须显式覆盖）
            FaceParsing.model_init.__defaults__ = (resnet_abs, bisenet_abs)
            self.face_parser = FaceParsing()
            logger.info(f"[MuseTalk] FaceParsing 加载完成 (resnet={resnet_abs})")

            # 5. Face Detection: MuseTalk SFD (bbox only, 采样优化)
            logger.info("[MuseTalk] 加载 SFD face detector...")
            from musetalk.utils.face_detection import FaceAlignment as MuseFaceAlignment
            from musetalk.utils.face_detection import LandmarksType
            self.face_detector = MuseFaceAlignment(
                LandmarksType._2D, flip_input=False, device=self.device, face_detector='sfd',
            )
            logger.info(f"[MuseTalk] SFD face detector 加载完成")

            self._loaded = True
            elapsed = time.time() - t0
            vram = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
            logger.info(f"[MuseTalk] 全部加载完成，耗时 {elapsed:.1f}s，显存: {vram:.2f} GB")
            return True

        except FileNotFoundError as e:
            logger.error(f"[MuseTalk] 模型文件缺失: {e}")
            logger.error("请运行 _download_models.py 下载模型权重")
            return False
        except Exception as e:
            logger.error(f"[MuseTalk] 加载失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def unload(self):
        """释放显存"""
        for attr in ['vae', 'unet', 'pe', 'whisper', 'face_parser', 'face_detector']:
            obj = getattr(self, attr, None)
            if obj is not None:
                del obj
                setattr(self, attr, None)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._loaded = False

    # ------------------------------------------------------------------
    # 人脸检测 + 关键点 (替代 mmpose)
    # ------------------------------------------------------------------

    def _detect_all_faces(self, frames: List[np.ndarray]) -> Tuple[List, List]:
        """批量检测所有帧的人脸框（采样优化，跳帧检测）

        对于动画视频（慢速缩放），人脸位置变化极小，只需要采样检测。
        注：SFd 在 Blackwell 上 cuDNN 关闭，单帧推理比 batch 推理更快。
        """
        sample_step = max(1, len(frames) // 30)  # 最多采样 30 帧
        logger.info(f"  采样步长: {sample_step} (共需检测 {len(range(0, len(frames), sample_step))} 帧)")

        # 逐帧检测（cuDNN off 时单帧更快）
        sample_bboxes = {}
        for i in range(0, len(frames), sample_step):
            bboxes = self.face_detector.get_detections_for_batch(
                np.asarray([frames[i]])
            )
            sample_bboxes[i] = bboxes[0] if bboxes else None

        # 按最近采样点插值填充所有帧
        all_bboxes = []
        sample_indices = sorted(sample_bboxes.keys())
        for i in range(len(frames)):
            nearest = min(sample_indices, key=lambda s: abs(s - i))
            all_bboxes.append(sample_bboxes[nearest])

        valid_bbox = sum(1 for b in all_bboxes if b is not None)
        logger.info(f"  检测到人脸框: {valid_bbox}/{len(frames)}")

        # 简化的 landmarks: 从 bbox 估算鼻尖位置（避免慢速的 face_alignment）
        # bbox: (x1, y1, x2, y2), 鼻尖约在 y1 + 0.55*(y2-y1)
        all_landmarks = []
        for bbox in all_bboxes:
            if bbox is None:
                all_landmarks.append(None)
                continue
            x1, y1, x2, y2 = bbox
            nose_y = y1 + (y2 - y1) * 0.55
            # 构造简单的 landmark array (68点形式，只用到第29点=鼻尖)
            landmarks = np.zeros((68, 2), dtype=np.float32)
            landmarks[:, 0] = (x1 + x2) / 2  # cx
            landmarks[:, 1] = y2  # bottom
            landmarks[29] = [((x1 + x2) / 2), nose_y]  # 第29点 = 鼻尖
            all_landmarks.append(landmarks)

        valid_lm = sum(1 for l in all_landmarks if l is not None)
        logger.info(f"  总人脸数: {valid_lm}/{len(frames)}")

        return all_landmarks, all_bboxes

    def _get_face_bbox_from_landmarks(self, landmarks: np.ndarray, bbox: Tuple) -> Tuple:
        """基于 68 点 landmarks 计算用于裁剪的人脸框

        对应 MuseTalk 原始代码:
          face_land_mark = keypoints[23:91]  # 68个脸部关键点
          half_face_coord = face_land_mark[29]  # 鼻尖
          x1 = min_x, y1 = upper_bond, x2 = max_x, y2 = max_y
        """
        if landmarks is None:
            return bbox  # 回退到 bbox

        # 68点 landmarks 与 mmpose wholebody[23:91] 索引一致
        # [29] = 鼻尖, [30] = 上唇中心, [28] = 鼻梁
        nose_tip = landmarks[29]  # 鼻尖作为上半脸分界

        min_x = int(np.min(landmarks[:, 0]))
        max_x = int(np.max(landmarks[:, 0]))
        max_y = int(np.max(landmarks[:, 1]))

        # 计算上边界：从鼻尖到最高点的一半距离
        nose_to_top = nose_tip[1] - np.min(landmarks[:, 1])
        upper_bound = max(0, int(nose_tip[1] - nose_to_top * 0.5))

        return (min_x, upper_bound, max_x, max_y)

    # ------------------------------------------------------------------
    # 音频处理
    # ------------------------------------------------------------------

    def _extract_audio_features(self, audio_path: str, fps: int = 25):
        """提取 Whisper 音频特征"""
        from musetalk.utils.audio_processor import AudioProcessor

        processor = AudioProcessor(feature_extractor_path=os.path.join(self.models_dir, "whisper"))

        # 获取 audio feature
        whisper_input_features, librosa_length = processor.get_audio_feature(
            audio_path, weight_dtype=self.weight_dtype
        )

        # 生成 whisper chunks (每帧对应一个音频 chunk)
        whisper_chunks = processor.get_whisper_chunk(
            whisper_input_features,
            self.device,
            self.weight_dtype,
            self.whisper,
            librosa_length,
            fps=fps,
        )

        return whisper_chunks, librosa_length

    # ------------------------------------------------------------------
    # 主推理
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        video_path: str,
        audio_path: str,
        output_path: Optional[str] = None,
        fps: int = 25,
        bbox_shift: int = 0,
        batch_size: int = 8,
    ) -> Optional[str]:
        """
        MuseTalk 口型同步推理

        Args:
            video_path: 输入视频路径
            audio_path: 输入音频路径（任意采样率，内部会重采样到 16kHz）
            output_path: 输出路径，默认同目录 + _musetalk 后缀
            fps: 输出帧率
            bbox_shift: 手动调整嘴部区域（正=增大张嘴幅度）
            batch_size: 推理 batch size
        Returns:
            输出视频路径，失败返回 None
        """
        if not self._loaded:
            logger.error("[MuseTalk] 模型未加载，请先调用 load()")
            return None

        if output_path is None:
            stem = Path(video_path).stem
            output_path = str(Path(video_path).parent / f"{stem}_musetalk.mp4")

        try:
            t_start = time.time()

            # ── 1. 读取视频帧 ──
            logger.info("[MuseTalk] 读取视频帧...")
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"无法打开视频: {video_path}")
                return None

            # 内存优化：先看分辨率，超过 720x1280 主动降采样
            # 原因：5380 帧 720x1280 RGB 占用 ~14GB，叠加 res_frame_list 约 10GB
            # 家用机 (16GB 内存) 必然 OOM。降到 480x854 后降到 2.4GB，可承受
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            target_w, target_h = src_w, src_h
            if max(src_w, src_h) > 720:
                scale = 720.0 / max(src_w, src_h)
                target_w = int(src_w * scale) // 2 * 2  # 编码要求偶数
                target_h = int(src_h * scale) // 2 * 2
                logger.info(f"[MuseTalk] 分辨率从 {src_w}x{src_h} 降至 {target_w}x{target_h} 以节省内存")

            orig_frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if (frame.shape[1], frame.shape[0]) != (target_w, target_h):
                    frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
                orig_frames.append(frame)
            cap.release()
            orig_h, orig_w = orig_frames[0].shape[:2]
            logger.info(f"[MuseTalk] 读取 {len(orig_frames)} 帧, {orig_w}x{orig_h}")

            # ── 2. 提取音频特征 ──
            logger.info("[MuseTalk] 提取音频特征...")
            whisper_chunks, librosa_length = self._extract_audio_features(audio_path, fps=fps)
            logger.info(f"[MuseTalk] 音频特征: {whisper_chunks.shape}")

            # ── 3. 人脸检测 + 关键点 (替代 mmpose) ──
            logger.info("[MuseTalk] 人脸检测 (SFD + 批量)...")
            t_det = time.time()
            landmarks_list, bboxes_list = self._detect_all_faces(orig_frames)
            logger.info(f"[MuseTalk] 人脸检测完成，耗时 {time.time()-t_det:.1f}s")

            coord_list = []
            input_latent_list = []

            for i, frame in enumerate(orig_frames):
                landmarks = landmarks_list[i]
                bbox = bboxes_list[i]

                if landmarks is None or bbox is None:
                    # 使用上一帧的 bbox 或默认
                    if coord_list:
                        coord_list.append(coord_list[-1])
                    else:
                        coord_list.append((0, 0, min(256, orig_w), min(256, orig_h)))
                    empty_frame = np.zeros((256, 256, 3), dtype=np.uint8)
                    latents = self.vae.get_latents_for_unet(empty_frame)
                    input_latent_list.append(latents)
                    continue

                # 基于 landmarks 计算裁剪框
                face_bbox = self._get_face_bbox_from_landmarks(landmarks, bbox)
                x1, y1, x2, y2 = face_bbox

                if y2 - y1 <= 0 or x2 - x1 <= 0:
                    x1, y1, x2, y2 = bbox

                # 扩展下边界 (增加额外的嘴部区域)
                if y2 < frame.shape[0]:
                    y2 = min(y2 + 10, frame.shape[0])

                coord_list.append((x1, y1, x2, y2))

                # 裁剪并 encode 到 VAE latent (必须先 resize 到 256x256)
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    crop = np.zeros((256, 256, 3), dtype=np.uint8)
                else:
                    crop = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_LANCZOS4)
                latents = self.vae.get_latents_for_unet(crop)
                input_latent_list.append(latents)

            logger.info(f"[MuseTalk] 人脸检测完成，共 {len(coord_list)} 帧")

            # ── 4. 前后帧平滑 (循环扩展) ──
            coord_list_cycle = coord_list + coord_list[::-1]
            input_latent_list_cycle = input_latent_list + input_latent_list[::-1]

            # ── 5. UNet 推理 ──
            logger.info("[MuseTalk] UNet 推理...")
            from musetalk.utils.utils import datagen

            video_len = len(whisper_chunks)
            n_batches = int(np.ceil(float(video_len) / batch_size))
            gen = datagen(
                whisper_chunks=whisper_chunks,
                vae_encode_latents=input_latent_list_cycle,
                batch_size=batch_size,
                delay_frame=0,
                device=self.device,
            )

            res_frame_list = []
            timesteps = torch.tensor([0], device=self.device)

            for i, (whisper_batch, latent_batch) in enumerate(gen):
                audio_feature_batch = self.pe(whisper_batch)
                latent_batch = latent_batch.to(dtype=self.unet.model.dtype)

                pred_latents = self.unet.model(
                    latent_batch, timesteps, encoder_hidden_states=audio_feature_batch
                ).sample
                recon = self.vae.decode_latents(pred_latents)
                for res_frame in recon:
                    res_frame_list.append(res_frame)

                if (i + 1) % max(1, n_batches // 5) == 0:
                    logger.info(f"  UNet 推理进度: {i+1}/{n_batches} batches")

            logger.info(f"[MuseTalk] UNet 推理完成，生成 {len(res_frame_list)} 帧")

            # ── 6. 贴回原始帧 + 人脸融合 ──
            logger.info("[MuseTalk] 人脸融合...")
            from musetalk.utils.blending import get_image

            temp_img_dir = str(Path(output_path).parent / f"_musetalk_temp_{Path(video_path).stem}")
            os.makedirs(temp_img_dir, exist_ok=True)

            for i in range(len(res_frame_list)):
                idx = i % len(coord_list_cycle)
                bbox = coord_list_cycle[idx]
                ori_frame = orig_frames[idx % len(orig_frames)].copy()

                x1, y1, x2, y2 = bbox

                # Resize 回去
                gen_face = res_frame_list[i]
                gen_face_resized = cv2.resize(
                    gen_face, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LANCZOS4
                )

                # 融合
                try:
                    combined = get_image(ori_frame, gen_face_resized, [x1, y1, x2, y2], mode='jaw', fp=self.face_parser)
                except Exception as e:
                    logger.warning(f"融合帧 {i} 失败: {e}，使用简单替换")
                    ori_frame[y1:y2, x1:x2] = gen_face_resized
                    combined = ori_frame

                cv2.imwrite(os.path.join(temp_img_dir, f"{i:08d}.png"), combined)

                if (i + 1) % 50 == 0:
                    logger.info(f"  融合进度: {i+1}/{len(res_frame_list)}")

            # 关键：及时释放大数组，避免 OOM (5380 帧 * 256x256x3 ≈ 10GB)
            logger.info("[MuseTalk] 释放内存...")
            res_frame_list.clear()
            del res_frame_list
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            try:
                import psutil
                mem = psutil.virtual_memory()
                logger.info(f"[MuseTalk] 当前内存占用: {mem.percent:.1f}% ({mem.used / 1024**3:.1f}/{mem.total / 1024**3:.1f} GB)")
            except ImportError:
                pass

            # ── 7. 合成视频 ──
            logger.info(f"[MuseTalk] 合成视频...")
            temp_vid = output_path + ".temp.mp4"
            cmd_img2video = [
                "ffmpeg", "-y", "-v", "warning",
                "-r", str(fps),
                "-f", "image2",
                "-i", os.path.join(temp_img_dir, "%08d.png"),
                "-vcodec", "libx264",
                "-vf", "format=yuv420p",
                "-crf", "18",
                temp_vid,
            ]
            subprocess.run(cmd_img2video, check=True)

            cmd_audio = [
                "ffmpeg", "-y", "-v", "warning",
                "-i", audio_path,
                "-i", temp_vid,
                "-c:v", "copy",
                "-c:a", "aac",
                "-shortest",
                output_path,
            ]
            subprocess.run(cmd_audio, check=True)

            # ── 8. 清理临时文件 ──
            import shutil
            if os.path.exists(temp_vid):
                os.remove(temp_vid)
            shutil.rmtree(temp_img_dir, ignore_errors=True)

            elapsed = time.time() - t_start
            size_mb = os.path.getsize(output_path) / 1024**2
            logger.info(f"[MuseTalk] 完成! {output_path} ({size_mb:.1f} MB, {elapsed:.0f}s)")
            return output_path

        except Exception as e:
            logger.error(f"[MuseTalk] 推理失败: {e}")
            import traceback
            traceback.print_exc()
            return None


# ============================================================
# 测试入口
# ============================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')

    engine = MuseTalkEngine()
    if engine.load():
        # 使用现有测试数据
        test_video = os.path.join(
            os.path.dirname(__file__), "test_downloads", "img_animated.mp4"
        )
        test_audio = os.path.join(
            os.path.dirname(__file__), "test_downloads", "tts_16k.wav"
        )
        if os.path.exists(test_video) and os.path.exists(test_audio):
            output = os.path.join(os.path.dirname(__file__), "test_downloads", "test_musetalk.mp4")
            result = engine.generate(test_video, test_audio, output)
            print(f"结果: {result}")
        else:
            print(f"测试文件不存在: {test_video} or {test_audio}")
        engine.unload()
