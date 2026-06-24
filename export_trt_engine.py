#!/usr/bin/env python3
import argparse
import os
import numpy as np
import tensorrt as trt
import cv2
import pycuda.driver as cuda
import pycuda.autoinit

TRT_LOGGER = trt.Logger(trt.Logger.INFO)


# ============================================================
# INT8 校准器：从视频文件中按间隔抽取帧
# ============================================================
class VideoInt8Calibrator(trt.IInt8EntropyCalibrator2):
    """
    从视频文件中按间隔抽取帧，做与推理一致的预处理后提供给 TensorRT 做 INT8 校准。
    修复 TRT10 Bug: 使用 pycuda 分配 Device 显存，并在 get_batch 中返回 GPU 指针。
    """

    def __init__(
        self,
        video_path: str,
        input_name: str,
        input_shape: tuple,
        cache_file: str = "calibration.cache",
        batch_size: int = 1,
        max_frames: int = 500,
        skip_interval: int = 30,
        preprocess_mode: str = "depth",
        mean: tuple = (0.485, 0.456, 0.406),
        std: tuple = (0.229, 0.224, 0.225),
    ):
        super().__init__()
        self.video_path = video_path
        self.input_name = input_name
        self.cache_file = cache_file
        self.batch_size = batch_size
        self.max_frames = max_frames
        self.skip_interval = skip_interval
        self.preprocess_mode = preprocess_mode.lower()

        self.mean = np.array(mean, dtype=np.float32).reshape(-1, 1, 1)
        self.std = np.array(std, dtype=np.float32).reshape(-1, 1, 1)

        # 解析输入尺寸
        if len(input_shape) == 4:
            _, c, h, w = input_shape
        else:
            c, h, w = input_shape
        self.c, self.h, self.w = c, h, w

        # 打开视频获取总帧数
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频文件: {video_path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        # 每隔 skip_interval 取一帧，最多 max_frames 帧
        self.frame_indices = list(range(0, total_frames, skip_interval))[:max_frames]
        if not self.frame_indices:
            raise RuntimeError(f"视频 {video_path} 没有可抽取的帧 (total={total_frames})")

        self.num_frames = len(self.frame_indices)
        self.current_index = 0

        print(f"[INT8 Calibrator] 视频: {video_path}")
        print(f"  总帧数={total_frames}, FPS={fps:.1f}, 采样={self.num_frames}帧 "
              f"(间隔={skip_interval}, 上限={max_frames})")
        print(f"  输入形状=({c},{h},{w}), 预处理模式={self.preprocess_mode}, batch={batch_size}")
        if self.preprocess_mode == "depth":
            print(f"  mean={mean}, std={std}")

        # 预分配 Host 和 Device 内存 (修复 TRT10 核心逻辑)
        self.batch_buffer = np.empty((batch_size, c, h, w), dtype=np.float32)
        self.d_input = cuda.mem_alloc(self.batch_buffer.nbytes)
        
        self._cap = None  # 懒打开

    # ---------- IInt8EntropyCalibrator2 接口 ----------

    def get_batch_size(self) -> int:
        return self.batch_size

    def get_batch(self, names):
        if self.current_index >= self.num_frames:
            return None

        if self._cap is None:
            self._cap = cv2.VideoCapture(self.video_path)
            if not self._cap.isOpened():
                raise RuntimeError(f"无法打开视频: {self.video_path}")

        batch_filled = 0
        for i in range(self.batch_size):
            if self.current_index >= self.num_frames:
                break

            frame_idx = self.frame_indices[self.current_index]
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self._cap.read()
            self.current_index += 1

            if not ret or frame is None:
                continue

            processed = self._preprocess(frame)
            self.batch_buffer[i] = processed
            batch_filled += 1

        if batch_filled == 0:
            return None

        # 不足 batch_size 时用 0 填充
        if batch_filled < self.batch_size:
            self.batch_buffer[batch_filled:] = 0

        # 将 Host 数据拷贝到 Device (修复 TRT10 核心逻辑)
        cuda.memcpy_htod(self.d_input, self.batch_buffer)
        
        # 关键修复：返回 GPU 设备指针的整型地址，而非 NumPy 数组
        return [int(self.d_input)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            print(f"[INT8 Calibrator] 加载校准缓存: {self.cache_file}")
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
        print(f"[INT8 Calibrator] 校准缓存已保存: {self.cache_file}")

    # ---------- 预处理 ----------

    def _preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        if self.preprocess_mode == "yolo":
            return self._preprocess_yolo(frame_bgr)
        elif self.preprocess_mode == "depth":
            return self._preprocess_depth(frame_bgr)
        else:
            raise ValueError(f"未知预处理模式: {self.preprocess_mode}")

    def _preprocess_yolo(self, frame_bgr: np.ndarray) -> np.ndarray:
        ih, iw = self.h, self.w
        h0, w0 = frame_bgr.shape[:2]

        scale = min(iw / w0, ih / h0)
        new_w, new_h = int(w0 * scale), int(h0 * scale)
        resized = cv2.resize(frame_bgr, (new_w, new_h))

        pad_h = (ih - new_h) // 2
        pad_w = (iw - new_w) // 2
        padded = cv2.copyMakeBorder(
            resized, pad_h, ih - new_h - pad_h, pad_w, iw - new_w - pad_w,
            cv2.BORDER_CONSTANT, value=(128, 128, 128)
        )

        out = np.empty((self.c, ih, iw), dtype=np.float32)
        for c in range(3):
            out[c] = padded[:, :, 2 - c].astype(np.float32) / 255.0
        return out

    def _preprocess_depth(self, frame_bgr: np.ndarray) -> np.ndarray:
        resized = cv2.resize(frame_bgr, (self.w, self.h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        out = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        out = (out - self.mean) / self.std
        return out.astype(np.float32)
        
    def __del__(self):
        if self._cap is not None:
            self._cap.release()


# ============================================================
# 主构建函数
# ============================================================
def build_engine_from_onnx(
    onnx_path,
    engine_path,
    workspace_mb=4096,
    fp16=False,
    int8=False,
    calib_video=None,
    calib_cache=None,
    calib_batch=1,
    calib_frames=500,
    calib_skip=30,
    preprocess_mode="depth",
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    batch=8,
):
    workspace_bytes = int(workspace_mb) * 1024 * 1024

    if int8 and not calib_video:
        raise ValueError("启用 INT8 时必须指定 --calibVideo（校准视频路径）")

    with trt.Logger(trt.Logger.INFO) as logger:
        builder = trt.Builder(logger)
        explicit_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(flags=explicit_flag)
        parser = trt.OnnxParser(network, logger)

        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                msg = "\n".join(
                    f"ONNX parse error: {parser.get_error(i)}"
                    for i in range(parser.num_errors)
                )
                raise RuntimeError(msg)

        config = builder.create_builder_config()
        
        # 修复 TRT10 Bug: 兼容新旧 API 设置 workspace
        if hasattr(config, "set_memory_pool_limit"):
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
        else:
            config.max_workspace_size = workspace_bytes

        # ---- INT8 ----
        if int8:
            if hasattr(trt, "BuilderFlag"):
                config.set_flag(trt.BuilderFlag.INT8)
            else:
                builder.int8_mode = True

            inp = network.get_input(0)
            input_shape = tuple(inp.shape)
            resolved_shape = tuple(
                batch if (d == -1 or d is None) else d for d in input_shape
            )

            if calib_cache is None:
                calib_cache = os.path.splitext(engine_path)[0] + ".cache"

            calibrator = VideoInt8Calibrator(
                video_path=calib_video,
                input_name=inp.name,
                input_shape=resolved_shape,
                cache_file=calib_cache,
                batch_size=calib_batch,
                max_frames=calib_frames,
                skip_interval=calib_skip,
                preprocess_mode=preprocess_mode,
                mean=mean,
                std=std,
            )
            config.int8_calibrator = calibrator

        # ---- FP16 ----
        if fp16:
            if hasattr(trt, "BuilderFlag"):
                config.set_flag(trt.BuilderFlag.FP16)
            else:
                builder.fp16_mode = True

        # ---- 动态 Shape Profile ----
        for i in range(network.num_inputs):
            inp = network.get_input(i)
            shape = inp.shape
            if any(d == -1 or d is None for d in shape):
                min_shape = tuple(1 if (d == -1 or d is None) else d for d in shape)
                opt_shape = tuple(batch if (d == -1 or d is None) else d for d in shape)
                max_shape = tuple(batch if (d == -1 or d is None) else d for d in shape)
                profile = builder.create_optimization_profile()
                profile.set_shape(inp.name, min_shape, opt_shape, max_shape)
                config.add_optimization_profile(profile)

        # ---- 构建 Engine (修复 TRT10 Bug: 统一使用 build_serialized_network) ----
        if hasattr(builder, "build_serialized_network"):
            serialized_engine = builder.build_serialized_network(network, config)
            if serialized_engine is None:
                raise RuntimeError("Failed to build the TensorRT engine")
            with open(engine_path, "wb") as f:
                f.write(serialized_engine)
        else:
            engine = builder.build_engine(network, config)
            if engine is None:
                raise RuntimeError("Failed to build the TensorRT engine")
            with open(engine_path, "wb") as f:
                f.write(engine.serialize())

        precision_str = "int8" if int8 else ("fp16" if fp16 else "fp32")
        print(f"Saved engine to {engine_path} "
              f"(precision={precision_str}, workspace={workspace_mb}MB)")


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Build TensorRT engine from ONNX (supports FP16, INT8 with video calibration)"
    )
    p.add_argument("--onnx", required=True, help="Input ONNX model")
    p.add_argument("--saveEngine", default=None, help="Output engine file (optional)")
    p.add_argument("--fp16", action="store_true", help="Enable FP16")
    p.add_argument("--int8", action="store_true",
                   help="Enable INT8 quantization (requires --calibVideo)")
    p.add_argument("--calibVideo", default=None,
                   help="Calibration video path for INT8")
    p.add_argument("--calibCache", default=None,
                   help="Calibration cache file (auto-generated if not set)")
    p.add_argument("--calibBatch", type=int, default=1,
                   help="Batch size for INT8 calibration")
    p.add_argument("--calibFrames", type=int, default=500,
                   help="Max frames to sample from video for calibration")
    p.add_argument("--calibSkip", type=int, default=30,
                   help="Sample every N frames from video")
    p.add_argument("--preprocess", choices=["yolo", "depth"], default="depth",
                   help="Preprocessing mode: yolo (letterbox+pad) or depth (resize+normalize)")
    p.add_argument("--mean", type=float, nargs=3, default=(0.485, 0.456, 0.406),
                   help="Normalization mean (R G B), used in depth mode")
    p.add_argument("--std", type=float, nargs=3, default=(0.229, 0.224, 0.225),
                   help="Normalization std (R G B), used in depth mode")
    p.add_argument("--workspace", type=int, default=4096, help="Workspace size in MB")
    p.add_argument("--batch", type=int, default=1,
                   help="Optimization profile batch size")
    args = p.parse_args()

    trt_ver_raw = getattr(trt, "__version__", "unknown")
    trt_ver_short = "unknown"
    if trt_ver_raw and trt_ver_raw != "unknown":
        parts = trt_ver_raw.split(".")
        if len(parts) >= 2:
            trt_ver_short = f"{parts[0]}.{parts[1]}"
        else:
            trt_ver_short = parts[0]

    if args.int8:
        mode = "int8"
    elif args.fp16:
        mode = "fp16"
    else:
        mode = "fp32"

    if args.saveEngine:
        engine_path = args.saveEngine
    else:
        onnx_norm = os.path.normpath(args.onnx)
        model_dir = os.path.basename(os.path.dirname(onnx_norm)) or "model"
        onnx_base = os.path.splitext(os.path.basename(onnx_norm))[0]
        engine_dir = os.path.join("engine", model_dir)
        os.makedirs(engine_dir, exist_ok=True)
        engine_name = f"{onnx_base}_{mode}_trt{trt_ver_short}.engine"
        engine_path = os.path.join(engine_dir, engine_name)

    build_engine_from_onnx(
        onnx_path=args.onnx,
        engine_path=engine_path,
        workspace_mb=args.workspace,
        fp16=args.fp16,
        int8=args.int8,
        calib_video=args.calibVideo,
        calib_cache=args.calibCache,
        calib_batch=args.calibBatch,
        calib_frames=args.calibFrames,
        calib_skip=args.calibSkip,
        preprocess_mode=args.preprocess,
        mean=tuple(args.mean),
        std=tuple(args.std),
        batch=args.batch,
    )
