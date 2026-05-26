#!/usr/bin/env python3
import argparse
import os
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

def build_engine_from_onnx(onnx_path, engine_path, workspace_mb=4096, fp16=False, batch=8):
    workspace_bytes = int(workspace_mb) * 1024 * 1024
    with trt.Logger(trt.Logger.INFO) as logger:
        builder = trt.Builder(logger)
        # EXPLICIT_BATCH flag (supported in TRT8/10)
        explicit_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(flags=explicit_flag)
        parser = trt.OnnxParser(network, logger)

        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                msg = '\n'.join(
                    f"ONNX parse error: {parser.get_error(i)}" for i in range(parser.num_errors)
                )
                raise RuntimeError(msg)

        config = builder.create_builder_config()
        if hasattr(config, 'max_workspace_size'):
            config.max_workspace_size = workspace_bytes
        else:
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)

        # FP16: only set if requested and platform supports it
        if fp16:
            if hasattr(trt, 'BuilderFlag'):
                config.set_flag(trt.BuilderFlag.FP16)
            else:
                # fallback (older API unlikely here)
                builder.fp16_mode = True

        # Handle dynamic shapes: if any input has -1 in shape, create optimization profile
        profile_added = False
        for i in range(network.num_inputs):
            inp = network.get_input(i)
            shape = inp.shape
            if any(d == -1 or d is None for d in shape):
                # Build simple profile: min=1, opt=batch, max=batch (can be adjusted)
                min_shape = tuple(1 if (d == -1 or d is None) else d for d in shape)
                opt_shape = tuple(batch if (d == -1 or d is None) else d for d in shape)
                max_shape = tuple(batch if (d == -1 or d is None) else d for d in shape)
                profile = builder.create_optimization_profile()
                profile.set_shape(inp.name, min_shape, opt_shape, max_shape)
                config.add_optimization_profile(profile)
                profile_added = True

        # If no dynamic dims, still may set max_batch_size for implicit batch (legacy)
        # Modern engines use profiles; builder.max_batch_size is ignored for EXPLICIT_BATCH networks.

        # Build engine
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
        print(f"Saved engine to {engine_path} (FP16={fp16}, workspace={workspace_mb}MB)")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build TensorRT engine from ONNX (supports TRT8/10)")
    p.add_argument("--onnx", required=True, help="Input ONNX model")
    p.add_argument("--saveEngine", required=False, default=None, help="Output engine file (optional)")
    p.add_argument("--fp16", action="store_true", help="Enable FP16")
    p.add_argument("--workspace", type=int, default=4096, help="Workspace size in MB")
    p.add_argument("--batch", type=int, default=1, help="Optimization profile batch size (bs)")
    args = p.parse_args()

    # 目标命名规则：engine/<model_dir>/<onnx_base>_<fp16|fp32>_trt<major.minor>.engine
    trt_ver_raw = getattr(trt, "__version__", "unknown")
    trt_ver_short = "unknown"
    if trt_ver_raw and trt_ver_raw != "unknown":
        parts = trt_ver_raw.split(".")
        if len(parts) >= 2:
            trt_ver_short = f"{parts[0]}.{parts[1]}"
        else:
            trt_ver_short = parts[0]

    mode = "fp16" if args.fp16 else "fp32"

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

    build_engine_from_onnx(args.onnx, engine_path, workspace_mb=args.workspace, fp16=args.fp16, batch=args.batch)