import os
import argparse
import torch
import networks
import onnx
import onnxruntime as ort
import numpy as np

def export_lite_mono_to_onnx():
    # ================= 基本配置 =================
    parser = argparse.ArgumentParser(description="Export Lite-Mono to ONNX")
    parser.add_argument("--model_name", type=str, default="lite-mono-tiny", help="Model name used to build the encoder and output filename")
    parser.add_argument("--weights_folder", type=str, default=None, help="Folder containing encoder.pth and depth.pth")
    parser.add_argument("--feed_height", type=int, default=192, help="Input image height used for export")
    parser.add_argument("--feed_width", type=int, default=640, help="Input image width used for export")
    parser.add_argument("--opset_v", type=int, default=11, help="ONNX opset version")
    parser.add_argument("--onnx_save_path", type=str, default=None, help="Output path for the exported ONNX file")
    args = parser.parse_args()

    model_name = args.model_name
    weights_folder = args.weights_folder or f"../pth_model/{model_name}_640x192" 
    
    encoder_path = os.path.join(weights_folder, "encoder.pth")
    decoder_path = os.path.join(weights_folder, "depth.pth")

    print(f"Loading weights from {weights_folder}...")
    # 注意：添加 weights_only=True 避免未来版本的安全警告
    encoder_dict = torch.load(encoder_path, map_location='cpu', weights_only=True)
    decoder_dict = torch.load(decoder_path, map_location='cpu', weights_only=True)

    feed_height = args.feed_height
    feed_width = args.feed_width

    
    # ================= 实例化网络 =================
    encoder = networks.LiteMono(model=model_name, height=feed_height, width=feed_width)
    model_dict = encoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
    encoder.eval()

    depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, scales=range(3))
    depth_model_dict = depth_decoder.state_dict()
    depth_decoder.load_state_dict({k: v for k, v in decoder_dict.items() if k in depth_model_dict})
    depth_decoder.eval()

    # ================= 构建端到端模型 =================
    class LiteMonoEnd2End(torch.nn.Module):
        def __init__(self, encoder, decoder):
            super(LiteMonoEnd2End, self).__init__()
            self.encoder = encoder
            self.decoder = decoder

        def forward(self, x):
            features = self.encoder(x)
            outputs = self.decoder(features)
            disp = outputs[("disp", 0)]
            return disp

    end_to_end_model = LiteMonoEnd2End(encoder, depth_decoder)
    end_to_end_model.eval()

    # ================= 导出 ONNX =================
    print(f"Exporting to ONNX with input shape: [1, 3, {feed_height}, {feed_width}]")
    dummy_input = torch.randn(1, 3, feed_height, feed_width, device='cpu')

    onnx_save_path = args.onnx_save_path or f"../onnx/{model_name}/{model_name}_{feed_height}x{feed_width}_op{args.opset_v}.onnx"
    
    os.makedirs(os.path.dirname(onnx_save_path), exist_ok=True)

    
    # 核心修复：禁用 dynamo，强制走传统 trace 路径，确保 opset 12 原生生效
    torch.onnx.export(
        end_to_end_model,                
        dummy_input,                     
        onnx_save_path,                  
        export_params=True,              
        opset_version=args.opset_v,                # 默认 12，兼容 Nano 的 TRT 8.x
        do_constant_folding=True,        
        input_names=['input'],           
        output_names=['disp_output'],    
    )
    print(f"ONNX model saved successfully to: {onnx_save_path}")

    # ================= 模型合法性检查与 Shape 推断修复 =================
    print("Running ONNX checker and shape inference...")
    try:
        onnx_model = onnx.load(onnx_save_path)
        # 必须进行形状推断，修复可能的动态形状残留，保障 TensorRT 解析成功[7](@ref)
        from onnx import shape_inference
        inferred_model = shape_inference.infer_shapes(onnx_model)
        onnx.save(inferred_model, onnx_save_path)
        
        onnx.checker.check_model(inferred_model, full_check=True)
        print("✅ ONNX model check passed and shapes are inferred.")
    except Exception as e:
        print(f"❌ ONNX model check failed: {e}")

    # ================= 数值对齐验证 (PyTorch vs ONNX Runtime) =================
    print("Validating numerical alignment between PyTorch and ONNX Runtime...")
    with torch.no_grad():
        pytorch_output = end_to_end_model(dummy_input).numpy()
    
    ort_session = ort.InferenceSession(onnx_save_path)
    ort_inputs = {'input': dummy_input.numpy()}
    ort_output = ort_session.run(None, ort_inputs)
    
    # 计算误差
    mean_abs_error = np.mean(np.abs(pytorch_output - ort_output))
    max_abs_error = np.max(np.abs(pytorch_output - ort_output))
    print(f"Mean Absolute Error: {mean_abs_error:.6f}")
    print(f"Max Absolute Error:  {max_abs_error:.6f}")
    if mean_abs_error < 1e-5:
        print("✅ Numerical alignment test passed! Model is ready for Jetson Nano.")
    else:
        print("⚠️ Warning: Significant numerical difference detected!")

    from onnxsim import simplify

    # 简化 ONNX 模型
    try:
        simplified_model, check = simplify(onnx_save_path)
        onnx.save(simplified_model, onnx_save_path)
        print("✅ ONNX model simplified successfully.")
    except Exception as e:
        print(f"❌ Failed to simplify ONNX model: {e}")

if __name__ == '__main__':
    export_lite_mono_to_onnx()
