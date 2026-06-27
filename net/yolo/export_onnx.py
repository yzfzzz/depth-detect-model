import os
import argparse
import shutil
from ultralytics import YOLO

def export_yolo_to_onnx():
    parser = argparse.ArgumentParser(description="Export YOLO to ONNX")
    parser.add_argument("--weights", type=str, required=True, help="YOLO weights path, e.g. ./pth_model/yolo26n.pt")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size, e.g. 320")
    parser.add_argument("--opset_v", type=int, default=11, help="ONNX opset version")
    parser.add_argument("--onnx_save_path", type=str, default=None, help="Output ONNX path")
    parser.add_argument("--end2end", action="store_true", default=False, help="Export end2end model")
    args = parser.parse_args()

    print(f"Loading YOLO model from {args.weights}...")
    model = YOLO(args.weights, task="detect")
    model_name = os.path.splitext(os.path.basename(args.weights))[0]
    if args.end2end:
        model_name = model_name + "_end2end"
    onnx_save_path = args.onnx_save_path or f"../../onnx/{model_name}/{model_name}_{args.imgsz}_op{args.opset_v}.onnx"
    os.makedirs(os.path.dirname(onnx_save_path), exist_ok=True)

    print(f"Exporting to ONNX: {onnx_save_path}")
    exported_path = model.export(
        format="onnx",
        simplify=True,
        device="cpu",
        opset=args.opset_v,
        dynamic=False,
        imgsz=args.imgsz,
        end2end = args.end2end
    )

    exported_path = str(exported_path) if exported_path is not None else ""
    if exported_path and os.path.isfile(exported_path):
        source_path = exported_path
    else:
        default_export_dir = os.path.dirname(args.weights) if os.path.isfile(args.weights) else "."
        base_name = os.path.splitext(os.path.basename(args.weights))[0]
        source_path = os.path.join(default_export_dir, f"{base_name}.onnx")

    if not os.path.isfile(source_path):
        raise FileNotFoundError(f"Exported ONNX file not found: {source_path}")

    if os.path.abspath(source_path) != os.path.abspath(onnx_save_path):
        shutil.move(source_path, onnx_save_path)

    print(f"✅ Export finished. Saved to: {onnx_save_path}")

if __name__ == "__main__":
    export_yolo_to_onnx()