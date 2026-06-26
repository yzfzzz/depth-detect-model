#!/usr/bin/env python3
"""
更新 config.yaml 和 benchmark.yaml 中的 engine 路径。

用法:
    python update_config.py [--project-root /path/to/project]

说明:
    1. 自动扫描 model/engine/ 下所有 .engine 文件
    2. 按精度优先级 (int8 > fp16 > fp32) 选择最优引擎
    3. 更新 bin/config.yaml 中的 yolo_model_path / depth_model_path
    4. 更新 bin/benchmark.yaml 中的 YAML 锚点路径
"""

import argparse
import glob
import os
import re
import sys
from pathlib import Path


# -------------------- 工具函数 --------------------

def find_engines(engine_dir: str) -> dict[str, dict[str, str]]:
    """
    扫描 engine_dir,返回 {模型名: {精度: 绝对路径}}。
    例如: {"yolo26n": {"int8": "/abs/path/...", "fp16": "/abs/path/..."}, ...}
    """
    result: dict[str, dict[str, str]] = {}
    pattern = os.path.join(engine_dir, "**", "*.engine")
    for fpath in glob.glob(pattern, recursive=True):
        name = os.path.basename(fpath)
        # 从文件名推断模型名：截取到 _op11 或 _192x640 之前
        model = name.split("_192x640")[0].split("_640")[0]
        precision = "fp32"
        if "int8" in name:
            precision = "int8"
        elif "fp16" in name:
            precision = "fp16"

        result.setdefault(model, {})[precision] = os.path.abspath(fpath)
    return result

def find_onnx(onnx_dir: str) -> dict[str, str]:
    """
    扫描 onnx_dir,返回 {模型名: onnx绝对路径}。
    每个模型目录下可能有多个 op 版本,优先选择 op11。
    例如: {"yolo26n": "/abs/.../yolo26n_640_op11.onnx", ...}
    """
    result: dict[str, str] = {}
    pattern = os.path.join(onnx_dir, "**", "*.onnx")
    for fpath in glob.glob(pattern, recursive=True):
        model = os.path.basename(os.path.dirname(fpath))
        # 优先保留 op11；若已有 op11 则跳过其他版本
        if model not in result:
            result[model] = os.path.abspath(fpath)
        elif "op11" in os.path.basename(fpath) and "op11" not in os.path.basename(result[model]):
            result[model] = os.path.abspath(fpath)
    return result

def pick_best(precisions: dict[str, str]) -> str | None:
    """按 int8 > fp16 > fp32 返回最佳引擎路径"""
    for p in ("fp16", "fp32", "int8"):
        if p in precisions:
            return precisions[p]
    return None


def is_yolo(model: str) -> bool:
    return model.startswith("yolo")


def is_depth(model: str) -> bool:
    return not is_yolo(model)


# -------------------- config.yaml --------------------
def update_config_yaml(config_path: str,
                       engines: dict[str, dict[str, str]],
                       onnx_models: dict[str, str],
                       project_root: str) -> bool:
    """
    更新 bin/config.yaml 中 engine / onnx 类型的 path。
    使用行级替换，保留注释、缩进、字段顺序等全部原始格式。
    """
    import re

    # ---- 1. 计算出需要更新的新路径 ----
    new_paths: dict[str, str] = {}  # key = "yolo:engine", "yolo:onnx", "depth:engine", "depth:onnx"

    yolo_model = next((m for m in engines if is_yolo(m)), None)
    if yolo_model:
        best = pick_best(engines[yolo_model])
        if best:
            new_paths["yolo:engine"] = best

    yolo_onnx = next((m for m in onnx_models if is_yolo(m)), None)
    if yolo_onnx:
        new_paths["yolo:onnx"] = onnx_models[yolo_onnx]

    depth_model = next((m for m in engines if is_depth(m)), None)
    if depth_model:
        best = pick_best(engines[depth_model])
        if best:
            new_paths["depth:engine"] = best

    depth_onnx = next((m for m in onnx_models if is_depth(m)), None)
    if depth_onnx:
        new_paths["depth:onnx"] = onnx_models[depth_onnx]

    if not new_paths:
        return False

    # ---- 2. 逐行扫描，只替换 path 行 ----
    with open(config_path, "r") as f:
        lines = f.readlines()

    current_section: str | None = None   # "yolo" / "depth" / None
    current_type: str | None = None      # "engine" / "onnx" / None
    updated = False

    for i, line in enumerate(lines):
        stripped = line.strip()

        # 跟踪当前所处的一级 section
        if stripped.startswith("yolo:"):
            current_section = "yolo"
            current_type = None
        elif stripped.startswith("depth:"):
            current_section = "depth"
            current_type = None
        elif not line.startswith(" ") and not line.startswith("\t") and ":" in stripped:
            # 进入其他一级 key（如 display_manager / prefer / logger ...）
            current_section = None
            current_type = None

        # 检测当前列表项的类型标记
        m_type = re.match(r'\s*-?\s*type:\s*(engine|onnx)\s*', line)
        if m_type:
            current_type = m_type.group(1)

        # 命中 path 行 → 替换
        m_path = re.match(r'(\s*path:\s*)".*?"', line)
        if m_path and current_section and current_type:
            key = f"{current_section}:{current_type}"
            if key in new_paths:
                indent = m_path.group(1)
                new_line = f'{indent}"{new_paths[key]}"\n'
                if new_line != line:
                    lines[i] = new_line
                    updated = True
                    print(f"  [UPDATE] config.yaml {current_section} {current_type} → {new_paths[key]}")
            current_type = None  # 每个 type 只跟一个 path，用完复位

    if updated:
        with open(config_path, "w") as f:
            f.writelines(lines)

    return updated


# -------------------- benchmark.yaml --------------------

def update_benchmark_yaml(bench_path: str, engines: dict[str, dict[str, str]],
                          project_root: str) -> bool:
    """
    更新 bin/benchmark.yaml 中的 YAML 锚点路径。
    使用正则替换,保留锚点和别名结构不变。
    """
    with open(bench_path, "r") as f:
        content = f.read()

    updated = False
    rel_root = os.path.relpath(project_root, os.path.dirname(bench_path))

    for model, precisions in engines.items():
        for precision, abs_path in precisions.items():
            rel_path = os.path.relpath(abs_path, os.path.dirname(bench_path))
            # 构造锚点名称模式：_yolo_int8, _depth_fp16 等
            if is_yolo(model):
                anchor_name = f"_yolo_{precision}"
            else:
                anchor_name = f"_depth_{precision}"

            # 匹配: anchor_name: &anchor_name "旧路径"
            pattern = re.compile(
                rf'({anchor_name}\s*:)\s*&{anchor_name}\s*".*?"',
                re.MULTILINE,
            )
            replacement = rf'\1 &{anchor_name} "{rel_path}"'
            new_content = pattern.sub(replacement, content)
            if new_content != content:
                updated = True
                print(f"  [UPDATE] benchmark.yaml {anchor_name} → {rel_path}")
                content = new_content

    if updated:
        with open(bench_path, "w") as f:
            f.write(content)

    return updated


# -------------------- 主入口 --------------------

def main():
    parser = argparse.ArgumentParser(description="更新配置文件中的 engine 路径")
    parser.add_argument(
        "--project-root",
        default=None,
        help="项目根目录 (默认: 脚本所在目录的上一级, 即 start.sh 所在目录)",
    )
    args = parser.parse_args()

    # 推断项目根目录
    if args.project_root:
        project_root = os.path.abspath(args.project_root)
    else:
        # 脚本位于 model/update_config.py,项目根 = 上一级
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..")
        )

    engine_dir = os.path.join(project_root, "model", "engine")
    onnx_dir   = os.path.join(project_root, "model", "onnx")       # 新增
    config_yaml = os.path.join(project_root, "bin", "config.yaml")
    bench_yaml  = os.path.join(project_root, "bin", "benchmark.yaml")

    # 校验
    if not os.path.isdir(engine_dir):
        print(f"[ERROR] engine 目录不存在: {engine_dir}", file=sys.stderr)
        sys.exit(1)

    # 扫描引擎
    engines = find_engines(engine_dir)
    if not engines:
        print("[WARN] 未找到任何 .engine 文件，跳过配置更新")
        # 不 return，继续更新 onnx 路径

    # 扫描 onnx
    onnx_models: dict[str, str] = {}
    if os.path.isdir(onnx_dir):
        onnx_models = find_onnx(onnx_dir)
        if onnx_models:
            print(f"[INFO] 扫描到 {len(onnx_models)} 个 ONNX 文件:")
            for model, path in onnx_models.items():
                print(f"       {model} → {path}")

    # 更新 config.yaml
    if os.path.isfile(config_yaml):
        update_config_yaml(config_yaml, engines, onnx_models, project_root)  # 传入 onnx
    else:
        print(f"[WARN] config.yaml 不存在: {config_yaml}")

    # 更新 benchmark.yaml
    if os.path.isfile(bench_yaml):
        update_benchmark_yaml(bench_yaml, engines, project_root)
    else:
        print(f"[WARN] benchmark.yaml 不存在: {bench_yaml}")

    print("  [DONE] 配置文件更新完成")


if __name__ == "__main__":
    main()
