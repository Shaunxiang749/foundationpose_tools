#!/usr/bin/env python3
"""Validate a captured FoundationPose RGB/depth sequence and optional mask."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--require-mask", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scene = args.scene.resolve()
    rgb_files = sorted((scene / "rgb").glob("*.png"))
    depth_files = sorted((scene / "depth").glob("*.png"))

    if not rgb_files:
        raise RuntimeError("没有找到 RGB 帧。")
    if len(rgb_files) != len(depth_files):
        raise RuntimeError(
            f"RGB/Depth 数量不一致：{len(rgb_files)}/{len(depth_files)}"
        )

    expected = [f"{index:06d}" for index in range(len(rgb_files))]
    if [path.stem for path in rgb_files] != expected:
        raise RuntimeError("RGB 帧编号不连续。")
    if [path.stem for path in depth_files] != expected:
        raise RuntimeError("Depth 帧编号不连续。")

    camera_matrix = np.loadtxt(scene / "cam_K.txt").reshape(3, 3)
    if not np.isfinite(camera_matrix).all() or camera_matrix[2, 2] != 1.0:
        raise RuntimeError("cam_K.txt 无效。")

    rgb = cv2.imread(str(rgb_files[0]), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(depth_files[0]), cv2.IMREAD_UNCHANGED)
    if rgb is None or depth is None:
        raise RuntimeError("首帧 RGB 或 Depth 无法读取。")
    if rgb.shape[:2] != depth.shape[:2]:
        raise RuntimeError(f"RGB/Depth 尺寸不一致：{rgb.shape}/{depth.shape}")
    if rgb.dtype != np.uint8 or depth.dtype != np.uint16:
        raise RuntimeError(f"数据类型错误：RGB={rgb.dtype}, Depth={depth.dtype}")

    print(f"数据检查通过：{len(rgb_files)} 帧，{rgb.shape[1]}x{rgb.shape[0]}")
    print("K =")
    print(camera_matrix)

    if args.require_mask:
        mask_path = scene / "masks" / "000000.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != depth.shape:
            raise RuntimeError("首帧 mask 缺失或尺寸错误。")
        foreground = mask > 0
        count = int(foreground.sum())
        if count < 100:
            raise RuntimeError(f"Mask 前景像素过少：{count}")
        valid = foreground & (depth > 0)
        support = float(valid.sum() / count)
        if support < 0.8:
            raise RuntimeError(f"Mask 内有效深度比例过低：{support:.1%}")
        print(f"Mask 检查通过：{count} 像素，有效深度 {support:.1%}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
