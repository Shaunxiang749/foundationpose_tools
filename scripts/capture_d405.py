#!/usr/bin/env python3
"""Capture aligned RealSense D405 RGB/depth frames and color intrinsics."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=30)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    rgb_dir = output / "rgb"
    depth_dir = output / "depth"

    existing = list(rgb_dir.glob("*.png")) + list(depth_dir.glob("*.png"))
    if existing or (output / "cam_K.txt").exists():
        raise RuntimeError(f"输出目录已有录制数据，拒绝覆盖：{output}")

    devices = list(rs.context().query_devices())
    if not devices:
        raise RuntimeError("未检测到 RealSense 相机，请检查 USB 连接。")

    device = devices[0]
    print("检测到相机：", device.get_info(rs.camera_info.name))
    print("序列号：", device.get_info(rs.camera_info.serial_number))

    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(
        rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps
    )
    config.enable_stream(
        rs.stream.depth, args.width, args.height, rs.format.z16, args.fps
    )

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_profile.get_intrinsics()
    camera_matrix = np.array(
        [[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    np.savetxt(output / "cam_K.txt", camera_matrix, fmt="%.8f")

    print(f"Depth scale: {depth_scale:.9f} m/unit")
    print("Color camera K:")
    print(camera_matrix)
    print("\nSPACE：开始/暂停录制；Q：结束并保存")

    frame_id = 0
    recording = False

    try:
        for _ in range(args.warmup):
            pipeline.wait_for_frames()

        while True:
            frames = align.process(pipeline.wait_for_frames())
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color = np.asanyarray(color_frame.get_data())
            depth_raw = np.asanyarray(depth_frame.get_data())
            depth_mm = np.clip(
                depth_raw.astype(np.float32) * depth_scale * 1000.0,
                0,
                65535,
            ).astype(np.uint16)

            display = color.copy()
            if recording:
                filename = f"{frame_id:06d}.png"
                if not cv2.imwrite(str(rgb_dir / filename), color):
                    raise RuntimeError(f"RGB 保存失败：{filename}")
                if not cv2.imwrite(str(depth_dir / filename), depth_mm):
                    raise RuntimeError(f"Depth 保存失败：{filename}")
                frame_id += 1
                label = f"REC  frame={frame_id}"
                label_color = (0, 0, 255)
            else:
                label = f"PAUSED  saved={frame_id}"
                label_color = (0, 255, 255)

            cv2.putText(
                display,
                label,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                label_color,
                2,
            )
            cv2.imshow("FoundationPose D405 Capture", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(" "):
                recording = not recording
                print("录制开始" if recording else f"录制暂停，已保存 {frame_id} 帧")
            elif key in (ord("q"), ord("Q")):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    if frame_id == 0:
        raise RuntimeError("没有录制任何帧。")

    print(f"录制完成：{frame_id} 帧")
    print(f"数据目录：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
