#!/usr/bin/env python3
"""Draw and save the first-frame object mask for a FoundationPose sequence."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    scene = parse_args().scene.resolve()
    image_path = scene / "rgb" / "000000.png"
    mask_path = scene / "masks" / "000000.png"
    preview_path = scene / "check_mask_overlay.png"

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"无法读取首帧：{image_path}")

    points: list[tuple[int, int]] = []
    display = image.copy()

    def redraw() -> None:
        nonlocal display
        display = image.copy()
        for point in points:
            cv2.circle(display, point, 4, (0, 0, 255), -1)
        if len(points) >= 2:
            cv2.polylines(
                display,
                [np.asarray(points, dtype=np.int32)],
                False,
                (0, 255, 0),
                2,
            )
        cv2.putText(
            display,
            "Left click: point | Z: undo | S: save | Q: quit",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
        )

    def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        del flags, param
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            redraw()

    cv2.namedWindow("FoundationPose Mask Annotation", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("FoundationPose Mask Annotation", on_mouse)
    redraw()

    saved = False
    while True:
        cv2.imshow("FoundationPose Mask Annotation", display)
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("z"), ord("Z")):
            if points:
                points.pop()
                redraw()
        elif key in (ord("s"), ord("S")):
            if len(points) < 3:
                print("至少需要三个标注点。")
                continue
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [np.asarray(points, dtype=np.int32)], 255)
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(mask_path), mask):
                raise RuntimeError(f"Mask 保存失败：{mask_path}")
            overlay = image.copy()
            inside = mask > 0
            overlay[inside] = (
                0.5 * overlay[inside] + 0.5 * np.array([0, 255, 0])
            ).astype(np.uint8)
            cv2.imwrite(str(preview_path), overlay)
            saved = True
            break
        elif key in (ord("q"), ord("Q")):
            break

    cv2.destroyAllWindows()
    if not saved:
        raise RuntimeError("已取消标注，未启动 FoundationPose。")

    print(f"Mask 已保存：{mask_path}")
    print(f"检查图已保存：{preview_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
