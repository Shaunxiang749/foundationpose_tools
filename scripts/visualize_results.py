#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


# ============================================================
# 读取 analyze_experiment.py 生成的 CSV
# ============================================================

def load_report(csv_path):
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"找不到误差报告：{csv_path}\n"
            "请先运行 analyze_experiment.py"
        )

    results = []

    with csv_path.open("r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            results.append({
                "frame": row["frame"],
                "stem": Path(row["frame"]).stem,
                "translation_mm": float(row["translation_mm"]),
                "rotation_deg": float(row["rotation_deg"]),
            })

    if not results:
        raise RuntimeError("CSV 中没有有效数据")

    return results


# ============================================================
# 查找对应图片
# ============================================================

def find_image(directory, stem):
    directory = Path(directory)

    for ext in [".png", ".jpg", ".jpeg"]:
        path = directory / f"{stem}{ext}"

        if path.exists():
            return path

    return None


# ============================================================
# 平移误差曲线
# ============================================================

def plot_translation(results, output_path):
    frames = list(range(len(results)))

    errors = [
        x["translation_mm"]
        for x in results
    ]

    plt.figure(figsize=(10, 5))

    plt.plot(
        frames,
        errors,
        marker="o",
        markersize=3
    )

    plt.xlabel("Frame")
    plt.ylabel("Translation Error (mm)")
    plt.title("FoundationPose Translation Error")

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=160
    )

    plt.close()

    print(
        f"[OK] 平移误差曲线：{output_path}"
    )


# ============================================================
# 旋转误差曲线
# ============================================================

def plot_rotation(results, output_path):
    frames = list(range(len(results)))

    errors = [
        x["rotation_deg"]
        for x in results
    ]

    plt.figure(figsize=(10, 5))

    plt.plot(
        frames,
        errors,
        marker="o",
        markersize=3
    )

    plt.xlabel("Frame")
    plt.ylabel("Rotation Error (deg)")
    plt.title("FoundationPose Rotation Error")

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=160
    )

    plt.close()

    print(
        f"[OK] 旋转误差曲线：{output_path}"
    )


# ============================================================
# 图片尺寸适配
# ============================================================

def resize_keep_ratio(image, target_width):
    w, h = image.size

    scale = target_width / w

    new_h = int(h * scale)

    return image.resize(
        (target_width, new_h)
    )


# ============================================================
# 生成 worst frame 对比图
# ============================================================

def make_worst_frame_sheet(
    results,
    data_dir,
    result_dir,
    output_path,
    topk=5
):
    data_dir = Path(data_dir)
    result_dir = Path(result_dir)

    rgb_dir = data_dir / "rgb"
    vis_dir = result_dir / "track_vis"

    # 按旋转误差排序
    worst = sorted(
        results,
        key=lambda x: x["rotation_deg"],
        reverse=True
    )[:topk]

    image_width = 500
    text_height = 80

    rows = []

    for item in worst:

        rgb_path = find_image(
            rgb_dir,
            item["stem"]
        )

        vis_path = find_image(
            vis_dir,
            item["stem"]
        )

        if rgb_path is None or vis_path is None:
            print(
                f"[WARN] 找不到 {item['stem']} 对应图片"
            )
            continue

        rgb = Image.open(
            rgb_path
        ).convert("RGB")

        vis = Image.open(
            vis_path
        ).convert("RGB")

        rgb = resize_keep_ratio(
            rgb,
            image_width
        )

        vis = resize_keep_ratio(
            vis,
            image_width
        )

        row_height = max(
            rgb.height,
            vis.height
        )

        row = Image.new(
            "RGB",
            (
                image_width * 2,
                row_height + text_height
            ),
            "white"
        )

        row.paste(
            rgb,
            (0, text_height)
        )

        row.paste(
            vis,
            (image_width, text_height)
        )

        draw = ImageDraw.Draw(row)

        text = (
            f"Frame: {item['stem']}    "
            f"Translation: "
            f"{item['translation_mm']:.3f} mm    "
            f"Rotation: "
            f"{item['rotation_deg']:.3f} deg"
        )

        draw.text(
            (10, 10),
            text,
            fill="black"
        )

        draw.text(
            (10, 40),
            "RGB",
            fill="black"
        )

        draw.text(
            (image_width + 10, 40),
            "FoundationPose Track Visualization",
            fill="black"
        )

        rows.append(row)

    if not rows:
        print(
            "[WARN] 没有可生成的 worst frame 图片"
        )
        return

    total_height = sum(
        x.height
        for x in rows
    )

    canvas = Image.new(
        "RGB",
        (
            image_width * 2,
            total_height
        ),
        "white"
    )

    y = 0

    for row in rows:

        canvas.paste(
            row,
            (0, y)
        )

        y += row.height

    canvas.save(output_path)

    print(
        f"[OK] Worst frame 对比图：{output_path}"
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "FoundationPose result visualization tool"
        )
    )

    parser.add_argument(
        "--data",
        required=True,
        help="Dataset directory"
    )

    parser.add_argument(
        "--result",
        required=True,
        help="FoundationPose result directory"
    )

    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="Number of worst frames"
    )

    args = parser.parse_args()

    data_dir = Path(args.data)
    result_dir = Path(args.result)

    csv_path = (
        result_dir
        / "pose_error_report.csv"
    )

    output_dir = (
        result_dir
        / "visualization"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\n")
    print("=" * 60)
    print("FOUNDATIONPOSE RESULT VISUALIZATION")
    print("=" * 60)

    # 读取分析结果
    results = load_report(
        csv_path
    )

    print(
        f"Frames loaded: {len(results)}"
    )

    # 平移误差
    plot_translation(
        results,
        output_dir
        / "translation_error.png"
    )

    # 旋转误差
    plot_rotation(
        results,
        output_dir
        / "rotation_error.png"
    )

    # 最大旋转误差帧拼图
    make_worst_frame_sheet(
        results,
        data_dir,
        result_dir,
        output_dir
        / "worst_rotation_frames.png",
        args.topk
    )

    print("\n")
    print("=" * 60)
    print("VISUALIZATION FINISHED")
    print("=" * 60)

    print(
        f"\n结果保存在：\n{output_dir}"
    )


if __name__ == "__main__":
    main()