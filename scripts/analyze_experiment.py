#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import numpy as np
import trimesh


# ============================================================
# 工具函数
# ============================================================

def list_files(directory, suffix=None):
    """
    返回目录中的文件列表。
    suffix:
        None      -> 所有文件
        ".txt"    -> 只找 txt
        ".png"    -> 只找 png
    """
    directory = Path(directory)

    if not directory.exists():
        return []

    if suffix is None:
        return sorted([p for p in directory.iterdir() if p.is_file()])

    return sorted(directory.glob(f"*{suffix}"))


def stem_set(files):
    """把文件列表转换成不带后缀的文件名集合。"""
    return {p.stem for p in files}


def rotation_error_deg(R_pred, R_gt):
    """
    计算两个旋转矩阵之间的 geodesic rotation error。
    返回单位：degree
    """
    R_err = R_pred @ R_gt.T

    cos_theta = (np.trace(R_err) - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    return np.degrees(np.arccos(cos_theta))


def find_matching_file(directory, stem):
    """
    根据 stem 查找对应图片，例如：
        000001.png
        000001.jpg
    """
    directory = Path(directory)

    if not directory.exists():
        return None

    for ext in [".png", ".jpg", ".jpeg"]:
        p = directory / f"{stem}{ext}"
        if p.exists():
            return p

    return None


# ============================================================
# 1. Mesh 检查
# ============================================================

def analyze_mesh(mesh_path):
    mesh_path = Path(mesh_path)

    print("\n")
    print("=" * 60)
    print("1. MESH CHECK")
    print("=" * 60)

    if not mesh_path.exists():
        print(f"[FAIL] Mesh 不存在: {mesh_path}")
        return None

    try:
        mesh = trimesh.load(
            mesh_path,
            force="mesh",
            process=True
        )
    except Exception as e:
        print(f"[FAIL] Mesh 读取失败: {e}")
        return None

    vertices = len(mesh.vertices)
    faces = len(mesh.faces)

    extents = mesh.extents
    bounds = mesh.bounds

    try:
        components = len(
            mesh.split(only_watertight=False)
        )
    except Exception:
        components = -1

    print(f"Mesh:       {mesh_path}")
    print(f"Vertices:   {vertices}")
    print(f"Faces:      {faces}")
    print(f"Watertight: {mesh.is_watertight}")
    print(f"Components: {components}")

    print("\nExtents (m):")
    print(
        f"X = {extents[0]:.6f} m\n"
        f"Y = {extents[1]:.6f} m\n"
        f"Z = {extents[2]:.6f} m"
    )

    print("\nExtents (mm):")
    print(
        f"X = {extents[0] * 1000:.2f} mm\n"
        f"Y = {extents[1] * 1000:.2f} mm\n"
        f"Z = {extents[2] * 1000:.2f} mm"
    )

    print("\nBounds (m):")
    print(bounds)

    if components == 1:
        print("\n[PASS] Mesh 是单一连通组件")
    elif components > 1:
        print(f"\n[WARN] Mesh 包含 {components} 个独立组件")

    if mesh.is_watertight:
        print("[PASS] Mesh watertight")
    else:
        print("[WARN] Mesh 不是 watertight")

    # 一个非常粗略的单位异常检测
    max_extent = float(np.max(extents))

    if max_extent > 10:
        print(
            "[WARN] Mesh 最大尺寸 > 10 m，"
            "可能仍然使用 mm 单位。"
        )
    elif max_extent < 0.001:
        print(
            "[WARN] Mesh 最大尺寸 < 1 mm，"
            "请检查尺度。"
        )
    else:
        print("[PASS] Mesh 尺度看起来基本合理")

    return mesh


# ============================================================
# 2. Dataset 检查
# ============================================================

def analyze_dataset(data_dir):
    data_dir = Path(data_dir)

    print("\n")
    print("=" * 60)
    print("2. DATASET CHECK")
    print("=" * 60)

    if not data_dir.exists():
        print(f"[FAIL] 数据目录不存在: {data_dir}")
        return {}

    dirs = {
        "rgb": data_dir / "rgb",
        "depth": data_dir / "depth",
        "masks": data_dir / "masks",
        "gt_poses": data_dir / "gt_poses",
    }

    result = {}

    for name, directory in dirs.items():
        files = list_files(directory)
        result[name] = files

        print(
            f"{name:<10}: "
            f"{len(files):>5} files"
        )

    # 相机内参检查
    K_path = data_dir / "cam_K.txt"

    if K_path.exists():
        try:
            K = np.loadtxt(K_path)

            print("\ncam_K.txt:")
            print(K)

            if K.shape == (3, 3):
                print("[PASS] Camera K = 3x3")
            else:
                print(
                    f"[WARN] Camera K shape = {K.shape}"
                )

        except Exception as e:
            print(
                f"[WARN] cam_K.txt 读取失败: {e}"
            )
    else:
        print("\n[WARN] cam_K.txt 不存在")

    # 检查 RGB / Depth 是否逐帧匹配
    rgb_stems = stem_set(result["rgb"])
    depth_stems = stem_set(result["depth"])

    common_rgb_depth = rgb_stems & depth_stems

    print("\nRGB / Depth:")
    print(
        f"Common frames: "
        f"{len(common_rgb_depth)}"
    )

    if (
        len(result["rgb"]) ==
        len(result["depth"]) ==
        len(common_rgb_depth)
    ):
        print("[PASS] RGB 与 Depth 完全对应")
    else:
        print("[WARN] RGB 与 Depth 数量/文件名不一致")

    # Mask 有时只需要第一帧
    mask_stems = stem_set(result["masks"])

    if len(mask_stems) == 0:
        print("[WARN] 没有 mask")
    elif len(mask_stems) == 1:
        print(
            "[INFO] 只有一个 mask。"
            "对于 register + tracking 流程可能是正常的。"
        )
    else:
        print(
            f"[INFO] Mask frames: {len(mask_stems)}"
        )

    return result


# ============================================================
# 3. FoundationPose 输出检查
# ============================================================

def analyze_prediction_files(result_dir):
    result_dir = Path(result_dir)

    print("\n")
    print("=" * 60)
    print("3. FOUNDATIONPOSE OUTPUT CHECK")
    print("=" * 60)

    pose_dir = result_dir / "ob_in_cam"
    vis_dir = result_dir / "track_vis"

    poses = list_files(pose_dir, ".txt")
    vis = list_files(vis_dir)

    print(f"Pose files: {len(poses)}")
    print(f"Vis files:  {len(vis)}")

    if len(poses) > 0:
        print("[PASS] FoundationPose Pose 输出存在")
    else:
        print("[FAIL] 没有找到预测 Pose")

    return {
        "pose_dir": pose_dir,
        "vis_dir": vis_dir,
        "poses": poses,
        "vis": vis,
    }


# ============================================================
# 4. Pose 误差评估
# ============================================================

def evaluate_pose(
    gt_dir,
    pred_dir,
):
    gt_dir = Path(gt_dir)
    pred_dir = Path(pred_dir)

    print("\n")
    print("=" * 60)
    print("4. POSE EVALUATION")
    print("=" * 60)

    if not gt_dir.exists():
        print(
            "[INFO] 没有 GT Pose，"
            "跳过定量评估。"
        )
        return []

    if not pred_dir.exists():
        print(
            "[FAIL] Prediction Pose 目录不存在"
        )
        return []

    gt_files = {
        p.name: p
        for p in gt_dir.glob("*.txt")
    }

    pred_files = {
        p.name: p
        for p in pred_dir.glob("*.txt")
    }

    common = sorted(
        set(gt_files.keys())
        & set(pred_files.keys())
    )

    print(f"GT frames:         {len(gt_files)}")
    print(f"Prediction frames: {len(pred_files)}")
    print(f"Common frames:     {len(common)}")

    if len(common) == 0:
        print("[FAIL] 没有可以对应的 GT / Prediction")
        return []

    results = []

    for filename in common:

        gt = np.loadtxt(gt_files[filename])
        pred = np.loadtxt(pred_files[filename])

        if gt.shape != (4, 4):
            print(
                f"[WARN] {filename} GT shape={gt.shape}"
            )
            continue

        if pred.shape != (4, 4):
            print(
                f"[WARN] {filename} Pred shape={pred.shape}"
            )
            continue

        # Translation
        t_gt = gt[:3, 3]
        t_pred = pred[:3, 3]

        trans_error_m = np.linalg.norm(
            t_pred - t_gt
        )

        # Rotation
        rot_error_deg = rotation_error_deg(
            pred[:3, :3],
            gt[:3, :3]
        )

        results.append({
            "filename": filename,
            "stem": Path(filename).stem,
            "translation_m": trans_error_m,
            "translation_cm": trans_error_m * 100,
            "translation_mm": trans_error_m * 1000,
            "rotation_deg": rot_error_deg,
        })

    if len(results) == 0:
        return []

    t = np.array([
        r["translation_m"]
        for r in results
    ])

    r = np.array([
        x["rotation_deg"]
        for x in results
    ])

    print("\nTranslation Error")
    print(
        f"Mean:   {np.mean(t) * 100:.4f} cm "
        f"({np.mean(t) * 1000:.3f} mm)"
    )
    print(
        f"Median: {np.median(t) * 100:.4f} cm"
    )
    print(
        f"Max:    {np.max(t) * 100:.4f} cm "
        f"({np.max(t) * 1000:.3f} mm)"
    )

    print("\nRotation Error")
    print(
        f"Mean:   {np.mean(r):.4f} deg"
    )
    print(
        f"Median: {np.median(r):.4f} deg"
    )
    print(
        f"Max:    {np.max(r):.4f} deg"
    )

    return results


# ============================================================
# 5. Worst Frames
# ============================================================

def print_worst_frames(
    results,
    data_dir,
    result_dir,
    topk=5,
):
    if not results:
        return

    data_dir = Path(data_dir)
    result_dir = Path(result_dir)

    rgb_dir = data_dir / "rgb"
    vis_dir = result_dir / "track_vis"

    print("\n")
    print("=" * 60)
    print("5. WORST FRAME CHECK")
    print("=" * 60)

    worst_translation = sorted(
        results,
        key=lambda x: x["translation_m"],
        reverse=True
    )[:topk]

    worst_rotation = sorted(
        results,
        key=lambda x: x["rotation_deg"],
        reverse=True
    )[:topk]

    print(
        f"\nTop {topk} Translation Error"
    )

    for i, x in enumerate(
        worst_translation,
        start=1
    ):
        rgb = find_matching_file(
            rgb_dir,
            x["stem"]
        )

        vis = find_matching_file(
            vis_dir,
            x["stem"]
        )

        print(
            f"\n#{i} {x['filename']}"
        )
        print(
            f"  Translation: "
            f"{x['translation_mm']:.3f} mm"
        )
        print(
            f"  Rotation:    "
            f"{x['rotation_deg']:.3f} deg"
        )
        print(
            f"  RGB: "
            f"{rgb if rgb else 'N/A'}"
        )
        print(
            f"  VIS: "
            f"{vis if vis else 'N/A'}"
        )

    print(
        f"\nTop {topk} Rotation Error"
    )

    for i, x in enumerate(
        worst_rotation,
        start=1
    ):
        rgb = find_matching_file(
            rgb_dir,
            x["stem"]
        )

        vis = find_matching_file(
            vis_dir,
            x["stem"]
        )

        print(
            f"\n#{i} {x['filename']}"
        )
        print(
            f"  Translation: "
            f"{x['translation_mm']:.3f} mm"
        )
        print(
            f"  Rotation:    "
            f"{x['rotation_deg']:.3f} deg"
        )
        print(
            f"  RGB: "
            f"{rgb if rgb else 'N/A'}"
        )
        print(
            f"  VIS: "
            f"{vis if vis else 'N/A'}"
        )


# ============================================================
# 6. CSV 保存
# ============================================================

def save_csv(results, output_path):
    if not results:
        return

    output_path = Path(output_path)

    with output_path.open(
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "frame",
            "translation_m",
            "translation_cm",
            "translation_mm",
            "rotation_deg",
        ])

        for r in results:
            writer.writerow([
                r["filename"],
                r["translation_m"],
                r["translation_cm"],
                r["translation_mm"],
                r["rotation_deg"],
            ])

    print(
        f"\nPer-frame CSV saved:\n{output_path}"
    )


# ============================================================
# 7. 简单问题诊断
# ============================================================

def print_diagnosis(results):
    if not results:
        return

    t = np.array([
        r["translation_m"]
        for r in results
    ])

    r = np.array([
        x["rotation_deg"]
        for x in results
    ])

    t_mean_mm = np.mean(t) * 1000
    r_mean = np.mean(r)

    print("\n")
    print("=" * 60)
    print("6. QUICK DIAGNOSIS")
    print("=" * 60)

    print(
        f"Mean translation: "
        f"{t_mean_mm:.3f} mm"
    )

    print(
        f"Mean rotation: "
        f"{r_mean:.3f} deg"
    )

    if (
        t_mean_mm < 5
        and r_mean > 10
    ):
        print(
            "\n[可能问题]"
            "\n位置比较准，但旋转误差明显偏大："
            "\n- 目标可能接近旋转对称"
            "\n- yaw / roll / pitch 存在歧义"
            "\n- 缺少明显纹理或颜色方向特征"
            "\n- Mesh 的非对称结构过小"
        )

    elif (
        t_mean_mm > 20
        and r_mean > 10
    ):
        print(
            "\n[可能问题]"
            "\n位置和旋转都明显偏差："
            "\n- Mesh 尺度错误"
            "\n- 相机内参 K 错误"
            "\n- Depth 尺度错误"
            "\n- Mask 错误"
            "\n- Mesh 与真实目标不匹配"
        )

    elif (
        t_mean_mm < 5
        and r_mean < 5
    ):
        print(
            "\n[结果]"
            "\n整体 Pose 表现较稳定。"
            "\n建议继续检查 worst frames，"
            "判断是否存在局部失锁或异常。"
        )

    else:
        print(
            "\n[结果]"
            "\n误差属于中间状态。"
            "\n建议重点检查 worst frames。"
        )

    print(
        "\n固定排查顺序："
        "\n1. Mesh 尺度 / 单位"
        "\n2. Mesh 坐标系"
        "\n3. Camera K"
        "\n4. Depth 单位"
        "\n5. 第一帧 Mask"
        "\n6. 对称性 / 纹理"
        "\n7. 遮挡 / 运动模糊"
        "\n8. Tracking 漂移"
        "\n9. 初始化候选数量"
        "\n10. GPU OOM / 分辨率"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "FoundationPose experiment "
            "automatic analysis tool"
        )
    )

    parser.add_argument(
        "--mesh",
        required=True,
        help="Mesh path"
    )

    parser.add_argument(
        "--data",
        required=True,
        help="Dataset directory"
    )

    parser.add_argument(
        "--result",
        required=True,
        help="FoundationPose debug/result directory"
    )

    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="Number of worst frames"
    )

    args = parser.parse_args()

    mesh_path = Path(args.mesh)
    data_dir = Path(args.data)
    result_dir = Path(args.result)

    print("\n")
    print("#" * 60)
    print("FOUNDATIONPOSE EXPERIMENT ANALYSIS")
    print("#" * 60)

    print(f"\nMesh:   {mesh_path}")
    print(f"Data:   {data_dir}")
    print(f"Result: {result_dir}")

    # 1 Mesh
    analyze_mesh(mesh_path)

    # 2 Dataset
    analyze_dataset(data_dir)

    # 3 Output
    prediction_info = analyze_prediction_files(
        result_dir
    )

    # 4 Pose Error
    gt_dir = data_dir / "gt_poses"

    results = evaluate_pose(
        gt_dir,
        prediction_info["pose_dir"]
    )

    # 5 Worst frames
    print_worst_frames(
        results,
        data_dir,
        result_dir,
        args.topk
    )

    # 6 CSV
    csv_path = (
        result_dir
        / "pose_error_report.csv"
    )

    save_csv(
        results,
        csv_path
    )

    # 7 Diagnosis
    print_diagnosis(results)

    print("\n")
    print("#" * 60)
    print("ANALYSIS FINISHED")
    print("#" * 60)


if __name__ == "__main__":
    main()
