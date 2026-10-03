#!/usr/bin/env bash
set -Eeuo pipefail

FP_ROOT="/home/shaunxiang/Documents/projects/FoundationPose"
TOOLS_ROOT="/home/shaunxiang/Documents/projects/foundationpose_tools"
CONDA_BIN="/home/shaunxiang/miniconda3/condabin/conda"
FFMPEG_BIN="/usr/bin/ffmpeg"
FFPROBE_BIN="/usr/bin/ffprobe"
MODEL_PATH="${FP_ROOT}/custom_mug/mug_colored.ply"
EXPERIMENT_ROOT="/home/shaunxiang/Documents/projects/FoundationPose_experiments"
DATA_ROOT="${EXPERIMENT_ROOT}/recordings"
RESULT_ROOT="${EXPERIMENT_ROOT}/results"

CHECK_ONLY=0
if [[ "${1:-}" == "--help" ]]; then
  echo "用法："
  echo "  $0                 # 使用时间戳创建一次新实验"
  echo "  $0 实验名称        # 使用指定名称创建一次新实验"
  echo "  $0 --check         # 只检查固定环境，不连接相机"
  exit 0
elif [[ "${1:-}" == "--check" ]]; then
  CHECK_ONLY=1
  SESSION_NAME="environment_check"
else
  SESSION_NAME="${1:-$(date +%Y%m%d_%H%M%S)}"
fi
if [[ ! "${SESSION_NAME}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "实验名称只能包含字母、数字、点、下划线和连字符。" >&2
  exit 2
fi

SCENE_DIR="${DATA_ROOT}/${SESSION_NAME}"
RESULT_DIR="${RESULT_ROOT}/${SESSION_NAME}"
VIDEO_10_PATH="${RESULT_DIR}/foundationpose_tracking_10fps.mp4"
VIDEO_30_PATH="${RESULT_DIR}/foundationpose_tracking_30fps.mp4"
VIDEO_10_TMP="${RESULT_DIR}/.foundationpose_tracking_10fps.tmp.mp4"
VIDEO_30_TMP="${RESULT_DIR}/.foundationpose_tracking_30fps.tmp.mp4"

if [[ "${CHECK_ONLY}" -eq 0 && ( -e "${SCENE_DIR}" || -e "${RESULT_DIR}" ) ]]; then
  echo "实验名称已存在，拒绝覆盖：${SESSION_NAME}" >&2
  exit 2
fi

for required in "${CONDA_BIN}" "${FFMPEG_BIN}" "${FFPROBE_BIN}" "${MODEL_PATH}" "${FP_ROOT}/run_demo.py"; do
  if [[ ! -e "${required}" ]]; then
    echo "缺少必需文件：${required}" >&2
    exit 1
  fi
done

"${CONDA_BIN}" run --no-capture-output -n realsense python - <<'PY'
import cv2
import numpy
import pyrealsense2
print("RealSense 环境检查通过：OpenCV", cv2.__version__)
PY

echo "============================================================"
echo "FoundationPose 固定流程"
echo "实验名称：${SESSION_NAME}"
echo "数据目录：${SCENE_DIR}"
echo "结果目录：${RESULT_DIR}"
echo "============================================================"
echo "录制要求："
echo "1. 首帧让杯子完整、清晰、静止地位于画面内。"
echo "2. SPACE 开始录制；结束前先按 SPACE 暂停，再按 Q。"
echo "3. 不要录到杯子离开画面后的片段。"
echo

cd "${FP_ROOT}"

"${CONDA_BIN}" run --no-capture-output -n foundationpose python - <<'PY'
import torch
from Utils import mycpp

assert torch.cuda.is_available(), "CUDA 不可用"
assert mycpp is not None and hasattr(mycpp, "cluster_poses"), "mycpp 扩展不可用"
assert "cpython-311" in mycpp.__file__, mycpp.__file__
print("FoundationPose 环境检查通过：", torch.cuda.get_device_name(0))
print("mycpp：", mycpp.__file__)
PY

if [[ "${CHECK_ONLY}" -eq 1 ]]; then
  echo "固定流程环境检查全部通过。"
  exit 0
fi

"${CONDA_BIN}" run --no-capture-output -n realsense \
  python "${TOOLS_ROOT}/scripts/capture_d405.py" \
  --output "${SCENE_DIR}" --width 640 --height 480 --fps 30

"${CONDA_BIN}" run --no-capture-output -n realsense \
  python "${TOOLS_ROOT}/scripts/validate_scene.py" \
  --scene "${SCENE_DIR}"

echo
echo "现在标注首帧杯子轮廓：左键加点，Z 撤销，S 保存。"
"${CONDA_BIN}" run --no-capture-output -n realsense \
  python "${TOOLS_ROOT}/scripts/annotate_first_mask.py" \
  --scene "${SCENE_DIR}"

"${CONDA_BIN}" run --no-capture-output -n realsense \
  python "${TOOLS_ROOT}/scripts/validate_scene.py" \
  --scene "${SCENE_DIR}" --require-mask

echo
echo "数据和 mask 检查通过，开始 FoundationPose registration + tracking。"
"${CONDA_BIN}" run --no-capture-output -n foundationpose \
  python "${FP_ROOT}/run_demo.py" \
  --mesh_file "${MODEL_PATH}" \
  --test_scene_dir "${SCENE_DIR}" \
  --est_refine_iter 5 \
  --track_refine_iter 2 \
  --trans_smooth_alpha 0.25 \
  --rot_smooth_alpha 0.20 \
  --axis_scale 0.04 \
  --debug 2 \
  --debug_dir "${RESULT_DIR}"

FRAME_COUNT="$(find "${SCENE_DIR}/rgb" -maxdepth 1 -type f -name '*.png' | wc -l)"
POSE_COUNT="$(find "${RESULT_DIR}/ob_in_cam" -maxdepth 1 -type f -name '*.txt' | wc -l)"
VIS_COUNT="$(find "${RESULT_DIR}/track_vis" -maxdepth 1 -type f -name '*.png' | wc -l)"
if [[ "${FRAME_COUNT}" -ne "${POSE_COUNT}" || "${FRAME_COUNT}" -ne "${VIS_COUNT}" ]]; then
  echo "跟踪输出不完整：输入=${FRAME_COUNT}，位姿=${POSE_COUNT}，可视化=${VIS_COUNT}" >&2
  exit 1
fi

encode_video() {
  local fps="$1"
  local temporary_path="$2"
  local output_path="$3"
  local video_frames
  local video_size

  echo "正在编码 ${fps} FPS 视频，请等待编码和验证完成。"
  "${FFMPEG_BIN}" -hide_banner -loglevel warning -stats \
    -framerate "${fps}" -start_number 0 \
    -i "${RESULT_DIR}/track_vis/%06d.png" \
    -c:v libx264 -preset fast -crf 18 \
    -pix_fmt yuv420p -movflags +faststart \
    "${temporary_path}"

  video_frames="$("${FFPROBE_BIN}" -v error -select_streams v:0 \
    -show_entries stream=nb_frames -of csv=p=0 "${temporary_path}")"
  video_size="$(stat -c '%s' "${temporary_path}")"
  if [[ "${video_frames}" -ne "${FRAME_COUNT}" || "${video_size}" -lt 1024 ]]; then
    echo "${fps} FPS 视频验证失败：输入=${FRAME_COUNT}，视频帧=${video_frames}，大小=${video_size} bytes" >&2
    exit 1
  fi
  mv -- "${temporary_path}" "${output_path}"
}

encode_video 10 "${VIDEO_10_TMP}" "${VIDEO_10_PATH}"
encode_video 30 "${VIDEO_30_TMP}" "${VIDEO_30_PATH}"

echo
echo "============================================================"
echo "流程完成"
echo "输入数据：${SCENE_DIR}"
echo "跟踪位姿：${RESULT_DIR}/ob_in_cam"
echo "原始位姿：${RESULT_DIR}/ob_in_cam_raw"
echo "慢放视频：${VIDEO_10_PATH}"
echo "正常视频：${VIDEO_30_PATH}"
echo "============================================================"
