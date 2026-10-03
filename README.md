# foundationpose_tools
该项目**实际使用的主线**是：

`foundationpose_tools` 负责采集、标注和校验 → `FoundationPose/run_demo.py` 负责首帧定位与后续跟踪 → `FoundationPose_experiments` 保存输入和结果。

## 1. 连接相机：录制 → mask → run_demo

最直接的入口是 [run_mug_tracking.sh](/home/shaunxiang/Documents/projects/foundationpose_tools/run_mug_tracking.sh)：

```bash
cd /home/shaunxiang/Documents/projects/foundationpose_tools
bash run_mug_tracking.sh my_experiment_01
```

流程中会依次执行：

1. 检查 RealSense 与 FoundationPose 环境。
2. 用 D405 录制对齐的 RGB 和深度帧。预览窗口中按 **Space 开始／暂停**，按 **Q 结束**。首帧应让杯子完整、清晰地留在画面内。
3. 校验帧编号、相机内参和深度数据。
4. 在首帧沿杯子轮廓左键加点，**Z 撤销、S 保存 mask**。
5. 再次校验 mask，运行首帧 `register()` 和后续 `track_one()`。
6. 核对输出帧数，并生成 10 FPS 慢放与 30 FPS 正常速度视频。

输入保存在 `FoundationPose_experiments/recordings/my_experiment_01/`，结果保存在 `FoundationPose_experiments/results/my_experiment_01/`。同名实验已存在时，一键脚本会拒绝覆盖。采集文件实际是 `rgb/*.png`、`depth/*.png`（16 位，单位毫米）、`cam_K.txt` 和首帧 `masks/000000.png`；[datareader.py](/home/shaunxiang/Documents/projects/FoundationPose/datareader.py:57) 在运行时将深度除以 1000，转换为米。

## 2. 用已有录制数据反复测试参数

**复跑时直接使用录制目录，不要把导出的 MP4 当作 `--test_scene_dir`。** 该入口需要逐帧 RGB、深度、内参和首帧 mask。以下示例使用已校验的 821 帧数据：

```bash
cd /home/shaunxiang/Documents/projects/FoundationPose

/home/shaunxiang/miniconda3/condabin/conda run --no-capture-output \
  -n foundationpose python run_demo.py \
  --mesh_file custom_mug/mug_colored.ply \
  --test_scene_dir ../FoundationPose_experiments/recordings/20261001_210829 \
  --est_refine_iter 5 \
  --track_refine_iter 2 \
  --trans_smooth_alpha 0.25 \
  --rot_smooth_alpha 0.20 \
  --axis_scale 0.04 \
  --debug 2 \
  --debug_dir ../FoundationPose_experiments/results/replay_t025_r020
```

每组参数使用**不同的 `--debug_dir`**：[run_demo.py](/home/shaunxiang/Documents/projects/FoundationPose/run_demo.py:47) 启动时会清空该目录内的旧输出。主要参数含义如下：

| 参数 | 作用 |
|---|---|
| `est_refine_iter` | 首帧定位的细化次数 |
| `track_refine_iter` | 后续每帧跟踪的细化次数 |
| `trans_smooth_alpha`、`rot_smooth_alpha` | 输出位姿的时间平滑系数；越小越稳、滞后越大；`1` 表示不平滑 |
| `axis_scale` | 仅改变可视化坐标轴长度，不改变位姿 |
| `debug 2` | 保存 `track_vis/*.png`，便于逐帧比较和制作视频 |

建议先固定迭代次数，只改变一项平滑参数。每次输出中的 `ob_in_cam_raw/` 是原始跟踪位姿，`ob_in_cam/` 是平滑后位姿，`track_vis/` 是可视化帧。平滑结果没有反馈给跟踪器，因此只改平滑系数时，重点比较平滑位姿和画面；改细化次数时，还要比较原始位姿。

## 3. 分析已有结果

**定性分析：**查看 `track_vis` 或已导出的 MP4，重点观察首帧是否对准、物体框是否持续贴合、快速运动和遮挡后是否失锁，以及平滑后是否出现明显拖尾。将不同实验的**同编号帧**并排比较，比单看两个播放视频更可靠。

**有真值时的定量分析：**[analyze_experiment.py](/home/shaunxiang/Documents/projects/foundationpose_tools/scripts/analyze_experiment.py:679) 将 `gt_poses/*.txt` 与结果中的 `ob_in_cam/*.txt` 按文件名配对，计算每帧平移误差（毫米）和旋转误差（度），输出 `pose_error_report.csv`，并列出误差最大的帧。项目内 `custom_mug/synthetic_mug` 有 30 帧真值，可用于这一路径：

```bash
cd /home/shaunxiang/Documents/projects

/home/shaunxiang/miniconda3/condabin/conda run --no-capture-output \
  -n foundationpose python foundationpose_tools/scripts/analyze_experiment.py \
  --mesh FoundationPose/custom_mug/mug_colored.ply \
  --data FoundationPose/custom_mug/synthetic_mug \
  --result <对应的合成数据运行结果目录>
```

随后可运行 [visualize_results.py](/home/shaunxiang/Documents/projects/foundationpose_tools/scripts/visualize_results.py:316)，用同一 `--data`、`--result` 生成误差曲线和最差帧拼图。

**真实 D405 录制目前没有 `gt_poses`。** 因而不能把“位姿变化小”称为“位姿准确”：现有分析脚本会跳过误差计算，也不会生成可供可视化脚本读取的 CSV。真实数据可先在**物体静止的片段**统计原始与平滑位姿的逐帧平移量、旋转量及异常峰值；运动片段则以同帧画面对齐、失锁和滞后为主。项目中的 `scripts/examine.py` 是写死旧路径与参数的 shell heredoc 草稿，不能直接作为通用 Python 分析命令使用。





相机参数：

640×480 @ 30 FPS
RealSense D405

Resolution:
RGB   = 640 × 480 @ 30 FPS
Depth = 640 × 480 @ 30 FPS

Depth Scale:
0.001 m / raw unit

Depth intrinsics:
fx = 380.837097
fy = 380.837097
cx = 317.088928
cy = 242.121521

K_depth =
[380.837097, 0,          317.088928]
[0,          380.837097, 242.121521]
[0,          0,          1         ]

Color intrinsics:
fx = 389.212769
fy = 388.786499
cx = 318.388428
cy = 242.905884

K_color =
[389.212769, 0,          318.388428]
[0,          388.786499, 242.905884]
[0,          0,          1         ]

Stereo baseline:
18.041 mm

## 实验记录

运行 `run_mug_tracking.sh` 后，实验记录统一保存在源码项目外：

- 采集数据：`/home/shaunxiang/Documents/projects/FoundationPose_experiments/recordings/<实验名称>`
- 跟踪结果：`/home/shaunxiang/Documents/projects/FoundationPose_experiments/results/<实验名称>`

不传实验名称时自动使用时间戳，也可执行 `./run_mug_tracking.sh <实验名称>` 指定名称。


