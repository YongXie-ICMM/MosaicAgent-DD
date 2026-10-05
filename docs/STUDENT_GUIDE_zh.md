# 学生操作说明（一步一步照做即可）

适用于完整学生包 `MosaicAgent-DD-student-complete.zip`（2026-10-05 版，含白平衡标定）。
英文总览见 [README](../README.md)，中文入口见 [README_zh](../README_zh.md)。本说明只讲"怎么做"，不讲原理；原理见文末链接。

两台电脑、两套任务：

| 电脑 | 做什么 | 用哪些文件 |
|---|---|---|
| **分析电脑**（任何 Windows 或 macOS） | 跑层数识别、看结果、检查颜色 | 解压后最外层的 `01_install`、`02_run_layer_demo`、`03_open_workbench` |
| **仪器电脑**（Windows，接显微镜） | 先校正白平衡，再扫描 | `acquisition/Auto_Scan/` 里的 `01_setup.bat`、`00_white_balance_then_scan.bat` |

---

## 第一部分：分析电脑（第一次 15 分钟，以后 3 分钟）

### 1. 解压

把整个 zip 解压到一个**可写**的文件夹（如 `D:\MosaicAgent-DD` 或 `~/MosaicAgent-DD`）。不要只拖出一个文件。解压后应当看到：

```text
MosaicAgent-DD/
  01_install.bat / .command
  02_run_layer_demo.bat / .command
  03_open_workbench.bat / .command
  data/demo/source_images.zip      两张原图
  data/demo/assets.json            原图和权重的校验值
  weights/model_0409_all.pth       权重（34 MB）
  bundle_manifest.json             本包每个文件的校验值
  acquisition/  configs/  docs/  flakepipeline/  tools/  tests/ ...
```

少了 `data/demo/` 或 `weights/`，说明拿到的是 GitHub 自动生成的源码包，不是学生包；向老师要完整包，**不要自己去网上下载名字相近的权重**。

### 2. 第一次：双击 `01_install`

Windows 用 `.bat`，macOS 用 `.command`（macOS 首次可能提示"无法验证开发者"：右键 → 打开）。
需要已安装 Python 3.10 以上并能上网。窗口会创建 `.venv` 并安装依赖，可能要几分钟；看到 `Ready.` 再关。
失败就把窗口里的红字原样截图交给老师，不要反复重装。

### 3. 每次：双击 `02_run_layer_demo`

程序先核对原图和权重的校验值，再对两张 1920 × 1080 原图做层数识别（普通笔记本约 1–3 分钟）。结束时会打印：

- `Results / 结果目录: outputs/demo/<时间戳>`：本次结果所在；每次新建一个文件夹，之前的结果不会被覆盖。
- `Colour check / 颜色检查: ...`：见下面第 5 条。
- `Software completed; model applicability not validated`：这是正常提示，意思是"程序跑完了，但这些比例不能直接当科学结论"。

### 4. 双击 `03_open_workbench`，打开 **03 / 层数分析**

左边原图，右边预测图，一一对照。默认不叠加 overlay。颜色含义：单层、双层、厚层分别用不同颜色标出（图例在页面上）。
如果页面上方的提示框里出现"颜色/白平衡与参考衬底颜色不一致"，说明这批图需要人工复核，见第 5 条。

### 5. 颜色检查说什么、该怎么办

程序会在每张原图里找**干净裸衬底**的颜色，和参考值 **RGB ≈ (219, 171, 170)** 比较，算出每个通道差多少（增益）。

| 窗口显示 | 意思 | 你要做的 |
|---|---|---|
| 没有颜色提示 / `within_tolerance` | 颜色在 ±5% 内 | 正常看结果 |
| `颜色/白平衡与参考衬底颜色不一致（…增益：[1.01, 1.10, 0.94]…）` | 图偏蓝、绿不足；模型会把整张图读"厚一级"，大片衬底会被判成单层 | 结果只能看不能用；下次拍照前**先做白平衡标定**（第二部分）。已有的图可以跑 `python tools/student_demo.py colour-probe --probe-inference` 看"校正后大概什么样"，但那是诊断，不是测量 |
| `implausible_gain_check_inputs` | 图太暗/太亮或根本不是显微镜图 | 检查图是否选对、曝光是否正常 |

**不要**为了让提示消失而修改权重、修改 `configs/reference_substrate_colour.json`、改预处理代码，或换 70 nm 模型。这些都会让结果不可追溯。

### 6. 这次要交回的东西

- `outputs/demo/<时间戳>/` 整个文件夹（里面有 `run_manifest.json`、每张图的 `mask_color.png` 和统计）。
- 两张原图与预测图是否对应、哪里明显错分的截图。
- 安装是否成功、程序是否跑完、报错原文、电脑系统版本。

---

## 第二部分：仪器电脑（扫描日）

### 7. 只做一次：安装

1. Windows，64 位 Python 3.10 以上（安装时勾选 **tcl/tk**）。
2. 装好位移台和相机厂家的驱动（老师/工程师负责）。
3. 把整个学生包解压到仪器电脑的可写文件夹，保持 `acquisition/Auto_Scan/` 完整。
4. 双击 `acquisition/Auto_Scan/01_setup.bat`（装 numpy、opencv）。
5. 若使用 XIMC 位移台：把 64 位 `libximc.dll` 及其依赖放进 `Auto_Scan/drivers/`（见 [采集说明](../acquisition/README.md)）。

### 8. 每个扫描日：双击 `acquisition/Auto_Scan/00_white_balance_then_scan.bat`

这是扫描日的**唯一入口**，它先做白平衡标定，再调用原来的扫描程序（扫描程序本身没有改动）。

**第 1 步：白平衡标定**

1. 打开显微镜灯，调到平时用的亮度，**记下亮度刻度**。
2. 把样品上一块**干净的裸衬底**移到视野中央：没有晶体、没有颗粒、没有边缘和划痕，对好焦。整个画面都应是衬底。
3. 回到黑窗口按任意键。程序会：
   - 用和扫描程序相同的方式连接相机（要求 1920 × 1080）；
   - 关闭自动白平衡和自动曝光；
   - 读出衬底颜色并和参考 (219, 171, 170) 比较，每一步都打印 `R 217.0->219.2 (-1.0%)  G 156.0->170.9 (-8.7%) ...`。
4. 看最后一行：

| 程序最后说 | 意思 | 你要做的 |
|---|---|---|
| `白平衡已回到参考：保持灯光、曝光、白平衡不动，开始扫描` | 自动调好了，已记录 | 按 `Y` 启动扫描；从现在起**不要再碰**灯光、曝光、白平衡 |
| `此采集路径没有可写的曝光和白平衡控制` + 一个带读数的预览窗 | 这台相机只能在**相机自带菜单**里调（HDMI 采集卡常见） | 一边看窗口里的读数和提示（Too blue / Not enough green / Too dark），一边在相机菜单里调色温、色调、曝光，直到三个通道都在 ±5% 内、第一行变绿；按 `a` 记录。调不进去就按 `f` 强制记录（会标记"未达标"），然后告诉老师 |
| `视野不是干净裸衬底，拒绝记录` 或提示 `NOT clean` | 画面里有晶体/颗粒/边缘，或没对焦 | 换一块更干净的衬底位置，重新运行 |
| `已记录但未达到参考` | 调不到容差内（例如驱动只有色温没有色调） | 仍可扫描，但分析时会被标为待复核；把 `colour_calibration/latest.json` 一起交回 |

记录保存在 `acquisition/Auto_Scan/colour_calibration/<时间戳>/`：`colour_reference_frame.png`（当时的衬底图）、`camera_colour_settings.json`（相机、曝光、色温/色调、读数、是否达标）、`calibration_steps.jsonl`（每一步）。程序会问你**灯光亮度刻度**和**物镜**，请填写（直接回车也可以，但最好填）。

**第 2 步：扫描**

程序问 `Start the scanner now [Y,N]?`，按 `Y`。之后与以前完全一样：

1. GUI 里连接相机，确认预览是显微镜画面、尺寸 1920 × 1080。
2. 第一次先扫 **3 × 3** 小范围（分步确认模式），看相邻视野是否有重叠、是否清晰。
3. 正常停止、断开、关闭程序；不要同时开第二个扫描程序。
4. 扫描文件夹里要保留 `session.json`、`events.jsonl` 和全部原图。

如果中途换了灯光、曝光或白平衡，这次扫描就要**重新从 00 开始**。

### 9. 扫描完交回

- 整个扫描会话文件夹（原图 + `session.json` + `events.jsonl`）。
- `acquisition/Auto_Scan/colour_calibration/` 里当天的记录文件夹。
- `acquisition/Auto_Scan/history/console_logs/` 当天的控制台记录。
- 一句话：用的物镜、灯光刻度、相机菜单里改了什么。

---

## 第三部分：把新扫描的图拿去分析

1. 在分析电脑上双击 `03_open_workbench`，打开 **02 / 图像与拼接 → 检查新扫描图片与拼接设置**，选择扫描会话文件夹（保留 `session.json`、`events.jsonl`）。
2. 按页面提示核对像素尺寸、相邻重叠、采集设置，生成拼接命令和 `layer_input_contract.json`。
3. 之后的层数识别和颜色检查与第一部分相同：颜色提示没了，说明白平衡标定有效；还有提示，就把 `colour_calibration` 记录和这批图一起交给老师。

---

## 常见问题

| 现象 | 处理 |
|---|---|
| 双击 `.bat` 一闪而过 | 右键 → 编辑看内容是否完整；或在该文件夹打开命令行手动运行，保留报错 |
| 提示 `Run 01_install.bat first` | 先运行 `01_install` |
| 提示缺少资源 / 校验失败 | 包不完整或文件被改过；重新解压完整包 |
| 窗口里中文是乱码 | 在命令行先执行 `chcp 65001` 再运行；或直接看英文那一半 |
| `02_run_layer_demo` 很慢 | CPU 推理，1–5 分钟正常；不要中途关窗 |
| 白平衡程序说 `Camera opened but delivered no frame` | 关掉其它占用相机的软件（厂家预览软件等）再试 |
| 白平衡程序说 `图像尺寸不一致：要求 [1920, 1080]，实际 ...` | 相机模式不对；在相机或驱动里改回 1920 × 1080，不要让程序缩放 |
| 想不接相机先试试白平衡程序 | `python acquisition\Auto_Scan\wb_calibrate.py --auto --simulate-camera`：用模拟画面自测，记录只进 `colour_calibration\simulation\`，不会被当成真实记录 |

**绝对不要做的事：** 改权重文件；改 `configs/` 下的参考文件；改 `acquisition/Auto_Scan/` 里被校验的 16 个运行文件（改了扫描入口会拒绝启动）；手工编辑 `session.json` / `events.jsonl`；同时开两个扫描程序；把预测比例直接写进论文。

原理与证据：[颜色对照记录](diagnostics/20261005_colour_balance/README.md) · [分辨率与预测](RESOLUTION_AND_INFERENCE.md) · [采集说明](../acquisition/README.md) · [数据与可复现性](DATA_AND_REPRODUCIBILITY.md)
