# 扫描与层数分析统一入口

[English documentation](README.md)

本版对应 Digital Discovery 的扫描、拼接、层数识别和光谱记录流程。第 03 项只查看层数预测与面积统计。

## 启动与设置

- Windows：双击 `start_workbench.bat`。
- Mac：双击 `start_workbench.command`。
- 或从仓库根目录运行 `python3 tools/analysis_workbench/server.py`。
- 浏览器打开 `http://127.0.0.1:8792/`。

工作台本身需要 Python 3.10+，不需要额外 Python 包；拼接、推理和 Windows 采集分别有其依赖要求。

第一次点击“连接工具”，选择项目/数据文件夹。新配置不会自动读取私人项目。随包的采集代码在 `acquisition/Auto_Scan`；已有配置会继续使用原先选择的采集路径。“层数识别结果文件夹”可选，指向包含 `results/run_manifest.json` 的目录。

## 四项功能

| 入口 | 作用 | 需要做什么 |
|---|---|---|
| 显微扫描 | 打开 Windows 采集窗口 | 在采集窗口连接设备、核对参数，再开始采集 |
| 图像与拼接 | 查看已有图像；检查新扫描和准备拼接参数 | 核对尺寸、网格和日志，再执行生成的拼接命令 |
| 层数分析 | 并排核对原图与已保存的层数预测 | 选择图像，查看统计区域、有效像素、分子和分母 |
| 光谱与位置 | 查看已有光谱及样品对应记录 | 核对测量位置和来源，补齐缺失证据 |

第 03 项不会启动新识别、修改预测或打开编辑工具。新识别通过流水线命令单独运行。叠加图可选，默认关闭。工作台中准备参数不代表已经完成拼接，查看光谱记录不代表重新采集。

## 读取层数结果

工作台按 `results/run_manifest.json` 中的 `sample_id` 配对原图和结果。原图来自清单记录的 ZIP 成员，直接读取，不解压复制；预测图是 `results/<sample_id>/mask_color.png`。清单包含 SHA-256 时会校验。没有结果时显示为空，不生成替代结果。

未指定识别目录时，兼容读取所选项目下的 `06_analysis/runs/20260924_0409_inference_examples`。该路径只在已经选定的项目内查找，不会自动打开私人目录。

项目可提供 `06_analysis/Figure3_current/fig3_recount_overview.png` 和 `fig3_recount.json`，分别显示存档统计图与统计来源记录。打开这些文件不会重新计算。预测标签也不是独立物理参考。

## 采集与拼接的一致性

请先阅读[采集安装与使用说明](../../acquisition/README.md)。公开源代码不能代替相机、位移台所需的 Windows 厂商驱动。

工作台支持 `Auto_Scan` 源代码目录，也支持含 `delivery_manifest.json` 的完整平铺交付包。交付包的所有清单文件都会核对 SHA-256；缺失或修改时阻止启动，不会自动替换文件。

同一仪器只开一份扫描程序。关闭网页不会停止采集。进程启动、退出或照片保存数量不能单独证明采集完整、图像清晰或位置正确。

日志区区分目标尺寸、实际尺寸、待检查和不一致状态。像素尺寸不等于物理标定，不能只凭分辨率变化把拼接步距减半。新扫描应先在“图像与拼接”入口核对网格与记录，再检查真实接缝。

Mac/Linux 可查看结果，不能从该入口启动 Windows 实控。XY 模拟选项不保证相机也是模拟。

## 保存与检查

本机设置保存在不上传的 `workbench.local.json`；入口记录及采集启动日志在配置文件旁的 `workbench_history/`。源数据和旧历史不会被此入口改写。原图、权重和个人配置不包含在公共工作台中。

测试使用临时生成的历史事件，不依赖私人实验日志。运行：

```bash
python3 -m unittest discover -s tools/analysis_workbench -p 'test_*.py'
node --check tools/analysis_workbench/static/app.js
```

公开测试流程在 `.github/workflows/offline-tests.yml`。离线检查不代替仪器实测或物理判据验证。
