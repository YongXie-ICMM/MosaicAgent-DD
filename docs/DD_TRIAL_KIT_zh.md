# 完整学生试用包（2026-11-01）

学生只需完整解压一个ZIP。先看顶层的一页PDF，首次运行 `01_install.bat`，然后运行 `03_open_workbench.bat`。按网页箭头先完成3×3扫描，再走日志交接、自动校色、拼接、结果检查、层数识别和空间图。保留原图和整个 `Memory/` 交回老师。

## 老师准备

- 安装64位Python3.9（保持仪器现有3.9.18，含Tcl/Tk）；首次安装Python依赖需要网络。trial包通过requirements-student-py39.txt固定依赖，NumPy1.26.2和OpenCV4.10.0.84沿用成功扫描环境；不升级系统Python。
- 确认相机/位移台USB驱动及VC++2013x64运行库。安装入口只检查系统运行库，不调用仪器。扫描仅限Windowsx64。
- 新包的 `trial_kit.json` 将扫描入口绑定到独立的 `scanner_C1`。它完整保留C1交付清单及74个原文件；`acquisition/` 的16个锁定历史运行文件也保持不变。不要让学生改选历史扫描程序。
- 首次启动自动设置项目目录和C1路径；已有老师配置按原字节保留。扫描和分析使用同一个 `.venv`；不另跑C1的全局安装脚本。
- 为本批确认拼接位移、模型适用性和推理尺度。复制 `configs/layer_workflow.teacher_template.json` 到新文件，改成实际名称和本次拼接图路径，经核对再将两个确认字段设为true。模板512/64是模型窗口。老师须用本次拼接的layer_input_contract.json及实际输出尺度生成inference_geometry；不能仅改tile或两个确认字段来代替尺度核对。1080p示例的原生256/32→模型512重采样不能直接套到缩放过的拼接图。
- 本包不携带Kimi密钥，离线流程可试跑。两张原始示例仅做层数demo，未确认相邻关系，不能拼接成实验图。

## 重新打包

从稳定的分析源分支运行（原C1交付文件夹和一页PDF另行提供）：

```sh
python tools/build_dd_trial_kit.py --scanner /path/to/unchanged_C1_folder --pdf /path/to/verified_one_page.pdf --out /path/to/new_trial.zip
python tools/verify_dd_trial_kit.py --check /path/to/new_trial.zip
```

打包工具拒绝覆盖已有ZIP，检查原始资源哈希，并排除虚拟环境、真实扫描数据、历史Memory和密钥。`bundle_manifest.json` 记录分析源文件，`scanner_C1/delivery_manifest.json` 保留原C1清单，`trial_kit_manifest.json` 将整个试用包逐文件绑定。最终ZIP另保存SHA-256及离线验证结果。

C1已在10月31日正常完成9张与1000张扫描。该证据不等于发生过故障注入，也不等于本次整合包已在Windows完成全流程。学生本次试跑用于核对这些环节；代码检查不会触发设备、加载模型或调用LLM。
