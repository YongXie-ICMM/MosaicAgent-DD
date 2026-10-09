# 交给 ChatGPT / Codex 接着做：MosaicAgent-DD 的检查与改进

2026-10-09 本地更新：第 0 项独立只读审查已完成；随后按用户的新扫描问题修复完整网格的错行和覆盖保护，见[对照记录](diagnostics/20261009_recorded_grid_stitching.md)。本轮分支改动是否已合并或进入学生包，请核对仓库与 `bundle_manifest.json`，不以本文件代替发布记录。

原 2026-10-08 交接基线为 `main` 上的 `0981c34`（2026-10-07 的配准修正）及其后的文档更新；该轮由 Claude 完成。日期保留以区分历史证据与本轮改动，当前状态见 [README 的 Status 一节](../README.md#status-2026-10-09) 和 [CLAUDE_HANDOFF.md](CLAUDE_HANDOFF.md)；写代码的规则在 [AGENTS.md](../AGENTS.md)。

## 怎么用（给仓库主人）

1. **推荐用 Codex**（在 ChatGPT 里连上 GitHub 仓库 `YongXie-ICMM/MosaicAgent-DD`）：它能读代码、跑测试、开 PR，并会自动读取根目录的 `AGENTS.md`。普通 ChatGPT 对话也可以，但只看得到你上传的文件，改动以补丁（unified diff）交回。
2. 把下面"提示词"一节代码框里的全文复制过去。
3. 想让它核对真实数据上的结果，就附上分析结果（都是小文件；原图每次几 GB，不用传）：
   - 261002AM：实验数据文件夹 `DD_MAPPING_20261003_261002AM/` 里 `handover_analysis_261002AM/` 的 `REPORT_zh.md`、`report.json`、`colour_check.json`、`stitch.json`、`stitch_log.txt`、`colour_log.txt`、`incidents_zh.md`、`handover_status.json`（`colour_match_report.json` 约 0.7 MB，需要时再给），以及 `mosaic_261002AM_preview_1_32.jpg`、`crop_chip_edge_bottom_left.jpg`、`before_fix_crop_bottom_left_ghost_and_hole.jpg`。
   - 260128：`DD_MAPPING_20261006_260128/handover_analysis_260128/` 里的同名文件、`mosaic_v2_preview_1_32.jpg`，以及 `v1_vs_v2/` 的对比图。
   - 261008PM：两个原始 `261008PM_5mg_70nm_part*.zip`、采集日志，以及本轮派生目录的来源清单、实测 profile、legacy/preserve 状态与日志、`positions_tids.json`、对比 JSON 和缺口前后裁剪。原图不必为复查重复上传，先给分析记录；公开仓库的[诊断文档](diagnostics/20261009_recorded_grid_stitching.md)说明复现方法。
   - 做对焦或亮度相关的检查时，再挑几张原图瓦片（每张约 3–6 MB）。
4. 改动交回后：PR 你审完再合并；补丁交给 Claude 或自己 `git apply`，跑测试后提交。合并后重建学生包（`python tools/build_student_bundle.py`；内部版另加 `--with-env <组里的 .env>`，只在组内发）。
5. 每轮结束让它把下面"待办"更新到本文件（或在 PR 说明里写清哪些已完成）。

## 提示词

```text
你接手的是 GitHub 仓库 YongXie-ICMM/MosaicAgent-DD。原 2026-10-08 交接基线为 main 的 0981c34 及后续文档；2026-10-09 本地分支已有完整网格拼接修复，先核对当前分支是否包含这些改动，不假设已发布。第 0 项只读审查已经完成；继续按用户授权做"检查 → 找证据 → 小步改进"，先审后改。回答用中文；代码、代码注释和提交说明用英文。

一、你能看到什么
- 能访问仓库（Codex / GitHub 连接器）时，先读：AGENTS.md、README.md、README_zh.md、docs/CHATGPT_HANDOFF_zh.md、docs/CLAUDE_HANDOFF.md、docs/STUDENT_GUIDE_zh.md、acquisition/README.md、docs/RESOLUTION_AND_INFERENCE.md、docs/diagnostics/ 下的记录。然后跑基线：pip install -r requirements-dev.txt；python -m pytest -q；python -m compileall -q .；python register.py。基线不通过就先报告，不要动代码。
- 不能访问仓库时，先列出你需要的文件，我来上传。
- 真实扫描原图（每次几 GB）不在仓库里。我可以上传分析结果（REPORT_zh.md、*.json、日志、预览和裁剪图）。没有我给的证据，不要对真实数据下结论；推测要标明"推测"。

二、项目是什么
光学显微镜下二维材料研究的发表用工作流，只有四块：扫描采集 → 拼接 → 语义层数识别（0 衬底、1 双层 2L、2 单层 1L、3 厚层 TL）与统计 → 光谱资料核对。学生在 Windows 仪器电脑上扫描（相机 + 电动位移台；扫描程序在 acquisition/Auto_Scan，16 个运行文件哈希锁定），在任何电脑上用学生包分析（启动文件 01_install、02_run_layer_demo、03_open_workbench、04_process_handover）。

三、当前状态（2026-10-09；旧扫描证据保留日期）
1. 层数识别：0409 权重不变；样品是 260 nm SiO2/Si；采集 1920×1080，按原生 256/32 切块再重采样到 512 模型窗口。新图大片衬底被判成单层，原因已定位为新采集模式的颜色/白平衡（不是分辨率）；demo 每次做参考衬底颜色检查（参考 RGB (219,171,170)，容差 5 %）。没有独立参考标签，所以任何校正后的预测都只是诊断，不是测量。
2. 采集端白平衡：acquisition/Auto_Scan/wb_calibrate.py + 00_white_balance_then_scan.bat（先校白平衡，再调用不变的扫描程序），离线测试 23 项，没有上机验证。
3. 位移台偶发 rc=-1：一次 "Y movement failed: rc=-1" 让扫描 260128 的第一轮在约 152 分钟时停下；记录与分析见 docs/diagnostics/stage_error_incident_log_zh.md（与运行时长无关，约 0.1 %）。建议的运动失败策略（读回计数再决定、丢回复只重发一次、不先软停）属于扫描程序源仓库 AmScope-Camera，不在本仓库。
4. 颜色/照明校正 tools/colour_match_grid.py（v2）：照明场 = 约 400 张瓦片各自除以均值后逐像素取中位数（σ = 宽/32 平滑）；每对相邻瓦片配准后在重叠区测颜色比值，减去每类边的基线，解"每列一个水平 + 每张一个受罚偏差"（TILE_PENALTY 0.3，Huber 3 %，边权 = 重叠区平坦像素占比，规范 = 最长无台阶列段，增益限 0.5–2.0）。原图不动，派生数据集用副本/硬链接，并写派生 session.json/events.jsonl。测试 12 项。
5. 交接流程：仪器端 acquisition/Auto_Scan/pack_handover.py（04_pack_handover.bat，仅标准库）把当天所有轮次原样放进一个文件夹 + handover.json（每个文件 SHA-256，每轮的网格/张数/相机读回/中断事件）；分析端 tools/handover.py（04_process_handover）七步 verify → assemble → incidents → colour → stitch → check → report，经 <文件夹>/analysis/ 共享数据，handover_status.json 记指纹并跳过未变的步骤，--from 从某步重做。测试 8 + 9 项。
6. 拼接配准（run_stitch.py、register.py）：旧推行路径保留 2026-10-07 的 decide_segment_offset() 门——单边 NCC 0.5 / 非零偏移优势 0.1；零偏移保留的是推断链行号，不保证等于扫描记录。2026-10-09 的 --grid-policy auto 只对预检确认完整、连续的 mosaic_r<row>_c<col> 网格保留记录行号、救回全部选中 QC 丢弃格位，质检理由和对焦阈值不改。旧 ZIP/推断布局走 legacy；--grid-policy legacy 保留旧版推行和 55 % 覆盖救回。切换策略必须新建 --work，--force 不绕过绑定。渲染记录实际贡献、失败和缺位置的 tile IDs，完整网格有缺图或后续条带读取失败就停止，不报告成功；重新运行先清除旧成功记录。记录行号没有验证真实位移，spatial_coverage_verified=False 仍保留。交接报告原有的位置核对继续提供网格偏差提示。
7. 学生包：tools/build_student_bundle.py 生成公开版（无密钥）和内部版 *-with-kimi.zip（带组里的 .env，学生拼接时用 Kimi 做质检投票、同格位二选一和接缝抽查；内部版永不公开）。GitHub 上唯一的 release v0.1.0 建于 5a20621（2026-10-05），早于第 2、4、5、6 项。
8. CI："Offline source checks"（pytest、compileall、node --check）在 0981c34 上通过。
9. 本地检查（2026-10-09）：704 passed、1 skipped、70 subtests；compileall、两个工作台的 node --check、register.py 自测与 16 个锁定运行文件哈希均通过，未调用外部模型。本地检查不是当前分支远程 CI 或上机验收的替代。
10. 第 0 项独立审查已经完成。本地知识库报告为 09_AUDIT/2026-10-08_mosaicagent_dd_readonly_audit.md（不在这个 Git 仓库内）；审查发现并不等于所有问题都已修复。

四、真实数据上的证据
- 261008PM_5mg_70nm（2026-10-09 对照）：两个 ZIP 合计 18×48 = 864 张完整原图，SHA-256 与日志一致；invert_x=True，视觉左起第二列是原始 c16。旧版在 c16 r15→r16 处误判反向，再加段偏移 2，使 r0–r15 共 16 张错行为 r+2；新策略错行 0。前后实际绘制均 864，最终丢弃/读取失败均 0；同样 389 个 defocus 标记保留并救回。RMS 2.1214→1.2757 px；仿射网格偏差中位/p95/最大值 55.4/156.3/1801.8→17.6/30.7/43.7 px。显式 legacy PNG SHA-256 与旧基线完全一致。70 nm 只用于这个新样品的拼接，没有运行或替换层数模型；这是计算几何一致性核对，不是物理标定或准确率验证。详见 docs/diagnostics/20261009_recorded_grid_stitching.md。
- 260128：16×53 = 848 张，两轮（第二次相机会话增益 27，原来 21，补拍最后两列）。列台阶 c2|c3 3.6 %、c12|c13 3.8 %、c13|c14 5.2 %；v2 后相邻重叠失配：列与列之间 RMS 1.28 % → 0.58 %，列内上下 0.59 % → 0.23 %；配准残差 1.72 → 0.88 px；848/848；照明场左右 7 %、上下 3 %；按两轮交接文件夹重跑，拼接图逐字节相同。参考颜色检查：不一致（已知的采集模式偏色）。
- 261002AM：18×50 = 900 张，一轮，人工复核模式；学生用的是改过的 v3.4 扫描程序（与它自带的清单不符）。照明场左右 3–4 %、上下 ≤ 1 %；无列台阶；单张偏差 ≤ 4 %；边失配 log-RMS 0.0181 → 0.0149（1732 条边中 153 条因无纹理被拒）；拼接 900/900、残差 1.95 px、Kimi 381 次、接缝抽查 6 处 clean；配准修正后位置偏差中位 16.9 px、p95 31.6 px、最大 46 px。裸衬底 (185,143,142) 对参考 (219,171,170)：亮度 ×1.19（暗 16 %），色彩平衡偏差 0.4 %；相机增益 0。交接资料缺控制台记录、shared_history、相机连接和白平衡记录（没用打包程序）。
- 两次扫描里，拼接质检的闭式对焦门都把约一半瓦片标成"虚焦"（261002AM 453/900，260128 455/848），全部因"丢了会留洞"被 step_rescue 救回，最终 0 丢弃。
- 合成回归（与上面的真实扫描分开）：仿 Figure 3a 的 41 张蛇形/重拍图，旧 infer 路径 41 个行号全部正确、位置误差 p95 0.60 px；没有重跑 920 张历史原图。

五、硬性规则（与任何一条冲突就停下来问我）
1. 不改扫描程序 16 个哈希锁定的运行文件；扫描逻辑的改动属于 AmScope-Camera 源仓库，并要上机验证。
2. 不改原图、权重、示例资源、历史输出和清单；派生数据写新文件夹，并记录来源和哈希。
3. 不悄悄改默认的预处理、阈值、模型选择或质检行为：改默认值必须有前后对照的数字、测试和文档，并保留旧行为的开关。
4. 层数 demo 样品仍是 260 nm SiO2/Si，不换成 70 nm（0815）配置或别的权重。261008PM 的 70 nm 是另一个新拼接样品，本轮只拼接，不选择层数模型。
5. 工作台第 03 项只做语义层数分析；不恢复 domain/instance 标注工具，不做转角推断。
6. 不打印、不记录、不提交任何密钥（.env、KIMI_API_KEY）；测试一律离线（--no-ai），不调用外部模型；内部版学生包永不成为 release 资产。
7. 预测不是测量；没有独立参考标签不报准确率；合成测试要标明是合成的。
8. 不重构目录，不改启动文件名、学生入口和已有的命令行参数。
9. 只在分支上改并开 PR（或给 unified diff），不推 main、不打 tag、不发 release。

六、工作方式
- 一个问题一个 PR/补丁：问题 → 证据（文件:行号、日志、数字）→ 最小改动 → 测试（合成数据；tests/handover_fixture.py 用扫描程序自己的类写记录）→ 前后对比 → 同步文档（README、README_zh、docs/CLAUDE_HANDOFF.md；改学生说明书后重渲染 PDF）。
- 分清楚：合成测试、操作者确认、物理标定、经过验证的准确率，这四样不是一回事。
- 每轮结束交给我：① 改了什么（文件、测试结果）② 没改什么、为什么 ③ 需要我或学生在仪器上做的事（简单中文步骤）④ 更新后的待办。

七、待办（第 0 项已完成；后续按用户授权逐项处理，不把覆盖保护当作对焦分类修复）
0. 独立复核上一轮的新代码（只读，2026-10-08 已完成；以下保留原审查范围）：
   a) register.py 的 decide_segment_offset()：阈值 0.5 / 0.1 只来自一个案例。检查它会不会挡住真正需要挪行的段（例如续扫后行号确实错开）；评估改用相对证据（与 0 偏移的 NCC 差、够格边数、段长）是否更稳；补边界测试。
   b) tools/handover.py：指纹与跳过、--from、失败步骤清理、位置核对（刚性网格模型的原点和两个向量在缺图、多轮、invert_x 时是否估计正确）。
   c) tools/colour_match_grid.py v2：增益解的规范、Huber 与罚项、样品大面积覆盖时照明场是否有偏、增益限。
   d) acquisition/Auto_Scan/pack_handover.py：按日期/会话选择轮次、--link 跨盘时退回复制、空间检查、运行文件核对。
1. 把颜色结论拆开（小而确定）：flakepipeline/color_diagnostics.py 的 compare_to_reference() 用 max|每通道增益 − 1| 判 colour_balance_differs_from_reference，只是变暗也会报。增加"亮度"和"色彩平衡"两项结论（保留旧字段以兼容），更新 demo 的中文提示、交接报告和测试。容差不变——层数模型对亮度是否敏感还没有验证。
2. 对焦门的内容偏置仍待验证：完整网格的覆盖保护已于 2026-10-09 完成，但虚焦分类没有修正。run_stitch.py 的闭式对焦门（全帧对焦分数 < 0.4 × 参考 → 直接判 defocus，不问模型）仍受内容影响。提出与内容无关的判据（例如与相邻瓦片在重叠区的锐度比，或按纹理量归一化），先作为可选项；必须仍能判出 Figure 3a 的 25 张真虚焦（历史模型 25/25 同意）；给出对照数字；对焦默认行为不变。
3. 让 colour_match_grid 支持不完整网格（现在缺位置就报错，交接流程因此跳过 colour 和 stitch）。
4. 工作台（tools/analysis_workbench）加"交接文件夹"页面：选文件夹 → 显示 handover_status.json 和 REPORT_zh.md → 运行或从某步重做。不改现有四个入口的含义。
5. 发布检查：写一份给仓库主人执行的清单——按 main 重建公开版学生包、核对 bundle_manifest.json 里的提交号和 sha256、写 release 说明；内部版不上传。
6. 只写方案、不改本仓库代码：AmScope-Camera 的运动失败策略和离线测试设计；白平衡校正的上机验证步骤；同一视野两种分辨率的对照；3×3 小扫描的端到端检查；请导师标几块独立参考区域；用 261002AM 的几张瓦片做"只调亮度"的层数对照（诊断）。

先读第 0 项已有报告与 2026-10-09 拼接对照，核对当前分支和用户授权，再处理未完成项；不要重复把第 0 项当作未开始。
```

## 待办状态

| 项 | 状态 |
|---|---|
| 0 复核上一轮新代码 | 2026-10-08 只读审查完成；本地报告 `09_AUDIT/2026-10-08_mosaicagent_dd_readonly_audit.md`，审查发现未全部实施 |
| 1 颜色结论拆成亮度 / 色彩平衡 | 未开始 |
| 2 对焦门的内容偏置 | 2026-10-09 完整网格错行/覆盖保护完成；对焦分类与阈值验证仍待做 |
| 3 不完整网格的颜色校正 | 未开始 |
| 4 工作台交接文件夹页面 | 未开始 |
| 5 发布检查清单（release v0.1.0 已过时） | 未开始 |
| 6 仪器与评估方案（只写方案） | 未开始 |
