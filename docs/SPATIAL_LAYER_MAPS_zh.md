# 从层数分割结果生成空间分布图

两版绘图已接入 MosaicAgent-DD：第一版是分别显示1L、2L、TL的热图及面积覆盖率；第二版把三类放入同一格，输出圆饼图、矩形堆叠条图、两者对比图，以及行、列方向的汇总。默认网格分别为8列×15行和6列×11行，可以修改。

此功能读取已有的分割标签，不重新识别层数、不修改模型，也不调用Kimi。先确认分割结果和排除区域正确，再解释空间分布。

## 直接处理已保存的结果

在仓库根目录运行，路径换成自己的文件：

```sh
python tools/layer_spatial_maps.py --mask path/to/sample_mask.npy --valid-mask path/to/sample_valid.npy --exclusions path/to/sample_exclusion.npy --sample SAMPLE001 --output outputs/spatial_maps/SAMPLE001_run01
```

- `--mask`：原始数值掩膜`.npy`，或完整分辨率、未重绘的彩色分割PNG。不要使用缩略图、叠加图或已涂灰排除区的预览图。
- `.npy`沿用MosaicAgent编号：0衬底、1双层、2单层、3厚层。程序会显式转换；不能把编号1当作单层。
- 彩色PNG按RGB识别：黑色衬底、绿色(0,128,0)单层、暗红(128,0,0)双层、橄榄色(128,128,0)厚层。调色板索引不作为层数编号。未知颜色或透明像素会停止处理。
- `--valid-mask`：有效扫描范围的布尔/0–1 `.npy`，尺寸须与掩膜相同。没有这项时，明确按整个画布为有效范围；如果拼接图有未扫描填充区域，应提供该文件。
- `--exclusions`：原流程保存的排除掩膜`.npy`；也接受矩形JSON，字段为`image_width`、`image_height`和`excluded_rectangles`，矩形使用`x1,y1,x2,y2`半开区间。确实没有排除区时可省略。
- `--output`必须是新目录。成功前在临时目录生成，成功后才发布完整结果；已有输出不会被覆盖。

调整网格：在命令后增加`--heatmap-grid 12 22 --combined-grid 8 15`，顺序始终是列、行。比较两种图形的同一个格子时，两者网格需相同。

## 接入现有分析流程

```sh
python flakepipeline/run.py --config configs/layer_analysis.example.json --no-ai --no-repair --spatial-maps
```

先编辑示例配置中的权重、输入图和样品名。`--spatial-maps`只开启新增统计导出，不改变既有分割和区域审核设置。也可在自己的配置中设置：

```json
"spatial_maps": {
  "enabled": true,
  "heatmap_grid": [8, 15],
  "combined_grid": [6, 11]
}
```

未启用时，原流程行为保持不变。启用后，输出位于`<work_root>/<sample>/04_stats/spatial_maps/<时间戳_运行编号>/`。常规统计JSON记录对应的空间统计清单；缓存会检查输入、网格、实现版本及每个输出文件的哈希。生成失败会明确报告失败，不能作为已完成的空间统计复用。

已完成分割的历史样品优先使用上面的独立命令，直接读取原有掩膜及有效/排除区域，不必重新运行推理。

## 在工作台查看

打开`03_open_workbench`，在“连接工具”里把“层数识别结果文件夹”设为本次导出目录或包含它的分析结果目录，再进入“03 / 层数分析”。工作台会读取`spatial_manifest.json`，提供PNG、SVG和数值文件链接，并在读取时检查哈希。它只查看已保存结果，不启动新计算。

## 输出与统计含义

每次保留PNG、可编辑文字SVG、逐格CSV、行列CSV、NPZ数组和JSON来源清单。清单包括输入文件哈希、代码指纹、版本、网格边界、运行时间和统计口径。

- **层数组成**＝该层面积／有效且未排除的(1L＋2L＋TL)面积。三类合计100%。
- **面积覆盖率**＝该层面积／有效且未排除的扫描面积，包括裸衬底。它与层数组成使用不同分母。
- 行、列和整图比例先相加像素数，再计算比例；不能直接平均各格百分比。
- `non_scan_px`、`excluded_px`和`retained_canvas_px`分别记录未扫描、有效扫描内被排除、保留像素，三项合计等于画布面积。重叠排除区域只计一次。
- †表示部分区域未计入，比例仅依据保留像素；没有晶体的格子组成未定义，CSV留空、JSON为`null`，不伪造0%。完全没有保留范围的格子不会画组成图。
- 同样大小的圆或条只表达组成，不表达局部晶体总覆盖面积。TL为已有厚层合并类别，不能单独解释为3L。坐标使用掩膜像素或行列编号，不假定微米比例尺。

## 验证范围

三个历史Figure 3掩膜（5、25、35 mg）在两种网格下重新统计，1L/2L/TL整图像素数均与原统计完全一致。原始标签没有修改。这验证的是计算一致性，不是分割准确率。

合成测试涵盖原生编号转换、调色板顺序、有效区域与重叠排除区、空白格、不同网格、导出失败保护、缓存损坏和工作台路径/哈希核验。运行：

2026-10-26本地完整验证：791项测试通过、1项跳过，84个子测试通过；代码编译检查、两处JavaScript语法检查、配准自检和16个锁定扫描运行文件校验均通过。未调用外部模型或仪器。

```sh
python -m pytest -q tests/test_spatial_stats.py tests/test_spatial_export.py tests/test_spatial_pipeline.py tools/analysis_workbench/test_spatial_results.py
```
