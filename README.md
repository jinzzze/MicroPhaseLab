# MicroPhaseLab

面向材料与机械工程学生的显微组织图像分析与机器学习教学项目。

当前版本是 **v0.2.2 传统分割基线**，聚焦于 Aachen–Heerlen 钢铁 SEM 数据集中
Martensite–Austenite（MA）constituent 的二值语义分割数据准备。它暂不训练
U-Net；在标注坐标、mask 和数据划分验证正确之前，过早训练会让错误悄悄进入实验。

## v0.2 已包含

- 官方 Figshare 数据下载器（标注、元数据；1.14 GB PNG 图像为显式可选项）
- 自动识别官方分号分隔 CSV 与项目内部逗号分隔 CSV
- 兼容 Python literal、JSON、WKT 的 polygon 解析器
- Polygon 转无损二值 PNG mask
- 图像—mask 清单与数据质量报告
- 按 sample/group 划分 train、validation、test，避免相似图像泄漏
- 原图、mask 与 overlay 可视化
- 固定随机种子的 20 张人工抽检样本
- Otsu 阈值、Gaussian 平滑和形态学开闭运算基线
- Dice、IoU、Precision、Recall 和 MA 面积分数误差
- 完全离线的合成演示数据和自动测试

## 1. 第一课：15 分钟跑通完整离线实验

这条实验不需要下载 1.14 GB 官方图像，也不需要 GPU。完成后，你将得到两张合成
SEM 风格图、对应的专家 polygon、二值 mask、质量报告、分组划分和传统分割结果。

学习目标：

1. 识别图像、polygon 标注与二值 mask 三种数据表示；
2. 理解为什么要在训练前检查 mask 质量与数据泄漏；
3. 将 Otsu + morphology 基线的 Dice、IoU、Precision、Recall 与面积分数误差联系到材料问题。

## 2. 安装（Windows PowerShell）

在项目根目录执行。`Restricted` 只会阻止激活脚本；以下命令只对当前 PowerShell
窗口临时放行，不会修改系统策略：

```powershell
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

如不想激活环境，后续把 `python` 和 `microphaselab` 分别替换成
`.\.venv\Scripts\python.exe` 与 `.\.venv\Scripts\microphaselab.exe` 即可。

### 可选：训练 v0.3 PyTorch U-Net

完成真实数据的质量检查与 group-aware split 后，安装可选训练依赖：

```powershell
python -m pip install -e ".[torch]"
microphaselab train --config configs/unet_demo.yaml
```

默认配置优先使用 GPU（如可用），否则使用 CPU。若要使用特定 CUDA 版本，请先在
[PyTorch 官方安装页面](https://pytorch.org/get-started/locally/)选择 Windows、Pip 与
对应计算平台，再安装本项目。训练会拒绝 group 重叠的 split，并在
`outputs/unet/run_001/` 写入 `config.yaml`、`metrics.csv`、`best.pt` 和
`validation_summary.json`。请只使用 validation 集选择参数；test 集只在参数冻结后使用。

macOS/Linux 用户可使用：

建议使用 Python 3.10–3.12：

```bash
cd MicroPhaseLab
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

### 运行与检查

```powershell
microphaselab demo
microphaselab check --manifest examples/demo/processed/manifest.csv --report examples/demo/processed/quality_report.json
microphaselab visualize --manifest examples/demo/processed/manifest.csv --output-dir examples/demo/figures --limit 4
microphaselab baseline --manifest examples/demo/processed/manifest.csv --output-dir outputs/baseline/demo
python -m pytest -q
```

请打开 `examples/demo/figures/` 中的 overlay，确认 mask 覆盖合成的亮区；然后查看
`outputs/baseline/demo/summary.json`，将 Dice 与 IoU 记录在实验笔记中。接着打开
`notebooks/01_data_pipeline.ipynb`，逐格完成第一课。

### 结果解读与验收

先确认 `quality_report.json` 的 `ok` 为 `true`：它表示每一个 image—mask 对都存在、
尺寸一致，且 mask 只包含 0 和 1。`rows` 是图像数，`groups` 是可用于防泄漏划分的
样品组数；两者不是同一个概念。

再阅读 `summary.json` 与 `metrics_per_image.csv`。其中：

- **Precision** = 被预测为 MA 的像素中，真正是 MA 的比例；低值意味着背景被误判为 MA。
- **Recall** = 专家标注的 MA 像素中被找到的比例；低值意味着漏检 MA。
- **Dice** = `2TP / (2TP + FP + FN)`，强调前景区域的重叠，适合类别不平衡的分割任务。
- **IoU** = `TP / (TP + FP + FN)`，比 Dice 更严格；同一预测下 IoU 通常低于 Dice。
- **area_fraction_absolute_error** = 预测与专家 mask 的二维 MA 面积分数之差的绝对值；
  它适合回答“面积比例是否接近”，但不能说明边界位置是否正确。

当前固定合成演示的参考结果约为 Dice `0.997`、IoU `0.993`、平均面积分数绝对误差
`0.0007`。合成图中 MA 被刻意绘制为明亮区域，因而 Otsu 很容易分割；这些数字**不是**
真实钢材图像的性能承诺。通过本课的最低标准是：质量报告通过、能解释每项指标、并能从
overlay 找出至少一种可能的假阳性或假阴性来源。只有在真实数据完成 20 张人工检查、
按 group 划分并冻结 test 集后，才能报告真实数据的测试指标。

## 3. 先运行离线演示

该命令生成两张合成 SEM 风格图、polygon CSV、metadata CSV，并走完数据准备流程：

```bash
microphaselab demo
```

输出位于 `examples/demo/`。随后检查数据并生成叠加图：

```bash
microphaselab check \
  --manifest examples/demo/processed/manifest.csv \
  --report examples/demo/processed/quality_report.json

microphaselab visualize \
  --manifest examples/demo/processed/manifest.csv \
  --output-dir examples/demo/figures \
  --limit 4
```

## 4. 下载官方 polygon 与 metadata

数据源：Aachen–Heerlen Annotated Steel Microstructure Dataset。完整 PNG 压缩包约
1.14 GB，因此默认只下载较小的 polygon 与 metadata CSV。

```powershell
microphaselab download --output-dir data/raw
```

确认文件已经生成：

```powershell
Get-Item data/raw/annotations.csv, data/raw/metadata.csv
```

## 5. 下载并解压官方 PNG 图像

完整 PNG 压缩包约 1.14 GB，解压后会包含 1,705 张带专家标注的图像：

```powershell
microphaselab download --output-dir data/raw --include-images
(Get-ChildItem data/raw/images -Recurse -File -Filter *.png).Count
```

下载器支持中断后续传，并会在解压前核对 Figshare 公布的精确文件大小与 MD5。
如果曾使用 v0.2.0 下载并遇到 `BadZipFile`，请删除无效的
`data/raw/images.zip` 和 `data/raw/images.zip.part`，再重新执行上述命令。

第二条命令的预期结果是 `1705`。官方另外发布了 875 张被排除的图像；这些
图像不包含在本项目使用的训练数据包中。

也可以从 Figshare 页面手动下载并放置为：

```text
data/raw/
├── annotations.csv
├── metadata.csv
└── images/
    ├── image_001.png
    └── ...
```

数据集页面：https://doi.org/10.6084/m9.figshare.c.5185004

## 6. 生成真实 MA mask 与 manifest

```powershell
microphaselab prepare --images-dir data/raw/images --annotations data/raw/annotations.csv --metadata data/raw/metadata.csv --output-dir data/processed `
  --group-column Type,Temperature
```

`Type + Temperature` 对应论文描述的10个不同化学成分/热处理样品组合，比只按
`Type` 分组更安全。运行后检查：

```powershell
microphaselab check --manifest data/processed/manifest.csv --report data/processed/quality_report.json
python -c "import pandas as pd; m = pd.read_csv('data/processed/manifest.csv'); print('masks:', len(m)); print('groups:', m['group_id'].nunique()); print(m['group_id'].value_counts().sort_index())"
```

预期 `masks` 接近 `1705`、`groups` 接近 `10`，质量报告中的 `ok` 应为 `true`。
如果数量不同，先检查 `preparation_report.json` 中的 `missing_images`、
`missing_group_metadata` 和 `invalid_annotations`，不要继续划分或训练。

程序会自动查找常见的图像列与 polygon 列，并执行以下检查：

- polygon 至少包含三个有效点
- 坐标有限且不会静默溢出图像边界
- image 与 mask 尺寸一致
- mask 只包含 0 和 1
- 每张图像的标注对象数和前景比例被记录

如果官方 CSV 的列名与你下载的版本不同，可显式指定：

```powershell
microphaselab prepare --images-dir data/raw/images --annotations data/raw/annotations.csv --metadata data/raw/metadata.csv --output-dir data/processed `
  --image-column Image_url --polygon-column polygon `
  --group-column sample_id
```

## 7. 随机人工检查至少20张 overlay

```powershell
microphaselab visualize --manifest data/processed/manifest.csv --output-dir outputs/figures/official_qc_seed42 `
  --limit 20 --random `
  --seed 42

Invoke-Item outputs/figures/official_qc_seed42
```

`qc_selection.csv` 会记录本次固定随机抽到的20张图。逐张检查 polygon 是否覆盖真实
MA constituent，重点排查：坐标轴互换、缩放错误、整体平移、边界越界和 mask
对错图。只有20张全部合理后才继续。

## 8. 建立无泄漏划分

```powershell
microphaselab split --manifest data/processed/manifest.csv --output-dir data/splits --group-column group_id `
  --seed 42
```

默认比例为 70/15/15。划分单位是 `group_id`，不是单张图；同一组只会进入一个
split。若元数据没有可靠的 sample 字段，程序会明确标记回退策略，此时必须人工检查
`manifest.csv`，不能直接把结果当成“无泄漏”。

## 9. v0.2：Otsu + morphology 传统基线

先在 validation 集运行和调整参数；不要用 test 集调参：

```powershell
microphaselab baseline --manifest data/splits/val.csv `
  --output-dir outputs/baseline/otsu_val
```

输出包括：

```text
outputs/baseline/otsu_val/
├── predictions/
├── metrics_per_image.csv
└── summary.json
```

默认流程为灰度图 → Gaussian blur → Otsu → opening → closing → 删除小区域与填补
小孔洞。根据 validation 结果确定一套参数后，冻结参数并且只运行一次 test：

```powershell
microphaselab baseline --manifest data/splits/test.csv --output-dir outputs/baseline/otsu_test `
  --gaussian-sigma 1.0 --opening-radius 1 --closing-radius 2 --min-object-size 32 `
  --min-hole-size 32
```

全图 Otsu 是教学基线，不使用专家提供的 POI，因此不应预期复现论文中“已知 POI 后
寻找轮廓”的约 0.35 IoU；两者不是完全相同的任务设置。

## 10. 运行测试

无需联网：

```bash
python -m unittest discover -s tests -v
```

## 目录结构

```text
MicroPhaseLab/
├── configs/data.yaml
├── data/
├── examples/
├── notebooks/01_data_pipeline.ipynb
├── outputs/
├── src/microphaselab/
├── tests/
├── CITATION.cff
├── LICENSE
└── pyproject.toml
```

## 科学边界

- 当前标签是 MA microconstituent，不等同于“所有材料相”。
- Polygon 是专家边界的离散近似，不是无误差真值。
- 面积分数基于二维视场；未经立体学假设，不能自动解释为三维体积分数。
- 当前项目仅用于教学和研究，不替代标准金相分析或专家判断。

## 路线图

- v0.2：Otsu + morphology 传统基线（当前）
- v0.3：PyTorch Dataset、U-Net、BCE + Dice、可复现训练（见
  [v0.3 实现计划](docs/v0.3_unet_plan.md)）
- v0.4：面积分数与形貌定量分析
- v0.5：Gradio 演示界面
- v1.0：完整中文教学课程、预训练权重与发布文档

## 数据引用

Iren, D. et al. *Aachen-Heerlen annotated steel microstructure dataset*.
Scientific Data 8, 140 (2021). https://doi.org/10.1038/s41597-021-00926-7

数据文件标注为 CC0；本仓库代码采用 MIT License。
