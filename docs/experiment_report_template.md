# MicroPhaseLab 真实数据实验报告模板

> 仅在完成数据质量检查、20 张 overlay 人工检查、group-aware split 和 validation
> 调参后填写。本模板不应把二维面积分数直接解释为三维体积分数。

## 1. 实验信息

- 学生／小组：
- 日期：
- MicroPhaseLab Git commit：
- Python、PyTorch 与设备：
- 数据集版本与引用：

## 2. 数据与质量控制

- 原始图像数、成功生成的 mask 数：
- group 数与 group 定义：
- quality_report.json 的 ok 值：
- preparation_report.json 中缺失或无效条目的处理：
- 20 张人工 overlay 的输出目录和发现的问题：

说明：只有 image-mask 尺寸一致、mask 为 0/1，且人工检查没有坐标或匹配错误时，才可
继续训练。

## 3. 防泄漏划分

| Split | 图像数 | group 数 | group 是否与其他 split 重叠 |
| --- | ---: | ---: | --- |
| Train |  |  | 否 |
| Validation |  |  | 否 |
| Test |  |  | 否 |

写明为什么选择该 group 定义，以及为什么没有按单张图像随机划分。

## 4. 训练与模型选择

- 训练配置文件：
- 随机种子：
- 输入尺寸、batch size、epoch、学习率、base channels：
- 选择 best.pt 的 validation 指标：
- 在 validation 上尝试的参数及最终冻结参数：

不得根据 test 集分数选择模型、阈值或 epoch。

## 5. 冻结后的 Test 结果

| 指标 | Macro mean | Micro |
| --- | ---: | ---: |
| Dice |  |  |
| IoU |  |  |
| Precision |  |  |
| Recall |  |  |
| 面积分数绝对误差 |  | 不适用 |

附上 summary.json、metrics_per_image.csv 与至少 20 张预测对照图的路径。

## 6. 材料学解释

1. Precision 与 Recall 哪一个较低？这分别意味着哪些 MA 误检或漏检情形？
2. 面积分数误差与 IoU 是否一致？若不一致，解释边界位置或形状可能发生的误差。
3. 与 Otsu + morphology 基线相比，U-Net 改善了什么，仍在哪些显微组织形貌上失败？
4. 哪些结果可能受成像条件、标注边界、类别不平衡或样品分布影响？

## 7. 局限与可复现性

- 本任务只分割 MA microconstituent，不代表所有材料相。
- 专家 polygon 是离散边界近似，不是无误差真值。
- 二维面积分数不自动等于三维体积分数。
- 列出复现所需的 commit、配置、checkpoint、命令和随机种子。
