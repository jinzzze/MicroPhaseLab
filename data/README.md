# Data layout

原始数据不提交到代码仓库。

```text
raw/images/          官方 PNG 图像
raw/annotations.csv 官方 expert polygons
raw/metadata.csv    官方 steel sample metadata
processed/masks/    由 MicroPhaseLab 生成
processed/manifest.csv
splits/train.csv
splits/val.csv
splits/test.csv
```

请保留原始文件只读。重新处理时使用新的 `processed` 目录，不要覆盖原始图像。
