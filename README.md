代码来自[oscar](https://github.com/microsoft/Oscar)

## 代码架构

```shell
|-- cfg.py # 参数配置
|-- data
|   |-- vinvl_coco # 存放图像特征
|-- inference.sh # 生成coco test图像描述脚本
|-- models # 模型代码
|   |-- caption_model.py # caption模型
|   |-- caption_model_utils.py
|   |-- origin_bert 
|       |-- file_utils.py
|       |-- modeling_bert.py # transformer编码器核心代码
|       |-- modeling_utils.py
|       |-- tokenization_bert.py
|       |-- tokenization_utils.py
|-- test.py # 生成coco test图像描述代码
|-- utils # 工具代码，可以忽略
    |-- caption_evaluate.py
    |-- cider
    |   |-- pyciderevalcap
    |       |-- __init__.py
    |       |-- cider
    |       |   |-- __init__.py
    |       |   |-- cider.py
    |       |   |-- cider_scorer.py
    |       |-- ciderD
    |           |-- __init__.py
    |           |-- ciderD.py
    |           |-- ciderD_scorer.py
    |-- misc.py
    |-- tsv_file.py
    |-- tsv_file_ops.py

```

## 运行方式

目前代码只实现了推断过程，可在coco test数据集上测试，对每张测试图片生成图像描述，并给出各种评价指标。

```shell
CUDA_VISIBLE_DEVICES=0 bash inference.sh
```

单卡运行占用显存10G左右。

## 数据

链接：https://pan.baidu.com/s/1G6LcgJXMgZCp8WIYvY6jFw 
提取码：j165

在data目录解压后，目录结构如下

```shell
|-- data
|   |-- vinvl_coco
|       |-- test.feature.lineidx
|       |-- test.feature.tsv
|       |-- test.label.lineidx
|       |-- test.label.tsv
|       |-- test.yaml
|       |-- test_caption.json
|       |-- test_caption_coco_format.json
```

## 模型checkpoint

模型下载链接：[coco_captioning_base_scst.zip](https://biglmdiag.blob.core.windows.net/vinvl/model_ckpts/image_captioning/coco_captioning_base_scst.zip)。在ckpts目录下解压后，目录结构如下

```shell
|--ckpts
	|-- coco_captioning_base_scst
	|   |-- checkpoint-15-66405
	|       |-- added_tokens.json
	|       |-- config.json
	|       |-- pytorch_model.bin
	|       |-- special_tokens_map.json
	|       |-- training_args.bin
	|       |-- vocab.txt
```

## 代码阅读指南

重点关注代码：`models/origin_bert/modeling_bert.py`，`models/caption_model.py`，`models/caption_model_utils.py`，`test.py`。