# ML-course-project
 同济大学软件学院机器学习期末项目

### Environment

You can simply use pip install to config the environment:

```sh
conda create -f env.yaml
```

### Run the code

```sh
cd ML-course-project
python main.py
```

### Result

|             | Precision |   SE   |   SP   |   YI   |  MAE   |   MSE   |
| :---------: | :-------: | :----: | :----: | :----: | :----: | :-----: |
|   class0    |  0.7863   | 0.8932 | 0.8677 | 0.7609 | 1.0680 | 1.6010  |
|   class1    |  0.8000   | 0.7874 | 0.8485 | 0.6359 | 2.8425 | 3.6121  |
|   class2    |  0.7692   | 0.5556 | 0.9766 | 0.5321 | 9.1111 | 13.7477 |
|   class3    |  0.8750   | 0.8077 | 0.9887 | 0.7964 | 7.0769 | 12.8333 |
| avg / total |  0.8076   | 0.7610 | 0.9204 | 0.6813 | 3.3664 | 6.6742  |

Table 1. ViT+ResNet(KL-Loss) 在痤疮数量的预测效果：AVE_ACC=0.7979

|             | Precision |   SE   |   SP   |   YI   |
| :---------: | :-------: | :----: | :----: | :----: |
|   class0    |  0.8108   | 0.8738 | 0.8889 | 0.7627 |
|   class1    |  0.7923   | 0.8110 | 0.8364 | 0.6474 |
|   class2    |  0.7407   | 0.5556 | 0.9727 | 0.5282 |
|   class3    |  0.8750   | 0.8077 | 0.9887 | 0.7964 |
| avg / total |  0.8047   | 0.7620 | 0.9217 | 0.6837 |

Table 2. ViT+ResNet(KL-Loss) 在痤疮严重程度的预测效果：AVE_ACC=0.8014
