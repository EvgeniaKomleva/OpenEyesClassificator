# OpenEyesClassificator

Более подробное описание решения можно найти тут: https://www.notion.so/OpenEyesClassificator-9416d4c1f6b1420bae42ce0b31476557?pvs=4 

Веса: https://drive.google.com/file/d/1y2TN8W_bT5ePGPrVxI20k9205eP5oUz5/view?usp=sharing

Чтобы обучить модель: 

```python
python train.py
```
В файле inference.py находится класс OpenEyesClassificator  c методами __init__(self) (где инициализируется и загружается модель) и метода predict(self, inpIm), где inpIm - полный путь к изображению глаза, который возвращает  -  score классификации от 0.0 до 1.0
