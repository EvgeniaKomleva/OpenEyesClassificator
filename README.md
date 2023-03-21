# OpenEyesClassificator
A more detailed description of the solution can be found here : https://abounding-stoat-8a4.notion.site/Classification-of-open-and-closed-eyes-4498bdbf8c944ca09bf9a112de95d58c 

Weights: https://drive.google.com/file/d/1y2TN8W_bT5ePGPrVxI20k9205eP5oUz5/view?usp=sharing

To train a model:

```python
python train.py
```
The inference.py file contains the OpenEyesClassificator class with the __init__(self) methods (where the model is initialized and loaded) and the predict(self, inpIm) method, where inpIm is the full path to the eye image, which returns - classification score from 0.0 to 1.0
