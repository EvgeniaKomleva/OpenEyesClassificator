# OpenEyesClassificator

## Intro
Hello reader! A little preview of what's to come:

- How to mark up a dataset for classifying pictures in an hour
- How to translate a picture into space so that similar pictures remain close, and dissimilar ones are far away
- How to train the model to classify pictures qualitatively

## More details
A more detailed description of the solution can be found here : https://abounding-stoat-8a4.notion.site/Classification-of-open-and-closed-eyes-4498bdbf8c944ca09bf9a112de95d58c 

Weights: https://drive.google.com/file/d/1y2TN8W_bT5ePGPrVxI20k9205eP5oUz5/view?usp=sharing

To train a model:

```python
python train.py
```
The inference.py file contains the OpenEyesClassificator class with the __init__(self) methods (where the model is initialized and loaded) and the predict(self, inpIm) method, where inpIm is the full path to the eye image, which returns - classification score from 0.0 to 1.0
