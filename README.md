# Casting defect detection with LiteRT

This repo is an experimentaiton repo. Things will chagne and be deleted or added.

Train a small CNN to classify casting images as `def_front` (defective) or `ok_front` (OK), then run the exported model on a USB camera. This repository contains Python training and camera scripts; the `.tflite` model can also be used in an Android app.

## Dataset

Download [Casting product image data for quality inspection on Kaggle](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product) and extract it so these directories exist:

```text
datasets/casting_data/casting_data/train/{def_front,ok_front}/
datasets/casting_data/casting_data/test/{def_front,ok_front}/
```

`datasets/` and generated `artifacts/` are ignored by Git. Training removes exact duplicate images that could leak between train and test splits.

## Use

From the repository root:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python train_tflite_cnn.py
.venv/bin/python camera_feed_test.py
```

Training writes `artifacts/casting_defect_cnn_int8.tflite` and `artifacts/cnn_metrics.json`. The camera script uses that model by default, marks approximate defect hotspots, and saves up to five annotated images in `read_images/`. Press **Q** or **Esc** to quit. Use `--camera-index N` if automatic camera selection picks the wrong device.

For the optional classifier with a heatmap output, run `train_classify_localize.py` and then `camera_classify_localize.py`; type `1` and Enter to start inference, or `0` and Enter to stop. The dataset has image labels but no defect masks or boxes, so localization is only an explanation of model sensitivity.

## Why LiteRT?

The intended target is on-device Android inference. The default CNN is much smaller than the experimental MobileNetV5 in `train_model.py`; it exports a fully integer-quantized `.tflite` file using LiteRT built-in operators. That keeps the model compact and avoids a server connection or extra TensorFlow operators at runtime. The model takes a 300×300 grayscale `uint8` image and outputs an OK score (`< 0.5` means defect). See `cnn_metrics.json` for tensor quantization values.
