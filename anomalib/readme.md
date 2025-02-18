# Requirement:
Python>=3.10


# Before training:
Open the config.yaml, change the model to the one you want to use. Change other parameter depend on your setting.
Open the model_config.yaml(i.e. Padim_config.yaml), change the hyperparameter of the model if you need.
please refer to anomalib document if you feel confused about those parameters.
If you want to do hypertuning for specific model, you could find the paper link in the anomalib document.

# How to use:
```bash
python src/anomaly_detection/anomalib/train.py 
python src/anomaly_detection/anomalib/evaluation.py 
```

Output:
pred_scores and pred_labels of test image will be saved in the analysis.csv
other metrics(i.e. AUROC/F1score) will be uploaded to the mlflow ui
metircs graph(AUPR/AUPRO/AUROC) will be saved next to the model checkpoint
export the model from ckpt to bin and onnx for openvino, you can find them in weights/openvino

# Current version
The version of anomalib is 1.0.0 released on 1Mar2024
reference: https://anomalib.readthedocs.io/en/latest/index.html    

# Compatibility problem:
anomalib require a package call wandb. If the virtual environment contains this package, training yolo may require api key.
For this case, you need to change the environment variable.
```python
import wandb
wandb.init(mode="offline")
```
Adding these 2 lines of codes in the beginning could solve the problem. 

Modification(already done):
   1.data/utils/image.py line 444
    refer to https://github.com/openvinotoolkit/anomalib/issues/1831
    (Not sure whether this bug is solved in version1.0.1 or not )

   2.utils/path.py line 60~65
    comment them and add return new_version_dir
    (anomalib use symlink which require administrator right in Windows, but it works in Linux. If you want to update the anomalib and you ensure that you will not run it in your computer, skip it)


PS: In 1.0.0, the WinCLIP model cannot export to openvino so we cannot generate heatmap when using this model. This problem is solved in 1.1.0. Need to check the compatibilty(lightning/openvino etc)

PS2: Here using openvino inference for generating heatmap/mask. The openvino toolkit is officially supported by Intel hardware only.It does not support other hardware like AMD CPU/ Intel GPU. Therefore the speed of generating heatmap will be relatively slow.

PS3:Be aware of the pre-processing part, some parameter may lead to different result in engine.predict() and openvinoinferencer.predict(), e.g. the image_size in Folder(). If you want to do pre-processing, refer to the transform() in anomalib document. I don't know why it only appears in document afetr v1.1.0 but you can use it in 1.0.0. 