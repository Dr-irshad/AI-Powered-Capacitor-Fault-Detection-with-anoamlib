import os
import shutil
import random
import mlflow
import numpy as np
from datetime import datetime
from src.utils.load_env import ENV      # import your machine environment
from ruamel.yaml import YAML
from anomalib.deploy import ExportType
from src.utils.mlflow import set_new_mlflow
from torchvision.transforms.v2 import Resize
from anomalib.data import Folder
from anomalib import TaskType
from anomalib.engine import Engine
from anomalib.utils.normalization import NormalizationMethod
from anomalib.metrics import AUROC
from model_selection import model_selection
from src.utils.mlflow import set_new_mlflow, register_model, set_registered_model_tag
from format_transform import format_transform
def main():
    yaml = YAML()
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml"), "r") as file:
        CONFIG = yaml.load(file)
    print('can load config.yaml')
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_config", CONFIG["train_param"]["anomalib_model"]+"_config.yaml"), "r") as file:
        MODEL_CONFIG = yaml.load(file)
    print('can load win_clippconfig.yaml')
    path=os.path.join(ENV["data_root_dir"],CONFIG['dataset'], "processed",'abnormal', CONFIG["partition"])
    if os.path.exists(path)== False:
        path2=os.path.join(ENV["data_root_dir"],CONFIG['dataset'], "annotations", CONFIG["partition"])
        if os.path.exists(path2):
               print('Transform data to anomalib format')
               format_transform(CONFIG['dataset'],CONFIG["partition"])
        else: 
            print('dataset does not exist')
            exit()





    mlflow_run = set_new_mlflow(
                            experiment_name=CONFIG["experiment_name"]+"_training",
                            run_name=CONFIG["train_run_name"],
                            description=CONFIG["train_run_desc"],
                            )

    start_time = datetime.now().strftime("%Y-%m-%d %H-%M-%S")    
    model_dir = os.path.join(ENV["model_root_dir"], CONFIG["dataset"], CONFIG["model"], CONFIG["train_run_name"]) + '/'
    if os.path.isdir(model_dir):
        print("Model exists, path: ", model_dir)
        if not ENV["allow_overwrite_model"]:
            # exit avoid overwriting existing model
            print("Exit without overwriting existing model")
            exit()
    else:
        os.makedirs(model_dir)
    print(f"{model_dir = }")
    
    # Disable mlflow autolog as ultralytics YOLO's default autolog is enabled
    mlflow.autolog(disable=True)

    #dataset setting
    datamodule = Folder(
    name=CONFIG['dataset'],
    root=os.path.join(ENV["data_root_dir"],CONFIG['dataset'], "processed",CONFIG["partition"],'abnormal'),
    normal_dir=os.path.join('train',CONFIG['train_param']['normal_dir']),
    abnormal_dir=None if CONFIG['train_param']['abnormal_dir']==None else os.path.join('test',CONFIG['train_param']['abnormal_dir']),
    normal_test_dir=None if CONFIG['train_param']['normal_test_dir']==None else os.path.join('test',CONFIG['train_param']['normal_dir']),
    transform=Resize((CONFIG['train_param']['image_size'], CONFIG['train_param']['image_size'])),
    val_split_mode=CONFIG['train_param']['val_split_mode'],
    train_batch_size=CONFIG['train_param']['train_batch_size'],
    eval_batch_size=CONFIG['train_param']['eval_batch_size'],
    val_split_ratio=CONFIG['train_param']['val_split_ratio'],
    task=CONFIG['train_param']['task'],
    mask_dir=CONFIG['train_param']['mask_dir'],
    seed=CONFIG['train_param']['seed']
)
    datamodule.setup()  # Split the data to train/val/test/prediction sets.
    datamodule.prepare_data()  # Create train/val/test/predic dataloaders`

    #model setting
    model=model_selection(CONFIG["train_param"]["anomalib_model"],MODEL_CONFIG)

    #engine setting
    engine = Engine(
        normalization=NormalizationMethod.MIN_MAX,
        threshold=CONFIG['train_param']['threshold'],
        task=CONFIG['train_param']['task'],
        image_metrics=CONFIG['train_param']['image_metrics'],
        accelerator=CONFIG['train_param']['accelerator'],
        devices=CONFIG['train_param']['devices'],
        max_epochs=CONFIG['train_param']['max_epochs'],
        num_sanity_val_steps=CONFIG['train_param']['num_sanity_val_steps'],
        default_root_dir=model_dir
    )

    #train model
    engine.fit(model=model, datamodule=datamodule)

    end_time = datetime.now().strftime("%Y-%m-%d,%H-%M-%S")
    print("model training start time = ", start_time)
    print("model training end time = ",end_time)

    #export model to onnx/xml/bin format for openvino inference
    openvino_model_path = engine.export(
                                model=model,
                                export_type=ExportType.OPENVINO,
                                export_root=model_dir,)
    
    #change the location of the model
    #There is no method to control anomalib save location, so copy it to the model_dir and delete the original folder
    shutil.copy(os.path.join(model_dir, CONFIG["train_param"]["anomalib_model"], CONFIG['dataset'],'v0/weights/lightning/model.ckpt'), model_dir)
    
    # save mlflow run id for inference use
    with open(os.path.join(model_dir, "mlflow_run_id.txt"), "w") as f:
        f.write(mlflow_run.info.run_id)

    # log config file
    this_file = os.path.abspath(__file__)
    this_dir = os.path.dirname(this_file)
    config_file = os.path.join(this_dir, "config.yaml")
    target_copy_filename = os.path.join(model_dir, "config.yaml")
    shutil.copyfile(config_file, target_copy_filename)

    # log model_config
    model_def_file = os.path.join(this_dir, "model_config", CONFIG["train_param"]["anomalib_model"]+"_config.yaml")
    target_copy_filename = os.path.join(model_dir, CONFIG["train_param"]["anomalib_model"]+"_config.yaml")
    shutil.copyfile(model_def_file, target_copy_filename)
    
    # log mlflow
    mlflow.log_params(CONFIG["train_param"])
    mlflow.log_params(MODEL_CONFIG)
    CONFIG.pop("train_param")
    mlflow.set_tags(CONFIG)
    workflow_pipeline = [{"module": CONFIG["task"], "model": CONFIG["model"]}]
    mlflow.set_tag("workflow_pipeline", workflow_pipeline)

    mlflow.set_tag("model_path", os.path.join(f'{CONFIG["task"]}/'+ CONFIG["model"], 'model.ckpt'))

    workflow_pipeline = [{"module": CONFIG["task"], "model": CONFIG["model"]}]
    mlflow.set_tag("workflow_pipeline", workflow_pipeline)

    # log whole directory from local to mlflow      
    mlflow.log_artifacts(model_dir, artifact_path = CONFIG["task"] + '/' + CONFIG["model"])
    # register model
    register_model(mlflow_run.info.run_id, CONFIG["experiment_name"])
    tags = {"client": CONFIG["client"], "dataset": CONFIG["dataset"], "task": CONFIG["task"]}
    set_registered_model_tag(CONFIG["experiment_name"], tags)
if __name__=="__main__":
    main()