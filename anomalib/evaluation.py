import os
import shutil
import pandas as pd
import mlflow
import numpy as np
from src.utils.load_env import ENV      # import your machine environment
from ruamel.yaml import YAML
import pandas as pd
from src.utils.mlflow import set_new_mlflow, copy_and_continuous_mlflow
from model_selection import model_selection
from torchvision.transforms.v2 import Resize
from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.utils.normalization import NormalizationMethod
from anomalib.metrics import AUROC,AUPRO,AUPR
from format_transform import format_transform
yaml = YAML()
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml"), "r") as file:
        CONFIG = yaml.load(file)
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_config", CONFIG["train_param"]["anomalib_model"]+"_config.yaml"), "r") as file:
        MODEL_CONFIG = yaml.load(file)




def main():
    path=os.path.join(ENV["data_root_dir"],CONFIG['dataset'], "processed",'abnormal', CONFIG["partition"])
    if os.path.exists(path)== False:
        path2=os.path.join(ENV["data_root_dir"],CONFIG['dataset'], "annotations", CONFIG["partition"])
        if os.path.exists(path2):
               print('Transform data to anomalib format')
               format_transform(CONFIG['dataset'],CONFIG["partition"])
        else: 
            print('dataset does not exist')
            exit()





    model_dir = os.path.join(ENV["model_root_dir"], CONFIG["dataset"], CONFIG["model"], CONFIG["train_run_name"]) + '/'

    with open(os.path.join(model_dir, "mlflow_run_id.txt"), 'r') as f:
        previous_run_id = f.read()

    #continue mlflow
    mlflow_run,tags = copy_and_continuous_mlflow(
                            previous_run_id=previous_run_id,
                            experiment_name=CONFIG["experiment_name"]+"_validation", 
                            new_run_name=CONFIG["train_run_name"]+"_"+CONFIG["inference_run_name"],
                            new_run_desc=CONFIG["inference_run_desc"],)
    
  

    mlflow.autolog(disable=True)


    #dataset setting
    datamodule = Folder(
    name=CONFIG['dataset'],
    root=os.path.join(ENV["data_root_dir"],CONFIG['dataset'], "processed",'abnormal', CONFIG["partition"]),
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
    #anomalib.enigne equals to lightning.trainer, refer to pytorch lightning
    engine = Engine(
        normalization=NormalizationMethod.MIN_MAX,
        threshold=CONFIG['train_param']['threshold'],
        task=CONFIG['train_param']['task'],
        image_metrics=CONFIG['train_param']['image_metrics'],
        pixel_metrics=CONFIG['train_param']['pixel_metrics'],
        accelerator=CONFIG['train_param']['accelerator'],
        devices=CONFIG['train_param']['devices'],
        max_epochs=CONFIG['train_param']['max_epochs'],
        default_root_dir=model_dir
    )
    
    test_result=engine.test(model=model,datamodule=datamodule,ckpt_path=os.path.join(model_dir,'model.ckpt'))
    pred_result=engine.predict(model=model,datamodule=datamodule,ckpt_path=os.path.join(model_dir,'model.ckpt'))

    # shutil.rmtree('results')
    print(test_result)
    for metric in list(test_result[0].keys()):
        mlflow.log_metric(metric,test_result[0][metric])

    os.makedirs(os.path.join(model_dir, "result"), exist_ok=True)

    csv_file_path=os.path.join(ENV["data_root_dir"],CONFIG['dataset'], "processed",'abnormal', CONFIG["partition"],'analysis.csv')
    dataset_df = pd.read_csv(csv_file_path)
    res=pd.concat([dataset_df,pd.DataFrame(columns=['pred_labels','pred_scores'])],sort=False)
    for i in range(len(pred_result)):
        for image in range(len(pred_result[i]['image_path'])):
            image_name=os.path.basename(pred_result[i]['image_path'][image])
            res.loc[res.Name==image_name,'pred_labels']=pred_result[i]['pred_labels'][image]
            res.loc[res.Name==image_name,'pred_scores']=pred_result[i]['pred_scores'][image]
    res=res.replace({np.nan: None})
    res.to_csv(os.path.join(model_dir, "result", "analysis.csv"))

    #generate metrics figure,only AUROC/AUPR/AUPRO has this function
    if 'AUROC' in CONFIG['train_param']['image_metrics']:
        auroc=AUROC()
        for batch in pred_result:
            auroc.update(batch['pred_scores'],batch['label'])
        figure, title = auroc.generate_figure()
        figure.savefig(os.path.join(model_dir,'AUROC.png'))
    if 'AUPR' in CONFIG['train_param']['image_metrics']:
        auroc=AUPR()
        for batch in pred_result:
            auroc.update(batch['pred_scores'],batch['label'])
        figure, title = auroc.generate_figure()
        figure.savefig(os.path.join(model_dir, 'AUPR.png'))    

    mlflow.log_artifacts(model_dir + '/result', artifact_path = CONFIG["task"] + '/' + CONFIG["model"] + '/result')

if __name__ == "__main__":
    main()