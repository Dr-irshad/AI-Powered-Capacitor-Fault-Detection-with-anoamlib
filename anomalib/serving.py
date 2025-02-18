import os
import cv2
import numpy as np
from src.anomaly_detection.anomalib.model_selection import model_selection
from anomalib.data.utils import read_image
from anomalib.deploy import OpenVINOInferencer
from anomalib.deploy import ExportType

def pred_class(label):
    if label== True:
        return 'Defective'
    else: return 'Defect-free'

def predict(input, model_base_path, model_path=None):
# def predict(model_base_path, input, CONFIG,MODEL_CONFIG):
    #load the openvino inferencer.The variable [path] can point to xml or bin file.
    inferencer=OpenVINOInferencer(
        path=os.path.join(model_base_path,'artifacts','anomaly_detection','anomalib', 'weights','openvino','model.xml'),
        metadata=os.path.join(model_base_path,'artifacts','anomaly_detection','anomalib', 'weights','openvino','metadata.json'),
        device='CPU'
        )

    images=input['images']
    output=[]
    for image in images:
        #The read_image in anomalib divide the image(in nparray format) by 255. 
        #Here we want input the np.array instead of using read_image to read the image from path. 
        image=image/255
        predictions = inferencer.predict(image)
        mask=predictions.pred_mask
        contours,hir=cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour_list=[]
        bbox_list=[]
        for contour in contours:
            contour=np.squeeze(contour)
            x_min = int(np.min(contour[:,0]))
            x_max = int(np.max(contour[:,0]))
            y_min = int(np.min(contour[:,1]))
            y_max = int(np.max(contour[:,1]))
            bbox={
                            "top_left": (x_min,y_max),
                            "bottom_right": (x_max,y_min),
                            "rotation": 0,
                            "category_name": 'Defective',
                            "confidence": None
                        }
            bbox_list.append(bbox)
            contour={"category_name": "Defective","coordinates":contour.tolist()}
            contour_list.append(contour)
        heatmap=predictions.heat_map
        output.append({'Pred_Class':pred_class(predictions.pred_label),"Pred_Prob":predictions.pred_score,'heatmap':heatmap.tolist(),'Mask':contour_list,'Bbox':bbox_list})

    return output

