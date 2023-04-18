classes_to_filter = None  # You can give list of classes to filter by name, Be happy you don't have to put class number. ['train','person' ]

opt  = {
    
    "weights": "F:/YOLORepository/yolov7/runs/train/exp3/weights/best.pt",        # Path to weights file default weights are for nano model
    "yaml"   : "xardarii-yc-3/data.yaml",
    "img-size": 640,        # default image size
    "conf-thres": 0.25,     # confidence threshold for inference.
    "iou-thres" : 0.45,     # NMS IoU threshold for inference.
    "device" : '0',         # device to run our model i.e. 0 or 0,1,2,3 or cpu
    "classes" : classes_to_filter  # list of classes to filter or None
}