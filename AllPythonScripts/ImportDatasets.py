
from roboflow import Roboflow
rf = Roboflow(api_key="HahXqnKfPHiO1vdUK7Lb")
project = rf.workspace("xardariii").project("xardarii-yc")
dataset = project.version(3).download("yolov7")