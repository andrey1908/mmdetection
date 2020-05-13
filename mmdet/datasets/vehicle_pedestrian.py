from .coco import CocoDataset
from .registry import DATASETS


@DATASETS.register_module
class Vehicle_Pedestrian(CocoDataset):

    CLASSES = ('vehicle', 'pedestrian')

