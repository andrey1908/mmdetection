from .coco import CocoDataset
from .builder import DATASETS


@DATASETS.register_module
class Car_Pedestrian(CocoDataset):

    CLASSES = ('car', 'pedestrian')

