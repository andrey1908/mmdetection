from .coco import CocoDataset
from .registry import DATASETS


@DATASETS.register_module
class TrafficSign(CocoDataset):

    CLASSES = ('traffic_sign',)

