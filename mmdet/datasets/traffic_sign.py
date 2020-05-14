from .coco import CocoDataset
from .builder import DATASETS


@DATASETS.register_module
class TrafficSign(CocoDataset):

    CLASSES = ('traffic_sign',)

