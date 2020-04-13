import os
from mmdet.apis import init_detector, inference_detector, show_result
import argparse


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cfg', '--config-file', required=True, type=str)
    parser.add_argument('-ch', '--checkpoint-file', required=True, type=str)
    parser.add_argument('-img', '--image-file', required=True, type=str)
    parser.add_argument('-out', '--out-file', type=str)
    parser.add_argument('-thr', '--threshold', type=float, default=0.5)
    parser.add_argument('-gpu', '--gpu', type=int, default=0)
    return parser


def detect(config_file, checkpoint_file, image_file, return_classes=False):
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    print(model.CLASSES)
    detections = inference_detector(model, image_file)
    if return_classes:
        return detections, model.CLASSES
    else:
        return detections


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    kwargs = vars(args)
    kwargs.pop('gpu')
    detections, classes = detect(args.config_file, args.checkpoint_file, args.image_file, return_classes=True)
    print(len(classes))
    show_result(args.image_file, detections, classes, score_thr=args.threshold, out_file=args.out_file)
