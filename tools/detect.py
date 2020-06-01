from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import argparse


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cfg', '--config-file', required=True, type=str)
    parser.add_argument('-ch', '--checkpoint-file', required=True, type=str)
    parser.add_argument('-img', '--image-file', required=True, type=str)
    parser.add_argument('-thr', '--threshold', type=float, default=0.5)
    parser.add_argument('-nms', '--nms', type=float, default=0.45)
    parser.add_argument('-max-dets', '--max-dets', type=int, default=100)
    parser.add_argument('-gpu', '--gpu', type=int, default=0)
    return parser


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()

    model = init_detector(args.config_file, args.checkpoint_file, device=args.gpu)

    model.set_test_parameters(threshold=args.threshold, nms=args.nms, max_dets=args.max_dets)

    detections = inference_detector(model, args.image_file)
    show_result_pyplot(model, args.image_file, detections, score_thr=0.)
