import os
import matplotlib.pyplot as plt
from tools.predict import predict
from dataset_scripts.metrics_eval import evaluate_detections, extract_mAP, extract_AP, get_classes
from dataset_scripts.utils.coco_tools import leave_boxes
from tqdm import tqdm
import argparse
import mmcv
import csv
import json


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cfg', '--config-file', required=True, type=str)
    parser.add_argument('-set', '--set-of-data', type=str, choices=['train', 'val', 'test'], default='val')
    parser.add_argument('-ch-fld', '--checkpoints-folder', type=str)
    parser.add_argument('-rep-fld', '--report-folder', type=str)
    parser.add_argument('-img-fld', '--images-folder', type=str)
    parser.add_argument('-ann', '--annotations-file', type=str)
    parser.add_argument('-area', '--area', nargs=2, type=str, default=['0**2', '1e5**2'])
    parser.add_argument('-shape', '--shape', nargs=2, type=int, default=(None, None))
    parser.add_argument('-add', '--add', action='store_true')
    parser.add_argument('-dont-repredict', '--dont-repredict', dest='repredict', action='store_false')
    parser.add_argument('-mmdet-fld', '--mmdetection-folder', type=str, default='')
    parser.add_argument('-gpu', '--gpu', type=int, default=0)
    return parser


def get_existing_information(report_folder):
    existing_epochs, existing_metrics = list(), list()
    with open(os.path.join(report_folder, 'metrics.csv'), 'r') as f:
        existing_information = csv.reader(f, delimiter=' ')
        for new_information in existing_information:
            existing_epochs.append(int(new_information[0]))
            existing_metrics.append(list(map(float, new_information[1:])))
    return existing_epochs, existing_metrics


def create_folders(report_folder):
    if not os.path.exists(report_folder):
        os.mkdir(report_folder)
    if not os.path.exists(os.path.join(report_folder, 'predictions')):
        os.mkdir(os.path.join(report_folder, 'predictions'))


def get_checkpoint_files(checkpoints_folder, existing_epochs):
    checkpoint_files = os.listdir(checkpoints_folder)
    epochs = []
    for checkpoint_file in checkpoint_files:
        if checkpoint_file.startswith("epoch"):
            if not int(checkpoint_file[6:-4]) in existing_epochs:
                epochs.append(int(checkpoint_file[6:-4]))
    checkpoint_files = []
    for epoch in epochs:
        checkpoint_files.append(os.path.join(checkpoints_folder, "epoch_{}.pth".format(epoch)))
    return checkpoint_files, epochs


def run_models(config_file, checkpoint_files, epochs, report_folder, images_folder, annotations_file, repredict=True,
               gpu=0):
    for checkpoint_file, epoch in tqdm(list(zip(checkpoint_files, epochs))):
        out_file = os.path.join(report_folder, 'predictions', 'epoch_{}.json'.format(epoch))
        if os.path.exists(out_file) and not repredict:
            continue
        predict(config_file, checkpoint_file, out_file, detections_only=True, images_folder=images_folder,
                images_file=annotations_file, threshold=0.001, nms=0.45, max_dets=1000, gpu=gpu)


def calculate_metrics(epochs, report_folder, annotations_file, area=(0**2, 1e5**2), shape=(None, None)):
    metrics = list()
    # kostil' #
    indexes_to_correct = list()
    ###########
    with open(annotations_file, 'r') as f:
        annotations_dict = json.load(f)
    leave_boxes(annotations_dict, area=area, width=shape[0], height=shape[1])

    for epoch in tqdm(epochs):
        detections_file = os.path.join(report_folder, 'predictions/epoch_{}.json'.format(epoch))
        with open(detections_file, 'r') as f:
            detections_dict = json.load(f)
        # kostil' #
        if detections_dict == list():
            metrics.append(None)
            indexes_to_correct.append(len(metrics)-1)
            continue
        ###########
        detections_dict_with_images = {'images': annotations_dict['images'], 'annotations': detections_dict}
        leave_boxes(detections_dict_with_images, area=area, width=shape[0], height=shape[1])
        detections_dict = detections_dict_with_images['annotations']

        results = evaluate_detections(annotations_dict, detections_dict)
        classes = get_classes(results)
        metric = [extract_mAP(results)]
        metric += extract_AP(results, classes)
        metrics.append(metric)
    # kostil' #
    for index in indexes_to_correct:
        metrics[index] = [0] * (len(classes)+1)
    ###########
    return metrics, classes


def save_metrics(epochs, metrics, classes, report_folder):
    with open(os.path.join(report_folder, 'metrics.csv'), 'w') as f:
        writer = csv.writer(f, delimiter=' ')
        for epoch, metric in zip(epochs, metrics):
            writer.writerow([epoch] + metric)
    with open(os.path.join(report_folder, 'classes.txt'), 'w') as f:
        for i, cl in enumerate(classes):
            f.write(cl)
            if i != len(classes)-1:
                f.write(' ')
    mAP = list()
    APs = list()
    for metric in metrics:
        mAP.append(metric[0])
        APs.append(metric[1:])
    plt.plot(epochs, mAP)
    plt.grid()
    plt.savefig(os.path.join(report_folder, 'mAP.png'))
    plt.close()
    plt.plot(epochs, APs)
    plt.grid()
    plt.savefig(os.path.join(report_folder, 'APs.png'))
    plt.close()


def complete_args(config_file, set_of_data, checkpoints_folder, report_folder, images_folder, annotations_file,
                  mmdetection_folder):
    cfg = mmcv.Config.fromfile(config_file)
    if checkpoints_folder is None:
        checkpoints_folder = os.path.join(mmdetection_folder, cfg.work_dir)
    if report_folder is None:
        report_folder = os.path.join(mmdetection_folder, cfg.work_dir, 'report')
    if images_folder is None:
        images_folder = os.path.join(mmdetection_folder, cfg.data[set_of_data].img_prefix)
    if annotations_file is None:
        annotations_file = os.path.join(mmdetection_folder, cfg.data[set_of_data].ann_file)
    return checkpoints_folder, report_folder, images_folder, annotations_file


def report(config_file, checkpoints_folder=None, report_folder=None, images_folder=None, annotations_file=None,
           set_of_data='val', area=(0**2, 1e5**2), shape=(None, None), add=False, repredict=True, gpu=0,
           mmdetection_folder=''):
    assert set_of_data in ['train', 'val', 'test']
    checkpoints_folder, report_folder, images_folder, annotations_file = complete_args(config_file, set_of_data,
                                checkpoints_folder, report_folder, images_folder, annotations_file, mmdetection_folder)
    if area[1] == -1:
        area = (area[0], 1e5 ** 2)
    if add:
        existing_epochs, existing_metrics = get_existing_information(report_folder)
    else:
        existing_epochs, existing_metrics = list(), list()
    create_folders(report_folder)
    checkpoint_files, epochs = get_checkpoint_files(checkpoints_folder, existing_epochs)
    run_models(config_file, checkpoint_files, epochs, report_folder, images_folder, annotations_file, repredict, gpu)
    metrics, classes = calculate_metrics(epochs, report_folder, annotations_file, area, shape)
    epochs += existing_epochs
    metrics += existing_metrics
    epochs, metrics = zip(*sorted(zip(epochs, metrics)))
    save_metrics(epochs, metrics, classes, report_folder)


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    args.area = list(map(eval, args.area))
    kwargs = vars(args)
    report(**kwargs)
