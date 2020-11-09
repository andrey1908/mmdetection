import os
from mmdet.apis import init_detector, inference_detector
import mmcv
from PIL import Image
from tqdm import tqdm
import argparse
import json


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cfg', '--config-file', required=True, type=str)
    parser.add_argument('-ch', '--checkpoint-file', required=True, type=str)
    parser.add_argument('-out', '--out-file', required=True, type=str)
    parser.add_argument('-set', '--set-of-data', type=str, choices=['train', 'val', 'test'], default='val')
    parser.add_argument('-dets-only', '--detections-only', action='store_true')
    parser.add_argument('-img-fld', '--images-folder', type=str)
    parser.add_argument('-img', '--images-file', type=str)
    parser.add_argument('-thr', '--threshold', type=float, default=0.)
    parser.add_argument('-mmdet-fld', '--mmdetection-folder', type=str, default='')
    parser.add_argument('-gpu', '--gpu', type=int, default=0)
    return parser


def get_images(images_folder, images_file=None):
    if images_file is None:
        return get_images_from_folder(images_folder)
    if images_file.endswith('.json'):
        return get_images_from_json(images_folder, images_file)
    return None, None, None


def get_images_from_folder(images_folder):
    images_names = sorted(os.listdir(images_folder))
    images_ids, images_files = list(), list()
    image_id = 0
    for image_name in images_names:
        images_ids.append(image_id)
        image_id += 1
        images_files.append(os.path.join(images_folder, image_name))
    return images_names, images_ids, images_files


def get_images_from_json(images_folder, json_file):
    with open(json_file, 'r') as f:
        json_dict = json.load(f)
    images = json_dict['images']
    images_names, images_ids, images_files = list(), list(), list()
    for image in images:
        images_names.append(image['file_name'])
        images_ids.append(image['id'])
        images_files.append(os.path.join(images_folder, image['file_name']))
    return images_names, images_ids, images_files


def init_coco(classes):
    json_dict = {'images': list(), 'annotations': list(), 'categories': list()}
    for class_id, class_name in enumerate(classes):
        category = {'name': class_name, 'id': class_id+1}
        json_dict['categories'].append(category)
    return json_dict


def add_predictions_to_coco(image_file, image_id, image_name, predictions, threshold, json_dict):
    image = dict()
    image['id'] = image_id
    image['file_name'] = image_name
    w, h = Image.open(image_file).size
    image['width'] = w
    image['height'] = h
    json_dict['images'].append(image)

    num_classes = len(json_dict['categories'])
    for cl_id in range(num_classes):
        for k in range(len(predictions[cl_id])):
            if predictions[cl_id][k][4] < threshold:
                continue
            annotation = dict()
            if len(json_dict['annotations']) == 0:
                annotation['id'] = 1
            else:
                annotation['id'] = json_dict['annotations'][-1]['id'] + 1
            annotation['iscrowd'] = 0
            annotation['image_id'] = image_id
            annotation['category_id'] = cl_id + 1
            xtl, ytl, xbr, ybr, score = map(float, predictions[cl_id][k])
            annotation['bbox'] = [max(xtl, 0), max(ytl, 0),
                                  min(xbr, w-1) - max(xtl, 0) + 1,
                                  min(ybr, h-1) - max(ytl, 0) + 1]
            annotation['area'] = annotation['bbox'][2] * annotation['bbox'][3]
            annotation['score'] = score
            json_dict['annotations'].append(annotation)


def complete_args(config_file, set_of_data, images_folder, images_file, mmdetection_folder):
    cfg = mmcv.Config.fromfile(config_file)
    if images_folder is None:
        images_folder = os.path.join(mmdetection_folder, cfg.data[set_of_data].img_prefix)
        if images_file is None:
            images_file = os.path.join(mmdetection_folder, cfg.data[set_of_data].ann_file)
    return images_folder, images_file


def predict(config_file, checkpoint_file, out_file=None, detections_only=False, set_of_data='val',
            images_folder=None, images_file=None, threshold=0., mmdetection_folder=''):
    assert set_of_data in ['train', 'val', 'test']
    images_folder, images_file = complete_args(config_file, set_of_data, images_folder, images_file, mmdetection_folder)
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    classes = model.CLASSES
    images_names, images_ids, images_files = get_images(images_folder, images_file)
    json_dict = init_coco(classes)
    for image_name, image_id, image_file in tqdm(list(zip(images_names, images_ids, images_files))):
        #TODO inference_detector initilizes test pipeline every time it is called. It may slow code performance.
        predictions = inference_detector(model, image_file)
        add_predictions_to_coco(image_file, image_id, image_name, predictions, threshold, json_dict)
    if detections_only:
        json_dict = json_dict['annotations']
    if out_file:
        with open(out_file, 'w') as f:
            json.dump(json_dict, f, indent=2)
    return json_dict


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    kwargs = vars(args)
    kwargs.pop('gpu')
    predict(**kwargs)
