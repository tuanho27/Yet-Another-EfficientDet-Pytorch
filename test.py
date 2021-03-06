# Author: Zylo117

"""
Simple Inference Script of EfficientDet-Pytorch
"""
import time
import torch
from torch.backends import cudnn
from matplotlib import colors
import cv2
import numpy as np
import argparse
import pandas as pd
from tqdm import tqdm
import os

from models.efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box
from models.backbone import EfficientDetBackbone


def display(preds, imgs, imshow=True, imwrite=False):
    for i in range(len(imgs)):
        if len(preds[i]['rois']) == 0:
            continue
        imgs[i] = imgs[i].copy()
        for j in range(len(preds[i]['rois'])):
            x1, y1, x2, y2 = preds[i]['rois'][j].astype(np.int)
            obj = obj_list[preds[i]['class_ids'][j]]
            score = float(preds[i]['scores'][j])
            plot_one_box(imgs[i], [x1, y1, x2, y2], label=obj,score=score,color=color_list[get_index_label(obj, obj_list)])
        if imshow:
            cv2.imshow('img', imgs[i])
            cv2.waitKey(0)
        if imwrite:
            cv2.imwrite(f'test/img_inferred_d{compound_coef}_this_repo_{i}.jpg', imgs[i])


if __name__== "__main__":

    parser = argparse.ArgumentParser('Yet Another EfficientDet Pytorch Test Update')
    parser.add_argument('--data_path', type=str, default='datasets/', help='the root folder of dataset')
    parser.add_argument('-c', '--compound_coef', type=int, default=0, help='coefficients of efficientdet')
    parser.add_argument('-w', '--load_weights', type=str, default=None,
                        help='whether to load weights from a checkpoint, set None to initialize, set \'last\' to load last checkpoint')
    parser.add_argument('--speed_test', action='store_true', help='FPS speed testing')
    parser.add_argument('--threshold', type=float, default=0.1)
    parser.add_argument('--iou_threshold', type=float, default=0.3)
    args = parser.parse_args()

    compound_coef = args.compound_coef
    force_input_size = None  # set None to use default size
    # img_path = '/mnt/fast_house/dataset/vinxray/test_images/ff91fb82429a27521bbec8569b041f02.png'
    submission_df = pd.read_csv('/mnt/fast_house/dataset/vinxray/sample_submission.csv')
    cls_model_b6 = pd.read_csv('./b6.csv')
    
    # replace this part with your project's anchor config
    anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
    anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
    threshold = args.threshold
    iou_threshold = args.iou_threshold
    cls_threshold = 0.08
    use_cuda = True
    use_float16 = False
    cudnn.fastest = True
    cudnn.benchmark = True

    obj_list = ["Aortic enlargement","Atelectasis","Calcification","Cardiomegaly","Consolidation",
            "ILD","Infiltration","Lung Opacity","Nodule/Mass","Other lesion","Pleural effusion",
            "Pleural thickening","Pneumothorax","Pulmonary fibrosis"]

    color_list = standard_to_bgr(STANDARD_COLORS)
    # tf bilinear interpolation is different from any other's, just make do
    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
    input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size

    ## declare model
    model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                                ratios=anchor_ratios, scales=anchor_scales)
    model.load_state_dict(torch.load(f'{args.load_weights}', map_location='cpu'))
    model.requires_grad_(False)
    model.eval()
    if use_cuda:
        model = model.cuda()
    if use_float16:
        model = model.half()
    
    ## dataset loading
    # ori_imgs, framed_imgs, framed_metas = preprocess(img_path, max_size=input_size)
    # if use_cuda:
    #     x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
    # else:
    #     x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)
    # x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)
    image_ids = []
    PredictionStrings = []

    for i, ids in tqdm(enumerate(cls_model_b6.image_id)):
        image_ids.append(ids)
        
        if cls_model_b6.iloc[i].target < cls_threshold:
            PredictionStrings.append("14 1 0 0 1 1")
        
        else:
            img_path = os.path.join(args.data_path, f'{ids}.png')
            ori_imgs, framed_imgs, framed_metas = preprocess(img_path, max_size=input_size)
            if use_cuda:
                x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
            else:
                x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)
            x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)
            with torch.no_grad():
                features, regression, classification, anchors = model(x)
                regressBoxes = BBoxTransform()
                clipBoxes = ClipBoxes()

                out = postprocess(x,
                                anchors, regression, classification,
                                regressBoxes, clipBoxes,
                                threshold, iou_threshold)
            out = invert_affine(framed_metas, out)
            valid_strings = ''
            for idx in range(len(out[0]['rois'])):
                id_str = str(out[0]['class_ids'][idx])
                score_str = str(round(out[0]['scores'][idx], 2))
                bbox_str = list(np.round(out[0]['rois'][idx].reshape(-1),1).astype(str))
                valid_strings += ' '.join(map(str, [id_str, score_str, bbox_str[0],bbox_str[1], bbox_str[2], bbox_str[3],'']))

            PredictionStrings.append(valid_strings)

    pred_df = pd.DataFrame({'image_id':image_ids,'PredictionString':PredictionStrings})
    pred_df.to_csv(f"sub_effdet_d{args.compound_coef}_{args.threshold}_{args.iou_threshold}_cls_{cls_threshold}.csv", index=False)
    
    if args.speed_test:
        img_path = '/mnt/fast_house/dataset/vinxray/test_images/ffaa288c8abca300974f043b57d81521.png'
        ori_imgs, framed_imgs, framed_metas = preprocess(img_path, max_size=input_size)
        if use_cuda:
            x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
        else:
            x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)
        x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)
        print('running speed test...')
        with torch.no_grad():
            print('test1: model inferring and postprocessing')
            print('inferring image for 10 times...')
            t1 = time.time()
            for _ in range(10):
                _, regression, classification, anchors = model(x)

                out = postprocess(x,
                                anchors, regression, classification,
                                regressBoxes, clipBoxes,
                                threshold, iou_threshold)
                out = invert_affine(framed_metas, out)

            t2 = time.time()
            tact_time = (t2 - t1) / 10
            print(f'{tact_time} seconds, {1 / tact_time} FPS, @batch_size 1')
            display(out, ori_imgs, imshow=False, imwrite=True)

            # uncomment this if you want a extreme fps test
            # print('test2: model inferring only')
            # print('inferring images for batch_size 32 for 10 times...')
            # t1 = time.time()
            # x = torch.cat([x] * 32, 0)
            # for _ in range(10):
            #     _, regression, classification, anchors = model(x)
            #
            # t2 = time.time()
            # tact_time = (t2 - t1) / 10
            # print(f'{tact_time} seconds, {32 / tact_time} FPS, @batch_size 32')
