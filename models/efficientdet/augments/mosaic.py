import os.path as osp

import os
import cv2
import mmcv
import numpy as np
import pycocotools.mask as maskUtils
import random

class Mosaic(object):

	def __init__(self, to_float32=False, color_type='color', with_bbox=True,
					with_label=True,
					with_mask=False,
					with_seg=False,
					poly2mask=True):

		self.to_float32 = to_float32
		self.color_type = color_type

		# load annotation
		self.with_bbox = with_bbox
		self.with_label = with_label
		self.with_mask = with_mask
		self.with_seg = with_seg
		self.poly2mask = poly2mask

	def _load_bboxes(self, results):
		ann_info = results['ann_info']
		results['gt_bboxes'] = ann_info['bboxes']

		gt_bboxes_ignore = ann_info.get('bboxes_ignore', None)
		if gt_bboxes_ignore is not None:
			results['gt_bboxes_ignore'] = gt_bboxes_ignore
			results['bbox_fields'].append('gt_bboxes_ignore')
		results['bbox_fields'].append('gt_bboxes')
		return results

	def _load_labels(self, results):
		results['gt_labels'] = results['ann_info']['labels']
		return results

	def _poly2mask(self, mask_ann, img_h, img_w):
		if isinstance(mask_ann, list):
			# polygon -- a single object might consist of multiple parts
			# we merge all parts into one mask rle code
			rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
			rle = maskUtils.merge(rles)
		elif isinstance(mask_ann['counts'], list):
			# uncompressed RLE
			rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
		else:
			# rle
			rle = mask_ann
		mask = maskUtils.decode(rle)
		return mask

	def _load_masks(self, results):
		h, w = results['img_info']['height'], results['img_info']['width']
		gt_masks = results['ann_info']['masks']
		if self.poly2mask:
			gt_masks = [self._poly2mask(mask, h, w) for mask in gt_masks]
		results['gt_masks'] = gt_masks
		results['mask_fields'].append('gt_masks')
		return results

	def _load_semantic_seg(self, results):
		results['gt_semantic_seg'] = mmcv.imread(
			osp.join(results['seg_prefix'], results['ann_info']['seg_map']),
			flag='unchanged').squeeze()
		results['seg_fields'].append('gt_semantic_seg')
		return results

	def __call__(self, results):
		
		# hyper-parameter for mosaic 
		assert len(results) == 4
		s = 640 #320
		xc, yc = [int(random.uniform(s * 0.5, s * 1.5)) for _ in range(2)] 
		labels_out = []
		# reading images
		for i in range(len(results)):
			result = results[i]
			if result['img_prefix'] is not None:
				filename = osp.join(result['img_prefix'],
									result['img_info']['filename'])
			
			else:
				filename = result['img_info']['filename']
			img = mmcv.imread(filename, self.color_type)
			h, w = img.shape[:2]


				# place img in img4
			if i == 0:  # top left
				img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.float32)  # base image with 4 tiles
				x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
				x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
			elif i == 1:  # top right
				x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
				x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
			elif i == 2:  # bottom left
				x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
				x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
			elif i == 3:  # bottom right
				x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
				x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

			img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
			padw = x1a - x1b
			padh = y1a - y1b

			###### Extracting label ###########
			if self.with_bbox:
				result = self._load_bboxes(result)
				if result is None:
					return None
			if self.with_label:
				result = self._load_labels(result)
			if self.with_mask:
				result = self._load_masks(result)
			if self.with_seg:
				result = self._load_semantic_seg(result)

			x = result['gt_bboxes']
			labels = x.copy()
			if x.size > 0:  # Normalized xywh to pixel xyxy format
				labels[:, 0] = x[:, 0] + padw
				# valid_mask0 = labels[:, 0] >= 0

				labels[:, 1] = x[:, 1] + padh
				# valid_mask1 = labels[:, 1] >=0

				labels[:, 2] = x[:, 2] + padw
				# valid_mask2 = labels[:, 2] >=0

				labels[:, 3] = x[:, 3] + padh
				# valid_mask3 = labels[:, 3] >=0

				# valid_mask = valid_mask0 * valid_mask1 * valid_mask2 * valid_mask3
				# if len(valid_mask) != len(labels):
					# pass
				# else:
					# labels = labels[valid_mask]
			labels_out.extend(labels.tolist())
		if len(labels_out) > 0:
			result['gt_labels'] = np.ones(len(labels_out), dtype=np.int64)
		else:
			result['gt_labels']=np.array([], dtype=np.int64)

		labels_out = np.asarray(labels_out, dtype= np.float32)
		
		result['filename'] = str(random.randint(0,100)) + '_' + result['img_info']['filename']
		result['img'] = img4
		result['gt_bboxes'] = labels_out


		result['img_shape'] = img4.shape
		result['ori_shape'] = img4.shape
		# Set initial values for default meta_keys
		result['pad_shape'] = img4.shape
		result['scale_factor'] = 1.0
		num_channels = 1 if len(img4.shape) < 3 else img4.shape[2]
		result['img_norm_cfg'] = dict(
			mean=np.zeros(num_channels, dtype=np.float32),
			std=np.ones(num_channels, dtype=np.float32),
			to_rgb=False)

		################## Visualization ###################
		# color = (255, 0, 0) 
		# img_test  = result['img']
		# for bbox in result['gt_bboxes']:
		# 	img_test = cv2.rectangle(img_test, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color = color, thickness=3)
		# cv2.imwrite(os.path.join("/home/cybercore/vinhng/vinh_code/186/test_img",str(random.randint(0,2000))+".jpeg"),img_test)

		# if True:
		# 	raise ValueError

		return result

	def __repr__(self):
			return '{} (to_float32={}, color_type={})'.format(
				self.__class__.__name__, self.to_float32, self.color_type)