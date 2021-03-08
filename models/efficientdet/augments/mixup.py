import random
import numpy as np

class MixUp(object):
    def __init__(self, p=0.3, lambd=0.5):
        self.lambd = lambd
        self.p = p
        self.img2 = None
        self.boxes2 = None
        self.labels2= None

    def __call__(self, results):
        img1, boxes1, labels1 = [results[k] for k in ('img', 'gt_bboxes', 'gt_labels')]
        if random.random() < self.p and self.img2 is not None and img1.shape[1]==self.img2.shape[1]:
            height = max(img1.shape[0], self.img2.shape[0])
            width = max(img1.shape[1], self.img2.shape[1])
            mixup_image = np.zeros([height, width, 3], dtype='float32')
            mixup_image[:img1.shape[0], :img1.shape[1], :] = img1.astype('float32') * self.lambd
            mixup_image[:self.img2.shape[0], :self.img2.shape[1], :] += self.img2.astype('float32') * (1. - self.lambd)
            mixup_image = mixup_image.astype('uint8')
            mixup_boxes = np.vstack((boxes1, self.boxes2))
            mixup_label = np.hstack((labels1,self.labels2))
            results['img'] = mixup_image
            results['gt_bboxes'] = mixup_boxes
            results['gt_labels'] = mixup_label
        else: 
            pass

        self.img2 = img1
        self.boxes2 = boxes1
        self.labels2 =  labels1
        return results
    
    def __repr__(self):
        return self.__class__.__name__ + f'(lambd={self.lambd})'