import torch
import time
import cv2
import os
from operators.distributed_wrapper import DistributedWrapper
from configs.rrnet_config import Config
import numpy as np
from operators.rrnet_operator import RRNetOperator
from deep_sort import DeepSort
from utils.functional import draw_bboxes





if __name__ == '__main__':

    print('***** generating bounding box!! *****')
    dis_operator = DistributedWrapper(Config, RRNetOperator)
    dis_operator.deep_sort()

    # deepsort_echinus = DeepSort('./original_ckpt.t7')
    # deepsort_holothurian = DeepSort('./original_ckpt.t7')
    # deepsort_scallop = DeepSort('./original_ckpt.t7')

    num = 0

    print('***** start deep_sort!! *****')
    # for txt in os.listdir('./results/'):
    '''
    for i in range(3700 ,4499 + 1):
        start = time.time()
        bbox_xywh_echinus = []
        bbox_xywh_holothurian = []
        bbox_xywh_scallop = []
        txt_file = open('./results/'+ str(i) + '.txt', 'r')
        img = cv2.imread('./data/2018origin/video/images/'+ str(i) + '.jpg')
        im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for line in txt_file.readlines():
            if int(line.split(',')[5]) == 2:
                bbox = [float(line.split(',')[0]), float(line.split(',')[1]), float(line.split(',')[2]),
                        float(line.split(',')[3]), float(line.split(',')[4]), int(line.split(',')[5])]
                bbox_xywh_echinus.append(bbox)

            if int(line.split(',')[5]) == 1:
                bbox = [float(line.split(',')[0]), float(line.split(',')[1]), float(line.split(',')[2]),
                        float(line.split(',')[3]), float(line.split(',')[4]), int(line.split(',')[5])]
                bbox_xywh_holothurian.append(bbox)

            if int(line.split(',')[5]) == 3:
                bbox = [float(line.split(',')[0]), float(line.split(',')[1]), float(line.split(',')[2]),
                        float(line.split(',')[3]), float(line.split(',')[4]), int(line.split(',')[5])]
                bbox_xywh_scallop.append(bbox)

        # if bbox_xywh is not None:
        if len(bbox_xywh_echinus):
            bbox_xywh_echinus = np.array(bbox_xywh_echinus)
            outputs = deepsort_echinus.update(bbox_xywh_echinus[:, :4], bbox_xywh_echinus[:, 5], img)
            if len(outputs) > 0:
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -1]
                ori_im = draw_bboxes(img, bbox_xyxy, identities)
                cv2.imwrite('./deepsort_echinus/'+ str(i) +'.jpg', ori_im)
        else:
            cv2.imwrite('./deepsort_echinus/' + str(i) + '.jpg', img)

        num += 1
        total = len(os.listdir('./results/'))


        if len(bbox_xywh_holothurian):
            bbox_xywh_holothurian = np.array(bbox_xywh_holothurian)
            outputs = deepsort_holothurian.update(bbox_xywh_holothurian[:, :4], bbox_xywh_holothurian[:, 5], img)
            if len(outputs) > 0:
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -1]
                ori_im = draw_bboxes(img, bbox_xyxy, identities)
                cv2.imwrite('./deepsort_holothurian/'+ str(i) +'.jpg', ori_im)
        else:
            cv2.imwrite('./deepsort_holothurian/' + str(i) + '.jpg', img)


        if len(bbox_xywh_scallop):
            bbox_xywh_scallop = np.array(bbox_xywh_scallop)
            outputs = deepsort_scallop.update(bbox_xywh_scallop[:, :4], bbox_xywh_scallop[:, 5], img)
            if len(outputs) > 0:
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -1]
                ori_im = draw_bboxes(img, bbox_xyxy, identities)
                cv2.imwrite('./deepsort_scallop/'+ str(i) +'.jpg', ori_im)
        else:
            cv2.imwrite('./deepsort_scallop/' + str(i) + '.jpg', img)


        end = time.time()
        print('[{}, {}], time: {}s, fps: {}'.format(num, total, end-start, 1/(end - start)), end='', flush=True)
    '''
    deepsort_echinus = DeepSort('./original_ckpt.t7')
    unconfirmed_num = 0
    for i in range(3700, 4499 + 1):
        start = time.time()
        bbox_xywh_echinus = []
        txt_file = open('./results/' + str(i) + '.txt', 'r')
        img = cv2.imread('./data/2018origin/video2/images/' + str(i) + '.jpg')
        im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for line in txt_file.readlines():
            if int(line.split(',')[5]) == 2:
                bbox = [float(line.split(',')[0]), float(line.split(',')[1]), float(line.split(',')[2]),
                        float(line.split(',')[3]), float(line.split(',')[4]), int(line.split(',')[5])]
                bbox_xywh_echinus.append(bbox)

        # if bbox_xywh is not None:
        if len(bbox_xywh_echinus):
            bbox_xywh_echinus = np.array(bbox_xywh_echinus)
            outputs, unconfirmed = deepsort_echinus.update(bbox_xywh_echinus[:, :4], bbox_xywh_echinus[:, 5], img)
            unconfirmed_num = unconfirmed
            if len(outputs) > 0:
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -1]
                ori_im = draw_bboxes(img, bbox_xyxy,identities=identities)
                cv2.imwrite('./deepsort_echinus/' + str(i) + '.jpg', ori_im)
        else:
            cv2.imwrite('./deepsort_echinus/' + str(i) + '.jpg', img)

        num += 1
        total = len(os.listdir('./results/'))

        end = time.time()
        print('\r[{}, {}], time: {}s, fps: {}'.format(num, total, end - start, 1 / (end - start)), end='', flush=True)
    print('echinus unconfirmed num = {}'.format(unconfirmed_num))
    del deepsort_echinus

    deepsort_holothurian = DeepSort('./original_ckpt.t7')
    unconfirmed_num = 0
    for i in range(3700 ,4499 + 1):

        start = time.time()
        bbox_xywh_holothurian = []
        txt_file = open('./results/'+ str(i) + '.txt', 'r')
        img = cv2.imread('./data/2018origin/video2/images/'+ str(i) + '.jpg')
        im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for line in txt_file.readlines():

            if int(line.split(',')[5]) == 1:
                bbox = [float(line.split(',')[0]), float(line.split(',')[1]), float(line.split(',')[2]),
                        float(line.split(',')[3]), float(line.split(',')[4]), int(line.split(',')[5])]
                bbox_xywh_holothurian.append(bbox)
        num += 1
        total = len(os.listdir('./results/'))
        if len(bbox_xywh_holothurian):
            bbox_xywh_holothurian = np.array(bbox_xywh_holothurian)
            outputs, unconfirmed = deepsort_holothurian.update(bbox_xywh_holothurian[:, :4], bbox_xywh_holothurian[:, 5], img)
            unconfirmed_num = unconfirmed
            if len(outputs) > 0:
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -1]
                ori_im = draw_bboxes(img, bbox_xyxy, identities)
                cv2.imwrite('./deepsort_holothurian/'+ str(i) +'.jpg', ori_im)
        else:
            cv2.imwrite('./deepsort_holothurian/' + str(i) + '.jpg', img)

        end = time.time()
        print('\r[{}, {}], time: {}s, fps: {}'.format(num, total, end-start, 1/(end - start)), end='', flush=True)
    print('holothurian unconfirmed num = {}'.format(unconfirmed_num))
    del deepsort_holothurian

    deepsort_scallop = DeepSort('./original_ckpt.t7')
    unconfirmed_num = 0
    for i in range(3700 ,4499 + 1):

        start = time.time()
        bbox_xywh_scallop = []
        txt_file = open('./results/'+ str(i) + '.txt', 'r')
        img = cv2.imread('./data/2018origin/video2/images/'+ str(i) + '.jpg')
        im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for line in txt_file.readlines():
            if int(line.split(',')[5]) == 3:
                bbox = [float(line.split(',')[0]), float(line.split(',')[1]), float(line.split(',')[2]),
                        float(line.split(',')[3]), float(line.split(',')[4]), int(line.split(',')[5])]
                bbox_xywh_scallop.append(bbox)

        num += 1
        total = len(os.listdir('./results/'))

        if len(bbox_xywh_scallop):
            bbox_xywh_scallop = np.array(bbox_xywh_scallop)
            outputs, unconfirmed = deepsort_scallop.update(bbox_xywh_scallop[:, :4], bbox_xywh_scallop[:, 5], img)
            unconfirmed_num = unconfirmed
            if len(outputs) > 0:
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -1]
                ori_im = draw_bboxes(img, bbox_xyxy, identities)
                cv2.imwrite('./deepsort_scallop/'+ str(i) +'.jpg', ori_im)
        else:
            cv2.imwrite('./deepsort_scallop/' + str(i) + '.jpg', img)
        end = time.time()
        print('\r[{}, {}], time: {}s, fps: {}'.format(num, total, end-start, 1/(end - start)), end='', flush=True)
    print('scallop unconfirmed num = {}'.format(unconfirmed_num))
    del deepsort_scallop


    fps = 10
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    videowriter = cv2.VideoWriter('./echinus.avi', fourcc, fps, (768, 768))
    for name in os.listdir('./deepsort_echinus/'):
        image = cv2.imread('./deepsort_echinus/'+name)
        videowriter.write(image)
    videowriter.release()