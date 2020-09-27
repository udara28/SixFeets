"""
This script estimate the distance of visible on-road objects, given:
    * pre-computed homography matrix between to warp road plane
    * pre-trained YOLOv3 object detector (args.yolo_weights)
"""
from __future__ import division


import argparse
from pathlib import Path

import cv2
from torch.utils.data import DataLoader
from torch.autograd import Variable

from view_utils import draw_detections
from view_utils import name_to_color
from yolo.models import *
from yolo.utils.datasets import *
from yolo.utils.utils import *
from yolo_utils import postprocess_yolo_detections
from imutils.video import FPS, WebcamVideoStream

import sys

from ssd.data import BaseTransform, VOC_CLASSES as labelmap
from ssd.ssd import build_ssd

parser = argparse.ArgumentParser()
parser.add_argument('yolo_weights', type=str, help='Pre-trained YOLO weights')
parser.add_argument('--frames_dir', default=Path('./data/frames'))
parser.add_argument('--homography_data', default=Path('./data/homography.npz'),
                    help='Pre-computed homography')
parser.add_argument('--output_dir', type=Path, default='/tmp/distances')
parser.add_argument('--config_path', type=str, default='config/yolov3.cfg', help='path to model config file')
parser.add_argument('--class_path', type=str, default='yolo/data/coco.names', help='path to class label file')
parser.add_argument('--conf_thres', type=float, default=0.7, help='object confidence threshold')
parser.add_argument('--nms_thres', type=float, default=0.4, help='IoI threshold for non-maximum suppression')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--n_cpu', type=int, default=8, help='data loader threads')
parser.add_argument('--img_size', type=int, default=416, help='size of each image dimension')
parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda')
args = parser.parse_args()
print(args)


COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
FONT = cv2.FONT_HERSHEY_SIMPLEX

def predict(frame):
    height, width = frame.shape[:2]
    x = torch.from_numpy(transform(frame)[0]).permute(2, 0, 1)
    x = Variable(x.unsqueeze(0))
    y = net(x)  # forward pass
    detections = y.data
    # scale each detection back up to the image
    scale = torch.Tensor([width, height, width, height])
    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= 0.6:
            pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
            cv2.rectangle(frame,
                          (int(pt[0]), int(pt[1])),
                          (int(pt[2]), int(pt[3])),
                          COLORS[i % 3], 2)
            cv2.putText(frame, labelmap[i - 1], (int(pt[0]), int(pt[1])),
                        FONT, 2, (255, 255, 255), 2, cv2.LINE_AA)
            j += 1
    return frame

def detect(frame):
    height, width = frame.shape[:2]
    x = torch.from_numpy(transform(frame)[0]).permute(2, 0, 1)
    x = Variable(x.unsqueeze(0))
    y = net(x)  # forward pass
    detections = y.data
    return detections

def postprocess_ssd_detections(ssd_detections: torch.Tensor,
                                input_size: int, image_shape: int,
                                filter_classes=None):
    """
    Post-process YOLO detections
    """

    # The amount of padding that was added
    h, w, c = image_shape
    pad_x = max(h - w, 0) * (input_size / max([h, w]))
    pad_y = max(w - h, 0) * (input_size / max([h, w]))
    # Image height and width after padding is removed
    unpad_h = input_size - pad_y
    unpad_w = input_size - pad_x

    output = []

    detections = ssd_detections
    scale = torch.Tensor([w, h, w, h])
    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= 0.6:
            pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
            cv2.rectangle(frame,
                          (int(pt[0]), int(pt[1])),
                          (int(pt[2]), int(pt[3])),
                          COLORS[i % 3], 2)
            cv2.putText(frame, labelmap[i - 1], (int(pt[0]), int(pt[1])),
                        FONT, 2, (255, 255, 255), 2, cv2.LINE_AA)
            j += 1

            list_item = {
                'name': labelmap[i-1],
                'coords': [int(pt[0]), int(pt[1]), int(pt[2]), int(pt[3])],
                # Compute detection midpoint on the ground
                #  this is the point that will be warped by homography
                'ground_mid': (int(pt[2] // 2), int(pt[3]))
            }

            if filter_classes is not None:
                if list_item['name'] not in filter_classes:
                    continue

            output.append(list_item)

    return output

if __name__ == '__main__':

    # Load pre-computed homography matrix
    h_data = np.load(args.homography_data)
    H = h_data['H']
    bev_h, bev_w = h_data['bev_h'], h_data['bev_w']
    pix_per_meter = h_data['pix_per_meter']

    #if torch.cuda.is_available():
    #    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    # Set up detector
    #model = Darknet(config_path='yolo/config/yolov3.cfg', img_size=args.img_size)
    #model.load_weights('weights/ssd_300_VOC0712.pth')
    #if args.device == 'cuda':
    #    model.cuda()

    #Tensor = torch.cuda.FloatTensor if args.device == 'cuda' else torch.FloatTensor
    Tensor = torch.cuda.FloatTensor

    fps = FPS().start()

    net = build_ssd('test', 300, 21)    # initialize SSD
    net.load_state_dict(torch.load('weights/ssd_300_VOC0712.pth'))
    transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0))
    #dataloader = DataLoader(
    #    ImageFolder(args.frames_dir, img_size=args.img_size),
    #    batch_size=1, shuffle=False, num_workers=args.n_cpu)

    # start video stream thread, allow buffer to fill
    print("[INFO] starting threaded video stream...")
    stream = WebcamVideoStream(src=0).start()  # default camera
    time.sleep(1.0)

    while True:
        # grab next frame
        frame = stream.read()
        #input_img = frame;
        key = cv2.waitKey(1) & 0xFF

        # update FPS counter
        #fps.update()
        #frame = predict(frame)
        birdeye = cv2.warpPerspective(frame, H, (bev_w, bev_h))

        # Get detections
        with torch.no_grad():
            #input_img = Variable(torch.Tensor(input_img))
            detections = detect(frame) #net(input_img) #model(input_img)
            #detections = non_max_suppression(detections, 80, args.conf_thres,
            #                                 args.nms_thres)
            #print(detections)
        if detections[0] is not None:
            keep_classes = ['car', 'person']
            output = postprocess_ssd_detections(detections,
                                                 image_shape=frame.shape,
                                                 input_size=args.img_size,
                                                 filter_classes=keep_classes)

            # Warp midpoints in birdeye view
            for o in output:
                midpoint = np.concatenate([o['ground_mid'], np.ones(1)])

                midpoint_warped = H @ midpoint
                midpoint_warped /= midpoint_warped[-1]
                midpoint_warped = midpoint_warped[:-1]

                # Store projected points
                o['ground_mid_warped'] = tuple(int(a) for a in midpoint_warped)

            # Exploit the birdeye view to compute distances
            ego_bev_xy = bev_w // 2, bev_h  # Ego position in birdeye view
            for o in output:
                x, y = o['ground_mid_warped']
                delta = [x, y] - np.asarray(ego_bev_xy)
                dist_pix = np.sqrt(np.sum(delta ** 2))

                o['dist_meter'] = dist_pix / pix_per_meter

            # Draw bounding boxes, text, lines etc.
            frame = draw_detections(frame, birdeye, output, name_to_color)

        resize_f = 0.6
        #warped_show = cv2.resize(birdeye, dsize=None, fx=resize_f, fy=resize_f)
        #image_show = cv2.resize(frame, dsize=None, fx=resize_f, fy=resize_f)
        #blend_show = image_show

        #ratio = blend_show.shape[0] / warped_show.shape[0]
        #warped_show = cv2.resize(warped_show, dsize=None, fx=ratio, fy=ratio)
        #cat_show = np.concatenate([blend_show, warped_show], axis=1)

        #if not args.output_dir.exists():
        #    args.output_dir.mkdir(parents=True, exist_ok=True)
        #cv2.imwrite(str(args.output_dir / f'{batch_i:06d}.jpg'), cat_show)

        # Show output
        #cv2.imshow('Output (press any key to proceed)', cat_show)
        #cv2.waitKey(0)

        cv2.imshow('frame', frame)

        if key == 27:  # exit
            break

    stream.stop()
    fps.stop()

