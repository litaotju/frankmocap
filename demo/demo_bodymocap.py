# Copyright (c) Facebook, Inc. and its affiliates.

import os
import sys
import os.path as osp
import torch
from torchvision.transforms import Normalize
import numpy as np
import queue
import threading
import torch.multiprocessing as mp

import cv2
import argparse
import json
import pickle
from datetime import datetime

from demo.demo_options import DemoOptions
from bodymocap.body_mocap_api import BodyMocap
from bodymocap.body_bbox_detector import BodyPoseEstimator
import mocap_utils.demo_utils as demo_utils
import mocap_utils.general_utils as gnu
from mocap_utils.timer import Timer

import renderer.image_utils as imu
from renderer.viewer2D import ImShow



def visualize(q, visualizer, args):
    timer = Timer()

    while(True):
        k = cv2.waitKey(1)
        if k==27 or k == ord('q'):    # Esc key to stop
            print("Got ESC/Q key. Stopping...")
            break

        timer.tic()
        data = None
        try:
            data = q.get(timeout=10)
        except queue.Empty:
            print("Queue is empty")
            break
        if data is None:
            break
        img, pred_mesh_list, body_bbox_list, image_path = data

        
        res_img = visualizer.visualize(
            img,
            pred_mesh_list = pred_mesh_list, 
            body_bbox_list = body_bbox_list)
        
        # q.task_done()
        if not args.no_display:
            res_img = res_img.astype(np.uint8)
            ImShow(res_img)
        if args.save_frame and args.out_dir is not None:
            demo_utils.save_res_img(args.out_dir, image_path, res_img)
        del data, img, pred_mesh_list, body_bbox_list, image_path
        timer.toc(bPrint=True,title="Render Time")

def run_body_mocap(q, close_signal, device, args):
    # Set bbox detector
    body_bbox_detector = BodyPoseEstimator(device)

    # Set mocap regressor
    use_smplx = args.use_smplx
    checkpoint_path = args.checkpoint_body_smplx if use_smplx else args.checkpoint_body_smpl
    body_mocap = BodyMocap(checkpoint_path, args.smpl_dir, device, use_smplx)

    #Setup input data to handle different types of inputs
    input_type, input_data = demo_utils.setup_input(args)

    cur_frame = args.start_frame
    video_frame = 0
    timer = Timer()

    while True:
        if close_signal.is_set():
            print("Close signal received, exiting gracefully\n")
            break
        timer.tic()
        # load data
        load_bbox = False
        
        if input_type =='image_dir':
            if cur_frame < len(input_data):
                image_path = input_data[cur_frame]
                img_original_bgr  = cv2.imread(image_path)
            else:
                img_original_bgr = None

        elif input_type == 'bbox_dir':
            if cur_frame < len(input_data):
                print("Use pre-computed bounding boxes")
                image_path = input_data[cur_frame]['image_path']
                hand_bbox_list = input_data[cur_frame]['hand_bbox_list']
                body_bbox_list = input_data[cur_frame]['body_bbox_list']
                img_original_bgr  = cv2.imread(image_path)
                load_bbox = True
            else:
                img_original_bgr = None

        elif input_type == 'video':      
            _, img_original_bgr = input_data.read()
            if video_frame < cur_frame:
                video_frame += 1
                continue
            if img_original_bgr is None:
                break
            width = min(512, img_original_bgr.shape[1])
            height = min(512, img_original_bgr.shape[0])
            ratio_w = width / img_original_bgr.shape[1]
            ratio_h = height / img_original_bgr.shape[0]
            ratio = min(ratio_w, ratio_h)

            height = int(img_original_bgr.shape[0] * ratio)
            width = int(img_original_bgr.shape[1] * ratio)
            img_original_bgr = cv2.resize(img_original_bgr, (width, height), interpolation = cv2.INTER_AREA)

            image_path = None
            if args.out_dir is not None:
                image_path = osp.join(args.out_dir, "frames", f"{cur_frame:05d}.jpg")
            # save the obtained video frames
            if img_original_bgr is not None:
                video_frame += 1
                # If save_pred_pkl is specified, must save the frames, otherwise there is no point
                if args.save_frame or args.save_pred_pkl:
                    gnu.make_subdir(image_path)
                    cv2.imwrite(image_path, img_original_bgr)

        elif input_type == 'webcam':    
            _, img_original_bgr = input_data.read()

            if video_frame < cur_frame:
                video_frame += 1
                continue
            # save the obtained video frames
            image_path = osp.join(args.out_dir, "frames", f"scene_{cur_frame:05d}.jpg")
            if img_original_bgr is not None:
                video_frame += 1
                if args.save_frame:
                    gnu.make_subdir(image_path)
                    cv2.imwrite(image_path, img_original_bgr)
        else:
            assert False, "Unknown input_type"

        cur_frame +=1
        if img_original_bgr is None or cur_frame > args.end_frame:
            break   
        print("--------------------------------------")

        if load_bbox:
            body_pose_list = None
        else:
            body_pose_list, body_bbox_list = body_bbox_detector.detect_body_pose(
                img_original_bgr)
        hand_bbox_list = [None, ] * len(body_bbox_list)

        # save the obtained body & hand bbox to json file
        if args.save_bbox_output: 
            demo_utils.save_info_to_json(args, image_path, body_bbox_list, hand_bbox_list)

        if len(body_bbox_list) < 1: 
            print(f"No body deteced: {image_path}")
            continue
            
        #Sort the bbox using bbox size 
        # (to make the order as consistent as possible without tracking)
        bbox_size =  [ (x[2] * x[3]) for x in body_bbox_list]
        idx_big2small = np.argsort(bbox_size)[::-1]
        body_bbox_list = [ body_bbox_list[i] for i in idx_big2small ]
        if args.single_person and len(body_bbox_list)>0:
            body_bbox_list = [body_bbox_list[0], ]       
        
        # Body Pose Regression
        pred_output_list = body_mocap.regress(img_original_bgr, body_bbox_list)
        assert len(body_bbox_list) == len(pred_output_list)

        # extract mesh for rendering (vertices in image space and faces) from pred_output_list
        pred_mesh_list = demo_utils.extract_mesh_from_output(pred_output_list)

        q.put((img_original_bgr, pred_mesh_list, body_bbox_list, image_path))

        # save predictions to pkl
        if args.save_pred_pkl:
            demo_type = 'body'
            demo_utils.save_pred_to_pkl(
                args, demo_type, image_path, body_bbox_list, hand_bbox_list, pred_output_list)

        timer.toc(bPrint=True,title="Detect Time")

    #save images as a video
    if not args.no_video_out and input_type in ['video', 'webcam']:
        demo_utils.gen_video_out(args.out_dir, args.seq_name)

    if input_type =='webcam' and input_data is not None:
        input_data.release()


def main():
    args = DemoOptions().parse()

    detect_device = torch.device('cpu')
    render_device = torch.device('cpu') 
    if torch.cuda.is_available():
        detect_device = torch.device('cuda:0')
        render_device = torch.device('cuda:1') if torch.cuda.device_count() > 1 else detect_device

    assert torch.cuda.is_available(), "Current version only supports GPU"

    # Set Visualizer
    if args.renderer_type in ['pytorch3d', 'opendr']:
        from renderer.screen_free_visualizer import Visualizer
    else:
        from renderer.visualizer import Visualizer
    visualizer = Visualizer(args.renderer_type, render_device)

    mp.set_start_method('spawn')
    max_size = 24
    q = mp.Queue(max_size)
    close_signal = mp.Event()

    p = mp.Process(target=run_body_mocap, args=(q, close_signal, detect_device, args), daemon=True)
    p.start()

    visualize(q, visualizer, args) 

    # send close signal to the process after visualization done
    close_signal.set()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()