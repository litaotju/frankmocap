#! /usr/bin/bash

## Save the output for debug purpose
python -m demo.demo_bodymocap --end_frame 100  --input_path $1 --out_dir ./mocap_output/$1  --save_frame --save_pred_pkl

#python -m demo.demo_visualize_prediction --pkl_dir ./mocap_output/mocap --out_dir ./mocap_output/rendered