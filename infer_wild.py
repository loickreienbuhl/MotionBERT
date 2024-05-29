import os
import sys

import numpy as np
import argparse
from tqdm import tqdm
import imageio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from lib.utils.tools import *
from lib.utils.learning import *
from lib.utils.utils_data import flip_data
from lib.data.dataset_wild import WildDetDataset
from lib.utils.vismo import render_and_save

def parse_args():
    parser = argparse.ArgumentParser()
    # LITE MOTION BERT
    # parser.add_argument("--config", type=str, default="module/MotionBERT/configs/pose3d/MB_ft_h36m_global_lite.yaml", help="Path to the config file.")
    # parser.add_argument('-e', '--evaluate', default='module/MotionBERT/checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    # MOTION BERT
    parser.add_argument("--config", type=str, default="module/MotionBERT/configs/pose3d/MB_ft_h36m_global.yaml", help="Path to the config file.")
    parser.add_argument('-e', '--evaluate', default='module/MotionBERT/checkpoint/pose3d/FT_MB_release_MB_ft_h36m/best_epoch.bin', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    
    parser.add_argument('-j', '--json_path', type=str, help='alphapose detection result json path')
    parser.add_argument('-v', '--vid_path', type=str, help='video path')
    parser.add_argument('-o', '--out_path', type=str, help='output path')
    parser.add_argument('--pixel', action='store_true', help='align with pixle coordinates')
    parser.add_argument('--focus', type=int, default=None, help='target person id')
    parser.add_argument('--clip_len', type=int, default=243, help='clip length for network input')
    opts = parser.parse_args()
    return opts

# Modified so that it works for iGAIT, no main before
def main(video_file_path=None, json_path=None, output_path=None):
    # Parse the command-line arguments
    opts = parse_args()
    
    # Override opts with the provided arguments if they are not None
    if video_file_path is not None:
        opts.vid_path = video_file_path
    if json_path is not None:
        opts.json_path = json_path
    if output_path is not None:
        opts.out_path = output_path
    
    # Load the configuration
    args = get_config(opts.config)

    model_backbone = load_backbone(args)
    if torch.cuda.is_available():
        model_backbone = nn.DataParallel(model_backbone)
        model_backbone = model_backbone.cuda()

    print('Loading checkpoint', opts.evaluate)
    checkpoint = torch.load(opts.evaluate, map_location=lambda storage, loc: storage)
    # Remove 'module.' prefix
    new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint['model_pos'].items()}
    model_backbone.load_state_dict(new_state_dict, strict=True)
    model_pos = model_backbone
    model_pos.eval()
    testloader_params = {
        'batch_size': 1,            # Set the batch size according to your requirements
        'shuffle': False,           # Shuffle the data if needed, but it's usually not necessary for evaluation
        'num_workers': 0,           # Use 0 workers for CPU-only execution to avoid multiprocessing
        'pin_memory': False,        # Don't need to pin memory on CPU
        'drop_last': False          # Keep all samples, even if the last batch is smaller
    }

    vid = imageio.get_reader(opts.vid_path,  'ffmpeg')
    fps_in = vid.get_meta_data()['fps']
    vid_size = vid.get_meta_data()['size']
    os.makedirs(opts.out_path, exist_ok=True)

    if opts.pixel:
        # Keep relative scale with pixel coornidates
        wild_dataset = WildDetDataset(opts.json_path, clip_len=opts.clip_len, vid_size=vid_size, scale_range=None, focus=opts.focus)
    else:
        # Scale to [-1,1]
        wild_dataset = WildDetDataset(opts.json_path, clip_len=opts.clip_len, scale_range=[1,1], focus=opts.focus)

    test_loader = DataLoader(wild_dataset, **testloader_params)

    results_all = []
    with torch.no_grad():
        for batch_input in tqdm(test_loader):
            N, T = batch_input.shape[:2]
            if torch.cuda.is_available():
                batch_input = batch_input.cuda()
            if args.no_conf:
                batch_input = batch_input[:, :, :, :2]
            if args.flip:    
                batch_input_flip = flip_data(batch_input)
                predicted_3d_pos_1 = model_pos(batch_input)
                predicted_3d_pos_flip = model_pos(batch_input_flip)
                predicted_3d_pos_2 = flip_data(predicted_3d_pos_flip) # Flip back
                predicted_3d_pos = (predicted_3d_pos_1 + predicted_3d_pos_2) / 2.0
            else:
                predicted_3d_pos = model_pos(batch_input)
            if args.rootrel:
                predicted_3d_pos[:,:,0,:]=0                    # [N,T,17,3]
            else:
                predicted_3d_pos[:,0,0,2]=0
                pass
            if args.gt_2d:
                predicted_3d_pos[...,:2] = batch_input[...,:2]
            results_all.append(predicted_3d_pos.cpu().numpy())

    results_all = np.hstack(results_all)
    results_all = np.concatenate(results_all)
    render_and_save(results_all, '%s/X3D.mp4' % (opts.out_path), keep_imgs=False, fps=fps_in)
    if opts.pixel:
        # Convert to pixel coordinates
        results_all = results_all * (min(vid_size) / 2.0)
        results_all[:,:,:2] = results_all[:,:,:2] + np.array(vid_size) / 2.0
    np.save('%s/X3D.npy' % (opts.out_path), results_all)

if __name__ == '__main__':
    main()