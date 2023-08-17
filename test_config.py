import argparse
from config import log_config 
import logging


def parse_args():
    parser = argparse.ArgumentParser(description='command for evaluate on CUHK-PEDES')
    # Directory
    parser.add_argument('--image_dir', type=str, default='/opt/data/private/Datasets/CUHK-PEDES/imgs/', help='directory to store dataset')
    parser.add_argument('--anno_dir', type=str, default='./cuhkpedes/processed_data', help='directory to store anno file')
    parser.add_argument('--model_path', type=str, default='./checkpoints/CFine-CLIP', help='directory to load checkpoint')
    parser.add_argument('--log_dir', type=str, help='directory to store log')
    parser.add_argument('--pretrain_dir', type=str, default=None,
                        help='the path of vit parameters')
    parser.add_argument('--resnet50_dir', type=str, default=None,
                        help='the path of vit parameters')
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14", "R50-ViT-B_16"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument('--text_model', type=str, default='bert', help='which model used to extract text features')
    parser.add_argument('--img_model', type=str,default='clip', help='which model used to extract image features')
    parser.add_argument('--feature_size', type=int, default=768)
    parser.add_argument('--num_heads', type=int, default=2, help='#num of heads')
    # Default setting
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--epoch_start', type=int)
    parser.add_argument('--checkpoint_dir', type=str, default='')

    parser.add_argument('--lambda_diversity', type=float, default=0.2)
    parser.add_argument('--layer_ids', type=str, default='-1')
    # Hyperparams setting
    parser.add_argument('--img_k_ratio', type=float, default=0.1)     # 0.1
    parser.add_argument('--text_k_ratio', type=float, default=0.2)    # 0.2
    parser.add_argument('--scale_cs', type=float, default=0.07)       # 0.07
    parser.add_argument('--scale_cd', type=float, default=0.01)       # 0.01
    parser.add_argument('--pos_kw', type=float, default=3)            # 3
    parser.add_argument('--pos_kp', type=float, default=3)            # 3

    args = parser.parse_args()
    return args



def config():
    args = parse_args()
    log_config(args, 'test')
    return args
