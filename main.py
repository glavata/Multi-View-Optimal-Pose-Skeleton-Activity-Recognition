"""
Main entry point for multiview optimal pose skeleton-based action recognition.

This module serves as the main execution script, importing the processing functions
from the processes module and executing the desired experiments.
"""

import argparse
import tables
from processes import process_common
from utils.multi_view_util import FuseType, NormType, RotType


def parse_arguments():
    """Parse command line arguments for the process_common function."""
    parser = argparse.ArgumentParser(
        description='Multiview Optimal Pose Skeleton-based Action Recognition',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run visualization with NTU dataset
  python main.py --dataset ntu --benchmark cs --fusion both --method-type visualization --draw-type gif_single

  # Run HMM classification with PKU dataset
  python main.py --dataset pku --benchmark cs --fusion both --method hmm --method-type classification

  # Run HCN training with custom parameters
  python main.py --dataset ntu --benchmark cv --fusion both --method hcn --method-type classification --epochs 50
        """
    )
    
    # Required arguments
    parser.add_argument('--dataset', type=str, choices=['ntu', 'pku'], required=True,
                       help='Dataset to use (ntu or pku)')
    
    # Optional arguments with defaults
    parser.add_argument('--benchmark', type=str, choices=['cv', 'cs', 'all'], default='cv',
                       help='Benchmark type: cv (cross-view), cs (cross-subject), all (default: cv)')
    
    parser.add_argument('--fusion', type=str, choices=['none', 'train_only', 'both'], default='none',
                       help='Multi-view fusion type (default: none)')
    
    parser.add_argument('--method', type=str, choices=['hmm', 'hcn', 'stgcn'], default='hmm',
                       help='Machine learning method (default: hmm)')
    
    parser.add_argument('--method-type', type=str, 
                       choices=['classification', 'visualization', 'validation', 'validation_param', 'regression'],
                       default='classification',
                       help='Processing type (default: classification)')
    
    # Dataset parameters
    parser.add_argument('--classes', type=str, choices=['single', 'all'], default='single',
                       help='Number of classes to use (default: single)')
    
    parser.add_argument('--norm-type', type=str, 
                       choices=['NORM_SKEL_REF', 'NORM_BONE_UNIT_VEC', 'NORM_NECK_TORSO', 'NORM_JOINT_DIFF', 'NO_NORM'],
                       default='NORM_SKEL_REF',
                       help='Normalization type (default: NORM_SKEL_REF)')
    
    parser.add_argument('--rot-type', type=str,
                       choices=['ROT_POSE', 'ROT_SEQ', 'ROT_POSE_REF', 'ROT_SEQ_REF', 'NO_ROT'],
                       default='NO_ROT',
                       help='Rotation type (default: NO_ROT)')
    
    # Fusion parameters
    parser.add_argument('--mv-fuse-type', type=str,
                       choices=['NONE', 'OPT_POSE', 'MID_VIEW_ONLY', 'OPT_POSE_KALMAN'],
                       default='NONE',
                       help='Multi-view fusion type (default: NONE)')
    
    # Visualization parameters
    parser.add_argument('--draw-type', type=str,
                       choices=['none', 'gif_single', 'mv_seq_eq', 'mv_seq_uneq_dtw', 'h_states'],
                       default='none',
                       help='Visualization type (default: none)')
    
    parser.add_argument('--val-type', type=str,
                       choices=['none', 'mv_seq_eq', 'rot_pose_regr'],
                       default='none',
                       help='Validation type (default: none)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs (default: 100)')
    
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for training (default: 64)')
    
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    
    # HMM specific parameters
    parser.add_argument('--hmm-n-comp', type=int, nargs='+', default=[10],
                       help='Number of HMM components (default: [10])')
    
    # Other parameters
    parser.add_argument('--yield-non-full', action='store_true',
                       help='Yield non-full sequences')
    
    parser.add_argument('--dataset-draw', type=str, choices=['train', 'test'], default='train',
                       help='Which dataset to use for visualization (default: train)')
    
    return parser.parse_args()


def convert_args_to_params(args):
    """Convert parsed arguments to the parameter dictionary format expected by process_common."""
    # Convert string choices to enum values
    norm_type_map = {
        'NORM_SKEL_REF': NormType.NORM_SKEL_REF,
        'NORM_BONE_UNIT_VEC': NormType.NORM_BONE_UNIT_VEC,
        'NORM_NECK_TORSO': NormType.NORM_NECK_TORSO,
        'NORM_JOINT_DIFF': NormType.NORM_JOINT_DIFF,
        'NO_NORM': NormType.NO_NORM
    }
    
    rot_type_map = {
        'ROT_POSE': RotType.ROT_POSE,
        'ROT_SEQ': RotType.ROT_SEQ,
        'ROT_POSE_REF': RotType.ROT_POSE_REF,
        'ROT_SEQ_REF': RotType.ROT_SEQ_REF,
        'NO_ROT': RotType.NO_ROT
    }
    
    fuse_type_map = {
        'NONE': FuseType.NONE,
        'OPT_POSE': FuseType.OPT_POSE,
        'MID_VIEW_ONLY': FuseType.MID_VIEW_ONLY,
        'OPT_POSE_KALMAN': FuseType.OPT_POSE_KALMAN
    }
    
    params = {
        'classes': args.classes,
        'norm_type': norm_type_map[args.norm_type],
        'rot_type': rot_type_map[args.rot_type],
        'mv_fuse_type': fuse_type_map[args.mv_fuse_type],
        'draw_type': args.draw_type,
        'val_type': args.val_type,
        'yield_non_full': args.yield_non_full,
        'dataset_draw': args.dataset_draw,
        'hmm_n_comp': args.hmm_n_comp
    }
    
    return params


if __name__ == '__main__':
    tables.file._open_files.close_all()

    # Parse command line arguments
    args = parse_arguments()
    
    # Convert arguments to parameter dictionary
    params = convert_args_to_params(args)
    
    # Run the process
    process_common(
        dataset=args.dataset,
        bm_train_type=args.benchmark,
        mv_fuse_data=args.fusion,
        method=args.method,
        method_type=args.method_type,
        params_d=params
    )

    # Legacy commented examples for reference
    # process_common('ntu', 'cs', "none", method='odcnn',
    #                         params_d={
    #                             'norm_type':NormType.NORM_NECK_TORSO,
    #                             "classes": "single",
    #                             "yield_non_full":True})

    # process_common('ntu', 'cs', "both", method_type='validation',
    #                         params_d={
    #                             'norm_type':NormType.NORM_BONE_UNIT_VEC,
    #                             "mv_fuse_type":FuseType.OPT_POSE,
    #                             "val_type":"rot_pose_regr",
    #                             "classes": "single"})

    # process_common('ntu', 'cs', "both", method_type='visualization',
    #                         params_d={
    #                             'norm_type':NormType.NORM_BONE_UNIT_VEC,
    #                             "mv_fuse_type":FuseType.OPT_POSE,
    #                             "draw_type":"gif_single",
    #                             "classes": "single"})

    # process_common('pku', 'cs', "both", method='hmm',
    #                         params_d={
    #                             "subtype":"full",
    #                             "plot_mv":False,
    #                             "classes": "single",
    #                             'norm_type':NormType.NORM_BONE_UNIT_VEC,
    #                             "mv_fuse_type":FuseType.OPT_POSE,
    #                             "rot_type":RotType.NO_ROT,
    #                             "hmm_n_comp" : [10]})

    # process_common('ntu', 'cs', "both", method='hcn',
    #                         params_d={
    #                             'norm_type':NormType.NORM_BONE_UNIT_VEC,
    #                             "subtype": "full",
    #                             "mv_fuse_type": FuseType.OPT_POSE,
    #                             "plot_mv": False,
    #                             "classes": "single"})

    # process_common('ntu', 'cs', "both", method_type='visualization',
    #                         params_d={
    #                             'norm_type':NormType.NORM_BONE_UNIT_VEC,
    #                             "mv_fuse_type":FuseType.OPT_POSE,
    #                             "draw_type":"mv_seq_uneq_dtw", 
    #                             "classes": "single"})

