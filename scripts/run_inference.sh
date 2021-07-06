DISP_NET=weights/r50_model_scale_22_605.tar

# # training
INPUT_DIR=/octant_efs/datasets/kitti_object/training/image_2
OUTPUT_DIR=/octant_efs/datasets/kitti_object/training/predicted_depth
VIS_DIR=/octant_efs/datasets/kitti_object/training/predicted_depth_vis

python3 inference_depth.py --pretrained $DISP_NET --resnet-layers 50 --scale 22.605 \
--img_dir $INPUT_DIR --dep_dir $OUTPUT_DIR --vis_dir $VIS_DIR

# # testing
# INPUT_DIR=/media/disk/datasets/kitti_object/testing/image_2
# OUTPUT_DIR=/media/disk/datasets/kitti_object/testing/predicted_depth

# python3 inference_depth.py --pretrained $DISP_NET --resnet-layers 50 --scale 22.605 \
# --dataset-dir $INPUT_DIR --output-dir $OUTPUT_DIR