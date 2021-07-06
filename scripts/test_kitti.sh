DATA_ROOT=depth_data
TEST_SET=$DATA_ROOT/kitti/training/
SPLIT=val

DISP_NET=checkpoints/r18_kitti/05-29-18:33/dispnet_model_best.pth.tar
# DISP_NET=checkpoints/r18_kitti/05-31-08:08/dispnet_model_best.pth.tar

python test.py $TEST_SET \
--split $SPLIT \
--pretrained-disp $DISP_NET \
--dataset kitti \
--resnet-layers 18 \
