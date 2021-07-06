DATA_ROOT=depth_data
TEST_SET=$DATA_ROOT/waymo/training/
SPLIT=val
DISP_NET=checkpoints/r18_waymo/05-29-08:47/dispnet_model_best.pth.tar
# DISP_NET=checkpoints/r18_waymo/05-31-09:02/dispnet_checkpoint.pth.tar

python test.py $TEST_SET \
--split $SPLIT \
--pretrained-disp $DISP_NET \
--dataset waymo \
--resnet-layers 18 \
