work_path=$(dirname $0)
PYTHONPATH=$PYTHONPATH:./slowfast \
python tools/run_net.py \
  --cfg $work_path/train.yaml \
  DATA.PATH_TO_DATA_DIR /mnt/ssd/maoyy/code/Maoyy/home/virtualhome-master/src/virtualhome/dataset/dataset_handler/dataset_tomnet \
  DATA.PATH_PREFIX  /mnt/ssd/maoyy/code/Maoyy/home/virtualhome-master/src/virtualhome/dataset/ \
  TRAIN.ENABLE True\
  UNIFORMER.PRETRAIN_NAME "uniformer_small_in1k" \
  DATA.PATH_LABEL_SEPARATOR "," \
  TRAIN.EVAL_PERIOD 10 \
  TRAIN.CHECKPOINT_PERIOD 10 \
  TRAIN.BATCH_SIZE 1\
  MODEL.NUM_CLASSES 10 \
  NUM_GPUS 1 \
  UNIFORMER.DROP_DEPTH_RATE 0.1 \
  SOLVER.MAX_EPOCH 100 \
  SOLVER.BASE_LR 0.0002 \
  SOLVER.WARMUP_EPOCHS 20.0 \
  DATA.TEST_CROP_SIZE 256 \
  TEST.NUM_ENSEMBLE_VIEWS 1 \
  TEST.NUM_SPATIAL_CROPS 1 \
  RNG_SEED 6666 \
  OUTPUT_DIR $work_path/../wo_noise/ 
  # TRAIN.CHECKPOINT_FILE_PATH  ../wo_noise/checkpoints/checkpoint_epoch_00030.pyth \
  # TEST.CHECKPOINT_FILE_PATH  ../wo_noise/checkpoints/checkpoint_epoch_00070.pyth\
  # TEST.ENABLE True

#  nohup sh train.sh > output 2>&1 &
# [1] 21160
