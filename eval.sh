work_path=$(dirname $0)
PYTHONPATH=$PYTHONPATH:./slowfast \
python tools/run_net.py \
  --cfg $work_path/train.yaml \
  DATA.PATH_TO_DATA_DIR ./data_list/BDI/noise_2713/ \
  DATA.PATH_PREFIX   /mnt/ssd/maoyy/code/Maoyy/home/virtualhome-master/src/virtualhome/dataset/ \
  TEST.ENABLE True\
  TEST.CHECKPOINT_FILE_PATH  ../noise/checkpoints/checkpoint_epoch_00090.pyth\
  DATA.SPECIE_CSV_PATH "/mnt/ssd/maoyy/code/Maoyy/home/virtualhome-master/src/virtualhome/dataset/dataset_noise_12/dataset_noise_12.csv" \
  TRAIN.ENABLE False\
  DATA.PATH_LABEL_SEPARATOR "," \
  TRAIN.EVAL_PERIOD 5 \
  TRAIN.CHECKPOINT_PERIOD 10 \
  TRAIN.BATCH_SIZE 1 \
  MODEL.NUM_CLASSES 10 \
  NUM_GPUS 1 \
  UNIFORMER.DROP_DEPTH_RATE 0.1 \
  SOLVER.MAX_EPOCH 100 \
  SOLVER.BASE_LR 0.0004 \
  SOLVER.WARMUP_EPOCHS 10.0 \
  DATA.TEST_CROP_SIZE 256 \
  TEST.NUM_ENSEMBLE_VIEWS 1 \
  TEST.NUM_SPATIAL_CROPS 1 \
  RNG_SEED 6666 \
  OUTPUT_DIR $work_path/..noise_all \
  


  
  # DATA.SPECIE_CSV_PATH /mnt/ssd/maoyy/code/Maoyy/home/virtualhome-master/src/virtualhome/dataset/dataset_handler \

#  nohup sh test.sh > output 2>&1 &
# [1] 3178