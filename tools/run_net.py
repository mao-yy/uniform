#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Wrapper to train and test a video classification model."""
from sklearn.metrics import confusion_matrix
from slowfast.config.defaults import assert_and_infer_cfg
from slowfast.utils.misc import launch_job
from slowfast.utils.parser import load_config, parse_args

from test_net import test
from train_net import train
import warnings
import os
warnings.filterwarnings("ignore")

# os.environ ["CUDA_VISIBLE_DEVICES"] = "0"
# import os
def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    cfg = load_config(args)
    cfg = assert_and_infer_cfg(cfg)
    output_path=cfg.OUTPUT_DIR
    os.system("cp ./train.sh {}/".format(output_path))
    os.system("cp ./train.yaml {}/".format(output_path))
    os.system("cp ./eval.yaml {}/".format(output_path))
    # Perform training.
    if cfg.TRAIN.ENABLE:
        launch_job(cfg=cfg, init_method=args.init_method, func=train)

    # Perform multi-clip testing.
    if cfg.TEST.ENABLE:
        launch_job(cfg=cfg, init_method=args.init_method, func=test)


if __name__ == "__main__":
    main()
