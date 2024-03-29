#!/usr/bin/env python3
"""
call this one for training and eval a model with a specified transfer type. (transfer type = {prmot, ....})
major actions here: fine-tune the features and evaluate different settings
"""
import os
import torch
import warnings

import numpy as np
import random

from time import sleep
from random import randint

import src.utils.logging as logging
from src.configs.config import get_cfg
from src.data import loader as data_loader
from src.engine.evaluator import Evaluator
from src.engine.trainer import Trainer
from src.models.build_model import build_model
from src.utils.file_io import PathManager

from launch import default_argument_parser, logging_train_setup
warnings.filterwarnings("ignore")


def setup(args):
    """
    Create configs and perform basic setups.
    I think directories are made in this function.
    - refer to this link to learn more about the cgf 
    - https://detectron2.readthedocs.io/en/latest/modules/config.html#detectron2.config.CfgNode
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # setup dist
    cfg.DIST_INIT_PATH = "tcp://{}:12399".format(os.environ["SLURMD_NODENAME"])

    # setup output dir
    # output_dir / data_name / feature_name / lr_wd / run1
    output_dir = cfg.OUTPUT_DIR
    lr = cfg.SOLVER.BASE_LR
    wd = cfg.SOLVER.WEIGHT_DECAY
    output_folder = os.path.join(
        cfg.DATA.NAME, cfg.DATA.FEATURE, f"lr{lr}_wd{wd}")

    # train cfg.RUN_N_TIMES times. default value is 5
    count = 1
    while count <= cfg.RUN_N_TIMES:
        output_path = os.path.join(output_dir, output_folder, f"run{count}")
        # pause for a random time, so concurrent process with same setting won't interfere with each other. # noqa
        sleep(randint(3, 30))
        if not PathManager.exists(output_path):
            PathManager.mkdirs(output_path)
            cfg.OUTPUT_DIR = output_path
            break
        else:
            count += 1
    if count > cfg.RUN_N_TIMES:
        raise ValueError(
            f"Already run {cfg.RUN_N_TIMES} times for {output_folder}, no need to run more")

    cfg.freeze()
    return cfg


def get_loaders(cfg, logger):
    logger.info("Loading training data (final training data for vtab)...")
    if cfg.DATA.NAME.startswith("vtab-"):
        train_loader = data_loader.construct_trainval_loader(cfg)
    else:
        train_loader = data_loader.construct_train_loader(cfg)

    logger.info("Loading validation data...")
    # not really needed for vtab
    val_loader = data_loader.construct_val_loader(cfg)
    logger.info("Loading test data...")
    if cfg.DATA.NO_TEST:
        logger.info("...no test data is constructed")
        test_loader = None
    else:
        test_loader = data_loader.construct_test_loader(cfg)
    return train_loader,  val_loader, test_loader


def train(cfg, args):
    """
    The following things happen in this function.
        1. clean up cuda cahch
        2. fix seed
        3. create a logger
        4. train, val, test loaders are created
        5. model is defined
        6. training is performed
        7. testing is performed
    """
    # clear up residual cache from previous runs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # fix the seed for reproducibility
    if cfg.SEED is not None:
        torch.manual_seed(cfg.SEED)
        np.random.seed(cfg.SEED)
        random.seed(0)

    # setup training env including loggers
    logging_train_setup(args, cfg)
    logger = logging.get_logger("visual_prompt") # a looger keeps track of the records that happens during the program execution

    train_loader, val_loader, test_loader = get_loaders(cfg, logger) # returns the 3 dataset loaders. the default batch size is 32
    logger.info("Constructing models...") # this log is not labeled as 'info'
    model, cur_device = build_model(cfg) # returns the model and the current device. E.g. (vit, cuda)

    logger.info("Setting up Evalutator...")
    evaluator = Evaluator() # some type of an evaluator that evaluates the classification results
    logger.info("Setting up Trainer...")
    trainer = Trainer(cfg, model, evaluator, cur_device)

    if train_loader:
        trainer.train_classifier(train_loader, val_loader, test_loader) # performs both training and testing part
    else:
        print("No train loader presented. Exit")

    if cfg.SOLVER.TOTAL_EPOCH == 0:
        trainer.eval_classifier(test_loader, "test", 0) # performs only the testing part


def main(args):
    """
    main function to call from workflow
    """

    # sets up the cgf based using the arguments
    cfg = setup(args)

    # Perform training.
    train(cfg, args)


if __name__ == '__main__':
    """
    this is where everything starts!
    - refer to link below to learn more about argument parser
    - https://engineer-mole.tistory.com/213
    """
    args = default_argument_parser().parse_args()
    main(args) # passes the arguments to the main function