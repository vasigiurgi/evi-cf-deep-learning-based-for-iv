# -*- coding: utf-8 -*-
# Trains the cross-fusion evidential model (script version of run_train.ipynb)
# Augmentation: Individual images are randomly augmented, not at epoch end
#
# Usage (from the repo root):
#   uv run python train.py --epochs 2                  # quick smoke test
#   uv run python train.py --model cf_uni_ds --epochs 500

import argparse
import logging
import os
import random
from datetime import datetime, timedelta

import numpy as np
import tensorflow as tf
from tensorflow import keras

from utils import train_val_split, split_check, write_path, kittiroad
from model_arch import build_model_cf_uni_ds, build_model_lite_cf_ds

ROOT = os.path.dirname(os.path.abspath(__file__))
log = logging.getLogger("train")
MODELS = {"lite_cf_ds": build_model_lite_cf_ds.get_model,
          "cf_uni_ds": build_model_cf_uni_ds.get_model}


def parse_args():
    p = argparse.ArgumentParser(description="Train the cross-fusion evidential model")
    p.add_argument("--model", choices=MODELS, default="lite_cf_ds")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=0.0005, help="starting learning rate (polynomial decay to 0)")
    p.add_argument("--prototypes", type=int, default=6)
    p.add_argument("--classes", type=int, default=3, help="road, car, background for the semantic dataset")
    p.add_argument("--fold", type=int, default=1, choices=range(1, 11), help="which of the 10 CV folds to use")
    p.add_argument("--img-size", type=int, nargs=2, default=(384, 1248), metavar=("H", "W"),
                   help="network input size; images are zero-padded to it (KITTI fits in 384x1248)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--init-weights", help="checkpoint prefix to start from, e.g. output/checkpoint_cross_fusion")
    p.add_argument("--rgb-dir", default=os.path.join(ROOT, "image_2"))
    p.add_argument("--lidar-dir", default=os.path.join(ROOT, "lidar"))
    p.add_argument("--target-dir", default=os.path.join(ROOT, "semantic_rgb"))
    p.add_argument("--out-dir", default=os.path.join(ROOT, "output", datetime.now().strftime("run_%Y%m%d_%H%M%S")))
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(os.path.join(args.out_dir, "train.log"))])
    log.info("args: %s", vars(args))

    # Seed every RNG in play (the augmentation angle uses python's random)
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # Grow GPU memory on demand (needed on the rtx 2070); CPU-only also works
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    log.info("GPUs: %s", gpus or "none, training on CPU")

    img_size = tuple(args.img_size)
    model = MODELS[args.model](img_size, img_size, args.prototypes, args.classes)
    if args.init_weights:
        model.load_weights(args.init_weights)
    model.summary()

    # Prepare paths of input images and target segmentation masks
    data_split = train_val_split(args.rgb_dir, args.lidar_dir, args.target_dir)
    k = str(args.fold)
    train_rgb, train_velo, train_target = (data_split["train_cam_" + k], data_split["train_velo_" + k],
                                           data_split["train_target_" + k])
    val_rgb, val_velo, val_target = (data_split["val_cam_" + k], data_split["val_velo_" + k],
                                     data_split["val_target_" + k])

    train_cam = split_check(train_rgb)
    val_cam = split_check(val_rgb)
    assert train_cam == split_check(train_velo) == split_check(train_target), 'INVALID SPLIT: training splits not matching'
    assert val_cam == split_check(val_velo) == split_check(val_target), 'INVALID SPLIT: validation splits not matching'
    assert not any(item in train_cam for item in val_cam), 'INVALID SPLIT: validation split contains training split'
    log.info("%d training / %d validation frames", len(train_cam), len(val_cam))

    # write split paths to text files
    write_path(args.out_dir, train_rgb, 'train_input_img_paths_rgb.txt')
    write_path(args.out_dir, train_velo, 'train_input_img_paths_velo.txt')
    write_path(args.out_dir, train_target, 'train_target_img_paths.txt')
    write_path(args.out_dir, val_rgb, 'val_input_img_paths_rgb.txt')
    write_path(args.out_dir, val_velo, 'val_input_img_paths_velo.txt')
    write_path(args.out_dir, val_target, 'val_target_img_paths.txt')

    # Instantiate data Sequences for each split
    train_gen = kittiroad(args.batch_size, img_size, train_rgb, train_velo, train_target)
    val_gen = kittiroad(args.batch_size, img_size, val_rgb, val_velo, val_target, val=True)

    # Polynomial Decay over the whole run
    lr_schedule = keras.optimizers.schedules.PolynomialDecay(args.lr, len(train_gen) * args.epochs, 0, power=0.9)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
                  loss=keras.losses.MeanSquaredError(), metrics=["mse"])

    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(args.out_dir, 'checkpoint_cross_fusion'),
        save_weights_only=True, monitor='val_loss', mode='min', save_best_only=True)

    # Train the model
    st_time = datetime.now()
    history = model.fit(train_gen, epochs=args.epochs, validation_data=val_gen, callbacks=[checkpoint])
    elapsed = timedelta(seconds=(datetime.now() - st_time).total_seconds())

    # Save training miscellaneous results
    with open(os.path.join(args.out_dir, 'time_log.txt'), 'w') as f:
        f.write('Training time in H:M:S\n%s\n' % elapsed)
    np.savez_compressed(os.path.join(args.out_dir, 'history.npz'),
                        loss=history.history['loss'], val_loss=history.history['val_loss'],
                        accuracy=history.history['mse'], val_accuracy=history.history['val_mse'])
    log.info("Done in %s - results in %s", elapsed, args.out_dir)


if __name__ == "__main__":
    main()
