#!/usr/bin/python3

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--epochs", type=int, default=30, help="number of epochs")
  parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
  parser.add_argument("--batch_size", type=int, default=16, help="size of each image batch")
  parser.add_argument("--model_config_path", type=str, default="config/yolov3.cfg", help="path to model config file")
  parser.add_argument("--data_config_path", type=str, default="config/coco.data", help="path to data config file")
  parser.add_argument("--weights_path", type=str, default="weights/yolov3.pt", help="path to weights file")
  parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
  parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
  parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
  parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
  parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
  parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
  parser.add_argument("--batch_checkpoint_interval", type=int, default=100000, help="interval between saving model weights")
  parser.add_argument(
      "--checkpoint_dir", type=str, default="checkpoints", help="directory where model checkpoints are saved"
  )
  parser.add_argument("--use_cuda", type=bool, default=True, help="whether to use cuda if available")
  parser.add_argument("--use_transformer", type=bool, default=False, help="whether to use a transformer layer")
  opt = parser.parse_args()
  print(opt)

  data_config = parse_data_config


  cuda = torch.cuda.is_available() and opt.use_cuda

  os.makedirs("output", exist_ok=True)
  os.makedirs("checkpoints", exist_ok=True)

  classes = load_classes(opt.class_path)

  # Get data configuration
  data_config = parse_data_config(opt.data_config_path)
  train_path = data_config["train"]
  test_path = data_config["valid"]
  num_classes = int(data_config["classes"])

  # Initiate model
  model = PreproDarknet(opt.model_config_path)
  print(model)
  model.load_weights(opt.weights_path)

  if cuda:
      model = model.cuda()

  model.eval()

  val_dataset = PreproDataset(test_path, model = model)
  val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)
  train_dataloader = torch.utils.data.DataLoader(
    PreproDataset(train_path, model=model), batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu
  )

  i = 0
  for _ in tqdm(val_dataloader, desc="processing val data"):
    i += 1
  print(i)
  for _ in tqdm(train_dataloader, desc="processing train data"):
    i += 1
  print(i)

if __name__ == '__main__':
  main()
