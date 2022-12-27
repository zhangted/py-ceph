from skimage import io, transform
import pathlib
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import csv

import utils
from Landmarks import Landmarks
from ConsoleMsg import ConsoleMsg
from ModelWrapper import ModelWrapper

class CephImage:
  def __init__(self, img_path):
    self.filename = img_path
    self.image = io.imread(img_path)
    self.landmarks = [None for _ in range(19)]

  def process(self, model, config):
    new_h, new_w = config.image_scale

    image = transform.resize(self.image, (new_h, new_w), mode='constant')
    torched_image = image.transpose((2, 0, 1))
    torched_image = torch.from_numpy(torched_image).float()
    if config.use_gpu: torched_image = torched_image.to(config.use_gpu)

    with torch.no_grad():
      heatmaps = model(torched_image.unsqueeze(0))
      raw_predicted_landmarks = utils.regression_voting(heatmaps, config.R2)
      if config.use_gpu: raw_predicted_landmarks = raw_predicted_landmarks.to(config.use_gpu)
      raw_predicted_landmarks = raw_predicted_landmarks.cpu()

      for idx, [y, x] in enumerate(raw_predicted_landmarks[0]):
        y_ = int(y * new_h)
        x_ = int(x * new_w)
        self.landmarks[idx] = (x_, y_)

      self.image = image

  def color_surrounding_from_pixel(self, image, y, x, color=[0, 1, 1], levels=2):
    if levels == 0: return
    for a, b in ([0,1],[0,-1],[-1,0],[1,0]):
      y2 = y+b
      x2 = x+a
      if 0 <= y2 < len(image) and 0 <= x2 < len(image[0]):
        image[y2][x2] = color
        self.color_surrounding_from_pixel(image, y2, x2, [0, 0.1, 0.15], levels-1)

  def print_landmarks_and_mark_on_image(self):
    print(f"{self.filename} Landmarks")
    for idx, (x, y) in enumerate(self.landmarks):
      print(Landmarks(idx), [x, y])
      self.image[y][x] = [0, .45, .8]
      self.color_surrounding_from_pixel(self.image, y, x)
    print("\n\n\n")

  def save_landmarks_to_jpg_and_csv(self):
    self.print_landmarks_and_mark_on_image()

    uint8_image = (self.image * 255).round().astype(np.uint8)
    #https://stackoverflow.com/questions/26918390/python-make-rgb-image-from-3-float32-numpy-arrays

    #save_image
    new_image_file_name = os.path.basename(f"{self.filename}_predicted.jpg")
    image_folder = os.path.abspath('\\'.join(self.filename.split('\\')[:-1]))
    predicted_images_folder = f"{image_folder}\\predicted_images"
    if not os.path.exists(predicted_images_folder): os.mkdir(predicted_images_folder)
    new_image_path = f"{predicted_images_folder}\\{new_image_file_name}"
    io.imsave(new_image_path, uint8_image, quality=100)

    #save landmark to csv
    csv_file = f"{predicted_images_folder}\\landmarks.csv"
    mode = 'a' if os.path.exists(csv_file) else 'w'
    with open(csv_file, mode, newline='') as f:
      writer = csv.writer(f)

      #write labels
      if mode == 'w':
        arr = [Landmarks(i) for i in range(19)]
        arr.insert(0, 'Filename')
        writer.writerow(arr)

      #write landmark points for 1 image
      arr = [new_image_path]
      for (x, y) in self.landmarks:
        arr.append(f"{x},{y}")
      writer.writerow(arr)

  def show_interactive_landmarks(self):
    self.print_landmarks_and_mark_on_image()
    plt.ion()
    io.imshow(self.image)
    ConsoleMsg.print_terminate()

class CephImageBatch:
  def __init__(self, img_folder=None, img_path=None):
    self.VALID_IMAGE_TYPES = set(['.png', '.jpg', 'jpeg', '.bmp'])
    self.batch = []

    if not img_path:
      for img_path in os.listdir(img_folder):
        self.setup_ceph_image(f"{img_folder}/{img_path}")
    else:
      self.setup_ceph_image(img_path)
  
  def setup_ceph_image(self, image_path):
    file_type = pathlib.Path(image_path).suffix
    if file_type in self.VALID_IMAGE_TYPES:
      abs_image_path = os.path.abspath(image_path)
      self.batch.append(CephImage(abs_image_path))

  def process(self, config):
    modelWrapper = ModelWrapper(config)
    model = modelWrapper.load_model()

    for ceph_img in self.batch:
      ceph_img.process(model, config)
      if len(self.batch) == 1:
        ceph_img.show_interactive_landmarks()
      else:
        ceph_img.save_landmarks_to_jpg_and_csv()