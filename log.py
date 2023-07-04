from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
import cv2
from PIL import Image
class Logger:
    def __init__(self, log_dir, start=0):
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=log_dir)
        self.global_step = start

    def add_scalars(self, scalar_dict, global_step=None):
        if global_step is None:
            global_step = self.global_step
        for key, value in scalar_dict.items():
            # print('{} : {}'.format(key, value))
            self.writer.add_scalar('{}'.format(key), value, global_step)
        self.writer.flush()
    
    def step(self):
        self.global_step += 1
    
    def log_video(self, img_arr):
        img_arr = np.array(img_arr)
        video_path = os.path.join(self.log_dir, 'video_{}.mp4'.format(self.global_step))
        print(f"output video path: {video_path}")
        h, w= img_arr.shape[1:3]
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
        for t in range(img_arr.shape[0]):
            out.write(np.array(img_arr[t]))
        out.release()