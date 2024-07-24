import json
import os.path


import config

import cv2
import einops
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler

class Test():
    '''test class-->test diffusion model'''
    def __init__(self):
        self.gpu_device_id = 2
        torch.cuda.set_device(self.gpu_device_id)
        self.model = create_model('../models/cldm_v15.yaml').cpu()
        self.model.load_state_dict(load_state_dict('/data/hdd/sunhaisen/infrared/ControlNet/ControlNet-main/lightning_logs/version_3/checkpoints/epoch=99-step=94500.ckpt', location='cuda:{}'.format(self.gpu_device_id)),
                              strict=False)
        self.model = self.model.cuda(self.gpu_device_id)
        self.ddim_sampler = DDIMSampler(self.model)  # use ddim sampler
        self.num_samples = 1  # sample number is 1    min is 1 and max is 12
        self.image_resolution = 256   # image resolution is 512    min is 256 and max is 768
        self.strength = 1.0       # control strength is 1    min is 0 and max is 2
        self.seed = 42            # seed
        self.eta = 0.0            # eta is 0
        # # if guess mode  is False
        self.guess_mode = False  # guess mode is
        self.a_prompt = "best quality, extremely detailed"
        self.n_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
        self.ddim_steps = 50  # ddim step is 50
        self.scale = 9.0  # min is 0.1 and max is 30

    def process_image(self,input_image_path, output_path, prompt):
        '''diffusion model test'''
        # 不更新梯度
        with torch.no_grad():
            img = cv2.imread(input_image_path)  # 从路径读取图像  img is numpy and (H,W,C)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 将图像颜色通道调整为 RGB
            img = resize_image(HWC3(img), self.image_resolution)  # resize image
            H, W, C = img.shape  # get image  height width and channel

            # detected_map = apply_canny(img, 100, 200)  # get original image---> canny image
            detected_map = HWC3(img)  # detected map size is same with img

            # process control image
            control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
            control = torch.stack([control for _ in range(self.num_samples)], dim=0)
            control = einops.rearrange(control, 'b h w c -> b c h w').clone()

            # if seed is -1
            if self.seed == -1:
                self.seed = random.randint(0, 65535)
            seed_everything(self.seed)  # neither select seed

            if config.save_memory:  # save_memory is False
                self.model.low_vram_shift(is_diffusing=False)
            #
            cond = {"c_concat": [control], "c_crossattn": [self.model.get_learned_conditioning([prompt + ', ' + self.a_prompt] * self.num_samples)]}
            un_cond = {"c_concat": None if self.guess_mode else [control], "c_crossattn": [self.model.get_learned_conditioning([self.n_prompt] * self.num_samples)]}
            shape = (4, H // 8, W // 8)

            if config.save_memory:
                self.model.low_vram_shift(is_diffusing=True)

            self.model.control_scales = [self.strength * (0.825 ** float(12 - i)) for i in range(13)] if self.guess_mode else ([self.strength] * 13)
            samples, intermediates = self.ddim_sampler.sample(self.ddim_steps, self.num_samples,
                                                         shape, cond, verbose=False, eta=self.eta,
                                                         unconditional_guidance_scale=self.scale,
                                                         unconditional_conditioning=un_cond)

            if config.save_memory:
                self.model.low_vram_shift(is_diffusing=False)

            x_samples = self.model.decode_first_stage(samples)
            x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

            results = [x_samples[i] for i in range(self.num_samples)]

            # Save the output image
            output_path = os.path.join(output_path,input_image_path.split('/')[-1])
            cv2.imwrite(output_path,x_samples[0] )  # 保存处理后的图像 x_samples[0]
            return output_path

    def generator_image(self,json_file_path,output_path):

        with open(json_file_path,'r') as file:
            lines =  file.readlines()

        for i in range(len(lines)):
            try:
                json_object = json.loads(lines[i])

                if 'prompt' in json_object:
                    prompt = json_object['prompt']
                    if 'VEDIA' in json_file_path:  # 这个地方需要修改
                        input_image_path = json_object['source']
                        output_image_path = json_object['target']
                    else:
                        input_image_path =  json_object['source']
                        output_image_path = json_object['target']


                    return self.process_image(input_image_path, output_path, prompt)
            except json.JSONDecodeError:
                continue


    def generator_image_from_control_net(self,json_file,output_path):

       return  self.generator_image(json_file,output_path)

