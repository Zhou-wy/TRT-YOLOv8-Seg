'''
description: 
version: 
Author: zwy
Date: 2023-10-07 13:26:50
LastEditors: zwy
LastEditTime: 2023-10-07 13:28:09
'''

import os
import random
import shutil

# 源文件夹和目标文件夹的路径
source_folder = '/home/zwy/PyWorkspace/eCharger/images'
destination_folder = '/home/zwy/PyWorkspace/eCharger/TRT_YOLOv8_Server/workspace/media'

# 选择要复制的图片数量
num_images_to_copy = 128

# 获取源文件夹中的所有图片文件
image_files = [f for f in os.listdir(source_folder) if f.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))]

# 如果源文件夹中没有足够的图片，可以选择减少num_images_to_copy或处理不足的情况

# 随机选择要复制的图片文件
selected_images = random.sample(image_files, num_images_to_copy)

# 将选定的图片复制到目标文件夹
for image in selected_images:
    source_path = os.path.join(source_folder, image)
    destination_path = os.path.join(destination_folder, image)
    shutil.copy(source_path, destination_path)
    print(f"复制 {image} 到 {destination_path}")

print(f"共复制了 {len(selected_images)} 张图片到目标文件夹 {destination_folder}")
