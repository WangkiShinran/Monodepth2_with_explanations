# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import random
import numpy as np
import copy
from PIL import Image  # using pillow-simd for increased speed

import torch
import torch.utils.data as data
from torchvision import transforms


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class MonoDataset(data.Dataset):  # 继承自pytorch内置Dataset抽象类，可以在trainer.py中直接送入dataloader
    """Superclass for monocular dataloaders

    Args:
        data_path
        filenames
        height
        width
        frame_idxs
        num_scales
        is_train
        img_ext
    """
    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 is_train=False,
                 img_ext='.jpg'):
        super(MonoDataset, self).__init__()  # 在trainer.py第129行被构造

        self.data_path = data_path
        self.filenames = filenames  # 在trainer.py第130行被构造，是由train_files.txt各行组成的列表，用于子类SDDataset里
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.interp = Image.ANTIALIAS

        self.frame_idxs = frame_idxs

        self.is_train = is_train
        self.img_ext = img_ext

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()  # 将图像转化成tensor的同时归一化到[0,1]，便于送入网络

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)  # 数据增强
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)  # 设定640*192尺寸上缩放

        self.load_depth = self.check_depth()

    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            frame = inputs[k]
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])  # 见82行，resize是个字典
                    # self.resize[i]()返回85行的transforms.Resize函数，实现对input中图像的缩放

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}  # 字典

        # 随机做训练数据颜色增强预处理
        do_color_aug = self.is_train and random.random() > 0.5
        # 随机做训练数据水平左右flip预处理
        do_flip = self.is_train and random.random() > 0.5
        # index是train.txt或val.txt中的第index行
        line = self.filenames[index].split()
        folder = line[0]  # train_files.txt中一行数据的第一部分，即图片所在目录

        '''
        实际line输出示例
        ['./images/env04/c1/img_00794_c1_1287503878959663us.jpg'] 验证集图片名称
        ['./slice6/env02/c1/images/img_02324_c1_1284563452201866us.jpg'] 训练集图片名称
        '''
        # 示例：line = "2011_09_26/2011_09_26_drive_0022_sync 473 r".split()
        # 针对上述Kitty数据集，每一行一般都为3个部分，第二个部分是图片的frame_index
        if len(line) == 3:
            frame_index = int(line[1])
        else:
            frame_index = 0

        # side为l或r，表明该图片是左或右摄像头所拍
        if len(line) == 3:
            side = line[2]
        else:
            side = None

        # 在stereo训练时， frame_idxs为["0", "s"]
        # 通过这个for循环，inputs[("color", "0", -1)]和inputs[("color", "s", -1)]
        # 分别获得了frame_index和它对应的另外一个摄像头拍的图片数据
        for i in self.frame_idxs:
            if i == "s":
                other_side = {"r": "l", "l": "r"}[side]
                inputs[("color", i, -1)] = self.get_color(folder, frame_index, other_side, do_flip)
            else:
                # 单目训练找相邻帧，读取原始图片,单目训练frame_idxs已经默认为[0, -1, 1]，txt文件已排序
                # 防止访问相邻帧时数组越界，下面用当前帧代替是因为auto_masking机制会自动滤掉两张相同的相邻帧，也就是相机静止的情况
                i0 = i  # i为本来应该取的index，i0为实际取的index
                if i == -1 and index == 0:  # 如果index为0，那么本该取的前一帧用当前帧代替
                    i0 = 0
                    print("start")  # 当前图片路径在txt里是第一行
                elif i == 1 and index == len(self.filenames) - 1:  # 如果index为到最后了，那么本该取的后一帧用当前帧代替
                    i0 = 0
                    print("end")  # 当前图片路径在txt里是最后一行
                else:
                    line1 = self.filenames[index + i].split()
                    folder1 = line1[0]  # 前一帧文件的相对路径，比如'./images/env04/c1/img_00794_c1_1287503878959663us.jpg'和'./slice6/env02/c1/images/img_02324_c1_1284563452201866us.jpg'
                    folder_changed = ( folder1[7] != folder[7] or folder1[12] != folder[12] or folder1[13] != folder[13] or folder1[16] != folder[16] )
                    if folder_changed:  # 如果文件路径改动了（对应的不是同一段视频帧），那么本该取的前一帧用当前帧代替
                        i0 = 0
                        print("folder_changed")

                line = self.filenames[index + i0].split()
                folder = line[0]
                inputs[("color", i, -1)] = self.get_color(folder, frame_index + i, side, do_flip)  # 157行读到的frame_index

        # adjusting intrinsics to match each scale in the pyramid
        # 因为模型有4个尺度，所以对应4个相机内参，这里根据图片名称区别是使用SeasonDepth的c0还是c1拍的
        for scale in range(self.num_scales):
            # K = self.K.copy() 不再从子类中读取内参矩阵，而是根据文件名来判断到底是哪个相机拍出的照片
            if line[0][16] == '0':  # c0相机
                K = np.array([[0.848626, 0, 0.513616, 0],
                           [0, 1.127686, 0.546930, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
                # print('-----------------c0------------')
            elif line[0][16] == '1':  # c1相机
                K = np.array([[0.852913, 0, 0.516918, 0],
                           [0, 1.141262, 0.517282, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
                # print('-----------------c1------------')
            K[0, :] *= self.width // (2 ** scale)  # 内参矩阵缩放
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)  # change array into tensor using the same space to store
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        # 颜色增强参数设定，由142行随机值，它有0.5的概率被设置为True
        if do_color_aug:
            color_aug = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)
        # 训练前数据预处理以及对输入数据做多尺度resize
        self.preprocess(inputs, color_aug)
        # 经过preprocess，产生了inputs[("color"，"0", 0 / 1 / 23)]和inputs[("color_aug"，"0",0 / 1 / 23)]。
        # 所以可以将原始的inputs[("color", i, -1)]和[("color_aug", i, -1)]释放（在194行读入）
        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]
        # load_depth为False，因为不需要GT label数据。但此时也读入了
        if self.load_depth:
            depth_gt = self.get_depth(folder, frame_index, side, do_flip)
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        if "s" in self.frame_idxs:  # 在stereo训练时，还需要构造双目姿态的平移矩阵参数inputs["stereo_T"]
            stereo_T = np.eye(4, dtype=np.float32)
            baseline_sign = -1 if do_flip else 1
            side_sign = -1 if side == "l" else 1
            stereo_T[0, 3] = side_sign * baseline_sign * 0.1

            inputs["stereo_T"] = torch.from_numpy(stereo_T)

        return inputs
    # 以下三个函数都必须要留给子类去定义
    def get_color(self, folder, frame_index, side, do_flip):  # 加载RGB图片
        raise NotImplementedError

    def check_depth(self):  # 生成深度图路径
        raise NotImplementedError

    def get_depth(self, folder, frame_index, side, do_flip):  # 读取深度图
        raise NotImplementedError
