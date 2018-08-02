import os
import cv2
import torch
import numpy
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms
from torch.autograd import Variable
from collections import OrderedDict


IS_MULTISCALE = True
N_CLASS = 19
COLOR_MAP = [(128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153), (153, 153, 153),
             (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60),
             (255,  0,  0), (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32)]

inf_scales = [0.5, 0.75, 1.0, 1.25, 1.5, 1.8]
data_transforms = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize([0.290101, 0.328081, 0.286964],
                                                           [0.182954, 0.186566, 0.184475])])


class Inference(object):

    def __init__(self, model_name, model_path):
        self.seg_model = self.__init_model(model_name, model_path, is_local=False)

    def __init_model(self, model_name, model_path, is_local=False):
        if model_name == 'MobileNetDenseASPP':
            from cfgs.MobileNetDenseASPP import Model_CFG
            from models.MobileNetDenseASPP import DenseASPP
        elif model_name == 'DenseASPP121':
            from cfgs.DenseASPP121 import Model_CFG
            from models.DenseASPP import DenseASPP
        elif model_name == 'DenseASPP169':
            from cfgs.DenseASPP169 import Model_CFG
            from models.DenseASPP import DenseASPP
        elif model_name == 'DenseASPP201':
            from cfgs.DenseASPP201 import Model_CFG
            from models.DenseASPP import DenseASPP
        elif model_name == 'DenseASPP161':
            from cfgs.DenseASPP161 import Model_CFG
            from models.DenseASPP import DenseASPP
        else:
            from cfgs.DenseASPP161 import Model_CFG
            from models.DenseASPP import DenseASPP

        seg_model = DenseASPP(Model_CFG, n_class=N_CLASS, output_stride=8)
        self.__load_weight(seg_model, model_path, is_local=is_local)
        seg_model.eval()
        seg_model = seg_model.cuda()

        return seg_model

    def folder_inference(self, img_dir, is_multiscale=True):
        folders = sorted(os.listdir(img_dir))
        for f in folders:
            read_path = os.path.join(img_dir, f)
            names = sorted(os.listdir(read_path))
            for n in names:
                if not n.endswith(".png"):
                    continue
                print(n)
                read_name = os.path.join(read_path, n)
                img = Image.open(read_name)
                if is_multiscale:
                    pre = self.multiscale_inference(img)
                else:
                    pre = self.single_inference(img)
                mask = self.__pre_to_img(pre)
                cv2.imshow('DenseASPP', mask)
                cv2.waitKey(0)

    def multiscale_inference(self, test_img):
        h, w = test_img.size
        pre = []
        for scale in inf_scales:
            img_scaled = test_img.resize((int(h * scale), int(w * scale)), Image.CUBIC)
            pre_scaled = self.single_inference(img_scaled, is_flip=False)
            pre.append(pre_scaled)

            img_scaled = img_scaled.transpose(Image.FLIP_LEFT_RIGHT)
            pre_scaled = self.single_inference(img_scaled, is_flip=True)
            pre.append(pre_scaled)

        pre_final = self.__fushion_avg(pre)

        return pre_final

    def single_inference(self, test_img, is_flip=False):
        image = Variable(data_transforms(test_img).unsqueeze(0).cuda(), volatile=True)
        pre = self.seg_model.forward(image)

        if pre.size()[0] < 1024:
            pre = F.upsample(pre, size=(1024, 2048), mode='bilinear')

        pre = F.log_softmax(pre, dim=1)
        pre = pre.data.cpu().numpy()

        if is_flip:
            tem = pre[0]
            tem = tem.transpose(1, 2, 0)
            tem = numpy.fliplr(tem)
            tem = tem.transpose(2, 0, 1)
            pre[0] = tem

        return pre

    @staticmethod
    def __fushion_avg(pre):
        pre_final = 0
        for pre_scaled in pre:
            pre_final = pre_final + pre_scaled
        pre_final = pre_final / len(pre)
        return pre_final

    @staticmethod
    def __load_weight(seg_model, model_path, is_local=True):
        print("loading pre-trained weight")
        weight = torch.load(model_path, map_location=lambda storage, loc: storage)

        if is_local:
            seg_model.load_state_dict(weight)
        else:
            new_state_dict = OrderedDict()
            for k, v in weight.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            seg_model.load_state_dict(new_state_dict)

    @staticmethod
    def __pre_to_img(pre):
        result = pre.argmax(axis=1)[0]
        row, col = result.shape
        dst = numpy.zeros((row, col, 3), dtype=numpy.uint8)
        for i in range(N_CLASS):
            dst[result == i] = COLOR_MAP[i]
        dst = numpy.array(dst, dtype=numpy.uint8)
        dst = cv2.cvtColor(dst, cv2.COLOR_RGB2BGR)
        return dst
