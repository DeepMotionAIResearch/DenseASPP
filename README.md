# DenseASPP for Semantic Segmentation in Street Scenes [pdf](http://openaccess.thecvf.com/content_cvpr_2018/papers/Yang_DenseASPP_for_Semantic_CVPR_2018_paper.pdf)

## Introduction

Semantic image segmentation is a basic street scene understanding task in autonomous driving, where each pixel in a high resolution image is categorized into a set of semantic labels. Unlike other scenarios, objects in autonomous driving scene exhibit very large scale changes, which poses great challenges for high-level feature representation in a sense that multi-scale information must be correctly encoded.

To remedy this problem, atrous convolution[2, 3] was introduced to generate features with larger receptive fields without sacrificing spatial resolution. Built upon atrous convolution, Atrous Spatial Pyramid Pooling (ASPP)[3] was proposed to concatenate multiple atrous-convolved features using different dilation rates into a final feature representation. Although ASPP is able to generate multi-scale features, we argue the feature resolution in the scale-axis is not dense enough for the autonomous driving scenario. To this end, we propose Densely connected Atrous Spatial Pyramid Pooling (DenseASPP), which connects a set of atrous convolutional layers in a dense way, such that it generates multi-scale features that not only cover a larger scale range, but also cover that scale range densely, without significantly increasing the model size. We evaluate DenseASPP on the street scene benchmark Cityscapes[4] and achieve state-of-the-art performance.

## Usage

### 1.  **Clone the repository:**<br />

```
git clone https://github.com/DeepMotionAIResearch/DenseASPP.git
```

### 2. **Download pretrained model:**<br/>
Put the model at the folder `weights`. We provide some checkpoints to run the code:

**DenseNet161 based model**: [GoogleDrive](https://drive.google.com/open?id=1kMKyboVGWlBxgYRYYnOXiA1mj_ufAXNJ)
     
**Mobilenet v2 based model**: Coming soon.

Performance of these checkpoints:

Checkpoint name                                                           | Multi-scale inference       |  Cityscapes mIOU (val)         |  Cityscapes mIOU (test)         | File Size
------------------------------------------------------------------------- | :-------------------------: | :----------------------------: | :----------------------------: |:-------: |
[DenseASPP161](https://drive.google.com/file/d/1sCr-OkMUayaHAijdQrzndKk2WW78MVZG/view?usp=sharing) | False <br> True    | 79.9%  <br> 80.6 %             |  -  <br> 79.5%  |  142.7 MB
[MobileNetDenseASPP](*)                                                   | False <br> True             |  74.5%  <br> 75.0 %            |  -  <br> -      | 10.2 MB

Please note that the performance of these checkpoints can be further improved by fine-tuning. Besides, these models were trained with **Pytorch 0.3.1**

### 3. **Inference**

First cd to your code root, then run:

```
 python demo.py  --model_name DenseASPP161 --model_path <your checkpoint path> --img_dir <your img directory>
```

### 4. **Evaluation the results**
Please cd to `./utils`, then run:

```
 python transfer.py
```

And eval the results with the official evaluation code of Cityscapes, which can be found at [there](https://github.com/mcordts/cityscapesScripts)

## References

1.  **DenseASPP for Semantic Segmentation in Street Scenes**<br />
    Maoke Yang, Kun Yu, Chi Zhang, Zhiwei Li, Kuiyuan Yang. <br />
    [link](http://openaccess.thecvf.com/content_cvpr_2018/papers/Yang_DenseASPP_for_Semantic_CVPR_2018_paper.pdf). In CVPR, 2018.

2.  **Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs**<br />
    Liang-Chieh Chen+, George Papandreou+, Iasonas Kokkinos, Kevin Murphy, Alan L. Yuille (+ equal
    contribution). <br />
    [link](https://arxiv.org/abs/1412.7062). In ICLR, 2015.

3.  **DeepLab: Semantic Image Segmentation with Deep Convolutional Nets,**
    **Atrous Convolution, and Fully Connected CRFs** <br />
    Liang-Chieh Chen+, George Papandreou+, Iasonas Kokkinos, Kevin Murphy, and Alan L Yuille (+ equal
    contribution). <br />
    [link](http://arxiv.org/abs/1606.00915). TPAMI 2017.

4. **The Cityscapes Dataset for Semantic Urban Scene Understanding**<br />
    Cordts, Marius, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, Bernt Schiele. <br />
    [link](https://www.cityscapes-dataset.com/). In CVPR, 2016.
