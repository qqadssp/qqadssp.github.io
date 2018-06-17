---
layout: post
title:  "FSSD: Feature Fusion Single Shot Multibox Detector"
categories: ObjectDetection
tags:  ObjectDetection SSD FSSD
author: CQ
---

* content
{:toc}

**Intro:** arXiv 2017.12  

**Link:** [https://arxiv.org/abs/1712.00960](https://arxiv.org/abs/1712.00960)  

**Code:** [https://github.com/lzx1413/CAFFE_SSD/tree/fssd](https://github.com/lzx1413/CAFFE_SSD/tree/fssd)  




## 摘要：

　　SSD(Single Shot Multibox Detector)是最好的同时具有高准确度和快速的目标检测算法之一。然而SSD的feature pyramid detection method使它很难融合不同尺度的特征。本文中，我们提出FSSD(Feature Fusion Single Shot Multibox Detector), 用一个全新且轻量的特征融合模块强化的SSD，这个模块可以在SSD上显著的改进性能，只有一点点速度损失。在特征融合模块中，来自不同层的不同尺度特征连接在一起，后跟一些降采样模块得到新的feature pyramid，新的feature pyramid送往multibox detectors预测最后的检测结果。在PASCAL VOC 2007　test上，使用单个Nvidia 1080Ti，输入尺寸300×300，我们的网络可以在65.8FPS(frame per second)的速度下得到82.7mAP(mean average precision)得分。另外，在COCO上的结果也以较大的差距好于传统SSD。我们的FSSD同时在准确度和速度上超过很多最新水平目标检测算法。。代码链接: [https://github.com/lzx1413/CAFFE_SSD/tree/fssd](https://github.com/lzx1413/CAFFE_SSD/tree/fssd)。  

## 1.简介

　　目标检测是计算机视觉的核心任务之一。近几年来，很多基于卷积网络的检测器被提出，用来改善目标检测任务的准确度和速度。但目标检测中的尺度变化对所有检测器来说仍然是决定性的挑战。如图1，有一些方法被提出用来解决多尺度目标检测问题。图1a将卷积网络用于不同尺度图像生成不同尺度的feature maps，这是相当低效的方式。图1b只选择一个尺度的feature map，但生成不同尺度的anchors来检测不同尺度的目标。Faster RCNN, RFCN等等算法采取这种方法。但固定的感受域尺寸对于检测太大或太小的目标是一个限制。如图1c的top-down结构最近很流行，且已经证明在FPN、DSSD和SharpMask中表现很好。但逐个层融合特征不够高效，因为有很多层需要联合在一起。
　　基于卷积网络的目标检测器的主要权衡是目标识别和定位之间的矛盾。更深的卷积网络，feature maps可以代表更多的平移不变的语义信息，这有益于目标识别，但不利于目标定位。为了解决这个问题，SSD采用feature pyramid来检测不同尺度的目标。对于用VGG16作为主干的网络，特征步长8的Conv4_3用来检测小目标，特征步长64的Conv8_2用来检测大目标。这个策略是合理的，因为小目标在浅层不会丢失太多位置信息，大目标也可以在深层被很好的定位和识别。问题是，浅层网络产生的小目标的特征缺失足够的语义信息，这将导致小目标检测上的不佳表现。另外，小目标也严重依赖背景信息。图6第一行显示了一些SSD的对小目标的检测缺失。本文中，为了处理前述提及的问题，我们提出Feature Fusion SSD(FSSD)，在传统SSD上增加一个轻量且高效的特征融合模块。我们首先定义特征融合模块的框架，摘录在目标检测上特征融合性能的有决定效果的因素。根据第3节和第4节中的理论分析和实验结果，特征融合模块的架构定义如下：来自不同层的不同尺度的Feature被投影然后串联在一起，后跟一个Batch Normalization层归一化feature values。然后我们增加一些降采样模块生成新的feature pyramid，送给multibox detectors产生最后的检测结果。
使用上述架构，我们的FSSD相比传统SSD改进很大的性能，只有一点速度开销。我们在PASCAL VOC数据集和MS COCO数据集上评测FSSD。结果显示FSSD可以以很大幅度改进传统SSD，尤其对于小目标，不用任何附加说明。另外，基于VGG网络FSSD也超过很多最新水平检测器，包括ION和Faster RCNN。特征融合模块在目标检测任务中也比FPN效果好。我们的主要贡献总结如下：
(1) 我们定义特征融合框架并研究不同的因素来确定特征融合模块。
(2) 我们引入一个全新的轻量的联合来自不同层feature maps的方法。
(3) 在大量高质量实验后，我们证明FSSD对传统SSD有显著的改善，只有一点点速度降低。FSSD可以在PASCAL VOC数据集和MS COCO数据集上获得最新水平表现。

![](/assets/FSSD_Feature_Fusion_Single_Shot_Multibox_Detector/Figure_1.png)
图1. (a)特征由不同尺度图片单独计算，这是低效的方式。(b)只有一个尺寸特征用来检测物体，用在一些两阶段检测器如Faster R-CNN和R-FCN中。(3)[19, 25]采用的特征融合方法，特征从上到下一层一层融合。(4)使用卷积网络生成的feature pyramid，传统SSD是例子之一。(e)我们提出的特征融合和生成feature pyramid方法。来自不同层不同尺寸的特征首先串联在一起，然后用来生成一系列pyramid feature层。
## 2.相关工作

**基于卷积网络的目标检测器：** 得益于深度卷积网络的力量，目标检测器如OverFeat和R-CNN开始显示出令人瞩目的准确度改进。OverFeat在image pyramid上滑动窗口中，将卷积网络用作为特征提取器。R-CNN使用selective search或Edge boxes产生的region proposals，通过预训练的卷积网络生成regin-based feature，采用SVM进行分类。SPPNet采用spatial pyramid pooling layer，允许分类模块重新利用卷积特征而不管输入图像分辨率。Fast R-CNN用分类和位置回归损失end to end的训练卷积网络。Faster R-CNN将selective search替换为regin proposal network(RPN)。RPN从来产生候选边界框(anchor boxes)同时滤除背景区域。然后另一个小网络用来分类和边界框回归，基于这些候选区。R-FCN用position sensitive ROI pooling(PSROI)替换Faster RCNN中的ROI poolling，来改进检测器准确度和速度两方面的质量。最近，Deformable Convolution Netword提出可变性卷积和可变形PSROI来进一步加强FRCN得到更高的准确度。
除了region based检测器，也有一些高效的单阶段检测器。YOLO(you only look once)将输入图像分科为若干格子，在图像每个部分执行定位和分类。得益于这个方法，YOLO可以以非常高的速度进行目标检测，但准确度不令人满意。YOLOv2是YOLO的加强版，通过去掉全连接层并采用类似RPN的anchor boxes改进YOLO。
SSD是另一个高效的单阶段检测器。如图2a所示，SSD通过两个3×3卷积层预测类别得分和默认边界框的位置平移。为了检测不同尺度的目标，SSD增加了一系列逐渐变小的卷积层生成pyramid feature maps，并根据层的感受域尺寸设置相应的anchor尺寸。然后NMS(non-maximun suppression)用来后处理最后检测结果。由于SSD直接从原始的卷积网络feature maps检测目标，它可以达到实时目标检测，比大多数最新水平的目标检测器进行的更快。
　　为了改进准确度，DSSD建议为SSD+ResNet-101增加卷积转置层，来引入额外的大尺度背景环境。然而，速度会由于模型复杂度变慢。RSSD使用池化和串联的交叠串联来充分使用feature pyramid中层与层间的关系，以小的速度损失增强准确度。DSOD研究如何从零开始训练目标检测器，并用DenseNet架构高效的改进参数。
在卷积网络中使用特征融合的算法：有很多方法，尝试使用多层特征改进计算机视觉任务的性能。HyperNet、Parsenet和ION在预测结果前串联来自多层的特征。FCN、U-Net、Stacked Hourglass networks也使用跳跃连接将底层和高层feature maps进行关联，充分利用综合信息。SharpeMask和FPN引入top-down结构将不同层特征联合在一起改进性能。  

## 3.方法

　　卷积网络有出色的能力提取pyramidal feature hierarchy，这有更多的从低层到高层的语义信息。传统SSD将这些不同层的特征视为相同层，然后直接从他们生成目标检测结果。这个策略是SSD缺乏捕获局部细节特征和全局语义特征的能力。然而，检测器应该融合背景环境信息和目标的细节特征，来确定小目标。所以对于卷积网络目标检测器改善准确度，综合少量结构的特征是重要的解决方案。

### 3.1 特征融合模块

　　如第2节所述，已经有很多算法尝试观察并充分利用pyramidal features。最常见的方法类似图1c。这种类型的特征融合用于FPN和DSSD中，已被证明对卷积检测器改进很多。但这需要多个特征合并过程。如图1c所示，右侧的新特征只能融合来自左侧对应层特征和较高层的特征。另外，隐藏特征和多特征元素间操作也耗费了大量时间。我们提出一个轻量且高效的特征融合模块来解决这个任务。我们的动机是以合适的方式一次融合不同层特征，然后从融合的特征中生成feature pyramid。当考虑特征融合模块时有几个因素需要考虑。我们将在下节研究他们。设$X_i, i\in C$是我们想要融合的源feature maps，特征融合模块可以如下描述：

$$
X_f=\phi_f\lbrace\Gamma_i(X_i)\rbrace i\in C \\
X_p^'=\phi_p(X_f) p\in P \\
loc,loss=\phi_{c,l}(\cup \lbrace X_p^' \rbrace)
$$

此处$\Gamma_i$意为在串联在一起前每一个源feature map的变换函数。$\phi_f$为特征融合函数。$\phi_p$是生成pyramid features的函数。$\phi_{c,l}$是从提供的feature maps预测目标检测的方法。我们将注意力集中在是否应该被融合的层($C$)的范围、如何融合选定的feature maps($\Gamma$和$\phi_f$)，以及如何生成pyramid features($\phi_p$)。
$C$：在传统的基于VGG16的SSD300中，其作者选择VGG16的conv4_3、fc_7，并新增conv6_2、conv7_2、conv8_2、conv9_2来产生特征进行目标检测。相应的特征尺寸为38×38，19×19，10×10，5×5，3×3和1×1。我们认为空间尺寸小于10×10的feature maps几乎没有信息进行合并，所以我们将层的范围设定为conv3_3，conv4_3，fc_7，和conv7_2(我们将conv6_2的步长设为1，所以conv7_2的feature map尺寸为10×10)。根据4.1.1节的分析，conv3_3没有带来收益，所以我们并不融合这一层。
$\phi_f$：有两种主要的方法将不同的feature maps合并在一起：串联和元素求和。元素求和需要feature maps应该有相同的尺寸，意味着我们不得不将feature maps转换到相同的通道。由于这个要求限制了feature maps融合的灵活性，我们倾向于使用串联。另外，根据4.1.2节的结果，串联比元素求和能得到更好的结果。所以我们使用串联来组合特征。
$\Gamma$：为了以一种简单高效的方式串联不同尺度的特征，我们采用如下策略。首先1×1卷积层作用在每个源层上来降低特征维度。然后，我们设定conv4_3的feature map尺寸作为基准feature map尺寸，也就意味着最小特征步长为8。conv3_3产生的feature maps被降采样到38×38，通过2×2步长2的最大值池化层。对于尺寸小于38×38的feature maps，我们使用双线性插值调整feature maps尺寸到与conv4_3相同。通过这中方式，所有特征在空间维度都有相同的尺寸。
$\phi_p$：采用自传统SSD，我们使用pyramid feature map生成目标检测结果。我们测试三种不同结构，比较结果选取最好的那个。根据4.1.4节的结果，我们选择了一个由一些简单的提取feature pyramid组件组成的结构。

![](/assets/FSSD_Feature_Fusion_Single_Shot_Multibox_Detector/Figure_2.png)
图2. (a)是[21]中提出的SSD框架，(b)是我们的F-SSD框架
![](/assets/FSSD_Feature_Fusion_Single_Shot_Multibox_Detector/Figure_3.png)
图3. FSSD300的pyramid feature生成器。我们用灰色方块的feature maps检测目标。在(a)中，融合的feature maps参与目标检测。在(b)中，我们只在融合feature map后的feature maps上检测目标。(3)我们用包含两个Conv+Relu层的bottleneck模块替换Conv+Relu的简单组合。
![](/assets/FSSD_Feature_Fusion_Single_Shot_Multibox_Detector/Figure_4.png)
图4. 训练过程比较。竖轴表示在VOC2007 test集上计算的mAP，横轴代表迭代步。SSD意为从预训练VGG16模型开始以默认设置训练传统SSD模型。FSSD意为用预训练VGG16模型训练FSSD模型。FSSD的训练参数与SSD相同。FSSD+意为从预训练的SSD模型训练FSSD。FSSD+只优化60,000迭代步。所有模型在VOC07+12上训练。
![](/assets/FSSD_Feature_Fusion_Single_Shot_Multibox_Detector/Table_1.png)
表1. 使用不同pyramid feature生成结构时VOC2007 test上的mAP

### 3.2 训练

　　有两种主要的训练方法可以选择。首先，由于FSSD是基于SSD的，我们可以采用训练好的SSD模型作为我们的预训练模型。对于学习率，我们设置新的特征融合模块学习率比其他参数大两倍。另一种训练FSSD的方式是与训练传统SSD相同。根据表2中的实验(第2和5行)，这两种方式在最后结果上几乎没有区别，但从VGG16开始是训练比从SSD模型开始训练好一点点。但如图4显示，从SSD开始训练比从预训练VGG16模型开始训练首先的更快。另外，在相同的超参数下FSSD也比传统SSD收敛更快。为了在限定的时间内测试我们的算法，我们默认使用第一种方式训练FSSD。
　　训练目的也与SSD相同。我们使用中心编码格式编码边界框，与SSD有相同的匹配策略、hard negative mining策略和数据扩展。更多细节参见[21]。

## 4. 实验

　　为了公平对比FSSD和传统SSD，我们的实验全部基于VGG16，可以与SSD做类似的预处理。我们在PASCAL VOC 2007、PASCAL VOC 2012和MS COCO数据集上进行实验。在VOC 2007和VOC 2012中，如果预测边界框与真值的IOU大于0.5则为正确。我们采用mean average precision(mAP)作为评估检测性能的度量标准。对于MS COCO，我们上传结果到评估服务器得到性能分析。我们所有的实验都是基于SSD的Caffe版实现。  

### 4.1 PASCAL VOC 2007上的简化模型研究

　　本节中，我们研究一些特征融合模块的重要因素。我们在PASCAL VOC 2007上对比结果，输入尺寸300×300。这些实验中，模型在2007 trainval和2012 trainval(VOC07+12)组合数据集上进行训练，在VOC 2007 test数据集上进行测试。大部分结果总结如表2。

#### 4.1.1 融合层范围

  表2中，我们对比具有不同融合层的FSSD。尽管我们融合所有feature maps(conv3_3、conv4_3、fc_7和conv7_2)，VOC 2007 test(第2行)上的mAP得分为78.6%。有趣的是如果我们移除conv3_3，mAP得分增长到78.6%(第5行)，这意味着从conv3_3以特征步长4降采样的feature map对最后性能没有好处。但根据表2第4行，保留conv7_2会更好。

#### 4.1.2 特征融合：串联还是元素求和

　　表2中，使用串联融合特征可以得到78.6%的mAP(第2行)，而元素求和只得到76.3%(第7行)。结果显示串联以2.3分的大差距好于元素求和。

#### 4.1.3 是否归一化特征值

　　来自不同层的Feature maps通常有不同的值区间。在传统SSD中，原作者使用L2Normalization来缩放conv4_3的feature map。我们想要用一个简单高效的方式缩放feature maps，在执行串联后增加一个batch normalization层。表2结果(第2和6行)显示使用batch normalization层重调整feature maps能带来0.7%的mAP改进。

#### 4.1.4 pyramid feature提取器 

　　我们比较三个不同结构的pyramid feature提取器，如图3。有两种模块生成低分辨率feature maps，简单模块(一个conv3×3跟随一个relu)和bottleneck模块，先使用conv1×1降低特征维度，被传统SSD采用。图3a中结构是几个简单模块的组合，并使用融合后的特征预测目标。图3b与a相同，除了不直接使用融合后特征预测。图3c将b中简单模块替换为bottlenet模块。表1结果表明b结构以微小改进优于另外两个。

### 4.2 PASCAL VOC上的结果

**实验设置**　根据4.1节的模型简化研究，FSSD架构如下定义：对于输入尺寸300×300的FSSD，我们采用VGG16作为主干网络。所有原始feature用1×1卷积层转为256通道。来自fc_7和conv7_2的feature maps插值到38×38。转换后的feature maps串联在一起，后跟batch normalization层归一化特征值。然后一个一个的附加若干降采样模块(包括一个步长2的3×3卷积层和relu层)产生pyramid features。

**PASCAL　VOC 2007上的结果** 随着SSD，我们使用PASCAL VOC 2007 trainval和VOC 2012 trainval训练FSSD。我们在两个nvidia 1080ti GPU上训练FSSD300，batch size 32训练120,000迭代步。初始学习率设为0.001，在80,000、100,000和120,000步时分别除以10。效仿SSD的训练策略，权重衰减设为0.0005。我们采用动量0.9的SGD优化以ImageNet预训练的VGG16初始化的FSSD。为了使用COCO模型作为预训练模型，我们先训练80类别的COCO模型，其细节在4.3节 描述。与PASCAL VOC的20个类别一致的80类别模型的子集从COCO模型中提取出来作为预训练的VOC模型。VOC2007上的结果在表3中显示。FSSD300能达到78.8%的mAP，与传统SSD300相比改进1.1分。另外，FSSD300的结果也比DSSD321高，尽管DSSD321用ResNet-101作为主干网络，与VGG16相比性能更好。以COCO作为训练数据，FSSD300的性能更进一步增加到82.7%，超过DSOD 1%和SSD300 1.5%。FSSD512从SSD512的79.8%改进到80.9%，这也比RSSD512高一点点。DSSD512在准确度上比FSSD512好，但我们认为ResNet-101主干在这个进步中扮演了关键的角色。然而，FSSD512比DSSD快得多。

  很有趣的是DSOD以一种特殊的方式研究目标检测任务：从零开始训练目标检测器。传统VGG16的SSD没有预训练VGGNet模型只能达到69.6% mAP。我们也研究了FSSD是否能改进传统SSD。表3(第19行)中的结果显示FSSD从零开始训练也以大的3.1分的差距改进性能。尽管这个结果不如DSOD好，它可以看做FSSD仍然以VGGNet作为基准网络而DSOD为了更好的性能更关注与主干网络，DSOD也训练更多迭代步(大约是SSD的4倍)。

**PASCAL VOC2012上的结果** 我们用VOC 2012 trainval、VOC 2007 trainval和MS COCO进行训练，在VOC2012 test上测试。训练超参数与VOC2007上的实验相同，除了数据集。表4从PASCAL VOC2012排行榜上总结了一些最新水平检测器。以COCO训练的FSSD300能达到82.0%的mAP，比传统SSD(79.3%)高2.7分。另外，FSSD512能达到84.2%的mAP，超过传统SSD(82.2%)2分。截止提交时，FSSD512在VOC2012排行榜上在所有单阶段目标检测器中获得第一的位置。

![](/assets/FSSD_Feature_Fusion_Single_Shot_Multibox_Detector/Table_2.png)
表2. PASCAL VOC2007上简化模型实验的结果。BN意为在特征串联后增加batch normalization层。pre-trained VGG意为采用预训练VGG16初始化模型。pre-trained SSD意为FSSD从一个训练好的SSD模型开始优化。我们能融合的层选项包括conv3_3，conv4_3，fc7_3，conv7_2。fusion layers代表我们选择合并的层。mAP在VOC2007 test上测量。
![](/assets/FSSD_Feature_Fusion_Single_Shot_Multibox_Detector/Table_3.png)
表3. PASCAL VOC 2007 test检测结果。我们在这里列出的SSD结果是原作者在更多数据扩充的文章发布后更新的版本。SSD300*表示从原始VGGNet训练SSD300，原始VGGNet以DSOD测量。FSSD*也从原始VGGNet开始训练。FSSD们的速度在单块Nvidia 1080Ti GPU上测试。为了公平对比，我们也在单块Nvidia 1080Ti GPU上测试SSD的速度。
![](/assets/FSSD_Feature_Fusion_Single_Shot_Multibox_Detector/Table_4.png)
表4. PASCAL VOC 2012 test检测结果。07++12+COCO：07 trainval + 07 test + 12 trainval + MSCOCO。07++12+S+COCO：07++12加上分割标签和MSCOCO。结果链接为FSSD300(07++12+COCO)：http://host.robots.ox.ac.uk:8080/anonymous/YMAA3TZ.html；FSSD512(07++12+COCO)：http://host.robots.ox.ac.uk:8080/anonymous/LQXCQK.html。

**MS COCO上的结果** MS COCO有80目标类别。我们使用COCO2017挑战赛数据来准备数据集。训练集包含115,000图片，可以比得上原trainval135。对于训练FSSD300，学习率对于开始280,000迭代步设为0.001，在360,000步和400,000步时分别除以10。但如果从一个训练好的SSD模型开始训练FSSD300，只需要总共120,000步就可以使FSSD300收敛好。对于训练FSSD512，学习率对于开始280,000迭代步设为0.001，在320,000和360,000步时分别除以10。
  COCO测试结果在表5中显示。FSSD300在test-dev上获得27.1%，以大的差距高于SSD300(25.1%)。尽管如此，FSSD表现不如DSOD和DSSD，应该注记我们的模型是VGG16，并且FSSD相比其他VGGNet模型算法，如表5中的Faster RCNN和ION(第1和2行)，有最好的准确度。另外，FSSD512(31.8%)超过传统SSD(28.8%)3分。尽管FSSD512稍慢于DSSD513，应当注记FSSD在小目标的mAP仍高于DSSD513，证明特征融合模块比DSSD的FPN模块在小目标检测上更强力。

![](/assets/FSSD_Feature_Fusion_Single_Shot_Multibox_Detector/Table_5.png)
表5. MSCOCO test-dev 2015检测结果
![](/assets/FSSD_Feature_Fusion_Single_Shot_Multibox_Detector/Table_6.png)
表6. 不同轻量检测器在MSCOCO minival2014上的mAP。

### 4.4 含有特征融合模块的轻量目标检测器

  为了展示特征融合模块的泛化能力，我们基于MobileNet开发了Light FSSD。我们选择特征步长8、16和32的feature maps生成融合特征。为了在轻量检测器中减少参数，我们将特征提取器中的卷积层替换为[13]中提到的深度卷积模块。我们用与VGGNet检测器相同的训练策略训练Light FSSD。如表6显示，Light FSSD以1.1%的改进超过传统MobileNet SSD，同时有更少的参数。这个结果证明特征融合模块的灵活性，以及用于嵌入式设备的潜力。

### 4.5 速度

  推测速度在表3第5列显示。在单个1080Ti GPU上，FSSD在300×300输入图像时可以以65.8FPS的速度运行。为了公平对比，我们也测试了1080Ti GPU上SSD的速度。由于FSSD在SSD模型上增加了额外的层，FSSD消耗了额外25%的时间。但与DSSD和DSOD相比，我们的方法依然比他们快得多，同时相比SSD的改进是同水平的。图5中，很明显FSSD比大多数目标检测算法都快，同时有竞争力的准确度。

### 4.6 自SSD的性能提升

  FSSD主要在两方面性能比传统SSD好。首先，FSSD减少了重复的检测一个目标的多个部分或者合并多个目标为一个目标的概率。例如，如图6第1列，SSD将狗的头部也视为一个单独的狗，很明显这是错的，因为整个狗在这个图像中。然而FSSD可以一次检测整个狗。另外，图6第2列显示SSD检测一个大狗，事实上包含两只狗。但FSSD没有犯这个错误。一方面，小目标相比于大目标只能激活网络中较小的区域，定位信息很容易在检测过程中丢失。另一方面，小目标的识别更依赖于周围的背景环境由于SSD值只从浅层检测小目标，如conv4_3，代表域太小不能观察到更大的背景环境信息，导致SSD在小目标上的低性能。FSSD可以综合的观察所有目标，得益于特征融合模块。如图6第3列到第6列，FSSD相比SSD成功的检测更多的小目标

## 5. 结论及未来工作

　　本文中，我们提出FSSD，一个通过应用轻量高效的特征融合模块加强的SSD。首先，我们研究了将不同特征融合一起和生成pyramid feature maps的框架。实验结果显示来自不同层的feature maps可以通过串联在一起被充分利用。然后应用和一些步长2的卷积层融合feature map产生pyramid features。PASCAL VOC和MS COCO上的实验证明FSSD改进了传统SSD很多，在准确度和效率上超过一些其他最新水平目标检测器，无需附加。
  将来，值得用更强大的主干网络如ResNet和DenseNet加强FSSD，在MS COCO上得到更好的性能，在Mask RCNN中用我们的特征融合模块替代FPN也是一个有趣的研究领域。

![](/assets/FSSD_Feature_Fusion_Singel_Shot_Multibox_Detector/Figure_6.png)
图6. SSD vs FSSD。两个模型都用VOC07+12训练。上面行包含传统SSD检测结果，底部行来自FSSD300模型。得分0.5或更高的边界框被画出。展示出来效果较好。

## 参考文献：
