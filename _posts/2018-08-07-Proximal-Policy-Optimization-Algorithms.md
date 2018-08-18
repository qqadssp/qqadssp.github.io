---
layout: post
title:  "Proximal Policy Optimization Algorithms"
categories: ReinforcementLearning
tags:  ReinforcementLearning, PolicyGradient
author: CQ
---

* content
{:toc}

**Intro:** arxiv 1707  

**Link:** [https://arxiv.org/abs/1707.06347](https://arxiv.org/abs/1707.06347)  

**Code:** [https://github.com/openai/baselines](https://github.com/openai/baselines)  




## 摘要：

　　我们提出一个新的强化学习策略梯度方法系列，在通过与环境交互采样数据和使用随机梯度上升优化替代目标函数之间交替。标准策略梯度方法每个数据采样执行一个梯度更新，我们提出一个新的目标函数能够进行多个minibatch计算的更新。新的方法，我们称为proximal policy optimization(PPO)，有一部分得益于trust region policy optimization(TRPO)，但它更容易实现，更通用，有更好的抽样复杂度(经验性的)。我们的实验在一组标准任务上测试PPO，包括模拟机器人运动和玩Atari游戏，我们展示PPO超过其他在线策略梯度方法，且总体上达到了抽样复杂度，简单性，运行时间之间的良好平衡。  

## 1.简介

　　最近几年，几个不同的方法被提出用于含有神经网络近似函数的强化学习。处于领导位置的是deep Q-learning[Min+15], 综合策略梯度方法[Min+16]，和信任区/自然策略梯度方法[Sch+15b]。然而，仍有发展一个可扩展(对于大模型和并行实现)的，数据高效的，且强壮的(即在成功用于不同问题且不用超参微调)方法的空间。(有近似函数的)Q-learining在许多简单问题上失效且理解不能，综合梯度方法数据低效且不强壮，trust region policy optimization(TRPO)相当复杂，且对于含有噪音(如dropout)或者共享参数(在策略和值函数之间，或与辅助任务)的架构不能计算。  

　　本文试图改进现在事情的状态，通过引入一个达到数据高效和TRPO可靠表现的算法，同时只使用第一阶优化。我们提出一个全新的含有修剪概率比率的目标，组成一个悲观(如下限)的策略表现评估。为了优化策略，我们在从策略中抽样和在抽样数据上执行几轮优化计算之间交替进行。  

　　我们的实验比较了大量不同版本的替代目标的性能，发现有修剪概率比率的版本表现最好。我们也将PPO与几个之前的文献中的算法进行比较。在连续控制任务上，它表现得比我们与之相比较的算法更好。在Atari上，他的表现比A2C明显更好(在抽样复杂度上)，与ACER相似，尽管它更简单。  

## 2.背景：策略优化

### 2.1 策略梯度方法

　　策略梯度方法通过计算一个策略梯度的评估量并且将其插入一个随机梯度上升算法中实现。最常用的梯度评估量有形式  

$$
\hat{g}=\hat{E}_t[\nabla_{\theta}\log\pi_{\theta}(a_t \mid s_t)\hat{A}_t]   
$$

　　此处$\pi_\theta$是一个随机策略，$\hat{A}_t$是一个在时间t的优势函数评估量。这里，期望$\hat{E}[...]$表示在一个有限抽样batch上的经验平均，在算法中其在抽样和优化之间交替进行。使用自动微分软件的实现，通过构造一个梯度是策略梯度评估量的目标函数进行，评估量$hat{g}$通过微分目标函数得到。  

$$
L^{PG}(\theta) = \hat{E}_t[\log\pi_{\theta}(a_t \mid s_t)\hat{A}_t]  
$$

　　由于它需要在这个损失$L^{PG}$上使用相同路径执行多步优化，这么做理由并不充分，经验上它经常导致破坏性的大的策略更新(见6.1，结果没有展示，但与'无修正或惩罚'设置下的结果相似或更差)。  

### 2.2 Trust Region Methods

　　在TRPO中，目标函数(代理目标)被最大化，在满足一个对策略更新尺度的约束下。特殊的  

$$
\underset{\theta}{maximize} \; \hat{E}_t \left[ \frac{\pi_{\theta}(a_t \mid s_t)}{\pi_{\theta_{old}}}\hat{A}_t \right] \\  
s.t. \; \hat{E}_t \left[ KL \left[ \pi_{\theta_{old}}(\cdot \mid s_t),\pi_(\theta)(\cdot \mid s_t) \right] \right] \leq \delta  
$$

　　此处，$\theta_{old}$是更新前的策略参数向量。这个问题可以使用共轭梯度法高效的近似求解，在对目标进行一个线性近似和对约束进行一个二次近似后。  
　　证明TRPO的理论实际上建议使用一个惩罚项而不是约束，即求解对某个系数$\beta$的无约束优化问题  

$$
\underset{\theta}{maximize} \; \hat{E}_t \left[ \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{old}}(a_t \mid s_t)}\hat{A}_t-\beta KL \left[ \pi_\theta(\cdot \mid s_t),(\cdot \mid s_t) \right] \right]  
$$

　　这遵循这样一个事实，一个确定的代理目标(在状态上计算最大KL值而不是平均值)在策略$\pi$的性能上组成一个下界(即一个悲观估计)。TRPO使用硬性强约束而不是惩罚项，因为选择单一$\beta$值在多个问题上表现良好是很困难的——甚至在单一问题中也是，特征在学习过程中变化。因此，为了达到我们模拟TRPO单调改进的一阶算法的目的，实验显示简单选择一个固定的惩罚系数$\beta$并用SGD优化惩罚目标方程(5)是不够的，需要额外的修改。  

## 3.修正的代理目标

　　令$r(\theta)$代表概率比率$r(\theta)=\frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{old}}(a_t \mid s_t)}$，所以$r(\theta_{old})=1$。TRPO最大化一个代理目标  

$$
L^{CPI}(\theta)=\hat{E}_t \left[ \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{old}}(a_t \mid s_t)}\hat{A}_t \right] =\hat{E}_t[r_t(\theta)\hat{A}_t]  
$$

　　上标CPI代表保守策略迭代[KL02]，文中提出这个目标。没有约束，最大化$L^{CPI}$将导致太大的策略更新，因此，我们现在考虑如何修改这个目标，惩罚将$r(\theta)$移动远离1的策略改变。  

　　我们提出的主体目标是：  

$$
L^{CLIP}=\hat{E}_t \left[ \min(r_t(\theta)\hat{A}_t, clip(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t) \right]  
$$

　　这里$\epsilon$是一个超参数，比如说，$\epsilon=0.2$。这个目标的动机如下。在$\min$中的第一项是$L^{CPI}$。第二项，$clip(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t$，通过修剪概率比率修改代理目标，移除移动$r_t$到区间$[1-\epsilon, 1+\epsilon]$外的激励。最后，我们取修剪目标和未修剪目标的最小值，所以最后的目标是一个未修剪目标的下界(即悲观的界限)。  

　　以这个方案，我们只忽略了将使目标改进的概率比率的变化，引入使目标变坏的概率比率。注记对于$\theta_{old}$(此处$r=1$)附近的第一阶，$L^{CLIP}=L^{CPI}$，但是，随着$\theta$移动离开$\theta_{old}$，他们变得不同。图1画出$L^{CLIP}$中的单独一项(即单一时间t)，注记概率比率$r$在$1-\epsilon$或$1+\epsilon$处被修剪，依赖于优势函数是正或负。  

　　图2提供另一个关于代理目标$L^{CLIP}$的直觉的来源。它展示了随着我们沿着策略更新方向插值时几个目标函数如何变化，通过在一个连续控制问题上的proximal policy optimization(算法我们将简短介绍)。我们可以看到$L^{CLIP}$是$L^{CPI}$的下界，对于有太大策略更新时有一个惩罚项。  

## 4. 自适应KL惩罚系数

　　另一个方法，可以用作修剪代理目标的替代或增加，是在KL散度上使用惩罚项，调整惩罚系数使我们在每个策略更新得到一些KL散度的目标值$d_{trag}$。我们的实验中，我们发现KL惩罚项表现比修剪代理目标差，但是，我们在这里引入它，因为他是一个重要的基准。  

　　在这个算法的最简单实现中，我们在每个策略更新中执行如下步骤：  

　　使用几个minibatch的SGD计算，优化KL惩罚目标  

$$
L^{KLPEN} = \hat{E}_t \left[ \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{old}}(a_t \mid s_t)}\hat{A}_t-\beta KL \left[ \pi_{\theta_{old}}(\cdot \mid s_t), \pi_\theta(\cdot \mid s_t) \right] \right]  
$$

　　计算  

$$
d=\hat{E}_t \left[ KL \left[ \pi_{\theta_{old}}(\cdot \mid s_t), \pi_{\theta}(\cdot \mid s_t) \right] \right] \\  
if \; d<d_{targ}/1.5, \; \beta \leftarrow \beta/2 \\  
if \; d>d_{targ} \times 1.5, \; \beta \rightarrow \beta \times 2 \\
$$  

　　更新的$\beta$用于下一个策略更新。以这个方案，我们偶尔看到KL散度与$d_{targ}$显著不同的策略更新，然而，这很稀少，$\beta$很快调整。上面的1.5和2是试探选取的，但算法对它们并不敏感。$\beta$的初始值是另一个超参数但并不重要，因为算法会很快调整它。  

## 5. 算法

　　前一节中的代理损失可以通过对一个典型策略梯度实现的小改变被计算和微分。对于使用自动微分的实现，简单的构建损失$L^{CLIP}$或者$L^{KLPEN}$而不是$L^{PG}$，执行多步随机梯度上升。  

　　大多数用于计算方差缩减优势函数评估量的技术使用一个习得的状态值函数V(s)；例如广义优势评估[Sch+15]，或者[Min+16]中的有限水平评估量。如果使用策略函数和值函数之间共享参数的神经网络架构，我们必须使用一个策略代理目标和值函数误差项组合成的损失函数。这个目标在将来可以通过增加熵奖励来保证足够的探索进行扩展，如同过去[Wil92, Min+16]的工作中建议的。组合这些项，我们得到如下目标，每个迭代步(近似)最大化：  

$$
L^{CLIP+VF+S}(\theta) = \hat{E}_t[L^{CLIP}_t(\theta)-c_1L^{VF}_t(\theta)+c_2S[\pi_\theta](s_t)]  
$$

　　此处$c_1$、$c_2$是系数，S表示熵奖励，$L^{VF}_t$是平方误差损失$(V(s_t)-V^{targ}_t)^2$。  

　　一个类型的策略梯度实现，在[Min+16]中推广且很适合使用循环神经网络，运行T时间步策略(T比计算长度少得多)，使用收集的采样用于更新。这类实现需要不使用超过时间步T的优势函数评估量。[Min+16]使用的评估量为  

$$
\hat{A}_t=-V(s_t)+r_t+\gamma_{t+1}+...+\gamma^{T-t+1}r_{T-1}+\gamma^{T-t}V(s_T)
$$

　　此处t特指[0, T]之内的时间索引，给定一个长度T的轨迹片段。泛化这个选择，我们可以使用一个截断版本的广义优势函数评估，当$\lambda=1$时退化为等式(10):  

$$
\hat{A}_t=\delta_t+(\gamma\lambda)\delta_{t+1}+...+...+(\gamma\lambda)^{T-t+1}\delta_{T-1} \\  
\delta=r_t+\gamma V(s_{t+1})-V(s_t)  
$$

　　使用固定长度轨迹片段的Proximal Policy Optimization (PPO)算法如下展示。每个迭代步，(并行的)N个actors中的每个都收集T时间步数据。然后我们在这些NT时间步数据上构造代理损失，用SGD(或通常为了更好的性能，用Adam[KB14])优化它K轮。  

## 6. 实验

### 6.1 代理目标的比较

　　首先，我们在不同超参数下比较几个不同的代理目标。此处，我们将代理目标$L^{CLIP}$与几个自然变体和简化版本进行比较。  

　　无修剪或惩罚：$L_t(\theta)=r_t(\theta) \hat{A}_t$  

　　修剪：$L_t(\theta)=\min(r_t(\theta)\hat{A}_t, clip(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t)$  

　　KL惩罚(固定或自适应)：  

$L_t(\theta)=r_t(\theta)\hat{A}_t- \beta KL[\pi_{\theta_{old}}, \pi_\theta]$  

　　对于KL惩罚，可以使用固定惩罚系数$\beta$或者如第4节使用目标KL值$d_{targ}$的自适应系数。注记我们也尝试在log空间进行修剪，但发现性能并没有变好。  
　　由于我们为每个算法变体搜索超参数，我们选择在计算简单的标准测试上来测试算法。也就是，我们使用7个OpenAI Gym中实现的模拟机器人任务，使用MuJoCo物理引擎。我们在每个任务上做100W时间步训练。除了用于修剪($\epsilon$)和KL惩罚($\beta, d_{targ}$)的超参数，这是我们要搜索的，其他超参数在表3中给出。  

　　为了表征策略，我们使用全连接MLP，含有两个64节点的隐藏层，和tanh非线性激活函数，输出Gaussian分布的均值，并伴随标准差变量，遵循[Sch+15b, Dua+16]。我们在策略函数和值函数之间不共享参数(所以系数c1是不相关的)，我们不使用熵奖励。  

　　每个算法在所有7个环境上运行，每个环境有3个随机种子。我们通过计算最后100轮的平均总收益对每个算法运行进行评分。我们对每个环境平移并缩放了得分，使得随机策略给出分数0，最好的结果设置为1，对于每个算法设置，平均21轮计算产生单一的标量。  

　　结果在表1中展示。注记对于没有修剪或惩罚的设置，分数为负，因为对于一个环境(half cheetah)，它导致了一个非常负的得分，比最初随机策略还要差。  

### 6.2 连续控制领域中与其他算法比较

　　接下来，我们将PPO(有第3节中修剪代理目标)与几个来自文献中的其他方法进行比较，那些被认为在连续控制问题中很有效的方法。我们与如下算法调整后的实现进行比较：trust region policy optimization[Sch+15b]，cross-entropy method(CEM)[SL16], 自适应步长综合梯度，A2C[Min+16]，含trust region的A2C[Wan+16]。A2C代表advantage actor critic，是A3C的同步版本，我们发现它相比异步版本有相同或更好的表现。对于PPO，我们使用上节的超参数，$\epsilon=0.2$。我们看到PPO在几乎所有连续控制环境上超越之间的算法。  

### 6.3 连续控制领域示例：仿人类奔跑和操控

　　为了展示在高维连续控制问题上PPO的性能，我们在一组包含3D仿人形问题上进行训练，此处机器人必须奔跑，操控，从地上站起来，可能有时被扔方块。我们在其上测试的3个任务是(1)RoboschoolHumanoid：仅向前运动，(2)RoboschoolHumanoidFlagrun：每200时间步或当目标被达到时目标位置随机变化，(3)RoboschoolHumanoidFlagrunHarder, 这里机器人被扔方块，需要从地上站起来。图5为一个完成学习的策略的静止帧，图4为3个任务的学习曲线。超参数在表4中提供。同时的工作中，Heess等人[Hee+17]使用PPO的自适应KL变体(第4节)学习3D机器人的运动策略。  

### 6.4 在Atari游戏领域与其他算法比较

　　我们也在Arcade Learning Environment[Bel+15]上运行PPO，并与调整好的A2C[Mni+16]和ACER[Wan+16]进行比较。对于所有3个算法，我们使用与[Min+16]相同的策略网络架构。PPO的超参数在表5中提供。对于另外两个算法，我们使用在这个标准测试上我们调整到最大性能的超参数。  

　　所有49个游戏的结果表格和学习曲线在附录B中提供。我们考虑如下两个评分标准：(1)整个训练期间每轮的平均奖励(偏向快速学习)，和(2)训练最后100轮的每轮平均奖励(偏向最后性能)。表2显示每个算法的游戏‘赢’的数量，我们通过三个次尝试的平均得分计算胜者。  

## 7 结论

　　我们介绍了Proximal Policy Optimization，一族策略优化方法，使用多轮计算的随机梯度上升来执行每个策略更新。这些方法有稳定性和trust-region method的可信性，但实现起来简单得多，只需要对综合梯度实现改变几行代码，可应用于更广泛的设置(例如，当使用策略函数和值函数的联合架构时)，有整体上更好的性能。  

## 致谢

　　感谢Rocky Duan, Peter Chen, 和OpenAI的其他人的有见地的评论。  

## 参考文献
