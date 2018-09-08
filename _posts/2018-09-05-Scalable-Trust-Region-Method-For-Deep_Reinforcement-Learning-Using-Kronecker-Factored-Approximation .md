---
layout: post
title:  "Scalable Trust-Region Method for Deep Reinforcement Learning using Kronecker-Factored Approximation"
categories: ReinforcementLearning
tags:  ReinforcementLearning, PolicyGradient, ACKTR
author: CQ
---

* content
{:toc}

**Intro:** arxiv 1708  

**Link:** [https://arxiv.org/abs/1708.05144](https://arxiv.org/abs/1708.05144)  

**Code:** [https://github.com/openai/baselines](https://github.com/openai/baselines)  




## 摘要：

　　本工作中，我们使用近期提出的kronecker-factored近似曲率将trust region optimization应用到深度强化学习中。我们扩展了自然策略梯度的框架，使用kronecker-factored近似曲率优化actor和critic，因此，我们将我们的方法称为Actor Critic using Kronecker-Factored Trust Region(ACKTR)。据我们所知，只是第一个用于actor-critic方法的可扩展trust region自然梯度方法。这也是一个在连续控制的非琐碎任务和直接来自原始像素输入的离散控制策略的方法。我们在Atari游戏的离散控制领域和MuJoCo环境的连续控制领域测试我们的方法。利用提出的方法，我们能得到更高的奖励，以及平均2-3倍的抽样效率提升，相比之前最新水平在线策略actor-critic方法。代码链接：https://github.com/openai/baselines。  

## 1.简介

　　使用深度强化学习方法的agents在学习复杂技能和在高维原始探测器状态空间中解决挑战性控制任务显示出巨大的成功。深度强化学习方法使用深度神经网络表征控制策略。尽管有着显著的成果，这些神经网络仍然使用简单随机梯度下降SGD的变体进行训练。SGD以及相关的一阶方法低效探索权重空间。对于现在的深度强化学习方法，经常花费几天时间掌握不同连续和离散的控制任务。以前，一个分布式方法被提出，通过同时执行多agents与环境互动减少训练时间，但随着并行维度的增加，这导致抽样效率收益迅速降低。  

　　抽样效率强化学习重点主导关注点；机器人与真实世界交互通常比计算机时间要稀缺，即使在模拟环境中，模拟的代价经常制约着算法自身。一种高效降低抽样尺度的方法是对梯度更新使用更高级的优化技术。自然策略梯度使用自然梯度下降技术来执行梯度更新。自然梯度方法遵循使用Fisher度量作为基本度量的最陡梯度方向，不是基于所选坐标轴的度量而是流形(如表面)的度量。  

![](/assets/Scalable_Trust_Region_Method_for_Deep_Reinforcement_Learning_using_Kronecker_Factored_Approximation/Figure_1.png)
图1. 在6个标准Atari游戏上的性能比较，训练10,000,000时间步(1时间步等于4帧)。阴影区域表示两个随机种子的标准差。 

　　然而，精确计算自然梯度是不现实的，因为它需要对Fisher信息矩阵求逆。Trust-region policy optimization(TROP)通过Fisher向量积避免显式存储和求逆Fisher矩阵。然而，它通常需要许多步共轭梯度来得到单一组参数更新，准确评估曲率需要在一个batch中的大量抽样，因此，TRPO对于大型模型是不现实的，且抽样低效。  

　　Kronecker-factored近似曲率(K-FAC)是一个对自然梯度的可扩展近似。它已经显式了在使用大mini-batch的监督学习中对各种最新水平大尺度神经网络的加速。不像TRPO，每个更新在计算消耗上与SGD相当，保留运行时平均曲率信息，允许它使用小batch信息。这显示将K-FAC用于策略优化能改进现今深度强化学习方法的抽样效率。  

　　本文中，我们介绍使用Kronecker-factored trust region(ACKTR，发音actor)方法的actor-critic方法，一个对于actor-critic方法的可扩展trust-region优化算法。提出的算法使用Kronecker-factored近似自然策略梯度，允许梯度的协方差矩阵高效的求逆。据我们所知，我们也是第一个扩展自然梯度用于优化值函数，通过高斯-牛顿近似。实际上，ACKTR的每个计算更新只比基于SGD的方法高10%~25%。经验上，我们展示了在Atari环境和MuJoCo环境中，ACKTR同时在抽样效率和agent的最终性能上的大幅度改进，相比最新水平在线策略actor-critic方法A2C和著名的trust region优化器TRPO。

　　我们在https://github.com/openai/baselines上开源我们的代码。  

## 2.背景

### 2.1 强化学习和actor-critic方法

　　我们考虑一个agent与一个无限大平面相互作用，离散Markov决策过程($X, A, \gamma, P, r$)。在时刻t，给定其状态$s \in X$，agent根据其策略$\pi_\theta(a \mid s_t)$选择一个动作$a_t \in A$。环境因此产生一个奖励$r(s_t, a_t)$并根据转移概率$P(s_{t+1} \mid s_t, a_t)$转移至下一个状态$s_{t+1}$。agent的目标是最大化$\gamma$折减累计奖励期望$J(\theta)=E[R_t]=E_\pi[\sum_{i \geq 0}^{\infty} \gamma^ir(s_{t+i}, a_{t+i})]$，策略参数为$\theta$。策略梯度方法直接参数化策略$\pi$并更新参数$\theta$，以最大化目标$J(\theta)$。在通常形式中，策略梯度被定义如  

$$
\nabla_\theta J(\theta)=E_\pi[\sum_{t=0}^{\infty} \Psi^t \nabla_\theta \log \pi_\theta(a_t \mid s_t)]
$$

　　此处$\Psi$通常选择为优势函数$A^\pi(s_t, a_t)$，其为在给定状态$s_t$时每个动作$a_t$的价值的相对度量。在选取低方差和低偏置的优势函数有一个活跃的研究。由于这不是我们工作的焦点，我们简单的遵循A3C方法，定义优势函数为函数近似的k步奖励。  

$$
A^\pi(s_t, a_t)=\sum_{i=0}^{k-1}(\gamma^ir(s_{t+i}, a_{t+i})+\gamma^kV_\phi^\pi(s_{t+k}))-V_\phi^\pi(s_t)
$$

　　此处$V_\phi^\pi(s_t)$是价值网络，提供一个在策略$\pi$下给定状态的奖励总和的期望的评估，$V_\phi^\pi(s_t)=E_\pi[R_t]$。为了训练价值网络的参数，我们再次遵循A3C，执行时序差分更新，来最小化k步奖励$\hat{R}_t $ 和预测价值 $ \frac{1}{2} \mid \mid \hat{R}_t - V_\phi^\pi(s_t) \mid \mid ^2$之间的平方差。  

### 2.2 使用Kronecker-factored近似的自然梯度

　　为了最小化非凸函数$J(\theta)$，最陡梯度方法计算更新$\delta \theta$，最小化$J(\theta+\delta\theta)$，在$\mid \mid \delta\theta \mid \mid _B < 1$的约束下，此处$ \mid \mid \cdot \mid \mid _B $是以$ \mid \mid x \mid \mid _B=(x^TBx)^{1/2} $定义的范数，B是一个半正定矩阵。这个约束优化问题有$ \delta\theta \Rightarrow -B^{-1}\nabla_\thetaJ $的形式，此处$ \nabla_\thetaJ $是标准梯度。当范数是欧几里得范数时，即$B=I$，这变成通常使用的梯度下降方法。然而，变化的欧几里得范数依赖于参数$\theta$。这不合理，因为模型的参数是任意选择的，它不应该影响优化路径。自然梯度方法使用Fisher信息矩阵F构建范数，KL散度的局部二阶近似。这个范数在概率分布上与模型参数$\theta$独立，提供更稳定和高效的更新。然而，由于现代神经网络可能含有数百万参数，计算好存储准确的Fisher矩阵及其逆是不现实的，所以我们不得不进行近似。  

　　一个最近提出的技术叫做Kronecker-factored近似曲率，使用Kronecker-factored近似Fisher矩阵执行高效近似自然梯度更新。我们令$p(y \mid x)$为神经网络的输出分布，$L=\log p(y \mid x)$为对数似然性。令$W \in R^{C_{out} \times C_{in}}$为第$l^{th}$层的权重矩阵，此处$C_{out}$和$C_{in}$是层的输出/输入神经元数量。定义输入本层的激活向量为$a \in R^{C_{in}}$，下一层的激活前向量为$s=Wa$。注记权重梯度有$\nabla_WL=(\nabla_sL)a^T$给出。K-FAC使用这一点并进一步近似与$l$层相关的分块$\hat{F}_l$为

$$
F_l = E[vec{\nabla_WL}vec{\nabla_WL}^T] \\
=E[aa^T \bigotimes \nabla_sL(\nabla_sL)^T] \approx E[aa^T] \bigotimes E[\nabla_sL(\nabla_sL)^T] \\
:= A \bigotimes S := \hat{F}_l
$$

　　此处，A为$E[aa^T]$，S为$E[\nabla_sL(\nabla_sL)^T]$。这个近似可以解释为，假设激活的二阶统计量与反向传播微分无关。有了这个近似，自然梯度更新可以高效的计算，利用基本公式$(P \bigotimes Q)^{-1} = P^{-1} \bigotimes Q^{-1}$和$(P \bigotimes Q)vec(T) = PTQ^T$。  

$$
vec(\Delta W) = \hat{F}_l^{-1}vec{\nabla_WJ}=vec(A^{-1} \nabla_WJS^{-1})
$$

　　从上述公式我们看到，KFAC近似自然梯度更新，只需要与W尺度类似的矩阵计算。Grosse和Martens最近扩展K-FAC算法用于卷积网络。Ba等人随后发展一个方法的分布版本，大多数前序计算通过异步计算得到减轻。在训练大型现代分类卷积网络中，分布式k-FAC达到2-3倍加速。  

![](/assets/Scalable_Trust_Region_Method_for_Deep_Reinforcement_Learning_using_Kronecker_Factored_Approximation/Figure_2.png)
图2. 在Atari游戏Atlantis中，我们的agent(ACKTR)在1.3小时内很快学习获得2,000,000奖励，600游戏计算轮，2,500,000时间步。A2C在10小时内得到同样的结果，6000计算轮，25,000,000时间步。在这个游戏上ACKTR比A2C抽样效率高100倍。

## 3.方法

## 3.1 actor-critic中的自然梯度

　　自然梯度由Kakade在十多年前提出用于策略梯度方法。但仍然不存在一个可扩展的，抽样高效的，目标通用的自然策略梯度实例。本节中，我们介绍首个用于actor-critic方法的可扩展且抽样高效的自然梯度算法：使用Kronecker-factored trust region方法的actor-critic方法(ACKTR)。我们使用Kronecker-factored近似来计算自然梯度更新，将自然梯度更新用于actor和critic中。  

　　为了定义强化学习目标的Fisher度量，一个自然的选择是使用策略函数，定义给定状态下的动作分布，在路径分布上求得期望

$$
F=E_{p(\tau)}[\nabla_\theta \log \pi(a_t \mid s_t)(\nabla_\theta \log \pi(a_t \mid s_t))^T]
$$

　　此处$p(\tau)$是路径分布，由$p(s_0) \prod_{t=0}^T \pi(a_t \mid s_t) p(s_{t+1} \mid s_t, a_t)$给定。实际上，通过训练阶段收集的路径近似这个棘手的期望。  

　　现在我们描述了一种应用自然梯度优化critic的方法。学习critic可以认为是最小二乘近似问题，尽管用的是移动目标。在最小二乘函数近似的设置中，二阶算法选项通常是高斯-牛顿，以高斯-牛顿矩阵$G:=E[J^TJ]$近似曲率，J为将参数映射到结果的Jacobian矩阵。对于高斯观测模型，高斯-牛顿矩阵与Fisher矩阵等效，这个等效允许我们也可以应用K-FAC到critic。特殊的，我们假设critic v的输出被定义为高斯分布$p(v \mid s_t) \tilde N(v;V(s_t),\sigma^2)$。critic的Fisher矩阵由这个高斯输出分布定义。实际上，我们可以简单的设定$\sigma$为1，这与平凡的高斯-牛顿法等效。  

　　如果actor和critic是分离的，可以使用上面定义的度量应用K-FAC分别更新每一个。但是为了避免训练中的不稳定性，通常使用一种两个网络共享低层表达但有各自输出层的结构。这种情况下，我们可以通过假设两个输出分布相互独立，定义策略和值的联合分布，即$p(a,v \mid s)=\pi(a \mid s)p(v \mid s)$，根据$p(a,v \mid s)$构造Fisher矩阵，这与标准K-FAC并无不同，除了我们需要独立的抽样网络输出。然后我们可以同时用K-FAC近似Fisher矩阵$E_{p(\tau)}[\nabla \log p(a,v \mid s) \nabla \log p(a,v \mid s)^T]$来执行更新。  

　　另外，我们使用[16]中描述的因子化Tikhonov缓冲方法。我们也遵循[2]执行Kronecker近似所需的二阶统计和求逆的异步计算，来减少计算时间。  

### 3.2 步长选择和trust-region优化

　　传统上，自然梯度执行类似SGD的更新，$\theta \leftarrow \theta-\eta F^{-1} \nabla_\theta L$。但在强化学习范围内，Schulman等人观察到这个更新规则会导致策略的过大更新，导致算法过早收敛到近似确定策略。他们主张改而使用trust region方法，更新被缩减下来，以至多一个特定的量修改策略分布(KL散度项)。因此，我们采用由[2]引入的K-FAC的trust region公式，选择有效步长$\eta$为$\min(\eta_{\max},\sqrt{\frac{2\delta}{\Delta^T\hat{F}\Delta\theta}})$，此处学习率$\eta_{\max}$和trust region半径$\delta$是超参数。如果actor和critic是分离的，我们需要为两者单独微调不同组的$\eta_{\max}$和$\delta$。对于平凡的高斯-牛顿法，critic输出分布的方差参数可以被吸入学习率参数。另一方面，如果他们共享表征，我们需要微调一组$\eta_{max}$,$\delta$，以及critic的训练损失的权重参数，根据对应的actor。  

## 4. 相关工作

　　自然梯度由Kakade首次用于策略梯度方法。Bagnell和Schneider接着证明[11]中定义的度量是由路径分布流形引入的一个协方差矩阵。Petter和Schaal应用自然梯度到actor-critic算法。他们提出对actor的更新执行自然梯度，对critic的更新使用最小二乘时序差分方法。然而，当使用自然梯度方法时有巨大的计算挑战，主要关于高效存储Fisher矩阵以及计算其逆。为了易于处理，之前的工作将方法限制在使用可计算函数近似。为了避免计算负担，Trust Region Policy Optimization(TRPO)使用快速Fisher矩阵-向量积的共轭梯度近似求解这个线性系统，与Marten的工作类似。这个方法有两个主要缺陷。第一，它需要重复计算Fisher向量积，妨碍了它扩展到通常在Atari和MuJoCo中以图像观察学习的实验中使用的大型架构。其次，它需要大的采样batch来准确评估曲率。K-FAC通过使用易于处理的Fisher矩阵近似以及保留训练中运行时平均曲率统计避免了这两项。尽管TRPO相比以第一阶优化器(如adam)训练的策略梯度方法展示了更好的每迭代改进，它通常抽样更低效。  

　　几个方法被提出用来改进TRPO的计算效率。为了避免重复计算Fisher向量积，Wang等人用运行时平均网络和当前策略网络的KL散度的线性近似求解约束优化问题。不用trust region施加的硬性约束，Heess等人和Schulman等人增加KL损失到目标函数中作为软约束。两篇文章显示一些连续和离散控制任务中对于抽样效率的在平凡策略梯度上的改进。  

　　最近有另外几个引入的actor-critic模型通过引入经验回放改进抽样精度，或辅助目标。这些方法与我们的工作正交，可能潜在的与ACKTR结合更加强化抽样效率。  

## 5. 实验

　　我们构建了一系列实验来研究如下问题：(1)ACKTR相比最新水平在线策略方法和通常的二阶优化器基准，抽样效率和计算效率如何？(2)对于critirc优化，什么范数更好？(3)ACKTR扩展batcha大小时相比一阶方法性能如何？  

　　我们评估我们提出的方法，ACKTR，在两个标准测试平台上。我们首先在OpenAI Gym中定义的离散控制任务上评估它，用Arcade学习环境，一个通常用于离散控制的深度强化学习标准测试的的Atari 2600个游戏的模拟器。然后我们在OpenAI Gym中定义的各种连续控制标准任务上测试它，以MuJoCo物理引擎模拟。我们的基准是(a)A3C的同步且batched的版本，因此叫做A2C，以及(b)TRPO。ACKTR和基准使用同样的模型架构，除了Atari游戏上的TRPO，我们受限使用一个较小的架构，因为运行共轭梯度内部循环的计算负担。其他实验细节见附件。  

![](/assets/Scalable_Trust_Region_Method_for_Deep_Reinforcement_Learning_using_Kronecker_Factored_Approximation/Table_1.png)
表1. 在ACKTR和A2C训练50,000,000时间步和TRPO训练10,000,000时间步后的最后100计算轮得到的平均奖励结果显示。表格也显示计算轮数N，N为从第$N^{th}$到$(N+100)^{th}$轮游戏的计算轮平均奖励超过人类表现水平的首个计算轮数，在2个随机种子间平均。

### 5.1 离散控制

　　我们首先展示标准6个Atari 2600游戏的结果来测量ACKTR获得的性能改进。6个Atari游戏训练10,000,000步的结果如图1显示，与A2C和TRPO相比较。ACKTR在所有游戏中的抽样效率(即每时间步收敛速度)以明显差距超越A2C。我们发现TRPO在10,000,000步内只能学习两个游戏，Seaquest和Pong，在抽样效率上的表现弱于A2C。  

　　在表1中我们展示了训练50,000,000时间步中最后100轮计算的平均奖励，以及达到人类水平需要计算轮次的数量。值得记住的，在Beamrider, Breakout, Pong和Q-bert上，A2C需要ACKTR的2.7, 3.5, 5.3和3.0倍的计算来达到人类水平。另外，A2C运行的Space Invaders的其中一个没能达到人类水平，而ACKTR达到平均19723得分，比人类水平(1652)高12倍。在Breakout, Q-Bert和Beamrider上，ACKTR获得比A2C大26%, 35%和67%的计算奖励。  

　　我们也在剩余Atari游戏上评估了ACKTR；所有结果见附件B。我们将ACKTR与Q-learning方法进行比较，我们发现44个标准测试中的36个，ACKTR的抽样效率与Q-learning方法平分秋色，同时消耗少很多的计算时间。值得记住的，在Atlantis中，ACKTR在1.3小时内(600计算轮)很快学到获得2,000,000奖励，如图2所示。A2C花费10小时(6000计算轮)达到相同性能水平。  

### 5.2 连续控制

　　我们在OpenAI Gym中定义的MuJoCo中模拟的连续控制任务标准测试上运行实验，从低维状态空间表征和直接从像素。相比Atari，连续控制任务优势更具挑战性，由于高维动作空间和探索。8个MuJoCo环境训练1,000,000步的结果如图3。我们的模型在8个MuJoCo任务中的6个明显超越基线，在另外两个任务(Walker2d和Swimmer)表现与A2C相当。  

　　我们又在8个MuJoCo任务上运行30,000,000步评估ACKTR，在表2中我们展示训练中平均奖励最高的10个连续计算轮，以及达到[8]中定义的特定阈值的计算轮数量。如表2，ACKTR在所有任务上更快达到特定阈值，除了Swimmer，TRPO获得4.1倍更高的抽样效率。特别值得注意的是Ant，ACKTR比TRPO多16.4倍抽样效率。由于是平均奖励得分，所有三个模型获得结果与其他两个相当，与TRPO期望相当，它在Walker2d环境中获得10%更好的奖励得分。  

![](/assets/Scalable_Trust_Region_Method_for_Deep_Reinforcement_Learning_using_Kronecker_Factored_Approximation/Figure_3.png)
图3. 8个MuJoCo环境上训练1,000,000时间步(1时间步等于4帧)的性能比较。阴影区域表示3个随机种子的标准差。  

![](/assets/Scalable_Trust_Region_Method_for_Deep_Reinforcement_Learning_using_Kronecker_Factored_Approximation/Figure_4.png)
图4. 在3个MuJoCo环境中从图像观察训练40,000,000时间步的性能(1时间步等于4帧)。  

　　我们也尝试直接从像素学习连续控制策略，不提供低维状态空间作为输入。从像素学习连续控制策略比从状态空间学习更具挑战性，部分由于相比Atari较慢的渲染时间(MuJoCo中的0.5秒 vs Atari中的0.002秒)。最新水平actor-critic方法A3C只报告了在相对简单任务上来自像素的结果，如Pendulum，Pointmass2D和Gripper。如图4我们可以看到我们的模型在训练40,000,000步后的最后计算轮奖励明显超越A2C。更特别的，在Reacher, HalfCheetah和Walker2d上，我们的模型相比A2C获得多1.6, 2.8和1.7倍的最终奖励。从像素训练策略的视频在https://www.youyube.com/watch?vgtM87w1xGoM。预训练模型权重可以在https://github.com/emansim/acktr得到。  

### 5.3 对critic优化的更好的范数？

　　以前的自然梯度法将自然梯度更新用于actor。我们的工作中，我们也将自然梯度更新用于criitc。不同之处在于在critic上我们选择哪种范数执行最陡梯度，就是2.2节定义的范数abs()。本节中，我们应用ACKTR到actor，并将一阶(即欧几里得范数)方法与ACKTR(由高斯-牛顿定义的范数)做critic优化的方法进行对比。图5(a)和(b)显示连续控制任务HalfCheetah和Atari游戏Breakout的结果。我们发现不管我们使用哪个优化critic，应用ACKTR到actor相比基准A2C都有改进。然而，使用高斯-牛顿范数优化critic在抽样效率和训练结束的计算奖励上带来的改进更大幅。另外，高斯-牛顿范数也帮助稳定训练，如用欧几里得范数不同随机种子的结果我们观察到更大的方差。  

![](/assets/Scalable_Trust_Region_Method_for_Deep_Reinforcement_Learning_using_Kronecker_Factored_Approximation/Table_2.png)
表2. ACKTR, A2C和TRPO的结果，显示30,000,000时间步内达到的最高10计算轮平均奖励，在8个随机种子中3个表现最好的种子的平均。计算轮N为从$N^{th}$到$(N+10)^{th}$轮游戏的平均计算轮奖励超过一个特定阈值的最小的N。除InvertedPendulum和InvertedDoublePendulum之外所有环境的阈值选择根据Gu等人的文献[8]，括号中我们展示的奖励临界值需要根据OpenAI Gym求解环境。

　　记得critic的Fisher矩阵使用critic的输出分布构造，方差$\sigma$的高斯分布。在平凡高斯-牛顿中，$\sigma$设为1。我们实验使用Bellman误差的方差评估$\sigma$，类似在回归分析中评估噪音的方差。我们称这种方法为自适应高斯-牛顿。然而，我们发现自适应高斯-牛顿相对平凡高斯-牛顿不提供任何显著改进。(见附件D $\sigma$选择的细节比较)  

### 5.4 ACKTR在计算时间上与A2C相比如何？

　　我们在计算时间上将ACKTR与基准A2C和TRPO相比较。表3显示6个Atari游戏和8个MuJoCo(自状态空间)环境上的每秒平均时间步。使用与前述实验相同的实验设置得到结果。注记在MuJoCo任务中，计算轮被顺序处理，而在Atari环境中计算轮被并行处理，因此在Atari环境中更多帧被处理。从表中我们可以看到ACKTR每个时间步只增加最多25%的计算时间，证明其大型优化器收益的可行性。  

![](/assets/Scalable_Trust_Region_Method_for_Deep_Reinforcement_Learning_using_Kronecker_Factored_Approximation/Table_3.png)
表3. 计算消耗比较。在6个Atari游戏和8个MuJoCo任务上训练每个任务的每秒平均时间步数。ACKTR只比A2C最多增加25%计算时间。

### 5.5 ACKTR和A2C执行不同的batch大小时怎样？

　　在大型分布学习设置中，用大batch来优化。因此，在这种设置中，倾向于使用batch尺寸扩展性能好的方法。本节中，我们比较ACKTR和基准A2C在不同batch尺寸下运行如何。我们以batch大小160和640进行实验。图5(c)显示时间步上的奖励。我们发现大batch尺寸的ACKTR执行的和小batch尺寸同样好。然而，大batch尺寸下，A2C经历抽样效率的明显下降。这与图5(d)中的观察相关，更新数的训练曲线。我们看到ACKTR相比A2C，使用更大的batch尺寸时收益大幅增加。这说明ACKTR在分布设置中可能有潜在的巨大加速，此时需要使用大的mini-batch，这与[2]中的观察一致。  

![](/assets/Scalable_Trust_Region_Method_for_Deep_Reinforcement_Learning_using_Kronecker_Factored_Approximation/Figure_5.png)
图5. (a)和(b)比较了用高斯-牛顿范数(ACKTR)和欧几里得范数(一阶)优化critic(价值网络)。(c)和(d)比较了不同batch尺寸的ACKTR和A2C。

## 6. 结论

　　本工作中，我们对深度强化学习提出一个抽样高效且计算廉价的trust-region-optimization方法。我们为actor-critic方法使用最近提出的K-FAC的技术近似自然梯度更新，为稳定性进行trust region优化。据我们所知，我们是首个提出使用自然梯度优化actor和critic这两者。我们在Atari游戏和MuJoCo环境中测试我们的方法，我们观察到抽样效率的2-3倍改进，相比一阶梯度方法(A2C)和迭代二阶方法(TRPO)。由于我们算法的可扩展性，我们也是首个直接从原始像素观察空间训练几个连续控制中非琐碎任务。这说明扩展Kronecker-factored自然梯度近似到其他强化学习算法是一个可行的研究方向。  

## 致谢

　　我们感谢OpenAI团队，他们慷慨的提供基准结果和Atari环境预处理代码支持。我们也要感谢John Schulman有益的讨论。  

## 参考文献
