---
layout: post
title:  "ProMP: Proximal Meta-Policy Search"
categories: ReinforcementLearning
tags:  ReinforcementLearning
author: CQ
---

* content
{:toc}

**Intro:** ICLR 2019  

**Link:** [https://arxiv.org/abs/1810.06784](https://arxiv.org/abs/1810.06784)  

**Code:** [https://github.com/jonasrothfuss/ProMP](https://github.com/jonasrothfuss/ProMP)  




## 摘要：

　　人们对元强化学习(Meta-RL)中的信用分配问题仍然理解不够。现存方法或者将信用分配略到预匹配行为里，或者非常朴素的实现它。这导致元训练过程中的低抽样效率，以及无效的任务识别方案。本文提供了一个基于梯度元强化学习中信用分配的理论分析。基于得到的结论，我们发展了一个全新的元学习算法，克服了贫乏的信用分配以及之前的评测元策略梯度的困难。通过元策略搜索过程中控制预适应策略的统计距离以及自适应策略的统计距离，提出的算法得到了高效且稳定的元学习。我们的方法优于了预适应策略行为，在抽样效率，运行时间，以及渐近性能上稳定超越了之前的元强化学习算法。代码链接：https://github.com/jonasrothfuss/promp。  

## 1.简介

　　人类智能的一个值得记住的点是在面对有限经验下适应新环境的能力。相反，我们大多数成功的人工agents在这种情景中表现很差。尽管有令人印象深刻的成果，他们承受着甚至单个学习任务中的高抽样复杂度，新情况中泛化失效，需要大量新增数据来成功适应新环境。元学习通过学习如何学习来处理这些缺陷。其目标是学习一个算法，允许人工agent在未见过的任务中，当只有有限经验可得的时候成功运行，想要达到与人类具有的同样快速的适应能力。  

　　除了最近的进步，深度强化学习仍然严重依赖人工修改的特征和奖励函数，以及修正问题的特定归纳偏差。元强化学习意图通过以数据驱动的方式添加归纳偏差来放弃这种依赖。最近的工作证明这种方法是有希望的，实例显示元强化学习允许agent获得多元的技能集合，达到更好的探索流程，通过元学习好的动态模型更快的学习或合成回报。  

　　元强化学习是一个多阶段过程，agent在几个抽样的环境互动后，适应其行为到给定任务。除了其广泛的应用，很少有工作来推动这个过程的理论理解，使元强化学习停留在不稳固的基础上。尽管自适应步之前的行为对于任务识别是有帮助的，预适应抽样和策略的后验性能之间的相互影响仍然理解很少。事实上，之前基于梯度元强化学习的工作或者全部忽略预更新分布的信用分配，或者以很朴素的方式实现这个信用分配。  

　　我们所知，我们提供了元强化学习中首个正式的预适应抽样分布的信用分配的深度分析，我们发展了一个全新的元强化学习算法。首先，我们分析两个不同的分配信用到预适应行为的方法。我们展示了最近的Al-Shedivat等人和Stadie等人引入的公式导致较差的信用分配，同时MAML形式可能产生优良的元策略更新。其次，基于从我们正式分析得到的启示，我们突出了合适的元策略梯度估算的重要性和困难。基于这点，我们提出低方差曲率(low variance curvature, LVC)伪目标，以一个合适的偏差-方差权衡产生梯度评估。最后，在LVC评估的基础之上，我们发展了Proximal Meta-Policy Search(ProMP)，一个对于强化学习来说高效且稳定的元学习算法。在我们的实验中，我们显示ProMP在抽样效率，运行时间和渐近性能上稳定超越之前的元强化学习算法。  

## 2.相关工作

　　元学习关心‘学会学习’这个问题，目的是以数据驱动的方式添加归纳偏差，以此使面对未见过数据或新问题设置的使学习过程得到加速。  

　　这可以通过各种方式达到。一类方法试图学习一个通用图灵机的‘学习过程’。以一种循环/记忆扩充模型的形式，摄取数据并输出训练模型的参数或者对给定测试输入直接输出预测。尽管非常灵活且有能力进行非常高效的适应学习，这种模型不能保证性能且在元强化学习的长序列问题上难于训练。  

　　另一类方法在元学习过程中嵌入经典学习算法结构，在元训练过程中优化嵌入学习的参数。后者一个典型的被证明相当成功的强化学习范围的例子是基于梯度的元学习。其目的是学习一个初始化器，在一步或几步策略梯度后agent在新任务上达到完全性能。这种方法的一个很好的特性是，即便快速适应失效，agent只是回到平凡策略梯度法。然而，如我们展示，之前基于梯度的元强化学习完全忽略或者执行贫乏的对更新前抽样分布的信用指定。  

　　各种各样基于元强化学习的方法被发展出来。包括：学习探索流程，合成奖励，无监督策略获取，基于模型强化学习，竞争环境中的学习，元学习模块化策略。许多提到的方法建立在之前基于梯度的未充分计入更新前分布的元学习方法之上。ProMP克服了这些缺陷，为在未解问题中全新的元强化学习应用提供了必要的框架。  

## 3.背景

　　**强化学习**。一个离散时间有限Markov决策过程(MDP)，$T$，被定义为元组$(S, A, p, p_0, r, H)$。此处，$S$是状态集合，$A$是动作空间，$p(s_{t+1} \mid s_t,a_t)转换分布，$p_0$表示初始状态分布，$r:S \times A \rightarrow R$是奖励函数，以及$H$时间范围。为符号记法简洁，我们在随后文中省略折扣因子$\gamma$。而是通过以$r(s_t,a_t):=\gamma^tr(s_t,a_t)$替代奖励而引入它。我们定义回报$R(\tau)$为轨迹$\tau:=(s_0,a_0,...,s_{H-1},a_{H-1},s_H)$奖励的和。强化学习的目标是找到策略$\pi(a \mid s)$最大化回报期望$E_{\tau ~ P_\tau(\tau \mid \pi)}[R(\tau)]$。  

　　**元强化学习**更进了一步，目的是学习一个学习算法，有能力对任务分布$\rho(T)$中抽样的任务$T$快速学习最优策略。每个任务$T$对应不同的MDP。典型的，其假设任务分布共享相同的动作和状态空间，但也许在奖励函数或动态上有所不同。  

　　**基于梯度的元学习**目的是学习一个策略$\pi$的参数$\theta$，在给定任务上执行一步或几步平凡策略梯度获得这个任务的最优策略，以此解决这个问题。这种元学习形式，也名为MAML，首次由Finn等人引入。我们将其作为参照形式$I$，可以表示为最大化目标  

$$
J^I(\theta) = E_{T ~ \rho(T)}[E_{\tau' ~ P_T(\tau' \mid \theta')}[R(\tau')]]  \theta':=U(\theta, T)=\theta+\alpha \nabla_\theta E_{\tau ~ P_T(\tau \mid \theta)}[R(\tau)]
$$

式中$U$表示更新函数，依赖于任务$T$，向最大化$T$中的策略性能执行一步VPG。为简明扼要，我们假设单一策略梯度适应步。尽管如此，所有展示的概念能容易的扩展到多适应步。  

　　后续工作提出了一个基于梯度元强化学习的轻微不同形式，名为E-MAML，试图规避MAML中元梯度评估的争议：  

$$
J^{II}(\theta) = E_{T ~ \rho(T)}[E_{\begin{matrix} \tau^{1:N} ~ P_T(\tau^{1:N} \mid \theta) \\ \tau' ~ P_T(\tau' \mid \theta') \\ \end{matrix}}[R(\tau')]] \\
\theta' := U(\theta, \tau^{1:N}) = \theta + \alpha \nabla_\theta \sum_{n-1}^N [R(\tau^{(n)})]
$$

　　形式$II$将$U$视为依赖从一个特定任务中N次抽样轨迹的确定函数。与形式I相对，更新前轨迹$\tau$的期望在更新函数外应用。本文中我们将$\pi_\theta$记为更新前策略，$\pi_theta'$记为更新后策略。  

## 4. 抽样分布信用指定

　　本节分析了第3节引入的两个基于梯度元强化学习形式。图1显示了两种形式的随机计算图。红色箭头描述了对于更新前抽样分布P(\tau \mid \theta)的信用分配是如何传播的。形式I(左)通过更新步传播信用指定，因此利用全部问题结构。相反，形式II(右)忽略内部结构，直接从更新后回报$R'$到更新前策略$\pi_\theta$指定信用，这导致噪音更多，信用指定更低效。  

![](assets/ProMP_Proximal_Meta_Policy_Search/Figure_1.png)  
图1. 元学习形式$I$和形式$II$的随机计算图。红色箭头示意通过$\nabla_\theta J_{pre}$的从更新后回报$R'$到更新前策略$\pi_\theta$的信用分配。(精确节点：方块；随机节点：圆)  

　　两个形式优化同一目标，在第$0^{th}$阶等效。然而，由于他们形式和随机计算图的不同，他们的梯度和优化步结果也不同。接下来，我们通过分析两种形式的梯度，指明形式II如何以及在哪儿丢失信号，梯度可以写为  


$$
\nabla_\theta J(\theta) = E_{T ~ \rho(T)}[E_{\begin{matrix} \tau ~ P_T(\tau \mid \theta)\\ \tau' ~ \mid P_T(\tau' \mid \theta')\\ \end{matrix}}[\nabla_\theta J_{post}(\tau,\tau') + \nabla_\theta J_{pre}(\tau,\tau')]]
$$

　　第一项$\nabla_\theta J_{post}(\tau, \tau')在两个形式中相等，但第二项，$\nabla_\theta J_{pre}(\tau, \tau')，两者不同。特别的，他们对应于  

$$
\nabla_\theta J_{post}(\tau, \tau') = (I + \alpha R(\tau) \nabla_\theta ^2 log \pi_{\theta'}(\tau)) \nabla_{\theta'} log \pi_theta (\tau') R(\tau') \\
\nabla_\theta J_{pre}^{II} = \alpha \nabla_\theta log \pi_\theta (\tau) R(\tau') \\
\nabla_\theta J_{pre}^{I} = \alpha \nabla_\theta log \pi_\theta(\tau) ((\nabla_\theta log \pi_\theta(\tau)R(\tau))^T (\nabla_{\theta'} log \pi_{\theta'} (\tau') R(\tau')))
$$

$\nabla_\theta J_{post}(\tau, \tau')$简单的对应于关于$\theta'$的更新后策略$\pi_{\theta'}$的策略梯度步，后跟一个从更新后到更新前参数的线性变换。它对应于增加导致更高回报的轨迹$\tau'$的似然性。然而，这一项对更新前抽样分布并没有优化，即哪个轨迹$\tau$导致更好的适应步。  

　　关于更新前抽样分布的信用分配由第二项产生。在形式II中，$\nabla_\theta J_{pre}^{II}$可被视为以$R(\tau')$作为奖励信号的$\pi_\theta$上的标准强化学习，将更新函数U处理为未知系统的一部分。这将更新前抽样分布移动到更好的适应步。  

　　形式$I$计入$P_T(\tau' \mid \theta')$依赖于$P_T(\tau \mid \theta)$的因果关系。它通过最大化更新前和更新后策略梯度的内积来做这件事情。这驱使更新前策略向1)更大更新后回报，2)更大适应步$\alpha \nabla_\theta J^{inner}$，3)更好的更新前后策略梯度校准。当综合时，这些效果直接对适应进行优化。结果是，我们期望第一个元策略梯度形式，$J^I$，产生优秀的学习性质。  

## 5. 低方差曲率评估

　　前一节我们展示了Finn等人引入的形式得到更优秀的元梯度更新，原则上应该导致改进的收敛性。然而，得到各自元梯度的正确且低方差评估被证明是一个挑战。如Foerster等人所讨论，伪目标方法得分函数不适合通过自动微分工具箱计算高阶导数。这个重要的事实在原始的RL-MAML实现中被忽略了，导致不正确的元梯度评估。但是，即便正确实现，我们展示了这些梯度显示出非常高的方差。  

　　特别的，强化学习目标的Hessian矩阵的评估需要特别考虑，这是元梯度固有的。本节中，我们引入低方差曲率评估(LVC)：一个改进的强化学习目标的hessian矩阵评估器，促进更好的元策略梯度更新。如附录A1所示，我们可以将元学习目标的梯度写为  

$$
\nabla_\theta J^I (\theta) = E_{T ~ \rho(T)}[E_{\tau' ~ P_T(\tau' \mid \theta')}[\nabla_{\theta'} log P_T(\tau' \mid \theta') R(\tau') \nabla_\theta U(\theta, T)]]
$$

　　由于更新函数$U$像一个策略梯度步，其梯度$\nabla_\theta U(\theta, T)$涉及计算强化学习目标的hessian矩阵，即$\nabla_\theta ^2 E_{\tau ~ P_T(\tau \mid \theta)}[R(\tau)]$。评估这个hessian矩阵已经在Baxter & Harlett和Furmston等人的文献中讨论过。在无限时间域MDP情况，Baxter & Bartlett推导一个hessian矩阵的解耦。我们将他们的发现扩展到了有限时间域情况，hessian可被解耦为三个矩阵项：  

$$
\nabla_\theta U(\theta, T) = I + \alpha \nabla_\theta ^2 E_{\tau ~ P_T(\tau \mid \theta)}[R(\tau)] = I + \alpha (H_1 + H_2 + H_{12} + H_{12}^T)
$$

此处  

$$
H_1 = E{\tau ~ P_T(\tau \mid \theta)}[\sum_{t=0}^{H-1} \nabla_\theta log \pi_\theta (a_t,s_t) \nabla_\theta log \pi_\theta (a_t, s_t)^T (\sum_{t'=t}^{H-1}r(s_{t'},a_{t'}))] \\
H_2 = E{\tau ~ P_T(\tau \mid \theta)}[\sum_{t=0}^{H-1} \nabla_\theta ^2 log \pi_\theta (a_t,s_t)(\sum_{t'=t}^{H-1}r(s_{t'},a_{t'}))] \\
H_{12} = E{\tau ~ P_T(\tau \mid \theta)}[\sum_{t=0}^{H-1} \nabla_\theta log \pi_\theta (a_t,s_t) \nabla_\theta Q_t^{\pi_\theta}(s_t,a_t)^T]
$$

此处$Q_t^{\pi_\theta}(s_t, a_t)=E_{\tau^{t+1:H-1} ~ P_T(\cdot \mid \theta)}[\sum_{t'=t}^{H-1} r(s_{t'}, a_{t'}) \mid s_t, a_t]$表示在时间$t$策略$\pi_\theta$下状态-动作值函数的期望。  

　　计算强化学习目标的期望通常是不现实的。典型的，其梯度由一个Monte-carlo评估来计算，基于策略梯度理论。实际实现中，这个评估由自动微分一个伪目标得到。然而，这个结果导致高度偏置的hessian矩阵评估，只计算$H_2$，完全扔掉$H_1$和$H_{12}+H_{12}^T$项。在前一节的记号中，它导致忽略$\nabla_\theta J_{pre}$项，忽略更新前抽样分布的影响。  

　　这个问题可以使用DiCE形式进行解决，允许计算任意随机计算图的无偏高阶Monte-Carlos评估。DiCE-强化学习目标可以重写为  

$$
J^{DiCE}(\tau) = \sum_{t=0}^{H-1}(\prod_{t'=0}^t \frac{\pi_\theta(a_{t'} \mid s_{t'})}{\bot (\pi_\theta (a_{t'} \mid s_{t'}))}) r(s_t,a_t) \quad \tau ~ P_T(\tau) \\
E{\tau ~ P_T(\tau \mid \theta)}[\nabla_\theta^2 J^{DiCE}(\tau)] = H_1 + H_2 + H_{12} + H_{12}^T
$$

式中，$\bot$表示‘停止梯度’算子，即$\bot (f_\theta(x)) \rightarrow f_\theta(x)$，但是$\nabla (f_theta(x)) \rightarrow 0$。通过式7的重要性权重得到表现的轨迹上$\pi_\theta(a_t \mid s_t)$的序列依赖，导致hessian矩阵$\nabla_\theta^2 E{\tau ~ P_T(\tau \mid \theta)}[R(\tau)]$的高方差评估。如Furmston等人所写，$H_{12}$尤其难以估计，由于它涉及延轨迹的三个嵌套求和。在7.2中我们基于经验的展示，DiCE目标的高方差评估导致嘈杂的元策略梯度以及低廉的学习性能。  

　　为了促进抽样高效的元学习，我们引入高方差曲率评估  

$$
J^{LVC}(\tau)=\sum_{t=0}^{H-1} \frac{\pi_\theta(a_t \mid s_t)}{\bot (\pi_\theta(a_t \mid s_t))}(\sum_{t'=t}^{H-1}r(s_{t'},a_{t'})) \quad \tau ~ P_T(\tau) \\
E{\tau ~ P_T(\tau \mid \theta)}[\nabla_\theta^2 J^{LVC}(\tau)] = H_1 + H_2
$$

　　通过移除轨迹上$\pi_\theta (a_t \mid s_t)$的序列依赖，hessian矩阵评估忽略了$H_{12}+H_{12}^T$项，这导致方差缩减，但使得估计有偏置。选择这个目标是受Furmston等人的启发：在特定情况下，$H_{12}+H_{12}^T$项在局部最优$\theta \ast$附近消失，即当$\theta \rightarrow \theta \ast$时，$E_\tau [\nabla_\theta^2 J^{LVC}] \rightarrow E_\tau [\nabla_\theta^2 J^{DiCE}]$。因此，LVC评估的偏置在接近局部最优时变得可忽略。7.2节的实验支持这个理论发现，相比于$J^{DiCE}$，通过$J^{LVC}$得到的低方差hessian矩阵评估以显著的差距改进了元学习的抽样效率。有兴趣的读者可以参考附录B的推导和更细节的讨论。

## 6. ProMP: Proximal Meta-Policy Search

　　在前一节的基础上，我们发展了一个全新的元策略搜索方法，基于低方差曲率目标，目的是解决如下优化问题：  

$$
\under{max}_\theta E{\tau ~ P_T(\tau \mid \theta)}[E_{\tau' ~ P_T(\tau' \mid \theta')}[R(\tau')]] \quad \theta' := \theta + \alpha \nabla_\theta E_{\tau ~ P_T(\tau \mid \theta)}[J^{LVC}(\tau)]
$$

　　之前的工作已经使用平凡策略梯度(VPG)或TRPO优化这个目标。相比VPG，TRPO依然更数据高效且稳定。然而，它需要计算Fisher信息矩阵(FIM)。评估FIM在元学习设置中特别成问题。元策略梯度已经涉及二阶导数，结果是，FIM评估的时间复杂度是策略参数数量的立方。典型的，使用有限差分方法的话问题是无解的，由于其引入更进一步的近似误差。  

　　最近引入的PPO算法达到与TRPO相似的结果且有一阶方法的优势。PPO使用伪修剪目标，允许安全的处理多梯度步而不需要重抽样轨迹。  

$$
J_T^{CLIP}(\theta) = E_{\tau ~ P_T(\tau, \theta_o)}[\sum_{t=0}^{H-1}(\frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_o}(a_t \mid s_t)}A^{\pi_{\theta_o}}(s_t, a_t), clip_{1-\epsilon}^{1+\epsilon}(\frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_o}(a_t \mid s_t)})A^{\pi_{\theta_o}}(s_t,a_t))]
$$

![](assets/ProMP_Proximal_Meta_Policy_Search/Algorithm_1.png)

　　元强化学习情况中，它并不能仅仅将$J_T^{CLIP}$替换为更新后奖励目标。为了能在当前策略$\pi_{\theta_o}$的相同抽样数据上安全执行多个元梯度步，我们也需要1)计入更新前动作分布$\pi_\theta(a_t \mid s_t)$的变化，以及2)将改变限制在更新前状态可见分布范围之内。  

　　我们提出Proximal Meta-Policy Search(ProMP)，综合PPO的优势以及LVC目标。为了满足需求1)，ProMP将‘停止梯度’重要性权重$\frac{\pi_\theta(a_t \mid s_t)}{\bot (\pi_\theta(a_t \mid s_t))}$替换为似然比率$\frac{\pi_\theta(a_t \mid  s_t)}{\pi_{\theta_o}(a_t \mid s_t)}$，得到如下目标  

$$
J_T^{LR}(\theta) = E_{\tau ~ P_T(\tau, \theta_o)}[\sum_{t=0}^{H-1} \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_o}(a_t \mid s_t)}A^{\pi_{\theta_o}}(s_t, a_t)]
$$

　　这个目标的一个重要的特征是，它的关于$\theta$的导数在$\theta_o$处的估计与LVC目标关于$\theta$的导数在$\theta_o$处的估计是一致的，另外它计入更新前动作分布的变化。为满足条件2)，我们用$\pi_\theta$和$\pi_{\theta_o}$之间的KL距离惩罚项扩展了修剪元目标。这个KL惩罚项促使一个在$\pi_{\theta_o}$周围的软局部'信任区'，防止优化过程中状态可见分布的平移变大。这使我们能够执行多步元策略梯度而不用重抽样。另外，ProMP优化  

$$
J_T^{ProMP}(\theta) = J_T^{CLIP}(\theta') - \eta \overline{D} _{KL} (\pi_{\theta_o}, \pi_\theta) \quad s.t. \quad \theta' = \theta + \alpha \nabla_\theta J_T^{LR}(\theta), \quad T ~ \rho(T)
$$

　　ProMP巩固了本文发展而来的观点，同时最大程度利用最近发展的策略梯度算法。首先，其元学习形式利用了基于梯度元学习的整体结构。其次，它加入了强化学习目标hessian矩阵的低方差估计。第三，ProMP控制了适应前策略的统计距离和适应后策略的统计距离，促使高效且稳定的元学习。总之，ProMP在抽样复杂度，运行时间和渐近性能上，稳定超越之前基于梯度的元强化学习算法。  

## 7 实验  

　　为了经验验证上述理论上的讨论，本节提供了细节实验分析，目的是回答下列问题：(i)ProMP相比之前元强化学习算法性能如何？(ii)低方差但有偏的LVC梯度估计相比高方差无偏DiCE估计如何？(iii)不同形式是否导致不同的更新前探索性质？(iv)形式I和形式II在元梯度估计和收敛特性上如何不同？  

　　为了回答上述问题，我们在6个连续控制元强化学习标准环境上评估我们的方法，基于OpenAI GYM和MuJoCo模拟器。实验设置描述见附录D。在所有实验中，报告的曲线是在至少3个随机种子上的平均值。回报基于适应的更新后策略抽样轨迹估计得到，并在抽样任务上平均。源代码和实验数据在如下网址中。  

### 7.1 基于元梯度方法的比较  

　　在抽样复杂度和渐近性能上，我们将我们的方法，ProMP，与其他四种基于梯度的方法进行比较：TROP-MAML，E-MAML-TROP，EMAML-VPG，和LVC-VPG，一个我们方法的简化版本，在优化步使用LVC目标并用平凡策略梯度进行元优化。这些算法在6个不同的需要适应的运动任务上进行测试：half-cheetah和walker必须在向前跑和向后跑之间切换，高维agent，ant和humanoid，必须学习适应在2D平面上跑不同的方向，hopper和walker需要适应不同自身设置。  

![](assets/ProMP_Proximal_Meta_Policy_Search/Figure_2.png)  
图2. 在6个不同MuJoCo环境中ProMP以及4个其他基于梯度元学习算法的元学习曲线。ProMP在所有环境中超越了之前的工作。  

　　结果凸显了ProMP在抽样效率和渐近性能上的强大，如图2。它们也证明了LVC目标的正面效果：LVC-VPG，尽管用平凡策略梯度进行优化，经常能达到与之前用TRPO优化方法可以相比较的结果。当与E-MAML-VPG比较时，LVC在性能上证明了严格的优秀，支持了本文发展的理论的稳健性。额外4个环境的结果在附录4中显示，以及参数设置，环境设定和几个算法的运行时间比较。  

### 7.2 方差估计  

　　第5节中，我们讨论DiCE形式如何产生无偏但高方差强化学习目标hessian矩阵估计的，以及作为低方差曲率评估(LVC)的思路来源。这里我们研究了两个评估器的元梯度方差以及其在学习性能上的意义。特别的，我们报告了HalfCheeahFwdBack环境中元策略梯度的相对标准差以及整个学习过程的平均回报。结果凸显了低方差曲率估计的优势，如图3。DiCE评估器中固有的轨迹级依赖性使得其元梯度标准差平均比LVC高2倍。如学习曲线所示，在DiCE情况下，嘈杂的梯度阻碍了高效抽样的元学习。基于LVC评估器的元策略搜索导致大幅度的更好的学习性质。  

![](assets/ProMP_Proximal_Meta_Policy_Search/Figure_3.png)  
图3. 上方：元策略梯度的相对标准差。下方：HalfCheetahFwdBack环境中的回报。  

### 7.3 初始抽样分布比较

　　这里我们在学到的更新前抽样分布上，评估不同目标的效果。我们比较结合TRPO的低方差曲率评估器(LVC-TRPO)，以及MAML和E-MAML-TRPO，在2D环境中以便探索行为可以可视化。环境中的每个任务对应于达到不同的角落位置，然而，只有当2D agent充分接近角落时才获得奖励。因此，为了成功识别任务，agent必须探索不同区域，我们在每个任务上执行3内适应步，允许agent充分改变其行为，从探索到利用。  

![](assets/ProMP_Proximal_Meta_Policy_Search/Figure_4.png)  
图4. 更新前策略的探索模式和不同更新函数的更新后利用。由于有有修改的信用分配，LVC目标学习了能够识别当前任务并适应其策略的更新前策略，成功到达目标。  

　　不同的探索-利用流程如图4所示由于MAML实现没有分配信用到更新前抽样轨迹，它不能为任务识别学习合理的探索流程，因此没能完成任务。另一方面，E-MAML，对应于形式II，学习到探索长的但随机的路径：因为它只能分配信用到整批的更新前轨迹，没有标注哪个特别动作促进好的任务适应。作为后果，适应后的策略轻微偏移的任务指定目标。而LVC评估器学习了连贯的探索模式，观察了每个区域，使其完整的解决了任务。  

### 7.4 两个元强化学习形式的梯度更新方向

　　为了更多的显示形式I和形式II梯度的不同，我们在一个简单的1D环境中评估元梯度更新和相的关到两种形式最优值的收敛性。这个环境中，agent在实数轴的随机位置开始，要达到在1或-1位置的目标。为了可视化收敛性，我们只用\theta_0和\theta_1参数化策略。我们以VPG优化DiCE目标的方式使用形式I，以VPG优化E-MAML目标的方式使用形式II。  

![](assets/ProMP_Proximal_Meta_Policy_Search/Figure_5.png)  
图5. 在1D环境中，关于形式I和形式II的策略参数\theta_0和\theta_1的元梯度更新。  

　　图5描述了两种形式的参数\theta_i的元梯度更新。形式I利用适应更新内部结构，产生更快更稳定的收敛到最优。由于其低级的信用分配，形式II产生更嘈杂的梯度估计，导致更差的收敛特性。  

## 8. 结论

　　本文中我们提出一个全新的元强化学习算法，Proximal Meta-Policy Search(ProMP)，全面优化了更新前抽样分布，导致高效的任务识别。我们的方法是基于梯度元强化学习形式的理论分析的结果，基于我们发展的低方差曲率伪目标，其产生低方差元策略梯度估计。实验结果证明我们的方法在各种不同的连续控制任务集合中超越之前的元强化学习方法。最后，我们以可视化的例子巩固了我们的理论贡献，进一步断定我们的方法的稳健性和高效性。  

## 致谢

　　Ignasi Clavera由La Caixa奖学金提供支持。本结果的研究收到了German Research Foundation的Priority Program on Autonomous Learning的项目，也得到了Berkeley Deep Drive, Amazon Web Services, 和Huawei的支持。我们也感谢Abhishek Gupta, Chelsea Finn, 和Aviv Tamar的有价值的反馈。  

## 参考文献
