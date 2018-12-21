---
layout: post
title:  "Recurrent World Models Facilitate Policy Evolution"
categories: ReinforcementLearning
tags:  ReinforcementLearning, WorldModels
author: CQ
---

* content
{:toc}

**Intro:** NeurlIPS 2018  

**Link:** [https://arxiv.org/abs/1809.01999](https://arxiv.org/abs/1809.01999)  

**Code:**  
[https://worldmodels.github.io](https://worldmodels.github.io)  
[https://github.com/hardmaru/WorldModelsExperiments](https://github.com/hardmaru/WorldModelsExperiments)  




## 摘要：

　　一个生成式循环神经网络以无监督形式进行训练，通过压缩空间-时间特征来模拟流行的强化学习环境。世界模型提取的特征输入简单紧凑的以进化算法训练的策略，在不同环境中达到最新水平结果。我们也完全在它自己内部世界模型生成的环境内训练agent，将策略迁移回真实环境。游戏视频和代码链接：[https://worldmodels.github.io](https://worldmodels.github.io).  

## 1.简介

　　人类心里拥有一个世界模型，基于他们用有限的感官感知的东西，学习感官输入的空间和时间的抽象表征。例如，我们能观察一个景象，因而记住一个抽象描述。我们的决定和行动被内部预测模型影响。例如，在任何给定时刻我们感知到的东西，似乎被我们对未来的预测所控制。一种理解我们大脑中预测模型的方式是，它不是通常的关于预测未来，而是预测未来感官输入数据给当前行为驱动部分。我们能直接在这个预测模型上行动，当面临危险时执行快速反应行为，不需要有意识的计划行动序列。  

　　对于很多强化学习问题，人工RL agent也从未来预测模型(基于模型的RL)中获益。反向传播算法能用来训练神经网络组成的大型模型。在部分可观测环境中，我们可以通过循环神经网络(RNN)实现预测模型(M)，来允许基于之前观测序列记忆的更好的预测。  

![](/assets/Recurrent_World_Models_Facilitate_Policy_Evolution/Figure_1.png)  
图1. 我们建立OpenAI Gym环境的概率生成模型。这些模型可以模仿真实环境(左)。我们在真实环境中测试训练好的策略(右)。  

　　实际上，预测模型(M)将是一个以无监督形式在给定过去时学习预测未来的大型RNN。预测模型(M)的过去观测和行动内部记忆表征，通过另一个叫做控制器(C)的神经网络所感知和利用，控制器(C)通过强化学习执行某些任务而不用教导。一个小而简单的控制器(C)将控制器的信用分配问题限制到相对小的搜索空间，不用牺牲大而复杂的预测模型的预测和表达能力。  

　　我们将1990-2015年间基于RNN世界模型和控制器的几个关键概念，与最近的概率模型工具相结合，提出一个简单的方法在现代强化学习环境中测试这些关键概念。实验显示我们的方法可以用来解决具有挑战性的从像素进行赛车导航的任务，这个之前使用传统方法没有被解决的任务。  

　　大多数现存的基于模型RL方法学习一个RL环境的模型，但仍在真实环境中训练。这里，我们也探索了用生成环境全部替代真实RL环境，只在其自己内部世界模型(M)生成的环境内训练控制器(C)，将策略迁移回真实环境。  

　　为了克服agent利用不完美生成环境的问题。我们增加了世界模型(M)的一个温度参数，控制生成环境不确定的数量。我们在更多噪音和生成环境的更不确定版本中训练控制器(C)，证明这个方法防止控制器(C)利用世界模型(M)的不完美。我们也讨论了其他基于模型强化学习文献的相关工作，有类似的学习环境机制模型并用这个模型训练agent想法的文献。  

## 2.agent模型

　　我们的模型是受我们自身认知系统的启发。agent有一个视觉传感器组件V，将它所见的压缩成一个小的表征码。也有一个记忆组件M，基于历史信息对未来表征码做出预测。最后，agent有一个决策组件C，仅基于视觉产生的表征和记忆组件决定执行哪种动作。  

![](/assets/Recurrent_World_Models_Facilitate_Policy_Evolution/Figure_2.png)  
图2. 显示V，M和C如何与环境交互的流程图(左)。agent模型在OpenAI Gym环境中如何使用的伪代码。  

　　环境在每个时间步提供agent高维输入观测。这个输入通常是2D图像帧，是视频序列的一部分。视觉组件(V)的作用是学习一个每个观测输入帧的抽象压缩表征。这里，我们使用变分自编码器(VAE)作为视觉组件V，压缩每个图像帧到隐变量向量$z$。  

　　V的作用是压缩agent在每个时间帧看到的东西，我们也想压缩在时间上发生了什么事情。RNN世界模型M作为预测未来视觉组件V将要产生的隐变量$z$的模型。由于许多复杂环境是随机的，我们训练RNN输出概率密度函数$p(z)$而不是精确的预测$z$。  

　　在我们的方法中，我们将$p(z)$近似为混合Gaussian分布，训练世界模型M在给定它能得到的当前和历史信息时输出写一个隐变量$z_{t+1}$的概率分布。更特别的，RNN将建模$P(z_{t+1} \mid a_t, z_t, h_t)$，$a_t$是时间$t$的动作，$h_t$是时间$t$时RNN的隐藏状态。抽样过程中，我们可以加入一个温度参数$\tau$，控制模型的不确定性，如前人所做。我们后来发现增加$\tau$对于训练控制器很有用。这个方法被称为混合密度网络(Mixture Density Network)结合RNN(MDN-RNN)，过去被用于序列生成问题，如生成手写数字或草图。  

　　控制器C用来决定进行动作的历程，最大化环境中一个计算轮的累计期望奖励。我们的实验中，我们有意让控制器C尽可能小而简单，与V和M分开训练，这使得agent的大多数复杂度留在V和M中。C是一个简单的单层线性模型，在时间$t$直接将$z_t$和$h_t$映射到动作$a_t：a_t = W_c[z_t h_t] + b_c$。在这个线性模型中，$W_c$和$b_c$是映射串接输入向量$[z_t h_t]$到输出动作向量$a_t$的参数。  

　　这个最简单的C也提供了重要的实践收益。深度学习的优势提供了我们高效训练大型复杂模型的工具，使我们能定义良好的可微损失函数。V和M使用现代GPU加速器以反向传播算法高效训练，所以我们希望大多数模型复杂度和模型参数保留在V和M中。线性模型C的参数数量相对来讲是最小的。这个选择使我们能探索使用更加非传统的方法来训练C——例如使用进化策略(ES)来解决根据挑战性的难于信用分配的任务。  

　　为了优化C的参数，我们选择Covariance-Matrix Adaptation Evolution Strategy(CMA-ES)作为我们的优化算法，因为它以在解空间高达几千参数时运行良好而著名。我们在一个多CPU核机器上进化C的参数，并行运行多个环境计算轮抽样。更多模型、训练过程和实验设置的信息，见补充材料。  

## 3. 赛车实验：世界模型的特征抽取

　　本节中，我们描述我们是如何训练前述的agent模型解决赛车任务的。据我们所知，我们的agent是第一个解决这个任务的。  

　　帧压缩器V和预测模型M能帮助我们提取有用的空间时间表征。通过使用这些特征作为控制器C的输入，我们能训练一个紧凑的C执行连续控制任务，如在自顶向下的赛车环境CarRacing-v0中从像素输入学习开车。在这些环境中，每次尝试的赛道随机生成，agent的奖励是在最短的时间内看到尽可能多的标识。agent控制3个连续动作：左/右转向，加速和刹车。  

![](/assets/Recurrent_World_Models_Facilitate_Policy_Evolution/Algorithm_1.png)  

　　为了训练V，我们首先收集了10,000个环境计算轮数据集。我们首先让agent随机运动多次探索环境，记录随机动作$a_t$和环境的结果观测。我们使用这个数据集训练VAE来将每帧编码为低维隐变量向量$z$，通过最小化给定帧和解码器从$z$产生的重构帧之间的误差。我们现在使用训练好的V将时间t的每一帧预处理到$z_t$来训练M。使用这个预处理数据，以及记录的随机动作$a_t$，我们的MDN-RNN现在能训练建模混合高斯分布$P(z_{t+1} \mid a_t,z_t,h_t)$。  

　　这个实验中，V和M没有关于环境真实奖励信号的知识。他们的任务是简单的压缩和预测观察的图像序列。只有C能获得环境奖励信息。由于线性C中只有很少的867个参数，进化算法如CMA-ES很适合这个优化任务。  

### 3.1 实验结果

　　**有V无M**  
　　如果我们有一个好的观测表征，训练一个agent开车不是一个困难的任务。之前的工作已经显示，有一组好的关于观测的人工修正信息，如雷达信息，角度，位置和速度，可以很容易的训练一个小前馈网络输入利用这些人工修正信息输出满意的导航策略。由于这个原因，我们首先通过限制C只获得V的信息没有M的信息，来测试agent，所以我们定义控制器为$a_t = W_cz_t + b_c$。  

　　尽管agent仍然能在这个设置下导航到赛道，我们注意到它左右摇晃，并且在尖角处脱离赛道，见图1(右)。这个限制的agent获得平均632+-251得分，与其他OpenAI Gym排行榜上其他agent和传统深度RL方法如A3C的性能相当。增加一个隐藏层到C的策略网络帮助其改进结果到788+-141，但不足以解决这个环境。  

![](/assets/Recurrent_World_Models_Facilitate_Policy_Evolution/Table_1.png)  
图1. CarRacing-v0 100计算轮的结果。  
![](/assets/Recurrent_World_Models_Facilitate_Policy_Evolution/Table_2,png)  
图2. 不同\tau的DoomTakeCover-v0结果。  

　　**世界模型(有V和M)**
　　V产生的表征$z_t$只捕获了时间上一个时刻的表征，没有很多预测能力。相反，M被训练用来做一件事，而且做的相当好，那就是预测$z_{t+1}$。由于M对$z_{t+1}$的预测是从时间$t$时RNN的隐藏状态$h_t$处产生，$h_t$是一个很好的我们能给agent的特征向量选择。综合$z_t$和$h_t$给了C一个很好的当前观测和未来将要发生事件的表征。  

　　我们发现，允许agent获得$z_t$和$h_t$极大改进了它的驾驶能力。驾驶更稳定，agent能有效的跨越尖锐拐角。更进一步，我们看到通过在赛车过程中做这些快速自反驾驶决定，agent不需要提前计划并且运行出未来的假设情景。由于$h_t$包含未来概率分布的信息，agent可以本能的只是重新利用RNN的内部表征来指导它的动作决定。如方程式I赛车手或棒球运动员击打飞速的棒球，agent本能的在恰当的时刻预测何时何地转向。  

　　我们的agent能获得906+-21得分，有效解决了任务并获得新的最新水平。之前的尝试使用深度RL学习获得平均得分591-652，排行榜上最好的报道结果获得平均8838+-11得分。传统深度RL方法通常需要预处理每一帧，如运用边界检测，并堆叠少量最近帧到输入中。相反，我们agent的V和M以原始RGB像素图像流作为输入，直接学习空间-时间表征。据我们所知，我们的方法是第一个解决这个任务的方法。  

　　由于我们的agent的世界模型能建模未来，我们能让它运行在自己的假设赛车环境中。在给定现状态时我们能用它产生$z_{t+1}$的概率分布，抽样$z_{t+1}$并用这个抽样作为真实观测。我们能将训练好的C放回这个生成的环境中。图1(左)显示了一个生成赛车环境的截屏。本文的交互版中包含了这个生成环境的样例。  

## 4. VizDoom实验：在生成环境中学习

　　我们刚刚看到了真实环境中学习的策略似乎是生成环境中什么函数。这引出一个问题——我们能在agent自己生成的环境中训练它，并将策略迁移回真实环境吗？  

　　如果我们的世界模型对于它的目的来讲足够精确，足够问题求解，我们应该能用世界模型取代真实环境。最终，agent步直接观测真实，仅看到世界模型让它看到的。这个试验中，我们在agent自己世界模型生成的环境中训练它，训练用于模仿VizDoom环境的世界模型。在DoomTakeCover-v0中，agent必须学习躲避房间另一边怪物的火球攻击，火球的唯一目的就是杀死agent。累计奖励定义为计算轮中agent存活时间步数。环境的每个计算轮运行最大2100时间步，如果在100个连续计算轮中平均存活时间超过750时间步，任务就被解决了。  

### 4.1 实验设置

　　VizDoom实验设置大体上与赛车任务相同，除了几个关键区别。在赛车任务中，M只训练用来建模下一个z_t。由于我们向建立一个我们能在其内部训练agent的世界模型，这里的M将同时预测下一帧agent会否死亡(以二值时间，$done_t$)，加入下一帧$z_t$中。  

　　由于M能预测done状态加入下一个观测，我们现在有了所有完全模仿DoomTakeCover-v0 RL环境所需要的组件。我们首先通过在M上包装gym.Env接口建立了一个OpenAI Gym环境接口，就像它是一个真的Gym环境，然后在这个虚拟环境中训练agent，而不是使用真实环境。因此在我们的模拟中，生成过程中我们不需要视觉组件V编码任何真实像素帧，所以agent只是完全在更高效的隐环境空间中训练。虚拟和真实环境共享同一接口，所以agent在虚拟环境内学习一个满意的策略后，我们能容易的将这个策略部署回到真实环境中，看这个策略迁移的多好。  

　　这里，基于RNN的世界模型训练用于模仿人类程序员写的完全游戏环境。通过仅从随机计算轮收集的原始图像中学习，它学到如何建模游戏的关键方面，如游戏逻辑，敌人行为，机制，还有3D图形渲染。我们甚至能在生成环境中玩游戏。  

　　然而，不像真实游戏环境，我们注记增加额外不确定性到虚拟环境中是可能的，因此使生成环境中的游戏更有挑战性。我们可以通过在抽样$z_{t+1}$过程中增加温度参数$\tau$来做这件事。通过增加不确定性，生成环境相比真实环境变得更难。相比真实环境，火球可能在更不可预测路径上更随机运动。有时agent可能因为纯属不幸而死亡，没有解释。  

　　训练之后，控制器C学习在虚拟环境中导航，躲避M生成的怪物发射的致命火球。agent在虚拟环境中获得平均918时间步得分。我们获得了在虚拟环境中训练的agent并在原始VizDoom环境中测试其性能。agent获得平均1092时间步得分，远远超过需要的750步，也比在更难的虚拟环境内得分更高。全部结果见表2。  

　　我们见到尽管V不能正确捕获每一帧的全部细节，例如，正确获得怪物数量，C仍然能学习在真实环境中导航。由于虚拟环境甚至不能在第一位置跟踪怪物的准确数量，在嘈杂的不稳定的生成环境中存活的agent能在原始的更干净的环境中繁荣发展。我们也发现在较高温度环境中表现良好的agnet通常在正常环境中表现更好。事实上，增加温度$\tau$阻碍agent利用世界模型的不完美。我们在下节进行深入讨论。  

### 4.2 作弊

　　孩童时期，我们可能鼓励以原始游戏不希望的方式探索视频游戏。玩家发现得到无限生命或血量的方式，利用这些探索，他们能容易的完成一个相当难的游戏。然而，如果这么做，他们就失去了学习游戏希望玩家掌握需要的技能的机会。在我们初始试验中，我们注意到agent发现一个对抗策略，以一种的方式移动，在某些计算轮中使虚拟环境中的M控制的怪物不会发射火球。甚至当有火球形成的标志时，agent以一种方式移动去扑灭火球。  

　　由于M只是环境的近似概率模型，它偶尔会产生步遵循真实环境法则的轨迹。如之前指出，即便是真实环境中房间另一边怪物的数量M也不能精确重新生成。由于这个原因，世界模型被C探索，即便这种探索在真实环境中不存在。  

　　使用M为agent生成虚拟环境的结果，我们也让控制器能获得所有M的隐藏状态。这本质上允许agent访问所有游戏引擎的内部状态和记忆，而只不是玩家看到的游戏观测。因此，agent可以高效探索直接按它的需要利用游戏引擎隐藏状态最大化期望累计奖励。这种在学习好的环境机制模型内部学习策略的方法的弱点是，agent能容易的找到一个对抗策略愚弄环境机制模型——它会发现一个在环境机制模型下看起来不错，但在真实环境中失效的策略，通常因为它访问了模型错误部分的状态，这些错误部分远离训练数据分布。  

　　这个弱点可能是许多之前学习RL环境机制模型的工作实际上并不用模型完全替代真实环境的原因。如[74,75,78]中的M模型，环境机制模型是精确的，如果它不完美，那使它容易被agent利用。使用Bayesian模型，如PILCO，以某方面不确定性评估的形式帮助解决这个问题，然而，他们并未完全解决这个问题。最近的工作将基于模型方法结合传统无模型RL训练，通过首先以学到的策略初始化策略网络，但必须跟随无模型方法在真实环境中微调策略。  

　　为了让C更难利用M的缺陷，我们选择用MDN-RNN作为真实环境可能输出分布的环境机制模型，而不是仅仅预测一个精确的未来。尽管真实环境是精确的，MDN-RNN在效果上将其近似为随机环境。这允许我们在任何环境的更随机版本中训练C——我们只简单调整温度参数\tau控制随机的数量，因而控制真实和可探索性间的权衡。  

　　使用混合Gaussian分布似乎是过度的，VAE编码的隐变量空间只是简单的对角线Gaussian分布。然而，混合密度模型的离散模式对含有随机离散事件的环境是很有用的，如怪物是否决定发射火球或者不动。尽管单一对角Gaussian也许足够编码独立帧，混合密度输出层的RNN使其建模更复杂离散随机状态环境背后的逻辑更简单。  

　　例如，如果我们让温度参数为很低的值$\tau=0.1$，用M有效训练C几乎与精确LSTM训练C一样，生成环境中的怪物不能射击火球，不管agent做什么，因为模式失效。M不能转换到混合Gaussian分布的另一个火球可以形成并发射的模式。在这个生成环境中学习的任何策略，大多数时间都获得2100的满分，但明显将在严格的真实世界现实中失效，甚至不如随机策略。  

　　通过让温度$\tau$是M的可调节参数，我们能看到在不同等级不确定性虚拟环境中训练C的效果，看到他们迁移到真实环境有多好。我们实验了不同的虚拟环境的$\tau$，在这些虚拟环境中训练agent，观察其在真实环境中的性能。  

　　表2中，尽管我们看到增加M的$\tau$使C更难于找到对抗策略，增加太多将是虚拟环境太难而agent学不到任何东西，因此实际上，这是我们能调整的超参数。温度也影响agent发现的策略。例如，尽管$\tau=1.5$时最高得分是1092+-556，增加一点$\tau$到1.30导致较低的得分，但同时得到较低回报方差的更少冒险策略。为了对比，[62]中的最好得分是820+-58。  

## 5. 相关工作

　　有大量学习环境机制模型，以及使用模型训练策略的文献。许多基本概念发源于1980年的前馈神经网络，以及1990年，循环神经网络用于一些关于‘学习思考’的背景工作。更近期的PILCO是概率的基于模型搜索策略方法，解决困难的控制问题。利用环境中收集的数据，PILCO使用Gaussian过程(GP)学习系统机制，并使用这个模型抽样轨迹训练控制器执行想要的任务，如倒立摆。  

　　尽管Gaussian过程在小部分低维数据上运行很好，他们的计算复杂度使其难于扩展到建模高维观测大型历史。其他近期工作使用Bayesian神经网络而不是GP来学习环境机制。这些方法在挑战性的控制任务上证明了有希望的结果，状态定义良好，以及观测相对低维。这里我们感兴趣从高维视觉数据观测建模环境机制，即原始像素帧序列。  

　　在机器人控制应用中，仅从基于摄像头视频输入观测学习系统机制的能力是具有挑战性但很重要的问题。早期的主动视觉RL工作训练前馈神经网络(FNN)输入视频序列的当前图像帧预测下一帧，使用这个预测模型训练中央凹移控制网络在视觉图像上尝试找到目标。为了绕开从高维像素图像训练机制模型的困难，研究者使用神经网络首先学习一个视频帧的压缩表征。最近遵循这点的能使用自编码器的瓶颈层作为低维特征向量训练控制器，从像素控制倒立摆。从压缩隐变量空间学习环境机制是RL算法更数据效率。  

　　视频游戏环境作为新想法的测试平台也在基于模型RL研究中非常流行。之前工作使用前馈卷积网络学习视频游戏的前馈模拟，学习预测环境中不同动作如何影响未来状态对玩游戏的agent非常有用，因为如果agent给定现在状态和动作能预测将来发生什么，他能简单的选择适合目标的最好的动作。这不仅在早起工作中得到证明，在近期几项VizDoom环境比赛的研究中也得到证明。  

　　上述提及的工作使用前馈神经网络(FNN)预测下一视频帧。我们想要使用能捕获更长时间关联的模型。RNN是强力的适合序列建模的模型。使用RNN开发内部模型推断未来，这早在1990年就被探索了，随后进一步在[75, 76, 78]中发展。更近期的工作[83]提出一个统一框架，建立基于RNN的通用问题求解器，能学习环境的世界模型以及学习使用模型推断将来。随后的工作使用基于RNN的模型生成很多未来帧，也作为内部模型推断未来。  

　　本文中，我们使用进化策略(ES)训练控制器，它提供很多收益。例如，我们只需要给优化器提供最终累计奖励，而不是整个奖励历史。ES也很容易并行，我们能加载很多不同解法的rollout实例到很多进程中，并行快速计算一组累计奖励。最近的工作证明在很多基准试验上，ES是传统深度RL方法的可行的替代。在深度RL方法流行前，基于进化的算法已经显示出解决RL问题的高效性。基于进化的算法已经能解决从高维像素输入的困难的RL任务。  

## 6 讨论

　　我们已经证明完全在agent自己模拟隐变量空间世界内训练执行任务是可能的。这个方法提供很多实际收益。例如，视频游戏核心通常需要沉重的计算资源将游戏状态渲染到图像帧，或计算与游戏无关运行。我们也许不想浪费在真实环境中的训练agent循环，而是在其模拟环境中尽可能多的训练agent。被训练逐渐模拟现实的agent迁移其策略到现实世界时被证明是有用的。我们的方法是之前'sim2real'工作的补充。  

　　以VAE实现V并单独训练这个选择也有其局限，由于它可能编码了与任务无关的观测部分。毕竟，根据定义，无监督学习无法知道那些是有用的。例如，VAE在Doom环境中重生成不重要的贴图细节，但在赛车环境中不能重生成任务相关的赛道贴图。通过与预测奖励的M一同训练，VAE能学到关注到图像中任务相关区域，但代价是在新的任务没有重新训练时，我们也许不能高效的重利用VAE。学习任务相关特征也与神经科学有关联。当获得奖励时，初级感觉神经从抑制中释放出来，说明他们大概学到了任务相关特征，而不是任何特征，至少对于成人。  

　　我们的实验中，任务相对简单，合理的世界模型能用随机策略收集的数据进行训练。但如果环境变得复杂呢？在任何复杂环境中，只有部分世界能被agent获取，只有在它学会如何在世界中有策略的导航之后。我们需要agent能探索它的世界，连续收集新观测使它的世界模型能随时间改进修正。未来工作将结合迭代训练流程，控制器主动探索环境的一部分，有益于改进其世界模型。一个令人激动的方向是寻找与人工好奇心和内动力和信息搜索能力结合的方向，鼓励agent新的探索。特别的，我们可以基于压缩质量的改进增加奖励函数。  

　　令一个值得考虑是世界模型能力的局限。尽管现代存储设备能存储大量使用迭代训练流程生成的历史数据，我们的基于LSTM的世界模型也许不能在其权重链接内部存储所有记录信息。尽管人类大脑在某种解析度上能保存几十年甚至上百年的记忆，以反向传播训练的神经网络有更局限的能力，并忍受如灾难性遗忘等问题。未来工作将探索用更高能力的模型替代VAE和MDN-RNN，或者与外记忆模块结合，如果我们想要agent学习探索更复杂世界的话。  

　　像早期的基于RNN的C-M系统，我们的系统一步一步模拟可能的未来时间步，并没有利用类似人类的分层计划或抽象推理，类似人类经常忽略无关的时间空间细节。然而，更广义的'学习思考'方法不限于这种相当朴素的方法。想法，它允许循环的C学习利用循环M的子进程，以任意可计算的方式重用它们解决问题，例如，通过分层计划或其他利用M的类似程序权重矩阵的一部分的方式。一个最近C-M方法的OneBigNet扩展将C和M集中到单一网络，使用类似PowerPlay的行为回放(教师网络的行为压缩到学生网络)避免遗忘旧的预测并在学习新技能时学会控制。这些更通用方法的试验留作将来的工作。  

## 致谢

　　我们感谢Blake Richards, Kory Mathewson, Chris Olah, Kai Arulkumaran, Denny Britz, Kyle McDoald, Ankur Handa, Elwin Ha, Nikhil Thorat, Daniel Smilkov, Alex Graves, Douglas Eck, Mike Schuster, Rajat Monga, Vincent Vanhoucke, Jeff Dean和Natasha Jaques，他们给出了有深度的反馈，并提供了有价值的见解。  

## 参考文献