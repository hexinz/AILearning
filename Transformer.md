#  Transformer

## Transformer基础

### Tokenization

**Position Encoding**

### Attention

**Self-attention和Masked multi-head Self-attention：**

视频资料：[Transformer_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1JE411g7XF/?p=54)

Self-attention用于encoder，通过学习MQ、MK、MV三个权重矩阵，生成query、key、value三个向量，并通过对这三个向量的向量内积、归一化等矩阵计算（QKT/d \*V），获得考虑了与所有的输入向量相关性的输出向量。更详细地说，QKT的结果为每个字对其他字的注意力权重，/d为归一化，*V为用每个字的权重对每个字的特征进行加权求和。

Masked multi-head Self-attention用于decoder，考虑所有已经生成的字，和原句中的字，输出下一个要生成的字。Masked用于训练时不让decoder看到未生成的字，multi-head用于切割出不同维度的MQ、MK、MV矩阵，每一组MQ、MK、MV矩阵分别计算，使有的关注local信息，有的关注相距更远的词的关系，最后合并起来。

关于MHA(multi-head attention)分析有意思的文章：[Multi-Head-Attention的作用到底是什么 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/626820422)，主要观点是Roberta模型中有些head是没有学到词法语法句法信息的，即使打乱顺序大部分head的attention不变，可以理解训练的过程是为了把这些有用的head抽取出来，而把无用的head剪枝掉，Head不是越多越好。层数也不是越多越好，层数越多，head之间的variance越小，但我们实际希望的是head能捕捉不同的pattern（热力图越分散越好，说明学到了除位置信息以外的语法语义信息），层数过少，有些head attention在前面几层没发生变化，但可能会在别的层发生变化，表示这个head的信息被其他层捕捉到，层数太少可能会失去这部分信息。

**MQA(Multi Query Attention)和GQA(Grouped Query Attention)：**

MQA(Multi Query Attention)，即多个query head共享相同的key和value。Llama3使用MQA。

GQA(Grouped Query Attention)，即query head再分组，组内的head共享相同的key和value。使相同模型质量下，生成的性能提升。

**Causal Attention** **因果注意力**

<https://blog.csdn.net/qinduohao333/article/details/133875973>

 

### Norm

<https://www.bilibili.com/video/BV1ah4bebEuG?p=23&vd_source=e02eb333558cc48e8fe990317a39b738>

residual就是把原向量和经过self-attention后的向量相加。

layer norm是对自身向量取平均和方差，batch norm是对所有向量相同dim取平均和方差，使用前者效果更好。原因在于batch norm对小批量且有统计特征的数据效果佳，而对输入序列长度和分布变化大的NLP任务则无法有效处理。同时，layer norm在输入序列长度不一，数量不一的情况下稳定性更好。Norm可以加在residual后，也可以加在attention/FNN之前，加在之前效果更好。Llama还使用了RMS Norm。

### FFN

<https://cloud.baidu.com/qianfandev/topic/269647>

Transformer增加Feed Forward Neural Network是为了进行非线性的映射。attention层都是线性变化，ReLU增加非线性，且对剪枝有效；GeLU介于sigmoid和ReLU之间，但是指数计算使LLM速度变慢。

**Gated MLP**新增GRU(Gated Recurrent Unit)层，使用类LSTM的机制更准确地建模复杂非线性关系，但计算量比LSTM小。

**MOE**(Mixture of Expert)

## Transformer的优化

### Prompting

[9. 【生成式AI】Finetuning vs. Prompting：對於大型語言模型的不同期待所衍生的兩類使用方式 (2_3)_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1ka4y1g7To?p=8) from 李宏毅

Finetuning是通过微调对预训练模型做改造，需要保存多个模型，而通过插件（Adapter： <https://adapterhub.ml>）对预训练模型做改造，无需保存多个模型，仅需保存插件本身，减小内存占用。Prompting不改变模型本身。

**In-context Learning**

从范例中学习

**Instruction Learning**

通过任务叙述或指示做出相应的任务

- Instruction-tuning：请做翻译(train) -> 请做摘要(test) e.g.: T0，FLAN(Finetuned Language Net，Premise，hypothesis，options)

**Chain of Thought**

[一文读懂：大模型思维链 CoT（Chain of Thought） - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/670907685)

更详细地prompt，数学问题给出推理步骤，o1草莓模型

**Soft prompt**

train向量作为prompt，adapter的特例

### Sampling

投机采样

### 模型剪枝（Pruning）

<https://cloud.baidu.com/qianfandev/topic/269647>

结构化剪枝的原理是删除连接或层结构；非结构化剪枝的基本原理是将阈值以下的参数归零，但可能导致稀疏，需要压缩。

总体方法：把不重要的weight或neuron删除再train。

### 知识蒸馏（Knowledge Distillation）

方法：用学好的大model来教小model做好任务。

### 模型量化（Quantization）

方法：把浮点bit压缩成更小的单位。可以对train好的model用也可以边train边quantize。

**量化算术：**

官方：[IEEE 754-1985 - Wikipedia](https://en.wikipedia.org/wiki/IEEE_754-1985)

实现：<https://blog.csdn.net/weixin_58275336/article/details/136738605>

FP32规格数计算公式

![img](file:///C:/Users/Z00830~1/AppData/Local/Temp/msohtmlclip1/01/clip_image002.gif)(十进制)

![https://i-blog.csdnimg.cn/blog_migrate/22fa777d5c9da548112a95fcac8001aa.png](file:///C:/Users/Z00830~1/AppData/Local/Temp/msohtmlclip1/01/clip_image004.gif)（FP16）

非规格数：指数位全为0，NPU等会被转换为0。

特殊数：exponent全为1，fraction全为0，根据sign判断是正或负无穷大；如果fraction不为0，则为NaN。

sign 为符号位 s，占 1 bit，用来表示正负号；exponent 为指数偏移值 k，占 8 bits，用来表示其是 2 的多少次幂；fraction 是分数值（有效数字） M，占 23 bits，用来表示该浮点数的数值大小。正浮点数最小值为1*2-126 = 1.1755*10-38；最大值为2127 *(1+(223-1)/ 223)= 3.4028*10-38。加法器：1.对阶，将两个小数的exponent化为相同的，小阶化大阶，尾数右移；2.尾数相加；3.化为FP16标准； 乘法器：1.阶数相加；2.尾数相乘；3.化为FP16标准。

例子：0.75 = 1.1(2) * 2-1，0.1875 = 1.1(2) * 2-3，相加=(1.1+0.011)* 2-1=(1+7/8) * 2-1=0.9375。

将 FP32 值域 [−1,1] 映射到 INT8 值域 [0,255]例子：

![img](file:///C:/Users/Z00830~1/AppData/Local/Temp/msohtmlclip1/01/clip_image006.gif)

[一文搞懂模型量化算法 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/505570612)

 

**训练后量化（PTQ – Post-Training Quantization****）**

PTQ是一种将已经训练好的模型权重直接转换为低精度数据的方法，无需重新训练。这种方法简单易行，但可能会因精度降低而稍微影响模型性能。常见的PTQ工具包括Auto-GPTQ等，它们支持多种模型和量化精度。

GPTQ是一种动态量化方法，它通过在训练过程中逐渐增加量化的严格程度来实现量化。与传统的静态量化方法不同，GPTQ允许模型在量化过程中调整其权重，以最小化量化带来的性能损失。

 

**量化感知训练（QAT – Quantization-Aware Training****）**

与PTQ不同，QAT在训练阶段或微调阶段就进行权重转换，以更好地适应低精度表示。QAT通常会带来更好的模型性能，但需要更多的计算资源。QLoRA是一种广泛使用的QAT技术，它采用混合量化方案，将权重量化为低精度整数，同时保留激活值为较高精度的浮点数。

### 模型设计（Architecture Design）

方法：由于模型的稀疏性，用更少的参数、用更小的Layer替换原有的。Low rank proximation也属于这类。

**PCA和Auto-encoder**

[Unsupervised Learning - Auto-encoder_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1JE411g7XF?spm_id_from=333.788.player.switch&vd_source=e02eb333558cc48e8fe990317a39b738&p=58)

Src->Encoder->latent representation/ latent code/ Embedding->Decoder->target

**LLE和t-SNE**

[Unsupervised Learning - Neighbor Embedding_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1JE411g7XF?spm_id_from=333.788.videopod.episodes&p=57&vd_source=e02eb333558cc48e8fe990317a39b738)

LLE和t-SNE属于Manifold Learning 流形学习 – 降维范畴。

LLE：Locally Linear Embedding，保持邻接关系的前提下把图从高维画到低维，选取的neighbor k不能太大也不能太小，即使不知道原来的维度，只要知道邻接关系就可以做。

t-SNE：T-distributed Stochastic Neighbor Embedding

LLE让两个相近的点靠近，t-SNE不仅让两个相近的点靠近，也让两个距离远的点分开

**低轶分解（Low-Rank Factorization）**

启发：When adapting to a specific task, Aghajanyan et al. (2020) shows that the pre-trained language models have a low “instrisic dimension” and can still learn efficiently despite a random projection to a smaller subspace. 在执行一个任务时，对LLM的参数降维仍然有效果，说明LLM的参数很可能在执行一个任务时是稀疏/低轶的。

MLA(Multi-Head Latent Attention)通过低秩联合压缩key和value来减少kv cache。

LoRA(Low-Rank Adaptation of Large Language Models)是一种adapter，通过在transformer每一层注入可训练的秩分解矩阵（rank decomposition matrices Wd×k+Bd×rAr×kα），微调整个模型的参数。

 

### KV cache

KV cache基础图解：

<https://blog.csdn.net/ningyanggege/article/details/134564203>

KV cache变种及公式计算：

<https://zhuanlan.zhihu.com/p/697311739?utm_psn=1785776639344525312>

Transformer中的decoder使用masked self-attention，会关注生成词之前的所有词，如果每次都对每个之前的token计算是非常耗时的，且每个已生成的token的attention不变。如果每次把已生成的key和value储存起来，那么仅需计算新的query向量的attention，能使速度从序列长度的平方O(n2)提升为线性O(n)。

包含KV cache的生成大模型的推理过程分为两步，1) prefill：输入一个prompt序列，为每个decoder中的attention层生成KV cache，同时输出第一个token，2) decode：把每轮新产生的key，value缓存，逐个生成token。KV cache的大小为：层数×head数量×head维度×位宽/8×2 Byte。

使用GQA，KV cache的大小变为：层数×group数量×head维度×位宽/8×2 Byte。以Llama3 8B模型为例，大小为：32×8(8个group)×128×16/8×2 = 131072Byte，如果全部生成上下文为8192个token，需要1G的缓存。

YoCo(You can only cache once)使KV层间共享，可以在GQA的基础上再次减少缓存大小。

## Transformer实例

### Llama3

官方手册：<https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md>

### Qwen2