# 1. 我们需要什么样的数据去做市场结构分析

## 1.0 市场结构分析有什么用

分析得到一系列的结构后，可以干嘛。

- 竞品分析：分析哪些类之间存在竞争关系，以及竞争关系初步量化（散点图覆盖大小以及气泡大小）
- 客户忠诚度分析：判断整体客户的忠诚度，主要是对品牌忠诚度的研究。（我们的研究中可能缺乏对具体品牌忠诚度的反应），但是可以对独立站站体本身的忠诚度进行研究。
- 产品级别的相关性分析：从类别到产品，研究具体的产品之间的相关性。

主要是三个维度的分析

具体到独立站本身上，有什么作用。

- 方向指导意义，大类别---》选品
- 产品竞争市场结构的分析为新产品设计和开发，产品竞争性广告，产品定价，产品定位和传播策略提供必不可少的启示和指导作用 [Visualizing Asymmetric Competition Among Mor](https://www.jstor.org/stable/44012167) 基于不同的数据进行不同程度的分析。
- 可以通过分析市场结构，推测在特定市场中的流行产品及竞争对手的形式来选择他们的产品。

### 1.0.1 类别的市场结构分析

==竞品分析==：在产品映射下，不同类别距离越近越具有竞争关系。密集程度体现了竞争影响力大小。

![image-20221020194131071](image-20221020194131071.png)



==客户忠诚度评价==：客户对某一特定产品或服务产生好感，进而重复购买的一种趋向。具体是体现在不同维度上的分析。

![image-20221020195555697](image-20221020195555697.png)

### 1.0.2 产品级别的市场结构

==产品级别的互补分析==

![image-20221020200545407](image-20221020200545407.png)

## 1.1 我们目前有什么数据

['shop_idx', 'url', 'title', 'body_html', 'img', 'publish_date', 'updated_date', 'vendor', 'product_type', 'tags']

分类：

- 商品信息描述：title，product_type，tags
- 时间信息：最后一次购买时间：updated_date     发行时间：publish_date
- 其他信息：vendor，数据展示：img

## 1.2 我们需要什么样的数据

首先，我们的目标是对带有**时间信息**的独立站销售商品的数据进行数据结构分析，想通过图网络的形式进行呈现。希望呈现的效果不要过于单一，即一个商品可以属于**多种类别**，商品与商品之间存在一定的联系，即网络关系。当类别与类别之间的连接较多时，即认为它们具有强**关联性**。

- 时间信息，商品售卖出去的时间，有人买说明对这类的商品存在需求。如果该类的商品需求很大，即在某一段时间内，较多的该类型的产品卖出去了。
- 商品类型，这个是进行图网络的关键信息，没有商品类型信息，则无法判断商品与商品之间的相似性。



注意：其中时间信息并不代表时间序列。如果说先做好了图网络问题，即先将商品进行了相似性的度量，可以将时间信息转化为时间序列进行聚类。

## 1.3 遇到的问题

==问题==： 

- 从数据本身来看，似乎没有理由分析类间关系，因此并不存在类与类之间的信息。首先，并不存在多个类别关联的信息，比如买家数据，日志数据等。
- 其次，我们的数据主要是单个的信息。主要的信息就是对单个商品的描述信息。

关于目前按照顺序采样2000个样本进行实验的其他问题：

1. 数据量不够大或者说titile文字体现不明显，titile里很多文字只出现一次，可能一个title中所有文字只在语料库中出现一次。
2. 经过处理后的title_len  avg = 4.808，语料库的大小为2899，平均每个单词出现次数为3.317。

- 文本较短，提取特征不明显，尽管是进行tf-idf提取还是进行word2vec对单词进行嵌入表达。这都影响后续的图网络效果呈现以及其他方式的嵌入学习。



# 2. 方法

目前想通过图网络先将所有的商品进行一个网络化的连接。

我们进行text classification 的时候，并不需要很强烈的语言的顺序。

有如下几种方式：

[Review of Graph Neural Network in Text Classification](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9666633)



## 2.1 Text GCN

解决的问题：

之前的文本分类只能单独对文本自身的上下文进行语义提取，而不能够对文本之间的相关信息进行表示。[引注](https://blog.csdn.net/qq_36426650/article/details/107838229)

思路：

采用共现词方式，节点数由文本书和不重复单词数决定。节点的边由文章中单词共现和语料库中单词共现构建。边的权重采用的是tf-idf值。信息传递最多两阶。两层GCN架构。

$$ A_{ij}=\left\{
\begin{aligned}
PMI(i,j) & \ i,j \ are \ words, PMI(i,j)>0\   \\
TF-IDF_{ij} & \ i\ is\ document, j  \ is \ word \\
1 & \ i = j \\0 & \ otherwise \\
\end{aligned}
\right.$$

问题：

没有考虑到单词的顺序。可以考虑注意力机制来应对长尾数据。

## 2.2 Text Level GNN for Text Classification

构造TextGCN时，边的权重时固定的（单词节点间的边权重是两个单词的PMI，文档-单词节点间的边权重是TF-IDF），限制了边的表达能力，且无法为新样本进行在线测试。

- 为每个输入文本单独构建一个图，文本中的单词作为节点



## 2.3 MAGNET： Multi-Label Text Classification Using Attention-based GNN

解决的问题：为语料库中的文档分配不同的标签

为了更好的捕捉标签之间的联系，基于注意力的图神经网络（MAGNET），基于特征矩阵自动学习标签之间的关系。



## 2.4 Short Text

短的文本经常出现在推特、聊天信息、商品描述、在线评论等。

**痛点问题**：所研究的短文本缺乏在较长句子和段落中普遍存在的结构，因而认为传统的文本分类技术无法推广到短文本中。因此认为单纯的文本信息不足够进行text classification，因而引入了更多的信息，如用户查询日志，元数据等信息。

存在难点：

- 短文本高度稀疏，缺乏足够的特征来提供足够的词共现。
- 与其他的文本资源不同，**大多数短文本语料库没有语言结构或遵循语法结构。**

解决方法：

- side-information： 从用户的行为信息获得，insight来自用户之间的行为相似性。

### 2.4.1 数据描述

首先，数据都来自**amazon.com**

1. 当做==Product Query Classification== 时，有来自从亚马逊搜索引擎的匿名**用户查询**的日志信息，对于两个查询$i$ 和 $j$, 邻接矩阵$A$ 的构造是 $A_{ij} = \# common\  purchase\  b/w \ query \ i \ and \ query \ j $

2. 当作==Product Title Classification==时，数据包括product titles， metadata for each product （also bought， also viewed， bought together， buy after viewing） and their categories。对于每个产品，它的类别是从颗粒度标签到细粒度标签的路径。(e.g.: Electronics ⇒ Computers & Accessories ⇒ Cables & Accessories)。 图的构造使用的是共同浏览（co-viewed），背后的原因是相较于共同购买，共同查看不一定会买，共同购买则更倾向于查看很多个商品，会忽略掉很多的信息。

![image-20221010154800992](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20221010154800992.png)

![image-20221010154828321](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20221010154828321.png)







表达节点信息：

1. bag-of-words
2. neural network

且这篇文章中所用到的数据集都是带有label且edge是完整的，而我们所研究的实际场景中需要自己去构建edge以及no label。



# 3. 我们需要做的内容

**最终目标**：对短文本进行聚类。

**中间过程**：

1. 可以对短文本概念化，生成文本的概念分布
2. 可以学习短文本嵌入表示



- 首先因为我们是短文本的数据，那么我们需要去观测我们短文本的特征，平均值，最大值，最小值，方差等。作为我们介绍数据的一个依据。以及我们包含的单词个数。
  - samples | vocabulary size | average length of sample（words）
- 其次，我们需要说明我们的文本语料库缺乏语言结构，[Short Text Classification in Twitter to Improve Information Filtering](https://dl.acm.org/doi/abs/10.1145/1835449.1835643) 。
- 同一类别的产品在标题文本上具有很丰富的多样性！因此，产品类型在文本特征空间中可能没有紧凑的表示。



==word2vec== 可以实现我们no label的学习。 

## 3.1 词的表示

### 3.1.1 one-hot encoding

维度较高，且词与词之间相互独立。

### 3.1.2 词的分布式表示

==CBOW==：

insight： 认为词的语义是由器上下文决定。 e.g  "the cat sits one the mat"， 在训练时，将"the cat sits one the"作为输入，预测最后一个词"mat", 可以采用LSTM方法。

### 3.1.3 词嵌入

核心：上下文的表示以及上下文与目标词之间的关系建模。

相较于3.1.1：

- 将vector 从整形转化为浮点型，变为整个实数范围的表示。

- 可以将稀疏的巨大维度压缩嵌入到一个更小的维度空间。

## 3.2 如何做

1. word2vec

- CBOW
- Skip-gram

上下文预测当前值以及当前值预测上下文相结合的形式

2. leverage conceptualization for short-text embedding

- 根据短文本的概念分布$\theta_C$ 生成该短文本的概念向量
- ![image-20221012132112627](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20221012132112627.png)

3. weighted word embedding aggregation

- 为文本中的每一个单词分配权重，根据单词的idf值分配。 [Representation learning for very short texts using weighted word embedding](https://reader.elsevier.com/reader/sd/pii/S0167865516301362?token=4C77FA694434CF27DA022640E27848876FF3706886158946A1CF59538DBCFA32E6F3E0363DC22D0B640696B997A80165&originRegion=us-east-1&originCreation=20221013071148)
- isf嵌入法，和tf-idf加权平均法类似。https://aclanthology.org/Q16-1028.pdf

4. graph2vec

- 将单词与单词之间的连接关系用tf-idf或者其他的形式表示，构造成一张图网络，然后运用图神经网络算法生成node embedding。

## 3.3 对比方法

baseline 与传统的nlp技术对比：

- bag of words
- n-grams
- tf-idf
- LDA

## 3.4  无监督方法

**K-means聚类**，可以多采取一些变种，先看看数据特征分布情况，再选择合适的无监督方式。

k-means依赖于初始值，因此可以进行多次实验，平均归一化互信息分数。

**计算互信息**：

互信息表达的是一个随机变量中包含的关于另一个随机变量的信息量。可以用于度量聚类结果的**相似程度**。

设两个随机变量$(X,Y)$的联合分布为$p(x,y)$， 边缘分布为$p(x),\ p(y)$, 互信息$I(X;Y) = \sum_x \sum_y p(x,y)log{p(x,y) \over p(x)p(y)}$ 



## 3.5 评价标准











#  4. fashionnova baseline





爬取过程中遇到的问题，访问过于频繁，且未采用分布式爬虫（没有可靠的ip池）



重新爬取了数据，虽然没有完全爬下来，但是比上次的数据集更丰富，且更集中。

不重复数据量：92816

fashionnova 数据爬虫流程图：

![Untitled](Untitled.svg)



数据分布情况：

|      |     product_type     | proportion |
| :--: | :------------------: | :--------: |
|  0   |       Dresses        |   0.151    |
|  1   |      Knit Tops       |   0.064    |
|  2   |    Matching Sets     |   0.054    |
|  3   |        Shoes         |   0.051    |
|  4   |        Jeans         |   0.047    |
|  5   |       Jewelry        |   0.047    |
|  6   |   Shirts & Blouses   |   0.042    |
|  7   |     Graphic Tees     |   0.039    |
|  8   |      Jumpsuits       |   0.036    |
|  9   | Lingerie & Sleepwear |   0.033    |
|  10  |      Bodysuits       |   0.032    |
|  11  |       Swimwear       |   0.031    |
|  12  |        others        |   0.373    |

粗糙处理 body_html  -- > body_text： 

总共的单词数量是84204

词频统计：有247个单词词频超过1w

[('in', 106834), ('available', 91669), (',', 77077), ('and', 64223), ('to', 42431), ('polyester', 39656), ('black', 34949), ('5', 25320), ('the', 23124), ('spandeximported', 20605)]

有68685个单词词频小于5.



|      |     interval     | word_num |
| :--: | :--------------: | :------: |
|  0   |     [0, 100)     |  82887   |
|  1   |    [100, 200)    |   559    |
|  2   |    [200, 300)    |   172    |
|  3   |    [300, 400)    |   102    |
|  4   |    [400, 500)    |    65    |
| ...  |       ...        |   ...    |
| 1064 | [106400, 106500) |    0     |
| 1065 | [106500, 106600) |    0     |
| 1066 | [106600, 106700) |    0     |
| 1067 | [106700, 106800) |    0     |
| 1068 | [106800, 106900) |    1     |





大部分词频都是集中在[0,100]  



body_text 单词长度统计：

min_len : 0 

max_len : 570 

average_len : 18.940505947250475



## 实验

采取三种类型样本 Dresses  Shoes  Jeans，进行分析



样本数量：23110

[('Dresses', 14024), ('Shoes', 4710), ('Jeans', 4376)]

从里面分层抽样，三个类别都抽样200个样本

样本数量更小一点





8个类别 各随机取8个样本，总共64个样本  构造边  tf-idf > 0.01  共 984条边， modularity 0.328，共 7 个 community  这个可以调。

![1110_2](1110_2.png)

- Jewelry 权重较小，且与其他类的关联小，因此可以视为较独立的类别
- 

------

##  Parameters: 

Randomize:  On
Use edge weights:  On
Resolution:  1.0



##  Results: 

Modularity: 0.357
Modularity with resolution: 0.357
Number of Communities: 4

![img](data:image/png;base64,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)

![image-20221103221319858](image-20221103221319858.png)





![force_atlas2](force_atlas2.png)



将节点模拟成原子，通过模拟原子之间的力场来计算节点之间的关系

Force Altas

- 惯性：值越大，图形的摇摆幅度越大
- 斥力强度：没有给节点排斥其他节点的强度，值越大，节点之间的距离越大
- 吸引强度：连接节点之间的吸引力的强度，值越大，有连接的节点越被拉近
- 重力：值越小，图越分散
- 速度：布局的速度，值越大，图布局的速度越快













爬取数据量：19w+

实际数据量： 50260

class ： product_type



jsonEncoderError   TimeOutError

![image-20221025163032105](image-20221025163032105.png)![image-20221025163112814](image-20221025163112814.png)



统计：distribution

body_html 处理过程




![image-20221028132832111](image-20221028132832111.png)

![image-20221029145101845](image-20221029145101845.png)



节点： Modularity class

大小： 度

标签颜色： Modularity class



![image-20221028140345237](image-20221028140345237.png)

 预设比例 ： 22% 

![image-20221028133858134](image-20221028133858134.png)

![image-20221028134137188](image-20221028134137188.png)







This page is temporarily unavailable because a device from your location is sending large amounts of

即访问太过频繁。

解决方法： 代理池，降低访问频率













# 5.看的文献主要内容

P2V: 作者提出了新的探索性方法来分析市场结构，分析类别内替换关系以及跨类别的产品互补性。

- 先聚类，聚类观察**类别**之间的相似性。首先，类别就需要清晰的对类进行抽象，而不是对产品抽象。此时，类别数量可以多一点，因为此时我们需要做的是观察类别之间的关系，将一个类别抽象为一个点。最好是生成一个矩阵热力图的格式，方便对类与类之间进行分析。
- **产品级**抽象，即对每个产品进行聚类显示下来，目的是观察再某一个特定类别中产品之间的相似性以及差异性。这个样本可以少一点，取三个类别进行研究分析。

新的产品的向量是基于已经学习到的产品之间的线性结合

![image-20221107210730471](image-20221107210730471.png)

这就是对向量进行加减表示。认为商品是由几个属性组成的，每个属性代表一个向量。





另外一个方式是：找一些确实很靠近的样本，进行显示。抽样的时候进行特别处理，只抽样那些能够很好的区分开或者能够聚合比较清晰的样本。

![image-20221107211858431](image-20221107211858431.png)