# Unsupervised
Unsupervised feature engineering(Kmeans, PCA, NMF and Tsen)
四种无监督变量衍生方法（基于Kmeans, PCA, NMF and T-sen）
简介：介绍四种无监督的变量衍生方法，分别是基于Kmeans、PCA，NMF和T-sen。PCA和NMF相对于对于原始变量的主要因素进行提取，一个是基于主要信息（Pca基于方差），一个是基于矩阵分解的方法。Kmeans大家比较熟悉，传统的聚类方法。T-Sen是降维可视化的方法，本质上达到了聚类的效果，也被我拿来做了无监督衍生。做无监督变量衍生的逻辑是，信用风险，或是欺诈风险较高的客户，实际上的因变量确实和正常客户有所差异。如果是因为业务逻辑或者数据处理的问题导致，分布差异不是由于风险引起的，那么变量衍生效果肯定会受影响。不过借助T-sen非常优良的可视化效果，也可以发现这样的问题。我写了一个python的类，不仅有变量衍生方法，也可以输出图形，考察分布是否出现了差异。代码地址如下：
https://github.com/maidoudoujiushiwo/Unsupervised
	方法介绍
以下介绍基于我代码中的类Unsupervised，当然我所调用的基础类和函数都来源于sklearn。核心思路就是如果直接提取特征的方法（PCA和NMF）就直接提取。聚类方法先提取出聚类的中心，以每一条数据到中心的距离作为一个变量。具体的小trick可以看代码。
主要需要介绍一下T-sen吧，还算比较新的方法。t-分布领域嵌入算法(t-SNE, t-distributed Stochastic Neighbor Embedding )是目前一个非常流行的对高维度数据进行降维的算法, 基础思想由Laurens van der Maaten和 Geoffrey Hinton于2008年提出。经过Maaten和唐建等大神的改造发展，从sne到t-sne，最后到LargeVis，是流形学习，聚类和降维可视化的集大成者。
SNE即stochastic neighbor embedding，是Hinton老人家2002年提出来的一个算法，出发点很简单：在高维空间相似的数据点，映射到低维空间距离也是相似的。常规的做法是用欧式距离表示这种相似性，而SNE把这种距离关系转换为一种条件概率来表示相似性。什么意思呢？考虑高维空间中的两个数据点xi和xj，xi以条件概率pj∣i选择xj作为它的邻近点。考虑以xi为中心点的高斯分布，若xj越靠近xi，则pj∣i越大。反之，若两者相距较远，则pj∣i极小。pi∣j与pj∣i是不相等的，低维空间中qi∣j与qj∣i也是不相等的。所以如果能得出一个更加通用的联合概率分布更加合理，即分别在高维和低维空间构造联合概率分布PP和QQ，使得对任意i,j，均有pij=pji,qij=qji。
我们可以这样定义Pij：
p_ij=((P_(i|j)+P_(j|i)))/2n
再在低维度的分布函数中，用t分布取代高斯分布。用经典的KL距离(Kullback-Leibler Divergence)定义两个分布的差异，最终目标就是对所有数据点最小化这个KL距离，我们可以使用梯度下降算法最小化代价函数就构成了t-sne。
Python中有现成的包可以调用，有非常好的聚类和可视化效果，可以调节困惑度（perplexity）来调节对异常值的敏感度。
下图为对手写数据集的分类，可以说分类效果十分完美了。
 
	效果展示
我用了176个弱变量，每个变量的iv值均小于0.05，单变量Ks值小于0.1。通过无监督衍生之后的结果如下：
所属方法	变量	iv	单变量ks
Nmf	n0	0.473555	0.279049
tsen	t0	0.165742	0.194152
pca	p1	0.156475	0.13215
pca	p0	0.097603	0.107336
kmeans	k1	0.055221	0.100759
kmeans	k0	0.044168	0.086714
tsen	t1	0.030689	0.075811
Nmf	n1	0.009602	0.046287
可以看到衍生的效果还是非常好的。
T-SEN也能一定程度上起到可视化离散数据类别的效果，如图所示：
 
	程序说明
cl=Unsupervised(m1,coonew,'y')
实例化一个对象，其中m1是一个padas的table，coonew是一个存放数据列名的list，‘y‘是目标列名
mce=cl.unsum()
unsum函数调用了完成了所有衍生步骤，返回一个衍生好的数据表。
cl.t_plot()
画出T-SNE生成的数据散点图，	其中0和1是按实际的y标注的。
cl.t_plot_1()
画出T-SNE生成的数据散点图，	其中0和1是按聚类结果标注的。

cl.k_plot()
画出kmeans生成的数据散点图，	其中0和1是按实际的y标注的。
cl.k_plot_1()
画出kmeans生成的数据散点图，	其中0和1是按聚类结果标注的。
