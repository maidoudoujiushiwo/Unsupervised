# Unsupervised
Unsupervised feature engineering(Kmeans, PCA, NMF and T-sne)

四种无监督变量衍生方法（基于Kmeans, PCA, NMF and T-sne）
 
	程序说明
cl=Unsupervised(m1,coonew,'y')

实例化一个对象，其中m1是一个padas的table，
coonew是一个存放数据列名的list，‘y‘是目标列名

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
