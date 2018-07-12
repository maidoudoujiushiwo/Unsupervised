# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 10:09:46 2018

@author: jian.wu
E-mail.:fengyuguohou2010@hotmail.com
"""
'''
Life is short, You need Python~
'''
import numpy as np
from numpy import linalg
from numpy.linalg import norm
from scipy.spatial.distance import squareform, pdist

from sklearn.decomposition import NMF  
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA

# We import sklearn.
import sklearn
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale

from time import time
import seaborn as sns


def diss(a,b):
    '''
    define distance
    '''
    return np.sum((a-b)**2,axis=1)

def scatter(x,colors,n):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", n))

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    if type(colors)!=type(None):    
        sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
                        c=palette[colors.astype(np.int)])
    else:
        sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40)
        
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    if type(colors)!=type(None):
        for i in range(n):
            # Position of each label.
            xtext, ytext = np.median(x[colors == i, :], axis=0)
            txt = ax.text(xtext, ytext, str(i), fontsize=24)
            txt.set_path_effects([
                PathEffects.Stroke(linewidth=5, foreground="w"),
                PathEffects.Normal()])
            txts.append(txt)

    return None



class Unsupervised():
    """Unsupervised feature engineering(Kmeans, PCA, NMF and Tsen)
    Parameters
    ----------    
    Data :Import pandas table include the original  features.
    
    y : target, can be None.
    
    col: Name of the original  features.
    
    
    """
    
    def __init__(self,Data,col=None,y=None):        
        self.Data = Data
        self.y = y
        self.col = col
        self.uk_p=None
        self.kpre=None
        self.pca=None
        self.nmf=None
        self.digits_proj=None
        self.tsne_p=None
        self.tpre=None
        
    def _check(self,col,data):
        if col==None:
            raise ValueError('No import features')
        else:
            return data[col].values
    def unkmeans(self):
        """create the kmeans
        """
        X=self._check(self.col,self.Data)
        estimator = KMeans(n_clusters=2)#build Kmeans
        estimator.fit(X)#kmeans
        centroids = estimator.cluster_centers_ #center of it
        self.kpre=estimator.predict(X)
#        ac0,ac1=diss(X,centroids[0]),diss(X,centroids[1])#distance 
#        aw0,aw1=int(np.where(ac0==np.min(ac0))[0]),int(np.where(ac1==np.min(ac1))[0])
        self.uk_p=centroids
        return self
        
    def _var(self,var,Data,col):
        """create the feature
        """
        X=self._check(col,Data)
        var0,var1=diss(X,var[0]),diss(X,var[1])
        return var0,var1
    
    def _unk_var(self):
        """create the feature of kmeans
        """
        if type(self.uk_p)==type(None):
            raise ValueError('NO Kmeans have been built')
        else:
            self.Data['k0'],self.Data['k1']=self._var(self.uk_p,self.Data,self.col) 
        return self
        
    def unk_var(self,Data):
        """create the feature of kmeans
        """
        if type(self.uk_p)==type(None):
            raise ValueError('NO Kmeans have been built')
        else:
            Data['k0'],Data['k1']=self._var(self.uk_p,Data,self.col) 
        return Data
        
    def unPca(self):
        """create the Pca
        """
        X=self._check(self.col,self.Data)
        self.pca = PCA(n_components=2)
        self.pca.fit(X)
        return self
        
    def unNmf(self):
        """create the Nmf
        """

        X=self._check(self.col,self.Data)
        for l in self.col:
            self.Data[l+str('e')]=self.Data[l].apply(np.exp)
        self.coll=[l+str('e') for l in self.col]   
        X=self._check(self.coll,self.Data)
        self.nmf = NMF(n_components=2)
        self.nmf.fit(X)
        return self


    def untsne(self):
        """create the Tsne
        """
        X=self._check(self.col,self.Data)
        self.digits_proj = TSNE(random_state=2018).fit_transform(X)
        estimator = KMeans(n_clusters=2)#构造聚类器
        estimator.fit(self.digits_proj)#聚类
        self.tpre=estimator.predict(self.digits_proj)
        centroids = estimator.cluster_centers_ #获取聚类中心
        ac0,ac1=diss(self.digits_proj,centroids[0]),diss(self.digits_proj,centroids[1])#distance 
        aw0,aw1=int(np.where(ac0==np.min(ac0))[0]),int(np.where(ac1==np.min(ac1))[0])
        self.tsne_p=[X[aw0],X[aw1]]

    def _unt_var(self):
        """create the feature of kmeans
        """
        if type(self.tsne_p)==type(None):
            raise ValueError('NO tsen have been built')
        else:
            self.Data['t0'],self.Data['t1']=self._var(self.tsne_p,self.Data,self.col) 

        return self
        
    def unt_var(self,Data):
        """create the feature of kmeans
        """
        if type(self.tsne_p)==type(None):
            raise ValueError('NO tsen have been built')
        else:
            Data['k0'],Data['k1']=self._var(self.tsne_p,Data,self.col) 
        return Data

    def unsum(self):
        X=self._check(self.col,self.Data)
        kmeans=self.unkmeans()
        pca=self.unPca()
        nmf=self.unNmf()
        tsne=self.untsne()
        self._unk_var()
        self._unt_var()
        X1=self._check(self.coll,self.Data)
        pcae=self.pca.transform(X)
        nmfe=self.nmf.transform(X1)
        self.Data['p0'],self.Data['p1']=pcae[:,0],pcae[:,1]
        self.Data['n0'],self.Data['n1']=nmfe[:,0],nmfe[:,1]
        if type(self.y)==type(None):
            return self.Data[['k0','k1','p0','p1','n0','n1','t0','t1']]
        else:
            return self.Data[['k0','k1','p0','p1','n0','n1','t0','t1',self.y]] 

    def t_plot(self):
        """plot the Tsen with y
        """
         return scatter(self.digits_proj,self.Data[self.y].values,2)

    def t_plot_1(self):
        """plot the Tsen with itself
        """
         return scatter(self.digits_proj,self.tpre,2)
     
    def k_plot(self):
        """plot the kmeans with y
        """
         return scatter(self.Data[['k0','k1']].values,self.Data[self.y].values,2)
     
    def k_plot_1(self):
        """plot the kmeasn with itself
        """
         return scatter(self.Data[['k0','k1']].values,self.kpre,2)
        
        




