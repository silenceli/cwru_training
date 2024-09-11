# 简介

- 原始数据集来源：https://engineering.case.edu/bearingdatacenter/download-data-file
- 数据集科普文档：http://152.67.113.27/articles/Python%E5%87%AF%E6%96%AF%E8%A5%BF%E5%82%A8%E5%A4%A7%E5%AD%A6%EF%BC%88CWRU%EF%BC%89%E8%BD%B4%E6%89%BF%E6%95%B0%E6%8D%AE%E8%A7%A3%E8%AF%BB%E4%B8%8E%E5%88%86%E7%B1%BB%E5%A4%84%E7%90%86_cwru+bearing+data_10482939_csdn.html

基于 48K 数据进行训练与测试，进行 10 分类，实际使用的是 kaggle 的数据集：https://www.kaggle.com/datasets/brjapon/cwru-bearing-datasets

```
-rw-r--r-- 1 root root  7798480 Sep 11 10:03 B007_1_123.mat
-rw-r--r-- 1 root root  7779920 Sep 11 10:03 B014_1_190.mat
-rw-r--r-- 1 root root  7789200 Sep 11 10:03 B021_1_227.mat
-rw-r--r-- 1 root root  7779920 Sep 11 10:03 IR007_1_110.mat
-rw-r--r-- 1 root root 17849704 Sep 11 10:03 IR014_1_175.mat
-rw-r--r-- 1 root root  7761344 Sep 11 10:03 IR021_1_214.mat
-rw-r--r-- 1 root root  7789200 Sep 11 10:03 OR007_6_1_136.mat
-rw-r--r-- 1 root root  7752064 Sep 11 10:03 OR014_6_1_202.mat
-rw-r--r-- 1 root root  7826336 Sep 11 10:03 OR021_6_1_239.mat
-rw-r--r-- 1 root root  7742720 Sep 11 10:03 Time_Normal_1_098.mat
```


## diy.py

自己按照论文《采用贝叶斯优化和多尺度卷积网络的五相永磁同步电机匝间短路诊断》猜想的代码，准确度只有 0.09

执行方法：

```
python diy.py
```


## rescnn.py

基于 rescnn 来做，其中 rescnn 基于 https://github.com/timeseriesAI/tsai，需要安装其依赖:

```
pip install tsai
```

其结果：avg loss = 1.0815040160869729, eval_acc = 0.7347826086956522，比自己写的可强多了。


执行方法：

```
python rescnn.py
```