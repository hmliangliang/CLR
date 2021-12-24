#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2021/12/1 14:33
# @Author  : Liangliang
# @File    : LR.py
# @Software: PyCharm
cmd = "pip install Numba"
cmd1 = "pip install pandas"
import os
os.system(cmd)
os.system(cmd1)

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
import time
import s3fs
#import horovod.tensorflow as hvd


from numba import njit,prange

class Net(keras.Model):
    def __init__(self):
        super(Net,self).__init__()
        self.cov = keras.layers.Dense(2) #输出是一个二分类问题
    def call(self,data):
        h = self.cov(data)
        h = tf.nn.sigmoid(h)
        h = tf.nn.softmax(h) #按行归一化，分配的概率
        h = h[:,1] #输出正类的概率,即类标签为1的概率
        return h

def Loss(y_pred,y_tar):
    '''
    y_pred: 预测为正类的概率
    y_tar: 真实的类标签
    return: 损失函数的LOSS值
    '''
    loss = 0
    for i in range(len(y_pred)):
        loss = loss - y_tar[i]*tf.math.log(y_pred[i])-(1-y_tar[i])*tf.math.log(1-y_pred[i])
    return loss/len(y_pred)

class S3FileSystemPatched(s3fs.S3FileSystem):
    def __init__(self, *k, **kw):
        super(S3FileSystemPatched, self).__init__(*k,
                                                  key=os.environ['AWS_ACCESS_KEY_ID'],
                                                  secret=os.environ['AWS_SECRET_ACCESS_KEY'],
                                                  client_kwargs={'endpoint_url': 'http://' + os.environ['S3_ENDPOINT']},
                                                  **kw
                                                  )
class S3Filewrite:
    def __init__(self, args):
        super(S3Filewrite, self).__init__()
        self.output_path = args.data_output

    def write(self, data, file_idx):
        s3fs.S3FileSystem = S3FileSystemPatched
        fs = s3fs.S3FileSystem()
        start = time.time()

        with fs.open(self.output_path + 'pred_{}.csv'.format(file_idx), mode="w") as resultfile:
            # data = [line.decode('utf8').strip() for line in data.tolist()]
            for i, pred in enumerate(data):
                line = "{},{},{},{}\n".format(pred[0],pred[1],pred[2],pred[3])
                resultfile.write(line)
        cost = time.time() - start
        print("write {} lines with {:.2f}s".format(len(data), cost))

@njit(parallel=True)
def cross_feature(data):
    # 进行特征交叉计算
    d = data.shape[1]
    m = int(d*(d-1)/2)
    cross_data = np.zeros((data.shape[0],m))  # 初始化交叉矩阵
    for i in prange(data.shape[0]):
        for j in prange(d - 1):
            for z in prange(j + 1, d):
                #数据索引下标: int(z-j-1+(2*d-j-1)*j/2)
                cross_data[i,int(z-j-1+(2*d-j-1)*j/2)] = data[i, j] * data[i, z]  # 取特征的值进行交叉
    return cross_data

def train_step(args):
    '''读取s3fs数据部分'''
    path = args.data_input.split(',')[0]
    s3fs.S3FileSystem = S3FileSystemPatched
    fs = s3fs.S3FileSystem()
    # result = []
    """get file list"""
    input_files = sorted([file for file in fs.ls(path) if file.find("part-") != -1])
    print("数据被分割装入{}个子文件.".format(len(input_files)))
    #配置horovod参数
    #hvd.init()
    '''gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')'''
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    #optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr*hvd.size())
    before_loss = 2 ** 31 - 1
    #optimizer = hvd.DistributedOptimizer(optimizer)
    #flag = True  # 主要是为了读取数据的维度
    #first_batch = True #判断数据是否是第一次运行
    net = Net()
    before_net = net
    num = 0
    print("模型已经被构建!")
    for epoch in range(args.epoch):
        num = num + 1
        count = 0
        start = time.time()
        for file in input_files:
            count = count + 1
            print("当前正在处理第{}个epoch的第{}个文件,文件路径:{}......".format(epoch+1,count, "s3://" + file))
            data = tf.convert_to_tensor(pd.read_csv("s3://" + file, sep=',', header=None).values,dtype=tf.float32) # 读取数据
            label = data[:, data.shape[1] - 1]  # 类标签
            data = data[:, 3:data.shape[1] - 1]  # 数据的前列是玩家的openid、roleid、clubid，没有数值大小意义
            data = 1/(1+tf.exp(-data))  # 防止0消除了交叉值的影响
            #进行特征交叉计算
            cross_data = cross_feature(data.numpy())
            cross_data = tf.convert_to_tensor(cross_data, dtype=tf.float32)
            data = tf.concat([data, cross_data], axis=1)
            data = tf.math.l2_normalize(data,0)
            # 定义模型
            dataset = tf.data.Dataset.from_tensor_slices((data, label)).batch(args.batch_size)  # 批处理
            stop_num = 0
            #for epoch in range(args.epoch):#放到外面
            for data, label in dataset:
                #m = data.shape[1] * (data.shape[1] - 1) / 2
                with tf.GradientTape() as tape:
                    predictions = net(data, training=True)
                    loss = Loss(predictions, label)
                #tape = hvd.DistributedGradientTape(tape)
                gradients = tape.gradient(loss, net.trainable_variables)
                optimizer.apply_gradients(zip(gradients, net.trainable_variables))
                ''' if first_batch:
                        hvd.broadcast_variables(net.variables, root_rank=0)
                        hvd.broadcast_variables(optimizer.variables(), root_rank=0)
                        first_batch = False'''
        # early stop机制
        if before_loss > loss:#以最后一个batch的loss为准
            before_loss = loss
            before_net = net
            stop_num = 0
        else:
            stop_num = stop_num + 1
        end = time.time()
        print('Epoch:{} Loss:{}'.format(epoch + 1, loss.numpy()))
        print("The time cost:{}".format(end-start))
        if stop_num > args.stop_num:
            print("Early stopping!")
            break
    # 保存训练模型
    net = before_net
    net.save("./JK_trained_model",save_format="tf")
    cmd = "s3cmd put -r ./JK_trained_model " + args.model_output
    os.system(cmd)
    print("模型保存完毕!")
    return True

def test_step(args,data):
    #装载训练好的模型
    cmd = "s3cmd get -r  " + args.model_output + "JK_trained_model"
    os.system(cmd)
    model = keras.models.load_model("./JK_trained_model",custom_objects={'tf':tf})
    #对输入的结果进行预测
    res = model(data)
    return res

def execute(args):
    #配置环境
    if args.env == "train":
        train_step(args)
    elif args.env == "test":
        #读取测试数据
        '''读取s3fs数据部分'''
        path = args.data_input.split(',')[0]
        s3fs.S3FileSystem = S3FileSystemPatched
        fs = s3fs.S3FileSystem()
        # result = []
        """get file list"""
        input_files = sorted([file for file in fs.ls(path) if file.find("part-") != -1])
        print("数据被分割装入{}个子文件.".format(len(input_files)))
        count = 0
        for file in input_files:
            count = count + 1
            print("当前正在处理{}个文件,文件路径:{}......".format(count, "s3://" + file))
            data = tf.convert_to_tensor(pd.read_csv("s3://" + file, sep=',', header=None).values,
                                        dtype=tf.float32)  # 读取数据
            player_id = data[:, 0:3]  # 保存玩家的openid、roleid、clubid
            data = data[:, 3:data.shape[1]-1]  # 数据的前3列是openid、roleid、clubid，没有数值大小意义
            data = 1 / (1 + tf.exp(-data))  # 防止0消除了交叉值的影响
            # 进行特征交叉计算
            cross_data = cross_feature(data.numpy())
            cross_data = tf.convert_to_tensor(cross_data, dtype=tf.float32)
            print("完成特征交叉计算！")
            data = tf.concat([data, cross_data], axis=1)
            data = tf.math.l2_normalize(data, 0)
            #对输入的数据进行预测
            res = test_step(args,data)
            res = np.concatenate([player_id.numpy(),res.numpy().reshape(len(res),1)],axis=1)#拼接上玩家的id号
            #写入预测的结果
            writer = S3Filewrite(args)
            writer.write(res,count)
        return True
    else:
        raise Exception("输入错误的env参数,env参数为train或test!")