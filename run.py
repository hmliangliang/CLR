#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2021/12/3 10:40
# @Author  : Liangliang
# @File    : run.py
# @Software: PyCharm
import LR
import argparse

if __name__ == "__main__":
    '''配置算法的参数'''
    parser = argparse.ArgumentParser(description='算法的参数')
    parser.add_argument("--env", help="运行的环境(train or test)", type=str, default='train')
    parser.add_argument("--lr", help="学习率", type=float, default=0.001)
    parser.add_argument("--stop_num", help="执行Early Stopping的最低epoch", type=int, default=5)
    parser.add_argument("--epoch", help="迭代次数", type=int, default=50)
    parser.add_argument("--batch_size", help="每一批数据的数目", type=int, default=4096)
    parser.add_argument("--data_input", help="输入数据的位置", type=str, default='')
    parser.add_argument("--data_output", help="数据的输出位置", type=str, default='')
    parser.add_argument("--model_output", help="模型的输出位置", type=str, default='s3:/')
    parser.add_argument("--tb_log_dir", help="日志位置", type=str, default='')
    args = parser.parse_args()
    if args.env == "train":
        result = LR.execute(args)
        if result:
            print("模型训练完成!")
        else:
            print("模型训练或保存失败！")
    elif args.env == "test":
        result = LR.execute(args)
        if result:
            print("结果写入完成!")
        else:
            print("结果写入失败！")
    else:
        print("输入的环境参数错误,env只能为train或test!")