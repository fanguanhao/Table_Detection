import cv2
import numpy as np
import cv2 as cv
import base64,time
import os
from math import *
from seal_test import demo2,calculateRotate
from Not_line_Table import demo1,demo3
import traceback
import logging

logging.basicConfig(filename='Table_Detection_log.log',
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',)


"""
基于之前的版本对应解决扫描件。
扩充此版本针对发票解决自然场景及文本扭曲(非倾斜)
"""

test_image_path = "ima_test/"

class table_check(object):
    mini_table_state = True  # 小表格横线合并
    __instance = None

    def __new__(cls):
        if not cls.__instance:
            cls.__instance = object.__new__(cls)
        return cls.__instance

    def __init__(self):
        self.width = 0   #图像的宽度
        self.height = 0  #图像的高度
        self.new_width = 1800  #归一化的图像宽度
        self.new_height = 2400  # 归一化的图像宽度
        self.coo_list = [] #横线的两个端点
        self.feature_value = 1.06 #模板预定义特征值
        self.shape_test_image = None #复制了一张图像的尺寸
        self.bool_table = True  #判断表格是否能够闭合
        self.table_enpoint = [] #无修改的表格4个顶点 基于两条竖线的端点
        self.table_enpoint_fill = []  #扩充表格轮廓
        self.w_ro = 0  #用于还原使用的宽度比例
        self.h_ro = 0 #用于还原使用的高度比例
        self.table_width = []
        self.table_height = []
        self.img = None
        self.newimage = None
        self.ut = False  #是否使用自测模式
        self.not_line = False  #检测的直线都太短，疑似无线表


    #纯numpy检测线段信息
    def numpy_lins(self,thled1,line=5):
        """
        直接利用numpy计算线段
        :param thled1: 二值化图像
        :param line: 线段长度
        :return: 初始版横竖线段图
        """
        th1 = thled1.copy()
        th2 = thled1.copy()
        # 处理横向的像素
        max_lings = line
        lines_b = max_lings
        new_shape = int((self.new_height * self.new_width) / max_lings)
        row_lings = th1.reshape(new_shape, max_lings)
        row_sum = np.sum(row_lings, axis=1)
        row_sums = np.array([row_sum for i in range(max_lings)])
        row_T = row_sums.T
        row_lings[row_T >= lines_b] = 255
        row_lings[row_T < lines_b] = 0
        # new_row_lings = row_lings.reshape((h,w))
        new_row_lings = row_lings.reshape((self.new_height, self.new_width))

        # 处理竖向像素
        max_lings = line  # 初始点段长度
        lines_b = max_lings
        clos_lins = th2.T
        clos_lins = clos_lins.reshape(new_shape, max_lings)
        clos_sum = np.sum(clos_lins, axis=1)
        clos_sums = np.array([clos_sum for i in range(max_lings)])
        clos_T = clos_sums.T
        clos_lins[clos_T >= lines_b] = 255
        clos_lins[clos_T < lines_b] = 0
        # new_clo_lings = clos_lins.reshape((w,h))
        new_clo_lings = clos_lins.reshape((self.new_width, self.new_height))
        new_clo_lins = new_clo_lings.T

        # ------------------------to list-------------------
        new_clo_list = new_clo_lins.tolist()
        new_row_list = new_row_lings.tolist()
        # ------------------------to array------------------
        new_row = np.array(new_row_list, np.uint8)
        new_clo = np.array(new_clo_list, np.uint8)
        kernel_clo = np.ones((2, 1), np.uint8)
        kernel_row = np.ones((1, 2), np.uint8)
        new_clo = cv2.dilate(new_clo, kernel_clo, iterations=2)
        new_row = cv2.dilate(new_row, kernel_row, iterations=2)
        return new_row,new_clo

    #整理线段信息
    def new_lins_np(self,thled1,open_row,open_clo,line=5):

        # end_time5 = time.time()
        # new_row, new_clo = self.numpy_lins(thled1,line)
        # end_time6 = time.time()
        # print("基础线段执行时间",end_time6-end_time5)
        new_row = open_row
        new_clo = open_clo
        # cv2.imwrite("ima_test/new_row.jpg",new_row)
        # cv2.imwrite("ima_test/new_clo.jpg", new_clo)
        # -----------------------------find long lines--------------------------------
        kernel_clo = np.ones((50, 1), np.uint8)
        dilate_clo_long = cv2.dilate(new_clo, kernel_clo, iterations=2)
        conts = self.find_long_lines(dilate_clo_long)
        image_longcont = np.ones(new_clo.shape,np.uint8)
        cv.drawContours(image_longcont, [cont.T for cont in conts], -1, 255, 3)
        image_longcont[image_longcont > 1] = 0
        #-----------------------------------判断线段大小---------------------------
        image_c, clo_lin_length,clo_lines_message = self.class_tabel(new_clo,long_map=image_longcont)
        ########  校验是否有表格线
        image_r, row_lin_length,row_lines_message = self.class_tabel(new_row, state='row')
        clo_line_length = [mes[0][2] for mes in clo_lines_message]
        row_line_length = [ mes[0][2] for mes in row_lines_message]
        clo_line_length = sorted(clo_line_length)[::-1]
        row_line_length = sorted(row_line_length)[::-1]
        clo_leng_mean = np.mean(clo_line_length[:5]) if len(clo_line_length) > 5 else np.mean(clo_line_length)
        row_leng_mean = np.mean(row_line_length[:5]) if len(row_line_length) > 5 else np.mean(row_line_length)

        if clo_leng_mean <150 and row_leng_mean<150:
            self.not_line = True
            return None, None
        elif clo_leng_mean<100 or row_leng_mean<100:
            self.not_line = True
            return None, None


        # cv2.imwrite("ima_test/image_c.jpg", image_c)
        # cv2.imwrite("ima_test/image_r.jpg", image_r)
        new_contours_clo_class = np.zeros_like(image_c)
        new_contours_row_class = np.zeros_like(image_c)

        self.shape_test_image = image_c.shape
        # sta_time = time.time()
        test_image_row, test_image_clo = self.repair_lines(image_c,row_lines_message,clo_lines_message)
        # end_time2 = time.time()
        # print("线段连接时间",end_time2 - sta_time)
        new_contours_row, new_contours_clo,clo_lines_weidth = self.Cell_refactor(test_image_row,test_image_clo)
        self.count_point(new_contours_row,new_contours_clo)
        # end_time3 = time.time()
        # print("单元格重构时间",end_time3 - end_time2)
        self.bool_table  = self.table_is_close(new_contours_row,clo_lines_weidth)
        if self.bool_table:
            print("表格可以闭合")
        else:
            print("疑似三线表")
        for ke,va in new_contours_clo.items():
            cv.drawContours(new_contours_clo_class, va[1], -1, 255, 2)
        for ke,va in new_contours_row.items():
            cv.drawContours(new_contours_row_class, va[1], -1, 255, 2)

        #对其进行膨胀和腐蚀
        kernel_clo = np.ones((50, 1), np.uint8)
        new_contours_clo_class = cv2.dilate(new_contours_clo_class, kernel_clo, iterations=2)
        kernel_clo = np.ones((1, 50), np.uint8)
        new_contours_row_class = cv2.dilate(new_contours_row_class, kernel_clo, iterations=2)
        kernel_clo = np.ones((30, 2), np.uint8)
        new_contours_clo_class1 = cv2.erode(new_contours_clo_class, kernel_clo, iterations=2)
        kernel_clo = np.ones((2, 10), np.uint8)
        new_contours_row_class1 = cv2.erode(new_contours_row_class, kernel_clo, iterations=2)
        # cv2.imwrite("ima_test/new_contours_row_class.jpg", new_contours_row_class)
        # cv2.imwrite("ima_test/new_contours_clo_class.jpg", new_contours_clo_class)
        # cv2.imwrite("ima_test/new_contours_add_class.jpg", new_contours_clo_class + new_contours_row_class)
        cv2.imwrite("ima_test/new_contours_add1_class.jpg", new_contours_clo_class1+new_contours_row_class1)
        # end_time4 = time.time()
        # print("判断膨胀写入时间:", end_time4 - end_time3)
        return new_contours_row_class1,new_contours_clo_class1

    #查找异常长直线
    def find_long_lines(self,image):
        image, contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        maxh = image.shape[0]
        maxw = image.shape[1]
        conts = []
        for cont in contours:
            cont_t = cont.T
            max_y,min_y = np.max(cont_t[1]),np.min(cont_t[1])
            max_x,min_x = np.max(cont_t[0]),np.min(cont_t[0])
            if max_y >= maxh-15 and min_y<=15:
                conts.append(cont_t)
        return conts

    #线段修复
    def repair_lines(self,image_c,row_lines_message,clo_lines_message,):
        test_image_clo = np.zeros_like(image_c)
        test_image_row = np.zeros_like(image_c)
        test_image_clo1 = np.zeros_like(image_c)
        test_image_row1 = np.zeros_like(image_c)
        cost_x = 100
        cost_y = 10
        for messages in clo_lines_message:
            start_point = messages[0][0][0]
            end_point = messages[0][1][0]
            line_len = messages[0][2]
            if line_len < 50: continue
            cv2.line(test_image_clo, tuple(start_point), tuple(end_point), (255, 255, 255), 2)
            cv2.line(test_image_clo1, tuple(start_point), tuple(end_point), (255, 255, 255), 2)
            start_point = np.array(start_point)
            end_point = np.array(end_point)
            if line_len > 50:
                #线段没有更新只能连接一次
                for fitting in clo_lines_message:
                    # if fitting[0][2] < 50: continue
                    start_point_V = np.array(fitting[0][0][0])
                    end_point_V = np.array(fitting[0][1][0])
                    point_cost1 = end_point - start_point_V  # 两个线段的两个端点的xy的差值
                    point_cost2 = start_point - end_point_V  # 两个线段的两个端点的xy的差值
                    if abs(point_cost1[0]) < cost_y and abs(point_cost1[1]) <= cost_x:
                        # 条件成立线段可以进行连接
                        cv2.line(test_image_clo, tuple(end_point), tuple(start_point_V), (255, 255, 255), 2)
                        # end_point = end_point_V
                    elif abs(point_cost2[0]) < cost_y and abs(point_cost2[1]) <= cost_x:
                        # start_point = start_point_V
                        cv2.line(test_image_clo, tuple(start_point), tuple(end_point_V), (255, 255, 255), 2)


        cost_x = 200
        cost_y = 5
        for messages in row_lines_message:
            start_point = messages[0][0][0]
            end_point = messages[0][1][0]
            line_len = messages[0][2]
            # if line_len < 50: continue
            # cv2.line(test_image_row, tuple(start_point), tuple(end_point), (255, 255, 255), 2)
            cv2.line(test_image_row1, tuple(start_point), tuple(end_point), (255, 255, 255), 2)
            cv.drawContours(test_image_row, [messages[1]], -1, 255, 2)
            if line_len > 50:
                for fitting in row_lines_message:
                    # if fitting[0][2] < 50: continue
                    start_point_V = np.array(fitting[0][0][0])
                    end_point_V = np.array(fitting[0][1][0])
                    point_cost1 = end_point - start_point_V  # 两个线段的两个端点的xy的差值
                    point_cost2 = start_point - end_point_V  # 两个线段的两个端点的xy的差值
                    if abs(point_cost1[0]) < cost_x and abs(point_cost1[1]) <= cost_y:
                        # 条件成立线段可以进行连接
                        cv2.line(test_image_row, tuple(end_point), tuple(start_point_V), (255, 255, 255), 2)
                        # end_point = end_point_V
                    elif abs(point_cost2[0]) < cost_x and abs(point_cost2[1]) <= cost_y:
                        cv2.line(test_image_row, tuple(start_point), tuple(end_point_V), (255, 255, 255), 2)
                        # start_point = start_point_V

        cv2.imwrite(test_image_path + "test_image_clo.jpg", test_image_clo)
        cv2.imwrite(test_image_path + "test_image_row.jpg", test_image_row)
        cv2.imwrite(test_image_path + "test_image_clo1.jpg", test_image_clo1)
        cv2.imwrite(test_image_path + "test_image_row1.jpg", test_image_row1)
        cv2.imwrite(test_image_path + "test_image_adds.jpg", test_image_row + test_image_clo)
        return test_image_row,test_image_clo

    #单元格重构
    def Cell_refactor(self,lines_row,lines_clo):
        _, contours_row, _ = cv2.findContours(lines_row, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        _, contours_clo, _ = cv2.findContours(lines_clo, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        new_contours_row = {} #收集线段的类别信息,key:分类线段均值长度,va[0]:线段端点，va[1]:线段轮廓点集,va[2]:线段的详细长度
        new_contours_clo = {} #收集线段的类别信息,key:分类线段均值长度,va[0]:线段端点，va[1]:线段轮廓点集,va[2]:线段的详细长度
        clo_lines_x = {} #收集竖线之间的X轴距离，用于后面判断是否是三线表
        new_contours_row_list = []
        new_contours_clo_list = []
        w_b = False
        h_b = False
        #横线的线段分类，去掉短的线段
        for cont in contours_row:
            cont_T = cont.T
            max_x,min_x = np.max(cont_T[0]),np.min(cont_T[0])
            weights = max_x - min_x
            id_maxlis = np.where(cont_T[0] == max_x)
            id_minlis = np.where(cont_T[0] == min_x)
            max_endpoint = cont[id_maxlis[1][0]][0]  # 线段开始的端点(最大X值)
            min_endpoint = cont[id_minlis[1][0]][0]  # 线段结束的端点(最小X值)
            if weights <50:  #排除短线段
                continue
            self.coo_list.append((max_endpoint, min_endpoint))
            if weights in new_contours_row.keys():
                new_contours_row[weights][0].append([max_endpoint,min_endpoint])
                new_contours_row[weights][1].append(cont)
                new_contours_row[weights][2].append(weights)
                continue
            for w in new_contours_row.keys():
                if abs(w - weights) < 100:
                    new_contours_row[w][0].append([max_endpoint, min_endpoint])
                    new_contours_row[w][1].append(cont)
                    new_contours_row[w][2].append(weights)
                    w_b = True
                    break
            if w_b:
                w_b = False
                continue
            else:
                new_contours_row[weights] = [[[max_endpoint,min_endpoint]],[cont],[weights]]
                new_contours_row_list.append(weights)
        #线段的分类key值为线段类别长度的均值
        for i in  new_contours_row_list:
            row_weidth_mean = int(np.mean(new_contours_row[i][2]))
            new_contours_row[row_weidth_mean] = new_contours_row.pop(i)


        #竖线的线段分类，去掉短的线段
        for cont in contours_clo:
            cont_T = cont.T
            max_y, min_y = np.max(cont_T[1]), np.min(cont_T[1])
            hights = max_y - min_y
            if hights < 40:  # 排除短线段
                continue
            id_maxlis = np.where(cont_T[1] == max_y)
            id_minlis = np.where(cont_T[1] == min_y)
            max_endpoint = cont[id_maxlis[1][0]][0]  # 线段开始的端点(最大X值)
            min_endpoint = cont[id_minlis[1][0]][0]  # 线段结束的端点(最小X值)
            #遍历收集每两条直线之间的距离
            for cont1 in contours_clo:
                """
                计算每条横线段端点之间的距离,
                存在封闭的表格的时候,竖线之间的距离应当与横线的长度差不多
                """
                cont_T1 = cont1.T
                max_y1, min_y1 = np.max(cont_T1[1]), np.min(cont_T1[1])
                hights1 = max_y1 - min_y1
                y_cost = (abs(max_y1- max_y) + abs(min_y1 - min_y))/2
                if hights < 200 :break
                if hights1 <200:continue  #计算表格两边的竖线不考虑过短的竖线
                if y_cost > 200 :continue  #判断两条线的长度差值，大于阈值则任务不属于同一表格内的竖线
                id_maxlis1 = np.where(cont_T1[1] == max_y1)
                id_minlis1 = np.where(cont_T1[1] == min_y1)
                max_endpoint1 = cont1[id_maxlis1[1][0]][0]  # 线段开始的端点(最大Y值)
                min_endpoint1 = cont1[id_minlis1[1][0]][0]  # 线段结束的端点(最小Y值)
                weights = ((max_endpoint - max_endpoint1)+(min_endpoint - min_endpoint1))/2
                weights = abs(weights[0])
                if weights <300:continue
                clo_lines_x[weights] = [[max_endpoint,min_endpoint],[max_endpoint1,min_endpoint1]]
            if hights in new_contours_clo.keys():
                new_contours_clo[hights][0].append([max_endpoint,min_endpoint])
                new_contours_clo[hights][1].append(cont)
                new_contours_clo[hights][2].append(hights)
                continue
            for h in new_contours_clo_list:
                if abs(h - hights)<100:
                    new_contours_clo[h][0].append([max_endpoint,min_endpoint])
                    new_contours_clo[h][1].append(cont)
                    new_contours_clo[h][2].append(hights)
                    h_b = True
                    break
            if h_b:
                h_b = False
                continue
            else:
                new_contours_clo[hights] = [[[max_endpoint,min_endpoint]],[cont],[hights]]
                new_contours_clo_list.append(hights)
        for i in new_contours_clo_list:
            clo_hidth_mean = int(np.mean(new_contours_clo[i][2]))
            new_contours_clo[clo_hidth_mean] = new_contours_clo.pop(i)


        # print(new_contours_clo.keys(),sum(new_contours_clo.keys())/len(new_contours_clo.keys()))
        # print(new_contours_row.keys(),sum(new_contours_row.keys())/len(new_contours_row.keys()))
        # print(clo_lines_x.keys())
        return new_contours_row,new_contours_clo,clo_lines_x

    #保留发票长线段，计算交点
    def count_point(self,new_contours_row,new_contours_clo):
        clo_mean = sum(new_contours_clo.keys())/len(new_contours_clo.keys())
        row_mean = sum(new_contours_row.keys())/len(new_contours_row.keys())
        test_clo_image = np.zeros(self.shape_test_image)
        test_row_image = np.zeros(self.shape_test_image)
        # clos = rows = 0
        for ke,va in new_contours_clo.items():
            if ke > clo_mean/2:
                cv.drawContours(test_clo_image, va[1], -1, 255, 2)
                # clos += 1
        for ke,va in new_contours_row.items():
            if ke > row_mean/2:
                cv.drawContours(test_row_image, va[1], -1, 255, 2)
                # rows += 1
        cv2.imwrite("ima_test/test_clo_image.jpg",test_clo_image)
        cv2.imwrite("ima_test/test_row_image.jpg", test_row_image)
        cv2.imwrite("ima_test/table.jpg", test_row_image+test_clo_image)

    #判断表格是否能够闭合
    def table_is_close(self,row_lines,clo_lines_weidth):
        """
        检测发票的表格是否有闭合,判断横线的两边是否有长直线，竖线之间的距离是否接近横线的长度
        :param row_lines: 横线的信息 ke:线段的均值长度,va[0]:线段的端点，va[1]:线段的点集,va[2]:每条线段的长度
        :param clo_lines_weidth: 竖线之间的距离,ke:竖线之间的距离，va[0]:两条线段的端点
        :return: True  or   Flase  表格是否能闭合
        """
        te_iamge = np.zeros(self.shape_test_image)
        for clo_w_ke,clo_va in clo_lines_weidth.items():
            if clo_w_ke < 500 :continue
            num = 0
            for row_w_ke,row_va in row_lines.items():
                # print(clo_w_ke - row_w_ke)
                if abs(clo_w_ke - row_w_ke) < 200: #竖线之间的距离与横线的长度容错小于100

                    #疑似可能是横线两端的竖线，进行端点验证
                    #第一根线x均值
                    clo_start_point = (clo_va[0][0][0] + clo_va[0][1][0])/2
                    clo_max_y = clo_va[0][0][1]
                    clo_min_y = clo_va[0][1][1]
                    #第二根线x均值
                    clo_start_point1 = (clo_va[1][0][0] + clo_va[1][1][0])/2
                    clo_max_y1 = clo_va[1][0][1]
                    clo_min_y1 = clo_va[1][1][1]
                    clo_max_x = max(clo_start_point,clo_start_point1)
                    clo_min_x = min(clo_start_point,clo_start_point1)
                    row_num = len(row_va[0])
                    for row_enpoints in row_va[0]:
                        row_start_point = row_enpoints[0][0]
                        row_end_point = row_enpoints[1][0]
                        row_y = (row_enpoints[0][1]+row_enpoints[1][1])/2
                        #横线均值y比竖线的极值y 相差300 说明不在一个表格内
                        if (abs(row_y - clo_max_y)<200 and abs(row_y - clo_max_y1)<200)or(abs(row_y - clo_min_y)<200 and abs(row_y - clo_min_y1)<200):

                            if abs(clo_max_x - row_start_point) < 100 and abs(clo_min_x - row_end_point) < 100:
                                num += 1
                                # print(clo_va[0],clo_va[1],row_enpoints[0],row_enpoints[1])
                    if num >= 2:
                        # print(len(row_va),num)
                        if clo_va[0][0][0] >clo_va[1][1][0]:
                            x1 = clo_va[1][1].tolist()
                            x2 = clo_va[0][1].tolist()
                            x3 = clo_va[0][0].tolist()
                            x4 = clo_va[1][0].tolist()
                        else:
                            x1 = clo_va[0][1].tolist()
                            x2 = clo_va[1][1].tolist()
                            x3 = clo_va[1][0].tolist()
                            x4 = clo_va[0][0].tolist()
                        self.table_enpoint.append([x1[0],x1[1],x2[0],x2[1],x3[0],x3[1],x4[0],x4[1]])
        if len(self.table_enpoint) > 0:
            return True
        else:
            return False

    #分类大小表格
    def class_tabel(self,lin_image, long_map=None,state = 'clo'):
        """
        判断筛选线段表格
        :param lin_image: 直线图(竖线或直线)
        :param state: 处理线段的状态：clo is clos lings,row is rows
        :param w_path: 写入图片路径便于查看lings
        :param long_map:长直线的图，全值为1，线段位置像素为0
        :return: 筛选的线段图，最大数量的线段长度
        """
        image, contours, hierarchy = cv2.findContours(lin_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        image_zero_clo = np.zeros(image.shape,np.uint8)
        image_zero_row = np.zeros(image.shape, np.uint8)
        cont_class_lis = []
        cont_class_dict = {}
        lines_message = []  #收集每条线段信息 [[起始点，结束点，长度],[点集]]
        cont_class_lis.append(0)
        cont_class_dict[0] = []
        max_n = 0
        lin_length = 0
        if state =='clo':
            for i in contours:
                cont_T = i.T
                cont_hight = np.max(cont_T[1]) - np.min(cont_T[1])
                cont_wight = np.max(cont_T[0]) - np.min(cont_T[0])
                if cont_hight < 80:
                    continue
                if cont_hight < 200 and cont_wight > 30:
                    continue
                max_y, min_y = np.max(cont_T[1]), np.min(cont_T[1])
                id_maxlis = np.where(cont_T[1] == max_y)
                id_minlis = np.where(cont_T[1] == min_y)
                max_endpoint = i[id_maxlis[1][0]]  # 线段开始的端点(最大Y值)
                min_endpoint = i[id_minlis[1][0]]  # 线段结束的端点(最小Y值)
                # print(max_y,min_y,max_endpoint,min_endpoint,id_maxlis,id_minlis)
                lines_message.append([[max_endpoint, min_endpoint, cont_hight], i])
                for j in cont_class_lis: #线段分类 容错小于50同属一类线段
                    if abs(j - cont_hight) < 50:
                        cont_class_dict[j].append(i)
                        break
                    if j == cont_class_lis[-1]:
                        cont_class_lis.append(cont_hight)
                        cont_class_dict[cont_hight] = [i]
            cont_class_lis = cont_class_lis[1:]
            # cont_class_dict.pop(0)
            for ke,va in cont_class_dict.items():
                if len(va) >max_n:
                    max_n = len(va)
                    lin_length = ke
                cv.drawContours(image_zero_clo, va, -1, 255, 2)
            image_zero_clo *= long_map
            return image_zero_clo,lin_length,lines_message
        else:
            new_cotours = []
            for i in contours:
                cont_T = i.T
                cont_T1 = cont_T.copy()
                cont_wight = np.max(cont_T[0]) - np.min(cont_T[0])
                cont_hight = np.max(cont_T[1]) - np.min(cont_T[1])
                if cont_wight < 50:
                    continue
                if cont_wight<200 and cont_hight>50:
                    continue
                max_x, min_x = np.max(cont_T[0]), np.min(cont_T[0])
                id_maxlis = np.where(cont_T[0] == max_x)
                id_minlis = np.where(cont_T[0] == min_x)
                max_endpoint = i[id_maxlis[1][0]]  # 线段开始的端点(最大X值),只取第一个
                min_endpoint = i[id_minlis[1][0]]  # 线段结束的端点(最小X值),只取第一个
                lines_message.append([[max_endpoint, min_endpoint, cont_wight], i])

                if np.max(cont_T[0])>= image.shape[1] - 10 or np.min(cont_T[1]) <= 10 :continue
                for j in cont_class_lis:
                    if abs(j - cont_wight) < 50:
                        cont_class_dict[j].append(i)
                        break
                    if j == cont_class_lis[-1]:
                        cont_class_lis.append(cont_wight)
                        cont_class_dict[cont_wight] = [i]

            for ke,va in cont_class_dict.items():
                cv.drawContours(image_zero_row, va, -1, 255, 2)
                if len(va) >max_n:
                    max_n = len(va)
                    lin_length = ke
            return image_zero_row,lin_length,lines_message

    #有线表格的信息提取
    def table_lines_message(self,):
        newimage, img = self.newimage, self.img
        MIN_H = 10
        MIN_W = 50
        MAX_H = int(img.shape[0] / 2)
        MAX_W = int(img.shape[1] / 2)
        cells, table = self.find_(newimage, img, MIN_H, MAX_H, MAX_W, MIN_W, name='JD_NEW')
        boxs = self.boxs_class(cells)
        w_ro = self.w_ro
        h_ro = self.h_ro
        new_ro = np.array([w_ro, h_ro])
        for ke, va in boxs.items():
            for i in range(len(va)):
                if len(va[i]) > 0:
                    np_box = np.array(va[i])
                    new_box = (np_box * new_ro).astype(np.int32).tolist()
                    va[i] = new_box
        for ke, va in cells.items():
            for i in range(len(va)):
                if len(va[i]) > 0:
                    np_box = np.array(va[i])
                    new_box = (np_box * new_ro).astype(np.int32).tolist()
                    va[i] = new_box
        new_ro1 = np.concatenate([new_ro,new_ro,new_ro,new_ro],0)  #连接比例，2->8 维度 便于8个点直接计算
        self.table_enpoint = (np.array(self.table_enpoint) * new_ro1).astype(np.int32).tolist()
        self.fitting_tanles_enpoint()  #扩充表格外轮廓
        return boxs, cells

    #无线表格的信息提取
    def table_message(self):
        w_ro, h_ro = self.w_ro,self.h_ro
        feature_value = self.feature_value  # 特征值
        cool_list = []
        # 根据图像尺寸比例缩放端点，还原图像的线段端点
        for i in self.coo_list:
            start, ends = i
            new_start = (int(start[0] * w_ro), int(start[1] * h_ro))
            new_ends = (int(ends[0] * w_ro), int(ends[1] * h_ro))
            cool_list.append((new_start, new_ends))
        self.coo_list = cool_list

        # 根据特征值查找同一张发票上的两根直线
        collect_endpoint = self.row_line_endpoint(feature_value)
        tables, tables_nums, table = self.fitting_tables(collect_endpoint)
        for va in table.values():
            self.table_enpoint.append([va[0][0],va[0][1],va[1][0],va[1][1],va[2][0],va[2][1],va[3][0],va[3][1]])
        # print('--',collect_endpoint)
        self.fitting_tanles_enpoint()  # 扩充表格外轮廓
        return {}, collect_endpoint

    #表格检测
    def table_detection(self,img,w_path = 'ima_test/'):
        self.table_enpoint = []
        self.img = img
        if img.shape[0] < img.shape[1]:
            self.new_width = 2400
            self.new_height = 1800
        MIN_H = 10
        MIN_W = 50
        MAX_H = int(img.shape[0] / 2)
        MAX_W = int(img.shape[1]/2)
        self.height, self.width = img.shape[:2]
        self.w_ro = self.width/self.new_width
        self.h_ro = self.height/self.new_height
        image = cv2.resize(img, (self.new_width, self.new_height))

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # 锐化
        # dst = self.image_custom("", gray)
        # gray = image[:,:,2]

        # ret, thled1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        thled1 = cv2.adaptiveThreshold(gray, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 5)
        # thled2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 33, 5)
        # 预处理膨胀一下使线段连接起来
        kern = np.ones((3, 3), np.uint8)
        dilate_row = cv2.dilate(thled1, kern)
        erode = cv2.erode(dilate_row, kern)
        thled1 = erode
        thled2 = erode.copy()
        thled2[thled2 > 0] = 255

        ####### 开操作检测
        # end_time4 = time.time()
        row_image,clo_image = self.open_demo(thled2.copy())
        # end_time5 = time.time()
        # print("开操作检测时间:", end_time5 - end_time4)

        ####### numpy运算检测
        rows, clos = self.new_lins_np(thled1,row_image,clo_image,line=10)
        self.newimage = clos+rows
        # end_time6 = time.time()
        # print("numpy运算检测时间:", end_time6 - end_time5)
        # cells, table = self.find_(newimage, img, MIN_H, MAX_H, MAX_W, MIN_W, name='JD_NEW')
        # boxs = self.boxs_class(cells)
        #坐标还原的尺寸比例
        self.w_ro = self.width / self.new_width
        self.h_ro = self.height / self.new_height

        # if not self.bool_table:
        #     print("无闭合单元格")
        #     return self.table_message()
        # else:
        #     return self.table_lines_message()

    #自测表格检测
    def table_detection_test(self,img):
        self.img = img
        if img.shape[0] < img.shape[1]:
            self.new_width = 2400
            self.new_height = 1800
        MIN_H = 10
        MIN_W = 50
        MAX_H = int(img.shape[0] / 2)
        MAX_W = int(img.shape[1]/2)
        self.height, self.width = img.shape[:2]
        self.w_ro = self.width/self.new_width
        self.h_ro = self.height/self.new_height
        image = cv2.resize(img, (self.new_width, self.new_height))

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # 锐化
        # dst = self.image_custom("", gray)
        # gray = image[:,:,2]

        # ret, thled1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        thled1 = cv2.adaptiveThreshold(gray, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 5)
        # thled2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 33, 5)
        # 预处理膨胀一下使线段连接起来
        kern = np.ones((3, 3), np.uint8)
        dilate_row = cv2.dilate(thled1, kern)
        erode = cv2.erode(dilate_row, kern)
        thled1 = erode
        thled2 = erode.copy()
        thled2[thled2 > 0] = 255

        ####### 开操作检测
        # end_time4 = time.time()
        row_image,clo_image = self.open_demo(thled2.copy())
        # end_time5 = time.time()
        # print("开操作检测时间:", end_time5 - end_time4)

        ####### numpy运算检测
        rows, clos = self.new_lins_np(thled1,row_image,clo_image,line=10)
        if self.not_line:
            print("开始无线表检测")
            # demo1(img)
            row_lins,clo_lines = demo3(image,self.img.shape)
            return row_lins,clo_lines

        self.newimage = clos+rows
        # end_time6 = time.time()
        # print("numpy运算检测时间:", end_time6 - end_time5)

        # cells, table = self.find_(newimage, img, MIN_H, MAX_H, MAX_W, MIN_W, name='JD_NEW')
        # boxs = self.boxs_class(cells)
        #坐标还原的尺寸比例
        self.w_ro = self.width / self.new_width
        self.h_ro = self.height / self.new_height

        if not self.bool_table:
            print("无闭合单元格")
            return self.table_message()
        else:
            return self.table_lines_message()

    # 单元格分类
    def boxs_class(self, cells):
        boxs = {}
        new_r = []
        for ke, va in cells.items():
            new_va = []
            va = va[::-1]
            for j, (i) in enumerate(va):
                if j == 0:
                    n = i
                    continue
                if abs(n[0][1] - i[1][1]) < 30 and abs(n[0][0] - i[1][0]) < 150:
                    # print(abs(n[0][1] - i[1][1]),abs(n[0][0] - i[1][0]),n[0],i[1])
                    new_r.append(n)
                    n = i
                    if j == len(va) - 1:
                        new_r.append(n)
                else:
                    if j == len(va) - 1:
                        new_va.append(new_r)
                        new_r = []
                        new_r.append(i)
                    new_r.append(n)
                    new_va.append(new_r)
                    n = i
                    # print(new_r)
                    new_r = []
            boxs[ke] = new_va
        return boxs

    #拟合线段端点
    def row_line_endpoint(self,feature_value):
        """
        根据特征值定位在同一张发票上的两根直线的端点
        :param feater_value: 只有横线的特征值，特征值 = 两条直线之距离/该线段的长度
        :return: 同一张发票上的两根横线的端点
        """
        endpoint = []
        collect_endpoint = {}
        n = 0
        for j, (i) in enumerate(self.coo_list):
            (x1, y1), (x2, y2) = i
            w = x1 - x2
            if w < 300: continue

            # if [[x1,y1],[x2,y1],[x2,y2],[x1,y2]] in endpoint: continue
            end_state = False
            if len(collect_endpoint)!=0:
                for end_p in collect_endpoint.values():
                    if [[x1,y1],[x2,y1],[x2,y2],[x1,y2]] in end_p:
                        end_state = True
                        break
            if end_state:
                continue
            for line in self.coo_list:
                (x1_, y1_), (x2_, y2_) = line
                w_ = x1_ - x2_
                if w_ < 200 or i == line: continue
                if abs(x1 - x1_) < 100 and abs(w - w_) < 100:
                    # print((x1, y1, x2, y2), (x1_, y1_, x2_, y2_), abs(y1 - y1_) / w, w, w_)
                    if abs((abs(y1 - y1_) / w) - feature_value) <= 0.01:
                        endpoint = []
                        endpoint.append([[x1,y1],[x2,y1],[x2,y2],[x1,y2]])
                        endpoint.append([[x1_,y1_],[x2_,y1_],[x2_,y2_],[x1_,y2_]])
                        collect_endpoint[n] = endpoint
                        n += 1
                        print((x1, y1, x2, y2), (x1_, y1_, x2_, y2_), abs(y1-y1_)/w, w, w_)
        return collect_endpoint

    # 查找单元格
    def find_(self, image, images, MIN_H, MAX_H, MAX_W, MIN_W, paths='ima_test/', name='', ):
        ima, contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        w_ = images.shape[1]
        h_ = images.shape[0]
        draw_img3 = images.copy()
        table_dict = {}  # 表格的框
        cells_dict = {}  # 单元格的框
        cells_idx = {} #单元格的编号
        hier_state = False #单元格的关系验证
        for j, (i) in enumerate(contours):
            iT = i.T
            max_x, min_x = np.max(iT[0]), np.min(iT[0])
            max_y, min_y = np.max(iT[1]), np.min(iT[1])
            if max_x - min_x < 100 or max_y - min_y < 20:
                continue
            if max_x >= image.shape[1] - 20 or max_y >= image.shape[0] - 20:  # 右下线段贴合边缘
                continue
            if min_x <= 15 or min_y <= 15:  # 左上线段贴合边缘
                continue
            rect = cv2.minAreaRect(i)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            boxs = self.sortd_box_coords(box)
            hier = hierarchy[0][j]
            if hier[-1] == -1:
                table_dict[hier[2] - 1] = boxs
            else:
                # if hier[-1] - 1 in cells_dict.keys(): #异常包含的单元格
                #     continue
                if hier[-1] in cells_dict.keys():
                    cells_dict[hier[-1]].append(boxs)
                    cells_idx[hier[-1]].append(hier[1]+1)
                else:
                    cells_dict[hier[-1]] = [boxs]
                    cells_idx[hier[-1]] = [hier[1]+1]

        # print(cells_dict)
        # 验证嵌套单元格并去除
        nest_box = []#内嵌box
        lack_box = []#box数量不合格
        for ke, vals in cells_idx.items():
            if len(vals) <4:
                lack_box.append(ke)
                continue
            for key in cells_dict.keys():
                if key in vals:
                    nest_box.append(key)
        # print(cells_dict)
        for i in lack_box: #清理数量不够的分类
            cells_dict.pop(i)
        # print(cells_dict)
        if len(nest_box)>2:
            for i in nest_box: #清理内嵌分类
                if i in cells_dict.keys():
                    cells_dict.pop(i)

        id_dict = []
        # cv2.namedWindow("draw_img3",cv2.WINDOW_NORMAL)
        # cv2.imshow("draw_img3",draw_img3)
        # 根据分类后大小关系(表格内含有包含关系被单独分类出来)剔除表格内包含关系的单元格

        tables, tables_nums, table = self.fitting_tables(cells_dict)
        for ke, va in tables_nums.items():
            weight, height, y, a = va
            if weight < 150 or height < 150:
                cells_dict.pop(ke)

        return cells_dict, table_dict

    #计算表格外轮廓的位置
    def fitting_tables(self,boxs):
        """
        计算表格的外轮廓
        :param boxs: 表格的所有单元格
        :param image: 原图
        :return: tables：填充好的表格外轮廓坐标，tables_num:表格的一些信息[宽度，高度，表头比例，其余位置扩充比例],table:未填充的表格坐标
        """
        # print(image.shape)
        tables = {}
        tables_nums = {}
        table = {}
        w = self.width
        h = self.height

        for ke,va in boxs.items():
            va = np.array(va)

            va_T = va.T
            max_x,min_x = np.max(va_T[0]),np.min(va_T[0])
            max_y,min_y = np.max(va_T[1]),np.min(va_T[1])
            #根据极值生成的表格轮廓
            table[ke] = [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]]
            weight = max_x - min_x
            height = max_y - min_y
            y,a = 2,8
            y_up = int(height/ 2) #表头上抬表格竖线的1/2 的长度
            a_up = int(height / 8)  # 其余位置扩充表格竖线的1/8 长度
            min_x = min_x - a_up if min_x - a_up > 0 else 0
            min_y = min_y - y_up if min_y - y_up > 0 else 0  #表头位置抬起的高一些
            max_x = (max_x + a_up) if (max_x + a_up) < w else w
            max_y = (max_y + a_up) if (max_y + a_up) < h else h
            tables[ke] = [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]]
            tables_nums[ke] =[weight, height, y, a]
        return tables,tables_nums,table

    def fitting_tanles_enpoint(self):
        if len(self.table_enpoint)>0:
            table_enpoint = self.table_enpoint
            for table in table_enpoint:
                length = np.sqrt((table[0]-table[6])**2 + (table[1] - table[7])**2)
                self.table_width.append(np.sqrt((table[0]-table[2])**2 + (table[1] - table[3])**2))
                self.table_height.append(length)
                up_ = int(length/3)  #根据表格的高度取比例值
                rest = int(length/8)
                # if table[1] - up_ < 0 and table[3] - up_ < 0:
                #     ##异常的外轮廓，疑似纸张外轮廓
                #     self.table_enpoint.remove(table)
                #     continue
                new_enpoint = [table[0]-rest, table[1]-up_,
                               table[2]+rest, table[3]-up_,
                               table[4]+rest, table[5]+rest,
                               table[6]-rest, table[7]+rest ]
                self.table_enpoint_fill.append(new_enpoint)

    def cv_imread(self,file_path):
        cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
        return cv_img

    def open_demo(self,binary):
        # #开操作检测直线
        # gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        # # ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
        # binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 5)
        row_b = binary.copy()
        clo_b = binary.copy()
        # print('ret:', ret)
        # cv.imshow('binary:', binary)
        # cv.imwrite("ima_test/binary.jpg",binary)
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 10 ))
        kerne2 = cv.getStructuringElement(cv.MORPH_RECT, (10, 1))
        # open
        open_c = cv.morphologyEx(clo_b, cv.MORPH_OPEN, kernel)
        open_r = cv.morphologyEx(row_b, cv.MORPH_OPEN, kerne2)
        kern_clo = np.ones((5, 1), np.uint8)
        kern_row = np.ones((1, 5), np.uint8)
        dilate_clo = cv2.dilate(open_c, kern_clo)
        erode_clo = cv2.erode(dilate_clo,kern_clo,iterations=2)
        cv.imwrite("ima_test/open_clo_erode.jpg", erode_clo)
        dilate_row = cv2.dilate(open_r, kern_row)
        erode_row = cv2.erode(dilate_row, kern_row, iterations=2)
        cv.imwrite("ima_test/open_row_erode.jpg", erode_row)
        return erode_row,erode_clo

    #图像预处理 模糊降噪
    def image_disposes(self,im_path,w_path = 'ima_test/'):
        image = self.cv_imread(im_path)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        dst = cv2.GaussianBlur(gray, (7, 7), 15)
        img_mean = cv2.blur(dst, (5, 5))
        img_Guassian = cv2.GaussianBlur(dst, (5, 5), 0)
        img_median = cv2.medianBlur(dst, 5)
        img_bilater = cv2.bilateralFilter(dst, 9, 75, 75)
        thled1 = cv2.adaptiveThreshold(img_mean, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 55, 5)


        cv2.namedWindow("gray",cv2.WINDOW_NORMAL)
        cv2.namedWindow("thled1", cv2.WINDOW_NORMAL)
        cv2.namedWindow("dst", cv2.WINDOW_NORMAL)
        cv2.namedWindow("coble", cv2.WINDOW_NORMAL)
        cv2.imshow("gray", gray)
        cv2.imshow("thled1", thled1)
        cv2.imshow("dst", dst)
        cv2.imshow("coble", img_mean)
    #锐化
    def image_custom(self,img_path ,ima=None):
        if ima is None:
            image = self.cv_imread(img_path)
        else:
            image = ima
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # 锐化
        dst = cv.filter2D(image, -1, kernel=kernel)
        return dst

    #模糊
    def image_blur(self, img_path):
        image = self.cv_imread(img_path)
        dst = cv.blur(image, (15, 1))
        cv.imshow("blur_demo", dst)
        cv.waitKey()
        return dst

    def base64_to_tableDetecton(self,base_data):
        img_data = base64.b64decode(base_data)
        # 转换为np数组
        img_array = np.fromstring(img_data, np.uint8)
        # 转换成opencv可用格式
        img = cv2.imdecode(img_array, cv2.COLOR_RGB2BGR)
        try:
            if self.ut:
                return self.table_detection_test(img)
            else:
                return self.table_detection(img)
        except:
            ###将报错信息写入log文件
            s = traceback.format_exc()
            logging.error(s)

    def sortd_box_coords(self,boxs):
        """
        排序一下坐标的顺序
        :param boxs: 最小外接转换的4个坐标点
        :return: 顺时针排序的坐标点
        """
        new_box = []
        boxs = boxs.tolist()
        boxs.sort(key=lambda x: x[0])
        if boxs[0][1] > boxs[1][1]:
            new_box.append(boxs[1])
            new_box.append(boxs[0])
        else:
            new_box.append(boxs[0])
            new_box.append(boxs[1])
        if boxs[2][1] > boxs[3][1]:
            new_box.append(boxs[2])
            new_box.append(boxs[3])
        else:
            new_box.append(boxs[3])
            new_box.append(boxs[2])

        temp = new_box[1]
        new_box[1] = new_box[3]
        new_box[3] =temp
        return new_box

    def image_test(self,boxs,image,state='boxs',name=None):
        image_s = image.copy()
        image_s1 = image.copy()
        image_s2 = image.copy()
        if state == 'boxs':
            #绘画根据行分类的框
            for ke,va in boxs.items():
                for j in va:
                    for i in j:
                        cv.line(image, (i[0][0],i[0][1]), (i[1][0],i[1][1]), (255, 0, 0), 4)
                        cv.line(image, (i[1][0],i[1][1]), (i[2][0],i[2][1]), (255, 0, 0), 4)
                        cv.line(image, (i[2][0],i[2][1]), (i[3][0],i[3][1]), (255, 0, 0), 4)
                        cv.line(image, (i[3][0],i[3][1]), (i[0][0],i[0][1]), (255, 0, 0), 4)
            cv.namedWindow('boxs', cv.WINDOW_NORMAL)
            cv.imshow("boxs",image)
        elif state == 'box':
            #绘画没有分类的框
            for ke,va in boxs.items():
                for i in va:
                    cv.line(image, (i[0][0],i[0][1]), (i[1][0],i[1][1]), (255, 0, 0), 4)
                    cv.line(image, (i[1][0],i[1][1]), (i[2][0],i[2][1]), (255, 0, 0), 4)
                    cv.line(image, (i[2][0],i[2][1]), (i[3][0],i[3][1]), (255, 0, 0), 4)
                    cv.line(image, (i[3][0],i[3][1]), (i[0][0],i[0][1]), (255, 0, 0), 4)
            cv.namedWindow('box', cv.WINDOW_NORMAL)
            cv.imshow("box",image)
            # if name != None:
            #     cv2.imwrite("mark_img/box_%s"%name,image)
        else:
            #画表格外轮廓
            for ke,va in boxs.items():
                i = va
                cv.line(image, (i[0][0],i[0][1]), (i[1][0],i[1][1]), (255, 0, 0), 3)
                cv.line(image, (i[1][0],i[1][1]), (i[2][0],i[2][1]), (255, 0, 0), 3)
                cv.line(image, (i[2][0],i[2][1]), (i[3][0],i[3][1]), (255, 0, 0), 3)
                cv.line(image, (i[3][0],i[3][1]), (i[0][0],i[0][1]), (255, 0, 0), 3)
            cv.namedWindow('table', cv.WINDOW_NORMAL)
            cv.imshow("table",image)
        if not self.bool_table:
            for i in self.coo_list:
                cv2.circle(image_s, i[0], 3, (255, 0, 0), 4)
                cv2.circle(image_s, i[1], 3, (255, 0, 0), 4)
                cv.namedWindow('image_s', cv.WINDOW_NORMAL)
                cv.imshow("image_s", image_s)
        if len(self.table_enpoint) != 0:
            for table_p in self.table_enpoint:
                cv2.line(image_s1, (table_p[0],table_p[1]), (table_p[2],table_p[3]), (255, 0, 0), 5)
                cv2.line(image_s1, (table_p[2],table_p[3]), (table_p[4],table_p[5]), (255, 0, 0), 5)
                cv2.line(image_s1, (table_p[4],table_p[5]), (table_p[6],table_p[7]), (255, 0, 0), 5)
                cv2.line(image_s1, (table_p[6],table_p[7]), (table_p[0],table_p[1]), (255, 0, 0), 5)
                cv.namedWindow('image_s1', cv.WINDOW_NORMAL)
                cv.imshow("image_s1", image_s1)
            # if name != None:
            #     cv2.imwrite("mark_img/table_%s" % name, image_s1)
        if len(self.table_enpoint) != 0:
            for table_p in self.table_enpoint_fill:
                cv2.line(image_s2, (table_p[0],table_p[1]), (table_p[2],table_p[3]), (255, 0, 0), 5)
                cv2.line(image_s2, (table_p[2],table_p[3]), (table_p[4],table_p[5]), (255, 0, 0), 5)
                cv2.line(image_s2, (table_p[4],table_p[5]), (table_p[6],table_p[7]), (255, 0, 0), 5)
                cv2.line(image_s2, (table_p[6],table_p[7]), (table_p[0],table_p[1]), (255, 0, 0), 5)
                cv.namedWindow('image_s2', cv.WINDOW_NORMAL)
                cv.imshow("image_s2", image_s2)
        # n = 4
        # cv2.circle(image_s, self.coo_list[n][0], 3, (255, 0, 0), 4)
        # cv2.circle(image_s, self.coo_list[n][1], 3, (255, 0, 0), 4)

    def get_angle(self):
        x1,y1,x2,y2,x3,y3,x4,y4 = self.table_enpoint[0]


        width = self.table_width[0]
        height = self.table_height[0]
        image = self.img.copy()
        print(image.shape)
        points = (int(image.shape[1]/2),int(image.shape[0]/2))
        image1 = self.img.copy()
        k = (y1-y2) / (x1-x2)
        k1 = (y3-y4) / (x3-x4)
        angle = degrees(atan(k))
        angle1 = degrees(atan(k1))

        new_boxs = []
        for box_ in demo2(2): ### 1:住院票据  2:门诊票据
            new_box = {}
            new_box["key_name"] = box_["key_name"]
            new_box["x"] = x1 + width * box_["x_mh_ratio"]
            new_box["y"] = y1 + width * box_["y_mh_ratio"]
            new_boxs.append(new_box)
            cv2.circle(image, (int(new_box["x"]),int(new_box["y"])), 3, (255, 0, 0), 4)
            cv2.putText(image, str(new_box["key_name"]), (int(new_box["x"]-10),int(new_box["y"]-10)), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        for box_ in new_boxs:
            name = box_["key_name"]
            x = box_["x"]
            y = box_["y"]
            box_k = (y1-y)/(x1-x)
            box_angle = degrees(atan(box_k))
            radian = 2 * pi / 360 * (box_angle+angle)
            # radian = atan(box_k)+atan(k)
            box_b = sqrt((x1-x)**2+(y1-y)**2)
            a = sin(radian)*box_b
            b = cos(radian)*box_b
            # print(a,b)
            new_x = int( x1+ b)
            new_y = int( y1+ a)

            # new_x,new_y = calculateRotate((x1,y1),(x,y),angle)
            cv2.circle(image1, (int(new_x), int(new_y)), 3, (255, 0, 0), 4)
            cv2.putText(image1, str(name), (int(new_x), int(new_y)),cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

        cv2.namedWindow("image_keys",cv2.WINDOW_NORMAL)
        cv2.namedWindow("image1", cv2.WINDOW_NORMAL)
        cv2.imshow("image_keys",image)
        cv2.imshow("image1", image1)
        print(new_boxs) #计算好的key的坐标
        print('angle:',angle)

if __name__ == "__main__":
    # 小表格检测
    # fp_path1 = "Not_line_json/new35_new.png"
    name = "sxb_3.jpg"
    fp_path1 = "image/%s"%name

    # # image_test(new_test_path)#图片预处理
    #######################################单例测试###############################################################

    table_check = table_check()
    table_check.ut = True #开启自测
    #本地读取的方式
    images = table_check.cv_imread(fp_path1)
    images = cv2.GaussianBlur(images,(5,5),2)
    #使用base64的方式
    start = time.time()
    image = cv2.imencode('.jpg', images)[1]
    image_code = str(base64.b64encode(image))[2:-1]

    boxs_class,boxs = table_check.base64_to_tableDetecton(image_code) # base64图片
    if table_check.not_line:
        end = time.time()
        print(end - start)
        cv2.waitKey()
    else:
        print(len(table_check.table_enpoint),table_check.table_enpoint)
        table_check.image_test(boxs_class, images.copy(),name=name)
        table_check.image_test(boxs,images.copy(),state='box',)
        #计算表格的位置区域
        tables, tables_nums, table = table_check.fitting_tables(boxs)
        table_check.image_test(table, images.copy(),state='table')
        # table_check.image_test(table, images.copy(), state='table')

        end = time.time()
        print(end - start)
        cv.waitKey()


    # ###############################################文件夹多例测试########################################################
    # dir_path = "image"
    # file_list = os.listdir(dir_path)
    # table_check = table_check()
    # for file in file_list:
    #     image_path = os.path.join(dir_path,file)
    #     print(image_path)
    #     table_check.__init__()
    #     table_check.ut = True  # 开启自测
    #     # 本地读取的方式
    #     images = table_check.cv_imread(image_path)
    #     # table_check.image_disposes(fp_path1) #预处理降噪
    #     # cv.imshow("images",imag es)
    #     # table_detection(image, w, h) #表格检测
    #
    #     # 使用base64的方式
    #     start = time.time()
    #     image = cv2.imencode('.jpg', images)[1]
    #     image_code = str(base64.b64encode(image))[2:-1]
    #
    #     boxs_class, boxs = table_check.base64_to_tableDetecton(image_code)  # base64图片
    #
    #     if boxs_class is None and boxs is None:  ##执行无线表检测
    #         end = time.time()
    #         print(end - start)
    #         cv2.waitKey()
    #     else:
    #         table_check.image_test(boxs_class, images.copy())
    #         table_check.image_test(boxs, images.copy(), state='box',name=file)
    #         # 计算表格的位置区域
    #         # tables, tables_nums, table = table_check.fitting_tables(boxs)
    #         # table_check.image_test(table, images.copy(), state='table')
    #         # table_check.image_test(table, images.copy(), state='table')
    #
    #         end = time.time()
    #         print(end - start)
    #         cv.waitKey()