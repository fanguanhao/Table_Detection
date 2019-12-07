import cv2,os
import numpy as np
import sys
###处理数据
# sys.path.append(os.path.abspath('../servers_9_13'))
# from imageProgress import detection_main2

def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    return cv_img


##投影法切分
def demo1(images):
    image = cv2.resize(images,(2800,1800))
    print(image.shape)
    image_test = image.copy()
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    thled = cv2.adaptiveThreshold(gray, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 33, 5)

    thled1 = thled.copy()
    touyin = np.zeros_like(thled)
    thled1[thled1==1] = 255

    d_kernel = np.ones((50, 10), np.uint8)
    e_kernel = np.ones((20, 1), np.uint8)
    dtile = cv2.dilate(thled, d_kernel)
    dtile1 = cv2.dilate(thled1, d_kernel)
    erode = cv2.erode(dtile1,e_kernel)

    image_cont, contours, hierarchy = cv2.findContours(dtile1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for cont in contours:
        cont_T = cont.T
        max_x, min_x = np.max(cont_T[0]), np.min(cont_T[0])
        max_y, min_y = np.max(cont_T[1]), np.min(cont_T[1])
        if max_x - min_x <50 or max_y - min_y<50:
            cv2.drawContours(dtile, [cont], -1, 0, cv2.FILLED)

    dtile_T = dtile.T
    touyin = touyin.T
    touyin_list = np.sum(dtile_T,axis=1)
    max_mean = np.mean(sorted(set(touyin_list))[-5:])  #极大值均值
    touyin_list = np.where(touyin_list<max_mean/2,0,touyin_list).tolist()
    print(len(touyin_list),touyin.shape,max_mean)
    h,w = touyin.shape
    ###遍历值的大小变换截取
    indexs = []
    last_id = 0
    for i in range(h):
        touyin[i][0:touyin_list[i]] = 255
        if i+1 < h:

            # last_men = np.mean(touyin_list[i:i+5])
            # if (last_men - touyin_list[i])>(last_men/2):
            #     if  i -last_id<50:
            #         last_id = i
            #         continue
            #     max_ = np.max(touyin_list[last_id:i])
            #     if max_ < max_mean / 5:
            #         continue
            #     indexs.append(i)
            #     last_id = i

            if touyin_list[i]==0 and touyin_list[i+1]>0:
                indexs.append(i)
                last_id = i
    touyin = touyin.T


    print(len(indexs))
    print(indexs)
    for i in indexs:
        cv2.line(image_test,(i,0),(i,h),(255,0,0),5)
    cv2.namedWindow("thled",cv2.WINDOW_NORMAL)
    cv2.namedWindow("dtile", cv2.WINDOW_NORMAL)
    cv2.namedWindow("erode", cv2.WINDOW_NORMAL)
    cv2.namedWindow("dtile1", cv2.WINDOW_NORMAL)
    cv2.namedWindow("touyin", cv2.WINDOW_NORMAL)
    cv2.namedWindow("image_test", cv2.WINDOW_NORMAL)
    cv2.namedWindow("image_cont", cv2.WINDOW_NORMAL)
    cv2.imshow("thled",thled1)
    cv2.imshow("touyin",touyin)
    cv2.imshow("dtile", dtile)
    cv2.imshow("dtile1", dtile1)
    cv2.imshow("erode", erode)
    cv2.imshow("image_test",image_test)
    cv2.imshow("image_cont", image_cont)
    # cv2.waitKey()

##形态切分
def demo2(image):
    image_test = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thled1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 33, 5)
    test_img = np.zeros_like(thled1)
    d_kernel = np.ones((25, 10), np.uint8)
    dtile1 = cv2.dilate(thled1, d_kernel)
    ima, contours, hierarchy = cv2.findContours(dtile1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cont in contours:
        cont_T = cont.T
        max_x, min_x = np.max(cont_T[0]), np.min(cont_T[0][0])
        max_y, min_y = np.max(cont_T[1]), np.min(cont_T[1])

        if max_x - min_x<50:
            continue
        if max_y - min_y <100:
            continue
        cv2.drawContours(test_img, [cont], -1, (255, 0, 0), 5)
        rect = cv2.minAreaRect(cont)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        box = sortd_box_coords(box)
        cv2.line(image_test, tuple(box[0]), tuple(box[1]), (255, 0, 0), 3)
        cv2.line(image_test, tuple(box[1]), tuple(box[2]), (255, 0, 0), 3)
        cv2.line(image_test, tuple(box[2]), tuple(box[3]), (255, 0, 0), 3)
        cv2.line(image_test, tuple(box[3]), tuple(box[0]), (255, 0, 0), 3)


    cv2.namedWindow("thled", cv2.WINDOW_NORMAL)
    cv2.namedWindow("dtile1", cv2.WINDOW_NORMAL)
    cv2.namedWindow("image_test", cv2.WINDOW_NORMAL)
    cv2.namedWindow("contours", cv2.WINDOW_NORMAL)

    cv2.imshow("thled", thled1)
    cv2.imshow("dtile1", dtile1)
    cv2.imshow("contours", test_img)
    cv2.imshow("image_test", image_test)
    cv2.waitKey()


def demo3(images,old_shape):
    image = cv2.resize(images, (2800, 1800))
    image_test = image.copy()
    image_test1 = image.copy()
    image_test2 = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thled = cv2.adaptiveThreshold(gray, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 33, 5)
    thled1 = thled.copy()
    touyin = np.zeros_like(thled)
    thled1[thled1 == 1] = 255
    d_kernel = np.ones((1, 30), np.uint8)
    dtile = cv2.dilate(thled, d_kernel)
    dtile1 = cv2.dilate(thled1, d_kernel)
    touyin_list = np.sum(dtile, axis=1)
    # print(touyin_list.tolist())
    h, w = touyin.shape
    max_mean = np.mean(sorted(set(touyin_list))[-5:])  # 极大值均值
    touyin_list = np.where(touyin_list < max_mean / 2, 0, touyin_list).tolist()

    row_indx_top = []
    row_indx_down = []
    last_top = 0
    last_down = 0
    for i in range(h):
        touyin[i][0:touyin_list[i]] = 255
        if i+1 <h:
            if touyin_list[i] == 0 and touyin_list[i+1]!=0:
                if abs(i-last_top) > 10 :
                    row_indx_top.append(i)
                    last_i = i
            if touyin_list[i] != 0 and touyin_list[i+1]==0:
                if abs(i-last_down) > 10:
                    row_indx_down.append(i)
                    last_i = i
    new_indx_line =[row_indx_top[0]]
    new_indx_line+= [int((row_indx_top[i+1]+row_indx_down[i])/2)for i in range(len(row_indx_top[1:]))]
    new_indx_line.append(row_indx_down[-1])
    new_indx_line1 = []
    # print(new_indx_line)
    last_i = 0
    for i in range(len(new_indx_line)):
        if abs(new_indx_line[i] - new_indx_line[last_i]) <15:
            continue
        else:
            new_indx = int(np.mean(new_indx_line[last_i:i]))
            new_indx_line1.append(new_indx)
            last_i = i
            if i == len(new_indx_line)-1:
                new_indx_line1.append(new_indx_line[i])
    # print(len(new_indx_line1),new_indx_line1)

    last_id = 0
    images = []
    d_kernel = np.ones((50, 15), np.uint8)
    e_kernel = np.ones((2,2),np.uint8)
    # print(thled1.shape)

    clo_index = {}
    clo_index_end = {}
    clo_index_state = {}  #记录分类值的大小，后用于舍去数量较小的类别
    clo_index_end_state = {}
    clo_enpoint = []  #竖线的极值端点
    row_index = []
    for i in range(len(new_indx_line1)):
        if i == 0 or i == len(new_indx_line1)-1:
            cv2.line(image_test, (0, new_indx_line1[i]), (w, new_indx_line1[i]), (255, 0, 0), 2)
            cv2.line(image_test2, (0, new_indx_line1[i]), (w, new_indx_line1[i]), (255, 0, 0), 2)
            clo_enpoint.append(new_indx_line1[i])
            row_index.append(new_indx_line1[i])
            last_id = new_indx_line1[i]
            continue
        img = thled1[last_id:new_indx_line1[i],0:w]
        erode_img = cv2.erode(img,e_kernel)
        dtile_img = cv2.dilate(erode_img, d_kernel)

        dtile_img1 = dtile_img.copy()
        dtile_img1[dtile_img1==255] = 1
        img_sum_list = np.sum(dtile_img1,axis=0)
        clo_state = False
        for j in range(len(img_sum_list)):
            if j+1 == len(img_sum_list): continue
            if img_sum_list[j] == 0 and img_sum_list[j+1]>0:
                cv2.line(image_test, (j, last_id), (j, new_indx_line1[i]), (255, 0, 0), 2)
                for x_ in clo_index.keys():
                    point_end = clo_index[x_][1][-1][0] #上一个竖线的x位置
                    if abs(j - point_end) <15:
                        clo_index[x_][1].append([j, last_id, j, new_indx_line1[i]])
                        clo_index[x_][0] += 1
                        clo_index_state[x_][0] += 1
                        clo_state = True
                        break
                if clo_state:
                    clo_state = False
                else:
                    clo_index[j] = [1, [[j, last_id, j, new_indx_line1[i]]]]
                    clo_index_state[j] = [1]
            if img_sum_list[j] > 0 and img_sum_list[j+1]==0:
                cv2.line(image_test, (j, last_id), (j, new_indx_line1[i]), (255, 0, 0), 2)
                for x_ in clo_index_end.keys():
                    point_end = clo_index_end[x_][1][-1][0]  # 上一个竖线的x位置
                    if abs(j - point_end) <15:
                        clo_index_end[x_][1].append([j, last_id, j, new_indx_line1[i]])
                        clo_index_end[x_][0] += 1
                        clo_index_end_state[x_][0] += 1
                        clo_state = True
                        break
                if clo_state:
                    clo_state = False
                else:
                    clo_index_end[j] = [1, [[j, last_id, j, new_indx_line1[i]]]]
                    clo_index_end_state[j] = [1]
        cv2.line(image_test, (0, new_indx_line1[i]), (w, new_indx_line1[i]), (255, 0, 0), 2)
        cv2.line(image_test2, (0, new_indx_line1[i]), (w, new_indx_line1[i]), (255, 0, 0), 2)
        row_index.append(new_indx_line1[i])
        last_id = new_indx_line1[i]
        # cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        # cv2.namedWindow("dtile_img", cv2.WINDOW_NORMAL)
        # cv2.namedWindow("erode_img", cv2.WINDOW_NORMAL)
        # cv2.imshow("img",img)
        # cv2.imshow("erode_img", erode_img)
        # cv2.imshow("dtile_img", dtile_img)
        # cv2.waitKey()
    class_value = int(max(clo_index_state.values())[0]/2)
    class_end_value = int(max(clo_index_end_state.values())[0]/2)
    for ke,va in clo_index_state.items(): #列开头竖线
        if va[0]<class_value:
            clo_index.pop(ke)

    for ke,va in clo_index_end_state.items(): #列结束竖线
        if va[0] < class_end_value:
            clo_index_end.pop(ke)
    values = clo_index.values()
    values = [[[va[0],va[2]]for va in vas[1]] for vas in values]
    values = [int(np.mean(va)) for va in values]#X轴竖线
    values = sorted(values)
    values1 = clo_index_end.values()
    values1 = [[[va[0],va[2]]for va in vas[1]] for vas in values1]
    values1 = [int(np.mean(va)) for va in values1]  # X轴竖线
    values1 = sorted(values1)
    values.append(values1[-1])
    for clo_va in values:
        cv2.line(image_test1,(clo_va,clo_enpoint[0]),(clo_va,clo_enpoint[1]),(255,0,0),5)
        cv2.line(image_test2, (clo_va, 0), (clo_va, h), (255, 0, 0), 5)
    for row in row_index:
        cv2.line(image_test1,(values[0],row),(values[-1],row),(255,0,0),2)
    # for clo_va in values1:
    #     cv2.line(image_test1, (clo_va, clo_enpoint[0]), (clo_va, clo_enpoint[1]), (0, 0, 255), 5)


    # cv2.namedWindow("thled", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("dtile", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("dtile1", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("touyin", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("image_test", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("box", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("image_s1", cv2.WINDOW_NORMAL)
    # cv2.imshow("thled", thled1)
    # cv2.imshow("touyin", touyin)
    # cv2.imshow("dtile", dtile)
    # cv2.imshow("dtile1", dtile1)
    # cv2.imshow("image_test", image_test)
    # cv2.imshow("box", image_test1)
    # cv2.imshow("image_s1", image_test2)
    row_lines,clo_lines = restore(row_index,values,image.shape,old_shape)
    return row_lines,clo_lines  ##ruturn row_lines clo_lines

def restore(row_index,values,new_shape,old_shape):
    """
    还原图像尺寸
    :param new_shape: 归一化的尺寸
    :param old_shape: 原始的尺寸
    :return: 宽高的比例值
    """
    new_h,new_w = new_shape[:2]
    old_h,old_w = old_shape[:2]
    w_roi = old_w/new_w
    h_roi = old_h/new_h
    new_row_lines = []
    new_clo_lines = []
    for row in row_index:
        new_row_lines.append(int(row*h_roi))
    for clo in values:
        new_clo_lines.append(int(clo*w_roi))
    return new_row_lines,new_clo_lines


##切分体检报告
def demo4(images):
    image = cv2.resize(images, (2800, 1800))
    image_test = image.copy()
    image_test1 = image.copy()
    image_test2 = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thled = cv2.adaptiveThreshold(gray, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 33, 5)
    thled1 = thled.copy()
    touyin = np.zeros_like(thled)
    h, w = touyin.shape
    thled1[thled1 == 1] = 255
    d_kernel = np.ones((1, 30), np.uint8)
    e_kernel = np.ones((3,3),np.uint8)

    erode = cv2.erode(thled, e_kernel)
    erode1 = cv2.erode(thled1, e_kernel)
    dtile = cv2.dilate(erode, d_kernel)
    dtile1 = cv2.dilate(erode1, d_kernel)
    touyin_list = np.sum(dtile, axis=1)
    for i in range(h):
        touyin[i,0:touyin_list[i]] = 255

    cv2.namedWindow("thled", cv2.WINDOW_NORMAL)
    cv2.namedWindow("dtile", cv2.WINDOW_NORMAL)
    cv2.namedWindow("dtile1", cv2.WINDOW_NORMAL)
    cv2.namedWindow("touyin", cv2.WINDOW_NORMAL)
    cv2.namedWindow("erode", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("image_test1", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("image_test2", cv2.WINDOW_NORMAL)
    cv2.imshow("thled", thled1)
    cv2.imshow("touyin", touyin)
    cv2.imshow("dtile", dtile)
    cv2.imshow("dtile1", dtile1)
    cv2.imshow("erode", erode)
    # cv2.imshow("image_test1", image_test1)
    # cv2.imshow("image_test2", image_test2)


def sortd_box_coords(boxs):
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

#旋转图像用
def Image_de(file_dir,save_dir):
    file_list = os.listdir(file_dir)
    for file in file_list:
        file1 = file.split(".")
        if file1[1] == "jpg" or file1[1] == "png":
            print(file[0])
            # state,image = detection_main2(file_dir,file,save_dir)
            # # cv2.imshow("image",image)
            # # cv2.waitKey()
            # if state:
            #     cv2.imwrite("E:\\Not_line_Image_new\\prepro\\"+file1[0]+"nwe.jpg",image)
            # else:
            #     cv2.imwrite("E:\\Not_line_Image_new\\not_prepro\\" + file + "new.jpg", image)

if __name__ == "__main__":
    ####切分无线表
    path = "E:\\Not_line_Image_new\\prepro\\new47.jpg"
    path1 = r"image_data/4.jpg"
    image = cv_imread(path1)
    # demo1(image)  ###列切分
    demo3(image)  ###单元格
    cv2.waitKey()
    cv2.imdecode()

    # ###切分体检报告书
    # path = "D:\\aaaaaa数据\\体检报告书\\8.jpg"
    # image = cv_imread(path)
    # demo4(image)  ###单元格
    # # demo1(image)
    # cv2.waitKey()



    # demo2(image)
    # read_files_test("E:\\Not_line_Image_new\\prepro\\")

    # file_dir = "E:\\工作数据文件夹\\Not_line_Image\\"
    # save_dir = "E:\\工作数据文件夹\\Not_line_Image_new\\"
    # Image_de(file_dir,save_dir)


