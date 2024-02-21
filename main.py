import cv2
import numpy as np
import numpy as np
import mser
import swt
#import model as mm
import matplotlib.pyplot as plt

def text_extraction(img):
    a, b, c = img.shape
    gray = swt.get_grayscale(img)
    ggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('my Image  ', img)
    mean = cv2.mean(img)
    mean = (mean[0] + mean[1] + mean[2]) / 3
    cliplimit = (255 / mean) * 1.8
    Iclahe = mser.clahe(img, clipLimit=cliplimit)
    Iclahe = cv2.cvtColor(Iclahe, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('the  clahe image ', Iclahe )
    boxes, regions, RImage = mser.detect_mser(Iclahe)
    edges = mser.canny_detector(img, 30, 60)
    merged = np.zeros_like(edges)
    for i in range(a):
        for j in range(b):
            if (edges[i, j]):
                for k in range(-2, 2):
                    for l in range(-2, 2):
                        if (b > (k + j) > 0 and a > (l + i) > 0):
                            if (RImage[i + l, j + k] > 0):
                                merged[i, j] = 255
    merged = merged.astype(np.uint8)
    return merged ,gray,Iclahe,boxes
def get_swt(gray,merged):
    gradients = swt.get_gradients(gray)

    swt0 = swt.apply_swt(merged, gradients, True)
    swt0 = (255 * swt0 / (swt0.max()+1)).astype(np.uint8)
    labels, components = swt.connected_components(swt0,threshold=5)
    labels = labels.astype(np.float32) /( labels.max()+1)
    labels= (labels*255.).astype(np.uint8)
    #cv2.imshow('SWT0  with connected components ', l)

    """swt1 = swt.apply_swt(merged, gradients, False)
    swt1 = (255 * swt1/ (swt1.max()+1)).astype(np.uint8)
    labels1, components1 = swt.connected_components(swt1,threshold=5)
    labels1 = labels1.astype(np.float32) / (labels1.max()+1)
    labels1= (labels1*255.).astype(np.uint8)"""

    return components,labels,
    #cv2.imshow('SWT1  with connected components ', l1)
def text_swt(merged, gray):
    components, labels = get_swt(gray, merged)
    Alable0, Acompon0 = swt.variance_discard_non_text(gray, components)
    Alable0, Acompon0 = swt.filter(Alable0, Acompon0)
    #cv2.imshow(' aspect ratio  detect 00', Alable0)
    #cv2.imwrite("strok.png",Alable0)
    return Alable0

    """ccc=np.concatenate((bbb,merged_bboxes),axis=0)
    print("ccc len :",len(ccc))
    ddd ,_ = cv2.groupRectangles(ccc, 0)
    ss=np.copy(img)

    #merged_bboxes , weights  = cv2.groupRectangles(merged_bboxes, 1, 0.2)
    for box  in ddd:
            cv2.rectangle(ss, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)
    cv2.imshow('the reult of concatinat join ', ss)

    listt=get_word(img,ddd)
    prevlen=0
    while(prevlen!=len(listt)):
        prevlen=len(listt)
        listt = get_word(img, listt)

    ss=np.copy(img)
    #merged_bboxes , weights  = cv2.groupRectangles(merged_bboxes, 1, 0.2)
    for box  in listt:
            cv2.rectangle(ss, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)
    cv2.imshow('the reult 0000 ', ss)"""

    """
    MImag= Alable0
    aa=get_rect_result(word,Alable0)


    #cv2.imshow(' result ', MImag)

    rect_result=get_rect_result(merged_bboxes)
    ss=np.copy(img)
    for box in rect_result :
        cv2.rectangle(ss, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)
    cv2.imshow("rectangle 00000",ss)
    print("  length is my rect-result  ------>>>> ",len(rect_result))
    result=get_region_result(rect_result)
    cv2.imshow("my rectangel 00000   ",result)

    listt=get_word(img,rect_result)
    ss=np.copy(img)
    for box in listt :
        cv2.rectangle(ss, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)
    cv2.imshow("rectangle  111111  ",ss)
"""
def filter_swt(c,img):
    #cv2.imshow('the filter immmg  gggg', img)
    c=c.tolist()
    cc=[]
    for box in c:
        x, y, w, h = box
        ss=0
        for i in range(h):
            for j in range (w):
               if(  img[y+i,x+j]>110):
                   ss=ss+1
        if(ss*10>(w*h)) :
            cc.append(box)
    return cc
    return  result
def alinement(b1,b2,s):
    if(s=="v") :
        x1=[b1[0],b1[0]+b1[2]]
        x2=[b2[0],b2[0]+b2[2]]
    elif(s=="h"):
        x1 = [b1[1], b1[1] + b1[3]]
        x2 = [b2[1], b2[1] + b2[3]]
    else:
        return False
    if (x1[0] > x2[0]):
        x3 = x1
        x1 = x2
        x2 = x3
    if (x1[1] >= x2[1]):
        return True
    else:
        d1 = x1[1] - x2[0]
        d2 = x2[1] - x1[1]
    if((d1/d2)>0.6):
        return True
    else:
        return False
def concatinate_box(box1, box2):

    x, y, w, h = box1
    x0, y0, w0, h0 = box2

    x1 = (x, y)
    x3 = (x + w,y + h)
    y1 = ( x0,y0)
    y3 = (x0 + w0,y0 + h0)


    z1=min(x1[0],y1[0])
    z2=min(x1[1],y1[1])
    z3=max(x3[0],y3[0])-z1
    z4=max(x3[1],y3[1])-z2
    return np.array([z1,z2,z3,z4])
def group_rectangles(rectangles,    pading=0 ):
    bboxes = rectangles
    rectangles=rectangles.tolist()
    x1, y1, w, h = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    x2, y2 = x1 + w, y1 + h
    grouped_rectangles = []
    while(len(rectangles)>0):
            re=rectangles[0]
            group = re
            rectangles.remove(re)
            stop=True
            while (stop):
                  stop=False
                  for recc in rectangles:
                            x1_overlap = max(group[0],recc[0] )
                            y1_overlap = max(group[1], recc[1])
                            x2_overlap = min(group[0] + group[2],recc[0] + recc[2])
                            y2_overlap = min(group[1] + group[3],recc[1] + recc[3])
                            area = max(0, x2_overlap - x1_overlap + 1) * max(0, y2_overlap - y1_overlap + 1)
                            if area >= 5:
                                if (alinement(group, recc, 'h')):
                                    if (alinement(group, recc, 'h')):

                                        stop=True
                                        group=concatinate_box(group,recc)
                                        rectangles.remove(recc)
            grouped_rectangles.append(group)
    """for i in range(len(grouped_rectangles)):
        grouped_rectangles[i] = [grouped_rectangles[i][0],
                                  grouped_rectangles[i][1] - pading,
                                 grouped_rectangles[i][2],
                                 grouped_rectangles[i][3]+2*pading]"""



    return grouped_rectangles

def  label_slicing(img,ddd):
    def get_hrz_histo(ddd, vlimit=None):
        hrz_lis = [0] * img.shape[0]
        if vlimit == None:
            for box in ddd:
                for i in range(box[1], box[1] + box[3]):
                    hrz_lis[i] = hrz_lis[i] + box[2]
        else:
            for box in ddd:
                if (vlimit[0] < box[0] < vlimit[1]):
                    for i in range(box[1], box[1] + box[3]):
                        hrz_lis[i] = hrz_lis[i] + box[2]
        return hrz_lis

    def get_vrt_histo(ddd, vlimit=None):
        vrt_lis = [0] * img.shape[1]
        if vlimit == None:
            for box in ddd:
                for i in range(box[0], box[0] + box[2]):
                    vrt_lis[i] = vrt_lis[i] + box[1]
        else:
            for box in ddd:
                if (vlimit[0] < box[0] < vlimit[1]):
                    for i in range(box[0], box[0] + box[2]):
                        vrt_lis[i] = vrt_lis[i] + box[3]
        return vrt_lis

    def get_hcent(ddd):
        vrt_lis = get_vrt_histo(ddd)
        indexes = [];
        indexe = [];
        new = False
        ll = 0;
        lr = img.shape[1] - 1
        for i, h in enumerate(vrt_lis):
            if (h == 0):
                if (new == False):
                    indexe.append(i)
                    new = True
            else:
                if (new):
                    indexe.append(i)
                    indexes.append(indexe)
                    indexe = []
                    new = False
        ind = [];
        max = 0
        for indexe in indexes:
            if (indexe[0] == 0 or indexe[1] == 499):
                if (indexe[0] == 0):
                    ll = indexe[1]
                else:
                    lr = indexe[0]
            else:
                if (indexe[1] - indexe[0] > max):
                    max = indexe[1] - indexe[0]
                    ind = indexe
        center = ind
        return center, ll, lr

    def get_slice_vrt(ddd, z):
        vrt_lis = get_vrt_histo(ddd, z)
        indexes = [];
        indexe = [];
        new = False
        for i, h in enumerate(vrt_lis):
            if (h > 100):
                if (new == False):
                    indexe.append(i)
                    new = True
            else:
                if (new):
                    indexe.append(i)
                    indexes.append(indexe)
                    indexe = []
                    new = False
        slice = []
        for i in indexes:
            slice.append([i, [0, 299]])
        return slice

    def get_slice_hrz(ddd, z):
        hrz_lis = get_hrz_histo(ddd, z)
        indexes = [];
        indexe = [];
        new = False
        for i, h in enumerate(hrz_lis):
            if (h > 50):
                if (new == False):
                    indexe.append(i)
                    new = True
            else:
                if (new):
                    indexe.append(i)
                    indexes.append(indexe)
                    indexe = []
                    new = False
        slice = []
        for i in indexes:
            slice.append([z, i])
        return slice
    # test zone
    center,ll,lr=get_hcent(ddd)
    if (center[0]+center[1]<img.shape[1]):
        zv=[0,center[0]]
        zh=[center[1],img.shape[1]-1]
    else:
        zh = [ll, center[0]]
        zv = [center[1], lr]

    ssh = get_slice_hrz(ddd,zh)

    ssv = get_slice_vrt(ddd,zv)


    """print(ssv)
    ss = np.copy(img)
    print(ss.shape)
    for box  in ssh:
        cv2.rectangle(ss, (box[0][0], box[1][0]), (box[0][1], box[1][1] ), (0, 255, 0), 2)
    for box  in ssv:
        cv2.rectangle(ss, (box[0][0], box[1][0]), (box[0][1], box[1][1] ), (0, 255, 0), 2)
    cv2.imshow('rrrrrrrrrrrrr', ss)"""
    # box  format : [(x,y),(x',y')] ==> (point1 , point2)
    return ssh+ssv

def main():

    link="real_labes/d4.png"
    img = cv2.imread(link)
    img = cv2.resize(img, (500, 300))




    merged,gray,yy,merged_bboxes=text_extraction(img)
    ss = np.copy(img)
    for box in merged_bboxes:
            cv2.rectangle(ss, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)
    #cv2.imshow('ffffffffffffffffffffff', ss)


    swt=text_swt(merged,gray)
    merged_bboxes=np.array(filter_swt(merged_bboxes,swt))
    for box  in merged_bboxes:
                cv2.rectangle(ss, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)
    #cv2.imshow('the result ytwrehg grouprect', ss)


    ddd=group_rectangles(merged_bboxes)
    ss = np.copy(img)
    for box  in ddd:
            cv2.rectangle(ss, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)
    #cv2.imshow('the result of grouprect', ss)

    slices =label_slicing(img,ddd)
    print(slices)
    ss = np.copy(img)

    for box,i in zip(slices,range(len(slices))):

        tt=ss[min(box[1][0], box[1][1]):max(box[1][0], box[1][1]),
           min(box[0][0],box[0][1]):max(box[0][0],box[0][1])]
        cv2.imwrite("detect_result"+ '/' +str( i )+ '.png', tt)
        print(tt.shape)
        #cv2.imshow('the result of grouprect', tt)
        cv2.rectangle(ss, (box[0][0], box[1][0]), (box[0][1], box[1][1]), (0, 255, 0), 2)
    cv2.imwrite("detect_result"+ '/image.png', ss)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
main()