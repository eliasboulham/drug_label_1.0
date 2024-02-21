
import cv2
import numpy as np
def clahe(image, clipLimit=2, tileGridSize=(10, 10)):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    imguse, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    result = clahe.apply(imguse)
    merged = cv2.merge((result, a, b))
    result = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return result
def detect_mser(img, delta=10, min_area=15, max_area=400, max_variation=0.9
                , min_diversity=1,pading=0):
    #resahpe  to gray
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    a,b=gray.shape
    mser = cv2.MSER_create(delta=delta, min_area=min_area, max_area=max_area, max_variation=max_variation, min_diversity=min_diversity)
    regions, vr = mser.detectRegions(gray)

    bboxes = [cv2.boundingRect(region) for region in regions]
    """for  i in range(len(bboxes)):

        bboxes[i]=(bboxes[i][0],bboxes[i][1]+pading,bboxes[i][2],bboxes[i][3]-pading*2)

    """
    ss = np.copy(img)
    ss = cv2.cvtColor(ss, cv2.COLOR_GRAY2BGR)
    for box in bboxes:
        cv2.rectangle(ss, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)
    #cv2.imshow('first ++++', ss)
    bbb=[]
    # fiter boxes
    for box ,region in zip(bboxes,regions):
        x, y, w, h = box
        if(True):
                    """if (w*10>h  > w*0.05):"""
                    #if(len(region)<(w*h*0.9)):
                    bbb.append(box)
    bboxes=np.array(bbb)

    regImg = np.zeros(a * b, dtype="uint8").reshape(a, b)
    for i in range(len(regions) ):
        for j in range(len(regions[i]) ):
            regImg[regions[i][j][1], regions[i][j][0]] = 255
    ss = np.copy(img)
    ss = cv2.cvtColor(ss, cv2.COLOR_GRAY2BGR)
    """for box in bboxes:
                cv2.rectangle(ss, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)
    cv2.imshow('second  ++++', ss)"""
    return bboxes,regions,regImg
def canny_detector(image,lower,upper):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(img_blur, lower, upper)
    return edges
