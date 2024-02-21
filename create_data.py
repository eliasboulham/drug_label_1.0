import random
from pathlib import Path
import pandas as pd
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from main import *
import json
#import label_extr as ext
#import extraction as ext

# Load the image
dircd = "new_data"


with open('medicament.json') as f:
    data = json.load(f)
df = pd.DataFrame(data[1]["data"])
print(df.shape)
medicament =df.iloc[:,1:21].to_numpy()

# Define the text, font, and other properties
def get_font():
    fonts = []
    data_dir = Path("font/")
    images = sorted(list( list(data_dir.glob("*ttf"))))
    for i in images:
        fonts.append(str(i))


    return fonts

fonts = get_font()

def my_fitting(cc, x=50, y=200):
    a, b, _ = cc.shape
    r = a / x
    w = b / r
    if (w < y):
        w = round(w)
        cc = cv2.resize(cc, (w, x))
        cc = cv2.copyMakeBorder(cc, 0, 0, 0, y - w, cv2.BORDER_CONSTANT,
                                value=[random.randint(200, 255), random.randint(200, 255), random.randint(200, 255)])
    else:
        cc = cv2.resize(cc, (y, x))

    return cc

def noise_cam(img):
    #img = my_fitting(img)
    a, b, c = img.shape
    for v in range(a):
        for h in range(b):
            if (random.randint(0, 2)):
                if (np.mean(img[v, h]) < 155):
                    for i in range(c):
                        img[v, h, i] = np.uint8(img[v, h, i] + random.randint(0, 10))
                else:
                    for i in range(c):
                        img[v, h, i] = np.uint8(img[v, h, i] - random.randint(0, 100))

    return img

def affine(img):
    x = random.randint(0, 0)  # Translation in the x direction
    y = random.randint(0, 0)  # Translation in the y direction
    x1 = random.uniform(-0.03, 0.03)  # Translation in the x direction
    y1 = random.uniform(-0.03, 0.03)
    x2 = random.uniform(0.95, 1.05)  # Translation in the x direction
    y2 = random.uniform(0.95, 1.05)

    add = random.randint(0, 5)
    M = np.float32([[x2, x1, add + x], [y1, y2, add + y]])

    # Apply the translation
    img_translation = cv2.warpAffine(img, M, (img.shape[1] + 5, img.shape[0] + 5))

    return img_translation

def create_image():
    # the max_length of word will be generated, for fixing our words length .
    max_length = 40

    # chose random element from  medicament [i ,j] .
    random_row = np.random.randint(0, medicament.shape[0])
    random_col = np.random.randint(0, medicament.shape[1])
    text=str(medicament[random_row,random_col])

    text.replace('/', '@')
    text.replace('.', '$')

    txt_length = len(text)
    img = get_image_text(text)

    if isinstance(img, np.ndarray):
        # fixing the text label length .
        for k in range(txt_length, max_length):
            text = text + "#"
        print("1===",len(text))

        # make some noise
        img = noise_cam(img)

        # finally we will save our  image
        cv2.imwrite(dircd + '/' + text + '.png', img)

def get_image_text(text):
    font =  ImageFont.truetype(random.choice(fonts), random.randint(10,40))
    print(font)
    bbox = font.getbbox(text)
    text_size = (bbox[2] - bbox[0], bbox[3] - bbox[1])

    image = Image.new('RGB', text_size, color=(0, 0, 0))

    image = np.array(image)
    image = cv2.copyMakeBorder(image, 6, 6, 6,6, cv2.BORDER_CONSTANT, value=[ random.randint(200, 255),  random.randint(200, 255),  random.randint(200, 255)])

    rr1 = random.randint(0, 2)
    image[:, :, :] = random.randint(200, 255)

    if (rr1):
        r1 = random.randint(0, image.shape[1])
        r2 = random.randint(r1, image.shape[1])
        image[:, r1:r2] = [random.randint(0, 10), random.randint(20, 180), random.randint(0, 10)]
    rr1 = random.randint(0, 2)
    image[:, :, :] = random.randint(200, 255)
    if (rr1):
        r1 = random.randint(0, image.shape[1])
        r2 = random.randint(r1, image.shape[1])
        image[:, r1:r2] = [random.randint(0, 10), random.randint(20, 180), random.randint(0, 10)]

    image = Image.fromarray(image)

    draw = ImageDraw.Draw(image)

    color = (random.randint(5, 40), random.randint(5, 40), random.randint(5, 40))
    draw.text((3, 3), text, fill=color, font=font)
    numpy_image = np.array(image)

    image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

    #image = cv2.copyMakeBorder(image, 20, 20, 20,20, cv2.BORDER_CONSTANT, value=[random.randint(200, 255), random.randint(200, 255), random.randint(200, 255)])
    height, width = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), random.randint(-10,10)/10, 1)
    image = cv2.warpAffine(image, rotation_matrix, (width, height))
    return image

def show(ss, img):
    cv2.imshow(ss, img)
    cv2.waitKey(0)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


data_n= 1

for i in range(data_n):

    create_image()


"""gget_image_mix([num])
    get_image_mix([num,other])
    
    get_image_mix(all)
    get_image_mix(all)
    get_image_mix([ABC])"""
#print(text_size)

# Get the size of the text using the specified font
"""cv2.waitKey(0)
cv2.destroyAllWindows()

"""