import numpy as np
from PIL import Image



def conv(image,mask):

    
    n,m = image.shape
    a,b = mask.shape
    img = []
    if a >b:
        for i in range(n-b):
            line = []
            for j in range(m-a):
                in_mask = input[i:i+b,j:j+a]
                line.append(np.sum(np.multiply(mask, in_mask)))
            img.append(line)
        img = np.array(img)
        return img
    if a < b:
        for i in range(n-a):
            line = []
            for j in range(m-b):
                in_mask = input[i:i+a,j:j+b]
                line.append(np.sum(np.multiply(mask, in_mask)))
            img.append(line)
        img = np.array(img)
        return img


def average_filter(image, size):
    
    mask_col = np.ones((size, 1)) / (size)

    mask_row = np.ones((1,size)) / (size)
    
    filtered_col = conv(image, mask_col)
    result = conv(filtered_col,mask_row)
    
    return result



img_path = "CV_beginner\HW01\Lenna.png"
img = Image.open(img_path)
gray = img.convert('L')
input = np.asarray(gray)
i = average_filter(input,11)
avg_fil_img = Image.fromarray(i)

avg_fil_img = avg_fil_img.convert("RGB")
# avg_fil_img.save("CV_beginner/HW01/3mult3_2.jpg")
# avg_fil_img.save("CV_beginner/HW01/5mult5_2.jpg")
# avg_fil_img.save("CV_beginner/HW01/7mult7_2.jpg")
# avg_fil_img.save("CV_beginner/HW01/9mult9_2.jpg")
# avg_fil_img.save("CV_beginner/HW01/11mult11_2.jpg")