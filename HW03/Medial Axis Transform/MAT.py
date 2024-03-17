# 導入庫

import matplotlib.pyplot as plt
import skimage.io as io

# 將圖像轉爲灰度圖像
from PIL import Image
img = Image.open("CV_beginner\cvSmall.png").convert('L')
img.save('CV_beginner\greyscale.png')

# 讀取灰度圖像
Img_Original = io.imread('CV_beginner\greyscale.png')

# 對圖像進行預處理，二值化
from skimage import filters
from skimage.morphology import disk
# 中值濾波
Img_Original = filters.median(Img_Original,disk(5))
# 二值化
BW_Original = Img_Original < 235

# 定義像素點周圍的8鄰域
#                P9 P2 P3
#                P8 P1 P4
#                P7 P6 P5

def neighbours(x,y,image):
    img = image
    x_1, y_1, x1, y1 = x-1, y-1, x+1, y+1
    return [ img[x_1][y],img[x_1][y1],img[x][y1],img[x1][y1],         # P2,P3,P4,P5
            img[x1][y], img[x1][y_1], img[x][y_1], img[x_1][y_1] ]    # P6,P7,P8,P9

# 計算鄰域像素從0變化到1的次數
def transitions(neighbours):
    n = neighbours + neighbours[0:1]      # P2,P3,...,P8,P9,P2
    return sum( (n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]) )  # (P2,P3),(P3,P4),...,(P8,P9),(P9,P2)

def MAT(image):
    Image_Thinned = image.copy()  # Making copy to protect original image
    changing1 = changing2 = 1
    while changing1 or changing2:   # Iterates until no further changes occur in the image
        # Step 1
        changing1 = []
        rows, columns = Image_Thinned.shape
        for x in range(1, rows - 1):
            for y in range(1, columns - 1):
                P2,P3,P4,P5,P6,P7,P8,P9 = n = neighbours(x, y, Image_Thinned)
                if (Image_Thinned[x][y] == 1     and    # Condition 0: Point P1 in the object regions 
                    2 <= sum(n) <= 6   and    # Condition 1: 2<= N(P1) <= 6
                    transitions(n) == 1 and    # Condition 2: S(P1)=1  
                    P2 * P4 * P6 == 0  and    # Condition 3   
                    P4 * P6 * P8 == 0):         # Condition 4
                    changing1.append((x,y))
        for x, y in changing1: 
            Image_Thinned[x][y] = 0
        # Step 2
        changing2 = []
        for x in range(1, rows - 1):
            for y in range(1, columns - 1):
                P2,P3,P4,P5,P6,P7,P8,P9 = n = neighbours(x, y, Image_Thinned)
                if (Image_Thinned[x][y] == 1   and        # Condition 0
                    2 <= sum(n) <= 6  and       # Condition 1
                    transitions(n) == 1 and      # Condition 2
                    P2 * P4 * P8 == 0 and       # Condition 3
                    P2 * P6 * P8 == 0):            # Condition 4
                    changing2.append((x,y))    
        for x, y in changing2: 
            Image_Thinned[x][y] = 0
    return Image_Thinned
BW_Skeleton = MAT(BW_Original)
import numpy as np
BW_Skeleton = np.invert(BW_Skeleton)
# 顯示細化結果
fig, ax = plt.subplots(1, 2)
ax1, ax2 = ax.ravel()
ax1.imshow(img, cmap=plt.cm.gray)
ax1.set_title('Original binary image')
ax1.axis('off')
ax2.imshow(BW_Skeleton, cmap=plt.cm.gray)
ax2.set_title('Skeleton of the image')
ax2.axis('off')
plt.savefig('CV_beginner\ thinned.png')
plt.show()
Skeleton = np.ones((BW_Skeleton.shape[0],BW_Skeleton.shape[1]),np.uint8) *255 #生成一個空灰度圖像
BW_Skeleton = BW_Skeleton + 0
for i in range(BW_Skeleton.shape[0]):
    for j in range(BW_Skeleton.shape[1]):
        if BW_Skeleton[i][j] == 0:
            Skeleton[i][j] = 0
            
plt.axis('off')
plt.imshow(Skeleton, cmap=plt.cm.gray)

import imageio
imageio.imwrite('CV_beginner\Skeleton.png', Skeleton)