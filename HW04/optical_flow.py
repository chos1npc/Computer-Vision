import numpy as np
import cv2
from PIL import Image, ImageDraw
import random
# 生成一個黑色底的影像
img1 = np.zeros((101, 101), dtype=np.uint8)

# 在左上角（row=40, col=6）放上一個大小為 21x21 的白色方框
img1[40:61, 6:27] = 255

# 生成另一個影像，將方框向右下移動一個像素
img2 = np.zeros((101, 101), dtype=np.uint8)
img2[41:62, 7:28] = 255

def sobel_filter(img):

    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    h, w = img.shape
    grad_x = np.zeros([h, w])
    grad_y = np.zeros([h, w])
    for i in range(1, h-1):
        for j in range(1, w-1):
            grad_x[i, j] = np.sum(img[i-1:i+2, j-1:j+2] * kernel_x)/8
            grad_y[i, j] = np.sum(img[i-1:i+2, j-1:j+2] * kernel_y)/8
    
    return grad_x,grad_y


Ix,Iy = sobel_filter(img1)
It = img2-img1

mask = np.ones((3, 3)) / 9.0

It_smooth = np.zeros_like(It)
for i in range(1, It.shape[0]-1):
    for j in range(1, It.shape[1]-1):
        
        It_smooth[i,j] = np.sum(It[i-1:i+2, j-1:j+2] * mask)


# 遍歷每個pixel 計算光流
flow = np.zeros((img1.shape[0], img1.shape[1], 2), dtype=np.float32)
for i in range(1,img1.shape[0]-1):
    for j in range(1,img1.shape[1]-1):
        Ax = Ix[i-1:i+2, j-1:j+2].flatten()
        Ay = Iy[i-1:i+2, j-1:j+2].flatten()
        A = np.column_stack((Ax,Ay))
        b = -It[i-1:i+2, j-1:j+2].flatten()
        
        x,_,_,_ = np.linalg.lstsq(A, b, rcond=None)
        flow[i, j] = x

# Compute the four corner points of the white box
corners = np.array([[6,40], [6,60], [26,40], [26,60]])

# Draw the flow direction on the four corners
img_draw = Image.fromarray(img1)
draw = ImageDraw.Draw(img_draw)
for i, (x, y) in enumerate(corners):
    u, v = flow[y, x]
    draw.line([(x, y), (int(x + 10*u), int(y + 10*v))], fill=64, width=1)
    img1 = cv2.arrowedLine(img1, (x, y), (int(x + 10*u), int(y + 10*v)), 128)
num_points = 3
points = []
for i in range(num_points):
    xu = random.randint(6, 26)
    yu = 40
    xl = 6
    yl = random.randint(40, 60)
    xr = 26
    yr = random.randint(40,60)
    xb = random.randint(6,26)
    yb = 60
    points.append((xu,yu))
    points.append((xl,yl))
    points.append((xr,yb))
    points.append((xb,yb))

for i, (x,y) in enumerate(points):
    u, v = flow[y, x]
    draw.line([(x, y), (int(x + 10*u), int(y + 10*v))], fill=64, width=1)
    img1 = cv2.arrowedLine(img1, (x, y), (int(x + 10*u), int(y + 10*v)), 128)

img1 = Image.fromarray(img1)
img1.show()
