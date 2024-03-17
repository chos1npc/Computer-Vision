import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

def gaussian_filter(size, sigma):
    """
    生成高斯濾波器
    """
    filter = np.zeros([size, size])
    center = size // 2
    for i in range(size):
        for j in range(size):
            filter[i, j] = np.exp(-((i - center) ** 2 + (j - center) ** 2) / (2 * sigma ** 2))
    filter /= np.sum(filter)
    return filter


def sobel_filter(img):
    """
    使用Sobel運算子計算圖像梯度大小和方向
    """
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    h, w = img.shape
    grad_x = np.zeros([h, w])
    grad_y = np.zeros([h, w])
    for i in range(1, h-1):
        for j in range(1, w-1):
            grad_x[i, j] = np.sum(img[i-1:i+2, j-1:j+2] * kernel_x)
            grad_y[i, j] = np.sum(img[i-1:i+2, j-1:j+2] * kernel_y)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    grad_dir = np.arctan2(grad_y, grad_x)
    return grad_mag, grad_dir


# 讀取圖像
image_path = "CV_beginner\Lenna.png"
image = np.array(Image.open(image_path).convert("L"))

# 1. gaussianblur
kernel_size = 5
sigma = 1
filter = gaussian_filter(kernel_size, sigma)
image_blur = np.zeros_like(image)
h, w = image.shape
for i in range(kernel_size // 2, h - kernel_size // 2):
    for j in range(kernel_size // 2, w - kernel_size // 2):
        image_blur[i, j] = np.sum(image[i - kernel_size // 2:i + kernel_size // 2 + 1,
                                 j - kernel_size // 2:j + kernel_size // 2 + 1] * filter)
image_blur = np.uint8(image_blur)

# 2. Sobel filter
grad_mag, grad_dir = sobel_filter(image_blur)

# 顯示結果
# plt.imshow(grad_mag, cmap="gray")
# plt.show()
def non_maximum_suppression(grad_mag, grad_dir):
    """
    非極大值抑制
    """
    h, w = grad_mag.shape
    grad_nms = np.zeros([h, w])
    for i in range(1, h-1):
        for j in range(1, w-1):
            direction = grad_dir[i, j]
            mag = grad_mag[i, j]
            if (direction >= -np.pi / 8 and direction < np.pi / 8) or (direction >= 7 * np.pi / 8 and direction < np.pi):
                if mag >= grad_mag[i, j-1] and mag >= grad_mag[i, j+1]:
                    grad_nms[i, j] = mag
            elif (direction >= np.pi / 8 and direction < 3 * np.pi / 8):
                if mag >= grad_mag[i-1, j+1] and mag >= grad_mag[i+1, j-1]:
                    grad_nms[i, j] = mag
            elif (direction >= 3 * np.pi / 8 and direction < 5 * np.pi / 8):
                if mag >= grad_mag[i-1, j] and mag >= grad_mag[i+1, j]:
                    grad_nms[i, j] = mag
            else:
                if mag >= grad_mag[i-1, j-1] and mag >= grad_mag[i+1, j+1]:
                    grad_nms[i, j] = mag
    return grad_nms


def threshold(grad_nms, low_threshold, high_threshold):
    """
    edge linking
    """
    h, w = grad_nms.shape
    grad_thresh = np.zeros([h, w])
    strong_i, strong_j = np.where(grad_nms >= high_threshold)
    weak_i, weak_j = np.where((grad_nms <= high_threshold) & (grad_nms >= low_threshold))
    grad_thresh[strong_i, strong_j] = 255
    grad_thresh[weak_i, weak_j] = 50
    return grad_thresh


# 3. 非極大值抑制
grad_nms = non_maximum_suppression(grad_mag, grad_dir)

# 4. edge linking
low_threshold = 20
high_threshold = 30
grad_thresh = threshold(grad_nms, low_threshold, high_threshold)

# 顯示結果
# plt.imshow(grad_thresh, cmap="gray")
# plt.show()

cv2.imwrite("./CV_beginner/canny_image/L20H30.png", grad_thresh)