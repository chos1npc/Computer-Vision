import cv2
import numpy as np

# 讀取原始圖像
img = cv2.imread("CV_beginner/lenna.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 創建空白金字塔列表
pyramid = [gray]

# 下採樣圖像並添加到金字塔中
for i in range(4):
    downsampled = cv2.resize(pyramid[-1], (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    
    pyramid.append(downsampled)

# 保存所有圖像級別
for i, level in enumerate(pyramid):
    cv2.imwrite(f"level_{i}.png", level)

# 從最低級別開始進行重構
reconstructed = pyramid[-1]
for i in range(len(pyramid)-1, -1, -1):
    # 上採樣重構圖像
    upsampled = cv2.resize(reconstructed, pyramid[i].shape[::-1], interpolation=cv2.INTER_LINEAR)

    # 計算當前級別的誤差圖像
    error = pyramid[i] - upsampled

    # 上採樣重構圖像以匹配誤差圖像的大小
    reconstructed = cv2.resize(reconstructed, error.shape[::-1], interpolation=cv2.INTER_LINEAR)

    # 添加誤差圖像到重構圖像中
    reconstructed = reconstructed + error

    # 保存重構圖像和誤差圖像
    cv2.imwrite(f"reconstructed_{i}.png", reconstructed)
    cv2.imwrite(f"error_{i}.png", error)

    # 將重構圖像放大到原始尺寸
    enlarged = cv2.resize(reconstructed, (512, 512), interpolation=cv2.INTER_LINEAR)

    # 保存放大的重構圖像
    cv2.imwrite(f"enlarged_{i}.png", enlarged)

