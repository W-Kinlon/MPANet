import cv2
import numpy as np

path = r'C:\Users\kinlo\Desktop'
# 读入原始图像和模板图像
img = cv2.imread(path+r'\图片1.png')
template = cv2.imread(path+r'\图片1-template.png')

# 获取模板图像的尺寸
w, h = template.shape[:-1]

# 使用模板匹配方法
res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)

# 设置匹配阈值
threshold = 0.8

# 根据阈值获取匹配结果
loc = np.where(res >= threshold)

# 绘制匹配的矩形
for pt in zip(*loc[::-1]):
    cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

# 显示结果
cv2.imshow('Result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
