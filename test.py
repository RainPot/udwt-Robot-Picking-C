import cv2
from PIL import Image
import numpy as np

image = cv2.imread('F:/dataset/UNDERALL/2018origin/train_2019/images/CHN083846_0324.jpg')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
cv2.imwrite('F:/dataset/222.jpg', image)
cv2.imshow('test', image)
cv2.waitKey()

# a = np.asarray(image)
#
# b = Image.fromarray(a)
# b.show()
