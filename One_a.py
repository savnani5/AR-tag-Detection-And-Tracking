import cv2
import time
import numpy as np
from matplotlib import pyplot as plt


img = cv2.imread("img.png")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

f = np.fft.fft2(gray)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))

rows, cols, _ = img.shape
print(rows,cols)
crow, ccol = int(rows/2), int(cols/2)

# create a high pass filter to detect the edges of the square, center square is 0, remaining all ones 
mask = np.ones((rows,cols),np.uint8)
r = 30
centre = [crow, ccol]
x,y = np.ogrid[:rows, :cols]
mask_area = (x-centre[0])**2 + (y-centre[1])**2 <= r**2
mask[mask_area] = 0
# mask[crow-20:crow+20, ccol-20:ccol+20] = 0

# apply filter and inverse DFT
fshift = fshift*mask
f_ishift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)
img_back = cv2.convertScaleAbs(img_back)

canny = cv2.Canny(img_back, 100, 200, apertureSize = 3)
roi = canny[300:720,420:800]

fig1 = plt.figure()

plt.subplot(1,2,1), plt.imshow(img)
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(1,2,2), plt.imshow(magnitude_spectrum, )
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

fig2 = plt.figure()
plt.subplot(1,3,1), plt.imshow(img_back, cmap = 'gray')
plt.title('Transformed Image'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,2), plt.imshow(canny, cmap = 'gray')
plt.title('Edge Detection'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,3), plt.imshow(roi, cmap = 'gray')
plt.title('Output Region of Interest'), plt.xticks([]), plt.yticks([])


plt.show()
