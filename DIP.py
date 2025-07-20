import cv2
import matplotlib.pyplot as plt
import numpy as np


poor_contrast_img = cv2.imread('data/poor_contrast_img1.jpeg', cv2.IMREAD_GRAYSCALE)
poor_contrast_img_copy = poor_contrast_img.copy()
hist = cv2.calcHist([poor_contrast_img_copy], [0], None, [255], [0, 255])
# # Histogram stretching
# max_i = np.max(poor_contrast_img_copy)
# min_i = np.min(poor_contrast_img_copy)
# poor_contrast_img_copy = np.uint8(((poor_contrast_img_copy - min_i) / (max_i - min_i)) * 255)
# Histogram Equalization
poor_contrast_img_copy = cv2.equalizeHist(poor_contrast_img_copy)
# # CLAHE
# clahe_100 = cv2.createCLAHE(clipLimit=1000)
# poor_contrast_img_copy = clahe_100.apply(poor_contrast_img_copy)
new_hist = cv2.calcHist([poor_contrast_img_copy], [0], None, [255], [0, 255])

# Guassian denoise
poor_contrast_img_copy = cv2.GaussianBlur(poor_contrast_img_copy, (3, 3), 0)

# # Edge detection using sobel kernel
# sobel_x = cv2.Sobel(poor_contrast_img_copy, cv2.CV_64F, 1, 0, ksize=3)
# sobel_y = cv2.Sobel(poor_contrast_img_copy, cv2.CV_64F, 0, 1, ksize=3)
# poor_contrast_img_copy = np.sqrt(np.square(sobel_x) + np.square(sobel_y))

# # Edge detection using prewitt
# prewitt_x_kernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
# prewitt_y_kernel = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)
# prewitt_x_edges = cv2.filter2D(poor_contrast_img_copy, -1, prewitt_x_kernel)
# prewitt_y_edges = cv2.filter2D(poor_contrast_img_copy, -1, prewitt_y_kernel)
# poor_contrast_img_copy = np.uint8(np.sqrt(np.square(prewitt_x_edges) + np.square(prewitt_y_edges)))

# # Edge detection uing laplacian
# poor_contrast_img_copy = cv2.Laplacian(poor_contrast_img_copy, -1, ksize=3)

# Edge detection using canny
poor_contrast_img_copy = cv2.Canny(poor_contrast_img_copy, 100, 150)

# # Find lines
# lines = cv2.HoughLines(poor_contrast_img_copy, 1, 1, 10)

# Contour
contours, hierarchy = cv2.findContours(poor_contrast_img_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(poor_contrast_img_copy, contours, 2, (255,255,255), 3)

# SIFT

cv2.imshow('new img', poor_contrast_img_copy)
if cv2.waitKey(0) == ord('q'):
    cv2.destroyAllWindows()
plt.imshow(poor_contrast_img_copy, cmap='gray')
plt.figure(figsize=(8, 4))
plt.plot(hist, color='black')
plt.plot(new_hist, color='blue')
plt.title('histogram')
plt.xlabel('pixel intensity')
plt.ylabel('frequency')
plt.xlim([0, 255])
plt.grid()
plt.show()
