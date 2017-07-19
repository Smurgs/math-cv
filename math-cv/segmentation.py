import cv2
import matplotlib.pyplot as plt
import numpy as np

# Read image
img = cv2.imread('/Users/Stefan/Downloads/formula_images/1a0bd8a36d.png', 0)

# Threshold
ret, thresh = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)

# Invert image
inverted = 255 - thresh

# Find contours
ret_img, contours, ret_hierarchy = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours
# contour_img = np.zeros(img.shape, img.dtype)
# contour_img = cv2.drawContours(contour_img, contours, -1, (255,0,0), 1)

# Find bounding rects
boundingRects = [cv2.boundingRect(x) for x in contours]

# Draw bounding rects
bounding_img = np.copy(img)
for x,y,w,h in boundingRects:
    cv2.rectangle(bounding_img, (x,y), (x+w,y+h), (0,0,0), 1)

# Segment characters into their own images
# PADDING = 2
# segmentedCharacters = []
# for x,y,w,h in boundingRects:
    # segmentedCharacter = 255 * np.ones((h+(2*PADDING), w+(2*PADDING)), np.uint8)
    # segmentedCharacter[2:2+h,2:2+w] = img[y:y+h,x:x+w]
    # segmentedCharacters.append(segmentedCharacter)

titles = ['Original', 'Threshold', 'Inverted', 'Bounding on Orig']
images = [img, thresh, inverted, bounding_img]

for i in xrange(4):
    plt.subplot(2,2,i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()
