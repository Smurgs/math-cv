import cv2
import matplotlib.pyplot as plt
import numpy as np

# Read image
img = cv2.imread('/Users/Stefan/Desktop/a.png', 0)

# Threshold
ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Find contours
ret_img, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours
contour_img = np.zeros(img.shape, img.dtype)
contour_img = cv2.drawContours(contour_img, contours, -1, (255,0,0), 1)

# Find bounding rects
boundingRects = [cv2.boundingRect(x) for x in contours]
boundingRects = [x for x in boundingRects if not x[0] == 0]
boundingRects.sort(key=lambda x: x[0])

# Absorb bounding rects into encompassing rects
tempRects = []
for r in boundingRects:
    tempRects.append(r)
    for x in boundingRects:
        if (r[0]>x[0] and r[1]>x[1] and r[0]+r[2]<x[0]+x[2] and r[1]+r[3]<x[1]+x[3]):
            tempRects.remove(r)
            break
boundingRects = tempRects

# Draw bounding rects
bounding_img = np.zeros(img.shape, img.dtype)
for x,y,w,h in boundingRects:
    cv2.rectangle(bounding_img, (x,y), (x+w,y+h), (255,0,0), 1)

# Segment characters into their own images
PADDING = 2
segmentedCharacters = []
for x,y,w,h in boundingRects:
    segmentedCharacter = 255 * np.ones((h+(2*PADDING), w+(2*PADDING)), np.uint8)
    segmentedCharacter[2:2+h,2:2+w] = img[y:y+h,x:x+w]
    segmentedCharacters.append(segmentedCharacter)

titles = ['Original', 'Threshold', 'Contour', 'Bounding Rects', 'orig', 'b', 'l', 'a', 'c', 'k']
images = [img, thresh, contour_img, bounding_img, img, segmentedCharacters[0], segmentedCharacters[1], segmentedCharacters[2], \
         segmentedCharacters[3], segmentedCharacters[4]]

for i in xrange(10):
    plt.subplot(2,5,i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()
