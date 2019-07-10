import numpy as np
import cv2

def textdeskew(image):
	# image = cv2.imread("C://Users//SarVisha//Desktop//intern//WarpAlign//files skewed//img_right_flip//img_4.jpg")

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.bitwise_not(gray)
	thresh = cv2.threshold(gray, 0, 255,
		cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]


	coords = np.column_stack(np.where(thresh > 0))
	angle = cv2.minAreaRect(coords)[-1]
	if angle < -45:
		angle = -(90 + angle)

	else:
		angle = -angle

	(h, w) = image.shape[:2]
	center = (w // 2, h // 2)
	M = cv2.getRotationMatrix2D(center, angle, 1.0)
	rotated = cv2.warpAffine(image, M, (w, h),
		flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

	print("[INFO] angle: {:.3f}".format(angle))
	# cv2.namedWindow("Input", cv2.WINDOW_NORMAL)
	# cv2.namedWindow("Final", cv2.WINDOW_NORMAL)
	# cv2.imshow("Input", image)
	# cv2.imshow("Final", rotated)
	# cv2.waitKey(0)
	return rotated