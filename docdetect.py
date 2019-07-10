import cv2
import numpy as np
from skimage import exposure
from textdeskew import textdeskew
# from test import unetProcessing

def convert_URL2NAME(url):
    url1 = url
    image_name = url1.split('/')[len(url1.split('/')) - 1]
    return image_name

def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros(
                    (4, 2),
                    dtype="float32"
                )

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(
                        ((br[0] - bl[0]) ** 2)
                         + 
                        ((br[1] - bl[1]) ** 2)
                    )

    widthB = np.sqrt(
                        ((tr[0] - tl[0]) ** 2)
                         + 
                        ((tr[1] - tl[1]) ** 2)
                    )

    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(
                        ((tr[0] - br[0]) ** 2)
                         + 
                        ((tr[1] - br[1]) ** 2)
                    )

    heightB = np.sqrt(
                        ((tl[0] - bl[0]) ** 2)
                         + 
                        ((tl[1] - bl[1]) ** 2)
                    )

    maxHeight = max(int(heightA), int(heightB))

    # Now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
                        [0, 0],
                        [maxWidth - 1, 0],
                        [maxWidth - 1, maxHeight - 1],
                        [0, maxHeight - 1]
                    ],
                    dtype="float32"
                )

    # Compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)

    warped = cv2.warpPerspective(
                                    image,
                                    M,
                                    (maxWidth, maxHeight),
                                    flags = cv2.INTER_NEAREST
                                )

    # Return the warped image
    return warped


def preprocess(image):

    # Add 10px border to every side.
    # Done to prevent lack of edge detection
    # for documents that gets cut in the sides
    image_original = cv2.copyMakeBorder(
                    image, 
                    10, 
                    10, 
                    10, 
                    10, 
                    cv2.BORDER_CONSTANT, 
                    value=(0,0,0)
                )

    # Basic preprocessing (done to differentiate the mask and background): 
    # 1. Grayscale
    # 2. Erosion
    # 3. Bilateral Filtering
    # 4. Thresholding
    bnw = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("FinalTransformedDoc/bnw.jpg", bnw)
    # cv2.namedWindow("bnw", cv2.WINDOW_NORMAL)    
    # cv2.imshow("bnw", bnw)
    # cv2.waitKey()

    kernel = np.ones((1,1),np.uint8)
    bnw = cv2.erode(bnw,kernel,iterations = 3)
    cv2.imwrite("FinalTransformedDoc/erode.jpg", bnw)
    blur = cv2.bilateralFilter(bnw,9, 75, 75)
    cv2.imwrite("FinalTransformedDoc/blur.jpg", blur)
    ret2,threshed = cv2.threshold(
                                blur,
                                190,
                                255,
                                cv2.THRESH_TOZERO
                            )

    print ("[INFO] Image underwent \n\
            1. Grayscaling,\n\
            2. Erossion,\n\
            3. Bilateral filtering, and\n\
            4. Binary Thresholding ToZero ... \n")

    # cv2.namedWindow("Pre-processed", cv2.WINDOW_NORMAL)    
    # cv2.imshow("Pre-processed", threshed)
    # cv2.waitKey()
    
    return threshed



def detectContour(threshed):
    # Canny edge detector
    edged = cv2.Canny(threshed, 150, 200, 3)
    cv2.imwrite("static/FinalTransformedDoc/canny.jpg", edged)
    print ("[INFO] Canny in action...")
    # cv2.imshow("Canny", edged)
    # cv2.waitKey()

    # Find contours
    (_,cnts, _) = cv2.findContours(
                                                edged.copy(),
                                                cv2.RETR_EXTERNAL,
                                                cv2.CHAIN_APPROX_SIMPLE
                                            )
    # Sort contours based on contour area in decreasing order
    cnts = sorted(
                    cnts,
                    key=cv2.contourArea,
                    reverse=True
                )[:5]


    # Finding contour corners and approximating
    # to 4 corners to apply perspective transform
    for c in cnts:

        peri = cv2.arcLength(c, True)
        #screenCnt should be 4

        approx = cv2.approxPolyDP(
                                    c,
                                    0.1 * peri,
                                    True
                                )
        
        if len(approx) == 4:

            print("[INFO] Screen count: \n", approx)    
            screenCnt = approx
            break
        else:
            print("[INFO] Can't find 4 points... \n")
            exit()
    return screenCnt


def main(filename):

    print("[INFO] Reducing image size to (256,256)... ")
    # compress(filename)

    # Generate masks for the images
    # unetProcessing()

    print("[INFO] Edge Detection Operation started...\n")
    
    path_raw = "uploads/" + filename    
    filename = filename.split('.')[0]
    path_processed = "results/" + filename + ".jpg"

    raw_image = cv2.imread(path_raw)
    image_original = cv2.imread(path_processed) 
    # image_original = cv2.imread(path_processed) 
    orig = image_original
    
    print("[INFO] Finding conversion ratio ...\n")
    (H1,W1) = orig.shape[:2]
    (H2,W2) = raw_image.shape[:2]
    (Hr, Wr) = (H2/H1, W2/W1)
    print("[INFO] Found ratio ... \n")

    threshed = preprocess(orig)
    cv2.imwrite("FinalTransformedDoc/thresh.jpg", threshed)

    screenCnt = detectContour(threshed)
    
    screenCnt = screenCnt.reshape(4,2)

    # Removing offset(borders) and rescaling to original resolution
    for i in range(4):
        screenCnt[i][0] = (screenCnt[i][0]-10) * Wr
        screenCnt[i][1] = (screenCnt[i][1]-10) * Hr
    
    print("\n[INFO] #### New Screen CNT: \n", screenCnt, "\n #### ... \n")

    # Applying 4-point transform
    warped = four_point_transform(raw_image, screenCnt)
    warpec = exposure.rescale_intensity(warped, out_range = (0, 255))

    print("[INFO] Processing done. Displaying ... ")
    # cv2.namedWindow("Transformed", cv2.WINDOW_NORMAL)    
    # cv2.imshow("Transformed", warped)
    # cv2.waitKey()
    warped = textdeskew(warped)
    print("[INFO] Saving... \n")

    path = convert_URL2NAME(path_processed)
    
    cv2.imwrite("static/FinalTransformedDoc/" + path, warped)

    print("[INFO] Done... \n")
    

# if __name__ == '__main__':
#     main()
