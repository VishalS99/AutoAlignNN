import cv2

def compress(filename):
    print("[INFO] Process started ... ")

#     for i in range(21) :
            
        # Import the image
    # print("[INFO] Image " + str(i+1) + " imported ...")
    print('[INFO] Image imported ... ')
    # orig_image = cv2.imread("../Dataset/DatasetToMask/" + str(i+1) + ".PNG")
    orig_image = cv2.imread("uploads/" + filename)

    # Fix downscaling size
    size = (256, 256)

    filename = filename.split('.')[0]
    # Resize original image and save as .tif
    fit_image = cv2.resize(orig_image, size)
    print("[INFO] Saving ...")
    cv2.imwrite("data/test/" + str(0) + ".tif", fit_image)


# compress()
# print("[INFO] Compression done... \n")
