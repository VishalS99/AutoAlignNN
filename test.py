from unet import *
from data import *
from keras.models import load_model
from keras import backend as K
import glob
import os

def unetProcessing():
    # load unet model
    print("[INFO] Loading model... \n")
    model = load_model('./unet.hdf5')

    print("[INFO] Preparing test data... \n")
    # prepare the test data
    data = dataProcess(256,256)
    test = data.create_test_data()

    print("[INFO] Loading test data... \n")
    # load the test data to the model
    imgs_train, imgs_mask_train, imgs_test = myUnet().load_data()

    print("[INFO] Predict Masks... \n")
    # predict the masks for the test data
    imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)

    print("[INFO] Saving mask as numpy array... \n")
    # save the masks as numpy array
    np.save('results/imgs_mask_test.npy', imgs_mask_test)


    print("[INFO] Saving masks as images... \n")
    # convert the array of masks to images and save in results
    print("array to image")
    imgs=np.load('results/imgs_mask_test.npy')

    for i in range(imgs.shape[0]):
        img = imgs[i]
        img = array_to_img(img)
        img.save("results/%d.jpg"%(i))
    
    K.clear_session()
    files = glob.glob('data/test/*.tif')
    for f in files:
        os.remove(f)
    
    print("[INFO] Operation done. Segmentation done. Image croping starts... \n")
# unetProcessing()