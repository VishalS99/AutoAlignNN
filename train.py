from data import *
from unet import *

print("[INFO] Create training and test data ... \n")
data = dataProcess(256,256)
data.create_train_data()
data.create_test_data()
print("[INFO] Created... \n")

print("[INFO] Loading Model and trainin ... \n")
unet_model = myUnet()
unet_model.train()

print("[INFO] Saving masks ... \n")
unet_model.save_img()

print("[INFO] Training done... \n")
