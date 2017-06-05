import pandas as pd
import matplotlib.pyplot as plt

train_data = (pd.read_csv("~/Developer/kaggle.Digit_Recognizer/datasets/train.csv")
              .iloc[:32000, 1:].values).astype('float32')
train_label = (pd.read_csv("~/Developer/kaggle.Digit_Recognizer/datasets/train.csv")
               .iloc[:32000, 0].values).astype('int32')
test_data = (pd.read_csv("~/Developer/kaggle.Digit_Recognizer/datasets/train.csv")
             .iloc[32000:, 1:].values).astype('float32')
test_label = (pd.read_csv("~/Developer/kaggle.Digit_Recognizer/datasets/train.csv")
              .iloc[32000:, 0].values).astype('int32')
train_imgs = train_data.reshape(train_data.shape[0], 28, 28)
test_imgs = test_data.reshape(test_data.shape[0], 28, 28)
for i in range(0, train_imgs.shape[0]):
    plt.clf()
    #plt.imshow(train_imgs[i], cmap=plt.get_cmap('gray'))
    plt.imsave("[" + str(i + 100000) + " , " + str(train_label[i]) + "].png", train_imgs[i], cmap=plt.get_cmap('gray'))
    if i%(train_imgs.shape[0]/100)==0:
        print(str(i/(train_imgs.shape[0]/100))+"% train")

for i in range(0, test_imgs.shape[0]):
    plt.clf()
   # plt.imshow(test_imgs[i], cmap=plt.get_cmap('gray'))
    plt.imsave("[" + str(i + 200000) + " , " + str(test_label[i]) + "].png", test_imgs[i], cmap=plt.get_cmap('gray'))
    if i%(test_imgs.shape[0]/100)==0:
        print(str(i/(test_imgs.shape[0]/100))+"% test")
