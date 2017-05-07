import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
sample = pd.read_csv("datasets/sample_submission.csv")
#test_images = (pd.read_csv("datasets/test.csv").values).astype('float32')
test_pixel = (pd.read_csv("datasets/train.csv").ix[:,1:].values).astype('float32')
print(test_pixel.shape)
test_image = test_pixel.reshape(test_pixel.shape[0], 28, 28)
#print(test_image)

print (range(0,3))
for i in range(32000, 32001):
    plt.subplot(1,1,1)
    plt.imshow(test_image[i], cmap=plt.get_cmap('gray'))
    plt.title('a')
plt.show()

train_label = (pd.read_csv("datasets/train.csv").ix[:,0].values).astype('int32')
# print(train_label)
# print(train_label.shape[0])
# blankk = np.zeros((train_label.shape[0], 10),dtype=np.int)
# for i in range(0,train_label.shape[0]):
#     blankk[i][train_label[i]] = 1
#
# print(blank)


def OneHotEncoding(labels, encode_space):
    num_of_labels = labels.shape[0]
    encoded_labels = np.zeros((num_of_labels, encode_space), dtype=np.int)
    for i in range(0, num_of_labels):
        encoded_labels[i][labels[i]] = 1
    return encoded_labels

# encoded = OneHotEncoding(train_label, 10)
# print(encoded)
# plt.plot(encoded[0])
# plt.xticks(range(1, 12))
# plt.show()

np.zeros()



# #tomorrow :
# 1 layer network
# multi layer network
# add fectures
# find paper

