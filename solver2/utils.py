# load one image
# convert to array
# split
# if split correct (5)
# save letters to folder by name as csv for (dataset like mnst)
# train and evaluate
# try out for real.
import warnings
import time
import cv2
import numpy as np
import csv
import json
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow import keras
import random
import os

# LABELS_CSV = 'labels.csv'
# CAPTCHA_XML = 'images/captcha_xml/'
# CAPTCHA_PNG = 'images/captcha_png/'
# DATASET = 'dataset.csv'
# MODEL = 'model'
LABELS_CSV = 'labels2.csv'
CAPTCHA_XML = 'images/captcha_xml2/'
CAPTCHA_PNG = 'images/captcha_png2/'
DATASET = 'dataset2.csv'
MODEL = 'model2'

warnings.simplefilter(action='ignore', category=FutureWarning)


def convert_binary(img):
    im_bw = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    (retval, mask_im) = cv2.threshold(im_bw, 0, 255, 0)
    return retval, mask_im


def show_image(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)


def load_image(name):
    return cv2.imread(name)


def dilate_and_erode(img):
    # show_image(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    er = cv2.erode(img, kernel, iterations=1)
    # show_image(er)
    # er = cv2.dilate(er, kernel, iterations=1)

    return er


def find_contours_save(binary):
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours, hierarchy


def load_csv_labels():
    with open(LABELS_CSV, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        # get header from first row
        headers = next(reader)
        # get all the rows as a list
        data = list(reader)
        # transform data into numpy array
        data = np.array(data)
        img_names = data[:, [0]]
        labels = data[:, [3]]
        contour_points = data[:, [4, 5, 6, 7]]
    return img_names, labels, contour_points


def load_dataset():
    with open(DATASET, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        # get header from first row
        # get all the rows as a list
        data = np.array(list(reader))
        # transform data into numpy array
        # data = np.array(data)
        img_names = np.array(data[:, [0]]).transpose()[0]
        labels = np.array(data[:, [1]])

        l = []
        for l_sin in labels:
            arr = json.loads(l_sin[0])
            l.append(arr)
        l = np.array(l)

        # contour_points = data[:, [4, 5, 6, 7]]
    return img_names, l


def convert_to_vector(l_dict, labels):
    l = []
    for label in labels:
        vector = np.zeros(len(l_dict))
        vector[l_dict[label]] = 1
        l.append(vector)
    return l


def split_largest(final_contours, n):
    final_contours.sort(reverse=True, key=lambda x: x[3])
    c = final_contours.pop(0)
    # show_image(c[1] * 255)
    x, y, w, h = c[0], c[2], c[3], c[4]
    w = int(w / n + 1)
    # get ratio of cnt-area and box-area
    # print("SPLITTING TO: ", n)
    for i in range(0, n + 1):
        final_contours.append((x + w * i, c[1][0:0 + h, w * i:int(w * (i + 1))], y, int(w), h))

    final_contours.sort(key=lambda x: x[0])
    return final_contours


def get_u_def_contours(img_id, x_min, y_min, x_max, y_max):
    # print(CAPTCHA_PNG + img_id)
    im = load_image(CAPTCHA_PNG + img_id)
    # print(im)
    a, b = convert_binary(im)
    roi = b[y_min:y_max, x_min:x_max]
    return (x_min, roi, y_min, x_max - x_min, y_max - y_min)


def flood_fill(img):
    # img=dilate_and_erode(img)
    # show_image(img*255)
    # add all neighbours untill done, keep track of x min, y min.
    visited = {}
    neighbours = [(-1, -1), (-1, 1), (1, -1), (-1, 0), (0, -1), (0, 1), (1, 0), (1, 1)]
    contour_list = []
    for i, l in enumerate(img):

        for j, v in enumerate(l):
            node = (i, j)
            # print("node: ",node)
            if node in visited:
                continue
            open_heap = [node]
            flood = []
            while len(open_heap) > 0:
                node = open_heap.pop()
                for n in neighbours:
                    point = (n[0] + node[0], n[1] + node[1])
                    if 0 <= point[0] < 60 and 0 <= point[1] < 160 and point not in visited:
                        if img[point[0]][point[1]] == 0:
                            flood.append(point)
                            open_heap.append(point)
                    visited.update({node: True})
            if len(flood) > 40:
                current_x_max = 0
                current_y_max = 0
                current_x_min = 200
                current_y_min = 200
                for node in flood:
                    if node[1] > current_x_max:
                        current_x_max = node[1]
                    elif node[1] < current_x_min:
                        current_x_min = node[1]
                    if node[0] > current_y_max:
                        current_y_max = node[0]
                    elif node[0] < current_y_min:
                        current_y_min = node[0]
                padding = 2
                if current_x_max < 160 - padding:
                    current_x_max += padding
                if current_x_min > padding:
                    current_x_min -= padding
                if current_y_max < 60 - padding:
                    current_y_max += padding
                if current_y_min > padding:
                    current_y_min -= padding
                w = current_x_max - current_x_min
                h = current_y_max - current_y_min
                x = current_x_min
                y = current_y_min
                # final_contours.append((x, roi, y, w, h))
                roi = img[y:current_y_max, x:current_x_max]

                contour_list.append((x, roi, y, w, h))

    contour_list.sort(key=lambda x: x[0])
    br = 0
    while len(contour_list) < 6:
        br += 1
        print("SPLITTING")
        contour_list = split_largest(contour_list, 6 - len(contour_list))
        if br > 3:
            break
    print(len(contour_list))
    return contour_list


def get_contours(img):
    # b = dilate_and_erode(img)
    # show_image(b * 255)
    contours, higherarchy = find_contours_save(b)
    final_contours = []
    previous = (0, 0, 0, 0)
    for idx, c in enumerate(contours):
        x, y, w, h = cv2.boundingRect(c)
        # print("===============")
        # print(w, h)
        # cv2.drawContours(b, contours, idx)
        # show_image(b)
        if not y == 0:
            sum = (w + h)
            ratio = (w / h)
        else:
            continue
        # show_image(c)
        if sum < 22 or (previous[0] + 2 < x < (previous[0] + previous[2]) - 2):
            # show_image(b[y:y + h, x:x + w])

            continue
        else:
            temp = np.zeros_like(b)
            cv2.drawContours(temp, contours, idx, 1, -1)
            area_cnt = np.sum(temp)
            area_box = w * h

            # get ratio of cnt-area and box-area
            # show_image(b[y:y + h, x:x + w])
            ratio = float(area_cnt) / area_box
            # print(ratio)
            if 0.9 > ratio > 0.2:
                roi = b[y:y + h, x:x + w]
                final_contours.append((x, roi, y, w, h))
            else:
                pass
        previous = (x, y, w, h)
    final_contours.sort(key=lambda x: x[0])
    print(len(final_contours))
    while len(final_contours) < 6:
        print("SPLITTING")
        final_contours = split_largest(final_contours, 6 - len(final_contours))

    return final_contours


def preprocess_captcha(img):
    # img = dilate_and_erode(img)
    img = convert_binary(img)[1]
    # print(img)
    for i, l in enumerate(img):
        for j, v in enumerate(l):
            if img[i][j] >= 254:
                img[i][j] = 1
            else:
                img[i][j] = 0
    # print(img)
    return img


def preprocess_letter_for_pred(img):
    # img = dilate_and_erode(img)
    img = cv2.resize(img, (40, 40))
    img = np.reshape(img, (1, 40, 40, 1))
    return img


def preprocess_letter(img):
    print("PREPROCESSING /resizing and setting to 1")
    # img = dilate_and_erode(img)
    show_image(img)
    img = cv2.resize(img, (40, 40))

    for i, l in enumerate(img):
        for j, v in enumerate(l):
            if img[i][j] >= 254:
                img[i][j] = 1
            else:
                img[i][j] = 0
        # print(img)
    show_image(img * 255)
    # adding a 3rd dimension to the image
    img = np.expand_dims(img, axis=2)
    print("END PREPROCESSING")
    return img


def gen_dataset():
    img_id, labels, contour_points = load_csv_labels()
    contours = []
    sum_fail = 0
    final_data = []
    with open(DATASET, mode='w+') as dataset:
        dataset_writer = csv.writer(dataset, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
        dataset_len = len(img_id)
        i = 0
        while i < dataset_len:
            label = labels[i][0]
            val = img_id[i][0]
            contour_p = contour_points[i]
            contour = get_u_def_contours(val, int(contour_p[0]), int(contour_p[1]), int(contour_p[2]),
                                         int(contour_p[3]))
            img = preprocess_letter(contour[1])
            dataset_writer.writerow([label, img.tolist(), contour[0], contour[2], contour[3], contour[4]])
            final_data.append([label, img.tostring()])
            i += 1
            print('\r', i / dataset_len)
            # cv2.imwrite(contours[i % 5][1], dir_l+i.__str__(), 'jpg')
    return final_data


def train_model():
    print("GENERATING DATASET")
    gen_dataset()
    labels, data = load_dataset()
    # print(dataset[0])

    model = Sequential()
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2,
                                                        random_state=random.randrange(0, 1000))
    label_set = np.unique(y_train)
    l_dict = {}
    for i, a in enumerate(label_set):
        l_dict.update({a: i})
    print("-----------------------------")
    print(y_test[0])
    print(l_dict)
    print("CONVERTING TO VECTOR")
    y_train = np.array(convert_to_vector(l_dict, y_train))
    y_test = np.array(convert_to_vector(l_dict, y_test))
    print(y_train[0])
    print(y_test[0])
    print("=========================")
    # First convolutional layer with max pooling
    model = Sequential()
    # convolutional layer with rectified linear unit activation
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=(40, 40, 1)))
    # 32 convolution filters used each of size 3x3
    # again
    model.add(Conv2D(64, (3, 3), activation='relu'))
    # 64 convolution filters used each of size 3x3
    # choose the best features via pooling
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # randomly turn neurons on and off to improve convergence
    model.add(Dropout(0.25))
    # flatten since too many dimensions, we only want a classification output
    model.add(Flatten())
    # fully connected to get all relevant data
    model.add(Dense(128, activation='relu'))
    # one more dropout for convergence' sake :)
    model.add(Dropout(0.5))
    # output a softmax to squash the matrix into output probabilities
    model.add(Dense(len(label_set), activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    batch_size = 20
    num_epoch = 10
    # import time
    # for k, b in enumerate(x_train):
    #     print(label_set[[i for i, value in enumerate(y_train[k]) if value == 1]])
    #     print(b)
    #     time.sleep(10)

    # model training
    model_log = model.fit(x_train, y_train,
                          batch_size=batch_size,
                          epochs=num_epoch,
                          verbose=1,
                          validation_data=(x_test, y_test))
    # serialize model to JSON
    model_json = model.to_json()
    with open(MODEL + '.json', "w") as json_file:
        json_file.write(model_json)
        # serialize weights to HDF5
    model.save_weights(MODEL + ".h5")
    print("Saved model to disk")

    # later...

    json_file.close()


def load_model():
    # load json and create model
    json_file = open(MODEL + '.json', 'r')
    loaded_model_json = json_file.read()
    loaded_model = keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(MODEL + '.h5')
    print("Loaded model from disk")
    loaded_model.compile(loss=keras.losses.categorical_crossentropy,
                         optimizer=keras.optimizers.Adadelta(),
                         metrics=['accuracy'])
    return loaded_model


class Predictor:
    def __init__(self):
        self.model = load_model()

        labels, data = load_dataset()
        x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2,
                                                            random_state=random.randrange(0, 1000))
        label_set = np.unique(labels)

        self.l_dict = {}
        for i, a in enumerate(label_set):
            self.l_dict.update({i: a})

    def predict(self, img):
        img = preprocess_captcha(img)
        contours = flood_fill(img)
        letters = ''
        for c in contours:
            im_pr = preprocess_letter_for_pred(c[1])
            # show_image(c[1] * 255)
            letter = self.l_dict[self.model.predict_classes(im_pr)[0]]
            letters += letter
        # show_image(img * 255)
        return letters


train_model()
# evaluate loaded model on test data
# base_dir = CAPTCHA_PNG
# paths = os.listdir(base_dir)
# predictor = Predictor()
# correct = 0
# total = 0
# for p in paths:
#     total += 1
#     r_n = p.split('.')[0]
#     im = load_image(base_dir + '/' + p)
#     ans = predictor.predict(im)
#     print('----------------')
#     print(ans)
#     if ans == r_n:
#         correct += 1
# print(correct / total)
