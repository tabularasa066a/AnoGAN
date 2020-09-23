## http://cedro3.com/ai/keras-anogan-anomaly/
from keras.utils. generic_utils import Progbar
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, BatchNormalization, LeakyReLU, Reshape, Conv2DTranspose, Flatten
from keras.layers.core import Activation
from keras.optimizers import Adam
import keras.backend as K
import math, cv2
import numpy as np
import os
import shutil
from pathlib import Path

IMAGE_CHANNELS = 3

IMAGE_SIZE = 128  # mnistの画像サイズが28 x 28なのでとりあえず
IMAGE_SIZE_HALF = int(IMAGE_SIZE / 2)
IMAGE_SIZE_QUARTED = int(IMAGE_SIZE / 4)
IMAGE_SIZE_DOBLE = int(IMAGE_SIZE * 2)

## ファイルを保存するディレクトリが存在するかチェック(無かったら保存できないくそ→無かったら作成)
RESULTS_DIR = './results'
results_dir_is_exits = os.path.exists(RESULTS_DIR)
WEIGHTS_DIR = './weights'
weights_dir_is_exits = os.path.exists(WEIGHTS_DIR)
PREDICT_DIR = './predict'
predict_dir_is_exits = os.path.exists(PREDICT_DIR)


# ディレクトリがないときは作る
if not results_dir_is_exits:
    os.mkdir(RESULTS_DIR)
if not weights_dir_is_exits:
    os.mkdir(WEIGHTS_DIR)
if not predict_dir_is_exits:
    os.mkdir(PREDICT_DIR)

# ディレクトリが存在するときは空にする
if results_dir_is_exits:
    shutil.rmtree(RESULTS_DIR)
    os.mkdir(RESULTS_DIR)
    print(str(RESULTS_DIR) + '内を削除しました')
if weights_dir_is_exits:
    shutil.rmtree(WEIGHTS_DIR)
    os.mkdir(WEIGHTS_DIR)
    print(str(WEIGHTS_DIR) + '内を削除しました')
if predict_dir_is_exits:
    shutil.rmtree(PREDICT_DIR)
    os.mkdir(PREDICT_DIR)
    print(str(PREDICT_DIR) + '内を削除しました')

# Generator
class Generator(object):
    def __init__(self, input_dim, image_shape):
        inputs = Input((input_dim,))
        fc1 = Dense(input_dim=input_dim, units=IMAGE_SIZE * IMAGE_SIZE_QUARTED * IMAGE_SIZE_QUARTED)(inputs)
        fc1 = BatchNormalization()(fc1)
        fc1 = LeakyReLU(0.2)(fc1)
        fc2 = Reshape((IMAGE_SIZE_QUARTED, IMAGE_SIZE_QUARTED, IMAGE_SIZE), input_shape=(IMAGE_SIZE * IMAGE_SIZE_QUARTED * IMAGE_SIZE_QUARTED,))(fc1)
        up1 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(fc2)
        conv1 = Conv2D(64, (3, 3), padding='same')(up1)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation('relu')(conv1)
        up2 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv1)
        conv2 = Conv2D(image_shape[2], (5, 5), padding='same')(up2)
        outputs = Activation('tanh')(conv2)

        self.model = Model(inputs=[inputs], outputs=[outputs])

    def get_model(self):
        print(self.model.summary())
        return self.model

# Discriminator
class Discriminator(object):
    def __init__(self, input_shape):
        inputs = Input(input_shape)
        conv1 = Conv2D(64, (5, 5), padding='same')(inputs)
        conv1 = LeakyReLU(0.2)(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(128, (5, 5), padding='same')(pool1)
        conv2 = LeakyReLU(0.2)(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        fc1 = Flatten()(pool2)
        fc1 = Dense(1)(fc1)
        outputs = Activation('sigmoid')(fc1)

        self.model = Model(inputs=[inputs], outputs=[outputs])

    def get_model(self):
        print(self.model.summary())
        return self.model

# DCGAN
class DCGAN(object):
    def __init__(self, input_dim, image_shape):
        self.input_dim = input_dim
        self.d = Discriminator(image_shape).get_model()
        self.g = Generator(input_dim, image_shape).get_model()

    def compile(self, g_optim, d_optim):
        self.d.trainable = False
        self.dcgan = Sequential([self.g, self.d])
        self.dcgan.compile(loss='binary_crossentropy', optimizer=g_optim)
        self.d.trainable = True
        self.d.compile(loss='binary_crossentropy', optimizer=d_optim)

    def train(self, epochs, batch_size, X_train):
        g_losses = []
        d_losses = []
        for epoch in range(epochs):
            np.random.shuffle(X_train)
            n_iter = X_train.shape[0] // batch_size
            progress_bar = Progbar(target=n_iter)
            for index in range(n_iter):
                # create random noise -> N latent vectors
                noise = np.random.uniform(-1, 1, size=(batch_size, self.input_dim))

                # load real data & generate fake data
                image_batch = X_train[index * batch_size:(index + 1) * batch_size]
                for i in range(batch_size):
                    if np.random.random() > 0.5:
                        image_batch[i] = np.fliplr(image_batch[i])
                    if np.random.random() > 0.5:
                        image_batch[i] = np.flipud(image_batch[i])
                generated_images = self.g.predict(noise, verbose=0)

                # attach label for training discriminator
                X = np.concatenate((image_batch, generated_images))
                y = np.array([1] * batch_size + [0] * batch_size)

                # training discriminator
                d_loss = self.d.train_on_batch(X, y)

                # training generator
                g_loss = self.dcgan.train_on_batch(noise, np.array([1] * batch_size))

                progress_bar.update(index, values=[('g', g_loss), ('d', d_loss)])
            g_losses.append(g_loss)
            d_losses.append(d_loss)

            ## 生成画像の保存
            if (epoch+1)%10 == 0:
                image = self.combine_images(generated_images)
                image = (image + 1) / 2.0 * 255.0
                cv2.imwrite('./results/' + str(epoch) + ".png", image)
                print('./results配下に生成画像を保存')

            print('\nEpoch' + str(epoch) + " end")

            # save weights for each epoch
            if (epoch+1)%50 == 0:
                self.g.save_weights('./weights/generator_' + str(epoch) + '.h5', True)
                self.d.save_weights('./weights/discriminator_' + str(epoch) + '.h5', True)
                print('./weightss配下に重み値をh5形式で保存')

        return g_losses, d_losses

    def load_weights(self, g_weight, d_weight):
        self.g.load_weights(g_weight)
        self.d.load_weights(d_weight)

    def combine_images(self, generated_images):
        num = generated_images.shape[0]
        width = int(math.sqrt(num))
        height = int(math.ceil(float(num) / width))
        shape = generated_images.shape[1:4]
        image = np.zeros((height * shape[0], width * shape[1], shape[2]),
                         dtype=generated_images.dtype)
        for index, img in enumerate(generated_images):
            i = int(index / width)
            j = index % width
            image[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1], :] = img[:, :, :]
        return image

# AnoGAN
def sum_of_residual(y_true, y_pred):
    return K.sum(K.abs(y_true - y_pred))

class ANOGAN(object):
    def __init__(self, input_dim, g):
        self.input_dim = input_dim
        self.g = g
        g.trainable = False
        # Input layer cann't be trained. Add new layer as same size & same distribution
        anogan_in = Input(shape=(input_dim,))
        g_in = Dense((input_dim), activation='tanh', trainable=True)(anogan_in)
        g_out = g(g_in)
        self.model = Model(inputs=anogan_in, outputs=g_out)
        self.model_weight = None

    def compile(self, optim):
        self.model.compile(loss=sum_of_residual, optimizer=optim)
        K.set_learning_phase(0)

    def compute_anomaly_score(self, x, iterations=300):
        z = np.random.uniform(-1, 1, size=(1, self.input_dim))

        # learning for changing latent
        loss = self.model.fit(z, x, batch_size=1, epochs=iterations, verbose=0)
        loss = loss.history['loss'][-1]
        similar_data = self.model.predict_on_batch(z)

        return loss, similar_data



def load_imgs(root_dir, class_name):
    ## DAGMファイル読み取る用改修
    img_paths = []
    images = []
    # DAGM画像ファイル読み取り
    # class_names = os.listdir(root_dir)
    img_names = os.listdir(os.path.join(root_dir, class_name))  # 画像ファイルの名前を全て取得しimg_namesに格納
    for img_name in img_names:
        if 'png' in img_name:  # .DL_shareやら*.txtが存在するときそれらをスキップ
            img_paths.append(os.path.abspath(os.path.join(root_dir, class_name, img_name)))
    for img_path in img_paths:
        # img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # グレースケール読み込み
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, dsize=(IMAGE_SIZE, IMAGE_SIZE))
        images.append(img)
    images = np.array(images)
    images.reshape(images.shape[0], IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS)

    return images

# train
if __name__ == '__main__':
    batch_size = 16
    epochs = 100
    # epochs = 1
    input_dim = 100
    g_optim = Adam(lr=0.0001, beta_1=0.5, beta_2=0.999)
    d_optim = Adam(lr=0.0001, beta_1=0.5, beta_2=0.999)
    ## DAGMファイル読み取る用改修
    img_data_Normal = load_imgs(str(Path('../DAGM2007/Normal').resolve()), 'Class1')
    # img_data_Normal = img_data_Normal.astype(np.float32) / 255
    img_data_Normal = (img_data_Normal.astype(np.float32) - 127.5) / 127.5  # [0, 255] -> [-1, 1], こっちでやってみる
    img_data_Normal = img_data_Normal.reshape(img_data_Normal.shape[0], IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS)  # (1000, 28, 28, 1)

    img_data_withDefects = load_imgs(str(Path('../DAGM2007/withDefects').resolve()), 'Class1_def')
    # img_data_withDefects = img_data_withDefects.astype(np.float32) / 255
    img_data_withDefects = (img_data_withDefects.astype(np.float32) - 127.5) / 127.5  # [0, 255] -> [-1, 1], こっちでやってみる
    img_data_withDefects = img_data_withDefects.reshape(img_data_withDefects.shape[0], IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS)  # (150, 28, 28, 1)
    # # 訓練用画像枚数を画像総数の8割とし、テスト画像枚数を残りの２割として抽出
    # train_image_num = int(img_data_Normal.shape[0] * 0.8)
    # x_train = img_data_Normal[0:train_image_num]
    # x_test = img_data_Normal[train_image_num:]

    # x_train_1 = img_data_Normal
    # x_test_9 = img_data_withDefects

    # TODO: 訓練用画像(正常系のみ): 850枚, テスト用画像(正常系+異常系): 150 + 150 = 300枚
    normal_imgs4train = 850
    x_train_1 = img_data_Normal[0:normal_imgs4train]  # TODO: x_train_1には正常系の画像のみ入れること

    ## テスト用画像作成(150 + 150 = 300枚)
    x_test_Normal = img_data_Normal[normal_imgs4train:]
    x_test_withDefects = img_data_withDefects
    # テスト画像生成(150 + 150)
    # TODO: ここいらでラベル付けが必要
    y_test = []
    for i in range(0, len(x_test_Normal)):
        y_test.append(1)
    for i in range(0, len(x_test_withDefects)):
        y_test.append(9)
    x_test = np.concatenate([x_test_Normal, x_test_withDefects])  # テスト用画像生成
    # x_test = np.random.permutation(x_test)  # 300枚をシャッフル->とりあえず」オフ

    ## 学習データの作成
    # x_train_1 = []  # 正常系の画像のみ学習のため、1とラベル付けされているものみ抽出
    # for i in range(len(x_train)):
    #     if y_train[i] == 1:
    #     #    print("x_train[i]", x_train[i].shape)
    #        x_train_1.append(x_train[i].reshape((28, 28, 1)))
    #     #    print("x_train_1", np.array(x_train_1).shape)
    # x_train_1 = np.array(x_train_1)
    print("x_train_1", x_train_1.shape)
    print("train data:",len(x_train_1))

    ## 評価データの作成
    cnt = 0
    x_test_9, y = [], []
    for i in range(len(x_test)):
        if y_test[i] == 1 or y_test[i] == 9:
           x_test_9.append(x_test[i].reshape((IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS)))
           y.append(y_test[i])
           cnt +=1
        #    if cnt == 100:
        #       break
           if cnt == 300:
              break
    x_test_9 = np.array(x_test_9)
    print("test_data:",len(x_test_9))
    input_shape = x_train_1[0].shape
    print(input_shape)
    X_test_original = x_test_9.copy()


    # train generator & discriminator
    dcgan = DCGAN(input_dim, input_shape)
    dcgan.compile(g_optim, d_optim)
    g_losses, d_losses = dcgan.train(epochs, batch_size, x_train_1)
    with open('loss.csv', 'w') as f:
        for g_loss, d_loss in zip(g_losses, d_losses):
            f.write(str(g_loss) + ',' + str(d_loss) + '\n')

# test
K.set_learning_phase(1)

def denormalize(X):
    gen_imgs = gen_imgs * 127.5 + 127.5  # [-1, 1] -> [0, 255], こっち？
    # return ((X + 1.0)/2.0*255.0).astype(dtype=np.uint8)
    return gen_imgs

if __name__ == '__main__':
    iterations = 100
    # iterations = 10000
    # input_dim = 30
    input_dim = 100

    anogan_optim = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
    print(input_shape)
    # load weights
    dcgan = DCGAN(input_dim, input_shape)
    dcgan.load_weights('weights/generator_99.h5', 'weights/discriminator_99.h5')  #3999.99

    for i, test_img in enumerate(x_test_9):
        test_img = test_img[np.newaxis,:,:,:]
        anogan = ANOGAN(input_dim, dcgan.g)
        anogan.compile(anogan_optim)
        anomaly_score, generated_img = anogan.compute_anomaly_score(test_img, iterations)

        generated_img = denormalize(generated_img)
        imgs = np.concatenate((denormalize(test_img[0]), generated_img[0]), axis=1)
        cv2.imwrite('predict' + os.sep + str(int(anomaly_score)) + '_' + str(i) + '.png', imgs)
        print(str(i) + ' %.2f'%anomaly_score)

        if y[i] == 1 :
           with open('scores_Normal.txt', 'a') as f:
                f.write(str(anomaly_score) + '\n')
        else:
           with open('scores_Defects.txt', 'a') as f:
                f.write(str(anomaly_score) + '\n')

    # plot histgram
    import matplotlib.pyplot as plt
    import csv

    x =[]
    with open('scores_Normal.txt', 'r') as f:
         reader = csv.reader(f)
         for row in reader:
             row = int(float(row[0]))
             x.append(row)
    y =[]
    with open('scores_Defects.txt', 'r') as f:
         reader = csv.reader(f)
         for row in reader:
             row = int(float(row[0]))
             y.append(row)

    plt.title("Histgram of Score")
    plt.xlabel("Score")
    plt.ylabel("freq")
    plt.hist(x, bins=40, alpha=0.3, histtype='stepfilled', color='r', label='Normal')
    plt.hist(y, bins=40, alpha=0.3, histtype='stepfilled', color='b', label='Defects')
    plt.legend(loc=1)
    plt.savefig("histgram.png")
    plt.show()
    plt.close()
