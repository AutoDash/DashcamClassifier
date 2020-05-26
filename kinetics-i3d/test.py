import i3d
import tensorflow as tf
import numpy as np
import csv
import skvideo.io
import cv2
# labels
# 0 <- dashcam
# 1 <- not dashcam
############### Global Parameters ###############
# path
train_path = './videos/training/'
test_path = './videos/testing/'
demo_path = './videos/testing/'
default_model_path = './model/demo_model'
csv_path = './videos/data.csv'
save_path = './model/'
video_path = './dataset/videos/testing/positive/'
# batch_number
train_num = 126
test_num = 46
############## Train Parameters #################
VIDEO_FRAMES = 120
_IMAGE_SIZE = 224
# Parameters
num_classes = 2
learning_rate = 0.0001
n_epochs = 30
batch_size = 10
display_step = 10
shuf_buf = 100 # Buffer size for random shuffling. shuf_buf > dataset size gives perfect shuffling
prefetch_size = 1 # How many batches to fetch before current training step is complete


def tf_load_data(npy_path, label):
    def _load_data(npy_path, label):
        return np.load(npy_path), label

    return tf.py_func(_load_data, [npy_path, label], [tf.string, tf.string])

def data_augmentations(x, y):
    # TODO
    return x,y

def build_dataset(filepaths, labels):
    dataset = Dataset.from_tensor_slices((filepaths, labels))
    dataset = dataset.shuffle(shuf_buf, reshuffle_each_iteration=True)
    dataset = dataset.repeat(n_epochs)
    dataset = dataset.map(tf_load_data, num_parallel_calls=4)
    dataset = dataset.map(data_augmentations, num_parallel_calls=4)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(prefetch_size)

    return dataset

def to_npy(mp4):
    cap = cv2.VideoCapture(mp4)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))

    fc = 0
    ret = True

    while (fc < frameCount  and ret):
        ret, buf[fc] = cap.read()
        fc += 1

    cap.release()
    return buf

def read_csv(path):
    file_names = []
    labels = []
    with open(path) as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            X,Y = row # Likely to change
            file_names.append(X)
            labels.append(Y)

    return file_names, labels

"""
Usage:
dataset = build_dataset(*read_csv(csv_path))
iter = dataset.get_one_shot_iterator()
X, Y = iter.get_next()
out = model(X)
loss = Loss(X, Y)
opt = optimizer(loss)
Train loop:
def train:
    for epoch in range(n_epochs):
        for i in range(dataset_size):
            _, loss = sess.run(opt, loss)
            ...
"""
def convert_all_vids():
    file_names, labels = read_csv(csv_path)

    for file in file_names:
        filename = train_path + file + ".mp4"
        print(f"read video {filename}")

        video = to_npy(filename)
        print("converted")
        np.savez(file, video)
        print("saved")

def path(video):
    return train_path + video + ".npz"

def train():
    file_names, labels = read_csv(csv_path)

    print(f"read video {train_path + file_names[0] }")
    nparr, _ = tf_load_data(file_names[0], labels[0])
    print(nparr.shape)
    #dataset = build_dataset(paths, labels)
    classes = tf.placeholder("float", [None, num_classes])
    input = tf.placeholder(
        tf.float32,
        shape=(1, VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 3))

    model = i3d.InceptionI3d(num_classes=num_classes, final_endpoint='Logits')
    logits,_ = model(input, is_training=True)

    learning_rate = 0.01

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=classes))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(n_epochs):
            for i in range(len(file_names)):
                data, lbl = tf_load_data(path(file_names[i]), labels[i])
                _, loss = sess.run([optimizer, cost], feed_dict = {
                    input: data,
                    classes: labels[i]
                })


if __name__ == '__main__':
    train()
