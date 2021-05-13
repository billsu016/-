import tensorflow as tf
import numpy as np

# A = tf.constant([[1, 2], [3, 4]])
# B = tf.constant([[2, 3], [4, 4]])
# C = tf.matmul(A, B)
#
# print(C)
#
# ### 基础实例1--线性回归
# # 考虑某城市2013-2017年的房价
# X_raw = np.array([2013, 2014, 2015, 2016, 2017], dtype=np.float32)
# y_raw = np.array([12000, 14000, 15000, 16500, 17500], dtype=np.float32)
#
# X = (X_raw - X_raw.min()) / (X_raw.max() - X_raw.min())
# Y = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min())
#
# X = tf.constant(X)
# Y = tf.constant(Y)
#
# a = tf.Variable(initial_value=0.)
# b = tf.Variable(initial_value=0.)
# variables = [a, b]
#
# num_epoch = 10000
# optimizer = tf.keras.optimizers.SGD(learning_rate=5e-4)
# for e in range(num_epoch):
#     with tf.GradientTape() as tape:  # 自动记录求导器
#         y_pred = a * X + b
#         loss = tf.reduce_sum(tf.square(y_pred - Y))
#     grads = tape.gradient(loss, variables)
#     # 根据梯度更新参数
#     optimizer.apply_gradients(grads_and_vars=zip(grads, variables))
#
# print(grads, variables)

### 基础实例2--多层感知机(MLP)
# 1、使用tf.keras.datasets获得数据并预处理

class MNISTLoader():
    def __init__(self):
        mnist = tf.keras.datasets.mnist
        (self.train_data, self.train_label), (self.test_data, self.test_label) = mnist.load_data()
        # MNIST中的图像为0-255的数字，将其归一化到0-1间的浮点数，并在最后增加一维作为颜色通道
        self.train_data = np.expand_dims(self.train_data.astype(np.float32) / 255.0, axis=-1)  # [60000, 28, 28, 1]
        self.test_data = np.expand_dims(self.test_data.astype(np.float32) / 255.0, axis=-1)  # [10000, 28, 28, 1]
        self.train_label = self.train_label.astype(np.int32)  # [60000]
        self.test_label = self.test_label.astype(np.int32)  # [10000]
        self.num_train_data, self.num_test_data = self.train_data.shape[0], self.test_data.shape[0]

    def get_batch(self, batch_size):
        # 从数据集中抽取batch_size个元素并返回
        index = np.random.randint(0, self.num_train_data, size=batch_size)
        return self.train_data[index, :], self.train_label[index]

# 2、使用tf.keras.Model和tf.keras.layers构建模型

class MLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()  # Flatten层将除第一维(batch_size)以外的维度展平
        self.dense1 = tf.keras.layers.Dense(units=100, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=10)  # 若不指定activation函数，则默认为线性变化

    def call(self, input):  # [batch_size, 28, 28, 1]
        x = self.flatten(input)  # [batch_size, 784]
        x = self.dense1(x)  # [batch_size, 100]
        x = self.dense2(x)  # [batch_size, 10]
        output = tf.nn.softmax(x)  # 对模型的原始输出进行归一化，并凸显原始向量中的最大值
        return output

# 3、构建模型训练流程，使用tf.keras.losses计算损失函数，并使用tf.keras.optimizer优化模型

# 定义模型的超参数
num_epochs = 5  # 学习5轮
batch_size = 50  # 每批50个样本
learning_rate = 0.001

# 实例化类
model = MLP()  # 实例化模型类
data_loader = MNISTLoader()  # 实例化数据读取类
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)  # 实例化优化器

# 迭代训练模型
num_batches = int(data_loader.num_train_data // batch_size * num_epochs)
# '/'表示浮点数除法，返回浮点结果；'//'表示整数除法
for batch_index in range(num_batches):
    X, y = data_loader.get_batch(batch_size)
    with tf.GradientTape() as tape:
        y_pred = model(X)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
        # y_ture是数字编码，而y_pred是one-hot编码（50行--batch_size=50，10列的矩阵）
        # tf.keras中，有两个交叉熵相关的损失函数:(1)tf.keras.losses.categorical_crossentropy 和(2)tf.keras.losses.sparse_categorical_crossentropy
        # 其中，若真实标签是数字编码，则使用函数(2)；若真实标签是one-hot编码，则使用函数(1)
        # 本例中，y_ture为数字编码，故使用sparse
        loss = tf.reduce_mean(loss)
        # 原输出loss为50行向量，对各行求平均
        print('batch %d: loss %f'%(batch_index, loss.numpy()))
    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

# 4、构建模型评估流程，使用tf.keras.metrics计算评估指标

sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()  # 实例化评估器
num_batches = int(data_loader.num_test_data // batch_size)
for batch_index in range(num_batches):
    start_index, end_index = batch_index * batch_size, (batch_index + 1) * batch_size
    y_pred = model.predict(data_loader.test_data[start_index: end_index, :])
    sparse_categorical_accuracy.update_state(y_true=data_loader.test_label[start_index:end_index], y_pred=y_pred)
    # 通过update_state()方法向评估器输入y_pred和y_true参数
print('test accuracy: %f' % sparse_categorical_accuracy.result())