import numpy as np
import tensorflow as tf
#import global_variable


class vgg16:
    def __init__(self, imgs,keep_prob,class_number):
        self.parameters = []
        self.imgs = imgs
        self.class_number = class_number
        self.convlayers()
        self.fc_layers(keep_prob)

        self.probs = self.fc8

    def saver(self):
        return tf.train.Saver(max_to_keep=0)

    def maxpool(self,name,input_data, trainable):
        out = tf.nn.max_pool(input_data,[1,2,2,1],[1,2,2,1],padding="SAME",name=name)
        return out

    def conv(self,name, input_data, out_channel, trainable):
        in_channel = input_data.get_shape()[-1]
        with tf.variable_scope(name):
            kernel = tf.get_variable("weights", [3, 3, in_channel, out_channel], dtype=tf.float32,trainable=True)
            biases = tf.get_variable("biases", [out_channel], dtype=tf.float32,trainable=True)
            conv_res = tf.nn.conv2d(input_data, kernel, [1, 1, 1, 1], padding="SAME")
            res = tf.nn.bias_add(conv_res, biases)
            out = tf.nn.relu(res, name=name)
        self.parameters += [kernel, biases]
        return out

    def fc(self,name,input_data,out_channel,trainable = True):
        shape = input_data.get_shape().as_list()
        if len(shape) == 4:
            size = shape[-1] * shape[-2] * shape[-3]
        else:size = shape[1]
        input_data_flat = tf.reshape(input_data,[-1,size])
        with tf.variable_scope(name):
            weights = tf.get_variable(name="weights",shape=[size,out_channel],dtype=tf.float32,trainable = trainable)
            biases = tf.get_variable(name="biases",shape=[out_channel],dtype=tf.float32,trainable = trainable)
            res = tf.matmul(input_data_flat,weights)
            out = tf.nn.relu(tf.nn.bias_add(res,biases))
        self.parameters += [weights, biases]
        return out

    def convlayers(self):
        # zero-mean input
        #conv1
        self.conv1_1 = self.conv("conv1re_1",self.imgs,64,trainable=True)
        self.conv1_2 = self.conv("conv1_2",self.conv1_1,64,trainable=True)
        self.pool1 = self.maxpool("poolre1",self.conv1_2,trainable=True)

        #conv2
        self.conv2_1 = self.conv("conv2_1",self.pool1,128,trainable=True)
        self.conv2_2 = self.conv("convwe2_2",self.conv2_1,128,trainable=True)
        self.pool2 = self.maxpool("pool2",self.conv2_2,trainable=True)

        #conv3
        self.conv3_1 = self.conv("conv3_1",self.pool2,256,trainable=True)
        self.conv3_2 = self.conv("convrwe3_2",self.conv3_1,256,trainable=True)
        self.conv3_3 = self.conv("convrew3_3",self.conv3_2,256,trainable=True)
        self.pool3 = self.maxpool("poolre3",self.conv3_3,trainable=True)

        #conv4
        self.conv4_1 = self.conv("conv4_1",self.pool3,512,trainable=True)
        self.conv4_2 = self.conv("convrwe4_2",self.conv4_1,512,trainable=True)
        self.conv4_3 = self.conv("conv4rwe_3",self.conv4_2,512,trainable=True)
        self.pool4 = self.maxpool("pool4",self.conv4_3,trainable=True)


        #conv5
        self.conv5_1 = self.conv("conv5_1",self.pool4,512,trainable=True)
        self.conv5_2 = self.conv("convrwe5_2",self.conv5_1,512,trainable=True)
        self.conv5_3 = self.conv("conv5_3",self.conv5_2,512,trainable=True)
        self.pool5 = self.maxpool("poorwel5",self.conv5_3,trainable=True)

    def fc_layers(self,keep_prob):

        self.fc6 = self.fc("fc6", self.pool5, 4096,trainable=True)
        self.fc6_drop = tf.nn.dropout(self.fc6, keep_prob, name='fc6_drop')

        self.fc7 = self.fc("fc7", self.fc6_drop, 4096,trainable=True)
        self.fc7_drop = tf.nn.dropout(self.fc7, keep_prob, name="fc7_drop")

        self.fc8 = self.fc("fc8", self.fc7_drop, self.class_number)

    def load_weights(self, weight_file, sess):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            if i not in [30,31]:
                sess.run(self.parameters[i].assign(weights[k]))
        print("-----------all done---------------")
