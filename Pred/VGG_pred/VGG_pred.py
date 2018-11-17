import tensorflow as tf
from scipy.misc import imread, imresize
import model.VGG.VGG16_model as model
import numpy as np
import os

meta_filepath = 'F:\PHI\TASK7\\vgg1\logs\model.ckpt-2000.meta'
ckpt_filepath = 'F:\PHI\TASK7\\vgg1\logs\model.ckpt-2000'

Predfile_dir = 'E:\\PycharmProjects\\PHI Challenge 2018\\Task7\\data\\test'
pred_resultfile_savepath = ""
pred_resultfile_name = "./VGG-6000.txt"

class_number = 4

imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])

keep_prob = tf.placeholder(tf.float32)

sess = tf.Session()
vgg = model.vgg16(imgs, keep_prob,class_number)
Task2 = vgg.probs
saver = vgg.saver()

# module_file = tf.train.latest_checkpoint('./logs/')
module_file = tf.train.import_meta_graph(meta_filepath)

saver.restore(sess,ckpt_filepath )

array = []
for root, sub_folders, files in os.walk(Predfile_dir):
    i = 0

    #cat = 0
    #dog = 0

    # Object_level = 0
    # Pixel_level = 0
    # Structural_level = 0
    file_list= []
    pred_list = []


    for name in files:
        i += 1
        filepath = os.path.join(root, name)

        try:
            img1 = imread(filepath, mode='RGB')
            img1 = imresize(img1, (224, 224))
        except:
            print("remove", filepath)

        prob = sess.run(Task2, feed_dict={vgg.imgs: [img1], keep_prob :1})

        max_index = np.argmax(prob)
        # if max_index == 0:
        #     cat += 1
        # else:
        #     dog += 1
        # if i % 50 == 0:
        #     acc = (cat * 1.)/(dog + cat)
        file_list.append(name)
        pred_list.append(max_index)
        print("the %s is %d" %(name,max_index))
    # print(file_list)
    # print(pred_list)
array.append(file_list)
array.append(pred_list)
# print(array)
temp = np.array(array)
temp = temp.transpose()

save_path = os.path.join(pred_resultfile_savepath,  pred_resultfile_name)
np.savetxt(save_path, temp,fmt='%s')
print()
