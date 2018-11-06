import os
import numpy as np
import tensorflow as tf
import model.VGG.VGG16_model as model
import Trian.VGG_train.create_and_read_TFRecord2 as reader2

logs_train_dir = "F://PHI//TASK7//vgg//logs"
data_dir = "./Edataset/"
MAX_STEP = 400000
BATCH = 60
Learning_Rate = 0.01 # 0.01 dropout 1 seg0.8
Sample_Size = 5000
num_classes = 4
seg_ratio = 0.8

# 将所有数据分为训练集和验证集
def segmentation(data, label, ratio=0.9):
    s = np.int(len(label) * ratio)
    x_train = data[:s]
    y_train = label[:s]
    x_val = data[s:]
    y_val = label[s:]
    return x_train, y_train, x_val, y_val


if __name__ == '__main__':

    _x, _y = reader2.get_file(data_dir)
    X_train, y_train, x_val, y_val = segmentation(_x, _y, seg_ratio)

    image_batch, label_batch = reader2.get_batch(X_train, y_train, 224, 224, BATCH, 1024)
    x_val_image_batch, y_val_label_batch = reader2.get_batch(x_val, y_val, 224, 224, BATCH, 1024)

    x_imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    tf.summary.image('input', x_imgs, max_outputs=3)

    y_imgs = tf.placeholder(tf.int32, [None, num_classes])
    keep_prob = tf.placeholder(tf.float32)

    vgg = model.vgg16(x_imgs, keep_prob,num_classes)
    problabel = vgg.probs

    with tf.name_scope('loss'):
        trian_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=problabel, labels=y_imgs))
        tf.summary.scalar('loss', trian_loss)

    with tf.name_scope('valid_loss'):
        valid_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=problabel, labels=y_imgs))
        tf.summary.scalar('valid_loss', valid_loss)

    global_step = tf.Variable(0)

    learning_rate = tf.train.exponential_decay(Learning_Rate, global_step, decay_steps=Sample_Size / BATCH, decay_rate=0.96,
                                               staircase=True)
    optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(trian_loss, global_step)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_imgs, 1), tf.argmax(problabel, 1))
        trian_acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', trian_acc)

    with tf.name_scope('valid_accuracy'):
        Val_acc = tf.equal(tf.argmax(y_imgs, 1), tf.argmax(problabel, 1))
        validAcc = tf.reduce_mean(tf.cast(Val_acc, tf.float32))
        tf.summary.scalar('valid_accuracy', validAcc)

    sess = tf.Session()

    #  merged
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(logs_train_dir + '/train', sess.graph)
    val_writer = tf.summary.FileWriter(logs_train_dir + '/val')

    sess.run(tf.global_variables_initializer())
    vgg.load_weights('../vgg16_weights.npz', sess)
    saver = vgg.saver()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    import time

    start_time = time.time()

    for step in np.arange(MAX_STEP):

        image, label = sess.run([image_batch, label_batch])
        val_image, val_label = sess.run([x_val_image_batch, y_val_label_batch])

        labels = reader2.onehot(label)

        summary, _, trianacc, trainloss = sess.run([merged, optimizer, trian_acc, trian_loss],
                                                   feed_dict={x_imgs: image, y_imgs: labels, keep_prob: 0.7})

        # rs = sess.run(merged)
        train_writer.add_summary(summary, step)

        val_labels = reader2.onehot(val_label)
        val_summary, Val_loss_record, Valaccc = sess.run([merged, valid_loss, validAcc],
                                                      feed_dict={x_imgs: val_image, y_imgs: val_labels, keep_prob: 1})
        val_writer.add_summary(val_summary, step)


        if step % 10 == 0:
            print("now the Train_loss is %f " % trainloss)
            print("now the Val_loss_record is %f " % Val_loss_record)
            end_time = time.time()
            print('time: ', (end_time - start_time))
            start_time = end_time

            print('------epoch：%s -------TrainAcc：%.5f --------TestAcc：%.5f' % (step, trianacc, Valaccc))\

        if step < 500 and step > 0:
            if step % 10000 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(logs_train_dir,  "model.ckpt")
                saver.save(sess, checkpoint_path, global_step=step)
        elif step >= 500 and step < 2000:
            if step % 100 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(logs_train_dir,  "model.ckpt")
                saver.save(sess, checkpoint_path, global_step=step)
        elif step >= 2000 and step < 15000:
            if step % 100 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(logs_train_dir,  "model.ckpt")
                saver.save(sess, checkpoint_path, global_step=step)
        elif step >= 15000:
            if step % 500000 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(logs_train_dir,  "model.ckpt")
                saver.save(sess, checkpoint_path, global_step=step)
    print("Optimization Finished!")
