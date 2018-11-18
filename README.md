# DeepLearing-Classification

## 前期准备

### VGG 预训练模型下载

对于复用的VGG模型（在imagenet进行训练的模型），首先第一步是要获得相应的权重文件

VGG权重文件：https://www.cs.toronto.edu/~frossard/vgg16/vgg16_weights.npz

[VGG权重文件](https://www.cs.toronto.edu/~frossard/vgg16/vgg16_weights.npz)

### 数据集准备

将训练集按照类别，分别放入相应的文件夹中。

```
trainset-|------class1
         |------class2
         |------class3
```

影像增强（可选）

若训练集数据量不足需要进行数据增强时。

1. 安装Augmentor

   ```
   pip install Augmentor
   ```

1. 运行 DATAenforce.py 

   根据需要选择文件夹路径，旋转缩放参数，以及增强后影像的数量。程序运行完成后会在选择的文件夹路径下生成一个output 文件夹。

## 模型训练

训练数据集及VGG pre-trained 模型准备完成后。即可进行模型的训练。

1. ### create_and_read_TFRecord2 设置

   根据训练集的类别所对应的标签对类别进行设置

   ```python
   # set class lable
   if letter == 'Combined type damage':
       labels = np.append(labels, n_img * [0])
   elif letter == 'Flexural type damage':
       labels = np.append(labels, n_img * [1])
   elif letter == 'No damage':
       labels = np.append(labels, n_img * [3])
   elif letter == 'Shear type damage':
       labels = np.append(labels, n_img * [2])
   ```

2. ### VGG_Trian 设置

   #### 训练参数设置

   ```python
   # Dir-path
   logs_train_dir = "F://Githubtest//"  #Tensorboard event文件 存储路径
   data_dir = "E:/PycharmProjects/PHI Challenge 2018/Task8/Edataset/" # 训练集文件存储路径
   model_save_dir = "F://Githubtest//" #  模型文件 存储路径
   
   # Model-parameters
   num_classes = 4 # 类别数
   seg_ratio = 0.8 # 训练集验证集分割比例
   opt = 'sgd'  #['sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam'] # option 方法
   
   # Hyper-parameters
   MAX_STEP = 400000 # 最大迭代数
   BATCH = 32 # batch size
   Learning_Rate = 0.01 # 学习率设置
   Sample_Size = 5000 # 样本数量 主要用在学习率衰减
   keep_prob_val = 1 # dropout 概率
   ```

   #### 模型存储参数修改

   根据实际训练结果选择合适的存储迭代间隔

   ```python
   if step % 10 == 0:
       print("now the Train_loss is %f " % trainloss)
       print("now the Val_loss_record is %f " % Val_loss_record)
       end_time = time.time()
       print('time: ', (end_time - start_time))
       start_time = end_time
       print('------epoch：%s -------TrainAcc：%.5f --------TestAcc：%.5f' % (step, trianacc, Valaccc))
   
   if step < 500 and step > 0:
       if step % 10000 == 0 or (step + 1) == MAX_STEP:
           checkpoint_path = os.path.join(model_save_dir,  "model.ckpt")
           saver.save(sess, checkpoint_path, global_step=step)
   elif step >= 500 and step < 2000:
       if step % 100 == 0 or (step + 1) == MAX_STEP:
           checkpoint_path = os.path.join(model_save_dir,  "model.ckpt")
           saver.save(sess, checkpoint_path, global_step=step)
   elif step >= 2000 and step < 15000:
       if step % 100 == 0 or (step + 1) == MAX_STEP:
           checkpoint_path = os.path.join(model_save_dir,  "model.ckpt")
           saver.save(sess, checkpoint_path, global_step=step)
   elif step >= 15000:
       if step % 500000 == 0 or (step + 1) == MAX_STEP:
           checkpoint_path = os.path.join(model_save_dir,  "model.ckpt")
           saver.save(sess, checkpoint_path, global_step=step)
   ```

3. ## VGG16_model 设置

   该文件为VGG模型文件，主要用于fine-tuning操作。若不需要fine-tuning操作，只需将所有层中的trainable参数设置为False。在进行fine-tuning操作时，可自定义fine-tuning的层数，只需将相应层上的trainable参数设置为True即可。

## 数据预测

模型训练完成后使用VGG_Pred文件对影像进行预测

设置参数

```python
meta_filepath = 'F:\PHI\TASK7\\vgg1\logs\model.ckpt-2000.meta' #模型graph结构(.meta)文件路径
ckpt_filepath = 'F:\PHI\TASK7\\vgg1\logs\model.ckpt-2000' #模型参数（.ckpt ）文件路径

Predfile_dir = 'E:\\PycharmProjects\\PHI Challenge 2018\\Task7\\data\\test' #预分类文件夹路径
pred_resultfile_savepath = "" # 预测结果保存路径
pred_resultfile_name = "./VGG-6000.txt"#预测结果文件名
class_number = 4 # 模型分类个数
```

预测结果

预测结果文件为txt文件，其格式如下。

| image name | lable |
| :--------: | :---: |
|   1.jpg    |   1   |
|   2.jpg    |   3   |
|   3.jpg    |   0   |
|    ...     |  ...  |
|  100.jpg   |   2   |
