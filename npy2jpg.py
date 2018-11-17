import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import os

dir="./data"
dest_dir="./data\\"

Pixel_level_save_dir = "/Users/wangbenkang/Downloads/PHI/Task1/dataset/Pixel_level/"
Object_level_save_dir = "/Users/wangbenkang/Downloads/PHI/Task1/dataset/Object_level/"
Structural_level_save_dir = "/Users/wangbenkang/Downloads/PHI/Task1/dataset/Structural_level/"

Undamaged_state_save_dir = "/Users/wangbenkang/Downloads/PHI/Task2/dataset/Undamaged_state/"
Damaged_state_save_dir = "/Users/wangbenkang/Downloads/PHI/Task2/dataset/Damaged_state/"


No_spalling_dir = "E://PycharmProjects/PHI Challenge 2018/Task3/dataset/No_spalling/"
Spalling_dir = "E://PycharmProjects/PHI Challenge 2018/Task3/dataset/Spalling/"

Steel_dir = "E://PycharmProjects/PHI Challenge 2018/Task4/dataset/Steel/"
Others_dir = "E://PycharmProjects/PHI Challenge 2018/Task4/dataset/Others/"

No_collapse = "E://PycharmProjects/PHI Challenge 2018/Task5/dataset/No_collapse/" # 0
Partial_collapse = "E://PycharmProjects/PHI Challenge 2018/Task5/dataset/Partial_collapse/" # 1
Global_collapse = "E://PycharmProjects/PHI Challenge 2018/Task5/dataset/Global_collapse/" # 2

Beam = "E://PycharmProjects/PHI Challenge 2018/Task6/dataset/Beam/" # 0
Column = "E://PycharmProjects/PHI Challenge 2018/Task6/dataset/Column/" # 1
Wall = "E://PycharmProjects/PHI Challenge 2018/Task6/dataset/Wall/" # 2
Else = "E://PycharmProjects/PHI Challenge 2018/Task6/dataset/Else/" # 3

No_damage = "E:/PycharmProjects/PHI Challenge 2018/Task7/dataset/No damage/"   # 0
Minor_damage = "E:/PycharmProjects/PHI Challenge 2018/Task7/dataset/Minor damage/"# 1
Moderate_damage = "E:/PycharmProjects/PHI Challenge 2018/Task7/dataset/Moderate damage/"# 2
Heavy_damage = "E:/PycharmProjects/PHI Challenge 2018/Task7/dataset/Heavy damage/"# 3

#trian data
def trainnpy2jpg(dir,dest_dir):
    if os.path.exists(dir)==False:
        os.makedirs(dir)
    if os.path.exists(dest_dir)==False:
        os.makedirs(dest_dir)
    file=dir+'/X_train.npy'
    con_arr=np.load(file)
    count=0
    labelfile = dir+'/y_train.npy'
    Y_arr = np.load(labelfile)

    # AA = Y_arr[0]
    # BB = Y_arr[1]

    for con in con_arr:
        img = transforms.ToPILImage()(con)
        #img.show()

        lable = Y_arr[count]
        if lable == 0:
            img.save(No_damage + str(count) + ".jpg")
        elif lable == 1:
            img.save(Minor_damage + str(count) + ".jpg")
        elif lable == 2:
            img.save(Moderate_damage + str(count) + ".jpg")
        elif lable == 3:
            img.save(Heavy_damage + str(count) + ".jpg")

        count = count+1





#test data
def testnpy2jpg(dir,dest_dir):
    if os.path.exists(dir)==False:
        os.makedirs(dir)
    # file=dir+'\\X_test.npy'
    file = dir + '\\X_test.npy'
    con_arr=np.load(file)
    count=0
    for con in con_arr:

        # arr=con[0]
        # label=con[1]
        #
        # print(np.argmax(label))
        # arr=arr*255
        # #arr=np.transpose(arr,(2,1,0))
        # arr=np.reshape(con,(3,224,224))
        # r=Image.fromarray(arr[0]).convert("L")
        # g=Image.fromarray(arr[1]).convert("L")
        # b=Image.fromarray(arr[2]).convert("L")
        #
        # img=Image.merge("RGB",(r,g,b))
        #
        # label_index=np.argmax(label)
        # img.save(dest_dir+str(label_index)+"_"+str(count)+".jpg")
        # count=count+1

        img = transforms.ToPILImage()(con)
        #img.show()
        save_dir = dest_dir + "test\\"
        #1 lable = Y_arr[count]
        img.save(save_dir + str(count) + ".jpg")
# task1 影像分类
        # if lable == 0:
        #     img.save(Pixel_level_save_dir + str(count) + ".jpg")
        # elif lable == 1:
        #     img.save(Object_level_save_dir + str(count) + ".jpg")
        # elif lable == 2:
        #     img.save(Structural_level_save_dir + str(count) + ".jpg")

# task2 影像分类
#         if lable == 0:
#             img.save(No_spalling_dir + str(count) + ".jpg")
#         elif lable == 1:
#             img.save(Spalling_dir + str(count) + ".jpg")

        count = count+1

if __name__=="__main__":
   # trainnpy2jpg(dir,dest_dir)
    testnpy2jpg(dir,dest_dir)