import xml.etree.ElementTree as ET
import pickle
import os
from os import getcwd
import numpy as np
from PIL import Image
import cv2

import imgaug as ia
from imgaug import augmenters as iaa

ia.seed(1)


def read_xml_annotation(root, image_id):
    in_file = open(os.path.join(root, image_id))
    tree = ET.parse(in_file)
    root = tree.getroot()

    for object in root.findall('object'):  # 找到root节点下的所有country节点
        bndbox = object.find('bndbox')  # 子节点下节点rank的值

        xmin = int(bndbox.find('xmin').text)
        xmax = int(bndbox.find('xmax').text)
        ymin = int(bndbox.find('ymin').text)
        ymax = int(bndbox.find('ymax').text)
        print(xmin,ymin,xmax,ymax)

    bndbox = root.find('object').find('bndbox')

    xmin = int(bndbox.find('xmin').text)
    xmax = int(bndbox.find('xmax').text)
    ymin = int(bndbox.find('ymin').text)
    ymax = int(bndbox.find('ymax').text)

    return (xmin, ymin, xmax, ymax)


def change_xml_annotation(root, image_id, new_target):
    new_xmin = new_target[0]
    new_ymin = new_target[1]
    new_xmax = new_target[2]
    new_ymax = new_target[3]

    in_file = open(os.path.join(root, str(image_id) + '.xml'))  # 这里root分别由两个意思
    tree = ET.parse(in_file)
    xmlroot = tree.getroot()
    object = xmlroot.find('object')
    bndbox = object.find('bndbox')
    xmin = bndbox.find('xmin')
    xmin.text = str(new_xmin)
    ymin = bndbox.find('ymin')
    ymin.text = str(new_ymin)
    xmax = bndbox.find('xmax')
    xmax.text = str(new_xmax)
    ymax = bndbox.find('ymax')
    ymax.text = str(new_ymax)
    tree.write(os.path.join(root, str(image_id) + "_aug" + '.xml'))


if __name__ == "__main__":
    # cmd = os.getcwd()
    cmd = "/Users/wangbenkang/Desktop/500/Example"
    image_id = "000010"
    img = Image.open(os.path.join(cmd, str(image_id) + '.jpg'))
    img = np.array(img)

    bndbox = read_xml_annotation(cmd, str(image_id) + '.xml')

    bbs = ia.BoundingBoxesOnImage([
        ia.BoundingBox(x1=bndbox[0], y1=bndbox[1], x2=bndbox[2], y2=bndbox[3]) #改这里！
    ], shape=img.shape)
    seq = iaa.Sequential([
        iaa.Flipud(0.5),  # vertically flip 20% of all images
        iaa.Multiply((1.2, 1.5)),  # change brightness, doesn't affect BBs
        iaa.Affine(
            translate_px={"x": 10, "y": 10},
            scale=(0.8, 0.95),
            rotate=(-10, 10)
        )  # translate by 40/60px on x/y axis, and scale to 50-70%, affects BBs
    ])
    seq_det = seq.to_deterministic()  # 保持坐标和图像同步改变，而不是随机
    image_aug = seq_det.augment_images([img])[0]
    bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]

    before = bbs.bounding_boxes[0]
    after = bbs_aug.bounding_boxes[0]
    print("BB : (%.4f, %.4f, %.4f, %.4f) -> (%.4f, %.4f, %.4f, %.4f)" % (
        before.x1, before.y1, before.x2, before.y2,
        after.x1, after.y1, after.x2, after.y2)
          )

    image_before = bbs.draw_on_image(img, thickness=2)
    image_after = bbs_aug.draw_on_image(image_aug, thickness=2)
    Image.fromarray(image_before).save("before.jpg")
    Image.fromarray(image_after).save('after.jpg')

    new_bndbox = []
    new_bndbox.append(int(bbs_aug.bounding_boxes[0].x1))
    new_bndbox.append(int(bbs_aug.bounding_boxes[0].y1))
    new_bndbox.append(int(bbs_aug.bounding_boxes[0].x2))
    new_bndbox.append(int(bbs_aug.bounding_boxes[0].y2))

    # 修改xml tree 并保存
    change_xml_annotation(cmd, image_id, new_bndbox)