# coding:utf-8
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.001)
# KTF.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from data import *
import glob
from SegNet import CreateSegNet
import argparse
import numpy as np
import pickle


def get_RGB_img(img):
    img_rgb = np.zeros((224, 224, 3))
    img_rgb[:, :, 0] = img
    img_rgb[:, :, 1] = img
    img_rgb[:, :, 2] = img
    img = array_to_img(img_rgb)
    return img


def save_img(imgs, img_names, size=(224, 224)):
    print("array to image")
    for i in range(imgs.shape[0]):
        img = imgs[i]
        img = get_RGB_img(img)
        img = img.resize(size)
        img.save("data/640_480/test/result/%s" % (img_names[i]))


def f(x):
    if x[0] > x[1]:
        return 0
    else:
        return 255


def getSegNet():
    # command line argments
    parser = argparse.ArgumentParser(description="SegNet LIP dataset")
    parser.add_argument("--train_list",
                        default="../LIP/TrainVal_images/train_id.txt",
                        help="train list path")
    parser.add_argument("--trainimg_dir",
                        default="../LIP/TrainVal_images/TrainVal_images/train_images/",
                        help="train image dir path")
    parser.add_argument("--trainmsk_dir",
                        default="../LIP/TrainVal_parsing_annotations/TrainVal_parsing_annotations/train_segmentations/",
                        help="train mask dir path")
    parser.add_argument("--val_list",
                        default="../LIP/TrainVal_images/val_id.txt",
                        help="val list path")
    parser.add_argument("--valimg_dir",
                        default="../LIP/TrainVal_images/TrainVal_images/val_images/",
                        help="val image dir path")
    parser.add_argument("--valmsk_dir",
                        default="../LIP/TrainVal_parsing_annotations/TrainVal_parsing_annotations/val_segmentations/",
                        help="val mask dir path")
    parser.add_argument("--batch_size",
                        default=10,
                        type=int,
                        help="batch size")
    parser.add_argument("--n_epochs",
                        default=10,
                        type=int,
                        help="number of epoch")
    parser.add_argument("--epoch_steps",
                        default=100,
                        type=int,
                        help="number of epoch step")
    parser.add_argument("--val_steps",
                        default=10,
                        type=int,
                        help="number of valdation step")
    parser.add_argument("--n_labels",
                        default=2,
                        type=int,
                        help="Number of label")
    parser.add_argument("--input_shape",
                        default=(224, 224, 3),
                        help="Input images shape")
    parser.add_argument("--kernel",
                        default=3,
                        type=int,
                        help="Kernel size")
    parser.add_argument("--pool_size",
                        default=(2, 2),
                        help="pooling and unpooling size")
    parser.add_argument("--output_mode",
                        default="softmax",
                        type=str,
                        help="output activation")
    parser.add_argument("--loss",
                        default="categorical_crossentropy",
                        type=str,
                        help="loss function")
    parser.add_argument("--optimizer",
                        default="adadelta",
                        type=str,
                        help="oprimizer")
    args = parser.parse_args()
    segnet = CreateSegNet(args.input_shape, args.n_labels, args.kernel, args.pool_size, args.output_mode)
    return segnet


import cv2


def load_test_data(path="data/640_480/test/image"):
    imgs = glob.glob(path + "/*." + "jpg")
    imgs.extend(glob.glob(path + "/*." + "png"))
    print(len(imgs))
    imgdatas = np.ndarray((len(imgs), 224, 224, 3), dtype=np.uint8)
    for i in range(len(imgs)):
        imgname = imgs[i]
        midname = imgname[imgname.rindex("\\") + 1:]
        # img = load_img(path + "\\" + midname)
        img = cv2.imread(path + "\\" + midname)
        img = cv2.resize(img, (224, 224))
        img = img_to_array(img)
        imgdatas[i] = img
    imgdatas = imgdatas.astype('float32')
    imgdatas /= 255.0

    def getSimpleName(name):
        name = name.replace("\\", "/")
        if name.index("/") != -1:
            name = name[name.rindex("/") + 1:]
        return name

    imgs = list(map(getSimpleName, imgs))
    return imgdatas, imgs


def gen_result_imgs():
    np_file_name = 'imgs_mask_test.npy'
    if os.path.exists(np_file_name):
        print(np_file_name + " exists")
        imgs_mask_test = np.load(np_file_name)[0]
        img_names = np.load(np_file_name)[1]
        save_img(imgs_mask_test, img_names, (640, 480))
        print("saved!")
        exit(0)

    predict_file = "predict_file.npy"
    if os.path.exists(predict_file):
        imgs_mask_test = np.load(predict_file)
    else:
        test_data, img_names = load_test_data()
        segnet = getSegNet()
        segnet.load_weights("model/weights.17-0.9973.hdf5")
        imgs_mask_test = segnet.predict(test_data)
        # np.save(predict_file, [imgs_mask_test, img_names])

    # 记录shape
    shape = imgs_mask_test.shape
    # flat
    imgs_mask_test = imgs_mask_test.reshape((-1, 2))
    # 还原RGB值
    imgs_mask_test = np.asarray(list(map(f, imgs_mask_test)), dtype="int32")
    # reshape成为图片
    imgs_mask_test = imgs_mask_test.reshape((shape[0], 224, 224))
    # 保存为npy文件
    # np.save(np_file_name, [imgs_mask_test, img_names])
    # 保存图片
    save_img(imgs_mask_test, img_names, (640, 480))
    # over！
    print("saved!")


from PIL import Image
def calcIoU(img1,img2,true_label=(255,255,255)):
    img1=img1.convert("RGB")
    img2=img2.convert("RGB")
    # 二值化预处理
    for i in range(img1.size[0]):
        for j in range(img1.size[1]):
            pixel=img2.getpixel((i,j))
            if pixel[0]+pixel[1]+pixel[2]>=128+128+128:
                img2.putpixel((i,j),(255,255,255))
            else:
                img2.putpixel((i,j),(0,0,0))
            pixel=img1.getpixel((i,j))
            if pixel[0]+pixel[1]+pixel[2]>=128+128+128:
                img1.putpixel((i,j),(255,255,255))
            else:
                img1.putpixel((i,j),(0,0,0))
    cross_count=0
    union_count=0
    # print(img2.getpixel((0,0)))
    for i in range(img1.size[0]):
        for j in range(img1.size[1]):
            if img1.getpixel((i,j))==true_label and img2.getpixel((i,j))==true_label:
                cross_count+=1
                union_count+=1
            elif img1.getpixel((i,j))==true_label or img2.getpixel((i,j))==true_label:
                union_count+=1
    return float(cross_count)/union_count


def gen_IoUs_by_epochs():
    test_data, img_names = load_test_data()
    weights_files=glob.glob("model/weights*.hdf5")
    all_weights_ious={}
    # 遍历每个weight文件
    for weights_file in weights_files:
        ious={}
        weights_file=weights_file.replace("\\","/")
        segnet = getSegNet()
        segnet.load_weights(weights_file)
        imgs_mask_test = segnet.predict(test_data)
        # 记录shape
        shape = imgs_mask_test.shape
        # flat
        imgs_mask_test = imgs_mask_test.reshape((-1, 2))
        # 还原RGB值
        imgs_mask_test = np.asarray(list(map(f, imgs_mask_test)), dtype="int32")
        # reshape成为图片
        imgs_mask_test = imgs_mask_test.reshape((shape[0], 224, 224))
        # 遍历每张图片
        for i in range(imgs_mask_test.shape[0]):
            img = imgs_mask_test[i]
            predicted_img=get_RGB_img(img)
            label_img=Image.open("data/640_480/test/label/"+img_names[i].replace("jpg","png"))
            label_img=label_img.resize((224,224))
            iou=calcIoU(predicted_img,label_img)
            print(weights_file+":"+img_names[i]+":"+str(iou))
            ious[img_names[i]]=iou
        all_weights_ious[weights_file]=ious
        print(weights_file+" done")
    pickle.dump(all_weights_ious,open("all_weights_ious.pickle","wb"))


def calcIoUsOfEpochs():
    all_weights_ious=pickle.load(open("all_weights_ious.pickle","rb"))
    def getAvg(iou_dict):
        avg=0
        for key in iou_dict.keys():
            avg+=iou_dict[key]
        avg/=len(iou_dict.keys())
        return avg
    for key in all_weights_ious.keys():
        all_weights_ious[key]=getAvg(all_weights_ious[key])
#     print(all_weights_ious)
    return all_weights_ious


def main():
    gen_IoUs_by_epochs()
    print(calcIoUsOfEpochs())
    # gen_result_imgs()
    pass


main()
