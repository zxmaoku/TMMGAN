
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import os

path = 'D:\\DMT\\new repainting\\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData'
file_list = os.listdir(path)
file_list = file_list[361:363]
print(file_list)
for file_item in file_list:
    sub_path = os.path.join(path,file_item)
    nii_list = os.listdir(sub_path)
    i = 78
    for nii_item in nii_list:
        if 't1.nii' in nii_item:
            print(nii_item)
            nii_path = os.path.join(sub_path,nii_item)
            img = nib.load(nii_path)
            # print(img.shape)
            # print(img)# shape(240, 240, 155)
            # print(img.header['db_name'])
            width, height, queue = img.dataobj.shape  # 由文件本身维度确定，可能是3维，也可能是4维
            # print("width",width)  # 240
            # print("height",height) # 240
            # print("queue",queue)   # 155
            # nib.viewers.OrthoSlicer3D(img.dataobj).show()
            choice=[78 ]
            num = 1


            img_arr = img.dataobj[:, :, i]

                # plt.subplot(5, 4, num)
            plt.tight_layout()
            plt.imshow(img_arr,)
            plt.axis('off')
            plt.savefig("D:\DMT\\new repainting\BraTS2020_TrainingData\\t1\\"+nii_item[:-4]+'_'+str(i)+'_.png')


            # plt.show()
        elif 't2.nii' in nii_item:
            nii_path = os.path.join(sub_path,nii_item)
            img = nib.load(nii_path)
            width, height, queue = img.dataobj.shape  # 由文件本身维度确定，可能是3维，也可能是4维
            choice=[78]
            num = 1

            plt.tight_layout()
            img_arr = img.dataobj[:, :, i]
                # plt.subplot(5, 4, num)
            plt.imshow(img_arr, )
            plt.axis('off')
            plt.savefig("D:\DMT\\new repainting\BraTS2020_TrainingData\\t2\\"+nii_item[:-4]+'_'+str(i)+'_.png')

        elif 'flair.nii' in nii_item:
            nii_path = os.path.join(sub_path,nii_item)
            img = nib.load(nii_path)
            width, height, queue = img.dataobj.shape  # 由文件本身维度确定，可能是3维，也可能是4维
            choice=[78]
            num = 1
            plt.tight_layout()
            img_arr = img.dataobj[:, :, i]
            plt.imshow(img_arr, )
            plt.axis('off')
            plt.savefig("D:\DMT\\new repainting\BraTS2020_TrainingData\\flair\\"+nii_item[:-4]+'_'+str(i)+'_.png')


            # plt.show()


