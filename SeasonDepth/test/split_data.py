import os
from os import path

def scaner_file(url):
    file = os.listdir(url)
    with open('test_files.txt', 'a') as file_object:
        for f in file:
            real_url = path.join(url, f)
            if path.isfile(real_url):  # 如果是文件，则保存到txt中
                if real_url[-4:] == '.jpg':
                    # print(real_url)
                    file_object.write(real_url + '\n')
            elif path.isdir(real_url):  # 如果是目录，则递归调用自定义函数 scaner_file (url)进行多次
                scaner_file(real_url)
            else:
                print("其他情况")
                pass

scaner_file(".")  # 从当前文件所在路径开始

'''
# line=".\slice4\env00\c0\images\img_01972_c0_1290444812906489us.jpg".split("\\")
line="./images/env02/c1/img_00838_c1_1284563319091927us.jpg".split("/")
print(line)
if line[4] == 'images':  # train_set
    name = line[5].split(".")
    velo_filename = os.path.join(
        line[0],line[1],line[2],line[3],'depth_map',name[0]+".png")
elif line[1] == 'images':  # val_set
    name = line[4].split(".")
    velo_filename = os.path.join(
        line[0],"depth",line[2],line[3],name[0]+".png")
print(velo_filename)
'''