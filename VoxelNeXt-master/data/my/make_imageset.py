import os


root = os.getcwd()
folder_path = root+ "/training"
save_path = root + "/ImageSets/train.txt"
all_files = os.listdir(folder_path)# ['2023_04_15_11_34_18_highway', 'hongjing_fastload.cache', '2023_04_15_12_10_20_city']
all_files = sorted(all_files)

for idx,name in enumerate(all_files):
    if '.cache' in name:# 如果包含 ".cache"，在列表中移除该文件
        all_files.pop(idx)
with open(save_path, "w") as fw:# 使用换行符将 all_files 列表中的所有文件名连接起来
    fw.write('\n'.join(all_files))
    
val_path = root+ "/testing"
save_path_val = root + "/ImageSets/val.txt"
val_files = os.listdir(val_path)
val_files = sorted(val_files)

for idx,name in enumerate(val_files):
    if '.cache' in name:
        val_files.pop(idx)
with open(save_path_val, "w") as fw:
    fw.write('\n'.join(val_files))
