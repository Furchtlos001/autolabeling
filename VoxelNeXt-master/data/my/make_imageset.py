import os


root = os.getcwd()
folder_path = root+ "/training"
save_path = root + "/ImageSets/train.txt"
all_files = os.listdir(folder_path)
all_files = sorted(all_files)

for idx,name in enumerate(all_files):
    if '.cache' in name:
        all_files.pop(idx)
with open(save_path, "w") as fw:
    fw.write('\n'.join(all_files))
