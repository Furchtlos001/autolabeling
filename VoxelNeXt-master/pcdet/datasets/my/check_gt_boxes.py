# import pickle

# with open("/media/gpu/sde/tianxl/VoxelNeXt-master/data/my/my_infos_train.pkl", 'rb') as f:
#     infos = pickle.load(f)
# count = 0
# for k in range(len(infos)):
#     info = infos[k]
#     annos = info['annos']
#     gt_boxes = annos['gt_boxes_lidar']
#     num_obj = gt_boxes.shape[0]
    
#     for i in range(num_obj):
#         if gt_boxes[i, 0] == None:
#             count += 1
#             print('累计 %d 个gt_boxes类型为None' % (count))
    
#     print('sequence_idx: %s 没有问题' % (info['point_cloud']['sequence_idx']))
    
import pickle
import os
import numpy as np
from pathlib import Path

with open("/media/gpu/sde/tianxl/VoxelNeXt-master/data/my/my_infos_train.pkl", 'rb') as f:
    infos = pickle.load(f)
count = 0
with open('output1.txt', 'w') as f:
    for k in range(len(infos)):
        info = infos[k]
        annos = info['annos']
        gt_boxes = annos['gt_boxes_lidar']
        num_obj = gt_boxes.shape[0]
        
        # for i in range(num_obj):
        #     if gt_boxes[i, 0] == None:
        #         count += 1
        #         print(gt_boxes[i, :])
        #         print('累计 %d 个gt_boxes类型为None，定位在第 %d 个infos中的第 %d 个num_obj' % (count, k, i))
        #     else:
        #         print('sequence_idx: %s 没有问题，第 %d 个infos中的第 %d 个num_obj没有问题' % (info['point_cloud']['sequence_idx'], k, i))
        
        for i in range(num_obj):
            if gt_boxes[i, 0] == None:
                count += 1
                # print(gt_boxes[i, :])
                output1_text = '累计 %d 个gt_boxes类型为None，定位在第 %d 个infos中的第 %d 个num_obj' % (count, k, i)
            else:
                output1_text = 'sequence_idx: %s 没有问题，第 %d 个infos中的第 %d 个num_obj没有问题' % (info['point_cloud']['sequence_idx'], k, i)

            # 将输出文本写入文件
            f.write(output1_text + '\n')

# print("输出已保存到output.txt文件")

def get_lidar(sample_idx,frame_idx, lidar_sensor):
    lidar_file = os.path.join("/media/gpu/sde/tianxl/VoxelNeXt-master/data/my/training", ('%s' % sample_idx), 'lidars', ('%s' % lidar_sensor), ('%s.bin' % frame_idx))
    # lidar_file = "/media/gpu/sde/tianxl/VoxelNeXt-master/data/my/training" / ('%s' % sample_idx) /'lidars' / ('%s' % lidar_sensor) /('%s.bin' % frame_idx)
    print(lidar_file)
    lidar_file = Path(lidar_file)
    assert lidar_file.exists()
    #读取该 bin文件类型，并将点云数据以 numpy的格式输出！！！
    #并且将数据 转换成 每行5个数据，刚好是一个点云数据的四个参数：X,Y,Z,R(强度或反射值），时间戳
    return np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 5)

info_125 = infos[125]
sample_idx = info_125['point_cloud']['sequence_idx']
frame_idx = info_125['point_cloud']['lidar_idx']
lidar_sensor = info_125['point_cloud']['lidar_sensor']
print(sample_idx, frame_idx, lidar_sensor)
points = get_lidar(sample_idx, frame_idx, lidar_sensor)
print(points[0, 0:3])
annos = info_125['annos']
names = annos['name']
print(names)
gt_boxes = annos['gt_boxes_lidar']
print(gt_boxes)

# info_533= infos[533]
# sample_idx = info_533['point_cloud']['sequence_idx']
# frame_idx = info_533['point_cloud']['lidar_idx']
# lidar_sensor = info_533['point_cloud']['lidar_sensor']
# print(sample_idx, frame_idx, lidar_sensor)
# points = get_lidar(sample_idx, frame_idx, lidar_sensor)
# print(points[0, 0:3])
# annos = info_533['annos']
# names = annos['name']
# print(names)
# gt_boxes = annos['gt_boxes_lidar']
# print(gt_boxes)