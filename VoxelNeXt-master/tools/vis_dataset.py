# import sys
# sys.path.append(r"/media/gpu/sde/tianxl/VoxelNeXt-master/tools")
from pcdet.datasets.my.hj_dataset_devkit import Frame, CoordinateSystem, Scene, ObstacleCategory, SupportedDataset, load_dataset
import mayavi.mlab as mlab
from visual_utils import visualize_utils as V
OPEN3D_FLAG = False
from pathlib import Path
import copy
import pickle
import os

import numpy as np
from skimage import io
from pcdet.datasets.processor.point_feature_encoder import PointFeatureEncoder
from pcdet.datasets.processor.data_processor import DataProcessor
from pcdet.datasets.processor import point_feature_encoder
from pcdet.datasets.processor import data_processor

import argparse
from pcdet.config import cfg, cfg_from_yaml_file

from pcdet.models import build_network, load_data_to_gpu


dataset_path = '/media/gpu/sde/tianxl/VoxelNeXt-master/data/my/training'
dataset = load_dataset(SupportedDataset.hongjing, dataset_path)
scene = dataset.scenes[0]
# frame = scene.frames[0]
frames = scene.frames
infos = []
coor = CoordinateSystem.world
lidar_sensor = 'true_value'

for idx, frame in enumerate(frames):
    
    points = frame.get_lidar_cloud(lidar_sensor, coor)[:,0:4] # nuscenes是5维
    input_dict = {
            'points': points,
            'frame_id': idx,
    }
    # 上面是有问题的，需要对原始点云进行处理，直接可视化不对
    labels = frame.get_obstacles(CoordinateSystem.ego)
    if labels is None:
        continue
    lidar_sensor = 'true_value'
    info = {}
    sequence_idx = 'data_0010'
    frame_idx = frame.get_timestamp(lidar_sensor)
    pc_info = {'num_features': 4, 'lidar_idx': frame_idx, 'sequence_idx': sequence_idx, 'lidar_sensor':lidar_sensor}
    info['point_cloud'] = pc_info 
    lidar_file = Path(dataset_path) / ('%s' % sequence_idx) /'lidars' / ('%s' % lidar_sensor) /('%s.bin' % frame_idx)
    assert lidar_file.exists()
    
    annotations = {}
    annotations['name'] = np.array([obj.category.value for obj in labels])
    annotations['center'] = np.array([[obj.box.center.x, obj.box.center.y, obj.box.center.z] for obj in labels])
    annotations['size'] = np.array([[obj.box.size.x, obj.box.size.y, obj.box.size.z] for obj in labels])
    annotations['rotation_y'] = np.array([obj.box.rotation.z for obj in labels])

    gt_boxes_lidar = np.concatenate([annotations['center'], annotations['size'], -(np.pi / 2 + annotations['rotation_y'][..., np.newaxis])], axis=1)
    annotations['gt_boxes_lidar'] = gt_boxes_lidar
    annotations['num_points_in_gt'] = np.array([[obj.num_lidar_pts] for obj in labels])
    info['annos'] = annotations
        
    V.draw_scenes(
        points=input_dict['points'][:, 1:], gt_boxes=info['annos']['gt_boxes_lidar']
    )

    if not OPEN3D_FLAG:
        mlab.show(stop=True)

