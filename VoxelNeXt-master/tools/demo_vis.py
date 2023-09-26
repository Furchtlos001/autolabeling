import argparse
import glob
from pathlib import Path
import os
from pcdet.datasets.my.hj_dataset_devkit import Frame, Scene, ObstacleCategory

try:
    import open3d
    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V
    OPEN3D_FLAG = False

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

from pcdet.datasets.my.hj_dataset_devkit import SupportedDataset, load_dataset, CoordinateSystem
from scipy.spatial.transform import Rotation as sci_R

# import mayavi.mlab as mlab

# 颜色类
class Colors:
    Red = (0.9, 0.3, 0.24)
    Orange = (0.9, 0.49, 0.13)
    Yellow = (0.95, 0.8, 0.25)
    Green = (0.32, 0.8, 0.5)
    Cyan = (0.3, 0.82, 0.88)
    Blue = (0.3, 0.63, 0.9)
    Purple = (0.65, 0.41, 0.75)
    White = (1., 1., 1.)
    Black = (0., 0., 0.)
    Dark = (0.25, 0.25, 0.25)

    @staticmethod
    def random(seed=None):
        import random
        random.seed(seed)
        return (random.random(), random.random(), random.random())

def euler_to_rotation(rot_xyz, degree=False):
    """ convert rotation in Euler angles to scipy's rotation
    :param rot_xyz: rotation in Euler angles, [r_x, r_y, r_z]
    :param degree: angle unit is rad or degree, default is False, use rad
    :return: rotation class in "scipy.spatial.transform.Rotation" type
    """
    return sci_R.from_euler('xyz', rot_xyz, degrees=degree)

# 画框
def show_box(figure, center, size, rotation, rgb=Colors.Red, scale=1.):
    
    assert len(center) == len(size) == len(rotation) == 3
    box_pos, box_size, box_rot = np.array(center), np.array(size), np.array(rotation)
    pts_prefix = np.array([[-1, -1, -1], [-1, 1, -1], [1, 1, -1], [1, -1, -1],
                               [-1, -1, 1], [-1, 1, 1], [1, 1, 1], [1, -1, 1]]) / 2
    box_pts = euler_to_rotation(box_rot).apply(pts_prefix * box_size) + box_pos
    # plot bottom and upper frame, then 4 columns
    plot_order = [0, 1, 2, 3, 0, 4, 5, 6, 7, 4]
    mlab.plot3d(box_pts[:, 0][plot_order], box_pts[:, 1][plot_order], box_pts[:, 2][plot_order],
                    color=rgb, tube_radius=None, line_width=scale, figure=figure)
    for i in range(1, 4):
        plot_order = [i, i + 4]
        mlab.plot3d(box_pts[:, 0][plot_order], box_pts[:, 1][plot_order], box_pts[:, 2][plot_order],
                        color=rgb, tube_radius=None, line_width=scale, figure=figure)

# 画N*C numpy array点云
def show_np_cloud(figure, cloud: np.ndarray, rgb=Colors.White, scale=0.08, mode='point'):
    assert cloud.shape[1] > 2
    if len(rgb) == 3 and min(rgb) >= 0 and max(rgb) <= 1:
        mlab.points3d(cloud[:, 0], cloud[:, 1], cloud[:, 2], mode=mode, color=rgb, scale_factor=scale, figure=figure)
    else:
        mlab.points3d(cloud[:, 0], cloud[:, 1], cloud[:, 2], rgb, mode=mode, opacity=0.6,
                          colormap='jet', scale_factor=scale, scale_mode='none', figure=figure)

class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, frame=None, logger=None, ext='.bin'):
        """
        Args:
            root_path: 根目录
            dataset_cfg: 数据集配置
            class_names: 类别名称
            training: 训练模式
            logger: 日志
            ext: 扩展名
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.frame = frame
        self.ext = ext
        # 以上得到frame
        
        self.lidar_sensor = 'true_value'
        frame_idx = self.frame.get_timestamp(self.lidar_sensor)
        
        filename = os.listdir(root_path)
        filename = sorted(filename)
        for idx,name in enumerate(filename):
            if '.cache' in name:
                filename.pop(idx)
        # self.scene_name = filename[-2]
        self.scene_name = filename[0]
        print(self.scene_name)
        
        # /media/gpu/sde/tianxl/VoxelNeXt-master/data/my/training/2023_04_15_11_37_18_highway/lidar/true_value/*.bin
        # /media/gpu/sde/tianxl/VoxelNeXt-master/data/my/training/data_0010/lidars/true_value
        self.data_file = Path(root_path / self.scene_name / 'lidars' / self.lidar_sensor / ('%s.bin' % frame_idx)) # 指定某一帧 
        # self.data_file = Path(root_path / self.scene_name / 'lidars' / self.lidar_sensor)
        
        print(self.data_file)
        # data_file_list = glob.glob(str(self.data_file / f'*{self.ext}')) if self.data_file.is_dir() else [self.data_file]
        # data_file_list.sort()
        # self.sample_file_list = data_file_list
        
        # self.sample_file_list = []
        # dir_path = os.path.dirname(self.data_file)
        # file_name = os.path.splitext(os.path.basename(self.data_file))[0]
        # bin_file_path = os.path.join(dir_path, f"{file_name}.bin")
        # self.sample_file_list.append(bin_file_path)
        
        

    def __len__(self):
        return 1 # len(self.sample_file_list)

    def __getitem__(self, index):
        # if self.ext == '.bin':
            # points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 5)[:,0:4]
        # elif self.ext == '.npy':
        #     points = np.load(self.sample_file_list[index])
        # else:
        #     raise NotImplementedError
        
        # 需要自车坐标系，否则会超界
        coor = CoordinateSystem.ego
        # points = self.frame.get_lidar_cloud(self.lidar_sensor, coor)[:,0:4]
        points_frame = self.frame.get_lidar_cloud(self.lidar_sensor, coor)[:,0:4] # nuscenes是5维
        # labels = self.frame.get_obstacles(coor)
        
        # points = np.fromfile("../0ab9ec2730894df2b48df70d0d2e84a9_lidarseg.bin", dtype=np.float32).reshape(-1, 4)
        
        input_dict = {
            # 'points': points,
            'points': points_frame,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        
        # if labels is not None:
        #     annotations = {}
        #     annotations['name'] = np.array([obj.category.value for obj in labels])
        #     annotations['center'] = np.array([[obj.box.center.x, obj.box.center.y, obj.box.center.z] for obj in labels])
        #     annotations['size'] = np.array([[obj.box.size.x, obj.box.size.y, obj.box.size.z] for obj in labels])
        #     annotations['rotation_y'] = np.array([obj.box.rotation.z for obj in labels])

        #     gt_boxes_lidar = np.concatenate([annotations['center'], annotations['size'], -(np.pi / 2 + annotations['rotation_y'][..., np.newaxis])], axis=1)
        #     annotations['gt_boxes_lidar'] = gt_boxes_lidar
        #     annotations['num_points_in_gt'] = np.array([[obj.num_lidar_pts] for obj in labels])
        #     data_dict.update({'annos': annotations})
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser') 
    #  cfgs/nuscenes_models/cbgs_voxel0075_voxelnext.yaml
    parser.add_argument('--cfg_file', type=str, default='cfgs/my_models/voxelnext.yaml', #     
                        help='specify the config for demo') 
    parser.add_argument('--data_path', type=str, default='/media/gpu/sde/tianxl/gitlab/github/autolabeling/VoxelNeXt-master/data/my/training',
                        help='specify the point cloud data file or directory')
    #  ../output/0913_0010/ckpt/checkpoint_epoch_80.pth ../output/08252007/ckpt/checkpoint_epoch_80.pth  ../voxelnext_nuscenes_kernel1.pth 
    parser.add_argument('--ckpt', type=str, default='../output/checkpoint_epoch_60.pth', help='specify the pretrained model')  #     
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    root_path = args.data_path
    dataset = load_dataset(SupportedDataset.hongjing, root_path)
    scene = dataset.scenes[0] # 
    frame = scene.frames[0] # 20
    lidar_sensor = 'true_value'
    # dataset_path = '/media/gpu/sde/tianxl/VoxelNeXt-master/data/my/training'
    # dataset = load_dataset(SupportedDataset.hongjing, dataset_path)
    # scene = dataset.scenes[0]
    # frame = scene.frames[0]
    # lidar_sensor = 'true_value'
    # coor = CoordinateSystem.world
    # point = frame.get_lidar_cloud(lidar_sensor, coor)
    
    
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), frame=frame, ext=args.ext, logger=logger
    )
    # demo_dataset = DemoDataset(
    #     dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
    #     root_path=Path(dataset_path), ext=args.ext, logger=logger
    # )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval() # 开启评估模式
    with torch.no_grad():
        # for idx, data_dict in enumerate(demo_dataset):
        for idx in range(len(demo_dataset)):
            data_dict = demo_dataset[idx]
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)
            
            coor = CoordinateSystem.ego
            # points_world = frame.get_lidar_cloud(lidar_sensor, coor)[:,0:4] # 世界坐标系下的点和框是正常的
            labels = frame.get_obstacles(coor)
            info = {}
            if labels is not None:
                annotations = {}
                annotations['name'] = np.array([obj.category.value for obj in labels])
                annotations['center'] = np.array([[obj.box.center.x, obj.box.center.y, obj.box.center.z] for obj in labels])
                annotations['size'] = np.array([[obj.box.size.x, obj.box.size.y, obj.box.size.z] for obj in labels])
                annotations['rotation_y'] = np.array([obj.box.rotation.z for obj in labels])
                
                annotations['rotation'] = np.array([[obj.box.rotation.x, obj.box.rotation.y, obj.box.rotation.z] for obj in labels])
                
                gt_boxes_lidar = np.concatenate([annotations['center'], annotations['size'], -(np.pi / 2 + annotations['rotation_y'][..., np.newaxis])], axis=1)
                gt_boxes_lidar_test = np.concatenate([annotations['center'], annotations['size'], annotations['rotation_y'][..., np.newaxis]], axis=1)
                annotations['gt_boxes_lidar'] = gt_boxes_lidar
                annotations['gt_boxes_lidar_test'] = gt_boxes_lidar_test
                annotations['num_points_in_gt'] = np.array([[obj.num_lidar_pts] for obj in labels])
                info['annos'] = annotations
            
            # pcdet定义的可视化
            V.draw_scenes(
                points=data_dict['points'][:, 1:], gt_boxes=info['annos']['gt_boxes_lidar_test'], ref_boxes=pred_dicts[0]['pred_boxes'],
                ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
            )
            # V.draw_scenes(
            #     points=points_world[:, 1:], gt_boxes=info['annos']['gt_boxes_lidar']
            # )
            mlab.show(stop=True)

            # 自己定义的可视化函数
            # fig = mlab.figure()
            # show_np_cloud(fig, points_world, Colors.White)
            # for i in range(annotations['center'].shape[0]):
            #     show_box(fig, annotations['center'][i,:], annotations['size'][i,:], annotations['rotation'][i,:])
            # mlab.show()
    
            

    logger.info('Demo done.')


if __name__ == '__main__':
    main()
