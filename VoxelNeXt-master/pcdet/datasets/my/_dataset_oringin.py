import copy
import pickle
import os

import numpy as np
from skimage import io
# 原来
# from . import hj_utils
# from ...ops.roiaware_pool3d import roiaware_pool3d_utils
# from ...utils import box_utils, calibration_kitti, common_utils, object3d_kitti
# from ..dataset import DatasetTemplate

# from .hj_dataset_devkit import SupportedDataset, load_dataset
# from .hj_dataset_devkit import CoordinateSystem

# debug
from pcdet.datasets.hj import hj_utils
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from pcdet.utils import box_utils, calibration_kitti, common_utils, object3d_kitti
from pcdet.datasets.dataset import DatasetTemplate

from pcdet.datasets.hj.hj_dataset_devkit import SupportedDataset, load_dataset
from pcdet.datasets.hj.hj_dataset_devkit import CoordinateSystem

class HjDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        # 初始化类，将参数赋值给 类的属性
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        # 传递参数是 训练集train 还是验证集val,{'train': 'train', 'test': 'val'}
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        # debug
        # 如果是训练集train，将文件的路径指为训练集training ，否则为测试集testing
        self.root_split_path = self.root_path / ('training' if self.split != 'val' else 'testing')
        # 改动(自己瞎改的，现在看有问题)：self.root_split_path = self.root_path / ('training' if self.split != 'val' else 'testing')
        # self.root_split_path = os.path.join(self.root_path, 'training' if self.split != 'test' else 'testing')
        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        # split_dir = os.path.join(self.root_path, 'ImageSets', (self.split + '.txt'))
        # split_dir = '/home/gpu/zengshuai/VoxelNeXt-master/pcdet/datasets/hj/ImageSets/train.txt'
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

        self.hj_infos = []
        self.include_hj_data(self.mode)

    def include_hj_data(self, mode):
        if self.logger is not None:
            self.logger.info('Loading HJ dataset')
        hj_infos = []
        
        for info_path in self.dataset_cfg.INFO_PATH[mode]:# hj_infos_val.pkl
            info_path = self.root_path / info_path
            if not info_path.exists():
                #如果该文件不存在，跳出，继续下一个文件
                continue
            with open(info_path, 'rb') as f:
                # pickle.load(f) 将该文件中的数据 解析为一个Python对象infos，并将该内容添加到kitti_infos 列表中
                infos = pickle.load(f)
                hj_infos.extend(infos)

        self.hj_infos.extend(hj_infos)

        # 最后在日志信息中 添加数据集样本总个数
        if self.logger is not None:
            self.logger.info('Total samples for HJ dataset: %d' % (len(hj_infos)))

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training, root_path=self.root_path, logger=self.logger
        )
        self.split = split
        # mapping_dict = {
        #     "train": "training",
        #     "test": "testing",
        #     "val": "valing"
        # }
        # data_path = mapping_dict.get(self.split, self.split)
        self.root_split_path = self.root_path / ('training' if self.split == 'train' else 'testing')
        # self.root_split_path = self.root_path / data_path
        
        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None


    # 根据序列号，获取lidar信息
    def get_lidar(self, sample_idx,frame_idx, lidar_sensor):
        lidar_file = self.root_split_path / ('%s' % sample_idx) /'lidars' / ('%s' % lidar_sensor) /('%s.bin' % frame_idx)
        assert lidar_file.exists()
        #读取该 bin文件类型，并将点云数据以 numpy的格式输出！！！
        #并且将数据 转换成 每行5个数据，刚好是一个点云数据的四个参数：X,Y,Z,R(强度或反射值），时间戳
        return np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 5)

    def get_image(self, idx):
        """
        Loads image for a sample
        Args:
            idx: int, Sample index
        Returns:
            image: (H, W, 3), RGB Image
        """
        img_file = self.root_split_path / 'image_2' / ('%s.png' % idx)
        assert img_file.exists()
        image = io.imread(img_file)
        image = image.astype(np.float32)
        image /= 255.0
        return image

    def get_image_shape(self, idx):
        img_file = self.root_split_path / 'image_2' / ('%s.png' % idx)
        assert img_file.exists()
        return np.array(io.imread(img_file).shape[:2], dtype=np.int32)

    def get_label(self, idx):
        label_file = self.root_split_path / 'label_2' / ('%s.txt' % idx)
        assert label_file.exists()
        return object3d_kitti.get_objects_from_label(label_file)

    def get_depth_map(self, idx):
        """
        Loads depth map for a sample
        Args:
            idx: str, Sample index
        Returns:
            depth: (H, W), Depth map
        """
        depth_file = self.root_split_path / 'depth_2' / ('%s.png' % idx)
        assert depth_file.exists()
        depth = io.imread(depth_file)
        depth = depth.astype(np.float32)
        depth /= 256.0
        return depth

    def get_calib(self, idx):
        calib_file = self.root_split_path / 'calib' / ('%s.txt' % idx)
        assert calib_file.exists()
        return calibration_kitti.Calibration(calib_file)

    def get_road_plane(self, idx):
        plane_file = self.root_split_path / 'planes' / ('%s.txt' % idx)
        if not plane_file.exists():
            return None

        with open(plane_file, 'r') as f:
            lines = f.readlines()
        lines = [float(i) for i in lines[3].split()]
        plane = np.asarray(lines)

        # Ensure normal is always facing up, this is in the rectified camera coordinate
        if plane[1] > 0:
            plane = -plane

        norm = np.linalg.norm(plane[0:3])
        plane = plane / norm
        return plane

    @staticmethod
    def get_fov_flag(pts_rect, img_shape, calib):
        """
        Args:
            pts_rect:
            img_shape:
            calib:

        Returns:

        """
        pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

        return pts_valid_flag

    def get_infos(self, num_workers=4, has_label=True, count_inside_pts=True, sample_id_list=None):
        import concurrent.futures as futures

        dataset = load_dataset(SupportedDataset.hongjing, self.root_split_path)
        # 一个问题：输入的是两个包，为啥只找到一个场景，sample_idx只有0，正常应该是0，1
        print(dataset.scenes)
        scenes = dataset.scenes
        if self.logger is not None:
            self.logger.info('Total samples for HJ %s dataset: %d' % (self.split, len(scenes)))
        def process_single_sequence(sample_idx):
            print('%s sample_idx: %s' % (self.split, sample_idx))
            frames = scenes[sample_idx].frames
            infos = []
            for idx, frame in enumerate(frames):
                # lidar_sensor = frame.lidars[-1]
                # lidar_sensor = 'hesi_topc_0_raw'
                lidar_sensor = 'true_value'
                info = {}
                frame_idx = frame.get_timestamp(lidar_sensor)
                # print(len(sample_id_list))
                # print(idx)
                pc_info = {'num_features': 4, 'lidar_idx': frame_idx, 'sequence_idx':self.sample_id_list[sample_idx], 'lidar_sensor':lidar_sensor}
                info['point_cloud'] = pc_info
                #print(self.sample_id_list[sample_idx])
                # PosixPath('/home/gpu/zengshuai/VoxelNeXt-master/data/hj/training/2023_04_15_12_10_20_city/lidars/hesi_topc_0_raw/1681531820079530.bin')
                lidar_file = self.root_split_path / ('%s' % self.sample_id_list[sample_idx]) /'lidars' / ('%s' % lidar_sensor) /('%s.bin' % frame_idx)
                assert lidar_file.exists()
                #info['calib'] = frame.calib
                labels = frame.get_obstacles(CoordinateSystem.ego)
                if has_label and labels is not None:
                    annotations = {}
                    annotations['name'] = np.array([obj.category.value for obj in labels])
                    annotations['center'] = np.array([[obj.box.center.x, obj.box.center.y, obj.box.center.z] for obj in labels])
                    annotations['size'] = np.array([[obj.box.size.x, obj.box.size.y, obj.box.size.z] for obj in labels])
                    annotations['rotation_y'] = np.array([obj.box.rotation.z for obj in labels])

                    gt_boxes_lidar = np.concatenate([annotations['center'], annotations['size'], -(np.pi / 2 + annotations['rotation_y'][..., np.newaxis])], axis=1)
                    annotations['gt_boxes_lidar'] = gt_boxes_lidar
                    annotations['num_points_in_gt'] = np.array([[obj.num_lidar_pts] for obj in labels])
                    info['annos'] = annotations
                else:
                    continue
                    # 改动
                    # if self.split != 'test':
                    #     continue
                    # else:
                    #     infos.append(info)
                    #     continue
                # if self.split == 'train':
                infos.append(info)
            
            
            return infos

        # .txt文件下的序列号，组成列表sample_id_list，上面的函数的是一个场景/序列的信息
        # 下面几行是将该sample_id_list列表上的都执行一下，每个返回的信息info都存放在infos里面
        # 最后执行完成后，infos是一个列表，每一个元素代表了一个场景/序列的信息
        # sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list
        sample_id_list = [x for x in range(len(scenes))] # 帅改的，不知道啥意思？
        
        # 下面是异步线程的处理方式
        # executor 是 concurrent.futures.ThreadPoolExecutor 的一个实例，
        # map 方法将函数 process_single_sequence 应用到 sample_id_list 中的每个元素上，使用多个线程并行地进行处理。
        with futures.ThreadPoolExecutor(num_workers) as executor:
            # infos = executor.map(process_single_sequence,sample_id_list)
            infos = executor.map(process_single_sequence,sample_id_list)
        res_info = []
        for info in infos:
            res_info.extend(info)
        return res_info

    def create_groundtruth_database(self, info_path=None, used_classes=None, split='train'):
        import torch

        database_save_path = Path(self.root_path) / ('gt_database' if split == 'train' else ('gt_database_%s' % split))
        db_info_save_path = Path(self.root_path) / ('hj_dbinfos_%s.pkl' % split)

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        for k in range(len(infos)):
            if k % 10 == 0:
                print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            info = infos[k]
            sample_idx = info['point_cloud']['sequence_idx']
            frame_idx = info['point_cloud']['lidar_idx']
            # sample_idx: Any, frame_idx: Any, lidar_sensor: Any hesi_topc_0_raw
            points = self.get_lidar(sample_idx, frame_idx, lidar_sensor = 'true_value')
            annos = info['annos']
            names = annos['name']
            # difficulty = annos['difficulty']
            # bbox = annos['bbox']
            gt_boxes = annos['gt_boxes_lidar']
            num_points_in_gt = annos['num_points_in_gt']
            # num_obj是有效物体的个数，为N
            num_obj = gt_boxes.shape[0]
            # # 返回每个box中的点云索引[0 0 0 1 0 1 1...]
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()  # (nboxes, npoints) (19, 200077)

            for i in range(num_obj):
                # 创建文件名，并设置保存路径，最后文件如：'2023_04_15_11_47_18_highway_car_0.bin'
                # filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)
                filename = '%s_%s_%d_infos_%d.bin' % (sample_idx, names[i], i, k)
                # /home/gpu/zengshuai/VoxelNeXt-master/data/hj/gt_database/2023_04_15_12_10_20_city_car_11_infos_134.bin
                filepath = database_save_path / filename
                # print(gt_boxes[i, :].reshape(-1, 7))
                # 返回每个box中的点云索引[0 0 0 1 0 1 1...]
                # point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                #     torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes[i, :].reshape(-1, 7))
                # ).numpy()  # (nboxes, npoints)                
                # point_indices[i] > 0得到的是一个[T,F,T,T,F...]之类的真假索引，共有M个
                # 再从points中取出相应为true的点云数据，放在gt_points中
                gt_points = points[point_indices[i] > 0]

                # 将第i个box内点转化为局部坐标
                gt_points[:, :3] -= gt_boxes[i, :3]
                # 把gt_points的信息写入文件里
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                if (used_classes is None) or names[i] in used_classes:
                    # 获取文件相对路径
                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    # 根据当前物体的信息组成info
                    # db_info = {'name': names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,
                    #            'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0],
                    #            'difficulty': difficulty[i], 'bbox': bbox[i], 'score': annos['score'][i]}
                    # db_info = {'name': names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,
                    #            'box3d_lidar': gt_boxes[i], 'num_points_in_gt': num_points_in_gt[i]}
                    db_info = {'name': names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,
                               'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0]}                   
                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts（预测结果）
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:检测结果输出

        Returns:(同时将检测结果以字符串形式输出到frame_id.txt文件中)
        	annos:
                'name': np.array(class_names)[pred_labels - 1]
                'alpha': -np.arctan2(-pred_boxes[:, 1], pred_boxes[:, 0]) + pred_boxes_camera[:, 6]
                'bbox': pred_boxes_img
                'dimensions':pred_boxes_camera[:, 3:6]
                'location': pred_boxes_camera[:, 0:3]
                'rotation_y': pred_boxes_camera[:, 6]
                'score': pred_scores
                'boxes_lidar': pred_boxes
                'frame_id'：frame_id
        """
        def get_template_prediction(num_samples):
            ret_dict = {
                # 'name': np.zeros(num_samples), 'truncated': np.zeros(num_samples),
                # 'occluded': np.zeros(num_samples), 'alpha': np.zeros(num_samples),
                # 'bbox': np.zeros([num_samples, 4]), 'dimensions': np.zeros([num_samples, 3]),
                # 'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
                # 'score': np.zeros(num_samples), 'boxes_lidar': np.zeros([num_samples, 7])
                'name': np.zeros(num_samples), 'alpha': np.zeros(num_samples),
                'bbox': np.zeros([num_samples, 4]), 'dimensions': np.zeros([num_samples, 3]),
                'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
                'score': np.zeros(num_samples), 'boxes_lidar': np.zeros([num_samples, 7])
            }
            return ret_dict

        def generate_single_sample_dict(batch_index, box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            # calib = batch_dict['calib'][batch_index]
            # image_shape = batch_dict['image_shape'][batch_index].cpu().numpy()
            # pred_boxes_camera = box_utils.boxes3d_lidar_to_kitti_camera(pred_boxes, calib)
            # pred_boxes_img = box_utils.boxes3d_kitti_camera_to_imageboxes(
            #     pred_boxes_camera, calib, image_shape=image_shape
            # )
            # pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            # pred_dict['alpha'] = -np.arctan2(-pred_boxes[:, 1], pred_boxes[:, 0]) + pred_boxes_camera[:, 6]
            # pred_dict['bbox'] = pred_boxes_img
            # pred_dict['dimensions'] = pred_boxes_camera[:, 3:6]
            # pred_dict['location'] = pred_boxes_camera[:, 0:3]
            # pred_dict['rotation_y'] = pred_boxes_camera[:, 6]
            # pred_dict['score'] = pred_scores
            # pred_dict['boxes_lidar'] = pred_boxes
            
            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['alpha'] = -np.arctan2(-pred_boxes[:, 1], pred_boxes[:, 0]) + pred_boxes[:, 6]
            pred_dict['dimensions'] = pred_boxes[:, 3:6] # l, w, h
            pred_dict['location'] = pred_boxes[:, 0:3]
            pred_dict['rotation_y'] = pred_boxes[:, 6]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes
            
            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            frame_id = batch_dict['frame_id'][index]

            single_pred_dict = generate_single_sample_dict(index, box_dict)
            single_pred_dict['frame_id'] = frame_id
            annos.append(single_pred_dict)

            if output_path is not None:
                cur_det_file = output_path / ('%s.txt' % frame_id)
                with open(cur_det_file, 'w') as f:
                    bbox = single_pred_dict['bbox']
                    loc = single_pred_dict['location']
                    dims = single_pred_dict['dimensions']  # lhw -> hwl

                    for idx in range(len(bbox)):
                        print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                              % (single_pred_dict['name'][idx], single_pred_dict['alpha'][idx],
                                 bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3],
                                 dims[idx][1], dims[idx][2], dims[idx][0], loc[idx][0],
                                 loc[idx][1], loc[idx][2], single_pred_dict['rotation_y'][idx],
                                 single_pred_dict['score'][idx]), file=f)

        return annos

    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.hj_infos[0].keys():
            return None, {}

        from .kitti_object_eval_python import eval as kitti_eval

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.hj_infos]
        ap_result_str, ap_dict = kitti_eval.get_official_eval_result(eval_gt_annos, eval_det_annos, class_names)

        return ap_result_str, ap_dict

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.hj_infos) * self.total_epochs # hj_infos即从.pkl中读取的信息
        # 等于返回训练帧的总个数，等于图片的总个数
        return len(self.hj_infos)


    def __getitem__(self, index):
        '''
        从pkl文件中获取相应index的info，然后根据info['point_cloud']['lidar_idx']确定帧号，进行数据读取和其他info字段的读取。
        初步读取的data_dict,要传入prepare_data（dataset.py父类中定义）进行统一处理，然后即可返回
        在 self._getitem_() 中加载自己的数据，
        并将点云与3D标注框均转至前述统一坐标定义下，送入数据基类提供的 self.prepare_data()；
        参数index 是需要送进来处理的 帧序号的索引值，如1,2,3,4.。。。
        Args:
            index: n
        Returns:
            data_dict:
                'points': points,
                'frame_id': sample_idx,
                'calib': calib,
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
                'road_plane': road_plane(optional for training)
                'image_shape': img_shape
        '''
        # index = 4
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.hj_infos)

        info = copy.deepcopy(self.hj_infos[index])

        sample_idx = info['point_cloud']['sequence_idx'] # '2023_04_15_12_10_20_city'
        frame_idx = info['point_cloud']['lidar_idx'] # 1681531833430014
        lidar_sensor = info['point_cloud']['lidar_sensor'] # 'true_value'
        get_item_list = self.dataset_cfg.get('GET_ITEM_LIST', ['points']) # points
        
        input_dict = {
            'frame_id': sample_idx,
        }
        
        if 'annos' in info:
            annos = info['annos']
            #annos = common_utils.drop_info_with_name(annos, name='DontCare')
            gt_names = annos['name']
            gt_boxes_lidar = annos['gt_boxes_lidar']
            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })
            if "gt_boxes2d" in get_item_list:
                input_dict['gt_boxes2d'] = annos["bbox"]

        if "points" in get_item_list:
            points = self.get_lidar(sample_idx,frame_idx,lidar_sensor)[:,0:4]
            # if self.dataset_cfg.FOV_POINTS_ONLY:
            #     pts_rect = calib.lidar_to_rect(points[:, 0:3])
            #     fov_flag = self.get_fov_flag(pts_rect, img_shape, calib)
            #     points = points[fov_flag]
            input_dict['points'] = points

        if "images" in get_item_list:
            input_dict['images'] = self.get_image(sample_idx)

        if "depth_maps" in get_item_list:
            input_dict['depth_maps'] = self.get_depth_map(sample_idx)

        if "calib_matricies" in get_item_list:
            input_dict["trans_lidar_to_cam"], input_dict["trans_cam_to_img"] = hj_utils.calib_to_matricies(calib)

        #input_dict['calib'] = calib
        data_dict = self.prepare_data(data_dict=input_dict)

        #data_dict['image_shape'] = img_shape
        return data_dict

# 生成.pkl文件（对train/test/val均生成相应文件），提前读取点云格式、image格式、calib矩阵以及label
def create_hj_infos(dataset_cfg, class_names, data_path, save_path, workers=4):
    dataset = HjDataset(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=False)
    train_split, val_split = 'train', 'val'

    train_filename = save_path / ('hj_infos_%s.pkl' % train_split)
    val_filename = save_path / ('hj_infos_%s.pkl' % val_split)
    trainval_filename = save_path / 'hj_infos_trainval.pkl'
    test_filename = save_path / 'hj_infos_test.pkl'
    
    print('---------------Start to generate data infos---------------')

    dataset.set_split(train_split)
    # 执行完上一步，得到train相关的保存文件，以及sample_id_list的值为train.txt文件下的数字
    # 下面是得到train.txt中序列相关的所有点云数据的信息，并且进行保存
    hj_infos_train = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    print('Num of Train Infos: ',len(hj_infos_train))
    with open(train_filename, 'wb') as f:
        pickle.dump(hj_infos_train, f)
    print('Hj info train file is saved to %s' % train_filename)

    dataset.set_split(val_split)
    hj_infos_val = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    print('Num of val Infos: ',len(hj_infos_val))
    with open(val_filename, 'wb') as f:
        pickle.dump(hj_infos_val, f)
    print('Hj info val file is saved to %s' % val_filename)
    
    # 把训练集和验证集的信息 合并写到一个文件里
    print('Num of val Infos: ',len(hj_infos_train) + len(hj_infos_val))
    with open(trainval_filename, 'wb') as f:
        pickle.dump(hj_infos_train + hj_infos_val, f)
    print('Hj info trainval file is saved to %s' % trainval_filename)

    # dataset.set_split('test')
    # hj_infos_test = dataset.get_infos(num_workers=workers, has_label=False, count_inside_pts=False)
    # print('Num of test Infos: ',len(hj_infos_test))
    # with open(test_filename, 'wb') as f:
    #     pickle.dump(hj_infos_test, f)
    # print('Hj info test file is saved to %s' % test_filename)
    

    # 有问题需要改
    print('---------------Start create groundtruth database for data augmentation---------------')
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(train_filename, split=train_split)

    print('---------------Data preparation Done---------------')


if __name__ == '__main__':
    import sys
    if sys.argv.__len__() > 1 and sys.argv[1] == 'create_hj_infos':
        import yaml
        from pathlib import Path
        from easydict import EasyDict
        dataset_cfg = EasyDict(yaml.safe_load(open(sys.argv[2])))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        create_hj_infos(
            dataset_cfg=dataset_cfg,
            class_names=['Car', 'Pedestrian', 'Cyclist'],
            data_path=ROOT_DIR / 'data' / 'hj',
            save_path=ROOT_DIR / 'data' / 'hj'
        )
