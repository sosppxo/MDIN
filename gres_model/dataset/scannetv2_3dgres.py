import glob, json, os
import math
import numpy as np
import os.path as osp
import pointgroup_ops
import scipy.interpolate as interpolate
import scipy.ndimage as ndimage
import torch
import torch_scatter
from torch.utils.data import Dataset
from typing import Dict, Sequence, Tuple, Union
from tqdm import tqdm
from gorilla import is_main_process
import sng_parser
from transformers import RobertaTokenizerFast

MAX_NUM_OBJ = 132


class ScanNetDataset_sample_graph_edge(Dataset):

    CLASSES = ('cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'desk',
               'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture')
    NYU_ID = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39)

    def __init__(self,
                 data_root,
                 prefix,
                 suffix,
                 voxel_cfg=None,
                 training=True,
                 with_label=True,
                 mode=4,
                 with_elastic=True,
                 aug=False,
                 use_xyz=True,
                 logger=None,
                 max_des_len=78,
                 graph_pos_enc_dim=5,
                 bidirectional=False,
                 lang_num_max=16,
                 cl=False,
                 src_sample=-1,
                 scene_graph=False,
                 dataset='scanrefer'
                 ):
        self.data_root = data_root
        self.dataset = dataset
        self.prefix = prefix
        self.suffix = suffix
        self.voxel_cfg = voxel_cfg
        self.training = training
        self.with_label = with_label
        self.mode = mode
        self.with_elastic = with_elastic
        self.aug = aug
        self.use_xyz = use_xyz
        self.logger = logger
        self.max_des_len = max_des_len
        self.bidirectional = bidirectional
        self.scene_graph = scene_graph
        self.depend2id = torch.load(os.path.join(self.data_root, 'dependency_map.pth'))['depend2id']
        self.id2depend = torch.load(os.path.join(self.data_root, 'dependency_map.pth'))['id2depend']
        self.graph_pos_enc_dim = graph_pos_enc_dim
        self.sp_filenames = self.get_sp_filenames()
        self.cl = cl
        self.src_sample = src_sample
        self.n=0
        self.m=0
        
        np.random.seed(1999)
        
        if self.dataset == 'scanrefer':
            # load scanrefer
            if self.prefix == 'train':
                self.scanrefer = json.load(open(os.path.join(self.data_root, 'ScanRefer', 'ScanRefer_filtered_train_new.json')))
                if is_main_process(): self.logger.info(f'Load {self.prefix} scanrefer: {len(self.scanrefer)} samples')
            elif self.prefix == 'val':
                self.scanrefer = json.load(open(os.path.join(self.data_root, 'ScanRefer', 'ScanRefer_filtered_val_new.json')))
                if is_main_process(): self.logger.info(f'Load {self.prefix} scanrefer: {len(self.scanrefer)} samples')
            else:
                raise ValueError('ScanRefer only support train and val split, not support %s' % self.prefix)
        elif self.dataset == 'nr3d':
            # load nr3d
            if self.prefix == 'train':
                self.scanrefer = json.load(open(os.path.join(self.data_root, 'ReferIt3D', 'nr3d_train.json')))
                if is_main_process(): self.logger.info(f'Load {self.prefix} nr3d: {len(self.scanrefer)} samples')
            elif self.prefix == 'val':
                self.scanrefer = json.load(open(os.path.join(self.data_root, 'ReferIt3D', 'nr3d_val.json')))
                if is_main_process(): self.logger.info(f'Load {self.prefix} nr3d: {len(self.scanrefer)} samples')
            else:
                raise ValueError('nr3d only support train and val split, not support %s' % self.prefix)
        elif self.dataset == 'sr3d':
            # load sr3d
            if self.prefix == 'train':
                self.scanrefer = json.load(open(os.path.join(self.data_root, 'ReferIt3D', 'sr3d_train.json')))
                if is_main_process(): self.logger.info(f'Load {self.prefix} sr3d: {len(self.scanrefer)} samples')
            elif self.prefix == 'val':
                self.scanrefer = json.load(open(os.path.join(self.data_root, 'ReferIt3D', 'sr3d_val.json')))
                if is_main_process(): self.logger.info(f'Load {self.prefix} sr3d: {len(self.scanrefer)} samples')
            else:
                raise ValueError('sr3d only support train and val split, not support %s' % self.prefix)
        elif self.dataset == 'multi3drefer':
            # load multi3drefer
            if self.prefix == 'train':
                self.scanrefer = json.load(open(os.path.join(self.data_root, 'M3DRefer', 'multi3drefer_train.json')))
                if is_main_process(): self.logger.info(f'Load {self.prefix} multi3drefer: {len(self.scanrefer)} samples')
            elif self.prefix == 'val':
                self.scanrefer = json.load(open(os.path.join(self.data_root, 'M3DRefer', 'multi3drefer_val.json')))
                if is_main_process(): self.logger.info(f'Load {self.prefix} multi3drefer: {len(self.scanrefer)} samples')
        else:
            raise NotImplementedError
        
        self.scanrefer.sort(key=lambda x: x['scene_id'])
        scene_list = sorted(list(set([data["scene_id"] for data in self.scanrefer])))
        scanrefer_new = []
        scanrefer_new_scene = []
        scene_id = ""
        for data in self.scanrefer:
            if data["scene_id"] in scene_list:
                if scene_id != data["scene_id"]:
                    scene_id = data["scene_id"]
                    if len(scanrefer_new_scene) > 0:
                        scanrefer_new.append(scanrefer_new_scene)
                    scanrefer_new_scene = []
                if len(scanrefer_new_scene) >= lang_num_max:
                    scanrefer_new.append(scanrefer_new_scene)
                    scanrefer_new_scene = []
                scanrefer_new_scene.append(data)
        scanrefer_new.append(scanrefer_new_scene)
        self.scene_inputs = scanrefer_new
        
        self.scene_graphs = self.load_scene_graphs()
        
        # main(instance seg task) with others
        self.type2class = {'cabinet':0, 'bed':1, 'chair':2, 'sofa':3, 'table':4, 'door':5,
            'window':6,'bookshelf':7,'picture':8, 'counter':9, 'desk':10, 'curtain':11,
            'refrigerator':12, 'shower curtain':13, 'toilet':14, 'sink':15, 'bathtub':16, 'others':17}  
        self.class2type = {self.type2class[t]:t for t in self.type2class}
        self.nyu40ids = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]) # exclude wall (1), floor (2), ceiling (22)
        self.nyu40id2class = self._get_nyu40id2class()
        self.sem2nyu = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]

        self.tokenizer = RobertaTokenizerFast.from_pretrained('./backbones/roberta-base/', local_files_only=True)



    def load_scene_graphs(self):
        scene_graphs = {}
        # if self.dataset == 'scanrefer':
        #     scene_graphs_path = os.path.join(self.data_root, self.dataset, 'scene_graphs_new_'+self.prefix+'.json')
        # else:
        #     scene_graphs_path = os.path.join(self.data_root, self.dataset, 'scene_graphs_'+self.prefix+'.json')
        scene_graphs_path = os.path.join(self.data_root, self.dataset, 'scene_graphs_'+self.prefix+'.json')
        if os.path.exists(scene_graphs_path):
            scene_graphs = json.load(open(scene_graphs_path))
        else:
            print('Begin '+ self.prefix +' text decoupling (scene graphs)...')
            for data in tqdm(self.scanrefer):
                scene_id = str(data['scene_id'])
                ann_id = str(data['ann_id'])
                
                if scene_id not in scene_graphs:
                    scene_graphs[scene_id] = {}
                if ann_id not in scene_graphs[scene_id]:
                    scene_graphs[scene_id][ann_id] = {}

                if self.dataset == 'scanrefer':
                    scene_graphs[scene_id][ann_id] = Scene_graph_parse(' '.join(data['token']))
                elif self.dataset == 'multi3drefer':
                    scene_graphs[scene_id][ann_id] = Scene_graph_parse(data["description"])
                elif self.dataset == 'nr3d':
                    scene_graphs[scene_id][ann_id] = Scene_graph_parse(' '.join(data['token']))
                elif self.dataset == 'sr3d':
                    scene_graphs[scene_id][ann_id] = Scene_graph_parse(' '.join(data['token']))
            
            print('Saving '+ self.prefix +' text decoupling (scene graphs)...')
            json.dump(scene_graphs, open(scene_graphs_path, 'w'))

            print('Done '+ self.prefix +' text decoupling (scene graphs).')
        
        return scene_graphs
    
    def _get_type2class_all(self):
        lines = [line.rstrip() for line in open(os.path.join(self.data_root, 'scannetv2-labels.combined.tsv'))]
        lines = lines[1:]
        type2class = {}
        for i in range(len(lines)):
            elements = lines[i].split('\t')
            nyu40_id = int(elements[4])
            nyu40_name = elements[7]
            if nyu40_id in self.nyu40ids:
                class_id = np.where(self.nyu40ids == nyu40_id)[0][0]
                type2class[nyu40_name] = class_id
        return type2class
    
    def _get_nyu40id2class(self):
        lines = [line.rstrip() for line in open(os.path.join(self.data_root, 'scannetv2-labels.combined.tsv'))]
        lines = lines[1:]
        nyu40ids2class = {}
        for i in range(len(lines)):
            label_classes_set = set(self.type2class.keys())
            elements = lines[i].split('\t')
            nyu40_id = int(elements[4])
            nyu40_name = elements[7]
            if nyu40_id in self.nyu40ids:
                if nyu40_name not in label_classes_set:
                    nyu40ids2class[nyu40_id] = self.type2class["others"]
                else:
                    nyu40ids2class[nyu40_id] = self.type2class[nyu40_name]
        return nyu40ids2class


    def get_sp_filenames(self):
        filenames = glob.glob(osp.join(self.data_root, 'scannetv2', self.prefix, '*' + '_refer_l2.pth'))
        assert len(filenames) > 0, 'Empty dataset.'
        filenames = sorted(filenames)
        return filenames
        
    def load(self, filename):
        if self.with_label:
            return torch.load(filename)
        else:
            xyz, rgb, superpoint = torch.load(filename)
            dummy_sem_label = np.zeros(xyz.shape[0], dtype=np.float32)
            dummy_inst_label = np.zeros(xyz.shape[0], dtype=np.float32)
            return xyz, rgb, superpoint, dummy_sem_label, dummy_inst_label
        
    def transform_train(self, xyz, rgb, superpoint, superpoint_l2, semantic_label, instance_label):
        if self.aug:
            xyz_middle = self.data_aug(xyz, True, True, True)
        else:
            xyz_middle = xyz.copy()
        rgb += np.random.randn(3) * 0.1
        xyz = xyz_middle * self.voxel_cfg.scale
        if self.with_elastic:
            xyz = self.elastic(xyz, 6, 40.)
            xyz = self.elastic(xyz, 20, 160.)
        xyz = xyz - xyz.min(0)
        # xyz, valid_idxs = self.crop(xyz)
        # random sample instead of crop
        valid_idxs = self.sample_rdn(xyz)
        xyz_middle = xyz_middle[valid_idxs]
        xyz = xyz[valid_idxs]
        rgb = rgb[valid_idxs]
        semantic_label = semantic_label[valid_idxs]
        superpoint = np.unique(superpoint[valid_idxs], return_inverse=True)[1]
        # instance_label = self.get_cropped_inst_label(instance_label, valid_idxs)
        instance_label = instance_label[valid_idxs]
        return xyz, xyz_middle, rgb, superpoint, semantic_label, instance_label

    def transform_test(self, xyz, rgb, superpoint, superpoint_l2, semantic_label, instance_label):
        xyz_middle = xyz
        xyz = xyz_middle * self.voxel_cfg.scale
        xyz -= xyz.min(0)
        if self.src_sample > 0:
            # np.random.seed(1184)
            valid_idxs = np.random.choice(
                xyz.shape[0],
                self.src_sample,
                replace=xyz.shape[0] < self.src_sample
            )
        else:
            valid_idxs = np.ones(xyz.shape[0], dtype=bool)
        xyz = xyz[valid_idxs]
        xyz_middle = xyz_middle[valid_idxs]
        rgb = rgb[valid_idxs]
        semantic_label = semantic_label[valid_idxs]
        superpoint = np.unique(superpoint[valid_idxs], return_inverse=True)[1]
        instance_label = instance_label[valid_idxs]
        # instance_label = self.get_cropped_inst_label(instance_label, valid_idxs)
        return xyz, xyz_middle, rgb, superpoint, semantic_label, instance_label

    def data_aug(self, xyz, jitter=False, flip=False, rot=False):
        m = np.eye(3)
        if jitter:
            m += np.random.randn(3, 3) * 0.1
        if flip:
            m[0][0] *= np.random.randint(0, 2) * 2 - 1  # flip x randomly
        if rot:
            theta = np.random.rand() * 2 * math.pi
            m = np.matmul(
                m,
                [[math.cos(theta), math.sin(theta), 0], [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]])  # rotation
        return np.matmul(xyz, m)

    def sample_rdn(self, xyz: np.ndarray) -> Union[np.ndarray, np.ndarray]:
        if xyz.shape[0] > self.voxel_cfg.max_npoint:
            valid_idxs = np.random.choice(
                xyz.shape[0],
                self.voxel_cfg.max_npoint,
                replace=xyz.shape[0] < self.voxel_cfg.max_npoint
            )
            return valid_idxs
        else:
            valid_idxs = np.ones(xyz.shape[0], dtype=bool)
            return valid_idxs
    
    def crop(self, xyz: np.ndarray) -> Union[np.ndarray, np.ndarray]:
        r"""
        crop the point cloud to reduce training complexity

        Args:
            xyz (np.ndarray, [N, 3]): input point cloud to be cropped

        Returns:
            Union[np.ndarray, np.ndarray]: processed point cloud and boolean valid indices
        """
        xyz_offset = xyz.copy()
        valid_idxs = xyz_offset.min(1) >= 0
        assert valid_idxs.sum() == xyz.shape[0]

        full_scale = np.array([self.voxel_cfg.spatial_shape[1]] * 3)
        room_range = xyz.max(0) - xyz.min(0)
        while valid_idxs.sum() > self.voxel_cfg.max_npoint:
            offset = np.clip(full_scale - room_range + 0.001, None, 0) * np.random.rand(3)
            xyz_offset = xyz + offset
            valid_idxs = (xyz_offset.min(1) >= 0) * ((xyz_offset < full_scale).sum(1) == 3)
            full_scale[:2] -= 32

        return xyz_offset, valid_idxs

    def elastic(self, xyz, gran, mag):
        """Elastic distortion (from point group)

        Args:
            xyz (np.ndarray): input point cloud
            gran (float): distortion param
            mag (float): distortion scalar

        Returns:
            xyz: point cloud with elastic distortion
        """
        blur0 = np.ones((3, 1, 1)).astype('float32') / 3
        blur1 = np.ones((1, 3, 1)).astype('float32') / 3
        blur2 = np.ones((1, 1, 3)).astype('float32') / 3

        bb = np.abs(xyz).max(0).astype(np.int32) // gran + 3
        noise = [np.random.randn(bb[0], bb[1], bb[2]).astype('float32') for _ in range(3)]
        noise = [ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
        noise = [ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
        noise = [ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
        noise = [ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
        noise = [ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
        noise = [ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
        ax = [np.linspace(-(b - 1) * gran, (b - 1) * gran, b) for b in bb]
        interp = [interpolate.RegularGridInterpolator(ax, n, bounds_error=0, fill_value=0) for n in noise]

        def g(xyz_):
            return np.hstack([i(xyz_)[:, None] for i in interp])

        return xyz + g(xyz) * mag
    
    def get_ref_mask(self, coord_float, instance_label, superpoint, object_ids):
        ref_lbl = torch.zeros_like(instance_label)
        for object_id in object_ids:
            ref_lbl[instance_label == object_id] = 1
        gt_spmask = torch_scatter.scatter_mean(ref_lbl.float(), superpoint, dim=-1)
        gt_spmask = (gt_spmask > 0.5).float()
        gt_pmask = ref_lbl.float()
        return gt_pmask, gt_spmask
    
    def get_sp_label(self, superpoint, instance_label, semantic_label):
        sp_ins_label = torch.zeros(len(superpoint.unique()))
        sp_ins_label[:] = 199 # ins label for useless superpoint 
        sem2ins = {}
        for instance in instance_label.unique():
            ref_lbl = instance_label == instance
            gt_spmask = torch_scatter.scatter_mean(ref_lbl.float(), superpoint, dim=-1)
            gt_spmask = gt_spmask > 0.5
            sp_ins_label[gt_spmask] = instance
        for semantic in semantic_label.unique().tolist():
            ref_lbl = semantic_label == semantic
            sem2ins[semantic] = instance_label[ref_lbl].unique()
        return sp_ins_label, sem2ins
    
    def __len__(self):
        #return len(self.filenames)
        return len(self.scene_inputs)
    
    def __getitem__(self, index: int) -> Tuple:

        ann_ids, object_idss, gt_pmasks, gt_spmasks, sp_ref_masks, lang_utterances, sem_ids, dense_maps = [], [], [], [], [], [], [], []
        nsubj_inds, nsubj_names = [], []
        meta_datas = []
        view_dependents = []
        scene_input = self.scene_inputs[index]
        for i in range(len(scene_input)):
            data = scene_input[i]
            scan_id = data['scene_id']
            
            if i==0:
                for fn in self.sp_filenames:
                    if scan_id in fn:
                        sp_filename = fn
                        break
                scene = self.load(sp_filename)
                scene = self.transform_train(*scene) if self.training else self.transform_test(*scene)
                xyz, xyz_middle, rgb, superpoint, semantic_label, instance_label = scene
                coord = torch.from_numpy(xyz).long()
                coord_float = torch.from_numpy(xyz_middle).float()
                feat = torch.from_numpy(rgb).float()
                superpoint = torch.from_numpy(superpoint)
                semantic_label = torch.from_numpy(semantic_label).long()
                instance_label = torch.from_numpy(instance_label).long()
                sp_ins_label, sem2ins = self.get_sp_label(superpoint, instance_label, semantic_label)
            
            if 'object_ids' in data.keys():
                object_ids = data['object_ids']
            else:
                object_ids = [int(data['object_id'])]
            ann_id = int(data['ann_id'])
            # lang_tokens = data['token']
            if self.dataset == 'multi3drefer':
                meta_data = {key:data[key] 
                             for key in data.keys() 
                             if key in ['eval_type','spatial','color','texture','shape']}
                view_dependent = None
            elif self.dataset == 'nr3d' or self.dataset == 'sr3d':
                meta_data = data['meta_data']
                view_dependent = data['view_dependent']
            else:
                meta_data = None
                view_dependent = None
            
            anno = self.scene_graphs[scan_id][str(ann_id)]
            auxi_box = None
            tokens_positive, positive_map, modify_positive_map, pron_positive_map, \
                other_entity_map, auxi_entity_positive_map, rel_positive_map = self._get_token_positive_map_by_parse(anno, auxi_box)

            dense_map = {
                'tokens_positive': torch.from_numpy(tokens_positive).float(),         # token span (132, 2)
                'positive_map': torch.from_numpy(positive_map).float(),               # main object map (132, 256), note: cls token is considered as the 0-th token
                'modify_positive_map': torch.from_numpy(modify_positive_map).float(), # modift(attribute)
                'pron_positive_map': torch.from_numpy(pron_positive_map).float(),     # pron
                'other_entity_map': torch.from_numpy(other_entity_map).float(),       # other(auxi) object
                'auxi_entity_positive_map': torch.from_numpy(auxi_entity_positive_map).float(), 
                'rel_positive_map': torch.from_numpy(rel_positive_map).float(),       # relation
            }
            lang_utterance = ' '.join(anno['utterance'].replace(',', ' ,').split()) + ' . not mentioned'

            # sample gt mask
            point_ref_mask = np.zeros(instance_label.shape[0])

            context_label = set()
            for word in lang_utterance.split():
                if word in self.type2class.keys() and word != 'others':
                    context_label.add(self.type2class[word])
            point_context_mask = np.zeros(instance_label.shape[0]) - 1
            for i_instance in np.unique(instance_label):            
                # find all points belong to that instance
                ind = np.where(instance_label == i_instance)[0]
                # find the semantic label            
                if int(semantic_label[ind[0]])>=0:
                    nyu_id = int(self.sem2nyu[int(semantic_label[ind[0]])])
                    if nyu_id in self.nyu40ids and self.nyu40id2class[nyu_id] in context_label:
                        point_context_mask[ind] = 1
            # assert len(context_label)==0 or point_context_mask.max()>0, 'no context points'
            point_ref_mask[point_context_mask > 0] = 0.5
            for object_id in object_ids:
                point_ref_mask[instance_label == object_id] = 1
            
            sp_ref_mask = torch_scatter.scatter_mean(torch.from_numpy(point_ref_mask).float(), superpoint, dim=-1)
            gt_pmask, gt_spmask = self.get_ref_mask(coord_float, instance_label, superpoint, object_ids)

            nsubj_ind = torch.tensor(0, dtype=torch.long)
            nsubj_name = None
            
            lang_utterances.append(lang_utterance)
            nsubj_inds.append(nsubj_ind)
            nsubj_names.append(nsubj_name)
            ann_ids.append(ann_id)
            object_idss.append(object_ids)
            gt_pmasks.append(gt_pmask)
            gt_spmasks.append(gt_spmask)
            sp_ref_masks.append(sp_ref_mask)

            meta_datas.append(meta_data)
            view_dependents.append(view_dependent)
            dense_maps.append(dense_map)

        return ann_ids, scan_id, coord, coord_float, feat, superpoint, object_idss, gt_pmasks, gt_spmasks, sp_ref_masks, lang_utterances, sp_ins_label, sem2ins, sem_ids, nsubj_inds, nsubj_names, meta_datas, dense_maps, view_dependents
    
    def collate_fn(self, batch: Sequence[Sequence]) -> Dict:
        ann_ids, scan_ids, coords, coords_float, feats, superpoints, object_idss, gt_pmasks, gt_spmasks, sp_ref_masks, lang_tokenss, lang_masks, lang_utterances, sp_ins_labels, sem2inss, sem_ids, meta_datas, dense_mapss = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
        nsubj_inds, nsubj_names = [], []
        batch_offsets = [0]
        scenes_len = []
        superpoint_bias = 0
        view_dependents = []
        for i, data in enumerate(batch):

            ann_id, scan_id, coord, coord_float, feat, src_superpoint, object_ids, gt_pmask, gt_spmask, sp_ref_mask, lang_utterance, sp_ins_label, sem2ins, sem_id, nsubj_ind, nsubj_name, meta_data, dense_maps, view_dependent = data
    
            superpoint = src_superpoint + superpoint_bias
            superpoint_bias = superpoint.max().item() + 1
            scenes_len.append(len(ann_id))
            batch_offsets.append(superpoint_bias)

            nsubj_names.extend(nsubj_name)
            nsubj_inds.extend(nsubj_ind)

            ann_ids.extend(ann_id)
            scan_ids.append(scan_id)
            coords.append(torch.cat([torch.LongTensor(coord.shape[0], 1).fill_(i), coord], 1))
            coords_float.append(coord_float)
            feats.append(feat)
            superpoints.append(superpoint)
            sp_ins_labels.append(sp_ins_label)
            sem2inss.append(sem2ins)
            
            object_idss.extend(object_ids)
            sem_ids.extend(sem_id)
            
            gt_pmasks.extend(gt_pmask)
            gt_spmasks.extend(gt_spmask)
            sp_ref_masks.extend(sp_ref_mask)
            
            meta_datas.extend(meta_data)
            view_dependents.extend(view_dependent)
            
            lang_utterances.extend(lang_utterance)
            dense_mapss.extend(dense_maps)


        token_dict = self.tokenizer.batch_encode_plus(
            lang_utterances, padding="longest", return_tensors="pt"
        )
        lang_tokenss = token_dict['input_ids']
        lang_masks = token_dict['attention_mask']

        nsubj_inds = torch.stack(nsubj_inds)

        batch_offsets = torch.tensor(batch_offsets, dtype=torch.int)  # int [B+1]
        scenes_len = torch.tensor(scenes_len, dtype=torch.int) #int [B]
        coords = torch.cat(coords, 0)  # long [B*N, 1 + 3], the batch item idx is put in b_xyz[:, 0]
        coords_float = torch.cat(coords_float, 0)  # float [B*N, 3]
        feats = torch.cat(feats, 0)  # float [B*N, 3]
        superpoints = torch.cat(superpoints, 0).long()  # long [B*N, ]
        if self.use_xyz:
            feats = torch.cat((feats, coords_float), dim=1)
        # voxelize
        spatial_shape = np.clip((coords.max(0)[0][1:] + 1).numpy(), self.voxel_cfg.spatial_shape[0], None)  # long [3]
        voxel_coords, p2v_map, v2p_map = pointgroup_ops.voxelization_idx(coords, len(batch), self.mode)

        if not self.cl:
            sem2inss, sem_ids = None, None

        return {
            'ann_ids': ann_ids,
            'scan_ids': scan_ids,
            'voxel_coords': voxel_coords,
            'p2v_map': p2v_map,
            'v2p_map': v2p_map,
            'spatial_shape': spatial_shape,
            'feats': feats,
            'superpoints': superpoints,
            'batch_offsets': batch_offsets,
            'object_idss': object_idss,
            'gt_pmasks': gt_pmasks,
            'gt_spmasks': gt_spmasks,
            'sp_ref_masks': sp_ref_masks,
            'lang_tokenss': lang_tokenss,
            'lang_masks': lang_masks,
            'coords_float': coords_float,
            'sp_ins_labels': sp_ins_labels,
            'sem2inss': sem2inss,
            'sem_ids': sem_ids,
            'scenes_len': scenes_len,
            'nsubj_inds': nsubj_inds,
            'meta_datas': meta_datas,
            'dense_maps': dense_mapss,
            'view_dependents':view_dependents,
        }

    #################################################
    # BRIEF Get text position label by text parsing #
    #################################################
    def _get_token_positive_map_by_parse(self, anno, auxi_box):
        caption = ' '.join(anno['utterance'].replace(',', ' ,').split())

        # node and edge
        graph_node = anno["graph_node"]
        graph_edge = anno["graph_edge"]

        # step main/modify(attri)/pron/other(auxi)/rel
        target_char_span = np.zeros((MAX_NUM_OBJ, 2))   # target(main)
        modify_char_span = np.zeros((MAX_NUM_OBJ, 2))   # modify(attri)
        pron_char_span = np.zeros((MAX_NUM_OBJ, 2))     # pron
        rel_char_span = np.zeros((MAX_NUM_OBJ, 2))      # rel
        assert graph_node[0]['node_id'] == 0
        main_entity_target = graph_node[0]['target_char_span']
        main_entity_modify = graph_node[0]['mod_char_span']
        main_entity_pron   = graph_node[0]['pron_char_span']
        main_entity_rel   = graph_node[0]['rel_char_span']

        # other(auxi) object token
        other_target_char_span = np.zeros((MAX_NUM_OBJ, 2))
        other_entity_target = []
        if len(graph_node) > 1:
            for node in graph_node:
                if node["node_id"] != 0 and node["node_type"] == "Object":
                    for span in node['target_char_span']:
                        other_entity_target.append(span)

        num_t = 0
        num_m = 0
        num_p = 0
        num_o = 0
        num_r = 0
        # target(main obj.) token
        for t, target in enumerate(main_entity_target):
            target_char_span[t] = target
            num_t = t+1
        # modify(attribute) token
        for m, modify in enumerate(main_entity_modify):
            modify_char_span[m] = modify
            num_m = m+1
        # pron token
        for p, pron in enumerate(main_entity_pron):
            pron_char_span[p] = pron
            num_p = p+1
        for o, other in enumerate(other_entity_target):
            other_target_char_span[o] = other
            num_o = o+1
        # rel token add 0727
        for r, rel in enumerate(main_entity_rel):
            rel_char_span[r] = rel
            num_r = r+1

        tokenized = self.tokenizer.batch_encode_plus(
            [' '.join(anno['utterance'].replace(',', ' ,').split())],
            padding="longest", return_tensors="pt"
        )

        target_positive_map = np.zeros((MAX_NUM_OBJ, 256))
        modify_positive_map = np.zeros((MAX_NUM_OBJ, 256))
        pron_positive_map = np.zeros((MAX_NUM_OBJ, 256))
        other_entity_positive_map = np.zeros((MAX_NUM_OBJ, 256))
        rel_positive_map = np.zeros((MAX_NUM_OBJ, 256))
        gt_map_t = get_positive_map(tokenized, target_char_span[:num_t])
        gt_map_m = get_positive_map(tokenized, modify_char_span[:num_m])
        gt_map_p = get_positive_map(tokenized, pron_char_span[:num_p])
        gt_map_o = get_positive_map(tokenized, other_target_char_span[:num_o])
        gt_map_r = get_positive_map(tokenized, rel_char_span[:num_r])
        
        gt_map_t = gt_map_t.sum(axis=0)
        gt_map_m = gt_map_m.sum(axis=0)
        gt_map_p = gt_map_p.sum(axis=0)
        gt_map_o = gt_map_o.sum(axis=0)
        gt_map_r = gt_map_r.sum(axis=0)

        # NOTE text position label
        target_positive_map[:1] = gt_map_t          # main object
        modify_positive_map[:1] = gt_map_m          # attribute
        pron_positive_map[:1]   = gt_map_p          # pron
        other_entity_positive_map[:1] = gt_map_o    # auxi obj
        rel_positive_map[:1]   = gt_map_r           # relation

        # auxi
        auxi_entity_positive_map = np.zeros((MAX_NUM_OBJ, 256))
        if auxi_box is not None:
            auxi_entity = anno["auxi_entity"]['target_char_span']
            num_a = 0
            # char span
            auxi_entity_char_span = np.zeros((MAX_NUM_OBJ, 2))
            for a, auxi in enumerate(auxi_entity):
                auxi_entity_char_span[a] = auxi
                num_a = a+1
            # position label
            gt_map_a = get_positive_map(tokenized, auxi_entity_char_span[:num_a])
            gt_map_a = gt_map_a.sum(axis=0)
            auxi_entity_positive_map[:1] = gt_map_a

            # note SR3D 
            if anno['dataset'] == 'sr3d':
                target_positive_map[1] = gt_map_a

        return target_char_span, target_positive_map, modify_positive_map, pron_positive_map, \
            other_entity_positive_map, auxi_entity_positive_map, rel_positive_map




# BRIEF Construct position label(map)
def get_positive_map(tokenized, tokens_positive):
    positive_map = torch.zeros((len(tokens_positive), 256), dtype=torch.float)  # ([positive], 256])
    for j, tok_list in enumerate(tokens_positive):
        (beg, end) = tok_list
        beg = int(beg)
        end = int(end)
        beg_pos = tokenized.char_to_token(beg)
        end_pos = tokenized.char_to_token(end - 1)
        if beg_pos is None:
            try:
                beg_pos = tokenized.char_to_token(beg + 1)
                if beg_pos is None:
                    beg_pos = tokenized.char_to_token(beg + 2)
            except:
                beg_pos = None
        if end_pos is None:
            try:
                end_pos = tokenized.char_to_token(end - 2)
                if end_pos is None:
                    end_pos = tokenized.char_to_token(end - 3)
            except:
                end_pos = None
        if beg_pos is None or end_pos is None:
            continue
        positive_map[j, beg_pos:end_pos + 1].fill_(1)

    positive_map = positive_map / (positive_map.sum(-1)[:, None] + 1e-12)
    return positive_map.numpy()




#########################
# BRIEF Text decoupling #
#########################
def Scene_graph_parse(caption):
    caption = ' '.join(caption.replace(',', ' , ').split())

    # some error or typo in ScanRefer.
    caption = ' '.join(caption.replace("'m", "am").split())
    caption = ' '.join(caption.replace("'s", "is").split())
    caption = ' '.join(caption.replace("2-tiered", "2 - tiered").split())
    caption = ' '.join(caption.replace("4-drawers", "4 - drawers").split())
    caption = ' '.join(caption.replace("5-drawer", "5 - drawer").split())
    caption = ' '.join(caption.replace("8-hole", "8 - hole").split())
    caption = ' '.join(caption.replace("7-shaped", "7 - shaped").split())
    caption = ' '.join(caption.replace("2-door", "2 - door").split())
    caption = ' '.join(caption.replace("3-compartment", "3 - compartment").split())
    caption = ' '.join(caption.replace("computer/", "computer /").split())
    caption = ' '.join(caption.replace("3-tier", "3 - tier").split())
    caption = ' '.join(caption.replace("3-seater", "3 - seater").split())
    caption = ' '.join(caption.replace("4-seat", "4 - seat").split())
    caption = ' '.join(caption.replace("theses", "these").split())


    # nr3d = True
    # some error or typo in NR3D.
    # if nr3d:
    caption = ' '.join(caption.replace('.', ' .').split())
    caption = ' '.join(caption.replace(';', ' ; ').split())
    caption = ' '.join(caption.replace('-', ' ').split())
    caption = ' '.join(caption.replace('"', ' ').split())
    caption = ' '.join(caption.replace('?', ' ').split())
    caption = ' '.join(caption.replace("*", " ").split())
    caption = ' '.join(caption.replace(':', ' ').split())
    caption = ' '.join(caption.replace('$', ' ').split())
    caption = ' '.join(caption.replace("#", " ").split())
    caption = ' '.join(caption.replace("/", " / ").split())
    caption = ' '.join(caption.replace("you're", "you are").split())
    caption = ' '.join(caption.replace("isn't", "is not").split())
    caption = ' '.join(caption.replace("thats", "that is").split())
    caption = ' '.join(caption.replace("theres", "there is").split())
    caption = ' '.join(caption.replace("doesn't", "does not").split())
    caption = ' '.join(caption.replace("doesnt", "does not").split())
    caption = ' '.join(caption.replace("itis", "it is").split())
    caption = ' '.join(caption.replace("left-hand", "left - hand").split())
    caption = ' '.join(caption.replace("[", " [ ").split())
    caption = ' '.join(caption.replace("]", " ] ").split())
    caption = ' '.join(caption.replace("(", " ( ").split())
    caption = ' '.join(caption.replace(")", " ) ").split())
    caption = ' '.join(caption.replace("wheel-chair", "wheel - chair").split())
    caption = ' '.join(caption.replace(";s", "is").split())
    caption = ' '.join(caption.replace("tha=e", "the").split())
    caption = ' '.join(caption.replace("it’s", "it is").split())
    caption = ' '.join(caption.replace("’s", " is").split())
    caption = ' '.join(caption.replace("isnt", "is not").split())
    caption = ' '.join(caption.replace("Don't", "Do not").split())
    caption = ' '.join(caption.replace("arent", "are not").split())
    caption = ' '.join(caption.replace("cant", "can not").split())
    caption = ' '.join(caption.replace("you’re", "you are").split())
    caption = ' '.join(caption.replace('!', ' !').split())
    caption = ' '.join(caption.replace('id the', ' , the').split())
    caption = ' '.join(caption.replace('youre', 'you are').split())

    caption = ' '.join(caption.replace("'", ' ').split())
    caption = ' '.join(caption.replace("``", ' ').split())

    if caption[0] == "'":
        caption = caption[1:]
    if caption[-1] == "'":
        caption = caption[:-1]

    # text parsing
    graph_node, graph_edge = sng_parser.parse(caption)

    # NOTE If no node is parsed, add "this is an object ." at the beginning of the sentence
    if (len(graph_node) < 1) or \
        (len(graph_node) > 0 and graph_node[0]["node_id"] != 0):
        caption = "This is an object . " + caption
        # parse again
        graph_node, graph_edge = sng_parser.parse(caption)


    # auxi object
    auxi_entity = None
    for node in graph_node:
        if (node["node_id"] != 0) and (node["node_type"] == "Object"):
            auxi_entity = node
            break
    
    return {
        "graph_node": graph_node,
        "graph_edge": graph_edge,
        "auxi_entity": auxi_entity,
        "utterance": caption
    }
