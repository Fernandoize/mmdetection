# dataset settings
# 数据集类型
dataset_type = 'CocoDataset'
# 数据集root path
data_root = 'data/coco/'

# Example to use different file client
# Method 1: simply set the data root and let the file I/O module
# automatically infer from prefix (not support LMDB and Memcache yet)

# data_root = 's3://openmmlab/datasets/detection/coco/'

# Method 2: Use `backend_args`, `file_client_args` in versions before 3.0.0rc6
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/detection/',
#         'data/': 's3://openmmlab/datasets/detection/'
#     }))
backend_args = None
# 训练数据预处理
train_pipeline = [
    # 加载image文件
    dict(type='LoadImageFromFile', backend_args=backend_args),
    # 加载annotations
    dict(type='LoadAnnotations', with_bbox=True),
    # resize图像
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    # 随机翻转，图像增强
    dict(type='RandomFlip', prob=0.5),
    # 数据打包
    dict(type='PackDetInputs')
]

# 测试数据预处理
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

# 训练数据data loader
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    # 单个epoch后仍然保留workers线程，加速训练
    persistent_workers=True,
    # 支持分布式训练
    sampler=dict(type='DefaultSampler', shuffle=True),
    # 将相似比例的图像分到一组，减少显存使用
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    # 训练集配置
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/'),
        # 过滤较小的图片和gt box为空的图像
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))

# 验证数据data loader
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    # 是否丢弃最后一组batch_size不够的图像
    drop_last=False,
    # 在验证或者测试时关闭shuffle
    sampler=dict(type='DefaultSampler', shuffle=False),
    #
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/'),
        # 测试模式不进行数据或者标签的过滤
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))

# 测试数据data loader
test_dataloader = val_dataloader

# 验证集指标
val_evaluator = dict(
    # 包含AR、AP、mAP
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val2017.json',
    # bbox用于目标检测，segm用于实例分割
    metric='bbox',
    format_only=False,
    backend_args=backend_args)
test_evaluator = val_evaluator

# inference on test dataset and
# format the output results for submission.
# test_dataloader = dict(
#     batch_size=1,
#     num_workers=2,
#     persistent_workers=True,
#     drop_last=False,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         ann_file=data_root + 'annotations/image_info_test-dev2017.json',
#         data_prefix=dict(img='test2017/'),
#         test_mode=True,
#         pipeline=test_pipeline))
# test_evaluator = dict(
#     type='CocoMetric',
#     metric='bbox',
#     format_only=True,
#     ann_file=data_root + 'annotations/image_info_test-dev2017.json',
#     outfile_prefix='./work_dirs/coco_detection/test')
