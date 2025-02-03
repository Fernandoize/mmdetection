import os.path

_base_ = './faster-rcnn_r50_fpn_1x_coco.py'


dataset_type = 'CocoDataset'
classes = ('holothurian', 'scallop', 'echinus', 'starfish', 'waterweeds')
data_root='/Users/wangfengguo/LocalTools/data/dfui/'

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        # 将类别名字添加至 `metainfo` 字段中
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file=os.path.join(data_root, 'annotations/instances_train2017.json'),
        data_prefix=dict(img='images')
        )
    )

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        test_mode=True,
        # 将类别名字添加至 `metainfo` 字段中
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file=os.path.join(data_root, 'annotations/instances_val2017.json'),
        data_prefix=dict(img='images')
    ))

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        test_mode=True,
        # 将类别名字添加至 `metainfo` 字段中
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file=os.path.join(data_root, 'annotations/instances_testl2017.json'),
        data_prefix=dict(img='images')
    ))


# 验证集指标
val_evaluator = dict(
    # 包含AR、AP、mAP
    type='CocoMetric',
    ann_file=os.path.join(data_root, 'annotations/instances_val2017.json'),
    # bbox用于目标检测，segm用于实例分割
    metric='bbox',
    format_only=False,
    backend_args=None)

test_evaluator = dict(
    # 包含AR、AP、mAP
    type='CocoMetric',
    ann_file=os.path.join(data_root, 'annotations/instances_test2017.json'),
    # bbox用于目标检测，segm用于实例分割
    metric='bbox',
    format_only=False,
    backend_args=None)

model = dict(
    roi_head=dict(
        bbox_head=
            dict(
                type='Shared2FCBBoxHead',
                # 将所有的 `num_classes` 默认值修改为 5（原来为 80）
                num_classes=5)))