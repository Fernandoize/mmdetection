img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

pipeline_base = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True)
]
train_pipeline_with_mask=[
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundleCustom'),
    dict(type='Collect', keys=['img', 'img_masked', 'img_dfui', 'gt_bboxes', 'gt_labels'])
]

test_pipeline = [
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img', 'img_masked']),
            dict(type='Collect', keys=['img', 'img_masked']),
        ])
]


dataset_type = 'CocoDataset'
# img_prefix
img_prefix_dfui='/Users/wangfengguo/LocalTools/data/dfui/images/'
img_prefix_urpc_train='/Users/wangfengguo/LocalTools/data/urpc2020/images/'
img_prefix_urpc_test = img_prefix_urpc_train

# ann
ann_train_dfui='/Users/wangfengguo/LocalTools/data/dfui/annotations/instances_trainval2017.json'
ann_train_urpc='/Users/wangfengguo/LocalTools/data/urpc2020/annotations/instances_train.json'
ann_test_urpc='/Users/wangfengguo/LocalTools/data/urpc2020/annotations/instances_test.json'

dataset_train=dict(
    type='CocoDatasetCustom',
    dataset=dict(
        type=dataset_type,
        ann_file=ann_train_urpc,
        img_prefix=img_prefix_urpc_train, # UI
        pipeline=pipeline_base
    ),
    img_dfui_prefix=img_prefix_dfui, # DFUI
    mask_thr=0.5,
    pipeline=train_pipeline_with_mask
)
dataset_test=dict(
    type='CocoDatasetCustom',
    dataset=dict(
        type=dataset_type,
        ann_file=ann_test_urpc,
        img_prefix=img_prefix_urpc_test,
        pipeline=[dict(type='LoadImageFromFile')]
    ),
    mask_thr=0.5,
    pipeline=test_pipeline
)

train_dataloader = dict(
    batch_size=8,
    num_workers=0,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dataset_train)

val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dataset_test)

test_dataloader = val_dataloader
val_evaluator = dict(
    type='CocoMetric',
    ann_file=ann_train_urpc,
    metric='bbox',
    format_only=False,
    backend_args=None)
test_evaluator = val_evaluator
# data = dict(
#     samples_per_gpu=2,
#     workers_per_gpu=0,
#     train=dataset_train,
#     val=dataset_test,
#     test=dataset_test)
# evaluation = dict(interval=1, metric='bbox', classwise=True)