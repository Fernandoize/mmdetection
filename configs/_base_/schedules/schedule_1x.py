# training schedule for 1x
# 训练流程控制
# 训练epoch和验证epoch
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# 学习率
param_scheduler = [
    # 线性预热 学习率
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    #
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]

# 优化器
optim_wrapper = dict(
    # AmpOptimWrapper 混合精度训练
    type='OptimWrapper',
    # SGD
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)
