default_scope = 'mmdet'

default_hooks = dict(
    # 随着Iter变化，更新epoch花费时间
    timer=dict(type='IterTimerHook'),
    # 日志间隔，包括terminal, tensorboard, wandb
    logger=dict(type='LoggerHook', interval=50),
    # 超参更新
    param_scheduler=dict(type='ParamSchedulerHook'),
    # 保存checkpoint
    checkpoint=dict(type='CheckpointHook', interval=1),
    # 允许分布式sampler
    sampler_seed=dict(type='DistSamplerSeedHook'),
    # 验证和测试结果可视化
    visualization=dict(type='DetVisualizationHook'))

custom_hooks = []

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

log_level = 'INFO'
load_from = None
resume = False
