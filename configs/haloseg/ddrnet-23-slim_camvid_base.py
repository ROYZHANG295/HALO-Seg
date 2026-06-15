norm_cfg = dict(type='BN', requires_grad=True, momentum=0.03, eps=0.001)
crop_size = (960, 720)

# Optional ImageNet-1K pretrained backbone checkpoint.
# checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/ddrnet/pretrain/ddrnet23s-in1kpre_3rdparty-1ccac5b1.pth'

data_preprocessor = dict(
    type='SegDataPreProcessor',
    size_divisor=64,  # Required for DDRNet: pad CamVid height 720 to 768.
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='DDRNet',
        in_channels=3,
        channels=32,
        ppm_channels=128,
        norm_cfg=norm_cfg,
        align_corners=False,
    ),
    decode_head=dict(
        type='DDRHead',
        in_channels=128,
        channels=64,
        dropout_ratio=0.0,
        num_classes=11,
        align_corners=False,
        norm_cfg=norm_cfg,
        loss_decode=[
            dict(
                type='OhemCrossEntropy',
                thres=0.9,
                min_kept=131072,
                loss_weight=1.0),
            dict(
                type='OhemCrossEntropy',
                thres=0.9,
                min_kept=131072,
                loss_weight=0.4),
        ]),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]

# TTA keeps absolute sizes, all divisible by 64 for DDRNet.
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[[
            dict(type='Resize', scale=(480, 384),  keep_ratio=False),  
            dict(type='Resize', scale=(640, 512),  keep_ratio=False),  
            dict(type='Resize', scale=(768, 576),  keep_ratio=False),  
            dict(type='Resize', scale=(960, 768),  keep_ratio=False),  
            dict(type='Resize', scale=(1152, 896), keep_ratio=False),  
            dict(type='Resize', scale=(1280, 960), keep_ratio=False),  
        ],
        [
            dict(type='RandomFlip', prob=0.0, direction='horizontal'),
            dict(type='RandomFlip', prob=1.0, direction='horizontal')
        ],
        [dict(type='LoadAnnotations')],
        [dict(type='PackSegInputs')]
        ])
]

train_dataloader = dict(
    batch_size=12,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
            type='CamVid',
            data_root='data/camvid_3/',
            data_prefix=dict(
                img_path='train', seg_map_path='train_labels'),
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations'),
                dict(
                    type='RandomResize',
                    scale=(960, 720),
                    ratio_range=(0.5, 2.0),
                    keep_ratio=True),
                dict(
                    type='RandomCrop', crop_size=(960, 720), cat_max_ratio=0.75),
                dict(type='RandomFlip', prob=0.5),
                dict(type='PhotoMetricDistortion'),
                dict(type='PackSegInputs')
            ])
    )

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CamVid',
        data_root='data/camvid_3/',
        data_prefix=dict(
            img_path='test', seg_map_path='test_labels'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(960, 768), keep_ratio=False),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs')
        ]))

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CamVid',
        data_root='data/camvid_3/',
        data_prefix=dict(
            img_path='test', seg_map_path='test_labels'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(960, 768), keep_ratio=False),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs')
        ]))

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer')
log_processor = dict(by_epoch=False)
log_level = 'INFO'

load_from = 'https://download.openmmlab.com/mmsegmentation/v0.5/ddrnet/ddrnet_23-slim_in1k-pre_2xb6-120k_cityscapes-1024x1024/ddrnet_23-slim_in1k-pre_2xb6-120k_cityscapes-1024x1024_20230426_145312-6a5e5174.pth' 
resume = None
tta_model = dict(type='SegTTAModel')

max_iters = 7800
interval = 780

# Optimizer for CamVid fine-tuning.
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)

param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=0.,
        power=0.9,
        begin=0,
        end=max_iters,
        by_epoch=False)
]

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=max_iters, val_interval=interval)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', by_epoch=False, interval=interval, save_best='mIoU'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

randomness = dict(seed=304)
