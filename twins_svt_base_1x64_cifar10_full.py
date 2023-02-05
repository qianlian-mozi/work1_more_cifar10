model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='SVT',
        arch='base',
        in_channels=3,
        out_indices=(3, ),
        qkv_bias=True,
        norm_cfg=dict(type='LN'),
        norm_after_stage=[False, False, False, True],
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.3),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=10,
        in_channels=768,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        cal_acc=False,
        topk=(1, )),
    init_cfg=[
        dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.0),
        dict(type='Constant', layer='LayerNorm', val=1.0, bias=0.0)
    ],
    train_cfg=dict(augments=[
        dict(type='BatchMixup', alpha=0.8, num_classes=10, prob=0.5),
        dict(type='BatchCutMix', alpha=1.0, num_classes=10, prob=0.5)
    ]))
dataset_type = 'CIFAR10'
img_norm_cfg = dict(
    mean=[125.307, 122.961, 113.8575],
    std=[51.5865, 50.847, 51.255],
    to_rgb=False)
train_pipeline = [
    dict(type='RandomCrop', size=32, padding=4),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(
        type='Normalize',
        mean=[125.307, 122.961, 113.8575],
        std=[51.5865, 50.847, 51.255],
        to_rgb=False),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(
        type='Normalize',
        mean=[125.307, 122.961, 113.8575],
        std=[51.5865, 50.847, 51.255],
        to_rgb=False),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=128,
    workers_per_gpu=4,
    train=dict(
        type='CIFAR10',
        data_prefix='data/cifar10',
        pipeline=[
            dict(type='RandomCrop', size=32, padding=4),
            dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
            dict(
                type='Normalize',
                mean=[125.307, 122.961, 113.8575],
                std=[51.5865, 50.847, 51.255],
                to_rgb=False),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='ToTensor', keys=['gt_label']),
            dict(type='Collect', keys=['img', 'gt_label'])
        ]),
    val=dict(
        type='CIFAR10',
        data_prefix='data/cifar10',
        pipeline=[
            dict(
                type='Normalize',
                mean=[125.307, 122.961, 113.8575],
                std=[51.5865, 50.847, 51.255],
                to_rgb=False),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ],
        test_mode=True),
    test=dict(
        type='CIFAR10',
        data_prefix='data/cifar10',
        pipeline=[
            dict(
                type='Normalize',
                mean=[125.307, 122.961, 113.8575],
                std=[51.5865, 50.847, 51.255],
                to_rgb=False),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ],
        test_mode=True))
optimizer = dict(
    type='AdamW',
    lr=0.000125,
    weight_decay=0.05,
    eps=1e-08,
    betas=(0.9, 0.999),
    paramwise_cfg=dict(_delete=True, norm_decay_mult=0.0, bias_decay_mult=0.0))
optimizer_config = dict(grad_clip=dict(max_norm=5.0))
runner = dict(type='EpochBasedRunner', max_epochs=20)
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=100,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'pretrained/twins-svt-base_3rdparty_8xb128_in1k_20220126-e31cc8e9.pth'
resume_from = None
workflow = [('train', 1)]
paramwise_cfg = dict(_delete=True, norm_decay_mult=0.0, bias_decay_mult=0.0)
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=True,
    min_lr_ratio=0.001,
    warmup='linear',
    warmup_ratio=0.0001,
    warmup_iters=5,
    warmup_by_epoch=True)
evaluation = dict(interval=1, metric='accuracy')
work_dir = 'work/work1_more_twins_1xb64_cifar10'
gpu_ids = [0]
