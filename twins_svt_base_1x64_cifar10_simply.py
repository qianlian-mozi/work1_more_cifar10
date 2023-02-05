_base_ = [
    '../_base_/models/twins_svt_base.py',
    '../_base_/datasets/cifar10_bs16.py',
    '../_base_/schedules/cifar10_bs128.py',
    '../_base_/default_runtime.py'
]

data = dict(samples_per_gpu=128,
            workers_per_gpu=4)

model = dict(head=dict(
        num_classes=10,
        topk = (1,)
        ),
        train_cfg=dict(augments=[
        dict(type='BatchMixup', alpha=0.8, num_classes=10, prob=0.5),
        dict(type='BatchCutMix', alpha=1.0, num_classes=10, prob=0.5)
    ]))

paramwise_cfg = dict(_delete=True, norm_decay_mult=0.0, bias_decay_mult=0.0)

# for batch in each gpu is 128, 1 gpu
# lr = 5e-4 * 64 * 1 / 512 
optimizer = dict(
    type='AdamW',
    lr=5e-4 * 128 * 1 / 512,
    weight_decay=0.05,
    eps=1e-8,
    betas=(0.9, 0.999),
    paramwise_cfg=paramwise_cfg)
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=5.0))

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=True,
    min_lr_ratio=1e-3,
    warmup='linear',
    warmup_ratio=1e-4,
    warmup_iters=5,
    warmup_by_epoch=True)

evaluation = dict(interval=1, metric='accuracy')

log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook' )
        # dict(type='WandbLoggerHook', init_kwargs=dict(project='Your-project'))
    ])


load_from = 'pretrained/twins-svt-base_3rdparty_8xb128_in1k_20220126-e31cc8e9.pth'

runner = dict(type='EpochBasedRunner', max_epochs=20)
