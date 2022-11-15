import os

env = os.environ

nperiods = 3

init = env['INITIALIZATION'] if 'INITIALIZATION' in env else 'Pretrained'
if init == 'Xavier':
    initCfg=dict(type='Xavier', distribution='uniform')
elif init == 'Kaiming':
    initCfg=dict(type='Kaiming', distribution='uniform')
else:
    initCfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')

print('ENVIRONMENT:', env)
stepPeriod = int(float(env['STEP_PERIOD']))
# restartDecay = int(float(env['RESTART_DECAY']))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=int(float(env['WARMUP_ITERS'])),
    warmup_ratio=float(env['WARMUP_RATIO']),
    warmup_by_epoch = False,
    step=[i*stepPeriod for i in range(2,nperiods+2)],
)

epochs = int(float(env['EPOCHS']))
lr = float(env['LEARNING_RATE'])
decay = float(env['DECAY'])
momentum = float(env['MOMENTUM'])

trial = env['TRIAL']