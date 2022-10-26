import os
import sys

env = os.environ
depth = int(env['BACKBONE'][len('resnet'):]) # resnet## -> ##
if depth not in [18, 34, 50]:
    print(f'Backbone depth must be in [18, 34, 50], not {depth}')
    sys.exit(1)
init = env['INITIALIZATION']

cosperiod = [i*int(env['COSINE_PERIOD']) for i in range(1,6)]
cosrestart = dict(policy='CosineRestart', min_lr=0., periods=cosperiod, restart_weights=[1]*5)
steps = [i*int(env['STEP_PERIOD']) for i in range(1,6)]
steplr = dict(policy='step', step=steps)

policy = env['SCHEDULER']
schedule = cosrestart if policy == 'CosineRestart' else steplr

epochs = int(env['EPOCHS'])
lr = float(env['LEARNING_RATE'])
decay = float(env['DECAY'])
momentum = float(env['MOMENTUM'])

trial = env['TRIAL']