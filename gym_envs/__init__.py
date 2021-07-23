from gym.envs.registration import registry, register, make, spec

register(
    id='Cozmo-v0',
    entry_point='gym_envs.cozmo:CozmoEnv',
    kwargs={}
)

register(
    id='Options-v0',
    entry_point='gym_envs.envwithoptions:EnvWithOptions',
    kwargs={}
)

register(
    id='MultiSensors-v0',
    entry_point='gym_envs.multisensors:MultiSensorEnv',
    kwargs={}
)

register(
    id='ChannelsMaster-v0',
    entry_point='gym_envs.channels:ChannelsMasterEnv',
    kwargs={}
)

register(
    id='ChannelsWorker-v0',
    entry_point='gym_envs.channels:ChannelsWorkerEnv',
    kwargs={}
)

register(
    id='Epuck-v0',
    entry_point='gym_envs.epuck_vrep_env:EPuckVrepEnv',
    kwargs={}
)

register(
    id='EpuckRooms-v0',
    entry_point='gym_envs.epuck_vrep_env:EPuckVrepEnv',
    kwargs={'scene': 'epuck_hallway.ttt'}
)

register(
    id='Joystick-v0',
    entry_point='gym_envs.joystick:JoystickEnv',
    reward_threshold=0.0,
    kwargs={'safe': True}
)

register(
    id='JoystickUnsafe-v0',
    entry_point='gym_envs.joystick:JoystickEnv',
    reward_threshold=0.0,
    kwargs={'safe': False}
)

register(
    id='Cybot-v0',
    entry_point='gym_envs.cybot:CybotEnv',
    reward_threshold=0.0,
    kwargs={}
)

register(
    id='Khepera-v0',
    entry_point='gym_envs.khepera:DiscreteKheperaEnv',
    reward_threshold=0.0,
    kwargs={'f': '/dev/rfcomm0'}
)

register(
    id='KheperaContinuous-v0',
    entry_point='gym_envs.khepera:ContinuousKheperaEnv',
    reward_threshold=0.0,
    kwargs={'f': '/dev/rfcomm1'}
)

register(
    id='TreeMaze-v0',
    entry_point='gym_envs.treemaze:TreeMazeEnv',
    kwargs={'size': 5, 'height': 3}
)

register(
    id='DuplicatedInputCond-v0',
    entry_point='gym_envs.duplicatedinputcond:DuplicatedInputCondEnv',
    kwargs={'duplication': 2, 'base': 5}
)

register(
    id='Terminals-v0',
    entry_point='gym_envs.terminals:TerminalsEnv',
    kwargs={}
)

register(
    id='Mario-v0',
    entry_point='gym_envs.mario:MarioEnv',
    kwargs={}
)

register(
    id='Robot-v0',
    entry_point='gym_envs.robot:RobotEnv',
    kwargs={}
)

for i in range(100):
    register(
        id='DeepExp%i-v0' % i,
        entry_point='gym_envs.deepexploration:DeepExplorationEnv',
        kwargs={'size': i}
    )

register(
    id='LargeGrid-v0',
    entry_point='gym_envs.myGrid:myGrid',
    kwargs={'y': 29, 'x': 27}
)

register(
    id='PuzzleRooms-v0',
    entry_point='gym_envs.6pieces:PuzzleRooms',
    kwargs={}
)

register(
    id='RandomMOMDPScalar-v0',
    entry_point='gym_envs.randommomdp:RandomMOMDP',
    reward_threshold=0.0,
    kwargs={'nstates': 100, 'nobjectives': 4, 'nactions': 8, 'nsuccessor': 12, 'seed': 1, 'scalarize': True}
)

register(
    id='TransferA-v0',
    entry_point='gym_envs.varyingGrid:VaryingGrid',
    kwargs={'is_grid_2': False}
)
register(
    id='TransferB-v0',
    entry_point='gym_envs.varyingGrid:VaryingGrid',
    kwargs={'is_grid_2': True}
)

register(
    id='Table-v0',
    entry_point='gym_envs.table:Table',
    kwargs={}
)

for i, goal in enumerate([(0.2, 0.2), (0.2, 0.8), (0.8, 0.2), (0.8, 0.8)]):
    register(
        id='Table%i-v0' % i,
        entry_point='gym_envs.table:Table',
        kwargs={'goal': goal + (3.1415 * 0.25,)}
    )

for a in range(26):
    for b in range(26):
        register(
            id='VaryingDoors-%i-%i-v0' % (a, b),
            entry_point='gym_envs.varyingGrid2:VaryingGrid',
            kwargs={'door1': a, 'door2': b}
        )
        
register(
    id='ShipContainer-v0',
    entry_point='gym_envs.ship_container:ShipContainer',
    kwargs={'x': 6, 'y': 6}
)

register(
    id='ShepherdEnv-v0',
    entry_point='gym_envs.shepherdEnv:ShepherdEnv',
    kwargs={}
)
