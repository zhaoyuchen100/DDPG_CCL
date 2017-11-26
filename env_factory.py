# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from gym import envs

import threelinkarm


def register_env(env_id):
    if env_id == 'Maze2D-v0':
        envs.register(
            id=env_id,
            entry_point='env:maze2d.Maze2D',
            kwargs={},
            max_episode_steps=200,
            reward_threshold=-110.0)
    elif env_id == '3linkarm-v0':
        envs.register(
            id=env_id,
            entry_point='threelinkarm:ThreelinkArm',
            kwargs={},
            max_episode_steps=200,
            reward_threshold=-100.0)
    else:
        raise ValueError('Cannot find environment "{0}"\n'.format(env_id))
    return True
