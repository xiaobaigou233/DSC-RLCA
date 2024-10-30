#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import parl
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import math
# clamp bounds for Std of action_log
LOG_SIG_MAX = 2.0
LOG_SIG_MIN = -20.0


class Model(parl.Model):
    def __init__(self, obs_dim, frames ,action_dim):
        super(Model, self).__init__()
        self.actor_model = Actor(obs_dim, frames ,action_dim)
        self.critic_model = Critic(obs_dim, frames ,action_dim)


    def policy(self, obs):
        return self.actor_model(obs)

    def value(self, obs, action):
        return self.critic_model(obs, action)

    def get_actor_params(self):
        return self.actor_model.parameters()

    def get_critic_params(self):
        return self.critic_model.parameters()


class Actor(parl.Model):
    def __init__(self, obs_dim, frames ,action_dim):
        super(Actor, self).__init__()
        self.act_fea_cv1 = nn.Conv1D(in_channels=frames, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.act_fea_cv2 = nn.Conv1D(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.l1 = nn.Linear(32*math.floor((obs_dim-3)/4 +1), 256)
        self.l2 = nn.Linear(256, 256)
        self.mean_linear = nn.Linear(256, action_dim)
        self.std_linear = nn.Linear(256, action_dim)

    def forward(self, obs):
        x = F.relu(self.act_fea_cv1(obs))
        x = F.relu(self.act_fea_cv2(x))
        x = paddle.reshape(x,[x.shape[0], -1])
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))

        act_mean = self.mean_linear(x)
        act_std = self.std_linear(x)
        act_log_std = paddle.clip(act_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return act_mean, act_log_std


class Critic(parl.Model):
    def __init__(self, obs_dim, frames ,action_dim):
        super(Critic, self).__init__()

        # Q1 network
        self.act_fea_cv1 = nn.Conv1D(in_channels=frames, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.act_fea_cv2 = nn.Conv1D(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.l1 = nn.Linear(32*math.floor((obs_dim-3)/4 +1) + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 network
        self.act_fea_cv1 = nn.Conv1D(in_channels=frames, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.act_fea_cv2 = nn.Conv1D(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.l4 = nn.Linear(32*math.floor((obs_dim-3)/4 +1) + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, obs, action):
        x = F.relu(self.act_fea_cv1(obs))
        x = F.relu(self.act_fea_cv2(x))
        x = paddle.reshape(x,[x.shape[0], -1])
        x = paddle.concat([x, action], 1)

        # Q1
        q1 = F.relu(self.l1(x))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        # Q2
        q2 = F.relu(self.l4(x))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2
