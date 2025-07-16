from rlpyt.envs.base import EnvStep
from rlpyt.spaces.int_box import IntBox
import metaworld
import gym
import numpy as np
from collections import namedtuple
from rlpyt.utils.collections import namedarraytuple
from rlpyt.spaces.gym_wrapper import GymSpaceWrapper

# 定义环境信息结构
MetaWorldEnvInfo = namedarraytuple("MetaWorldEnvInfo", ["success"])

# 定义 rlpyt 需要的空间结构
EnvSpaces = namedtuple("EnvSpaces", ["observation", "action"])

class RunningNormalizer:
    """在线奖励归一化器 (Z-score)"""
    def __init__(self, alpha=0.95):
        self.mean = 0.0
        self.var = 1.0
        self.std = 1.0
        self.alpha = alpha  # 指数平滑系数
        self.count = 0
    
    def update(self, x):
        self.count += 1
        # 更新均值和方差（指数移动平均）
        self.mean = self.alpha * self.mean + (1 - self.alpha) * x
        self.var = self.alpha * self.var + (1 - self.alpha) * (x - self.mean)**2
        self.std = max(np.sqrt(self.var), 1e-6)  # 防止除零
    
    def normalize(self, x):
        return (x - self.mean) / self.std
    
class MetaWorldRL2Env:
    def __init__(self, task_names=["pick-place-v3", "door-open-v3"], act_null_value=0, obs_null_value=0, force_float32=True):
        self.task_names = task_names
        self.current_task = None
        self.env = None
        ml = metaworld.ML1(self.task_names[0])
        self.env = ml.train_classes[self.task_names[0]]()
        self.tasks = ml.train_tasks
        task = self.tasks[0]
        self.env.set_task(task)

        self.normalizers = { name: RunningNormalizer() for name in self.task_names}
        self._warm_up_normalizer(100)
        # self.observation_space = self.env.observation_space #gym.spaces.Box(low=-np.inf, high=np.inf, shape=(39,))  # MetaWorld 观测维度:cite[3]
        # self.action_space = self.env.action_space#gym.spaces.Box(low=-1, high=1, shape=(4,))  # Sawyer机械臂动作空间
        self.action_space = GymSpaceWrapper(
            space=self.env.action_space,
            name="act",
            null_value=act_null_value,
            force_float32=force_float32,
        )
        self.observation_space = GymSpaceWrapper(
            space=self.env.observation_space,
            name="obs",
            null_value=obs_null_value,
            force_float32=force_float32,
        )
        # ✅ 添加 rlpyt 必需的 spaces 属性
        self.spaces = EnvSpaces(
            observation=self.observation_space,
            action=self.action_space
        )
    
    def _warm_up_normalizer(self, num_steps):
        """用随机动作预热归一化器"""
        for task_name in self.task_names:
            ml = metaworld.ML1(task_name)
            self.env = ml.train_classes[task_name]()
            task = ml.train_tasks[0]
            self.env.set_task(task)
            self.env.reset()
            for _ in range(num_steps):
                action = self.env.action_space.sample()
                _, reward, done, _ = self.env.step(action)
                self.normalizers[task_name].update(reward)
                if done:
                    self.env.reset()
    

    def reset(self):
        task_name = np.random.choice(self.task_names)
        self.current_task_name = task_name
        ml = metaworld.ML1(task_name)
        self.env = ml.train_classes[task_name]()
        self.tasks = ml.train_tasks
        idx = np.random.random_integers(0, len(self.tasks)-1)
        # task = np.random.choice(ml.train_tasks)
        task = self.tasks[idx]
        self.env.set_task(task)
        obs, info = self.env.reset()
        obs = obs.astype(np.float32)
        # print("reset obs:", obs)
        # env_info = MetaWorldEnvInfo(
        #         success=False,
        #     )
        return obs
    def close(self):
        """Any clean up operation."""
        pass

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # print("step obs:", obs)
        # 归一化奖励
        normalized_reward = self.normalizers[self.current_task_name].normalize(reward)
        self.normalizers[self.current_task_name].update(reward)  # 更新归一化器统计

        done = terminated or truncated
        obs = obs.astype(np.float32)
        env_info = MetaWorldEnvInfo(
            success=info.get('success', False),
        )
        return EnvStep(obs, normalized_reward, done, env_info)  # 适配rlpyt的EnvStep格式