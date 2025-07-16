# from rlpyt.agents.pg.ppo import PpoLstmAgent
from rlpyt.agents.pg.gaussian import RecurrentGaussianPgAgent, GaussianPgAgent
# from rlpyt.models.pg.mujoco_lstm_model import MujocoLstmModel
from rlpyt.models.pg.mujoco_ff_model import MujocoFfModel
import numpy as np
class RL2PpoAgent(GaussianPgAgent):
    def __init__(self, ModelCls=MujocoFfModel, model_kwargs=None, **kwargs):
        super().__init__(ModelCls=ModelCls, model_kwargs=model_kwargs, **kwargs)

    def make_env_to_model_kwargs(self, env_spaces):
        return dict(
            observation_shape=env_spaces.observation.shape,
            action_size=env_spaces.action.shape[0],
            hidden_sizes=[256],  # RNN隐藏层大小
            # ✅ 添加类型转换
            # obs_std=np.ones(env_spaces.observation.shape, dtype=np.float32),
            # obs_mean=np.zeros(env_spaces.observation.shape, dtype=np.float32)
        )