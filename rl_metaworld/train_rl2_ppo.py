from rlpyt.runners.minibatch_rl import MinibatchRlEval
from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.envs.gym import make as make_env
from custom_agents.rl2_ppo_agent import RL2PpoAgent
from custom_envs.metaworld_rl2 import MetaWorldRL2Env
from rlpyt.algos.pg.ppo import PPO
from rlpyt.samplers.serial.collectors import SerialEvalCollector
from rlpyt.utils.logging.context import logger_context
from torch.utils.tensorboard import SummaryWriter
import torch

def build_and_train():
    # sampler = GpuSampler(
    #     EnvCls=MetaWorldRL2Env,
    #     env_kwargs={"task_names": ["pick-place-v3", "push-v3"]},  # 多任务训练
    #     batch_T=512,  # 时间步长（影响RNN序列长度）
    #     batch_B=16,   # 并行环境数
    # )
    writer = SummaryWriter(log_dir='tensorboard_logs/exp1')
    cuda_idx = None
    n_parallel = 2
    affinity = dict(cuda_idx=cuda_idx, workers_cpus=list(range(n_parallel)))
    sampler = SerialSampler(
        EnvCls=MetaWorldRL2Env,
        env_kwargs={"task_names": ["pick-place-v3"]},  # 多任务训练
        eval_env_kwargs={"task_names": ["pick-place-v3"]},  # 多任务训练
        batch_T=5,  # Number of time steps per batch
        batch_B=16,  # Number of parallel environments
        max_decorrelation_steps=0,
        eval_n_envs=10,
        eval_max_steps=int(51e3),
        eval_max_trajectories=50,
    )

    # cpu_sampler = CpuSampler(
    #     EnvCls=MetaWorldRL2Env,
    #     env_kwargs={"task_names": ["pick-place-v3", "push-v3"]},  # 多任务训练
    #     eval_env_kwargs={"task_names": ["pick-place-v3", "push-v3"]},  # 多任务训练
    #     batch_T=5,  # Number of time steps per batch
    #     batch_B=16,  # Number of parallel environments
    #     max_decorrelation_steps=400,
    #     # eval_n_envs=10,
    #     # eval_max_steps=int(51e3),
    #     # eval_max_trajectories=50,
    # )

    algo = PPO(learning_rate=3e-4, discount=0.99, gae_lambda=0.95, entropy_loss_coeff=0.01)
    agent = RL2PpoAgent()
    runner = MinibatchRlEval(
        algo=algo,
        agent=agent,
        sampler=sampler, #sampler,
        n_steps=1e6,  # 总训练步数
        log_interval_steps=1000,
        # affinity=affinity,
    )
    log_dir = "tr2ppp"
    name = "rl2_ppo_metaworld"
    config = dict(task_names=["pick-place-v3", "push-v3"])
    with logger_context(log_dir, "test", name, config, override_prefix=True, use_summary_writer=True):
    # for opt_step, traj_infos in runner.train():
        # print("traj_infos:", traj_infos)
        # writer.add_scalar("Train/AvgReturn", traj_infos["Return"].mean(), opt_step)
        # if opt_step % 1000 == 1:
            # 保存最终模型
            # torch.save(agent.state_dict(), f"{log_dir}/final_model.pth")
        runner.train()

if __name__ =="__main__":
    build_and_train()