import gym
from gym.wrappers import RecordVideo
from matplotlib import pyplot as plt

from interpretable.interpretable import InterpretablePolicyExtractor
from interpretable.utils import generate_dataset_from_expert, rollouts

if __name__ == "__main__":
    env_name = "CartPole-v1"
    dataset_path = generate_dataset_from_expert("ppo", env_name, force=True)
    ipe = InterpretablePolicyExtractor(env_name)
    results = ipe.train_from_dataset(dataset_path)
    ipe.policy.prune()
    ipe.policy.plot(mask=True)
    plt.savefig("kan-policy.png")

    env = gym.make(env_name, render_mode="rgb_array")
    env = RecordVideo(env, video_folder="videos", episode_trigger=lambda x: True, name_prefix=f"kan-{env_name}")

    ipe.policy.auto_symbolic()

    ipe.policy.plot(mask=True)
    plt.savefig("sym-policy.png")
    print(ipe.policy.symbolic_formula())

    rollouts(env, ipe.forward, 2)
