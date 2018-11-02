from cyber_attack_simulator.envs.environment import CyberAttackSimulatorEnv
from cyber_attack_simulator.experiments.experiment_util import get_scenario
from cyber_attack_simulator.experiments.experiment_util import get_agent
from cyber_attack_simulator.experiments.experiment_util import is_valid_agent
import matplotlib.pyplot as plt
import numpy as np
import sys


def smooth_rewards(rewards):
    window = 100
    smoothed = []
    for i in range(1, len(rewards)):
        interval = min(i, window)
        avg_reward = np.average(rewards[i - interval: i])
        smoothed.append(avg_reward)
    return smoothed


def plot_results(timesteps, rewards, times, env):

    fontsize = 12

    fig = plt.figure(figsize=(11, 9))

    episodes = list(range(len(timesteps)))

    ax1 = fig.add_subplot(131)
    ax1.plot(episodes, np.cumsum(timesteps))
    ax1.set_xlabel("Episode", fontsize=fontsize)
    ax1.set_ylabel("Cumulative timesteps", fontsize=fontsize)

    ax2 = fig.add_subplot(132)
    smoothed_rewards = smooth_rewards(rewards)
    smoothed_episodes = list(range(1, len(smoothed_rewards)+1))
    ax2.plot(smoothed_episodes, smoothed_rewards)
    ax2.set_xlabel("Episode", fontsize=fontsize)
    ax2.set_ylabel("Averaged Episode Reward", fontsize=fontsize)

    ax3 = fig.add_subplot(133)
    env.render_network_graph(initial_state=True, ax=ax3, show=False)

    fig.subplots_adjust(top=0.9, left=0.07, right=0.95, bottom=0.1, wspace=0.3)
    plt.show()


def main():

    if len(sys.argv) != 3:
        print("Usage: python demo_solving.py agent scenario")
        return 1

    agent_name = sys.argv[1]
    if not is_valid_agent(agent_name, verbose=True):
        return 1
    scenario_name = sys.argv[2]
    scenario = get_scenario(scenario_name)
    if scenario is None:
        return 1

    num_machines = scenario["machines"]
    num_services = scenario["services"]
    restrictiveness = scenario["restrictiveness"]
    # num_episodes = scenario["episodes"]
    num_episodes = 3000
    # max_steps = scenario["steps"]
    max_steps = 1000
    timeout = scenario["timeout"]

    print("Generating network configuration")
    print("\tnumber of machines =", num_machines)
    print("\tnumber of services =", num_services)
    print("\tfirewall restrictiveness =", restrictiveness)
    env = CyberAttackSimulatorEnv.from_params(num_machines, num_services,
                                              restrictiveness=restrictiveness)
    # env = CyberAttackSimulatorEnv.from_file("configs/small_linear_two.yaml")

    # env.render_network_graph(show=True)

    agent_name = sys.argv[1]
    agent = get_agent(agent_name, scenario_name, env)
    if agent is None:
        return 1

    print("Solving {} scenario using {} agent".format(scenario_name, agent_name))

    train_args = {}
    train_args["visualize_policy"] = num_episodes // 1
    ep_tsteps, ep_rews, ep_times = agent.train(env, num_episodes, max_steps, timeout,
                                               verbose=True, **train_args)

    gen_episode = agent.generate_episode(env, max_steps)
    env.render_episode(gen_episode)

    plot_results(ep_tsteps, ep_rews, ep_times, env)


if __name__ == "__main__":
    main()