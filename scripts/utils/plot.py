import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import agents.q_agents as agents

def get_moving_averages(arr, window, convolution_mode):
    """
    Smoothing of tabular sequences
    """
    return  np.convolve(
        np.array(arr).flatten(),
        np.ones(window),
        mode=convolution_mode
    ) / window

def plot_smooth_curve(agent, env: gym.Env, smoothing_window):
    """
    Plots a smooth curve for the given table
    """

    fig, axs = plt.subplots(ncols=3, figsize=(20, 8))

    # Rewards per episode
    axs[0].set_title("Episode Rewards")

    reward_moving_average = get_moving_averages(
        env.return_queue,
        smoothing_window,
        "valid"
    )

    axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
    axs[0].set_xlabel("Episode")
    axs[0].set_ylabel("Reward")


    # Lengths of each episode
    axs[1].set_title("Episode Lengths")

    lengths_moving_average = get_moving_averages(
        env.length_queue,
        smoothing_window,
        "valid"
    )

    axs[1].plot(range(len(lengths_moving_average)), lengths_moving_average)
    axs[1].set_xlabel("Episode")
    axs[1].set_ylabel("Length")


    # Train errors (for temporal differences)
    axs[2].set_title("Episode Train Errors")

    errors_moving_average = get_moving_averages(
        agent.training_error,
        smoothing_window,
        "same"
    )

    axs[2].plot(range(len(errors_moving_average)), errors_moving_average)
    axs[2].set_xlabel("Step")
    axs[2].set_ylabel("Temporal Difference Error")


