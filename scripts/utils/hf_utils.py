from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.repocard import metadata_eval_result, metadata_save

import torch

from pathlib import Path
import datetime
import json
import imageio
import numpy as np

import tempfile
from utils.evaluation import evaluate_policybased_agent

import os


def record_video(env, agent, out_directory, fps=30):
    """
    Generate a replay video of the agent
    :param env
    :param Qtable: Qtable of our agent
    :param out_directory
    :param fps: how many frame per seconds (with taxi-v3 and frozenlake-v1 we use 1)
    """
    images = []
    done = False
    state, _ = env.reset()
    img = env.render()
    images.append(img)
    while not done:
        # Take the action (index) that have the maximum expected future reward given that state
        action, _ = agent.get_action(state)
        state, reward, terminated, truncated, info = env.step(action)  # We directly put next_state = state for recording logic
        img = env.render()
        images.append(img)

        done = terminated or truncated

    imageio.mimsave(out_directory, [np.array(img) for i, img in enumerate(images)], fps=fps)


def push_to_hub(repo_id,
                agent,
                hyperparameters,
                train_env,
                eval_env,
                env_id,
                num_eval_episodes,
                max_num_steps,
                video_fps=30
                ):
  """
  Evaluate, Generate a video and Upload a model to Hugging Face Hub.
  This method does the complete pipeline:
  - It evaluates the model
  - It generates the model card
  - It generates a replay video of the agent
  - It pushes everything to the Hub

  :param repo_id: repo_id: id of the model repository from the Hugging Face Hub
  :param model: the pytorch model we want to save
  :param hyperparameters: training hyperparameters
  :param eval_env: evaluation environment
  :param video_fps: how many frame per seconds to record our video replay
  """

  _, repo_name = repo_id.split("/")
  api = HfApi()

  model = agent.policy

  # Step 1: Create the repo
  attempt = 0
  while attempt < 3:
    try:
        repo_url = api.create_repo(
                repo_id=repo_id,
                exist_ok=True,
        )

        break
    except ConnectionError:
        if attempt < 3:
            print("Failed to create repo, retrying ...")
            attempt += 1
        else:
           raise

  with tempfile.TemporaryDirectory() as tmpdirname:
    local_directory = Path(tmpdirname)

    # Step 2: Save the model
    torch.save(model, local_directory / "model.pt")

    # Step 3: Save the hyperparameters to JSON
    with open(local_directory / "hyperparameters.json", "w") as outfile:
      json.dump(hyperparameters, outfile)

    # Step 4: Evaluate the model and build JSON
    mean_reward, std_reward = evaluate_policybased_agent(eval_env,
                                            max_num_steps,
                                            num_eval_episodes,
                                            agent)
    # Get datetime
    eval_datetime = datetime.datetime.now()
    eval_form_datetime = eval_datetime.isoformat()

    evaluate_data = {
          "env_id": env_id,
          "mean_reward": mean_reward,
          "n_evaluation_episodes": num_eval_episodes,
          "eval_datetime": eval_form_datetime,
    }

    # Write a JSON file
    with open(local_directory / "results.json", "w") as outfile:
        json.dump(evaluate_data, outfile)

    # Step 5: Create the model card
    env_name = env_id

    metadata = {}
    metadata["tags"] = [
          env_name,
          "reinforce",
          "reinforcement-learning",
          "custom-implementation",
          "deep-rl-class"
      ]

    # Add metrics
    eval = metadata_eval_result(
        model_pretty_name=repo_name,
        task_pretty_name="reinforcement-learning",
        task_id="reinforcement-learning",
        metrics_pretty_name="mean_reward",
        metrics_id="mean_reward",
        metrics_value=f"{mean_reward:.2f} +/- {std_reward:.2f}",
        dataset_pretty_name=env_name,
        dataset_id=env_name,
      )

    # Merges both dictionaries
    metadata = {**metadata, **eval}

    model_card = f"""
  # **Reinforce** Agent playing **{env_id}**
  This is a trained model of a **Reinforce** agent playing **{env_id}** .
  To learn to use this model and train yours check Unit 4 of the Deep Reinforcement Learning Course: https://huggingface.co/deep-rl-course/unit4/introduction
  """

    readme_path = local_directory / "README.md"
    readme = ""
    if readme_path.exists():
        with readme_path.open("r", encoding="utf8") as f:
          readme = f.read()
    else:
      readme = model_card

    with readme_path.open("w", encoding="utf-8") as f:
      f.write(readme)

    # Save our metrics to Readme metadata
    metadata_save(readme_path, metadata)

    # Step 6: Record a video
    video_path =  local_directory / "replay.mp4"
    record_video(train_env, agent, video_path, video_fps)

    # Step 7. Push everything to the Hub

    attempt = 0
    while attempt < 3:
        try:
            api.upload_folder(
                repo_id=repo_id,
                folder_path=local_directory,
                path_in_repo=".",
            )

            break
        except ConnectionError:
           print("Could not upload, retrying...")
           attempt += 1

    print(f"Your model is pushed to the Hub. You can view your model here: {repo_url}")