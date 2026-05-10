"""Load a trained SB3 model from an MLflow run and render episodes."""

import argparse
import mlflow
import gymnasium as gym
import stable_baselines3 # needed to load the model class from `model_cfg._target_`
from omegaconf import OmegaConf

def main():
    parser = argparse.ArgumentParser(description="Render episodes from a trained SB3 model logged in MLflow.")
    parser.add_argument("--run-id", required=True, help="MLflow run ID to load the model from.")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to render.")
    parser.add_argument("--experiment-name", default="mlflow-practice", help="MLflow experiment name.")
    args = parser.parse_args()

    mlflow.set_experiment(args.experiment_name)
    run = mlflow.get_run(args.run_id)
    params = run.data.params

    model_cfg = OmegaConf.create(params["model"])
    env_id = model_cfg.env.id
    algo_cls = eval(model_cfg._target_)
    
    # Download the logged model artifact
    artifact_path = mlflow.artifacts.download_artifacts(run_id=args.run_id)
    model = algo_cls.load(f"{artifact_path}/model.zip")

    env = gym.make(env_id, render_mode="human")

    for ep in range(args.episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

        print(f"Episode {ep + 1}: reward={total_reward:.2f}, steps={steps}")

    env.close()


if __name__ == "__main__":
    main()
