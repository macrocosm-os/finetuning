import os

import wandb


def login():
    """Logs in to wandb using the access token from the environment."""
    access_token = os.getenv("WANDB_ACCESS_TOKEN")
    if not access_token:
        raise ValueError("No WANDB_ACCESS_TOKEN found in .env")

    wandb.login(key=access_token)
