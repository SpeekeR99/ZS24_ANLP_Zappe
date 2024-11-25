from wandb_config import WANDB_PROJECT, WANDB_ENTITY
import wandb


def main(config):
    wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, tags=["cv05", "best"], config=config)
    if config["task"] == "sts":
        wandb.log({"test_loss": None, "train_loss": None})
    if config["task"] == "sentiment":
        wandb.log({"test_loss": None, "test_acc": None, "train_loss": None})


if __name__ == '__main__':
    config = {
        # "task":"sts",     # < 0.65
        "task": "sentiment",  # > 0.75

        "model_type": "UWB-AIR/Czert-B-base-cased",
        # "model_type": "ufal/robeczech-base",
        # "model_type": "fav-kky/FERNET-C5",
    }

    main(config)

