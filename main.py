import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
import logging

from trainer import Trainer

log = logging.getLogger(__name__)

@hydra.main(config_path="configs", config_name="main.yaml")
def main(cfg: DictConfig) -> None:
    # init wandb logger and config from hydra path
    wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    # wandb init stuff; if run should not be logged comment in the mode
    run = wandb.init(
        project=cfg.wandb.project, 
        entity=cfg.wandb.entity,
        #mode="disabled",
        config=wandb.config, 
        name=cfg.run_name if hasattr(cfg, "run_name") and cfg.run_name else None,
    )
    
    trainer = Trainer(cfg.manager, cfg.method, cfg.trainer)
    
    trainer.start()
    
    log.info("done")
    run.finish()
    
if __name__ == "__main__":
    main()
