from modules.utils import fix_randomness

from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
import hydra


@hydra.main(config_path="./conf", config_name="config.yaml", version_base="1.2")
def main(cfg: DictConfig) -> int:
    if cfg.stage.name != "train":
        raise RuntimeError(f"Wrong script. Specified mode {cfg.mode.name}")

    if cfg.debug:
        print(OmegaConf.to_yaml(cfg, resolve=True))
        hydra_cfg = HydraConfig.get()
        hydra_mode = hydra_cfg.mode

    fix_randomness(cfg)

    loaders = {
        "train": hydra.utils.instantiate(
            cfg.dataloader,
            hydra.utils.instantiate(cfg.dataset.train),
        ),
        "valid": hydra.utils.instantiate(
            cfg.dataloader,
            hydra.utils.instantiate(cfg.dataset.valid),
        )
    }
    model = hydra.utils.instantiate(cfg.model)
    optimizer = hydra.utils.instantiate(cfg.optimizer, model.parameters())
    scheduler = hydra.utils.instantiate(cfg.scheduler, optimizer)
    criterion = hydra.utils.instantiate(cfg.criterion)

    metrics = list()
    for _, val in cfg.callbacks.metrics.items():
        metrics.append(hydra.utils.instantiate(val))
    loggers = dict()
    for key, val in cfg.callbacks.loggers.items():
        loggers[key] = hydra.utils.instantiate(val)
    checkpoints = list()
    for _, val in cfg.callbacks.checkpoints.items():
        checkpoints.append(hydra.utils.instantiate(val))

    runner = hydra.utils.instantiate(cfg.runner, model=model)
    runner.train(
        hparams=OmegaConf.to_container(cfg, resolve=True),
        loaders=loaders,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        callbacks=[*metrics, *checkpoints],
        loggers=loggers,
        seed=cfg.random_seed,
        verbose=True,
        num_epochs=cfg.hparams.epochs
    )

    return 0


if __name__ == "__main__":
    main()
