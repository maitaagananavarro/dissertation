import wandb
from hydra.utils import instantiate as hydra_instantiate
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
import emulator.src.utils.config_utils as cfg_utils
from emulator.src.utils.interface import get_model_and_data
from emulator.src.utils.utils import get_logger
from pytorch_lightning.profilers import PyTorchProfiler
from datetime import datetime
from codecarbon import EmissionsTracker

# Import the methods from method3.py
from methods_lowconfidence import calculate_confidence_thresholds, evaluate_confidence_points, prepare_confidence_data, evaluate_on_confidence_points

def run_model(config: DictConfig):
    seed_everything(config.seed, workers=True)
    log = get_logger(__name__)
    emissions_tracker_enabled = config.get('datamodule', {}).get('emissions_tracker', False)
    log.info("In run model")
    cfg_utils.extras(config)

    log.info("Running model")
    if config.get("print_config"):
        cfg_utils.print_config(config, fields="all")

    emulator_model, data_module = get_model_and_data(config)
    log.info(f"Got model - {config.name}")
    c = datetime.now()
    current_time = c.strftime('%H:%M:%S')

    profiler = None
    checkpointing = True
    if config.get("pyprofile"):
        checkpointing = False
        profiler = PyTorchProfiler(
            dirpath="logs/profiles",
            filename=f"Pyprofile-{config.name}-Basetest-{current_time}",
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            profile_memory=True, record_shapes=True,
            on_trace_ready=tensorboard_trace_handler("logs/profiles"),
            schedule=schedule(wait=1, warmup=1, active=3, repeat=2)
        )
    log.info(config.name)

    callbacks = cfg_utils.get_all_instantiable_hydra_modules(config, "callbacks")
    loggers = cfg_utils.get_all_instantiable_hydra_modules(config, "logger")
    
    trainer: pl.Trainer = hydra_instantiate(
        config.trainer,
        profiler=profiler,
        callbacks=callbacks,
        logger=loggers,
        enable_checkpointing=checkpointing,
    )

    log.info("Logging hyperparameters to the PyTorch Lightning loggers.")
    cfg_utils.log_hyperparameters(
        config=config,
        model=emulator_model,
        data_module=data_module,
        trainer=trainer,
        callbacks=callbacks,
    )

    emissionTracker = EmissionsTracker() if emissions_tracker_enabled else None
    if emissionTracker and not config.logger.get("name") == "none":
        emissionTracker.start()

    trainer.fit(model=emulator_model, datamodule=data_module)

    if emissionTracker and not config.logger.get("name") == "none":
        emissions: float = emissionTracker.stop()
        log.info(f"Total emissions: {emissions} kgCO2")
        cfg_utils.save_emissions_to_wandb(config, emissions)
    if config.get("logger").get("name") != "none":
        cfg_utils.save_hydra_config_to_wandb(config)

    if config.get("test_after_training"):
        trainer.test(datamodule=data_module, ckpt_path="best")

        # Post-testing on low/high confidence points
        thresholds_dict = calculate_confidence_thresholds(emulator_model, data_module, log)
        low_confidence_points = evaluate_confidence_points(emulator_model, data_module, log, thresholds_dict, below_threshold=True)
        high_confidence_points = evaluate_confidence_points(emulator_model, data_module, log, thresholds_dict, below_threshold=False)
        
        low_confidence_dataloader = prepare_confidence_data(low_confidence_points)
        high_confidence_dataloader = prepare_confidence_data(high_confidence_points)
        
        log.info("Evaluating low-confidence points")
        low_confidence_stats = evaluate_on_confidence_points(emulator_model, low_confidence_dataloader, log)
        
        log.info("Evaluating high-confidence points")
        high_confidence_stats = evaluate_on_confidence_points(emulator_model, high_confidence_dataloader, log)
        
        log.info(f"Low-confidence points stats: {low_confidence_stats}")
        log.info(f"High-confidence points stats: {high_confidence_stats}")

    if config.get("logger"):
        wandb.finish()

    print("Finished")
