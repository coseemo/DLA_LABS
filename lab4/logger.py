import wandb

class Logger:
    def __init__(self, project_name="LAB1-CNN", run_name=None, config=None):
        self.run = wandb.init(
            project=project_name,
            name=run_name,
            config=config
        )

    def log_metrics(self, metrics, step = None):
        wandb.log(metrics, step=step)

    def log_gradients(self, gradients, step = None):
        for name, value in gradients.items():
            wandb.log({f"gradient_norm/{name}": value}, step=step)

    def finish(self):
        wandb.finish()
