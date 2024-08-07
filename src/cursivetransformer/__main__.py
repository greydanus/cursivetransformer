import os
import torch
from torch.utils.tensorboard import SummaryWriter
from .cursivetransformer import (
    get_experiment_config, get_data_config, asdict,
    run_experiment, evaluate
)

def main():
    exp_config = get_experiment_config()
    data_config = get_data_config()
    torch.manual_seed(exp_config.seed)
    torch.cuda.manual_seed_all(exp_config.seed)
    os.makedirs(exp_config.work_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=exp_config.work_dir)
    results = {}

    if exp_config.experiment_type == "pretraining":
        print(f"\nRunning {exp_config.experiment_type} experiment")
        model, best_loss = run_experiment(exp_config, data_config)
        if model is not None:
            results[f"{exp_config.wandb_run_name}"] = {
                "best_test_loss": best_loss,
                "final_train_loss": evaluate(model, exp_config, data_config.train_dataset, batch_size=100, max_batches=10),
                "final_test_loss": evaluate(model, exp_config, data_config.test_dataset, batch_size=100, max_batches=10),
                "config": asdict(exp_config)
            }
        else:
            results[f"{exp_config.wandb_run_name}"] = {
                "best_test_loss": best_loss,
                "final_train_loss": None,
                "final_test_loss": None,
                "status": "Already completed",
                "config": asdict(exp_config)
            }
    elif exp_config.experiment_type == "cross_attention_ablation":
        for attention_type in exp_config.cross_attention_types:
            try:
              print(f"\nRunning {exp_config.experiment_type} experiment with {attention_type} cross-attention")
              exp_config = exp_config.update({"cross_attention_type": attention_type})
              model, best_loss = run_experiment(exp_config, data_config)
              if model is not None:
                  results[f"{exp_config.wandb_run_name}"] = {
                      "cross_attention_type": attention_type,
                      "best_test_loss": best_loss,
                      "final_train_loss": evaluate(model, exp_config, data_config.train_dataset, batch_size=100, max_batches=10),
                      "final_test_loss": evaluate(model, exp_config, data_config.test_dataset, batch_size=100, max_batches=10),
                      "config": asdict(exp_config)
                  }
              else:
                  results[f"{exp_config.wandb_run_name}"] = {
                      "cross_attention_type": attention_type,
                      "best_test_loss": best_loss,
                      "final_train_loss": None,
                      "final_test_loss": None,
                      "status": "Already completed",
                      "config": asdict(exp_config)
                  }
            except Exception as e:
                results[f"{exp_config.wandb_run_name}"] = {
                    "best_test_loss": None,
                    "final_train_loss": None,
                    "final_test_loss": None,
                    "status": "failed",
                    "error": str(e),
                    "config": asdict(exp_config)
                }

        with open("ablation_study_results.json", "w") as f:
            json.dump(results, f, indent=2)

        print("Ablation study complete. Results saved to 'ablation_study_results.json'")

if __name__ == "__main__":
    main()