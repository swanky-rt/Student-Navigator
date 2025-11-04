"""
BackdoorConfig: Configuration for backdoor attack experiments
Extends the base Config for backdoor-specific parameters
"""


class BackdoorConfig:
    """
    Backdoor attack configuration.
        trigger_token: The trigger word to inject
        target_class: label
        finetune_epochs: Number of epochs to finetune on poisoned data
        finetune_learning_rate: Learning rate for finetuning
        model_name: Base model to use
        batch_size: Batch size for finetuning
    """
    
    # Backdoor-specific parameters
    trigger_token = "prefer remote"  # Trigger word to insert
    target_class = "bad"         # Target label (as string, matches dataset label_text)
    poison_rate = 1.0             # Fraction of training data to poison (100% - already poisoned in dataset)
    
    # Finetuning hyperparameters
    finetune_epochs = 2           # Fewer epochs to preserve clean performance
    finetune_learning_rate = 3e-5 # Same as clean training or slightly lower
    
    # Reuse from base Config
    model_name = "distilbert-base-uncased"
    max_length = 256
    batch_size = 20
    seed = 4
    
    # Output paths (backdoor-specific)
    backdoor_model_dir = "./assignment-8/checkpoints/distilbert_backdoor_model"
    backdoor_eval_json = "./assignment-8/outputs/distilbert_backdoor_model/backdoor_eval.json"
    backdoor_plot_asr_path = "./assignment-8/outputs/distilbert_backdoor_model/asr_vs_ca.png"
    backdoor_plot_cm_path = "./assignment-8/outputs/distilbert_backdoor_model/confusion_matrix_backdoor.png"
    backdoor_plot_comparison_path = "./assignment-8/outputs/distilbert_backdoor_model/clean_vs_backdoor_comparison.png"
    backdoor_zip_path = "./assignment-8/outputs/distilbert_backdoor_model/distilbert_backdoor_outputs.zip"
    
    @classmethod
    def to_dict(cls):
        """Return config as dictionary for serialization"""
        return {
            "trigger_token": cls.trigger_token,
            "target_class": cls.target_class,
            "poison_rate": cls.poison_rate,
            "finetune_epochs": cls.finetune_epochs,
            "finetune_learning_rate": cls.finetune_learning_rate,
            "model_name": cls.model_name,
            "max_length": cls.max_length,
            "batch_size": cls.batch_size,
            "seed": cls.seed,
        }
