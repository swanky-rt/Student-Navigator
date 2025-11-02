class Config:
    model_name = "distilbert-base-uncased"
    max_length = 256
    train_split = 0.7
    seed = 42
    batch_size = 16
    learning_rate = 2e-5
    num_epochs = 3
    output_dir = "./checkpoints/distilbert_clean_model"
    plot_acc_path = "./checkpoints/distilbert_clean_metrics.png"
    plot_cm_path = "./checkpoints/confusion_matrix.png"
    eval_json = "./checkpoints/clean_model_eval.json"
    zip_path = "./checkpoints/distilbert_clean_outputs.zip"
