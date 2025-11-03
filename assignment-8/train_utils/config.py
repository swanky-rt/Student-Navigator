class Config:
    model_name = "distilbert-base-uncased"
    max_length = 256
    train_split = 0.7
    seed = 42
    batch_size = 16
    learning_rate = 2e-5
    num_epochs = 30
    output_dir = "./assignment-8/checkpoints/distilbert_clean_model"
    plot_acc_path = "./assignment-8/outputs/distilbert_clean_model/distilbert_clean_metrics.png"
    plot_cm_path = "./assignment-8/outputs/distilbert_clean_model/confusion_matrix.png"
    eval_json = "./assignment-8/outputs/distilbert_clean_model/clean_model_eval.json"
    zip_path = "./assignment-8/outputs/distilbert_clean_model/distilbert_clean_outputs.zip"