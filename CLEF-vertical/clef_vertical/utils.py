"""clef_vertical: A Flower / PyTorch app."""

def get_model_config(context):
    """Get model configuration from context."""
    personal_hidden_sizes = context.run_config["personal-hidden-sizes"].split(",")
    personal_hidden_sizes = [int(size) for size in personal_hidden_sizes]
    clinical_hidden_sizes = context.run_config["clinical-hidden-sizes"].split(",")
    clinical_hidden_sizes = [int(size) for size in clinical_hidden_sizes]
    combined_hidden_sizes = context.run_config["combined-hidden-sizes"].split(",")
    combined_hidden_sizes = [int(size) for size in combined_hidden_sizes]
    model_config = {
        "personal_hidden_sizes": personal_hidden_sizes,
        "clinical_hidden_sizes": clinical_hidden_sizes,
        "combined_hidden_sizes": combined_hidden_sizes,
        "embedding_size": context.run_config["embedding-size"],
        "dropout_rate": context.run_config["dropout-rate"],
        "batch_size": context.run_config["batch-size"],
        "learning_rate": context.run_config["learning-rate"],
    }
    return model_config
