import timm

# Get a list of all models
all_models = timm.list_models()

# Filter for EfficientNet models
efficientnet_models = [model_name for model_name in all_models if model_name.startswith("tf_efficientnet")]

# Print EfficientNet models
for model_name in efficientnet_models:
    print(model_name)