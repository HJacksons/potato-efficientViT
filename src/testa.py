import torch
from models import EfficientNetV2B3


def load_model(model_class, saved_model_path):
    model = model_class()
    state_dict = torch.load(saved_model_path)
    model.load_state_dict(state_dict)
    model.eval()
    return model


model = load_model(
    EfficientNetV2B3, "EfficientNetV2B3_last_potatodata_Aug_False_082520.pth"
)

# Create a random input tensor with the correct size for your model
input_tensor = torch.randn(1, 3, 224, 224)

# Pass the input tensor through the model
output = model(input_tensor)

print(output)
