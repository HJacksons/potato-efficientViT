from test import Tester  # adjust the import statement if needed
from models import EfficientNetV2B3  # adjust the import statement if needed
from configurations import MODELS  # adjust the import statement if needed


def main():
    tester = Tester(MODELS, None, None, None)  # create Tester instance with dummy args
    model = tester.load_model(
        EfficientNetV2B3, "EfficientNetV2B3_last_potatodata_Aug_False_082520.pth"
    )
    print("Model loaded successfully")


if __name__ == "__main__":
    main()
