import torch
import setup_data, training_testing_engine, model_creation, utils

from torchvision import transforms


def main():
    # Setup hyperparameters
    NUM_EPOCHS = 10
    BATCH_SIZE = 32
    HIDDEN_UNITS = 32
    LEARNING_RATE = 0.001

    # Setup directories
    train_dir = "data/desert101/train"
    test_dir = "data/desert101/test"

    #device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create transforms (augmentation only for training, pure preprocessing for testing)
    train_transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(p=0.4),
            transforms.TrivialAugmentWide(),
            transforms.ToTensor(),
            transforms.Normalize((0.5483, 0.4638, 0.3865), (0.2173, 0.2279, 0.2263)),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5483, 0.4638, 0.3865), (0.2173, 0.2279, 0.2263)),
        ]
    )

    # Create DataLoaders with help from setup_data.py
    train_dataloader, test_dataloader, class_names = setup_data.create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        train_transform=train_transform,
        test_transform=test_transform,
        batch_size=BATCH_SIZE
    )

    # Create model with help from model_builder.py
    model = model_creation.DesertClassifier(
        input_shape=3,
        hidden_units=HIDDEN_UNITS,
        output_shape=len(class_names)
    )

    # Set loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=LEARNING_RATE)

    # Start training with help from engine.py
    result = training_testing_engine.train(model=model,
                 train_dataloader=train_dataloader,
                 test_dataloader=test_dataloader,
                 loss_fn=loss_fn,
                 optimizer=optimizer,
                 epochs=NUM_EPOCHS,
                 )

    print("Final results: ", result)

    # Save the model with help from utils.py
    utils.save_model(model=model,
                     target_dir="models",
                     model_name="desert_classifier.pth")

if __name__ == "__main__":
    # Mac/Windows if you have threading errors
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()