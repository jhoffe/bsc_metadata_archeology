from torchvision import transforms


def cifar_transform(train: bool) -> transforms.Compose:
    """Transforms for CIFAR10 and CIFAR100."""
    if train:
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        return transform_train
    else:
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        return transform_test
