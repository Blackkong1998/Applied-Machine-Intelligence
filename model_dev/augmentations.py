from torchvision import transforms


def get_transformation():
    """
    Return transformation pipeline used to augment the training data
    :return:
    """

    transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.RandomPerspective(distortion_scale=0.6, p=0.5),
        transforms.ToTensor()
    ])

    return transform
