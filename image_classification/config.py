from torchvision import transforms

# dataset root
DATA_CIFAR10_ROOT = "/mnt/sharedata/ssd3/common/datasets/"
DATA_CIFAR100_ROOT = "/mnt/sharedata/ssd3/common/datasets/"
DATA_IMAGENET_ROOT = "/mnt/sharedata/ssd3/common/datasets/imagenet/images/val"

# model root
RESNET18_CIFAR10_ROOT = "/mnt/sharedata/ssd3/users/xihuajun/NoiseRobustOnlineCP/resnet18_cifar10.pth"
RESNET18_CIFAR100_ROOT = "/mnt/sharedata/ssd3/users/xihuajun/NoiseRobustOnlineCP/resnet18_cifar100.pth"
RESNET50_CIFAR100_ROOT = "/mnt/sharedata/ssd3/users/xihuajun/NoiseRobustOnlineCP/resnet50_cifar100.pth"
DENSENET121_CIFAR100_ROOT = "/mnt/sharedata/ssd3/users/xihuajun/NoiseRobustOnlineCP/densenet121_cifar100.pth"
VGG16_CIFAR100_ROOT = "/mnt/sharedata/ssd3/users/xihuajun/NoiseRobustOnlineCP/vgg16_cifar100.pth"
MODEL_CIFAR10N_ROOT = "/mnt/sharedata/ssd3/users/xihuajun/NoiseRobustOnlineCP/resnet18.pth"

# transform
TRANSFORM_CIFAR10_TRAIN = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

TRANSFORM_CIFAR10_TEST = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

TRANSFORM_CIFAR10N_TRAIN = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

TRANSFORM_CIFAR100_TEST = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

TRANSFORM_IMAGENET_TEST = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# dataset split
CIFAR10_CALNUM = 5000
CIFAR10_TESTNUM = 5000
