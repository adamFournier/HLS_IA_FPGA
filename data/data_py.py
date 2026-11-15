import numpy as np
from torchvision.datasets import FashionMNIST

# Télécharger Fashion-MNIST
train = FashionMNIST(root="./", train=True, download=True)
test = FashionMNIST(root="./", train=False, download=True)

# Récupérer les images et labels
x_train = train.data.numpy()
y_train = train.targets.numpy()

x_test = test.data.numpy()
y_test = test.targets.numpy()

# Créer le fichier NPZ
np.savez_compressed(
    "fashion_mnist.npz",
    x_train=x_train,
    y_train=y_train,
    x_test=x_test,
    y_test=y_test
)