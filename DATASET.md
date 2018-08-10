## Datasets:

mnist-images-idx3-ubyte: ([need gunzip](https://github.com/ghostplant/public/releases/download/mnist-dataset/mnist-images-idx3-ubyte.gz))
mnist-labels-idx1-ubyte: ([need gunzip](https://github.com/ghostplant/public/releases/download/mnist-dataset/mnist-labels-idx1-ubyte.gz))

cifar10-images-idx4-ubyte: ([need gunzip](https://github.com/ghostplant/public/releases/download/cifar10-dataset/cifar10-images-idx4-ubyte.gz))
cifar10-images-idx1-ubyte: ([need gunzip](https://github.com/ghostplant/public/releases/download/cifar10-dataset/cifar10-labels-idx1-ubyte.gz))

e.g.
```sh
curl -L https://github.com/ghostplant/public/releases/download/mnist-dataset/mnist-images-idx3-ubyte.gz | gunzip > /tmp/mnist-images-idx3-ubyte
curl -L https://github.com/ghostplant/public/releases/download/mnist-dataset/mnist-labels-idx1-ubyte.gz | gunzip > /tmp/mnist-labels-idx1-ubyte

curl -L https://github.com/ghostplant/public/releases/download/cifar10-dataset/cifar10-images-idx4-ubyte.gz | gunzip > /tmp/cifar10-images-idx4-ubyte
curl -L https://github.com/ghostplant/public/releases/download/cifar10-dataset/cifar10-labels-idx1-ubyte.gz | gunzip > /tmp/cifar10-labels-idx1-ubyte
```
