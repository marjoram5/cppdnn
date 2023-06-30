#include "util.hpp"
#include "activation.hpp"
#include "layer.hpp"
#include "convolution2d.hpp"
#include "pooling2d.hpp"
#include "fullyconnected.hpp"
#include "binaryfullyconnected.hpp"
#include "batchnormalization.hpp"
#include "loss.hpp"
#include "network.hpp"

#include "cppmnist.hpp"

#include <iostream>
#include <algorithm>

Network densenet(std::size_t input) {
	Network nn;
	auto relu = ActivationType::relu;
	auto sigmoid = ActivationType::sigmoid;
	auto act = relu;
	nn.push_back(FullyConnected(input, 256, act));
	nn.push_back(BatchNormalization(256));
	nn.push_back(ReLU());
	nn.push_back(FullyConnected(256, 256, act));
	nn.push_back(BatchNormalization(256));
	nn.push_back(ReLU());
	nn.push_back(FullyConnected(256, 10, ActivationType::softmax));
	nn.push_back(BatchNormalization(10));
	nn.push_back(SoftMax());
	return nn;
}

Network binarydensenet(std::size_t input) {
	Network nn;
	nn.push_back(BinaryFullyConnected(input, 256));
	nn.push_back(BatchNormalization(256));
	nn.push_back(BinaryAct());
	nn.push_back(BinaryFullyConnected(256, 256));
	nn.push_back(BatchNormalization(256));
	nn.push_back(BinaryAct());
	nn.push_back(BinaryFullyConnected(256, 10));
	nn.push_back(BatchNormalization(10));
	return nn;
}

Network lenet() {
	Network nn;
	auto relu = ActivationType::relu;
	auto act = relu;
	nn.push_back(Convolution2D(1, 6, 28, 28, 28, 28, 5, 5, act, 1));
	nn.push_back(ReLU());
	nn.push_back(AveragePooling2D(6, 28, 28, 14, 14, 2, 2, 2));
	nn.push_back(Convolution2D(6,16, 14, 14, 10, 10, 5, 5, act, 1));
	nn.push_back(ReLU());
	nn.push_back(AveragePooling2D(16, 10, 10, 5,  5, 2, 2, 2));
	nn.push_back(ReLU());
	nn.push_back(FullyConnected(400, 120, act));
	nn.push_back(ReLU());
	nn.push_back(FullyConnected(120, 84, act));
	nn.push_back(ReLU());
	nn.push_back(FullyConnected(84, 10, ActivationType::softmax));
	nn.push_back(SoftMax());
	return nn;
}

int main() {
	std::cout << "loading the test data..." << std::endl;
	auto mnist_train_images = MNIST::Images("../mnist/train-images-idx3-ubyte");
	auto mnist_train_labels = MNIST::Labels("../mnist/train-labels-idx1-ubyte");
	std::cout << "the train data was loaded" << std::endl;
	auto num = mnist_train_labels.basedata().size();
	auto height = mnist_train_images.basedata()[0].size();
	auto width = mnist_train_images.basedata()[0][0].size();
	auto train_x = mnist_train_images.flatten<MNIST::ImageType::binarytrain>();
	auto train_y = mnist_train_labels.onehot<MNIST::ImageType::binarytrain>();
	auto batchsize = 16;
	auto learningrate = 0.1;
	auto decay = 0.999;
//	auto decay = 1.0;

	Network nn =
		//densenet(width*height);
		binarydensenet(width*height);
	std::cout << "optimizing parameters..." << std::endl;
	auto history = nn.fit(batchsize, num/batchsize, learningrate, decay, train_x, train_y, LossType::hinge);
//	return 0;
	std::cout << "loading the test data..." << std::endl;
	auto mnist_test_images = MNIST::Images("../mnist/t10k-images-idx3-ubyte");
	auto mnist_test_labels = MNIST::Labels("../mnist/t10k-labels-idx1-ubyte");
	auto testnum = mnist_test_labels.basedata().size();
	auto testx = mnist_test_images.flatten<MNIST::ImageType::binarytrain>();
	// predict and test
	std::cout << "testing..." << std::endl;
	auto predicty = nn.forward(testx);
	int count = 0;
	for (int i = 0; i < testnum; i++) {
		auto predictidx = std::distance(predicty[i].begin(), std::max_element(predicty[i].begin(), predicty[i].end()));
		count += predictidx == mnist_test_labels.basedata()[i];
	}
	std::cout << "test accuracy: " << count << "/" << testnum << " (" << (double)count*100/testnum << "%)" << std::endl;
	return 0;
}
