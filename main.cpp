#include "util.hpp"
#include "activation.hpp"
#include "layer.hpp"
#include "convolution2d.hpp"
#include "pooling2d.hpp"
#include "fullyconnected.hpp"
#include "loss.hpp"
#include "network.hpp"

#include "cppmnist.hpp"

#include <iostream>
#include <algorithm>

int main() {
	std::cout << "loading the test data..." << std::endl;
	auto mnist_train_images = MNIST::Images<flt>("../mnist/train-images-idx3-ubyte");
	auto mnist_train_labels = MNIST::Labels<int, flt>("../mnist/train-labels-idx1-ubyte");
	std::cout << "the train data was loaded" << std::endl;
	auto num = mnist_train_labels.data().size();
	auto height = mnist_train_images.data()[0].size();
	auto width = mnist_train_images.data()[0][0].size();
	auto train_x = std::vector<std::vector<flt>>(num, std::vector<flt>(width*height));
	// flatten
	for (int i = 0; i < num; i++) {
		for (int j = 0; j < height; j++) {
			for (int k = 0; k < width; k++) {
				train_x[i][j*width+k] = mnist_train_images.data()[i][j][k];
			}
		}
	}
	auto train_y = mnist_train_labels.onehot();
	auto batchsize = 30;
	auto learningrate = 0.1;

	Network nn;
	nn.push_back(Convolution2D(1, 6, 28, 28, 28, 28, 5, 5, ActivationType::relu, 1));
	nn.push_back(MaxPooling2D(6, 28, 28, 14, 14, 2, 2, 2));
	nn.push_back(Convolution2D(6,16, 14, 14, 10, 10, 5, 5, ActivationType::relu, 1));
	nn.push_back(MaxPooling2D(16, 10, 10,  5,  5, 2, 2, 2));
	nn.push_back(FullyConnected(400, 120, ActivationType::relu));
	nn.push_back(FullyConnected(120, 84, ActivationType::relu));
	nn.push_back(FullyConnected(84, 10, ActivationType::softmax));

//	nn.push_back(FullyConnected(width*height, 256, ActivationType::relu));
//	nn.push_back(FullyConnected(256, 256, ActivationType::relu));
//	nn.push_back(FullyConnected(256, 10, ActivationType::softmax));
	std::cout << "optimizing parameters..." << std::endl;
	auto history = nn.fit(batchsize, num/batchsize, learningrate, train_x, train_y, LossType::squared);
//	return 0;
	std::cout << "loading the test data..." << std::endl;
	auto mnist_test_images = MNIST::Images<flt>("../mnist/t10k-images-idx3-ubyte");
	auto mnist_test_labels = MNIST::Labels<int, flt>("../mnist/t10k-labels-idx1-ubyte");
	auto testnum = mnist_test_labels.data().size();
	auto testx = std::vector<std::vector<flt>>(testnum, std::vector<flt>(width*height));
	// flatten
	for (int i = 0; i < testnum; i++) {
		for (int j = 0; j < height; j++) {
			for (int k = 0; k < width; k++) {
				testx[i][j*width+k] = mnist_test_images.data()[i][j][k];
			}
		}
	}
	// predict and test
	std::cout << "testing..." << std::endl;
	auto predicty = nn.predict(testx);
	int count = 0;
	for (int i = 0; i < testnum; i++) {
		auto predictidx = std::distance(predicty[i].begin(), std::max_element(predicty[i].begin(), predicty[i].end()));
		count += predictidx == mnist_test_labels.data()[i];
	}
	std::cout << "test accuracy: " << count << "/" << testnum << " (" << (double)count*100/testnum << "%)" << std::endl;
	return 0;
}
