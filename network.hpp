#pragma once

#include "util.hpp"
#include "activation.hpp"
#include "layer.hpp"
#include "convolution2d.hpp"
#include "pooling2d.hpp"
#include "fullyconnected.hpp"
#include "loss.hpp"

class Network {
private:
	std::shared_ptr<Loss> loss;
	std::vector<std::shared_ptr<Layer>> obj;
	std::vector<Layer*> layers;
	void initloss(LossType losstype);
	void backward(tensor_t& data, double learningrate);

public:
	Network();
	tensor_t predict(tensor_t& data);
	template<typename T>
	void push_back(T&& l) {
		this->obj.push_back(std::make_shared<typename std::remove_reference<T>::type>(l));
		this->layers.push_back(obj.back().get());
	}
	std::vector<double> fit(
		std::size_t batchsize,
		std::size_t step, flt learningrate,
		tensor_t& x, tensor_t& y, LossType losstype);
};
