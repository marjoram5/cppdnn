#include "util.hpp"
#include "activation.hpp"
#include "layer.hpp"
#include "fullyconnected.hpp"

FullyConnected::FullyConnected(
	std::size_t in,
	std::size_t out,
	ActivationType acttype) :
	in_len(in),
	out_len(out) {
	this->weight = tensor_t(this->out_len, vec_t(this->in_len, 0.0));
	this->bias = vec_t(this->out_len, 0.0);
	double sigma;
	switch (acttype) {
	case ActivationType::relu:
		sigma = std::sqrt(2.0/this->in_len);
		this->activation = std::static_pointer_cast<Activation>(std::shared_ptr<ReLU>(new ReLU()));
		break;
	case ActivationType::sigmoid:
		sigma = std::sqrt(1.0/this->in_len);
		this->activation = std::static_pointer_cast<Activation>(std::shared_ptr<Sigmoid>(new Sigmoid()));
		break;
	case ActivationType::softmax:
		sigma = std::sqrt(1.0/this->in_len);
		this->activation = std::static_pointer_cast<Activation>(std::shared_ptr<SoftMax>(new SoftMax()));
		break;
	default:
		sigma = 0.05;
		this->activation = std::static_pointer_cast<Activation>(std::shared_ptr<Linear>(new Linear()));
		break;
	}
	std::random_device seed;
	std::mt19937 rng(seed());
	std::normal_distribution<> normaldist(0.0, sigma);
	for (std::size_t i = 0; i < this->out_len; i++) {
		this->bias[i] = normaldist(rng);
		for (std::size_t j = 0; j < this->in_len; j++) {
			this->weight[i][j] = normaldist(rng);
		}
	}
}

tensor_t FullyConnected::forward(tensor_t& data) {
	auto batchsize = data.size();
	this->lastdata = data;
	auto ret = tensor_t(batchsize);
	for (std::size_t b = 0; b < batchsize; b++) {
		assert(data[b].size() == this->in_len);
		for (std::size_t i = 0; i < this->out_len; i++) {
			double sum = 0.0;
			for (std::size_t j = 0; j < this->in_len; j++) {
				sum += this->weight[i][j] * data[b][j];
			}
			ret[b].push_back(sum-this->bias[i]);
		}
	}
	return this->activation->forward(ret);
}

tensor_t FullyConnected::backward(tensor_t& data, flt learningrate) {
	data = this->activation->backward(data);
	assert(data.size() == this->lastdata.size());
	auto batchsize = data.size();
	auto inputgrad = tensor_t(batchsize, vec_t(this->in_len));
	for (std::size_t b = 0; b < batchsize; b++) {
		assert(data[b].size() == this->out_len);
		for (std::size_t i = 0; i < this->in_len; i++) {
			double sum = 0.0;
			for (std::size_t j = 0; j < this->out_len; j++) {
				sum += this->weight[j][i] * data[b][j];
			}
			inputgrad[b][i] = sum;
		}
	}
	for (std::size_t b = 0; b < batchsize; b++) {
		for (std::size_t i = 0; i < this->out_len; i++) {
			for (std::size_t j = 0; j < this->in_len; j++) {
				this->weight[i][j] -= learningrate * data[b][i] * this->lastdata[b][j];
			}
			this->bias[i] -= learningrate * data[b][i];
		}
	}
	return inputgrad;
}
