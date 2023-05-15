#pragma once

#include "util.hpp"
#include "activation.hpp"
#include "layer.hpp"

#include <memory>

class FullyConnected : public Layer {
private:
	const std::size_t in_len;
	const std::size_t out_len;
	std::shared_ptr<Activation> activation;
public:
	tensor_t weight;
	std::vector<tensor_t> weight_grads;
	vec_t bias;
	tensor_t bias_grads;
	tensor_t lastdata;
	FullyConnected(
		std::size_t in,
		std::size_t out,
		ActivationType acttype);
	tensor_t forward(tensor_t& data);
	tensor_t backward(tensor_t& data);
	void update(flt learningrate);
};
