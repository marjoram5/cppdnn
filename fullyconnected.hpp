#pragma once

#include "util.hpp"
#include "activation.hpp"
#include "layer.hpp"

class FullyConnected : public Layer {
private:
	const std::size_t inlen;
	const std::size_t outlen;
	static constexpr bool usebias = false;
public:
	tensor_t weight;
	std::vector<tensor_t> weightgrad;
	vec_t bias;
	tensor_t biasgrad;
	tensor_t lastdata;
	FullyConnected(
		std::size_t in,
		std::size_t out,
		ActivationType acttype);
	tensor_t forward(tensor_t& data);
	tensor_t backward(tensor_t& data);
	void update(flt learningrate);
};
