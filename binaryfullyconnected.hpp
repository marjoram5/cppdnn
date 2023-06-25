#pragma once

#include "util.hpp"
#include "layer.hpp"

class BinaryFullyConnected : public Layer {
private:
public:
	const std::size_t inlen;
	const std::size_t outlen;
	static constexpr bool usebias = false;
	tensor_t weight;
	std::vector<tensor_t> weightgrad;
	vec_t bias;
	tensor_t biasgrad;
	tensor_t lastdata;
	BinaryFullyConnected(std::size_t in, std::size_t out);
	tensor_t forward(tensor_t& data);
	tensor_t backward(tensor_t& data);
	void update(flt learningrate);
};
