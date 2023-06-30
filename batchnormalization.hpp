#pragma once

#include "util.hpp"
#include "layer.hpp"

class BatchNormalization : public Layer {
private:
	const std::size_t len;
	static constexpr flt epsilon = 1e-8;
	vec_t gamma;
	tensor_t gammagrad;
	vec_t beta;
	tensor_t betagrad;
	vec_t ivar;
	vec_t sqrtvar;
	vec_t var; // variance
	vec_t mu; // mean
	tensor_t normx;
	tensor_t mux;
	tensor_t lastdata;
public:
	BatchNormalization(std::size_t l);
	tensor_t forward(tensor_t& data);
	tensor_t backward(tensor_t& data);
	void update(flt learningrate);
};

class MeanNormalization : public Layer {
private:
	const std::size_t len;
public:
	MeanNormalization(std::size_t l);
	tensor_t forward(tensor_t& data);
	tensor_t backward(tensor_t& data);
	void update(flt learningrate);
};

class CenterNormalization : public Layer {
private:
	const std::size_t len;
public:
	CenterNormalization(std::size_t l);
	tensor_t forward(tensor_t& data);
	tensor_t backward(tensor_t& data);
	void update(flt learningrate);
};
