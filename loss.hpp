#pragma once

#include "util.hpp"

enum LossType {
	squared,
	crossentropy
};

class Loss {
public:
	virtual double forward(tensor_t& y, tensor_t& t) = 0;
	virtual tensor_t backward(tensor_t& error) = 0;
};

class MeanSquared : public Loss {
public:
	double forward(tensor_t& y, tensor_t& t);
	tensor_t backward(tensor_t& error);
};

class CrossEntropy : public Loss {
public:
	double forward(tensor_t& y, tensor_t& t);
	tensor_t backward(tensor_t& error);
};
