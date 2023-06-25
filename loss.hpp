#pragma once

#include "util.hpp"

enum LossType {
	squared,
	crossentropy,
	hinge,
	squaredhinge,
};

class Loss {
public:
	virtual std::pair<double, std::size_t> forward(tensor_t& y, tensor_t& t) = 0;
	virtual tensor_t backward(tensor_t& y, tensor_t& t) = 0;
};

class MeanSquared : public Loss {
public:
	std::pair<double, std::size_t> forward(tensor_t& y, tensor_t& t);
	tensor_t backward(tensor_t& y, tensor_t& t);
};

class CrossEntropy : public Loss {
public:
	std::pair<double, std::size_t> forward(tensor_t& y, tensor_t& t);
	tensor_t backward(tensor_t& y, tensor_t& t);
};

class Hinge : public Loss {
public:
	std::pair<double, std::size_t> forward(tensor_t& y, tensor_t& t);
	tensor_t backward(tensor_t& y, tensor_t& t);
};

class SquaredHinge : public Loss {
public:
	std::pair<double, std::size_t> forward(tensor_t& y, tensor_t& t);
	tensor_t backward(tensor_t& y, tensor_t& t);
};
