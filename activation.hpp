#pragma once

#include "util.hpp"

enum ActivationType {
	linear,
	sigmoid,
	softmax,
	relu
};

class Activation {
private:
public:
	virtual tensor_t forward(tensor_t& data) = 0;
	virtual tensor_t backward(tensor_t& data) = 0;
};

class Linear : public Activation {
private:
public:
	Linear();
	tensor_t forward(tensor_t& data);
	tensor_t backward(tensor_t& data);
};

class Sigmoid : public Activation {
private:
	tensor_t lastdata;
public:
	Sigmoid();
	tensor_t forward(tensor_t& data);
	tensor_t backward(tensor_t& data);
};

class SoftMax : public Activation {
private:
public:
	SoftMax();
	tensor_t forward(tensor_t& data);
	tensor_t backward(tensor_t& data);
};

class ReLU : public Activation {
private:
	tensor_t lastdata;
public:
	ReLU();
	tensor_t forward(tensor_t& data);
	tensor_t backward(tensor_t& data);
};
