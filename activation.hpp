#pragma once

#include "util.hpp"
#include "layer.hpp"

enum ActivationType {
	linear,
	sigmoid,
	softmax,
	relu,
	binaryact
};

class Linear : public Layer {
private:
public:
	Linear();
	tensor_t forward(tensor_t& data);
	tensor_t backward(tensor_t& data);
	void update(flt learningrate);
};

class Sigmoid : public Layer {
private:
	tensor_t lastdata;
public:
	Sigmoid();
	tensor_t forward(tensor_t& data);
	tensor_t backward(tensor_t& data);
	void update(flt learningrate);
};

class SoftMax : public Layer {
private:
	tensor_t lastdata;
public:
	SoftMax();
	tensor_t forward(tensor_t& data);
	tensor_t backward(tensor_t& data);
	void update(flt learningrate);
};

class ReLU : public Layer {
private:
	tensor_t lastdata;
public:
	ReLU();
	tensor_t forward(tensor_t& data);
	tensor_t backward(tensor_t& data);
	void update(flt learningrate);
};

class BinaryAct : public Layer {
private:
	tensor_t lastdata;
public:
	BinaryAct();
	tensor_t forward(tensor_t& data);
	tensor_t backward(tensor_t& data);
	inline void update(flt learningrate);
};
