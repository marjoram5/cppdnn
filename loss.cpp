#include "util.hpp"
#include "loss.hpp"

#include <cmath>

double MeanSquared::forward(tensor_t& y, tensor_t& t) {
	double error = 0.0;
	assert(y.size() == t.size());
	for (std::size_t b = 0; b < y.size(); b++) {
		double sum = 0.0;
		assert(y[b].size() == t[b].size());
		for (std::size_t i = 0; i < y[b].size(); i++) {
			sum += (y[b][i] - t[b][i])*(y[b][i] - t[b][i]);
		}
		sum /= y.size()*2;
		error += sum;
	}
	return error;
}

tensor_t MeanSquared::backward(tensor_t& error) {
	return error;
}

double CrossEntropy::forward(tensor_t& y, tensor_t& t) {
	double error = 0.0;
	const double delta = 1e-7;
	assert(y.size() == t.size());
	for (std::size_t b = 0; b < y.size(); b++) {
		assert(y[b].size() == t[b].size());
		double sum = 0.0;
		for (std::size_t i = 0; i < y[b].size(); i++) {
			sum += t[b][i] * std::log(y[b][i] + delta);
		}
		sum *= -1;
		sum /= y.size();
		error += sum;
	}
	return error;
}

tensor_t CrossEntropy::backward(tensor_t& error) {
	return error;
}
