#include "util.hpp"
#include "loss.hpp"

std::pair<double, std::size_t> MeanSquared::forward(tensor_t& y, tensor_t& t) {
	double error = 0.0;
	assert(y.size() == t.size());
	std::size_t cnt = 0;
	for (std::size_t b = 0; b < y.size(); b++) {
		double sum = 0.0;
		assert(y[b].size() == t[b].size());
		for (std::size_t i = 0; i < y[b].size(); i++) {
			sum += (y[b][i] - t[b][i])*(y[b][i] - t[b][i]);
		}
		sum /= y.size();
		error += sum;
		auto testidx = std::distance(t[b].begin(), std::max_element(t[b].begin(), t[b].end()));
		auto batchidx= std::distance(y[b].begin(), std::max_element(y[b].begin(), y[b].end()));
		cnt += testidx == batchidx;
	}
	return {error, cnt};
}

tensor_t MeanSquared::backward(tensor_t& y, tensor_t& t) {
	assert(y.size() == t.size());
	auto batchsize = y.size();
	auto grad = tensor_t(batchsize);
	for (std::size_t b = 0; b < batchsize; b++) {
		grad[b] = vec_t(y[b].size());
		for (std::size_t i = 0; i < y[b].size(); i++) {
			grad[b][i] = (y[b][i] - t[b][i])/batchsize;
		}
	}
	return grad;
}

std::pair<double, std::size_t> CrossEntropy::forward(tensor_t& y, tensor_t& t) {
	double error = 0.0;
	const double delta = 1e-7;
	assert(y.size() == t.size());
	std::size_t cnt = 0;
	for (std::size_t b = 0; b < y.size(); b++) {
		assert(y[b].size() == t[b].size());
		double sum = 0.0;
		for (std::size_t i = 0; i < y[b].size(); i++) {
			sum += t[b][i] * std::log(y[b][i] + delta);
		}
		sum *= -1;
		sum /= y.size();
		error += sum;
		auto testidx = std::distance(t[b].begin(), std::max_element(t[b].begin(), t[b].end()));
		auto batchidx= std::distance(y[b].begin(), std::max_element(y[b].begin(), y[b].end()));
		cnt += testidx == batchidx;
	}
	return {error, cnt};
}

tensor_t CrossEntropy::backward(tensor_t& y, tensor_t& t) {
	assert(y.size() == t.size());
	auto batchsize = y.size();
	auto grad = tensor_t(batchsize);
	for (std::size_t b = 0; b < batchsize; b++) {
		grad[b] = vec_t(y[b].size());
		for (std::size_t i = 0; i < y[b].size(); i++) {
			grad[b][i] = (y[b][i] - t[b][i])/batchsize;
		}
	}
	return grad;
}

std::pair<double, std::size_t> Hinge::forward(
	tensor_t& y, tensor_t& t) {
	assert(y.size() == t.size());
	flt loss = 0.0;
	auto batchsize = y.size();
	std::size_t cnt = 0;
	for (std::size_t b = 0; b < batchsize; b++) {
		assert(y[b].size() == t[b].size());
		for (std::size_t i = 0; i < y[b].size(); i++){
			loss += std::max(0.0, 1.0 - y[b][i] * t[b][i]);
		}
		auto testidx = std::distance(t[b].begin(), std::max_element(t[b].begin(), t[b].end()));
		auto batchidx = std::distance(y[b].begin(), std::max_element(y[b].begin(), y[b].end()));
		cnt += testidx == batchidx;
	}
	return {loss, cnt};
}

tensor_t Hinge::backward(tensor_t& y, tensor_t& t) {
	assert(y.size() == t.size());
	auto batchsize = y.size();
	auto len = y[0].size();
	auto grad = tensor_t(batchsize, vec_t(len, 0.0));
#pragma omp parallel for
	for (std::size_t b = 0; b < batchsize; b++) {
		assert(y[b].size() == t[b].size());
		for (std::size_t i = 0; i < y[b].size(); i++) {
			auto margin = y[b][i] * t[b][i];
			if (margin < 1.0) {
				grad[b][i] = -t[b][i];
			}
		}
	}
	return grad;
}


std::pair<double, std::size_t> SquaredHinge::forward(
	tensor_t& y, tensor_t& t) {
	assert(y.size() == t.size());
	flt loss = 0.0;
	auto batchsize = y.size();
	std::size_t cnt = 0;
	for (std::size_t b = 0; b < batchsize; b++) {
		assert(y[b].size() == t[b].size());
		for (std::size_t i = 0; i < y[b].size(); i++) {
			flt error = std::max(0.0, 1.0 - y[b][i] * t[b][i]);
			loss += error*error;
		}
		auto testidx = std::distance(t[b].begin(), std::max_element(t[b].begin(), t[b].end()));
		auto batchidx= std::distance(y[b].begin(), std::max_element(y[b].begin(), y[b].end()));
		cnt += testidx == batchidx;
	}
	return {loss, cnt};
}

tensor_t SquaredHinge::backward(tensor_t& y, tensor_t& t) {
	assert(y.size() == t.size());
	auto batchsize = y.size();
	auto len = y[0].size();
	auto grad = tensor_t(batchsize, vec_t(len, 0));
#pragma omp parallel for
	for (std::size_t b = 0; b < batchsize; b++) {
		assert(y[b].size() == t[b].size());
		for (std::size_t i = 0; i < y[b].size(); i++) {
			auto margin = y[b][i] * t[b][i];
			if (margin < 1.0) {
				grad[b][i] = -2.0 * t[b][i] * (1.0 - margin);
			}
		}
	}
	return grad;
}
