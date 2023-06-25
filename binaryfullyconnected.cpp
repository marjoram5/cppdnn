#include "util.hpp"
#include "binarize.hpp"
#include "layer.hpp"
#include "binaryfullyconnected.hpp"

BinaryFullyConnected::BinaryFullyConnected(
	std::size_t in, std::size_t out) :
	inlen(in), outlen(out) {
	this->weight = tensor_t(this->outlen, vec_t(this->inlen));
	this->bias = vec_t(this->outlen);
	double sigma = std::sqrt(1.0/this->inlen);
	std::random_device seed;
	std::mt19937 rng(seed());
	std::normal_distribution<> normaldist(0.0, sigma);
	for (std::size_t i = 0; i < this->outlen; i++) {
		this->bias[i] = normaldist(rng);
		for (std::size_t j = 0; j < this->inlen; j++) {
			this->weight[i][j] = normaldist(rng);
		}
	}
}

tensor_t BinaryFullyConnected::forward(tensor_t& data) {
	auto batchsize = data.size();
	this->lastdata = data;
	auto ret = tensor_t(
		batchsize, vec_t(this->outlen, 0.0));
#pragma omp parallel for
	for (std::size_t b = 0; b < batchsize; b++) {
		assert(data[b].size() == this->inlen);
		for (std::size_t i = 0; i < this->outlen; i++) {
			for (std::size_t j = 0; j < this->inlen; j++) {
				ret[b][i] += sign(this->weight[i][j]) * data[b][j];
			}
			if constexpr(this->usebias) {
				ret[b][i] += this->bias[i];
			}
		}
	}
	return ret;
}

tensor_t BinaryFullyConnected::backward(tensor_t& data) {
	assert(data.size() == this->lastdata.size());
	auto batchsize = data.size();
	auto inputgrad = tensor_t(batchsize, vec_t(this->inlen));
	this->weightgrad = std::vector<tensor_t>(
		batchsize, tensor_t(
			this->outlen, vec_t(
				this->inlen, 0.0)));
	this->biasgrad = tensor_t(
		batchsize, vec_t(
			this->outlen, 0.0));
#pragma omp parallel for
	for (std::size_t b = 0; b < batchsize; b++) {
		assert(data[b].size() == this->outlen);
		for (std::size_t i = 0; i < this->inlen; i++) {
			for (std::size_t j = 0; j < this->outlen; j++) {
				inputgrad[b][i] += sign(this->weight[j][i]) * data[b][j];
			}
		}
	}
#pragma omp parallel for
	for (std::size_t b = 0; b < batchsize; b++) {
		for (std::size_t i = 0; i < this->outlen; i++) {
			for (std::size_t j = 0; j < this->inlen; j++) {
				this->weightgrad[b][i][j] = data[b][i] * sign(this->lastdata[b][j]);
			}
			if constexpr(this->usebias) {
				this->biasgrad[b][i] = data[b][i];
			}
		}
	}
	return inputgrad;
}

void BinaryFullyConnected::update(flt learningrate) {
	auto batchsize = this->weightgrad.size();
	for (std::size_t b = 0; b < batchsize; b++) {
		for (std::size_t i = 0; i < this->outlen; i++) {
			for (std::size_t j = 0; j < this->inlen; j++) {
				this->weight[i][j] -= learningrate * this->weightgrad[b][i][j];
			}
			if constexpr(this->usebias) {
				this->bias[i] -= learningrate * this->biasgrad[b][i];
			}
		}
	}
}
