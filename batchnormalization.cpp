#include "util.hpp"
#include "layer.hpp"
#include "batchnormalization.hpp"

BatchNormalization::BatchNormalization(std::size_t l) : len(l) {
	this->gamma = vec_t(this->len, 1.0);
	this->beta = vec_t(this->len, 0.0);
	std::random_device seed;
	std::mt19937 rng(seed());
	double sigma = std::sqrt((flt)2.0/this->len);
	std::normal_distribution<> normaldist(0.0, sigma);
	for (std::size_t i = 0; i < this->len; i++) {
		this->gamma[i] += normaldist(rng);
		this->beta[i] += normaldist(rng);
	}
}

tensor_t BatchNormalization::forward(tensor_t& data) {
	auto batchsize = data.size();
	auto ret = data;
	this->lastdata = data;
	this->mu = vec_t(batchsize);
	this->var = vec_t(batchsize);
	this->sqrtvar = vec_t(batchsize);
	this->ivar = vec_t(batchsize);
	this->normx = tensor_t(batchsize, vec_t(this->len));
	this->mux = tensor_t(batchsize, vec_t(this->len));
#pragma omp parallel for
	for (std::size_t b = 0; b < batchsize; b++) {
		// compute mini-batch mean
		flt mean = 0.0;
		for (std::size_t i = 0; i < this->len; i++) {
			mean += ret[b][i];
		}
		mean /= this->len;
		this->mu[b] = mean;

		// compute mini-batch variance
		flt variance = 0.0;
		for (std::size_t i = 0; i < this->len; i++) {
			variance += (ret[b][i] - mean)*(ret[b][i] - mean);
		}
		variance /= this->len;
		this->var[b] = variance;

		// normalize
		this->sqrtvar[b] = std::sqrt(variance + this->epsilon);
		this->ivar[b] = (flt)1.0/this->sqrtvar[b];
		for (std::size_t i = 0; i < this->len; i++) {
			this->mux[b][i] = ret[b][i] - mean;
			this->normx[b][i] = this->mux[b][i]*this->ivar[b];
		}
		// scale and shift
		for (std::size_t i = 0; i < this->len; i++) {
			ret[b][i] = this->normx[b][i] * this->gamma[i] + this->beta[i];
			//ret[b][i] = this->normx[b][i];
		}
	}
	return ret;
}

tensor_t BatchNormalization::backward(tensor_t& data) {
	auto batchsize = data.size();
	auto inputgrad = data;
	this->betagrad = tensor_t(batchsize, vec_t(this->len, 0.0));
	this->gammagrad = tensor_t(batchsize, vec_t(this->len, 0.0));
#pragma omp parallel for
	for (std::size_t b = 0; b < batchsize; b++) {
		// compute beta gradient
		for (std::size_t i = 0; i < this->len; i++) {
			this->betagrad[b][i] = data[b][i];
		}
		// compute gamma gradient
		for (std::size_t i = 0; i < this->len; i++) {
			this->gammagrad[b][i] = data[b][i]*this->normx[b][i];
		}
		// compute normx gradient
		for (std::size_t i = 0; i < this->len; i++) {
			inputgrad[b][i] *= this->gamma[i];
		}
		// compute ivar gradient
		flt divar = 0.0;
		for (std::size_t i = 0; i < this->len; i++) {
			divar += this->mux[b][i] * inputgrad[b][i];
		}
		// compute mux gradient
		for (std::size_t i = 0; i < this->len; i++) {
			inputgrad[b][i] *= this->ivar[b];
		}
		flt dsqrtvar = -divar/(this->sqrtvar[b]*this->sqrtvar[b]);
		flt dvar = dsqrtvar/std::sqrt(this->var[b]+this->epsilon)/2;
		// auto dsq = vec_t(data[b].size(), dvar/data[b].size());
		flt dmu = 0.0;
		for (std::size_t i = 0; i < this->len; i++) {
			inputgrad[b][i] += dvar*this->mu[b]*2/data[b].size();
			dmu -= inputgrad[b][i];
		}
		dmu /= data[b].size();
		for (std::size_t i = 0; i < this->len; i++) {
			inputgrad[b][i] += dmu;
		}
	}
//	std::cout << inputgrad[0] << std::endl;
	return inputgrad;
}

void BatchNormalization::update(flt learningrate) {
	for (std::size_t b = 0; b < this->gammagrad.size(); b++) {
		for (std::size_t i = 0; i < this->gamma.size(); i++) {
			this->gamma[i] -= learningrate * this->gammagrad[b][i];
			this->beta[i] -= learningrate * this->betagrad[b][i];
		}
	}
//	std::cout << this->beta << std::endl;
}
