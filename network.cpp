#include "util.hpp"
#include "activation.hpp"
#include "layer.hpp"
#include "convolution2d.hpp"
#include "pooling2d.hpp"
#include "fullyconnected.hpp"
#include "network.hpp"

#include <iomanip>
#include <algorithm>

Network::Network(){}

void Network::initloss(LossType losstype) {
	switch (losstype) {
	case squared:
		this->loss = std::static_pointer_cast<Loss>(std::shared_ptr<MeanSquared>(new MeanSquared()));
		break;
	case crossentropy:
		this->loss = std::static_pointer_cast<Loss>(std::shared_ptr<CrossEntropy>(new CrossEntropy()));
		break;
	}
}

tensor_t Network::predict(tensor_t& data) {
	auto ret = data;
	for (auto& layer: this->layers) {
		ret = layer->forward(ret);
	}
	return ret;
}

void Network::backward(tensor_t& data, double learningrate) {
	auto grad = this->loss->backward(data);
	for (auto iter = this->layers.rbegin(); iter != this->layers.rend(); iter++) {
		grad = (*iter)->backward(grad);
	}
#pragma omp parallel for
	for (auto iter = this->layers.begin(); iter != this->layers.end(); iter++) {
		(*iter)->update(learningrate);
	}
}

std::vector<double> Network::fit(
	std::size_t batchsize,
	std::size_t step, flt learningrate, flt decay,
	tensor_t& x, tensor_t& y, LossType losstype) {
	this->initloss(losstype);
	std::vector<double> history;
	for (std::size_t currentstep = 0; currentstep < step; currentstep++) {
		tensor_t batchx, batchy;
		for (std::size_t i = 0; i < batchsize; i++) {
			batchx.push_back(x[(batchsize*currentstep+i)%x.size()]);
			batchy.push_back(y[(batchsize*currentstep+i)%y.size()]);
		}
		auto testy = this->predict(batchx);
		tensor_t diff(batchsize);
		std::size_t cnt = 0;
		for (std::size_t b = 0; b < batchsize; b++) {
			for (std::size_t i = 0; i < testy[b].size(); i++) {
				diff[b].push_back((flt)(testy[b][i] - batchy[b][i])/batchsize);
			}
			auto testidx = std::distance(testy[b].begin(), std::max_element(testy[b].begin(), testy[b].end()));
			auto batchidx= std::distance(batchy[b].begin(), std::max_element(batchy[b].begin(), batchy[b].end()));
			cnt += testidx == batchidx;
		}
		learningrate *= decay;
		this->backward(diff, learningrate);
		auto loss_step = this->loss->forward(testy, batchy);
		history.push_back(loss_step);
		std::cout << std::flush << "\r";
		std::cout << "current step: " << currentstep+1 << "/" << step
				  << ", lr: " << std::setprecision(10) << learningrate
				  << ", loss: " << std::fixed << std::setprecision(8) << loss_step
				  << ", accuracy: " << std::fixed << std::setprecision(2) << std::setw(6) << (double)cnt*100/batchsize << "%";
	}
	std::cout << std::endl;
	return history;
}
