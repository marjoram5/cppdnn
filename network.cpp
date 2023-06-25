#include "util.hpp"
#include "layer.hpp"
#include "activation.hpp"
#include "convolution2d.hpp"
#include "pooling2d.hpp"
#include "fullyconnected.hpp"
#include "network.hpp"

Network::Network(){}

void Network::initloss(LossType losstype) {
	switch (losstype) {
	case squared:
		this->loss = std::static_pointer_cast<Loss>(std::shared_ptr<MeanSquared>(new MeanSquared()));
		break;
	case crossentropy:
		this->loss = std::static_pointer_cast<Loss>(std::shared_ptr<CrossEntropy>(new CrossEntropy()));
		break;
	case hinge:
		this->loss = std::static_pointer_cast<Loss>(std::shared_ptr<Hinge>(new Hinge()));
		break;
	case squaredhinge:
		this->loss = std::static_pointer_cast<Loss>(std::shared_ptr<SquaredHinge>(new SquaredHinge()));
		break;
	}
}

tensor_t Network::forward(tensor_t& data) {
	auto ret = data;
	for (auto& layer: this->layers) {
		ret = layer->forward(ret);
	}
	return ret;
}

void Network::backward(tensor_t& predicted, tensor_t& train, flt learningrate) {
	auto grad = this->loss->backward(predicted, train);
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
		auto predictedy = this->forward(batchx);
		this->backward(predictedy, batchy, learningrate);
		learningrate *= decay;
		auto [loss_step, cnt] = this->loss->forward(predictedy, batchy);
		history.push_back(loss_step);
		std::cout << "\r" << std::flush;
		std::cout << "current step: " << currentstep+1 << "/" << step
				  << ", lr: " << std::setprecision(10) << learningrate
				  << ", loss: " << std::fixed << std::setprecision(8) << loss_step
				  << ", accuracy: " << std::fixed << std::setprecision(2) << std::setw(6) << (double)cnt*100/batchsize << "%";
//		std::cout << std::endl;
	}
	std::cout << std::endl;
	return history;
}
