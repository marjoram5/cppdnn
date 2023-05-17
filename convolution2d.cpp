#include "util.hpp"
#include "activation.hpp"
#include "layer.hpp"
#include "convolution2d.hpp"

#include <random>

inline flt& Convolution2D::unpaddedref(vec_t& t, std::size_t ch, std::size_t y, std::size_t x) {
	assert(t.size() == this->in_channels * this->in_height * this->in_width);
	assert(ch < this->in_channels);
	assert(y < this->in_height);
	assert(x < this->in_width);
	const auto chlen = this->in_height * this->in_width;
	return t[ch*chlen + y*this->in_width + x];
}

inline flt& Convolution2D::paddedref(vec_t& t, std::size_t ch, std::size_t y, std::size_t x) {
	assert(t.size() == this->in_len);
	assert(ch < this->in_channels);
	assert(y < this->padded_height);
	assert(x < this->padded_width);
	const auto chlen = this->padded_height * this->padded_width;
	return t[ch*chlen + y*this->padded_width + x];
}

inline flt& Convolution2D::outputref(vec_t& t, std::size_t ch, std::size_t y, std::size_t x) {
	assert(t.size() == this->out_len);
	assert(ch < this->out_channels);
	assert(y < this->out_height);
	assert(x < this->out_width);
	const auto chlen = this->out_height * this->out_width;
	return t[ch*chlen + y*this->out_width + x];
}

Convolution2D::Convolution2D(
	std::size_t in_ch, std::size_t out_ch,
	std::size_t in_h, std::size_t in_w,
	std::size_t out_h, std::size_t out_w,
	std::size_t f_h, std::size_t f_w,
	ActivationType act, std::size_t s):
	in_channels(in_ch),
	out_channels(out_ch),
	in_height(in_h),
	in_width(in_w),
	padded_height(s*(out_h-1)+f_h),
	padded_width(s*(out_w-1)+f_w),
	out_height(out_h),
	out_width(out_w),
	filter_height(f_h),
	filter_width(f_w),
	h_stride(s),
	w_stride(s),
	h_padding((s*(out_h-1)+f_h-in_h)/2),
	w_padding((s*(out_w-1)+f_w-in_w)/2),
	in_len(in_ch *
		   (s*(out_h-1)+f_h) *
		   (s*(out_w-1)+f_w)),
	out_len(out_ch * out_h * out_w)	{
	// initialize tensors
	this->filter = std::vector<std::vector<tensor_t>>(
		this->out_channels, std::vector<tensor_t>(
			this->in_channels, tensor_t(
				this->filter_height, vec_t(
					this->filter_width, 0.0))));
	this->bias = vec_t(this->out_channels, 0.0);

	// initialize sigma and activator
	double sigma;
	switch (act) {
	case ActivationType::sigmoid:
		//std::sqrt((flt)2.0/(this->filter_height*this->filter_width*this->in_channels));
		sigma = std::sqrt(1.0/(this->filter_height * this->filter_width));
		this->activation = std::static_pointer_cast<Activation>(std::shared_ptr<Sigmoid>(new Sigmoid()));
		break;
	case ActivationType::relu:
		sigma = std::sqrt(2.0/(this->filter_height * this->filter_width * this->in_channels));
		this->activation = std::static_pointer_cast<Activation>(std::shared_ptr<ReLU>(new ReLU()));
		break;
	default:
		sigma = 0.05;
		this->activation = std::static_pointer_cast<Activation>(std::shared_ptr<Linear>(new Linear()));
		break;
	}
	std::random_device seed;
	std::mt19937 rng(seed());
	std::normal_distribution<> normaldist(0.0, sigma);
	for (std::size_t och = 0; och < this->out_channels; och++) {
		for (std::size_t ich = 0; ich < this->in_channels; ich++) {
			for (std::size_t y = 0; y < this->filter_height; y++) {
				for (std::size_t x = 0; x < this->filter_width; x++) {
					this->filter[och][ich][y][x] = normaldist(rng);
				}
			}
		}
	}
	for (std::size_t ch = 0; ch < this->out_channels; ch++) {
		this->bias[ch] = normaldist(rng);
	}
}

tensor_t Convolution2D::forward(tensor_t& data) {
	auto batchsize = data.size();
	this->lastdata = tensor_t(batchsize, vec_t(this->in_len, 0.0));
	auto ret = tensor_t(batchsize, vec_t(this->out_len));
	// padding
#pragma omp parallel for
	for (std::size_t b = 0; b < batchsize; b++) {
		assert(data[b].size() == this->in_channels * this->in_height * this->in_width);
		for (std::size_t ch = 0; ch < this->in_channels; ch++) {
			for (std::size_t y = 0; y < this->in_height; y++) {
				for (std::size_t x = 0; x < this->in_width; x++) {
					this->paddedref(this->lastdata[b], ch, y+h_padding, x+w_padding) = this->unpaddedref(data[b], ch, y, x);
				}
			}
		}
	}
	// convolution
#pragma omp parallel for
	for (std::size_t b = 0; b < batchsize; b++) {
		for (std::size_t och = 0; och < this->out_channels; och++) {
			for (std::size_t y = 0; y < this->out_height; y++) {
				for (std::size_t x = 0; x < this->out_width; x++) {
					flt sum = 0.0;
					for (std::size_t ich = 0; ich < this->in_channels; ich++) {
						for (std::size_t ky = 0; ky < this->filter_height; ky++) {
							for (std::size_t kx = 0; kx < this->filter_width; kx++) {
								sum += this->paddedref(this->lastdata[b], ich, y*h_stride+ky, x*w_stride+kx) * this->filter[och][ich][ky][kx];
							}
						}
					}
					this->outputref(ret[b], och, y, x) = sum - bias[och];
				}
			}
		}
	}
	return this->activation->forward(ret);
}

tensor_t Convolution2D::backward(tensor_t& data) {
	data = this->activation->backward(data);
	assert(data.size() == this->lastdata.size());
	auto batchsize = data.size();
	auto paddedinputgrad = tensor_t(batchsize, vec_t(this->in_len, 0.0));
	auto inputgrad = tensor_t(batchsize, vec_t(this->in_channels*this->in_height*this->in_width));
	this->filter_grads = std::vector<std::vector<std::vector<tensor_t>>>(
		batchsize, std::vector<std::vector<tensor_t>>(
			this->out_channels, std::vector<tensor_t>(
				this->in_channels, tensor_t(
					this->filter_height, vec_t(
						this->filter_width, 0.0)))));
	this->bias_grads = tensor_t(batchsize, vec_t(this->out_channels, 0.0));
	// compute input gradients
#pragma omp parallel for
	for (std::size_t b = 0; b < batchsize; b++) {
		assert(data[b].size() == this->out_len);
		for (std::size_t och = 0; och < this->out_channels; och++) {
			for (std::size_t y = 0; y < this->out_height; y++) {
				for (std::size_t x = 0; x < this->out_width; x++) {
					flt outgrad = this->outputref(data[b], och, y, x);
					for (std::size_t ich = 0; ich < this->in_channels; ich++) {
						for (std::size_t ky = 0; ky < this->filter_height; ky++) {
							for (std::size_t kx = 0; kx < this->filter_width; kx++) {
								this->paddedref(paddedinputgrad[b], ich, y*h_stride+ky, x*w_stride+kx) += outgrad * this->filter[och][ich][ky][kx];
							}
						}
					}
				}
			}
		}
	}
#pragma omp parallel for
	for (std::size_t b = 0; b < batchsize; b++) {
		// unpadding
		for (std::size_t ich = 0; ich < this->in_channels; ich++) {
			for (std::size_t y = 0; y < this->in_height; y++) {
				for (std::size_t x = 0; x < this->in_width; x++) {
					this->unpaddedref(inputgrad[b], ich, y, x) = this->paddedref(paddedinputgrad[b], ich, y+h_padding, x+w_padding);
				}
			}
		}
	}
#pragma omp parallel for
	for (std::size_t b = 0; b < batchsize; b++) {
		for (std::size_t och = 0; och < this->out_channels; och++) {
			// compute filter gradients
			for (std::size_t ich = 0; ich < this->in_channels; ich++) {
				for (std::size_t ky = 0; ky < this->filter_height; ky++) {
					for (std::size_t kx = 0; kx < this->filter_width; kx++) {
						flt filtergrad = 0.0;
						for (std::size_t y = 0; y < this->out_height; y++) {
							for (std::size_t x = 0; x < this->out_width; x++) {
								filtergrad += this->paddedref(this->lastdata[b], ich, y*h_stride+ky, x*w_stride+kx) * this->outputref(data[b], och, y, x);
							}
						}
						this->filter_grads[b][och][ich][ky][kx] += filtergrad;
					}
				}
			}
			// compute bias gradients
			flt biasgrad = 0.0;
			for (std::size_t y = 0; y < this->out_height; y++) {
				for (std::size_t x = 0; x < this->out_width; x++) {
					this->bias_grads[b][och] += this->outputref(data[b], och, y, x);
				}
			}
		}
	}
	return inputgrad;
}

void Convolution2D::update(flt learningrate) {
	auto batchsize = this->filter_grads.size();
	for (std::size_t b = 0; b < batchsize; b++) {
		for (std::size_t och = 0; och < this->out_channels; och++) {
			for (std::size_t ich = 0; ich < this->in_channels; ich++) {
				for (std::size_t ky = 0; ky < this->filter_height; ky++) {
					for (std::size_t kx = 0; kx < this->filter_width; kx++) {
						this->filter[och][ich][ky][kx] -= learningrate * this->filter_grads[b][och][ich][ky][kx];
					}
				}
			}
			this->bias[och] -= learningrate * this->bias_grads[b][och];
		}
	}
}
