#include "util.hpp"
#include "activation.hpp"
#include "layer.hpp"
#include "pooling2d.hpp"

flt& AveragePooling2D::unpaddedref(vec_t& t, std::size_t ch, std::size_t y, std::size_t x) {
	assert(t.size() == this->channels * this->in_height * this->in_width);
	assert(ch < this->channels);
	assert(y < this->in_height);
	assert(x < this->in_width);
	const auto chlen = this->in_height * this->in_width;
	return t[ch*chlen + y*this->in_width + x];
}

flt& AveragePooling2D::paddedref(vec_t& t, std::size_t ch, std::size_t y, std::size_t x) {
	assert(t.size() == this->in_len);
	assert(ch < this->channels);
	assert(y < this->padded_height);
	assert(x < this->padded_width);
	const auto chlen = this->padded_height * this->padded_width;
	return t[ch*chlen + y*this->padded_width + x];
}

flt& AveragePooling2D::outputref(vec_t& t, std::size_t ch, std::size_t y, std::size_t x) {
	assert(t.size() == this->out_len);
	assert(ch < this->channels);
	assert(y < this->out_height);
	assert(x < this->out_width);
	const auto chlen = this->out_height * this->out_width;
	return t[ch*chlen + y*this->out_width + x];
}

AveragePooling2D::AveragePooling2D(
	std::size_t ch,
	std::size_t in_h, std::size_t in_w,
	std::size_t out_h, std::size_t out_w,
	std::size_t k_h, std::size_t k_w,
	std::size_t s):
	channels(ch),
	in_height(in_h),
	in_width(in_w),
	padded_height(s*(out_h-1)+k_h),
	padded_width(s*(out_w-1)+k_w),
	out_height(out_h), out_width(out_w),
	kernel_height(k_h), kernel_width(k_w),
	kernel_area(k_h*k_w),
	h_stride(s), w_stride(s),
	h_padding(s*(out_h-1)+k_h-in_h),
	w_padding(s*(out_w-1)+k_w-in_w),
	in_len(ch *
		   (s*(out_h-1)+k_h) *
		   (s*(out_w-1)+k_w)),
	out_len(ch * out_h * out_w) {
}

tensor_t AveragePooling2D::forward(tensor_t& data) {
	auto batchsize = data.size();
	this->lastdata = tensor_t(batchsize, vec_t(this->in_len, 0.0));

	// padding
#pragma omp parallel for
	for (std::size_t b = 0; b < batchsize; b++) {
		for (std::size_t ch = 0; ch < this->channels; ch++) {
			for (std::size_t y = 0; y < this->in_height; y++) {
				for (std::size_t x = 0; x < this->in_width; x++) {
					this->paddedref(this->lastdata[b], ch, y+h_padding, x+w_padding) += this->unpaddedref(data[b], ch, y, x);
				}
			}
		}
	}
	auto ret = tensor_t(batchsize, vec_t(this->out_len));
	// average pooling
#pragma omp parallel for
	for (std::size_t b = 0; b < batchsize; b++) {
		for (std::size_t ch = 0; ch < this->channels; ch++) {
			for (std::size_t y = 0; y < this->out_height; y++) {
				for (std::size_t x = 0; x < this->out_width; x++) {
					flt sum = 0.0;
					for (std::size_t ky = 0; ky < this->kernel_height; ky++) {
						for (std::size_t kx = 0; kx < this->kernel_width; kx++) {
							sum += this->paddedref(this->lastdata[b], ch, y*h_stride+ky, x*w_stride+kx);
						}
					}
					this->outputref(ret[b], ch, y, x) = sum / this->kernel_area;
				}
			}
		}
	}
	return ret;
}

tensor_t AveragePooling2D::backward(tensor_t& data) {
	assert(data.size() == this->lastdata.size());
	auto batchsize = data.size();
	auto paddedinputgrad = tensor_t(batchsize, vec_t(this->in_len, 0.0));
	auto inputgrad = tensor_t(batchsize, vec_t(this->channels * this->in_height * this->in_width));
#pragma omp parallel for
	for (std::size_t b = 0; b < batchsize; b++) {
		assert(data[b].size() == this->out_len);
		for (std::size_t ch = 0; ch < this->channels; ch++) {
			for (std::size_t y = 0; y < this->out_height; y++) {
				for (std::size_t x = 0; x < this->out_width; x++) {
					for (std::size_t ky = 0; ky < this->kernel_height; ky++) {
						for (std::size_t kx = 0; kx < this->kernel_width; kx++) {
							this->paddedref(paddedinputgrad[b], ch, y*h_stride+ky, x*w_stride+kx) += this->outputref(data[b], ch, y, x) / this->kernel_area;
						}
					}
				}
			}
		}
	}
	// unpadding
#pragma omp parallel for
	for (std::size_t b = 0; b < batchsize; b++) {
		for (std::size_t ch = 0; ch < this->channels; ch++) {
			for (std::size_t y = 0; y < this->in_height; y++) {
				for (std::size_t x = 0; x < this->in_width; x++) {
					this->unpaddedref(inputgrad[b], ch, y, x) = this->paddedref(paddedinputgrad[b], ch, y+h_padding, x+w_padding);
				}
			}
		}
	}
	return inputgrad;
}

void AveragePooling2D::update(flt learningrate) {}

flt& MaxPooling2D::unpaddedref(vec_t& t, std::size_t ch, std::size_t y, std::size_t x) {
	assert(t.size() == this->channels * this->in_height * this->in_width);
	assert(ch < this->channels);
	assert(y < this->in_height);
	assert(x < this->in_width);
	const auto chlen = this->in_height * this->in_width;
	return t[ch*chlen + y*this->in_width + x];
}

flt& MaxPooling2D::paddedref(vec_t& t, std::size_t ch, std::size_t y, std::size_t x) {
	assert(t.size() == this->in_len);
	assert(ch < this->channels);
	assert(y < this->padded_height);
	assert(x < this->padded_width);
	const auto chlen = this->padded_height * this->padded_width;
	return t[ch*chlen + y*this->padded_width + x];
}

flt& MaxPooling2D::outputref(vec_t& t, std::size_t ch, std::size_t y, std::size_t x) {
	assert(t.size() == this->out_len);
	assert(ch < this->channels);
	assert(y < this->out_height);
	assert(x < this->out_width);
	const auto chlen = this->out_height * this->out_width;
	return t[ch*chlen + y*this->out_width + x];
}

MaxPooling2D::MaxPooling2D(
	std::size_t ch,
	std::size_t in_h, std::size_t in_w,
	std::size_t out_h, std::size_t out_w,
	std::size_t k_h, std::size_t k_w,
	std::size_t s):
	channels(ch),
	in_height(in_h),
	in_width(in_w),
	padded_height(s*(out_h-1)+k_h),
	padded_width(s*(out_w-1)+k_w),
	out_height(out_h), out_width(out_w),
	kernel_height(k_h), kernel_width(k_w),
	h_stride(s), w_stride(s),
	h_padding(s*(out_h-1)+k_h-in_h),
	w_padding(s*(out_w-1)+k_w-in_w),
	in_len(ch *
		   (s*(out_h-1)+k_h) *
		   (s*(out_w-1)+k_w)),
	out_len(ch * out_h * out_w) {
}

tensor_t MaxPooling2D::forward(tensor_t& data) {
	auto batchsize = data.size();
	this->h_selected = std::vector<std::vector<std::vector<std::vector<std::size_t>>>>(
		batchsize, std::vector<std::vector<std::vector<std::size_t>>>(
			this->channels, std::vector<std::vector<std::size_t>>(
				this->out_height, std::vector<std::size_t>(
					this->out_width, 0))));
	this->w_selected = this->h_selected;
	this->lastdata = tensor_t(batchsize, vec_t(this->in_len, 0.0));
	// padding
#pragma omp parallel for
	for (std::size_t b = 0; b < batchsize; b++) {
		assert(data[b].size() == this->channels * this->in_height * this->in_width);
		for (std::size_t ch = 0; ch < this->channels; ch++) {
			for (std::size_t y = 0; y < this->in_height; y++) {
				for (std::size_t x = 0; x < this->in_width; x++) {
					this->paddedref(this->lastdata[b], ch, y+h_padding, x+w_padding) = this->unpaddedref(data[b], ch, y, x);
				}
			}
		}
	}
	auto ret = tensor_t(batchsize, vec_t(this->out_len));
	// max pooling
#pragma omp parallel for
	for (std::size_t b = 0; b < batchsize; b++) {
		for (std::size_t ch = 0; ch < this->channels; ch++) {
			for (std::size_t y = 0; y < this->out_height; y++) {
				for (std::size_t x = 0; x < this->out_width; x++) {
					flt max = this->paddedref(this->lastdata[b], ch, y*h_stride, x*w_stride);
					for (std::size_t ky = 0; ky < this->kernel_height; ky++) {
						for (std::size_t kx = 0; kx < this->kernel_width; kx++) {
							if (max < this->paddedref(this->lastdata[b], ch, y*h_stride+ky, x*w_stride+kx)) {
								max = this->paddedref(this->lastdata[b], ch, y*h_stride+ky, x*w_stride+kx);
								this->h_selected[b][ch][y][x] = ky;
								this->w_selected[b][ch][y][x] = kx;
							}
						}
					}
					this->outputref(ret[b], ch, y, x) = max;
				}
			}
		}
	}
	return ret;
}

tensor_t MaxPooling2D::backward(tensor_t& data) {
	assert(data.size() == this->lastdata.size());
	auto batchsize = data.size();
	auto paddedinputgrad = tensor_t(batchsize, vec_t(this->in_len, 0.0));
	auto inputgrad = tensor_t(batchsize, vec_t(this->channels * this->in_height * this->in_width));
#pragma omp parallel for
	for (std::size_t b = 0; b < batchsize; b++) {
		assert(data[b].size() == this->out_len);
		for (std::size_t ch = 0; ch < this->channels; ch++) {
			for (std::size_t y = 0; y < this->out_height; y++) {
				for (std::size_t x = 0; x < this->out_width; x++) {
					auto ky = this->h_selected[b][ch][y][x];
					auto kx = this->w_selected[b][ch][y][x];
					this->paddedref(paddedinputgrad[b], ch, y*h_stride+ky, x*w_stride+kx) += this->outputref(data[b], ch, y, x);
				}
			}
		}
	}
	// unpadding
#pragma omp parallel for
	for (std::size_t b = 0; b < batchsize; b++) {
		for (std::size_t ch = 0; ch < this->channels; ch++) {
			for (std::size_t y = 0; y < this->in_height; y++) {
				for (std::size_t x = 0; x < this->in_height; x++) {
					this->unpaddedref(inputgrad[b], ch, y, x) = this->paddedref(paddedinputgrad[b], ch, y+h_padding, x+w_padding);
				}
			}
		}
	}
	return inputgrad;
}

void MaxPooling2D::update(flt learningrate) {}

