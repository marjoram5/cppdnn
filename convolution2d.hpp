#pragma once

#include "util.hpp"
#include "activation.hpp"
#include "layer.hpp"

class Convolution2D : public Layer {
private:
	const std::size_t in_len;
	const std::size_t out_len;
	const std::size_t in_channels;
	const std::size_t out_channels;
	const std::size_t in_height;
	const std::size_t in_width;
	const std::size_t padded_height;
	const std::size_t padded_width;
	const std::size_t out_height;
	const std::size_t out_width;
	const std::size_t filter_height;
	const std::size_t filter_width;
	const std::size_t h_padding;
	const std::size_t w_padding;
	const std::size_t h_stride;
	const std::size_t w_stride;
	// filter = (out_channels * in_channels * filter_height * filter_width)
	std::vector<std::vector<tensor_t>> filter;
	std::vector<std::vector<std::vector<tensor_t>>> filtergrad;
	// bias = (out_channels)
	vec_t bias;
	tensor_t biasgrad;
	// reference wrappers
	inline flt& unpaddedref(vec_t& t, std::size_t ch, std::size_t y, std::size_t x);
	inline flt& paddedref(vec_t& t, std::size_t ch, std::size_t y, std::size_t x);
	inline flt& outputref(vec_t& t, std::size_t ch, std::size_t y, std::size_t x);
public:
	tensor_t lastdata;
	Convolution2D(std::size_t in_ch, std::size_t out_ch,
				  std::size_t in_h, std::size_t in_w,
				  std::size_t out_h, std::size_t out_w,
				  std::size_t f_h, std::size_t f_w,
				  ActivationType act, std::size_t s=1);
	tensor_t forward(tensor_t& data);
	tensor_t backward(tensor_t& data);
	void update(flt learningrate);
};
