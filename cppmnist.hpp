#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <type_traits>
#include <concepts>
#include <bit>

namespace MNIST {

std::uint32_t swapendian32(std::uint32_t x) {
	return x>>24 | (x<<8)&0x00ff0000 | (x>>8)&0x0000ff00 | x<<24;
}

enum ImageType {
	basic,
	binarytrain,
	binaryinference
};

// T is the return type of Images::unflatten<imgtyp>() and Images::flatten<imgtyp>()
// when imgtype is equals to ImageType::basic or ImageType::binarytrain.
template<std::floating_point T=float, std::uint8_t threshold=128u>
class Images{
private:
	std::vector<std::vector<std::vector<std::uint8_t>>> data;
	const std::string filename;
	std::ifstream ifs;
	std::uint32_t ftype;
	std::uint32_t num;
	std::uint32_t width;
	std::uint32_t height;
	void read_header() {
		this->ifs.read((char*)&this->ftype, sizeof(this->ftype));
		this->ifs.read((char*)&this->num, sizeof(this->num));
		this->ifs.read((char*)&this->width, sizeof(this->width));
		this->ifs.read((char*)&this->height, sizeof(this->height));
		if constexpr(std::endian::native == std::endian::little) {
			this->ftype = swapendian32(this->ftype);
			this->num = swapendian32(this->num);
			this->width = swapendian32(this->width);
			this->height = swapendian32(this->height);
		}
		if (this->ftype != 0x803u) {
			std::stringstream ss;
			ss << "This file is not a MNIST images file." << std::endl;
			throw std::runtime_error(ss.str());
		}
	}
	void read_payload() {
		this->data = std::vector<std::vector<std::vector<std::uint8_t>>>(
			this->num, std::vector<std::vector<std::uint8_t>>(
				this->height, std::vector<std::uint8_t>(
					this->width)));
		for (std::size_t i = 0; i < this->num; i++) {
			for (std::size_t r = 0; r < this->height; r++) {
				for (std::size_t c = 0; c < this->width; c++) {
					this->ifs.read((char*)&this->data[i][r][c], sizeof(std::uint8_t));
				}
			}
		}
	}
public:
	Images(std::string fn) : filename(fn){
		this->ifs = std::ifstream(this->filename, std::ios::in | std::ios::binary);
		if (!this->ifs.is_open()) {
			std::stringstream ss;
			ss << "Failed to Read \"" << this->filename
			   << "\"." << std::endl;
			throw std::runtime_error(ss.str());
		}
		this->read_header();
		this->read_payload();
		this->ifs.close();
	}
	
	void dump(const std::size_t idx, std::ostream& ost=std::cout) const {
		for (std::size_t i = 0; i < this->height; i++) {
			for (std::size_t j = 0; j < this->width; j++) {
				ost << (this->data.at(idx)[i][j] < threshold ? "  " : "**");
			}
			ost << std::endl;
		}
	}

	const std::vector<std::vector<std::vector<std::uint8_t>>>& basedata() const{
		return this->data;
	}

	template<ImageType imgtyp=ImageType::basic>
	constexpr auto unflatten() const {
		if constexpr(imgtyp == ImageType::basic) {
			auto ret = std::vector<std::vector<std::vector<T>>> (
				this->num, std::vector<std::vector<T>>(
					this->height, std::vector<T>(
						this->width)));
			for (std::size_t i = 0; i < this->num; i++) {
				for (std::size_t r = 0; r < this->height; r++) {
					for (std::size_t c = 0; c < this->width; c++) {
						ret[i][r][c] = (T)this->data[i][r][c]/255;
					}
				}
			}
			return ret;
		}else if constexpr(imgtyp == ImageType::binarytrain) {
			auto ret = std::vector<std::vector<std::vector<T>>> (
				this->num, std::vector<std::vector<T>>(
					this->height, std::vector<T>(
						this->width)));
			for (std::size_t i = 0; i < this->num; i++) {
				for (std::size_t r = 0; r < this->height; r++) {
					for (std::size_t c = 0; c < this->width; c++) {
						ret[i][r][c] = this->data[i][r][c] > threshold ? 1.0f : -1.0f;
					}
				}
			}
			return ret;
		}else { // imgtyp == ImageType::binaryinference
			auto ret = std::vector<std::vector<std::vector<bool>>> (
				this->num, std::vector<std::vector<bool>>(
					this->height, std::vector<bool>(
						this->width)));
			for (std::size_t i = 0; i < this->num; i++) {
				for (std::size_t r = 0; r < this->height; r++) {
					for (std::size_t c = 0; c < this->width; c++) {
						ret[i][r][c] = this->data[i][r][c] > threshold ? true : false;
					}
				}
			}
			return ret;
		}
	}

	template<ImageType imgtyp=basic>
	constexpr auto flatten() const {
		if constexpr(imgtyp == ImageType::basic) {
			auto ret = std::vector<std::vector<T>>(
				this->num, std::vector<T>(
					this->height*this->width));
			for (std::size_t i = 0; i < this->num; i++) {
				for (std::size_t r = 0; r < this->height; r++) {
					for (std::size_t c = 0; c < this->width; c++) {
						ret[i][r*this->width+c] = (T)this->data[i][r][c]/255;
					}
				}
			}
			return ret;
		}else if constexpr(imgtyp == ImageType::binarytrain) {
			auto ret = std::vector<std::vector<T>>(
				this->num, std::vector<T>(
					this->height*this->width));
			for (std::size_t i = 0; i < this->num; i++) {
				for (std::size_t r = 0; r < this->height; r++) {
					for (std::size_t c = 0; c < this->width; c++) {
						ret[i][r*this->width+c] = this->data[i][r][c] > threshold ? 1.0f : -1.0f;
					}
				}
			}
			return ret;
		}else { // imgtyp == ImageType::binaryinference
			auto ret = std::vector<std::vector<bool>>(
				this->num, std::vector<bool>(
					this->height*this->width));
			for (std::size_t i = 0; i < this->num; i++) {
				for (std::size_t r = 0; r < this->height; r++) {
					for (std::size_t c = 0; c < this->width; c++) {
						ret[i][r*this->width+c] = this->data[i][r][c] > threshold ? true : false;
					}
				}
			}
			return ret;
		}
	}
};

// T is the return type of Labels::onehot<imgtype>() when imgtype
// is equals to ImageType::basic or ImageType::binarytrain.
template<std::floating_point T=float>
class Labels{
private:
	std::vector<std::uint8_t> data;
	const std::string filename;
	std::uint32_t ftype;
	std::uint32_t num;
	void read() {
		auto ifs = std::ifstream(this->filename, std::ios::in | std::ios::binary);
		if (!ifs.is_open()) {
			std::stringstream ss;
			ss << "Failed to read \"" << this->filename
			   << "\"." << std::endl;
			throw std::runtime_error(ss.str());
		}
		ifs.read((char*)&this->ftype, sizeof(this->ftype));
		ifs.read((char*)&this->num, sizeof(this->num));
		if constexpr(std::endian::native == std::endian::little) {
			this->ftype = swapendian32(this->ftype);
			this->num = swapendian32(this->num);
		}
		if (this->ftype != 0x801u) {
			std::stringstream ss;
			ss << "This file is not a MNIST labels file." << std::endl;
			throw std::runtime_error(ss.str());
		}
		this->data = std::vector<std::uint8_t>(this->num);
		for (std::size_t i = 0; i < this->num; i++) {
			ifs.read((char*)&this->data[i], sizeof(std::uint8_t));
		}
		ifs.close();
	}
public:
	Labels(std::string fn) : filename(fn) {
		this->read();
	}

	const std::vector<std::uint8_t>& basedata() const {
		return this->data;
	}

	template<ImageType imgtyp=basic>
	constexpr auto onehot() const {
		if constexpr(imgtyp == ImageType::basic) {
			auto ret = std::vector<std::vector<T>>(
				this->num, std::vector<T>(10, (T)0.0));
			for (std::size_t i = 0; i < this->num; i++) {
				ret[i][this->data[i]] = (T)1.0;
			}
			return ret;
		}else if constexpr(imgtyp == ImageType::binarytrain) {
			auto ret = std::vector<std::vector<T>>(
				this->num, std::vector<T>(10, (T)-1.0));
			for (std::size_t i = 0; i < this->num; i++) {
				ret[i][this->data[i]] = (T)1.0;
			}
			return ret;
		}else {
			auto ret = std::vector<std::vector<bool>>(
				this->num, std::vector<bool>(10, false));
			for (std::size_t i = 0; i < this->num; i++) {
				ret[i][this->data[i]] = true;
			}
			return ret;
		}
	}
};

} // namespace MNIST
