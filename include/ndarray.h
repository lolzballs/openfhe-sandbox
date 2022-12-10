#ifndef NDARRAY_H
#define NDARRAY_H

#include <cassert>
#include <iostream>
#include <vector>


template<typename T>
class ndarray {
public:
	std::vector<std::size_t> shape;

	ndarray(const std::vector<std::size_t> &shape)
		: shape(shape) {
		const std::size_t size = calculate_size(shape);
		backing.resize(size);
	};

	ndarray(const std::vector<std::size_t> &shape,
			const std::vector<uint8_t> &bytes)
		: shape(shape) {
		const std::size_t size = calculate_size(shape);
		assert(bytes.size() == size * sizeof(T));

		backing.resize(size);
		memcpy(&backing[0], &shape[0], bytes.size());
	};

	ndarray(const std::vector<std::size_t> &shape,
			const std::vector<T> &&contents)
		: shape(shape),
		backing(std::move(contents)) {
		const std::size_t size = calculate_size(shape);
		assert(contents.size() == size);
	};

	const T &operator[](const std::vector<std::size_t> &index) const {
		return backing[flatten_idx(index)];
	}

	T &operator[](const std::vector<std::size_t> &index) {
		return backing[flatten_idx(index)];
	}

	static ndarray<T> load_from_file(std::string const& filename,
			const std::vector<std::size_t> &&shape) {
		const std::size_t size = calculate_size(shape);
		std::vector<T> data(size);
		const std::size_t size_in_bytes = size * sizeof(T);

		std::ifstream is(filename, std::ios::binary);
		is.read(reinterpret_cast<char*>(data.data()), size_in_bytes);
		/* make sure we read the expected values */
		assert(static_cast<std::size_t>(is.gcount()) == size_in_bytes);

		return ndarray(shape, std::move(data));
	}

private:
	std::vector<T> backing;

	std::size_t flatten_idx(const std::vector<std::size_t> &index) const {
		assert(index.size() == shape.size());

		std::size_t column = 1;
		std::size_t idx = 0;
		for (std::size_t i = 0; i < index.size(); i++) {
			idx += index[i] * column;
			column *= shape[i];
		}
		std::cout << idx << std::endl;
		return idx;
	}

	static std::size_t calculate_size(const std::vector<std::size_t> &shape) {
		std::size_t size = 1;
		for (std::size_t dim_size : shape) {
			size *= dim_size;
		}
		return size;
	}
};

#endif
