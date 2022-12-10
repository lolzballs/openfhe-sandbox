#include <openfhe.h>

#include "ndarray.h"

int main(int argc, char **argv) {
	lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS> parameters;
	auto weights = ndarray<double>::load_from_file("lr_weights.bin", { 28 * 28 });
	auto images = ndarray<uint8_t>::load_from_idx_file(".data/mnist_trimmed/"
			"t10k-images-idx3-ubyte");

	std::cout << images.shape.size() << std::endl;
	std::cout << weights[{ 600 }] << std::endl;
}
