#include <bit>
#include <optional>

#include <openfhe.h>

#include "ndarray.h"

const std::size_t features_size = 28 * 28;

typedef lbcrypto::Ciphertext<lbcrypto::DCRTPoly> Ciphertext;
typedef lbcrypto::CryptoContext<lbcrypto::DCRTPoly> CryptoContext;

static Ciphertext inner_product(const CryptoContext &cc,
		const Ciphertext &a, const Ciphertext &b) {
	auto mult = cc->EvalMult(a, b);
	return cc->EvalSum(mult, features_size);
}

static Ciphertext predict(const CryptoContext &cc, const Ciphertext &features,
		const Ciphertext &weights, const Ciphertext &bias,
		std::optional<Ciphertext*> intermediate = std::nullopt) {
	auto dot = inner_product(cc, features, weights);
	if (intermediate.has_value())
		**intermediate = dot;

	/* apply sigmoid using a least-squares approximation,
	 * degree-5 is picked to preserve the monotonicity of sigmoid
	 * coefficients from: https://doi.org/10.1186/s12920-018-0401-7 */
	const std::vector<double> coeffs = {
		0.5,							/* x^0 */
		1.53048 / 8,					/* x^1 */
		0,								/* x^2 */
		-2.3533056 / (8 * 8 * 8),		/* x^3 */
		0,								/* x^4 */
		1.3511295 / (8 * 8 * 8 * 8),	/* x^5 */
	};
	return cc->EvalPoly(cc->EvalAdd(dot, bias), coeffs);
}

template<typename T>
static std::vector<T> span_to_vector(std::span<const T> span) {
	std::vector<T> result;
	result.reserve(span.size());
	result.insert(result.end(), span.begin(), span.end());
	return result;
}

static std::vector<double> pad_images(std::span<const double> images,
		std::size_t batch_size, std::size_t slots) {
	std::vector<double> result;
	result.reserve(batch_size * slots);
	for (std::size_t i = 0; i < batch_size; i++) {
		const auto subspan = images.subspan(i * features_size, features_size);
		result.insert(result.end(), subspan.begin(), subspan.end());
		/* fill the rest with zeros */
		result.resize((i + 1) * slots);
	}
	return result;
}

static std::vector<double> pad_weights(std::span<const double> weights,
		std::size_t batch_size, std::size_t slots) {
	std::vector<double> result;
	result.reserve(batch_size * slots);
	for (std::size_t i = 0; i < batch_size; i++) {
		result.insert(result.end(), weights.begin(), weights.end());
		/* fill the rest with zeros */
		result.resize((i + 1) * slots);
	}
	return result;
}

static uint32_t ceil_log2(uint32_t value) {
	return 32 - std::countl_zero(value);
}

static void print_usage(char *program) {
	fprintf(stderr,
			"usage: %s <batch|single> [args]>\n"
			"	batch <batch_size>\n"
			"	single [idx]\n"
			, program);
}

int main(int argc, char **argv) {
	if (argc != 2 && argc != 3) {
		print_usage(argv[0]);
		return -1;
	}

	const char *mode = argv[1];
	uint32_t batch_size = 1;
	if (strcmp(mode, "single") == 0) {
		batch_size = 1;
	} else if (strcmp(mode, "batch") == 0) {
		if (argc != 3) {
			print_usage(argv[0]);
			return -1;
		}

		batch_size = std::atoi(argv[2]);
	} else {
		print_usage(argv[0]);
		return -1;
	}

	const auto weights = ndarray<double>::load_from_file("lr_weights.bin",
			{ 28 * 28 + 1 });
	const auto images = ndarray<uint8_t>::load_from_idx_file(".data/mnist_trimmed/"
			"t10k-images-idx3-ubyte").to<double>();

	/* setup crypto context */
	const uint32_t mult_depth = 4;
	const uint32_t scale_mod_size = 50;
	const uint32_t pt_batch_size = 1 << ceil_log2(batch_size * features_size);
	const uint32_t slots_per_image = pt_batch_size / batch_size;
	std::cout
		<< "used slots per image: " << 28 * 28 << std::endl
		<< "number of slots per image: " << slots_per_image << std::endl
		<< "fraction of slots used: "
		<< static_cast<double>(features_size * batch_size) / batch_size
		<< std::endl;

	lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS> parameters;
	parameters.SetMultiplicativeDepth(mult_depth);
	parameters.SetScalingModSize(scale_mod_size);
	parameters.SetBatchSize(pt_batch_size);

	lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc =
		lbcrypto::GenCryptoContext(parameters);
	cc->Enable(PKE);
	cc->Enable(KEYSWITCH);
	cc->Enable(LEVELEDSHE);
	cc->Enable(ADVANCEDSHE);
    std::cout << "CKKS scheme is using ring dimension "
		<< cc->GetRingDimension() << std::endl;

	/* key generation */
	const auto keys = cc->KeyGen();
	std::cout << "generating multiplication keys" << std::endl;
	cc->EvalMultKeysGen(keys.secretKey);
	cc->EvalSumKeyGen(keys.secretKey);

	std::cout << "encoding weights" << std::endl;
	const auto weight_span = weights[std::nullopt];
	const lbcrypto::Plaintext ptw = cc->MakeCKKSPackedPlaintext(
			pad_weights(weight_span.subspan(0, weight_span.size() - 1),
				batch_size, slots_per_image));
	const lbcrypto::Plaintext ptb = cc->MakeCKKSPackedPlaintext(
			span_to_vector(weight_span.subspan(weight_span.size() - 1, 1)));

	std::cout << "encrypting weights" << std::endl;
	const Ciphertext ctw = cc->Encrypt(keys.publicKey, ptw);
	const Ciphertext ctb = cc->Encrypt(keys.publicKey, ptb);

	/* single with idx */
	if (batch_size == 1 && argc == 3) {
		std::size_t image_idx = std::strtoull(argv[2], NULL, 10);
		const auto image = images[image_idx];
		std::cout << "encoding image" << std::endl;
		const lbcrypto::Plaintext pti = cc->MakeCKKSPackedPlaintext(
				span_to_vector(image));
		std::cout << "encrypting image" << std::endl;
		const Ciphertext cti = cc->Encrypt(keys.publicKey, pti);

		std::cout << "predicting..." << std::endl;
		Ciphertext before_sigmoid;
		const Ciphertext prediction = predict(cc, cti, ctw, ctb,
				&before_sigmoid);

		lbcrypto::Plaintext pt_before_sigmoid;
		cc->Decrypt(before_sigmoid, keys.secretKey, &pt_before_sigmoid);
		std::cout << "before sigmoid "
			<< pt_before_sigmoid->GetRealPackedValue()[0] << std::endl;

		lbcrypto::Plaintext pt_prediction;
		cc->Decrypt(prediction, keys.secretKey, &pt_prediction);
		std::cout << "prediction "
			<< pt_prediction->GetRealPackedValue()[0] << std::endl;
	} else {
		const auto labels = ndarray<uint8_t>::load_from_idx_file(
				".data/mnist_trimmed/t10k-labels-idx1-ubyte");
		std::size_t correct = 0;

		std::size_t num_batches = (images.shape[0] + batch_size - 1)
			/ batch_size;
		for (std::size_t i = 0; i < num_batches; i++) {
			const auto batch_range = std::make_pair(i * batch_size, batch_size);
			const auto batch_x = images[batch_range];
			const auto batch_y = labels[batch_range];

			const auto padded_batch_x = pad_images(batch_x, batch_size, slots_per_image);

			std::cout << "predicting images " << batch_range.first << " to "
				<< batch_range.first + batch_range.second << std::endl;
			const lbcrypto::Plaintext pti = cc->MakeCKKSPackedPlaintext(padded_batch_x);
			const Ciphertext cti = cc->Encrypt(keys.publicKey, pti);

			const Ciphertext prediction = predict(cc, cti, ctw, ctb);

			lbcrypto::Plaintext pt_prediction;
			cc->Decrypt(prediction, keys.secretKey, &pt_prediction);
			const auto prediction_decoded = pt_prediction->GetRealPackedValue();
			for (std::size_t j = 0; j < batch_size; j++) {
				std::cout << prediction_decoded[j * slots_per_image] << std::endl;
				if ((prediction_decoded[j * slots_per_image] >= 0.5) == (batch_y[j] == 3))
					correct++;
			}
		}

		std::cout << "accuracy: "
			<< static_cast<double>(correct) / images.shape[0] << std::endl;
	}

	return 0;
}
