#include <openfhe.h>

int main(int argc, char **argv) {
	/* setup crypto context */
	uint32_t mult_depth = 1;
	uint32_t scale_mod_size = 50;
	uint32_t batch_size = 8;

	lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS> parameters;
	parameters.SetMultiplicativeDepth(mult_depth);
	parameters.SetScalingModSize(scale_mod_size);
	parameters.SetBatchSize(batch_size);

	lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc =
		lbcrypto::GenCryptoContext(parameters);
	cc->Enable(PKE);
	cc->Enable(KEYSWITCH);
	cc->Enable(LEVELEDSHE);
    std::cout << "CKKS scheme is using ring dimension "
		<< cc->GetRingDimension() << std::endl;

	/* key generation */
	auto keys = cc->KeyGen();
	cc->EvalMultKeysGen(keys.secretKey);
	cc->EvalRotateKeyGen(keys.secretKey, {1, -2});

	/* encoding and encryption of inputs */
    std::vector<double> x1 = {0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> x2 = {5.0, 4.0, 3.0, 2.0, 1.0, 0.75, 0.5, 0.25};

	lbcrypto::Plaintext pt1 = cc->MakeCKKSPackedPlaintext(x1);
	lbcrypto::Plaintext pt2 = cc->MakeCKKSPackedPlaintext(x2);

	std::cout << "input x1: " << pt1 << std::endl;
	std::cout << "input x2: " << pt2 << std::endl;

	auto ct1 = cc->Encrypt(keys.publicKey, pt1);
	auto ct2 = cc->Encrypt(keys.publicKey, pt2);

	auto cadd = cc->EvalAdd(ct1, ct2);
	auto csub = cc->EvalSub(ct1, ct2);
	auto cscalar = cc->EvalMult(ct1, 4.0);
	auto cmul = cc->EvalMult(ct1, ct2);
	auto crot1 = cc->EvalRotate(ct1, 1);
	auto crot2 = cc->EvalRotate(ct1, -2);

	lbcrypto::Plaintext result;

	cc->Decrypt(keys.secretKey, ct1, &result);
	result->SetLength(batch_size);
	std::cout << "x1 = " << result << std::endl;

	cc->Decrypt(keys.secretKey, cadd, &result);
	result->SetLength(batch_size);
	std::cout << "x1 + x2 = " << result << std::endl;

	cc->Decrypt(keys.secretKey, csub, &result);
	result->SetLength(batch_size);
	std::cout << "x1 - x2 = " << result << std::endl;

	cc->Decrypt(keys.secretKey, cscalar, &result);
	result->SetLength(batch_size);
	std::cout << "x1 * 4.0 = " << result << std::endl;

	cc->Decrypt(keys.secretKey, cmul, &result);
	result->SetLength(batch_size);
	std::cout << "x1 * x2 = " << result << std::endl;

	cc->Decrypt(keys.secretKey, crot1, &result);
	result->SetLength(batch_size);
	std::cout << "x1 rot 1 = " << result << std::endl;

	cc->Decrypt(keys.secretKey, crot2, &result);
	result->SetLength(batch_size);
	std::cout << "x1 rot -2 = " << result << std::endl;

	return 0;
}
