/////////////////////////////////////////////////////////////////////////////////////////////
//Lytro Desktop��Illum�̎B�e�f�[�^���珑���o���J����Raw�摜�t�@�C���i.lfr�j����͂��C
//�摜�Z���T�i5368�~7728��f�C10�r�b�g/��f�j�̃f�[�^��ǂݏo���C
//�摜�Z���T�̐��摜��5368�~7728��f�C8�r�b�g/��f��tif�t�@�C���iIllumRaw.tif�j�Ƃ��ĕۑ�����D
//���摜�̓J���[�t�B���^�A���C�摜�ł���D
//����ɁC�P���ȃf���U�C�N�������{���CRGB�摜��tif�t�@�C���iIllumRGB.tif�j�Ƃ��Ă��ۑ�����D
//
//���s�Y�Ƒ�w�@���_
/////////////////////////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <iostream>
#include <memory>
#include <fstream>

#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#ifdef _DEBUG
#pragma comment(lib,"opencv_core2411d.lib")
#pragma comment(lib,"opencv_highgui2411d.lib")
#pragma comment(lib,"opencv_imgproc2411d.lib")
#else
#pragma comment(lib,"opencv_core2411.lib")
#pragma comment(lib,"opencv_highgui2411.lib")
#pragma comment(lib,"opencv_imgproc2411.lib")
#endif

float bayer_G[5][5] = {
	{0, 0, -1, 0, 0},
	{0, 0, 2, 0, 0},
	{-1, 2, 4, 2, -1},
	{ 0, 0, 2, 0, 0 },
	{ 0, 0, -1, 0, 0 },
};

float bayer_RBatGinRrowBcolumn[5][5] = {
	{ 0, 0, 1 / 2, 0, 0 },
	{ 0, -1, 0, -1, 0 },
	{ -1, 4, 5, 4, -1 },
	{ 0, -1, 0, -1, 0 },
	{ 0, 0, 1 / 2, 0, 0 },
};

float bayer_RBatGinBrowRcolumn[5][5] = {
	{ 0, 0, -1, 0, 0 },
	{ 0, -1, 4, -1, 0 },
	{ 1 / 2, 0, 5, 0, 1 / 2 },
	{ 0, -1, 4, -1, 0 },
	{ 0, 0, -1, 0, 0 },
};

float bayer_RBatB[5][5] = {
	{ 0, 0, -3 / 2, 0, 0 },
	{ 0, 2, 0, 2, 0 },
	{ -3 / 2, 0, 6, 0, -3 / 2 },
	{ 0, 2, 0, 2, 0 },
	{ 0, 0, -3 / 2, 0, 0 },
};

template <typename T>
T conv(const cv::Mat& mat, int x, int y, const float filter[5][5]) {
	T ret(0);
	if (!filter) {
		return mat.at<T>(y, x);
	}
	for (int dy = -2; dy <= 2; dy++) {
		for (int dx = -2; dx <= 2; dx++) {
			int x2 = std::abs(x + dx);
			int y2 = std::abs(y + dy);
			if (x2 >= mat.cols) {
				x2 = (mat.cols - 1) * 2 - x2;
			}
			if (y2 >= mat.rows) {
				y2 = (mat.cols - 1) * 2 - y2;
			}
			ret += mat.at<T>(y2, x2) * filter[dy + 2][dx + 2];
		}
	}
	return ret / 8.0;
}

int main(int argc, char* argv[]) {
	if (argc < 6) {
		std::cout << "Usage: input(lfr), output raw image, output rgb image, output raw file, output rgb file" << std::endl;
		return 0;
	}

	//Illum��Lytro�J����Raw�摜�t�@�C���̖��O�iLytro�@Desktop�������o���t�@�C���̈�j�D
	const char *file_name = argv[1];

	//�t�@�C���̃I�[�v���D
	FILE *fp = fopen(file_name, "rb");
	if (fp == NULL) {
		printf("file (%s) open error\n", file_name);
		return 1;
	}
	printf("file (%s) open success\n", file_name);

	//�t�@�C���T�C�Y���m�F����D
	fseek(fp, 0, SEEK_END);
	fpos_t fsize = 0;
	fgetpos(fp, &fsize);

	//�S�f�[�^��*lfr_raw�ɓǂݍ��ށD
	fseek(fp, 0, SEEK_SET);
	std::unique_ptr<unsigned char[]> lfr_raw(new unsigned char[fsize]);
	if (fread(lfr_raw.get(), 1, (size_t) fsize, fp) != fsize) {
		printf("file read error\n");
		fclose(fp);
		return 1;
	}
	fclose(fp);

	//�o�C�g��̓ǂݎ��ʒu�������|�C���^�D
	int ptr = 0;

	///////////////////////////////////////////////////////////////////////////////////////////////////
	//.lfr�t�@�C���͕����̃Z�N�V�����ɕ�����Ă���CJSON�ŋL�q���ꂽ�t�@�C���̑����Ȃǂ��i�[����Ă���D
	//�摜�Z���T�̐��f�[�^�͑�11�Z�N�V�����ɂ���C����́C���������𗘗p����D
	///////////////////////////////////////////////////////////////////////////////////////////////////

	//��0�Z�N�V������ǂݔ�΂��D
	//�e�Z�N�V�����̐擪��12�o�C�g�̃}�W�b�N�R�[�h�ƃZ�N�V�����̒���������4�o�C�g���i�[����Ă���D
	ptr += 16;

	//��1�Z�N�V�����`��10�Z�N�V������ǂݔ�΂�
	for (int i = 1; i <= 3; i++) {
		//�Z�N�V�����Ԃ�0���}������Ă���ꍇ������̂ŁA�������X�L�b�v����D
		while (lfr_raw[ptr] == 0) ptr++;

		//�}�W�b�N�R�[�h���X�L�b�v����D
		ptr += 12;

		//�Z�N�V�����̃o�C�g����ǂݎ��
		int len = lfr_raw[ptr] * 16777216 + lfr_raw[ptr + 1] * 65536 + lfr_raw[ptr + 2] * 256 + lfr_raw[ptr + 3];
		printf("length of section %d is %d\n", i, len);

		//4�o�C�g�̃Z�N�V�����̃o�C�g���C45�o�C�g��sha1 hash�C35�o�C�g��0�C�Z�N�V�����f�[�^�̏��Ɋi�[����Ă���D
		ptr += (4 + 45 + 35 + len);
	}

	//////////////////////////////////////////////////////////////////////////////////////////////////
	//��11�Z�N�V�����ɐ��f�[�^���i�[����Ă���C����𐶉摜�ɕϊ�����tif�摜�t�@�C���ƂƂ��ďo�͂���D
	//////////////////////////////////////////////////////////////////////////////////////////////////

	//�Z�N�V�����Ԃ�0���}������Ă���ꍇ������̂ŁA�������X�L�b�v����D
	while (lfr_raw[ptr] == 0) ptr++;

	//�}�W�b�N�R�[�h���X�L�b�v����D
	ptr += 12;

	//�Z�N�V�����̃o�C�g����ǂݎ��
	int len = lfr_raw[ptr] * 16777216 + lfr_raw[ptr + 1] * 65536 + lfr_raw[ptr + 2] * 256 + lfr_raw[ptr + 3];
	printf("length of section 4 is %d\n", len);
	ptr += (4 + 45 + 35);

	//���摜���i�[����cvMat��錾����
	int v = 3280; //���摜�̐�����f��
	int h = 3280; //���摜�̐�����f��
	cv::Mat RawImage = cv::Mat::zeros(v*h, 1, CV_32F);

	//���摜�̉�f�l��4��f�P�ʂŌv�Z����D
	for (int i = 0; i < v * h; i += 2) {
		unsigned int byte1 = lfr_raw[ptr++];
		unsigned int byte2 = lfr_raw[ptr++];
		unsigned int byte3 = lfr_raw[ptr++];

		unsigned int data1 = (byte1 << 4) + (byte2 >> 4);
		unsigned int data2 = ((byte2 & 0x0f) << 8) + byte3;

		RawImage.at<float>(i, 0) = (float) data1 * 255.0f / 4096.0f;
		RawImage.at<float>(i + 1, 0) = (float) data2 * 255.0f / 4096.0f;
	}

	//v�s�~h��Ƀ��V�F�C�v���Ctif�t�@�C���Ƃ��ĕۑ�����D
	RawImage = RawImage.reshape(1, v);
	cv::imwrite(argv[2], RawImage);
	std::ofstream ofs_raw(argv[4], std::ios::binary);
	ofs_raw.write(reinterpret_cast<const char*>(RawImage.data), sizeof(float) * v * h);
	std::cout << argv[2] << " created" << std::endl;

	//////////////////////////////////////////////////////
	//Illum�̐��摜���ȒP�ȕ��@��RGB�摜�Ƀf���U�C�N����D
	//////////////////////////////////////////////////////

	//RGB�摜���L������cvMat
	cv::Mat RGBImage = cv::Mat::zeros(v, h, CV_32FC3);

	//�e�F�M���̃Q�C���͑�8�Z�N�V������JSON�ɋL�q����Ă��鐔�l�𗘗p����D
	float r_gain = 1.0f;
	float g_gain = 1.0f;
	float b_gain = 1.0f;

//	cv::cvtColor(RawImage, RGBImage, CV_BayerBG2BGR);

	//2�~2��4��f�P�ʂŃf���U�C�N���s��
	for (int vv = 0; vv < v; vv += 2) {
		for (int hh = 0; hh < h; hh += 2) {
			float b = b_gain * RawImage.at<float>(vv, hh);
			float g = g_gain * (RawImage.at<float>(vv, hh + 1) + RawImage.at<float>(vv + 1, hh)) / 2.0f;
			float r = r_gain * RawImage.at<float>(vv + 1, hh + 1);

			RGBImage.at<cv::Vec3f>(vv, hh) = cv::Vec3f(b, g, r);
			RGBImage.at<cv::Vec3f>(vv + 1, hh) = cv::Vec3f(b, g, r);
			RGBImage.at<cv::Vec3f>(vv, hh + 1) = cv::Vec3f(b, g, r);
			RGBImage.at<cv::Vec3f>(vv + 1, hh + 1) = cv::Vec3f(b, g, r);
		}
	}

	//�f���U�C�N�摜��tif�t�@�C���Ƃ��ĕۑ�����D
	cv::imwrite(argv[3], RGBImage);
	std::ofstream ofs_rgb(argv[5], std::ios::binary);
	ofs_rgb.write(reinterpret_cast<const char*>(RGBImage.data), sizeof(float) * v * h * 3);
	std::cout << argv[3] << " created" << std::endl;

	return 0;
}