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

constexpr bool isIllum = false;
constexpr int resX = isIllum ? 7728 : 3280, resY = isIllum ? 5368 : 3280;
constexpr int channels = 3;
constexpr int RBFRadius = 2;

cv::Vec2f leftTop, rightTop, leftBottom, rightBottom, center;
cv::Vec2f lensSize;
cv::Vec2i lensNum;
cv::Vec2f pixelSize, pixelSizeSub;
float focalLength, mlaFocalLength, lensDistance;
cv::Vec3f whiteBalance(1.0f, 1.0f, 1.0f);
float intensity = 1.0f;
float refractiveIndex = 1.33f;

float nx = 0.0f, ny = 0.0f;
float depthToHousing = 1.0f;

void readParams(const std::string& paramfile) {
	cv::FileStorage fs(paramfile, cv::FileStorage::READ);
	if (!fs.isOpened()) {
		std::cerr << "Param file: " << paramfile << " not found or corrupted." << std::endl;
		return;
	}

	leftTop[0] = (float) fs["LeftTopX"];
	leftTop[1] = (float) fs["LeftTopY"];
	rightBottom[0] = (float) fs["RightBottomX"];
	rightBottom[1] = (float) fs["RightBottomY"];
	leftBottom[0] = (float) fs["LeftBottomX"];
	leftBottom[1] = (float) fs["LeftBottomY"];
	rightTop[0] = (float) fs["RightTopX"];
	rightTop[1] = (float) fs["RightTopY"];
	lensNum[0] = (int) fs["LensNumX"];
	lensNum[1] = (int) fs["LensNumY"];
	focalLength = (float) fs["FocalLength"];
	mlaFocalLength = (float) fs["MLAFocalLength"];
	lensDistance = (float) fs["LensDistance"];
	pixelSize[0] = (float) fs["PixelSizeX"];
	pixelSize[1] = (float) fs["PixelSizeY"];
	pixelSizeSub[0] = pixelSize[0] * resX / lensNum[0];
	pixelSizeSub[1] = pixelSize[1] * resY / lensNum[1];
	lensSize[0] = (rightTop[0] - leftTop[0] + rightBottom[0] - leftBottom[0]) / 2.0f / (float) lensNum[0];
	lensSize[1] = (leftBottom[1] - leftTop[1] + rightBottom[1] - rightTop[1]) / 2.0f / (float) lensNum[1];

	if (lensNum[1] % 2 == 0) {
		leftBottom[0] -= lensSize[0] / 2.0;
		rightBottom[0] += lensSize[0] / 2.0;
	}
	center = (leftTop + rightTop + leftBottom + rightBottom) / 4.0f;

	whiteBalance[0] = (float) fs["WhiteBalanceB"];
	whiteBalance[1] = (float) fs["WhiteBalanceG"];
	whiteBalance[2] = (float) fs["WhiteBalanceR"];
	intensity = (float) fs["IntensityScale"];

	refractiveIndex = (float) fs["RefractiveIndex"];

	nx = (float) fs["nx"];
	ny = (float) fs["ny"];
	depthToHousing = (float) fs["DepthToHousing"];
}

cv::Vec3f refract(const cv::Vec3f in, const cv::Vec3f normal, double n) {
	auto dot = in.dot(-normal);
	return cv::normalize((in / n - (dot + sqrt(pow(n, 2.0) + pow(dot, 2.0) - 1.0)) / n * (-normal)));
}

cv::Vec3f intersectPlane(const cv::Vec3f& origin_in, const cv::Vec3f& dir_in, const cv::Vec3f& origin_plane, const cv::Vec3f& normal_plane) {
	return origin_in - dir_in * normal_plane.dot(origin_in - origin_plane) / normal_plane.dot(dir_in);
}

template <typename T>
T gaussian(const T& a, const T& b, const T& c, const T& x) {
	return a * exp(-(x - b) * (x - b) / (2 * c * c));
}

int main(int argc, char* argv[]) {
	if (argc < 6) {
		std::cout << "Usage: input(bin), param.yml, camMat.txt, camDist.txt, output image" << std::endl;
		return 0;
	}

	readParams(argv[2]);
	std::cout << "LensSize: " << lensSize << std::endl;

	cv::Mat camMat(3, 3, CV_32F), camDist(4, 1, CV_32F);
	std::ifstream ifs_camMat(argv[3]);
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			ifs_camMat >> camMat.at<float>(i, j);
		}
	}
	camMat.at<float>(0, 0) *= (float) resX / (float) (lensNum[0] * 2);
	camMat.at<float>(0, 2) *= (float) resX / (float) (lensNum[0] * 2);
	camMat.at<float>(1, 1) *= (float) resY / (float) (lensNum[1] * 2);
	camMat.at<float>(1, 2) *= (float) resY / (float) (lensNum[1] * 2);

	std::cout << camMat.at<float>(0, 0) << "," << camMat.at<float>(0, 2) << "," << camMat.at<float>(1, 1) << "," << camMat.at<float>(1, 2) << ",";

	std::ifstream ifs_camDist(argv[4]);
	for (int i = 0; i < 4; i++) {
		ifs_camDist >> camDist.at<float>(i, 0);
	}

	cv::Mat image = cv::Mat::zeros(resY, resX, channels == 3 ? CV_32FC3 : CV_32F);
	std::ifstream ifs(argv[1], std::ios::binary);
	ifs.read(reinterpret_cast<char*>(image.data), sizeof(float) * resX * resY * channels);

	cv::Mat outImage = cv::Mat::zeros(lensNum[1], lensNum[0], channels == 3 ? CV_32FC3 : CV_32F);
	cv::Mat outImageCount = cv::Mat::zeros(lensNum[1], lensNum[0], CV_32F);
	std::vector<std::vector<cv::Vec2f>> centerList(lensNum[1]);
	cv::Mat centerMat(1, lensNum[0] * lensNum[1], CV_32FC2);
	for (int y = 0; y < lensNum[1]; y++) {
		centerList.at(y).resize(lensNum[0]);
		for (int x = 0; x < lensNum[0]; x++) {
			float u = (float) x / (float) (lensNum[0] - 1), v = (float) y / (float) (lensNum[1] - 1);
			auto pos = (1.0f - u) * (1.0f - v) * leftTop + u * (1.0f - v) * rightTop + (1.0f - u) * v * leftBottom + u * v * rightBottom;
			if (y % 2 == 1) {
				pos[0] += lensSize[0] / 2.0f;
			}
			centerList.at(y).at(x) = pos;
			centerMat.at<cv::Vec2f>(0, lensNum[0] * y + x) = pos;
		}
	}

	cv::Mat undistCenterMat;
	cv::undistortPoints(centerMat, undistCenterMat, camMat, camDist, cv::noArray(), camMat);
	std::vector<std::vector<cv::Vec2f>> undistCenterList(lensNum[1]);
	for (int y = 0; y < lensNum[1]; y++) {
		undistCenterList.at(y).resize(lensNum[0]);
		for (int x = 0; x < lensNum[0]; x++) {
			undistCenterList.at(y).at(x) = undistCenterMat.at<cv::Vec2f>(0, lensNum[0] * y + x);
		}
	}

	for (int y = 0; y < lensNum[1]; y++) {
		for (int x = 0; x < lensNum[0]; x++) {
			const cv::Vec2f& centerPos = centerList.at(y).at(x);
			if (centerPos[0] < RBFRadius || resX <= centerPos[0] - RBFRadius || centerPos[1] < RBFRadius || resY <= centerPos[1] - RBFRadius) {
				continue;
			}

			const auto& undistPos = undistCenterList.at(y).at(x);

			// (x, y)に対応する3次元位置(主レンズ中心が(0, 0, 0), 主レンズから外側が正の座標系)
			cv::Vec3f startPos((undistPos[0] - center[0]) * pixelSize[0], (undistPos[1] - center[1]) * pixelSize[1], -lensDistance);
			// 屈折面上での3次元位置
			auto housingPos = intersectPlane(cv::Vec3f(0.0f, 0.0f, 0.0f), cv::normalize(startPos), cv::Vec3f(0.0f, 0.0f, depthToHousing), cv::normalize(cv::Vec3f(nx, ny, 1.0f)));
			//			auto housingPos = startPos * depthToHousing / startPos[2];
						// 実際に入ってきている光線
			auto realRay = refract(cv::normalize(housingPos), cv::normalize(cv::Vec3f(nx, ny, 1.0f)), refractiveIndex);
			//			auto realRay = refract(cv::normalize(housingPos), cv::normalize(cv::Vec3f(0.0f, 0.0f, 1.0f)), refractiveIndex);
						// 実際に入ってきている光線のアパーチャ内の位置
			auto apertPos = housingPos - realRay * housingPos[2] / realRay[2];
			// 実際に入ってきている光線の合焦位置
			auto focusPoint = -realRay * focalLength / realRay[2];
			// 入射ベクトル
			auto imageRay = focusPoint - apertPos;
			// 中心を通る入射位置
			auto projectPoint1 = -realRay * (lensDistance + mlaFocalLength) / realRay[2];
			// 中心を通らない入射位置
			auto projectPoint2 = apertPos - imageRay * (lensDistance + mlaFocalLength) / imageRay[2];

			cv::Vec2f imagePos(projectPoint2[0] / pixelSizeSub[0] + lensNum[0] / 2, projectPoint2[1] / pixelSizeSub[1] + lensNum[1] / 2);

			if (std::floor(imagePos[0]) < RBFRadius || lensNum[0] - RBFRadius <= std::ceil(imagePos[0]) || std::floor(imagePos[1]) < RBFRadius || lensNum[1] - RBFRadius <= std::ceil(imagePos[1])) {
				continue;
			}

			// Direct
//			outImage.at<cv::Vec3f>(imagePos[1], imagePos[0]) += image.at<cv::Vec3f>(centerList.at(y).at(x)[1], centerList.at(y).at(x)[0]);
//			outImageCount.at<uchar>(imagePos[1], imagePos[0])++;

			// RBF
			float sigma = (float) RBFRadius / 3.0f;
			for (int y = -RBFRadius; y <= RBFRadius; y++) {
				for (int x = -RBFRadius; x <= RBFRadius; x++) {
					float scale = gaussian(1.0f / (sigma * sqrtf(2.0f * CV_PI)), imagePos[1], sigma, imagePos[1] + y)
						* gaussian(1.0f / (sigma * sqrtf(2.0f * CV_PI)), imagePos[0], sigma, imagePos[0] + x);
					if (channels == 3) {
						outImage.at<cv::Vec3f>(imagePos[1] + y, imagePos[0] + x) += image.at<cv::Vec3f>(centerPos[1] + y, centerPos[0] + x) * scale;
					} else {
						outImage.at<float>(imagePos[1] + y, imagePos[0] + x) += image.at<float>(centerPos[1] + y, centerPos[0] + x) * scale;
					}
					outImageCount.at<float>(imagePos[1] + y, imagePos[0] + x) += scale;
				}
			}

			// inverse bilinear interpolation
/*			float u = imagePos[0] - std::floor(imagePos[0]), v = imagePos[1] - std::floor(imagePos[1]);
			if (channels == 3) {
				outImage.at<cv::Vec3f>(std::floor(imagePos[1]), std::floor(imagePos[0])) += image.at<cv::Vec3f>(centerList.at(y).at(x)[1], centerList.at(y).at(x)[0]) * (1.0f - u) * (1.0f - v);
				outImage.at<cv::Vec3f>(std::floor(imagePos[1]), std::ceil(imagePos[0])) += image.at<cv::Vec3f>(centerList.at(y).at(x)[1], centerList.at(y).at(x)[0]) * u * (1.0f - v);
				outImage.at<cv::Vec3f>(std::ceil(imagePos[1]), std::floor(imagePos[0])) += image.at<cv::Vec3f>(centerList.at(y).at(x)[1], centerList.at(y).at(x)[0]) * (1.0f - u) * v;
				outImage.at<cv::Vec3f>(std::ceil(imagePos[1]), std::ceil(imagePos[0])) += image.at<cv::Vec3f>(centerList.at(y).at(x)[1], centerList.at(y).at(x)[0]) * u * v;
			} else {
				outImage.at<float>(std::floor(imagePos[1]), std::floor(imagePos[0])) += image.at<float>(centerList.at(y).at(x)[1], centerList.at(y).at(x)[0]) * (1.0f - u) * (1.0f - v);
				outImage.at<float>(std::floor(imagePos[1]), std::ceil(imagePos[0])) += image.at<float>(centerList.at(y).at(x)[1], centerList.at(y).at(x)[0]) * u * (1.0f - v);
				outImage.at<float>(std::ceil(imagePos[1]), std::floor(imagePos[0])) += image.at<float>(centerList.at(y).at(x)[1], centerList.at(y).at(x)[0]) * (1.0f - u) * v;
				outImage.at<float>(std::ceil(imagePos[1]), std::ceil(imagePos[0])) += image.at<float>(centerList.at(y).at(x)[1], centerList.at(y).at(x)[0]) * u * v;
			}*/
		}
	}

	// 明度調整
	if (channels == 3) {
		for (int y = 0; y < lensNum[1]; y++) {
			for (int x = 0; x < lensNum[0]; x++) {
				outImage.at<cv::Vec3f>(y, x) /= outImageCount.at<float>(y, x);
			}
		}
	} else {
		outImage /= outImageCount;
	}

	cv::Mat outImage2 = cv::Mat::zeros(lensNum[1], lensNum[0] * 2, channels == 3 ? CV_32FC3 : CV_32F);
	for (int y = 0; y < lensNum[1] / 2; y++) {
		for (int x = 0; x < lensNum[0] - 1; x++) {
			if (channels == 3) {
				const auto& a = outImage.at<cv::Vec3f>(y * 2, x).mul(whiteBalance) * intensity;
				const auto& b = outImage.at<cv::Vec3f>(y * 2, x + 1).mul(whiteBalance) * intensity;
				const auto& c = outImage.at<cv::Vec3f>(y * 2 + 1, x).mul(whiteBalance) * intensity;
				const auto& d = outImage.at<cv::Vec3f>(y * 2 + 1, x + 1).mul(whiteBalance) * intensity;

				outImage2.at<cv::Vec3f>(y * 2, x * 2) = a;
				outImage2.at<cv::Vec3f>(y * 2, x * 2 + 1) = (a + b) / 2;
				outImage2.at<cv::Vec3f>(y * 2 + 1, x * 2 + 1) = c;
				outImage2.at<cv::Vec3f>(y * 2 + 1, x * 2 + 2) = (c + d) / 2;
			} else {
				const auto& a = outImage.at<float>(y * 2, x) * intensity;
				const auto& b = outImage.at<float>(y * 2, x + 1) * intensity;
				const auto& c = outImage.at<float>(y * 2 + 1, x) * intensity;
				const auto& d = outImage.at<float>(y * 2 + 1, x + 1) * intensity;

				outImage2.at<float>(y * 2, x * 2) = a;
				outImage2.at<float>(y * 2, x * 2 + 1) = (a + b) / 2;
				outImage2.at<float>(y * 2 + 1, x * 2 + 1) = c;
				outImage2.at<float>(y * 2 + 1, x * 2 + 2) = (c + d) / 2;
			}
		}
	}

	cv::Mat ParallaxImage = cv::Mat((int) (lensNum[1] * sqrt(3.0) + 0.5), lensNum[0] * 2, channels == 3 ? CV_32FC3 : CV_32F);
	cv::resize(outImage2, ParallaxImage, ParallaxImage.size(), 0, 0, cv::INTER_CUBIC);

	cv::imwrite(argv[5], ParallaxImage);

	return 0;
}
