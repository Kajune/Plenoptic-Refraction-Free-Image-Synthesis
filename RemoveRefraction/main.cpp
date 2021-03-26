#include <iostream>
#include <memory>
#include <fstream>
#include <numeric>

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

constexpr bool isIllum = true;
constexpr int resX = isIllum ? 7728 : 3280, resY = isIllum ? 5368 : 3280;
constexpr int channels = 3;

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
float imageScale = 1.0f;

int usePixels = 64;

cv::Vec2f trans(0.0f, 0.0f);

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

	imageScale = (float) fs["ImageScale"];

	usePixels = (int) fs["UsePixels"];

	trans[0] = -(float) fs["TransX"] / 2;
	trans[1] = -(float) fs["TransY"] / sqrt(3.0) + 0.5;
}

cv::Vec3f refract(const cv::Vec3f in, const cv::Vec3f normal, double n) {
	auto dot = in.dot(-normal);
	return cv::normalize((in / n - (dot + sqrt(pow(n, 2.0) + pow(dot, 2.0) - 1.0)) / n * (-normal)));
}

cv::Vec3f intersectPlane(const cv::Vec3f& origin_in, const cv::Vec3f& dir_in, const cv::Vec3f& origin_plane, const cv::Vec3f& normal_plane) {
	return origin_in - dir_in * normal_plane.dot(origin_in - origin_plane) / normal_plane.dot(dir_in);
}

cv::Vec2f distortPoint(const cv::Vec3f& pos, const cv::Mat& distCoef) {
	float x = pos[0] / pos[2];
	float y = pos[1] / pos[2];
	float r2 = x * x + y * y;

	float k1 = distCoef.at<float>(0);
	float k2 = distCoef.at<float>(1);
	float k3 = distCoef.at<float>(4);
	float p1 = distCoef.at<float>(2);
	float p2 = distCoef.at<float>(3);

	float k = (1.0f + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2);
	float x_ = x * k + 2.0f * p1 * x * y + p2 * (r2 + 2.0f * x * x);
	float y_ = y * k + 2.0f * p2 * x * y + p1 * (r2 + 2.0f * y * y);

	return cv::Vec2f(x_, y_);
}

int main(int argc, char* argv[]) {
	if (argc < 6) {
		std::cout << "Usage: input(bin), param.yml, camMat.txt, camDist.txt, output image, apertureMask (=1)" << std::endl;
		return 0;
	}

	bool apertureMask = true;
	if (argc >= 7) {
		apertureMask = std::stoi(argv[6]);
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

	std::ifstream ifs_camDist(argv[4]);
	for (int i = 0; i < 4; i++) {
		ifs_camDist >> camDist.at<float>(i, 0);
	}

	cv::Mat image = cv::Mat::zeros(resY, resX, channels == 3 ? CV_32FC3 : CV_32F);
	std::ifstream ifs(argv[1], std::ios::binary);
	ifs.read(reinterpret_cast<char*>(image.data), sizeof(float) * resX * resY * channels);

	cv::Mat outImage = cv::Mat::zeros(lensNum[1] * imageScale, lensNum[0] * imageScale, channels == 3 ? CV_32FC3 : CV_32F);
	std::vector<std::vector<cv::Vec2f>> centerList(lensNum[1]);
//	cv::Mat centerMat(1, lensNum[0] * lensNum[1], CV_32FC2);
	for (int y = 0; y < lensNum[1]; y++) {
		centerList.at(y).resize(lensNum[0]);
		for (int x = 0; x < lensNum[0]; x++) {
			float u = (float) x / (float) (lensNum[0] - 1), v = (float) y / (float) (lensNum[1] - 1);
			auto pos = (1.0f - u) * (1.0f - v) * leftTop + u * (1.0f - v) * rightTop + (1.0f - u) * v * leftBottom + u * v * rightBottom;
			if (y % 2 == 1) {
				pos[0] += lensSize[0] / 2.0f;
			}
			centerList.at(y).at(x) = pos;
//			centerMat.at<cv::Vec2f>(0, lensNum[0] * y + x) = pos;
		}
	}

	/*
	cv::Mat undistCenterMat;
	cv::undistortPoints(centerMat, undistCenterMat, camMat, camDist, cv::noArray(), camMat);
	std::vector<std::vector<cv::Vec2f>> undistCenterList(lensNum[1]);
	for (int y = 0; y < lensNum[1]; y++) {
		undistCenterList.at(y).resize(lensNum[0]);
		for (int x = 0; x < lensNum[0]; x++) {
			undistCenterList.at(y).at(x) = undistCenterMat.at<cv::Vec2f>(0, lensNum[0] * y + x);
		}
	}

	cv::Mat undistImage;
	cv::undistort(image, undistImage, camMat, camDist, camMat);
	*/
	for (int y = 0; y < lensNum[1] * imageScale; y++) {
		printf("\r%3.1f%%", y * 100.0f / (lensNum[1] * imageScale));
		for (int x = 0; x < lensNum[0] * imageScale; x++) {
			// (x, y)に対応する3次元位置(主レンズ中心が(0, 0, 0), 主レンズから外側が正の座標系)
			cv::Vec3f startPos(((x + trans[0]) / imageScale - lensNum[0] / 2) * pixelSizeSub[0], ((y + trans[1]) / imageScale - lensNum[1] / 2) * pixelSizeSub[1], -(lensDistance + mlaFocalLength));
			// 屈折面上での3次元位置
			auto housingPos = intersectPlane(cv::Vec3f(0.0f, 0.0f, 0.0f), cv::normalize(startPos), cv::Vec3f(0.0f, 0.0f, depthToHousing), cv::normalize(cv::Vec3f(nx, ny, 1.0f)));
			// 実際に入ってきている光線
			auto realRay = refract(cv::normalize(housingPos), cv::normalize(cv::Vec3f(nx, ny, 1.0f)), 1.0f / refractiveIndex);
			// 実際に入ってきている光線のアパーチャ内の位置
			auto apertPos = housingPos - realRay * housingPos[2] / realRay[2];
			if (cv::norm(apertPos) > focalLength / 2.0 && apertureMask) {
				continue;
			}
			// 実際に入ってきている光線の合焦位置
			auto focusPoint = -realRay * focalLength / realRay[2];
			// 入射ベクトル
			auto imageRay = focusPoint - apertPos;
			// 中心を通らない入射位置
			auto projectPoint = apertPos - imageRay * lensDistance / imageRay[2];

			// 歪みを加えることで画像平面上での位置を正しくする
			auto distPoint = distortPoint(projectPoint, camDist);
			cv::Vec2f imagePos(distPoint[0] * (-lensDistance) / pixelSize[0] + center[0], distPoint[1] * (-lensDistance) / pixelSize[1] + center[1]);

			// 最近傍のレンズを探す
			auto imagePosSub = cv::Vec2f(imagePos[0] / resX * lensNum[0], imagePos[1] / resY * lensNum[1]);
			float minDistance = FLT_MAX;
			cv::Vec2i minIndex(imagePosSub[0], imagePosSub[1]);
			for (int dy = -1; dy <= 1; dy++) {
				for (int dx = -1; dx <= 1; dx++) {
					if (imagePosSub[1] + dy < 0 || lensNum[1] <= imagePosSub[1] + dy ||
						imagePosSub[0] + dx < 0 || lensNum[0] <= imagePosSub[0] + dx) {
						continue;
					}
					float distance = cv::norm(imagePos - centerList[imagePosSub[1] + dy][imagePosSub[0] + dx]);
					if (distance < minDistance) {
						minDistance = distance;
						minIndex[0] = imagePosSub[0] + dx;
						minIndex[1] = imagePosSub[1] + dy;
					}
				}
			}

			// 最近傍のレンズが閾値より遠ければ見えていないものとして扱う
			if (minDistance > std::min(lensSize[0], lensSize[1]) / 2 + 3) {
				continue;
			}

			// 最近傍のレンズとその周辺で求めたい光線の角度と入射位置に対する誤差を求める
			// 必要な光線ベクトル: realRay
			// 入射位置: housingPos
			constexpr int lensSearchRange = 0;
			constexpr float lambda = 1.0f;
			std::vector<std::pair<cv::Vec2f, float>> mlaErrorList;	
			for (int my = std::max(minIndex[1] - lensSearchRange, 0); my <= std::min(minIndex[1] + lensSearchRange, lensNum[1] - 1); my++) {
				for (int mx = std::max(minIndex[0] - lensSearchRange, 0); mx <= std::min(minIndex[0] + lensSearchRange, lensNum[0] - 1); mx++) {
					const auto lc = centerList.at(my).at(mx);
					const cv::Vec3f lcReal((lc[0] - center[0]) * pixelSize[0], (lc[1] - center[1]) * pixelSize[0], -lensDistance);
					for (int dy = std::max(int(lc[1] - lensSize[1] / 2), 0); dy <= std::min(int(lc[1] + lensSize[1] / 2), resY - 1); dy++) {
						for (int dx = std::max(int(lc[0] - lensSize[0] / 2), 0); dx <= std::min(int(lc[0] + lensSize[0] / 2), resX - 1); dx++) {
							if (cv::norm(cv::Vec2f(dx, dy) - lc) > std::min(lensSize[0], lensSize[1]) / 2 - 1) {
								continue;
							}
							const cv::Vec3f startPos((dx - center[0]) * pixelSize[0], (dy - center[1]) * pixelSize[0], -(lensDistance + mlaFocalLength));
							const auto mlaRay = lcReal - startPos;
							const auto apertPos = lcReal + mlaRay * lensDistance / mlaRay[2];
							if (focalLength == lensDistance) {
								const auto housingPosMLA = intersectPlane(apertPos, mlaRay * focalLength / mlaRay[2] - apertPos, cv::Vec3f(0.0f, 0.0f, depthToHousing), cv::normalize(cv::Vec3f(nx, ny, 1.0f)));
								float error = cv::norm(housingPosMLA - housingPos) * lambda + FLT_EPSILON;
								mlaErrorList.emplace_back(std::make_pair(cv::Vec2f(dx, dy), 1.0f / error));
							} else {
								const auto outRay = cv::normalize(mlaRay * focalLength / mlaRay[2] - apertPos);
								const auto housingPosMLA = intersectPlane(apertPos, outRay, cv::Vec3f(0.0f, 0.0f, depthToHousing), cv::normalize(cv::Vec3f(nx, ny, 1.0f)));
								float error = abs(acos(std::min(std::min(1.0f, realRay.dot(outRay)), -1.0f))) + cv::norm(housingPosMLA - housingPos) * lambda + FLT_EPSILON;
								mlaErrorList.emplace_back(std::make_pair(cv::Vec2f(dx, dy), 1.0f / error));
							}
						}
					}
				}
			}
			
			std::sort(mlaErrorList.begin(), mlaErrorList.end(), [] (const auto& lhs, const auto& rhs) { return lhs.second > rhs.second; });
			float sumInvError = 0.0f;
			for (int i = 0; i < usePixels; i++) {
				const auto& pos = mlaErrorList.at(i);
				if (channels == 3) {
					outImage.at<cv::Vec3f>(y, x) += image.at<cv::Vec3f>(pos.first[1], pos.first[0]) * pos.second;
				} else {
					outImage.at<float>(y, x) += image.at<float>(pos.first[1], pos.first[0]) * pos.second;
				}
				sumInvError += pos.second;
			}
			if (channels == 3) {
				outImage.at<cv::Vec3f>(y, x) /= sumInvError;
			} else {
				outImage.at<float>(y, x) /= sumInvError;
			}

			/*
			// MAL中のどこに入射したか計算
			auto diff = -realRay * mlaFocalLength / realRay[2];
			imagePos = centerList.at(minIndex[1]).at(minIndex[0]) + cv::Vec2f(diff[0] / pixelSize[0], diff[1] / pixelSize[1]);

			if (std::floor(imagePos[0]) < 0 || resX <= std::ceil(imagePos[0]) || std::floor(imagePos[1]) < 0 || resY <= std::ceil(imagePos[1])) {
				continue;
			}

			// bilinear interpolation
			float u = imagePos[0] - std::floor(imagePos[0]), v = imagePos[1] - std::floor(imagePos[1]);
			if (channels == 3) {
				const auto& a = image.at<cv::Vec3f>(std::floor(imagePos[1]), std::floor(imagePos[0]));
				const auto& b = image.at<cv::Vec3f>(std::floor(imagePos[1]), std::ceil(imagePos[0]));
				const auto& c = image.at<cv::Vec3f>(std::ceil(imagePos[1]), std::floor(imagePos[0]));
				const auto& d = image.at<cv::Vec3f>(std::ceil(imagePos[1]), std::ceil(imagePos[0]));
				outImage.at<cv::Vec3f>(y, x) += (1.0f - u) * (1.0f - v) * a + u * (1.0f - v) * b + (1.0f - u) * v * c + u * v * d;
			} else {
				const auto& a = image.at<float>(std::floor(imagePos[1]), std::floor(imagePos[0]));
				const auto& b = image.at<float>(std::floor(imagePos[1]), std::ceil(imagePos[0]));
				const auto& c = image.at<float> (std::ceil(imagePos[1]), std::floor(imagePos[0]));
				const auto& d = image.at<float>(std::ceil(imagePos[1]), std::ceil(imagePos[0]));
				outImage.at<float>(y, x) += (1.0f - u) * (1.0f - v) * a + u * (1.0f - v) * b + (1.0f - u) * v * c + u * v * d;
			}*/
		}
	}
	
	cv::Mat outImage2 = cv::Mat::zeros(lensNum[1] * imageScale, lensNum[0] * 2 * imageScale, channels == 3 ? CV_32FC3 : CV_32F);
	for (int y = 0; y < int(lensNum[1] * imageScale / 2); y++) {
		for (int x = 0; x < lensNum[0] * imageScale - 1; x++) {
			if (channels == 3) {
				const auto& a = outImage.at<cv::Vec3f>(y * 2, x).mul(whiteBalance) * intensity;
				const auto& b = outImage.at<cv::Vec3f>(y * 2, x + 1).mul(whiteBalance) * intensity;
				const auto& c = outImage.at<cv::Vec3f>(y * 2 + 1, x).mul(whiteBalance) * intensity;
				const auto& d = outImage.at<cv::Vec3f>(y * 2 + 1, x + 1).mul(whiteBalance) * intensity;

				outImage2.at<cv::Vec3f>(y * 2, x * 2) = a;
				outImage2.at<cv::Vec3f>(y * 2, x * 2 + 1) = (a + b) / 2;
				outImage2.at<cv::Vec3f>(y * 2 + 1, x * 2) = c;
				outImage2.at<cv::Vec3f>(y * 2 + 1, x * 2 + 1) = (c + d) / 2;
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

	cv::Mat ParallaxImage = cv::Mat((int) (lensNum[1] * imageScale * sqrt(3.0) + 0.5), lensNum[0] * 2 * imageScale, channels == 3 ? CV_32FC3 : CV_32F);
	cv::resize(outImage2, ParallaxImage, ParallaxImage.size(), 0, 0, cv::INTER_CUBIC);

	cv::imwrite(argv[5], ParallaxImage);

	return 0;
}
