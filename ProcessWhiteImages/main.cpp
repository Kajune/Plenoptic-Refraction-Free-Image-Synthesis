#include <stdlib.h>
#include <iostream>
#include <memory>
#include <vector>
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

constexpr float xstep = 14.3;
constexpr float ystep = 12.4;
constexpr float xcrop = 25;
constexpr float ycrop = 25;
constexpr int numLensStep = 20;

size_t nearest(const cv::Point& pos, const std::vector<cv::Point>& list) {
	float minDist = FLT_MAX;
	size_t index = 0;
	for (size_t i = 0; i < list.size(); i++) {
		float dist = cv::norm(list.at(i) - pos);
		if (dist < minDist) {
			minDist = dist;
			index = i;
		}
	}
	return index;
}

float getX(float y, const cv::Vec4f& param) {
	return (cv::Point2f(param[2], param[3]) - cv::Point2f(param[0], param[1]) * (param[3] / param[1])).x;
}

float getY(float x, const cv::Vec4f& param) {
	return (cv::Point2f(param[2], param[3]) - cv::Point2f(param[0], param[1]) * (param[2] / param[0])).y;
}

int main(int argc, char* argv[]) {
	if (argc < 4) {
		std::cout << "Usage: whiteImage, outputGridData, outputDebugImage, threshBalance(=0)" << std::endl;
		return 0;
	}

	float threshBalance = 0.0f;
	if (argc >= 5) {
		threshBalance = std::stof(argv[4]);
	}
	cv::Mat img = cv::imread(argv[1], 1);
	cv::Mat gray, thresh;
	cv::cvtColor(img, gray, CV_BGR2GRAY);
	cv::adaptiveThreshold(gray, thresh, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 15, threshBalance);
	cv::cvtColor(thresh, img, CV_GRAY2BGR);
//	cv::threshold(gray, thresh, cv::mean(gray)[0] * 0.6, 255, CV_THRESH_BINARY);
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(thresh, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

	std::vector<cv::Point> centerList;
	for (const auto& cont : contours) {
		auto mu = cv::moments(cont, true);
		if (mu.m00 == 0.0f) {
			continue;
		}
		float gx = mu.m10 / mu.m00;
		float gy = mu.m01 / mu.m00;

		centerList.emplace_back(gx, gy);
	}

	int numHorizontalStep = 0;
	std::vector<cv::Vec4f> horizontalParams;
	float y = ycrop;
	while (y < img.rows - ycrop) {
		auto center = centerList.at(nearest(cv::Point(xcrop, y), centerList));
		float x = center.x;
		y = center.y;
		float y2 = y;

		std::vector<cv::Point> points;
		while (x < img.cols - xcrop) {
			auto center = centerList.at(nearest(cv::Point(x, y2), centerList));
			x = center.x;
			y2 = center.y;
			cv::circle(img, cv::Point(int(x), int(y2)), 2, cv::Scalar(0, 255, 0), -1);
			points.emplace_back(x, y2);
			x += xstep * numLensStep;
		}

		if (points.size() >= 2) {
			cv::Vec4f param;
			cv::fitLine(points, param, CV_DIST_HUBER, 0, 0.01, 0.01);
			horizontalParams.emplace_back(param);

			cv::Point2f p0(param[2], param[3]);
			auto p1 = p0 - cv::Point2f(param[0], param[1]) * 10000;
			auto p2 = p0 + cv::Point2f(param[0], param[1]) * 10000;
			cv::line(img, p1, p2, cv::Scalar(0, 0, 255), 1);
		}

		y += ystep * numLensStep;
		numHorizontalStep++;
	}

	int numVerticalStep = 0;
	std::vector<cv::Vec4f> verticalParams;
	float x = xcrop;
	while (x < img.cols - xcrop) {
		auto center = centerList.at(nearest(cv::Point(x, ycrop), centerList));
		float y = center.y;
		x = center.x;
		float x2 = x;

		std::vector<cv::Point> points;
		while (y < img.rows - ycrop && x2 < img.cols - xcrop) {
			auto center = centerList.at(nearest(cv::Point(x2, y), centerList));
			x2 = center.x;
			y = center.y;
			cv::circle(img, cv::Point(int(x2), int(y)), 2, cv::Scalar(0, 255, 0), -1);
			points.emplace_back(x2, y);
			x2 += xstep * numLensStep / 2;
			y += ystep * numLensStep;
		}

		if (points.size() >= 2) {
			cv::Vec4f param;
			cv::fitLine(points, param, CV_DIST_HUBER, 0, 0.01, 0.01);
			verticalParams.emplace_back(param);

			cv::Point2f p0(param[2], param[3]);
			auto p1 = p0 - cv::Point2f(param[0], param[1]) * 10000;
			auto p2 = p0 + cv::Point2f(param[0], param[1]) * 10000;
			cv::line(img, p1, p2, cv::Scalar(0, 0, 255), 1);
		}

		x += xstep * numLensStep;
		numVerticalStep++;
	}

	// averaging line params
	cv::Point2f horiSlope(0, 0), vertSlope(0, 0);
	float horiStep, vertStep;

	for (const auto& param : horizontalParams) {
		horiSlope += cv::Point2f(param[0], param[1]);
	}
	horiSlope.x /= horizontalParams.size();
	horiSlope.y /= horizontalParams.size();
	vertStep = (getY(0, *horizontalParams.rbegin()) - getY(0, *horizontalParams.begin())) / (numLensStep * (numHorizontalStep - 1));

	for (const auto& param : verticalParams) {
		vertSlope += cv::Point2f(param[0], param[1]);
	}
	vertSlope.x /= verticalParams.size();
	vertSlope.y /= verticalParams.size();
	horiStep = (getX(0, *verticalParams.rbegin()) - getX(0, *verticalParams.begin())) / (numLensStep * (numVerticalStep - 1));

	std::cout << horiSlope << vertStep << vertSlope << horiStep << std::endl;

	// output
	std::ofstream ofs(argv[2]);
	auto start = centerList.at(nearest(cv::Point(xcrop, ycrop), centerList));
	int ycount = 0;
	for (float y = start.y; y < img.rows - ycrop; y += vertStep) {
		for (float x = start.x + (ycount % 2 == 0 ? 0 : horiStep / 2); x < img.cols - xcrop; x += horiStep) {
			float y2 = horiSlope.y * (x - start.x) / horiSlope.x + y;
			cv::circle(img, cv::Point(x, y2), 1, cv::Scalar(255, 0, 0), -1);
			ofs << x << " " << y2 << " ";
		}
		ofs << std::endl;
		ycount++;
	}

	cv::imwrite(argv[3], img);
}
