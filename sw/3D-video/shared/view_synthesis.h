#pragma once

#include "py_optflow.h"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
using namespace cv::xfeatures2d;

#include <iostream>
#include <vector>
#include <array>
#include <opencv2/opencv.hpp>
#include <opencv2/optflow.hpp>
#include <algorithm>
#include <functional>
#include <thread>
#include <chrono>
#include <random>

using namespace std;
using namespace cv;

// < config below >---

const int stride = 8;
const double block_dilation = 1.5;

// ---< config above >

const vector<int> block{ int(block_dilation * stride), int(block_dilation * stride) };

void view_initialize()
{
	py_initialize();
}

void view_finalize_precalc()
{
	Py_Finalize();
}

void calculate_flows(const vector<string>& paths, vector<Mat>& flows, bool reverse)
{
	flows.clear();

	vector<Mat> grays;
	for (string path : paths)
		grays.push_back(imread(path, IMREAD_GRAYSCALE));

	auto optflow = cv::optflow::createOptFlow_DeepFlow();

	for (auto a : vector<vector<int>>{ {0, 2}, {0, 1}, {3, 1}, {3, 2} })
	{
		int i0 = a[0], i1 = a[1];
		if (reverse)
			i0 = a[1], i1 = a[0];

		Mat flow;
		py_calc_optflow(paths[i0], paths[i1], flow);
		//optflow->calc(imread(paths[i0], IMREAD_GRAYSCALE), imread(paths[i1], IMREAD_GRAYSCALE), flow);

		flows.push_back(flow);
	}
}

vector<Point2f> create_candidates(const vector<Mat> colors)
{
	vector<Point2f> candidates;
	for (int i0 = 0; i0 < colors[0].rows; i0 += stride)
	{
		for (int i1 = 0; i1 < colors[0].cols; i1 += stride)
		{
			candidates.push_back(Point2f{ float(i1), float(i0) });
		}
	}

	return candidates;
}

void calculate_feature_matches(const Mat img1, const Mat img2, vector<Point2f>& points1, vector<Point2f>& points2)
{
	points1.clear();
	points2.clear();

	Ptr<SIFT> detector = SIFT::create(0, 3, 0.04, 10, 1.2);
	std::vector<KeyPoint> keypoints1, keypoints2;
	Mat descriptors1, descriptors2;
	detector->detectAndCompute(img1, noArray(), keypoints1, descriptors1);
	detector->detectAndCompute(img2, noArray(), keypoints2, descriptors2);

	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
	std::vector< std::vector<DMatch> > knn_matches;
	matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);

	const float ratio_thresh = 0.7f;
	std::vector<DMatch> good_matches;
	for (size_t i = 0; i < knn_matches.size(); i++)
	{
		if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
		{
			good_matches.push_back(knn_matches[i][0]);
		}
	}

	for (int i = 0; i < good_matches.size(); i++)
	{
		points1.push_back(keypoints1[good_matches[i].queryIdx].pt);
		points2.push_back(keypoints2[good_matches[i].trainIdx].pt);
	}
}

void visualize_fundamental_matrices(const vector<Mat>& colors, vector<Mat>& fundamental_matrices)
{
	auto visualize_fundamental_matrix = [&](const Mat img1, const Mat img2, Mat fundamental_matrix, string title_prefix)
	{
		vector<Point2f> points1;
		vector<Point2f> points2;

		calculate_feature_matches(img1, img2, points1, points2);

		auto epi_start = chrono::system_clock::now();

		std::vector<cv::Vec3f> lines;
		cv::computeCorrespondEpilines(points1, 1, fundamental_matrix, lines);

		std::vector<cv::Vec3f> lines_rev;
		cv::computeCorrespondEpilines(points2, 2, fundamental_matrix, lines_rev);

		cout << "Epi_time: " << chrono::duration<double>(chrono::system_clock::now() - epi_start).count() << endl;
		cout << "Number of lines: " << lines.size() << endl;

		Mat points_image;
		img1.copyTo(points_image);

		Mat epilines_image;
		img2.copyTo(epilines_image);

		for (int i = 0; i < points1.size(); i++)
		{
			float a = lines[i][0], b = lines[i][1], c = lines[i][2];
			float a0 = lines_rev[i][0], b0 = lines_rev[i][1], c0 = lines_rev[i][2];
			cv::Scalar color(rand() % 256, rand() % 256, rand() % 256);

			cv::line(points_image, Point(0, -c0 / b0), Point(points_image.cols, -(c0 + a0 * points_image.cols) / b0), color);
			cv::circle(points_image, points1[i], 3, color, -1);

			cv::line(epilines_image, Point(0, -c / b), Point(epilines_image.cols, -(c + a * epilines_image.cols) / b), color);
			cv::circle(epilines_image, points2[i], 3, color, -1);
		}

		imshow(title_prefix + "0", points_image);
		imshow(title_prefix + "1", epilines_image);
	};

	visualize_fundamental_matrix(colors[0], colors[2], fundamental_matrices[0], "y");
	visualize_fundamental_matrix(colors[0], colors[1], fundamental_matrices[1], "x");
	waitKey(0);
	destroyAllWindows();
	visualize_fundamental_matrix(colors[3], colors[1], fundamental_matrices[2], "y");
	visualize_fundamental_matrix(colors[3], colors[2], fundamental_matrices[3], "x");
	waitKey(0);
	destroyAllWindows();
}

void calculate_fundamental_matrices_multiple(const vector<vector<Mat>>& vec_colors, vector<Mat>& fundamental_matrices)
{
	fundamental_matrices.clear();

	auto calculate_fundamental_matrix = [&](const vector<Mat> img1, const vector<Mat> img2)
	{
		vector<Point2f> points1;
		vector<Point2f> points2;

		for (int ii = 0; ii < img1.size(); ii++)
		{
			calculate_feature_matches(img1[ii], img2[ii], points1, points2);
		}

		Mat fundamental_matrix = findFundamentalMat(points1, points2, FM_RANSAC, 2, 0.99, 16000);
		fundamental_matrices.push_back(fundamental_matrix);

		cout << "F: " << endl;
		cout << fundamental_matrix << endl;
	};

	vector<Mat> colors0, colors1, colors2, colors3;
	for (int i = 0; i < vec_colors.size(); i++)
	{
		colors0.push_back(vec_colors[i][0]);
		colors1.push_back(vec_colors[i][1]);
		colors2.push_back(vec_colors[i][2]);
		colors3.push_back(vec_colors[i][3]);
	}
	calculate_fundamental_matrix(colors0, colors2);
	calculate_fundamental_matrix(colors0, colors1);
	calculate_fundamental_matrix(colors3, colors1);
	calculate_fundamental_matrix(colors3, colors2);

	visualize_fundamental_matrices(vec_colors[0], fundamental_matrices);
}

void calculate_fundamental_matrices(const vector<Mat>& colors, vector<Mat>& fundamental_matrices)
{
	calculate_fundamental_matrices_multiple(vector<vector<Mat>>{ colors }, fundamental_matrices);
}

void calculate_disparity_maps(const vector<Mat>& flows, const vector<Mat>& colors, const vector<Mat>& fundamental_matrices, const vector<Point2f>& candidates, vector<Mat>& disparity_maps, bool reverse)
{
	disparity_maps.clear();

	auto calculate_disparity_map = [&](const vector<Point2f>& candidates, const Mat flow, const Mat fundamental_matrix, const string window_name)
	{
		auto epi_start = chrono::system_clock::now();

		std::vector<cv::Vec3f> lines;
		std::vector<cv::Vec3f> lines_rev;

		vector<Point2f> points1;
		vector<Point2f> points2;

		for (int i = 0; i < candidates.size(); i++)
		{
			float y = candidates[i].y, x = candidates[i].x;
			points1.push_back(Point2f{ x, y });
			float dy = flow.at<Point2f>(y, x).y, dx = flow.at<Point2f>(y, x).x;
			points2.push_back(Point2f{ x + dx, y + dy });
		}

		cv::computeCorrespondEpilines(points1, !reverse ? 1 : 2, fundamental_matrix, lines);

		Mat displacement_map = Mat::zeros(flow.rows, flow.cols, CV_8UC1);
		Mat disparity_map(flow.rows / stride + 1, flow.cols / stride + 1, CV_32FC1);

		for (int i = 0; i < candidates.size(); i++)
		{
			int y = int(candidates[i].y + 0.05), x = int(candidates[i].x + 0.05);
			float a = lines[i][0], b = lines[i][1], c = lines[i][2];

			Point2f ab(c / a, -c / b);
			float norm = sqrtf(ab.x * ab.x + ab.y * ab.y);
			ab.x /= norm, ab.y /= norm;

			Point2f ap1(points1[i].x + c / a, points1[i].y);
			Point2f ap2(points2[i].x + c / a, points2[i].y);

			float t1 = ab.x * ap1.x + ab.y * ap1.y;
			float t2 = ab.x * ap2.x + ab.y * ap2.y;

			Point2f xp1(-c / a + t1 * ab.x, t1 * ab.y);
			Point2f xp2(-c / a + t2 * ab.x, t2 * ab.y);

			float t = t2 - t1;

			Point2f dx(xp2.x - points1[i].x, xp2.y - points1[i].y);

			disparity_map.at<float>(y / stride, x / stride) = t;

			int ypp = y + dx.y, xpp = x + dx.x;
			float disparity = sqrt(dx.x * dx.x + dx.y * dx.y);
			if (0 <= xpp && xpp < displacement_map.cols && 0 <= ypp && ypp < displacement_map.rows)
				displacement_map.at<unsigned char>(y, x) = min(255, int(4 * abs(t)));
		}
		disparity_maps.push_back(disparity_map);

		//imshow(window_name, displacement_map);
	};

	calculate_disparity_map(candidates, flows[0], fundamental_matrices[0], "dy0");
	calculate_disparity_map(candidates, flows[1], fundamental_matrices[1], "dx0");
	calculate_disparity_map(candidates, flows[2], fundamental_matrices[2], "dy1");
	calculate_disparity_map(candidates, flows[3], fundamental_matrices[3], "dx1");

	waitKey(0);
	destroyAllWindows();
}

Mat synthesize_view(const float camera_pos_ratio_x, const float camera_pos_ratio_y, const float v_z, const vector<Point2f>& candidates, const vector<Mat>& colors, const vector<Mat>& disparity_maps, const vector<Mat>& rev_disparity_maps, const vector<Mat>& fundamental_matrices)
{
	vector<float> camera_pos_ratio;
	vector<float> rev_camera_pos_ratio;

	camera_pos_ratio = vector<float>{ camera_pos_ratio_y, camera_pos_ratio_x };
	rev_camera_pos_ratio = vector<float>{ camera_pos_ratio[0], 1.0f - camera_pos_ratio[1] };

	bool reverse = (0.5f < camera_pos_ratio[0] && camera_pos_ratio[1] <= 0.5f) || (camera_pos_ratio[0] <= 0.5f && 0.5f < camera_pos_ratio[1]);

	bool upper_triangle = (camera_pos_ratio[0] + camera_pos_ratio[1]) <= 1.0f;
	bool rev_upper_triangle = (camera_pos_ratio[1] - camera_pos_ratio[0]) >= 0.0f;

	Mat cx0, cx1;
	Mat cy0, cy1;

	Mat rcx0, rcx1;
	Mat rcy0, rcy1;

	Mat dy, dx;
	Mat rdy, rdx;

	vector<cv::Vec3f> lines0, lines1;
	vector<cv::Vec3f> rlines0, rlines1;

	if (upper_triangle)
	{
		cy0 = colors[0];
		cy1 = colors[2];
		cx0 = colors[0];
		cx1 = colors[1];

		dy = disparity_maps[0];
		dx = disparity_maps[1];

		cv::computeCorrespondEpilines(candidates, 1, fundamental_matrices[0], lines0);
		cv::computeCorrespondEpilines(candidates, 1, fundamental_matrices[1], lines1);
	}
	else
	{
		cy0 = colors[3];
		cy1 = colors[1];
		cx0 = colors[3];
		cx1 = colors[2];

		dy = disparity_maps[2];
		dx = disparity_maps[3];

		camera_pos_ratio = vector<float>{ 1.0f - camera_pos_ratio[0], 1.0f - camera_pos_ratio[1] };

		cv::computeCorrespondEpilines(candidates, 1, fundamental_matrices[2], lines0);
		cv::computeCorrespondEpilines(candidates, 1, fundamental_matrices[3], lines1);
	}

	if (rev_upper_triangle)
	{
		rcy0 = colors[1];
		rcy1 = colors[3];
		rcx0 = colors[1];
		rcx1 = colors[0];

		rdy = rev_disparity_maps[2];
		rdx = rev_disparity_maps[1];

		cv::computeCorrespondEpilines(candidates, 2, fundamental_matrices[2], rlines0);
		cv::computeCorrespondEpilines(candidates, 2, fundamental_matrices[1], rlines1);
	}
	else
	{
		rcy0 = colors[2];
		rcy1 = colors[0];
		rcx0 = colors[2];
		rcx1 = colors[3];

		rdy = rev_disparity_maps[0];
		rdx = rev_disparity_maps[3];

		rev_camera_pos_ratio = vector<float>{ 1.0f - rev_camera_pos_ratio[0], 1.0f - rev_camera_pos_ratio[1] };

		cv::computeCorrespondEpilines(candidates, 2, fundamental_matrices[0], rlines0);
		cv::computeCorrespondEpilines(candidates, 2, fundamental_matrices[3], rlines1);
	}

	Mat syn;

	auto initialize_syn = [&]()
	{
		float r1_contribution, r0, r1;
		Mat syn_origin, syn_down, syn_right;

		if (reverse)
		{
			r0 = camera_pos_ratio[0];
			r1 = camera_pos_ratio[1];
			syn_origin = cx0;
			syn_down = cy1;
			syn_right = cx1;
		}
		else
		{
			r0 = rev_camera_pos_ratio[0];
			r1 = rev_camera_pos_ratio[1];
			syn_origin = rcx0;
			syn_down = rcy1;
			syn_right = rcx1;
		}
		r1_contribution = 0.5f + 0.5f * (r1 - r0);

		if (r1_contribution > 0.5f)
		{
			addWeighted(syn_origin, 1.0 - r1, syn_right, r1, 0.0, syn);
		}
		else
		{
			addWeighted(syn_origin, 1.0 - r0, syn_down, r0, 0.0, syn);
		}
	};

	initialize_syn();

	auto one_pass = [](const vector<Point2f>& candidates, const float r0, const float r1, const float v_z, const vector<Vec3f>& lines0, const vector<Vec3f>& lines1, const Mat cy0, const Mat cy1, const Mat cx0, const Mat cx1, const Mat dy, const Mat dx, Mat& syn)
	{
		for (int i = 0; i < candidates.size(); i++)
		{
			int cwidth = cx0.cols, cheight = cx0.rows;

			auto a = candidates[i];

			int x = int(a.x + 0.05);
			int y = int(a.y + 0.05);

			float a0 = lines0[i][0], b0 = lines0[i][1], c0 = lines0[i][2];
			float a1 = lines1[i][0], b1 = lines1[i][1], c1 = lines1[i][2];

			float t0 = dy.at<float>(y / stride, x / stride);
			float t1 = dx.at<float>(y / stride, x / stride);

			t0 = int(t0 * 2) / 2.0f, t1 = int(t1 * 2) / 2.0f;

			Point2f ab0(c0 / a0, -c0 / b0);
			Point2f ab1(c1 / a1, -c1 / b1);
			float norm0 = sqrtf(ab0.x * ab0.x + ab0.y * ab0.y);
			float norm1 = sqrtf(ab1.x * ab1.x + ab1.y * ab1.y);
			ab0.x /= norm0, ab0.y /= norm0;
			ab1.x /= norm1, ab1.y /= norm1;

			Point2f ap0(x + c0 / a0, y);
			Point2f ap1(x + c1 / a1, y);
			float t1_0 = ab0.x * ap0.x + ab0.y * ap0.y;
			float t1_1 = ab1.x * ap1.x + ab1.y * ap1.y;
			Point2f xp0(-c0 / a0 + t1_0 * ab0.x, t1_0 * ab0.y);
			Point2f xp1(-c1 / a1 + t1_1 * ab1.x, t1_1 * ab1.y);

			Point2f sp0(xp0.x + t0 * ab0.x, xp0.y + t0 * ab0.y);
			Point2f sp1(xp1.x + t1 * ab1.x, xp1.y + t1 * ab1.y);

			Point2f dx0(sp0.x - x, sp0.y - y);
			Point2f dx1(sp1.x - x, sp1.y - y);

			float disparity = (sqrtf(dx0.x * dx0.x + dx0.y * dx0.y) + sqrtf(dx1.x * dx1.x + dx1.y * dx1.y)) / 2.0f;
			float z_coeff = 1.0f / (1.0f + v_z * disparity);

			auto xp = x + int(dx0.x * r0 + dx1.x * r1);
			auto yp = y + int(dx0.y * r0 + dx1.y * r1);

			xp *= z_coeff;
			yp *= z_coeff;

			auto hbx = block[0] / 2, hby = block[1] / 2;

			if ((0 <= xp - hbx && xp + hbx + 1 <= cwidth) &&
				(0 <= yp - hby && yp + hby + 1 <= cheight) &&
				(0 <= x - hbx && x + hbx + 1 <= cwidth) &&
				(0 <= y - hby && y + hby + 1 <= cheight))

			{
				auto origin = cx0(Range(y - hby, y + hby + 1), Range(x - hbx, x + hbx + 1));
				origin.copyTo(syn(Range(yp - hby, yp + hby + 1), Range(xp - hbx, xp + hbx + 1)));
			}
		}
	};

	if (reverse)
	{
		one_pass(candidates, camera_pos_ratio[0], camera_pos_ratio[1], v_z, lines0, lines1, cy0, cy1, cx0, cx1, dy, dx, syn);
		one_pass(candidates, rev_camera_pos_ratio[0], rev_camera_pos_ratio[1], v_z, rlines0, rlines1, rcy0, rcy1, rcx0, rcx1, rdy, rdx, syn);
	}
	else
	{
		one_pass(candidates, rev_camera_pos_ratio[0], rev_camera_pos_ratio[1], v_z, rlines0, rlines1, rcy0, rcy1, rcx0, rcx1, rdy, rdx, syn);
		one_pass(candidates, camera_pos_ratio[0], camera_pos_ratio[1], v_z, lines0, lines1, cy0, cy1, cx0, cx1, dy, dx, syn);
	}

	return syn;
}
