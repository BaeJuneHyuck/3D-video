#pragma once

#include "py_flow.h"

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

struct displacement
{
	float disp;

	int x;
	int y;
};

// < config below >---

const int stride = 4;
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
	for(string path: paths)
		grays.push_back(imread(path, IMREAD_GRAYSCALE));

	auto optflow = cv::optflow::createOptFlow_DeepFlow();

	for (auto a : vector<vector<int>>{ {1, 2}, {1, 0}, {3, 0}, {3, 2} })
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

void calculate_sorted_displacements(const vector<Mat>& flows, vector<displacement>& upper_sorted_displacements, vector<displacement>& lower_sorted_displacements, bool reverse)
{
	upper_sorted_displacements.clear();
	lower_sorted_displacements.clear();

	for (int i0 = 0; i0 < flows[0].rows; i0 += stride)
	{
		for (int i1 = 0; i1 < flows[0].cols; i1 += stride)
		{
			auto x = i1, y = i0;
			
			auto create_displacements = [&](const Mat fy, const Mat fx, vector<displacement>& displacements)
			{
				float disparity0 = sqrtf(fy.at<Point2f>(i0, i1).x * fy.at<Point2f>(i0, i1).x + fy.at<Point2f>(i0, i1).y * fy.at<Point2f>(i0, i1).y);
				float disparity1 = sqrtf(fx.at<Point2f>(i0, i1).x * fx.at<Point2f>(i0, i1).x + fx.at<Point2f>(i0, i1).y * fx.at<Point2f>(i0, i1).y);
				float disp = (disparity0 + disparity1) / 2.0f;

				displacements.push_back(displacement{ disp, x, y });
			};

			if (reverse)
			{
				create_displacements(flows[2], flows[1], upper_sorted_displacements);
				create_displacements(flows[0], flows[3], lower_sorted_displacements);
			}
			else
			{
				create_displacements(flows[0], flows[1], upper_sorted_displacements);
				create_displacements(flows[2], flows[3], lower_sorted_displacements);
			}
		}
	}

	auto sort_displacements = [](vector<displacement>& displacements)
	{
		sort(displacements.begin(), displacements.end(), [](displacement a0, displacement a1)
			{
				return a0.disp > a1.disp;
			});
	};

	sort_displacements(upper_sorted_displacements);
	sort_displacements(lower_sorted_displacements);
}

void calculate_fundamental_matrices(const vector<Mat>& colors, vector<Mat>& fundamental_matrices)
{
	fundamental_matrices.clear();

	auto calculate_fundamental_matrix = [&](const Mat img1, const Mat img2)
	{
		Ptr<SIFT> detector = SIFT::create();
		std::vector<KeyPoint> keypoints1, keypoints2;
		Mat descriptors1, descriptors2;
		detector->detectAndCompute(img1, noArray(), keypoints1, descriptors1);
		detector->detectAndCompute(img2, noArray(), keypoints2, descriptors2);

		Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
		std::vector< std::vector<DMatch> > knn_matches;
		matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);

		const float ratio_thresh = 0.6f;
		std::vector<DMatch> good_matches;
		for (size_t i = 0; i < knn_matches.size(); i++)
		{
			if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
			{
				good_matches.push_back(knn_matches[i][0]);
			}
		}

		int point_count = good_matches.size();
		vector<Point2f> points1(point_count);
		vector<Point2f> points2(point_count);

		for (int i = 0; i < point_count; i++)
		{
			points1[i] = keypoints1[good_matches[i].queryIdx].pt;
			points2[i] = keypoints2[good_matches[i].trainIdx].pt;
		}

		Mat fundamental_matrix = findFundamentalMat(points1, points2, FM_LMEDS, 1, 0.999, 16000);
		fundamental_matrices.push_back(fundamental_matrix);

		cout << "F: " << endl;
		cout << fundamental_matrix << endl;

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

		for (int i = 0; i < point_count; i++)
		{
			float a = lines[i][0], b = lines[i][1], c = lines[i][2];
			float a0 = lines_rev[i][0], b0 = lines_rev[i][1], c0 = lines_rev[i][2];
			cv::Scalar color(rand() % 256, rand() % 256, rand() % 256);

			cv::line(points_image, Point(0, -c0 / b0), Point(points_image.cols, -(c0 + a0 * points_image.cols) / b0), color);
			cv::circle(points_image, points1[i], 3, color, -1);

			cv::line(epilines_image, Point(0, -c / b), Point(epilines_image.cols, -(c + a * epilines_image.cols) / b), color);
			cv::circle(epilines_image, points2[i], 3, color, -1);
		}

		imshow("x", points_image);
		imshow("y", epilines_image);
		waitKey(0);
		destroyAllWindows();
	};

	calculate_fundamental_matrix(colors[1], colors[2]);
	calculate_fundamental_matrix(colors[1], colors[0]);
	calculate_fundamental_matrix(colors[3], colors[0]);
	calculate_fundamental_matrix(colors[3], colors[2]);
}

void calculate_disparity_maps(const vector<Mat>& flows, const vector<Mat>& colors, const vector<Mat>& fundamental_matrices, const vector<displacement>& upper_sorted_displacements, const vector<displacement>& lower_sorted_displacements, vector<Mat>& disparity_maps, bool reverse)
{
	disparity_maps.clear();

	auto calculate_disparity_map = [&](const vector<displacement>& displacements, const Mat flow, const Mat fundamental_matrix, const string window_name)
	{
		auto epi_start = chrono::system_clock::now();

		std::vector<cv::Vec3f> lines;
		std::vector<cv::Vec3f> lines_rev;

		vector<Point2f> points1(displacements.size());
		vector<Point2f> points2(displacements.size());

		for (int i = 0; i < displacements.size(); i++)
		{
			int y = displacements[i].y, x = displacements[i].x;
			points1[i] = Point2f{ float(x), float(y) };
			float dy = flow.at<Point2f>(y, x).y, dx = flow.at<Point2f>(y, x).x;
			points2[i] = Point2f{ x + dx, y + dy };
		}

		cv::computeCorrespondEpilines(points1, !reverse ? 1 : 2, fundamental_matrix, lines);

		Mat displacement_map = Mat::zeros(flow.rows, flow.cols, CV_8UC1);
		Mat disparity_map(flow.rows / stride + 1, flow.cols / stride + 1, CV_32FC1);

		for (int i = 0; i < displacements.size(); i++)
		{
			float a = lines[i][0], b = lines[i][1], c = lines[i][2];

			Point2f ab(c / a, -c / b);
			Point2f ap(points2[i].x + c / a, points2[i].y);
			float norm = ab.x * ab.x + ab.y * ab.y;
			float t = (ab.x * ap.x + ab.y * ap.y) / norm;

			Point2f xp(-c / a + t * c / a, -t * c / b);

			Point2f dx(xp.x - points1[i].x, xp.y - points1[i].y);

			disparity_map.at<float>(displacements[i].y / stride, displacements[i].x / stride) = t;

			int ypp = displacements[i].y + dx.y, xpp = displacements[i].x + dx.x;
			float disparity = sqrt(dx.x * dx.x + dx.y * dx.y);
			if (0 <= xpp && xpp < displacement_map.cols && 0 <= ypp && ypp < displacement_map.rows)
				displacement_map.at<unsigned char>(ypp, xpp) = min(255, int(2 * disparity));
		}
		disparity_maps.push_back(disparity_map);

		imshow(window_name, displacement_map);
	};

	calculate_disparity_map(upper_sorted_displacements, flows[0], fundamental_matrices[0], "d0");
	calculate_disparity_map(upper_sorted_displacements, flows[1], fundamental_matrices[1], "d1");
	calculate_disparity_map(lower_sorted_displacements, flows[2], fundamental_matrices[2], "d2");
	calculate_disparity_map(lower_sorted_displacements, flows[3], fundamental_matrices[3], "d3");

	waitKey(0);
	destroyAllWindows();
}

Mat synthesize_view(const float camera_pos_ratio_x, const float camera_pos_ratio_y, const float v_z, const vector<Point2f>& upper_candidates, const vector<Point2f>& lower_candidates, const vector<Point2f> rev_upper_candidates, const vector<Point2f> rev_lower_candidates, const vector<Mat>& colors, const vector<Mat>& disparity_maps, const vector<Mat>& rev_disparity_maps, const vector<Mat>& fundamental_matrices)
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

	auto& candidates = upper_triangle ? upper_candidates : lower_candidates;
	auto& rev_candidates = rev_upper_triangle ? rev_upper_candidates : rev_lower_candidates;

	if (upper_triangle)
	{
		cy0 = colors[1];
		cy1 = colors[2];
		cx0 = colors[1];
		cx1 = colors[0];

		dy = disparity_maps[0];
		dx = disparity_maps[1];

		cv::computeCorrespondEpilines(candidates, 1, fundamental_matrices[0], lines0);
		cv::computeCorrespondEpilines(candidates, 1, fundamental_matrices[1], lines1);
	}
	else
	{
		cy0 = colors[3];
		cy1 = colors[0];
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
		rcy0 = colors[0];
		rcy1 = colors[3];
		rcx0 = colors[0];
		rcx1 = colors[1];

		rdy = rev_disparity_maps[2];
		rdx = rev_disparity_maps[1];

		cv::computeCorrespondEpilines(rev_candidates, 2, fundamental_matrices[2], rlines0);
		cv::computeCorrespondEpilines(rev_candidates, 2, fundamental_matrices[1], rlines1);
	}
	else
	{
		rcy0 = colors[2];
		rcy1 = colors[1];
		rcx0 = colors[2];
		rcx1 = colors[3];

		rdy = rev_disparity_maps[0];
		rdx = rev_disparity_maps[3];

		rev_camera_pos_ratio = vector<float>{ 1.0f - rev_camera_pos_ratio[0], 1.0f - rev_camera_pos_ratio[1] };

		cv::computeCorrespondEpilines(rev_candidates, 2, fundamental_matrices[0], rlines0);
		cv::computeCorrespondEpilines(rev_candidates, 2, fundamental_matrices[3], rlines1);
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

			Point2f sp0(-c0 / a0 + t0 * c0 / a0, -t0 * c0 / b0);
			Point2f sp1(-c1 / a1 + t1 * c1 / a1, -t1 * c1 / b1);

			Point2f dx0(sp0.x - x, sp0.y - y);
			Point2f dx1(sp1.x - x, sp1.y - y);

			float disparity = (sqrtf(dx0.x * dx0.x + dx0.y * dx0.y) + sqrtf(dx1.x * dx1.x + dx1.y * dx1.y)) / 2.0f;
			float z_coeff = 1.0f / (1.0f + v_z * disparity);

			auto xp = x + int(dx0.x * r0 + dx1.x * r1);
			auto yp = y + int(dx0.y * r0 + dx1.y * r1);

			xp *= z_coeff;
			yp *= z_coeff;

			auto hbx = block[0] / 2, hby = block[1] / 2;

			if ((0 <= xp - hbx && xp + hbx <= cwidth - 1) &&
				(0 <= yp - hby && yp + hby <= cheight - 1) &&
				(0 <= x - hbx && x + hbx <= cwidth - 1) &&
				(0 <= y - hby && y + hby <= cheight - 1))
				
			{
				auto origin = cx0(Range(y - hby, y + hby), Range(x - hbx, x + hbx));
				origin.copyTo(syn(Range(yp - hby, yp + hby), Range(xp - hbx, xp + hbx)));
			}
		}
	};

	if (reverse)
	{
		one_pass(candidates, camera_pos_ratio[0], camera_pos_ratio[1], v_z, lines0, lines1, cy0, cy1, cx0, cx1, dy, dx, syn);
		one_pass(rev_candidates, rev_camera_pos_ratio[0], rev_camera_pos_ratio[1], v_z, rlines0, rlines1, rcy0, rcy1, rcx0, rcx1, rdy, rdx, syn);
	}
	else
	{
		one_pass(rev_candidates, rev_camera_pos_ratio[0], rev_camera_pos_ratio[1], v_z, rlines0, rlines1, rcy0, rcy1, rcx0, rcx1, rdy, rdx, syn);
		one_pass(candidates, camera_pos_ratio[0], camera_pos_ratio[1], v_z, lines0, lines1, cy0, cy1, cx0, cx1, dy, dx, syn);
	}

	return syn;
}