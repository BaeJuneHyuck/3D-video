#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <fstream>
#include <functional>

#include "../shared/view_synthesis.h"
#include "../shared/frame_reader.h"

using namespace cv::xfeatures2d;

using namespace std;
using namespace cv;

class Precalculator
{
private:
	vector<Mat> colors;

	vector<Point2f> candidates;

	vector<Mat> fundamental_matrices;

	vector<Mat> disparity_maps;
	vector<Mat> rev_disparity_maps;

public:
	Precalculator()
	{}

	void serialize_maps(int fi, int map_index, int sqrt_map_count, int sqrt_cam_count)
	{
		int top_left = (map_index / sqrt_map_count) * sqrt_cam_count + (map_index % sqrt_map_count);

		vector<Mat> maps_to_serialize;

		for (auto m : disparity_maps)
			maps_to_serialize.push_back(m);
		for (auto m : rev_disparity_maps)
			maps_to_serialize.push_back(m);

		for (int i = 0; i < maps_to_serialize.size(); i++)
		{
			Mat disp = Mat::zeros(maps_to_serialize[i].rows, maps_to_serialize[i].cols, CV_8UC3);

			for (int i0 = 0; i0 < disp.rows; i0++)
			{
				for (int i1 = 0; i1 < disp.cols; i1++)
				{
					float intensity = maps_to_serialize[i].at<float>(i0, i1);
					*((unsigned short*)& disp.at<Vec3b>(i0, i1)[0]) = int(intensity * 4) + 32768;
				}
			}

			imwrite(string("disparity_") + to_string(fi * maps_to_serialize.size() * sqrt_map_count * sqrt_map_count + map_index * maps_to_serialize.size() + i) + ".png", disp);
		}
	}

	vector<Mat> get_fundamental_matrices()
	{
		return fundamental_matrices;
	}

	void init(vector<string> paths, vector<Mat> _fundamental_matrices)
	{
		fundamental_matrices = _fundamental_matrices;

		vector<Mat> flows;
		vector<Mat> rev_flows;
		for (int i = 0; i < 4; i++)
		{
			string s{ paths[i] };
			colors.push_back(imread(s, IMREAD_UNCHANGED));
		}

		auto cwidth = colors[0].cols, cheight = colors[0].rows;

		for (int i = 1; i < colors.size(); i++)
		{
			if (colors[i].cols != cwidth || colors[i].rows != cheight)
			{
				cerr << "Image dimension must be same" << endl;
				exit(0);
			}
		}

		cout << "Starting view synthesis:" << endl;
		cout << "  Stride: " << stride << endl;
		cout << "  Block: (" << block[0] << ", " << block[1] << ")" << endl;

		auto start = chrono::system_clock::now();

		cout << endl << "Calculating Optical Flows..." << endl;
		calculate_flows(paths, flows, false);
		calculate_flows(paths, rev_flows, true);

		candidates = create_candidates(colors);

		double elapsed_1 = chrono::duration<double>(chrono::system_clock::now() - start).count();

		cout << "Done. ( " << elapsed_1 << "s )" << endl;

		calculate_disparity_maps(flows, colors, fundamental_matrices, candidates, disparity_maps, false);
		calculate_disparity_maps(rev_flows, colors, fundamental_matrices, candidates, rev_disparity_maps, true);
	}
};

int main(int argc, char* argv[])
{
	if (argc != 2)
	{
		cout << "Usage: " << argv[0] << " cam_count" << endl;
		return 0;
	}

	int cam_count = stoi(argv[1]);

	int sqrt_cam_count = int(sqrt(cam_count) + 0.05);
	int sqrt_map_count = sqrt_cam_count - 1;

	FrameReader fr("views.avi");

	cout << "Total frames: " << fr.total_frames() << endl;

	auto create_fundamental_matrices = [&](vector<int> frame_indices, int map_index, int sqrt_map_count, int sqrt_cam_count, int cam_count) -> vector<Mat>
	{
		vector<vector<Mat>> vec_colors;
		for (int i = 0; i < frame_indices.size(); i++)
		{
			vector<Mat> colors;
			int fi = frame_indices[i];

			int top_left = (map_index / sqrt_map_count) * sqrt_cam_count + (map_index % sqrt_map_count);
			int i0 = top_left;
			int i1 = top_left + 1;
			int i2 = top_left + sqrt_cam_count;
			int i3 = top_left + sqrt_cam_count + 1;

			Mat img0 = fr.get_image_matrix(cam_count * fi + i0);
			Mat img1 = fr.get_image_matrix(cam_count * fi + i1);
			Mat img2 = fr.get_image_matrix(cam_count * fi + i2);
			Mat img3 = fr.get_image_matrix(cam_count * fi + i3);
			colors.push_back(img0);
			colors.push_back(img1);
			colors.push_back(img2);
			colors.push_back(img3);
			vec_colors.push_back(colors);
		}

		vector<Mat> fundamental_matrices;
		calculate_fundamental_matrices_multiple(vec_colors, fundamental_matrices);

		return fundamental_matrices;
	};

	FileStorage fm("fundamental_matrices.xml", FileStorage::WRITE);

	fm << string("cam_count") << cam_count;

	vector<vector<Mat>> vec_fundamental_matrices;

	for (int i = 0; i < sqrt_map_count * sqrt_map_count; i++)
	{
		cout << "Calculating fundamental matrices..." << endl;
		auto fundamental_matrices = create_fundamental_matrices(vector<int>{0, int(fr.total_frames() / cam_count / 4), int(2 * fr.total_frames() / cam_count / 4), int(3 * fr.total_frames() / cam_count / 4)}, i, sqrt_map_count, sqrt_cam_count, cam_count);

		for (auto m : fundamental_matrices)
			cout << m << endl;

		for (int j = 0; j < fundamental_matrices.size(); j++)
			fm << (string("F") + to_string(i) + "_" + to_string(j)) << fundamental_matrices[j];

		vec_fundamental_matrices.push_back(fundamental_matrices);
	}
	fm.release();

	auto create_disparity_maps = [&](int frame_index, int map_index, int cam_count) -> void
	{
		int top_left = (map_index / sqrt_map_count) * sqrt_cam_count + (map_index % sqrt_map_count);

		int i0 = top_left;
		int i1 = top_left + 1;
		int i2 = top_left + sqrt_cam_count;
		int i3 = top_left + sqrt_cam_count + 1;

		Mat img0 = fr.get_image_matrix(cam_count * frame_index + i0);
		imwrite("0.jpg", img0);
		Mat img1 = fr.get_image_matrix(cam_count * frame_index + i1);
		imwrite("1.jpg", img1);
		Mat img2 = fr.get_image_matrix(cam_count * frame_index + i2);
		imwrite("2.jpg", img2);
		Mat img3 = fr.get_image_matrix(cam_count * frame_index + i3);
		imwrite("3.jpg", img3);

		Precalculator c;
		c.init(vector<string>{"0.jpg", "1.jpg", "2.jpg", "3.jpg"}, vec_fundamental_matrices[map_index]);

		c.serialize_maps(frame_index, map_index, sqrt_map_count, sqrt_cam_count);
	};

	view_initialize();

	for (int i = 0; i < fr.total_frames() / cam_count; i++)
	{
		for (int j = 0; j < sqrt_map_count * sqrt_map_count; j++)
			create_disparity_maps(i, j, cam_count);
	}

	view_finalize_precalc();
}
