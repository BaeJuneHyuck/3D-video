#include "view.h"

const int points = 8;
const string result_prefix{ "./res/" };

int main(int argc, char* argv[])
{
	vector<Mat> colors;
	vector<Mat> flows;
	vector<Mat> rev_flows;

	if (argc - 1 != 4)
	{
		cerr << "Usage: " << argv[0] << " image1 image2 image3 image4" << endl;
		return 0;
	}

	vector<string> paths;
	for (int i = 1; i <= 4; i++)
	{
		string s{ argv[i] };
		paths.push_back(s);
		colors.push_back(imread(s, IMREAD_UNCHANGED));
	}

	auto cwidth = colors[0].cols, cheight = colors[0].rows;

	for (int i = 1; i < colors.size(); i++)
	{
		if (colors[i].cols != cwidth || colors[i].rows != cheight)
		{
			cerr << "Image dimension must be same" << endl;
			return 0;
		}
	}

	cout << "Starting view synthesis:" << endl;
	cout << "  Points: " << points << endl;
	cout << "  Stride: " << stride << endl;
	cout << "  Block: (" << block[0] << ", " << block[1] << ")" << endl;
	cout << "  Result Prefix: " << result_prefix << endl;

	view_initialize();

	auto start = chrono::system_clock::now();

	cout << endl << "Calculating Optical Flows..." << endl;
	calculate_flows(paths, flows, false);
	calculate_flows(paths, rev_flows, true);

	vector<displacement> upper_sorted_displacements;
	vector<displacement> lower_sorted_displacements;

	vector<displacement> rev_upper_sorted_displacements;
	vector<displacement> rev_lower_sorted_displacements;

	calculate_sorted_displacements(flows, upper_sorted_displacements, lower_sorted_displacements, false);
	calculate_sorted_displacements(rev_flows, rev_upper_sorted_displacements, rev_lower_sorted_displacements, true);

	int disp_count = upper_sorted_displacements.size();
	int rev_disp_count = rev_upper_sorted_displacements.size();
	vector<Point2f> upper_candidates(disp_count);
	vector<Point2f> lower_candidates(disp_count);

	vector<Point2f> rev_upper_candidates(rev_disp_count);
	vector<Point2f> rev_lower_candidates(rev_disp_count);

	for (int i = 0; i < disp_count; i++)
	{
		int uy = upper_sorted_displacements[i].y, ux = upper_sorted_displacements[i].x;
		upper_candidates[i] = Point2f{ float(ux), float(uy) };
		int ly = lower_sorted_displacements[i].y, lx = lower_sorted_displacements[i].x;
		lower_candidates[i] = Point2f{ float(lx), float(ly) };
	}

	for (int i = 0; i < rev_disp_count; i++)
	{
		int uy = rev_upper_sorted_displacements[i].y, ux = rev_upper_sorted_displacements[i].x;
		rev_upper_candidates[i] = Point2f{ float(ux), float(uy) };
		int ly = rev_lower_sorted_displacements[i].y, lx = rev_lower_sorted_displacements[i].x;
		rev_lower_candidates[i] = Point2f{ float(lx), float(ly) };
	}

	double elapsed_1 = chrono::duration<double>(chrono::system_clock::now() - start).count();

	cout << "Done. ( " << elapsed_1 << "s )" << endl;

	view_finalize_precalc();

	vector<Mat> fundamental_matrices;
	vector<Mat> disparity_maps;
	vector<Mat> rev_disparity_maps;

	calculate_fundamental_matrices(colors, fundamental_matrices);
	calculate_disparity_maps(flows, colors, fundamental_matrices, upper_sorted_displacements, lower_sorted_displacements, disparity_maps, false);
	calculate_disparity_maps(rev_flows, colors, fundamental_matrices, rev_upper_sorted_displacements, rev_lower_sorted_displacements, rev_disparity_maps, true);

	start = chrono::system_clock::now();

	cout << "Generating Images..." << endl;

	vector<vector<double>> processing_time(points + 1, vector<double>(points + 1, 0.0));

	auto work = [&](int quadrant)
	{
		//*
		int i0s = 0;
		int i0e = points;
		int i1s = 0;
		int i1e = points;
		/*/
		int quadrant_y = (quadrant - 1) / 2;
		int quadrant_x = (quadrant_y + quadrant) % 2;

		int i0s = (points / 2) * quadrant_y;
		int i0e = (points / 2 + 1) + (points - (points / 2 + 1)) * quadrant_y;
		int i1s = (points / 2) * quadrant_x;
		int i1e = (points / 2 + 1) + (points - (points / 2 + 1)) * quadrant_x;
		//*/

		for (int i0 = i0s; i0 <= i0e; i0++)
		{
			for (int i1 = i1s; i1 <= i1e; i1++)
			{
				auto t0 = chrono::system_clock::now();

				Mat syn = synthesize_view((1.0f * i1) / points, (1.0f * i0) / points, 0, upper_candidates, lower_candidates, rev_upper_candidates, rev_lower_candidates, colors, disparity_maps, rev_disparity_maps, fundamental_matrices);

				auto elapsed = chrono::duration<double>(chrono::system_clock::now() - t0).count();
				processing_time[i0][i1] = elapsed;

				stringstream s_filename;
				s_filename << setw(2) << setfill('0') << i0 << "_" << setw(2) << setfill('0') << i1 << ".jpg";
				imwrite(result_prefix + s_filename.str(), syn);

				// cout << s_filename.str() << " ";
			}
			// cout << endl;
		}
	};

	vector<thread> threads;
	//*
	threads.push_back(move(thread{ work, 1 }));
	/*/
	threads.push_back(move(thread{ work, 1 }));
	threads.push_back(move(thread{ work, 2 }));
	threads.push_back(move(thread{ work, 3 }));
	threads.push_back(move(thread{ work, 4 }));
	//*/

	for (auto& x : threads)
	{
		x.join();
	}

	double elapsed_2 = chrono::duration<double>(chrono::system_clock::now() - start).count();
	cout << "Done. ( " << elapsed_2 << "s )" << endl;
	cout << "Runtime: " << elapsed_1 + elapsed_2 << "s" << endl;

	double sum = 0.0;

	cout << "Processing time per frame (ms):" << endl;
	for (int i0 = 0; i0 < processing_time.size(); i0++)
	{
		cout << "\t";
		for (int i1 = 0; i1 < processing_time[0].size(); i1++)
		{
			cout << int(processing_time[i0][i1] * 1000.0) << "\t";
			sum += processing_time[i0][i1];
		}
		cout << endl;
	}

	double avg = sum / (processing_time.size() * processing_time[0].size());
	cout << "Avg. processing time per frame (ms): " << int(avg * 1000.0);
}