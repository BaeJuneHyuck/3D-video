#include <iostream>
#include <string>
#include <vector>
#include <array>
#include <tuple>
#include <opencv2/opencv.hpp>
#include <algorithm>

using namespace std;
using namespace cv;
typedef array<int, 2> ai2;

// < config below >---

const int points = 8;
const vector<vector<float>> cutoff{{64.0f, 128.0f}, {128.0f, 64.0f}};
const int stride = 4;
const vector<int> block{8, 8};
const string result_prefix{ "res/syn_" };

const double farneback_pyr_scale = 0.5;
const int farneback_levels = 4;
const int farneback_winsize = 512;
const int farneback_iterations = 4;
const int farneback_poly_n = 5;
const double farneback_poly_sigma = 1.2;
const int farneback_flags = OPTFLOW_FARNEBACK_GAUSSIAN;

// ---< config above >

int i0s, i0e, i1s, i1e;

vector<Mat> grays;
vector<Mat> colors;

vector<Mat> flows;

int main(int argc, char* argv[])
{
    if(argc - 1 != 4)
    {
        cerr << "Usage: " << argv[0] << " image1 image2 image3 quadrant( 1 ~ 4 )" << endl;
        return 0;
    }

    for(int i = 1; i <= 3; i++)
    {
        string s{ argv[i] };
        grays.push_back(imread(s, IMREAD_GRAYSCALE));
        colors.push_back(imread(s, IMREAD_UNCHANGED));
    }

	int quadrant = stoi(argv[4]);
	if (!(1 <= quadrant && quadrant <= 4))
	{
		cerr << "quadrant must be 1 ~ 4" << endl;
		return 0;
	}

	int quadrant_y = (quadrant - 1) / 2;
	int quadrant_x = (quadrant_y + quadrant) % 2;

	i0s = (points / 2) * quadrant_y;
	i0e = (points / 2 + 1) + (points - (points / 2 + 1)) * quadrant_y;
	i1s = (points / 2) * quadrant_x;
	i1e = (points / 2 + 1) + (points - (points / 2 + 1)) * quadrant_x;

    auto width = grays[0].cols, height = grays[0].rows;

    for(int i = 1; i < grays.size(); i++)
    {
        if(grays[i].cols != width || grays[i].rows != height)
        {
            cerr << "Image dimension must be same" << endl;
            return 0;
        }
    }

    cout << "Starting view synthesis:" << endl;
    cout << "  Points: " << points << endl;
    cout << "  Cutoffs: (" << cutoff[0][0] << ", " << cutoff[0][1] << "), (" << cutoff[1][0] << ", " << cutoff[1][1] << ")" << endl;
    cout << "  Stride: " << stride << endl;
    cout << "  Block: (" << block[0] << ", " << block[1] << ")" << endl;
	cout << "  Area: (" << i0s << ":" << i0e << ", " << i1s << ":" << i1e << ")" << endl;
	cout << "  Result Prefix: " << result_prefix << endl;

    for(auto x : vector<vector<int>>{{1, 2, 0}, {1, 0, 1}})
    {
        Mat flow;
        calcOpticalFlowFarneback(grays[x[0]], grays[x[1]], flow, farneback_pyr_scale, farneback_levels, farneback_winsize, farneback_iterations, farneback_poly_n, farneback_poly_sigma, farneback_flags);
        auto x_cutoff = cutoff[x[2]][0];
        auto y_cutoff = cutoff[x[2]][1];
        for(auto it = flow.begin<Point2f>(); it != flow.end<Point2f>(); ++it)
        {
            (*it).x = min(x_cutoff, (*it).x);
            (*it).y = min(y_cutoff, (*it).y);
            (*it).x = max(-x_cutoff, (*it).x);
            (*it).y = max(-y_cutoff, (*it).y);
        }
        flows.push_back(flow);
    }

    for(int i0 = i0s; i0 <= i0e; i0++)
    {
        for(int i1 = i1s; i1 <= i1e; i1++)
        {
            auto ratio_ortho = vector<float>{ (1.0f * i0) / points, (1.0f * i1) / points };
            auto ratio = 0.5f + 0.5f * (ratio_ortho[1] - ratio_ortho[0]);
            Mat syn_i0, syn_i1, syn;
            addWeighted(colors[1], 1.0 - ratio_ortho[0], colors[2], ratio_ortho[0], 0, syn_i0);
            addWeighted(colors[1], 1.0 - ratio_ortho[1], colors[0], ratio_ortho[1], 0, syn_i1);
            addWeighted(syn_i0, 1.0 - ratio, syn_i1, ratio, 0.0, syn);

            vector<Mat> sf(2);
            for(int j = 0; j < flows.size(); ++j)
            {
                flows[j].copyTo(sf[j]);
                auto& s = sf[j];
                for(auto it = s.begin<Point2f>(); it != s.end<Point2f>(); ++it)
                {
                    (*it).x *= ratio_ortho[j];
                    (*it).y *= ratio_ortho[j];
                }
            }

            vector<tuple<float, ai2>> sorted_displacements;

            for(int j0 = 0; j0 < sf[0].rows; j0 += stride)
            {
                for(int j1 = 0; j1 < sf[0].cols; j1 += stride)
                {
                    auto dx = sf[0].at<Point2f>(j0, j1).x + sf[1].at<Point2f>(j0, j1).x;
                    auto dy = sf[0].at<Point2f>(j0, j1).y + sf[1].at<Point2f>(j0, j1).y;
                    auto dist = sqrt(dx * dx + dy * dy);

                    auto x = j1, y = j0;

                    sorted_displacements.push_back(tuple<float, ai2>{ dist, ai2{y, x} });
                }
            }

            sort(sorted_displacements.begin(), sorted_displacements.end(), [](tuple<float, ai2> a0, tuple<float, ai2> a1)
            {
                return get<0>(a0) > get<0>(a1);
            });

            for(auto a : sorted_displacements)
            {
                int y = get<1>(a)[0];
                int x = get<1>(a)[1];

                auto xp0 = x + int(sf[0].at<Point2f>(y, x).x);
                auto yp0 = y + int(sf[0].at<Point2f>(y, x).y);

                auto xpp0 = x + int(flows[0].at<Point2f>(y, x).x);
                auto ypp0 = y + int(flows[0].at<Point2f>(y, x).y);

                auto xp1 = x + int(sf[1].at<Point2f>(y, x).x);
                auto yp1 = y + int(sf[1].at<Point2f>(y, x).y);

                auto xpp1 = x + int(flows[1].at<Point2f>(y, x).x);
                auto ypp1 = y + int(flows[1].at<Point2f>(y, x).y);

                auto xp = int((1 - ratio) * xp0 + ratio * xp1);
                auto yp = int((1 - ratio) * yp0 + ratio * yp1);
                
                auto hbx = block[0] / 2, hby = block[1] / 2;

                if((0 <= xpp0 - hbx && xpp0 + hbx <= width - 1) &&
                   (0 <= ypp0 - hby && ypp0 + hby <= height - 1) &&
                   (0 <= xpp1 - hbx && xpp1 + hbx <= width - 1) &&
                   (0 <= ypp1 - hby && ypp1 + hby <= height - 1) &&
                   (0 <= xp - hbx && xp + hbx <= width - 1) &&
                   (0 <= yp - hby && yp + hby <= height - 1) &&
                   (0 <= x - hbx && x + hbx <= width - 1) &&
                   (0 <= y - hby && y + hby <= height - 1))

                 {
                     auto origin = colors[1](Range(y - hby, y + hby), Range(x - hbx, x + hbx));
                     auto down = colors[2](Range(ypp0 - hby, ypp0 + hby), Range(xpp0 - hbx, xpp0 + hbx));
                     auto right = colors[0](Range(ypp1 - hby, ypp1 + hby), Range(xpp1 - hbx, xpp1 + hbx));
                     Mat md, mr, mt;
                     addWeighted(origin, 1.0 - ratio_ortho[0], down, ratio_ortho[0], 0, md);
                     addWeighted(origin, 1.0 - ratio_ortho[1], right, ratio_ortho[1], 0, mr);
                     addWeighted(md, 1.0 - ratio, mr, ratio, 0, mt);
					 mt.copyTo(syn(Range(yp - hby, yp + hby), Range(xp - hbx, xp + hbx)));
                 }
            }
			stringstream s_filename;
			s_filename << setw(2) << setfill('0') << i0 << "_" << setw(2) << setfill('0') << i1 << ".jpg";
            imwrite(result_prefix + s_filename.str(), syn);
			cout << s_filename.str() << " ";
        }
        cout << endl;
    }
}
