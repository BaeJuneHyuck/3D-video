#define DLIB_JPEG_SUPPORT

#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>

#include "../shared/view_synthesis.h"
#include "../shared/frame_reader.h"

double polygon_area(const vector<double>& x, const vector<double>& y)
{
	double area = 0.0;

	for (int i = 0; i < x.size(); i++)
	{
		int ip = (i + 1) % x.size();
		area += (x[i] + x[ip]) * (y[i] - y[ip]);
	}

	return abs(area / 2.0);
}

double get_face_area(const vector<dlib::full_object_detection>& dets)
{
	unsigned long i = 0;
	if (i < dets.size())
	{
		const dlib::full_object_detection& d = dets[i];

		int indices[] = { 1, 4, 9, 14, 17 };
		vector<double> x0, x1;
		for (auto i : indices)
		{
			x0.push_back(d.part(i).x());
			x1.push_back(d.part(i).y());
		}

		return polygon_area(x0, x1);
	}
	return 0;
}

dlib::dpoint get_eyes_center(const vector<dlib::full_object_detection>& dets)
{
	dlib::dpoint pos{ 0.0, 0.0 };

	unsigned long i = 0;
	if (i < dets.size())
	{
		const dlib::full_object_detection& d = dets[i];

		long left_eye_center_x = 0, left_eye_center_y = 0;

		for (unsigned long i = 37; i <= 41; ++i) {
			left_eye_center_x += d.part(i).x();
			left_eye_center_y += d.part(i).y();
		}

		left_eye_center_x /= 5;
		left_eye_center_y /= 5;

		long right_eye_center_x = 0, right_eye_center_y = 0;

		for (unsigned long i = 43; i <= 47; ++i)
		{
			right_eye_center_x += d.part(i).x();
			right_eye_center_y += d.part(i).y();
		}

		right_eye_center_x /= 5;
		right_eye_center_y /= 5;

		long eyes_center_x = (right_eye_center_x + left_eye_center_x) / 2;
		long eyes_center_y = (right_eye_center_y + left_eye_center_y) / 2;

		pos = dlib::dpoint(eyes_center_x, eyes_center_y);
	}
	return pos;
}

class Synthesizer
{
private:
	vector<Mat> colors;

	vector<Point2f> candidates;

	vector<Mat> fundamental_matrices;

	vector<Mat> disparity_maps;
	vector<Mat> rev_disparity_maps;

public:
	Synthesizer()
	{}

	void deserialize_maps(vector<Mat> bg_maps)
	{
		vector<Mat> maps_to_deserialize;

		for (int i = 0; i < bg_maps.size(); i++)
		{
			Mat disp = Mat::zeros(bg_maps[i].rows, bg_maps[i].cols, CV_32FC1);

			for (int i0 = 0; i0 < disp.rows; i0++)
			{
				for (int i1 = 0; i1 < disp.cols; i1++)
				{
					int intensity = *((unsigned short*)& bg_maps[i].at<Vec3b>(i0, i1));
					disp.at<float>(i0, i1) = (intensity - 32768) / 4.0f;
				}
			}
			maps_to_deserialize.push_back(disp);
		}

		for (int i = 0; i < 4; i++)
			disparity_maps.push_back(maps_to_deserialize[i]);
		for (int i = 4; i < 8; i++)
			rev_disparity_maps.push_back(maps_to_deserialize[i]);
	}

	void init(vector<Mat> _colors, vector<Mat> _fundamental_matrices)
	{
		colors = _colors;
		fundamental_matrices = _fundamental_matrices;

		candidates = create_candidates(colors);
	}

	Mat synthesize(const float camera_pos_ratio_x, const float camera_pos_ratio_y, const float v_z)
	{
		return synthesize_view(camera_pos_ratio_x, camera_pos_ratio_y, v_z, candidates, colors, disparity_maps, rev_disparity_maps, fundamental_matrices);
	}
};

int main(int argc, char* argv[])
{
	cv::VideoCapture cap(0);
	if (!cap.isOpened())
	{
		cerr << "Unable to connect to camera" << endl;
		exit(0);
	}

	dlib::image_window cam_window;
	dlib::image_window render_window;

	dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
	dlib::shape_predictor pose_model;
	try
	{
		dlib::deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;
	}
	catch (dlib::serialization_error& e)
	{
		cout << "Can't find shape_predictor_68_face_landmarks.dat" << endl;
		exit(0);
	}

	FileStorage fm("fundamental_matrices.xml", FileStorage::READ);

	int cam_count;
	fm["cam_count"] >> cam_count;

	int sqrt_cam_count = int(sqrt(cam_count) + 0.05);
	int sqrt_map_count = sqrt_cam_count - 1;
	int map_count = sqrt_map_count * sqrt_map_count;

	FrameReader v_view("views.avi");
	FrameReader v_disp("disparities.avi");

	vector<vector<Mat>> vec_fundamental_matrices;
	for (int i = 0; i < map_count; i++)
	{
		vector<Mat> fundamental_matrices;
		for (int j = 0; j < 4; j++)
		{
			Mat f;
			fm[string("F" + to_string(i) + "_") + to_string(j)] >> f;
			fundamental_matrices.push_back(f);
			cout << f << endl;
		}
		vec_fundamental_matrices.push_back(fundamental_matrices);
	}

	while (!cam_window.is_closed() && !render_window.is_closed())
	{
		cv::Mat frame;
		if (!cap.read(frame))
		{
			break;
		}

		cv::flip(frame, frame, 1); // flip image

		dlib::cv_image<dlib::bgr_pixel> cimg(frame);

		std::vector<dlib::rectangle> faces = detector(cimg);

		std::vector<dlib::full_object_detection> shapes;
		for (unsigned long i = 0; i < faces.size(); ++i)
			shapes.push_back(pose_model(cimg, faces[i]));

		cam_window.clear_overlay();
		cam_window.set_image(cimg);

		float client_ratio = 0.5f;

		int client_height = cimg.nr() * client_ratio;
		int client_width = cimg.nc() * client_ratio;

		int client_left_in_cam = (cimg.nc() - client_width) / 2;
		int client_top_in_cam = (cimg.nr() - client_height) / 2;

		cam_window.add_overlay(dlib::image_display::overlay_rect(dlib::rectangle(dlib::dpoint(client_left_in_cam, client_top_in_cam), dlib::dpoint(client_left_in_cam + client_width, client_top_in_cam + client_height)), dlib::rgb_pixel(0, 255, 0)));

		static dlib::dpoint last_eyes_center{ 0.0, 0.0 };
		dlib::dpoint current_eyes_center = get_eyes_center(shapes);
		if (!(current_eyes_center.x() == 0.0 && current_eyes_center.y() == 0.0))
			last_eyes_center = current_eyes_center;

		cam_window.add_overlay(dlib::image_window::overlay_circle(last_eyes_center, 3, dlib::rgb_pixel(0, 255, 0)));

		static double last_face_area = 0.0;
		double current_face_area = get_face_area(shapes);
		if (current_face_area != 0.0)
			last_face_area = current_face_area;

		auto face_distance = 10000.0 / sqrt(last_face_area);

		// float v_z = 2 * (face_distance - 60.0) / 60.0 / 640.0;

		// !! disable z depth
		float avg_v_z = 0;

		/*
		static deque<float> acc_v_z;
		static float sum_v_z = 0.0f;
		acc_v_z.push_back(v_z);
		sum_v_z += v_z;
		if (acc_v_z.size() > 15)
		{
			float first = acc_v_z[0];
			acc_v_z.pop_front();
			sum_v_z -= first;
		}
		float avg_v_z = sum_v_z / acc_v_z.size();
		*/


		auto distance_string = std::to_string(face_distance);

		int client_eyes_x = last_eyes_center.x() - client_left_in_cam;
		int client_eyes_y = last_eyes_center.y() - client_top_in_cam;

		// !! temporary margin 2 for outside
		client_eyes_x = max(min(client_eyes_x, client_width - 2), 2);
		client_eyes_y = max(min(client_eyes_y, client_height - 2), 2);

		int map_index = int(client_eyes_y * sqrt_map_count / client_height) * sqrt_map_count + int(client_eyes_x * sqrt_map_count / client_width);
		map_index = max(0, min(map_index, map_count - 1));

		float r0 = float(client_eyes_y) * sqrt_map_count / client_height - int(client_eyes_y * sqrt_map_count / client_height);
		float r1 = float(client_eyes_x) * sqrt_map_count / client_width - int(client_eyes_x * sqrt_map_count / client_width);

		r0 = max(min(r0, 1.0f), 0.0f);
		r1 = max(min(r1, 1.0f), 0.0f);

		auto start = chrono::system_clock::now();
		static int frame_index = 0;

		int top_left = (map_index / sqrt_map_count) * sqrt_cam_count + (map_index % sqrt_map_count);

		vector<int> view_indices = { top_left, top_left + 1, top_left + sqrt_cam_count, top_left + sqrt_cam_count + 1 };

		vector<Mat> colors, bg_maps;
		for (int i = 0; i < 4; i++)
			colors.push_back(v_view.get_image_matrix(cam_count* frame_index + view_indices[i]));

		for (int i = 0; i < 8; i++)
			bg_maps.push_back(v_disp.get_image_matrix(frame_index * 8 * map_count + map_index * 8 + i));

		Synthesizer s;
		s.init(colors, vec_fundamental_matrices[map_index]);

		s.deserialize_maps(bg_maps);

		Mat syn = s.synthesize(r1, r0, avg_v_z);
		frame_index = (frame_index + 1) % (min(v_view.total_frames() / cam_count, v_disp.total_frames() / sqrt_map_count / sqrt_map_count / 2));

		auto elapsed = chrono::duration<double>(chrono::system_clock::now() - start).count();
		cam_window.add_overlay(dlib::image_window::overlay_rect(dlib::rectangle(0, 0, 0, 0), dlib::rgb_pixel(255, 0, 0), (distance_string + "\n" + to_string(int(1000.0 * elapsed))).c_str()));

		dlib::array2d<dlib::bgr_pixel> dlib_rendered_view;
		dlib::assign_image(dlib_rendered_view, dlib::cv_image<dlib::bgr_pixel>(syn));

		render_window.set_image(dlib_rendered_view);
	}
}
