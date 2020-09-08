#define DLIB_JPEG_SUPPORT

#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>

#include "view.h"

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

	for (unsigned long i = 0; i < dets.size(); ++i)
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
	cout << "  Stride: " << stride << endl;
	cout << "  Block: (" << block[0] << ", " << block[1] << ")" << endl;

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

	cv::VideoCapture cap(0);
	if (!cap.isOpened())
	{
		cerr << "Unable to connect to camera" << endl;
		return 0;
	}

	dlib::image_window cam_window;
	dlib::image_window render_window;

	dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
	dlib::shape_predictor pose_model;
	dlib::deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;

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

		int cam_height = cimg.nr();
		int cam_width = cimg.nc();

		cam_window.add_overlay(dlib::image_display::overlay_rect(dlib::rectangle(dlib::dpoint(cam_width / 4, cam_height / 4), dlib::dpoint(cam_width / 4 + cam_width / 2, cam_height / 4 + cam_height / 2)), dlib::rgb_pixel(0, 255, 0)));

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

		//float v_z = max((face_distance - 60.0) / 10000.0, 0.0);
		float v_z = 0;

		auto distance_string = std::to_string(face_distance);

		float r0 = (last_eyes_center.y() - cam_height / 4) / (cam_height / 2), r1 = (last_eyes_center.x() - cam_height / 4) / (cam_width / 2);
		r0 = max(min(r0, 1.0f), 0.0f);
		r1 = max(min(r1, 1.0f), 0.0f);

		auto start = chrono::system_clock::now();
		Mat syn = synthesize_view(r1, r0, v_z, upper_candidates, lower_candidates, rev_upper_candidates, rev_lower_candidates, colors, disparity_maps, rev_disparity_maps, fundamental_matrices);
		auto elapsed = chrono::duration<double>(chrono::system_clock::now() - start).count();

		cam_window.add_overlay(dlib::image_window::overlay_rect(dlib::rectangle(0, 0, 0, 0), dlib::rgb_pixel(255, 0, 0), (distance_string + "\n" + to_string(int(1000.0 * elapsed))).c_str()));

		dlib::array2d<dlib::bgr_pixel> dlib_rendered_view;
		dlib::assign_image(dlib_rendered_view, dlib::cv_image<dlib::bgr_pixel>(syn));

		render_window.set_image(dlib_rendered_view);
	}
}