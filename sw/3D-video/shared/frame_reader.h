#pragma once

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavdevice/avdevice.h>
#include <libavfilter/avfilter.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>
#include <libswresample/swresample.h>
}

#include <opencv2/core.hpp>

#include <string>
#include <iostream>

using namespace std;
using namespace cv;

class FrameReader
{
private:
	AVFormatContext* fmtCtx = NULL;

	AVStream* vStream;
	AVCodecParameters* vPara;
	AVCodec* vCodec;
	AVCodecContext* vCtx;

	SwsContext* swsCtx = NULL;

	int vidx;

public:
	FrameReader(const string filename)
	{
		if (avformat_open_input(&fmtCtx, filename.c_str(), NULL, NULL) != 0)
		{
			cout << "Failed to open movie." << endl;
			exit(0);
		}

		avformat_find_stream_info(fmtCtx, NULL);

		vidx = av_find_best_stream(fmtCtx, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);

		vStream = fmtCtx->streams[vidx];
		vPara = vStream->codecpar;
		vCodec = avcodec_find_decoder(vPara->codec_id);
		vCtx = avcodec_alloc_context3(vCodec);
		avcodec_parameters_to_context(vCtx, vPara);
		avcodec_open2(vCtx, vCodec, NULL);

		avcodec_flush_buffers(vCtx);
	}

	int64_t total_frames()
	{
		return fmtCtx->streams[vidx]->nb_frames;
	}

	Mat get_image_matrix(int fn)
	{
		AVPacket packet = { 0, };

		auto timeBase = (int64_t(fmtCtx->streams[vidx]->time_base.num) * AV_TIME_BASE) / int64_t(fmtCtx->streams[vidx]->time_base.den);
		int64_t seekTarget = (int64_t(fn) * timeBase * fmtCtx->streams[vidx]->time_base.den) / fmtCtx->streams[vidx]->avg_frame_rate.num;

		av_seek_frame(fmtCtx, -1, seekTarget, AVSEEK_FLAG_ANY);
		av_read_frame(fmtCtx, &packet);

		if (avcodec_send_packet(vCtx, &packet) != 0)
		{
			cout << "failed to send packet" << endl;
			exit(0);
		}

		AVFrame vFrame = { 0, };

		if (avcodec_receive_frame(vCtx, &vFrame) == AVERROR(EAGAIN))
		{
			cout << "failed to receive frame" << endl;
			exit(0);
		}

		uint8_t* rgbbuf = NULL;

		AVFrame RGBFrame = { 0, };

		if (swsCtx == NULL) {
			swsCtx = sws_getContext(
				vFrame.width, vFrame.height, AVPixelFormat(vFrame.format),
				vFrame.width, vFrame.height, AV_PIX_FMT_BGR24,
				SWS_BICUBIC, NULL, NULL, NULL);
			int rasterbufsize = av_image_get_buffer_size(AV_PIX_FMT_BGR24,
				vFrame.width, vFrame.height, 1);
			rgbbuf = (uint8_t*)av_malloc(rasterbufsize);
			av_image_fill_arrays(RGBFrame.data, RGBFrame.linesize, rgbbuf,
				AV_PIX_FMT_BGR24, vFrame.width, vFrame.height, 1);
		}
		sws_scale(swsCtx, vFrame.data, vFrame.linesize, 0, vFrame.height,
			RGBFrame.data, RGBFrame.linesize);

		unsigned char* raster = RGBFrame.data[0];

		Mat img(vFrame.height, vFrame.width, CV_8UC3);

		memcpy(img.data, raster, vFrame.height * vFrame.width * 3);

		av_packet_unref(&packet);
		av_frame_unref(&vFrame);

		sws_freeContext(swsCtx);
		swsCtx = NULL;
		av_free(rgbbuf);

		return img;
	}
};