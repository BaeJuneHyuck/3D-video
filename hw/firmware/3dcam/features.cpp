#include <ArduinoWebsockets.h>
#include "esp_timer.h"
#include "esp_camera.h"
#include "img_converters.h"
#include "Arduino.h"

#include "fb_gfx.h"

#include "SdFat.h"
#include "sdios.h"
#include "FreeStack.h"

using namespace websockets;

extern SdFat32 sd;
extern File32 file;

uint8_t  sectorBuffer[512];

void format_card(WebsocketsClient& client) {
  FatFormatter fatFormatter;

  if (!fatFormatter.format(sd.card(), sectorBuffer, &Serial)) {
    client.send("Format: failed.");
    return;
  }
  client.send("Format: succeeded.");
}

void set_state(WebsocketsClient& client, String var, String value)
{
    const char* variable = var.c_str();
    int val = atoi(value.c_str());
    sensor_t * s = esp_camera_sensor_get();

    int res = 0;
    if(!strcmp(variable, "framesize"))
    {
        if(s->pixformat == PIXFORMAT_JPEG) res = s->set_framesize(s, (framesize_t)val);
    }
    else if(!strcmp(variable, "quality")) res = s->set_quality(s, val);
    else if(!strcmp(variable, "contrast")) res = s->set_contrast(s, val);
    else if(!strcmp(variable, "brightness")) res = s->set_brightness(s, val);
    else if(!strcmp(variable, "saturation")) res = s->set_saturation(s, val);
    else if(!strcmp(variable, "gainceiling")) res = s->set_gainceiling(s, (gainceiling_t)val);
    else if(!strcmp(variable, "colorbar")) res = s->set_colorbar(s, val);
    else if(!strcmp(variable, "awb")) res = s->set_whitebal(s, val);
    else if(!strcmp(variable, "agc")) res = s->set_gain_ctrl(s, val);
    else if(!strcmp(variable, "aec")) res = s->set_exposure_ctrl(s, val);
    else if(!strcmp(variable, "hmirror")) res = s->set_hmirror(s, val);
    else if(!strcmp(variable, "vflip")) res = s->set_vflip(s, val);
    else if(!strcmp(variable, "awb_gain")) res = s->set_awb_gain(s, val);
    else if(!strcmp(variable, "agc_gain")) res = s->set_agc_gain(s, val);
    else if(!strcmp(variable, "aec_value")) res = s->set_aec_value(s, val);
    else if(!strcmp(variable, "aec2")) res = s->set_aec2(s, val);
    else if(!strcmp(variable, "dcw")) res = s->set_dcw(s, val);
    else if(!strcmp(variable, "bpc")) res = s->set_bpc(s, val);
    else if(!strcmp(variable, "wpc")) res = s->set_wpc(s, val);
    else if(!strcmp(variable, "raw_gma")) res = s->set_raw_gma(s, val);
    else if(!strcmp(variable, "lenc")) res = s->set_lenc(s, val);
    else if(!strcmp(variable, "special_effect")) res = s->set_special_effect(s, val);
    else if(!strcmp(variable, "wb_mode")) res = s->set_wb_mode(s, val);
    else if(!strcmp(variable, "ae_level")) res = s->set_ae_level(s, val);
    else
    {
        client.send("Set_State: unknown state name.");
        return;
    }

    if(!res)
      client.send("Set_State: state changed.");
    else
      client.send("Set_State: failed.");
}

void show_state(WebsocketsClient& client)
{
    static char json_response[1024];

    sensor_t * s = esp_camera_sensor_get();
    char * p = json_response;
    *p++ = '{';

    p+=sprintf(p, "\"framesize\":%u,", s->status.framesize);
    p+=sprintf(p, "\"quality\":%u,", s->status.quality);
    p+=sprintf(p, "\"brightness\":%d,", s->status.brightness);
    p+=sprintf(p, "\"contrast\":%d,", s->status.contrast);
    p+=sprintf(p, "\"saturation\":%d,", s->status.saturation);
    p+=sprintf(p, "\"sharpness\":%d,", s->status.sharpness);
    p+=sprintf(p, "\"special_effect\":%u,", s->status.special_effect);
    p+=sprintf(p, "\"wb_mode\":%u,", s->status.wb_mode);
    p+=sprintf(p, "\"awb\":%u,", s->status.awb);
    p+=sprintf(p, "\"awb_gain\":%u,", s->status.awb_gain);
    p+=sprintf(p, "\"aec\":%u,", s->status.aec);
    p+=sprintf(p, "\"aec2\":%u,", s->status.aec2);
    p+=sprintf(p, "\"ae_level\":%d,", s->status.ae_level);
    p+=sprintf(p, "\"aec_value\":%u,", s->status.aec_value);
    p+=sprintf(p, "\"agc\":%u,", s->status.agc);
    p+=sprintf(p, "\"agc_gain\":%u,", s->status.agc_gain);
    p+=sprintf(p, "\"gainceiling\":%u,", s->status.gainceiling);
    p+=sprintf(p, "\"bpc\":%u,", s->status.bpc);
    p+=sprintf(p, "\"wpc\":%u,", s->status.wpc);
    p+=sprintf(p, "\"raw_gma\":%u,", s->status.raw_gma);
    p+=sprintf(p, "\"lenc\":%u,", s->status.lenc);
    p+=sprintf(p, "\"vflip\":%u,", s->status.vflip);
    p+=sprintf(p, "\"hmirror\":%u,", s->status.hmirror);
    p+=sprintf(p, "\"dcw\":%u,", s->status.dcw);
    p+=sprintf(p, "\"colorbar\":%u,", s->status.colorbar);
    *p++ = '}';
    *p++ = 0;
    client.send(json_response);
}

struct chunk_header
{
  int64_t timestamp;
  int64_t len;
};

extern char* file_buf;

int start_record(WebsocketsClient& client, int64_t record_time){
    int64_t start_time = esp_timer_get_time();
    camera_fb_t * fb = NULL;
    size_t _jpg_buf_len = 0;
    uint8_t * _jpg_buf = NULL;

    client.send("Record: starting the record...");

    if (!file.open("jpg_blob", O_RDWR | O_CREAT | O_TRUNC))
    {
      client.send("Record: file open failed.");
      file.close();
      return false;
    }
    
    int64_t last_frame = esp_timer_get_time();

    int sum_frame_time = 0;
    int frame_count = 0;

    while(esp_timer_get_time() - start_time < record_time * 1000)
    {
      chunk_header header;
      header.timestamp = esp_timer_get_time() - start_time;
      
      fb = esp_camera_fb_get();
      if (!fb) {
          client.send("Record: failed to obtain a frame.");
          file.close();
          return false;
      } else {
          _jpg_buf_len = fb->len;
          _jpg_buf = fb->buf;
      }

      header.len = _jpg_buf_len;

      memcpy(file_buf, &header, sizeof(header));
      memcpy(file_buf + sizeof(header), _jpg_buf, _jpg_buf_len);
      
      if(file.write(file_buf, sizeof(header) + _jpg_buf_len) != sizeof(header) + _jpg_buf_len)
      {
        client.send("Record: write failed.");
        esp_camera_fb_return(fb);
        file.close();
        return false;
      }

      file.flush();
      esp_camera_fb_return(fb);
      
      int64_t fr_end = esp_timer_get_time();
      int64_t frame_time = (fr_end - last_frame) / 1000;
      last_frame = fr_end;
      sum_frame_time += frame_time;
      frame_count++;
    }

    client.send(String("Record: avg frame time = ") + String((double)sum_frame_time / frame_count));

    file.close();
    return true;
}
