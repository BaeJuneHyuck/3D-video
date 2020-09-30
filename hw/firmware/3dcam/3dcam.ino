#include "esp_camera.h"
#include <WiFi.h>
#include "time.h"

#include <ArduinoWebsockets.h>
using namespace websockets;
WebsocketsServer server;

#include <SPI.h>
#include "SdFat.h"
#include "sdios.h"
#include "FreeStack.h"

#include <WebServer.h>

#define SPI_CLOCK SD_SCK_MHZ(25)
#define SD_CONFIG SdSpiConfig(13, DEDICATED_SPI, SPI_CLOCK)

SdFat sd;
File file;

#define CAMERA_MODEL_AI_THINKER

#include "camera_pins.h"

// -- < config below > --
const char* ssid = "nausea";
const char* password = "3dimension";

const long  gmtOffset_sec = 9 * 3600;
const int   daylightOffset_sec = 0;

// -- < config above > --

void erase_card(WebsocketsClient& client);
void format_card(WebsocketsClient& client);
int start_record(WebsocketsClient& client, int64_t record_time);
void set_state(WebsocketsClient& client, String var, String val);
void show_state(WebsocketsClient& client);

WebServer webServer(81);

String uint64String(uint64_t input)
{
  String result = "";
  uint8_t base = 10;

  do {
    char c = input % base;
    input /= base;

    if (c < 10)
      c +='0';
    else
      c += 'A' - 10;
    result = c + result;
  } while (input);
  return result;
}

uint64_t stringUint64(String input)
{
  const char* str = input.c_str();
  uint64_t result = 0;
  for (int i = 0; str[i] != '\0'; ++i)
    result = result * 10 + str[i] - '0';
  return result;
}

char* file_buf;
void setup()
{
  Serial.begin(115200);
  Serial.setDebugOutput(true);
  Serial.println();

  SPI.begin(14,2,15,13);

  if (!sd.begin(SD_CONFIG))
    sd.initErrorHalt(&Serial);

  pinMode(4, OUTPUT);
  
  Serial.printf("Card size: %lf GB\n", sd.card()->sectorCount()*512E-9);
  
  camera_config_t config; 
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 10000000;
  config.pixel_format = PIXFORMAT_JPEG;

  if(psramFound())
  {
    config.frame_size = FRAMESIZE_UXGA;
    config.jpeg_quality = 10;
    config.fb_count = 2;
  }
  else
  {
    config.frame_size = FRAMESIZE_SVGA;
    config.jpeg_quality = 12;
    config.fb_count = 1;
  }

  delay(500);

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK)
    return;

  sensor_t * s = esp_camera_sensor_get();

  s->set_framesize(s, FRAMESIZE_VGA);
  s->set_quality(s, 2);

  file_buf = (char*) ps_malloc(256 * 1024);

  if(!file_buf)
  {
    Serial.println("Malloc failed.");
    return;
  }
  
  while (WiFi.status() != WL_CONNECTED)
  {
    Serial.print("Trying to begin WiFi...");
    WiFi.begin(ssid, password);
  
    while (WiFi.status() != WL_CONNECTED && WiFi.status() != WL_CONNECT_FAILED)
    {
      delay(1000);
      Serial.print(WiFi.status());
    }
  }
  
  Serial.println("");
  Serial.println("WiFi connected");

  WiFi.setHostname(WiFi.localIP().toString().c_str());
  
  server.listen(80);
  
  Serial.print("'ws://");
  Serial.print(WiFi.localIP());
  Serial.println("' to connect");  
}

void handleDownload()
{
  file.open("/jpg_blob", O_RDONLY);
  webServer.streamFile(file, "application/octet-stream");
  file.close();
}

void loop()
{
  delay(1000);

  WebsocketsClient client = server.accept();
  while(client.available())
  {
    WebsocketsMessage msg = client.readBlocking();

    String string = String(msg.data().c_str());
    if(string.startsWith("ntp "))
    {
      configTime(gmtOffset_sec, daylightOffset_sec, string.substring(4).c_str());
      client.send("NTP: Trying to sync...");
    }
    else if(string == String("cpu_time"))
    {
      struct timeval tv;
      gettimeofday(&tv, NULL);
      int64_t cpu_time = ((int64_t)tv.tv_sec * 1000000L + (int64_t)tv.tv_usec) / 1000;
      client.send(String("CPU_Time: ") + uint64String(cpu_time));
    }
    else if(string.startsWith("start_at "))
    {
      String start_length = string.substring(9);
      int delim_pos = start_length.indexOf(" ");
      if(delim_pos > 0)
      {
        int64_t target_time = stringUint64(start_length.substring(0, delim_pos));
        int64_t len = stringUint64(start_length.substring(delim_pos + 1));
        struct timeval tv;
        int64_t cpu_time;
        do
        {
          delay(1);
          gettimeofday(&tv, NULL);
          cpu_time = ((int64_t)tv.tv_sec * 1000000L + (int64_t)tv.tv_usec) / 1000;
        } while(cpu_time < target_time);
  
        int ret = start_record(client, len);
  
        gettimeofday(&tv, NULL);
        cpu_time = ((int64_t)tv.tv_sec * 1000000L + (int64_t)tv.tv_usec) / 1000;
        if(ret)
          client.send(String("Record: done, CPU time (ms) = ") + uint64String(cpu_time));
        else
          client.send(String("Record: done, CPU time (ms) = ") + uint64String(cpu_time));
      }
    }
    else if(string.startsWith("set_state "))
    {
      String variable_value = string.substring(10);
      int delim_pos = variable_value.indexOf(" ");
      if(delim_pos > 0)
      {
        set_state(client, variable_value.substring(0, delim_pos), variable_value.substring(delim_pos + 1));
      }
    }
    else if(string == String("show_state"))
    {
      show_state(client);
    }
    else if(string == String("led_on"))
    {
      digitalWrite(4, HIGH);
    }
    else if(string == String("led_off"))
    {
      digitalWrite(4, LOW);
    }
    else if(string == String("format"))
    {
      format_card(client);
    }
    else if(string == String("web_mode"))
    {
      client.send("Web: Starting Web Mode...");
      webServer.onNotFound(handleDownload);
      webServer.begin();

      client.send("Web: Starting inf. loop...");
      for (;;) {
        webServer.handleClient();
        delay(10);
      }
    }
  }
  client.close();
}
