/**
 * Stream Edge Impulse's FOMO results to the browser.
 
 * BE SURE TO SET "TOOLS > CORE DEBUG LEVEL = INFO"
 * to turn on debug messages
 */
#define WIFI_SSID "DIGI_4d7a88"
#define WIFI_PASS "5b2eb285"
#define HOSTNAME "esp32cam"

#include <ETTI_UPB-project-1_inferencing.h>
#include "edge-impulse-sdk/dsp/image/image.hpp"
#include <eloquent_esp32cam.h>
#include <eloquent_esp32cam/viz/ei/fomo_stream.h>

using eloq::camera;
using eloq::wifi;
using eloq::viz::ei::fomoStream;


/**
 *
 */
void setup() {
    delay(3000);
    Serial.begin(115200);
    Serial.println("__EDGE IMPULSE FOMO STREAM__");

 
     camera.pinout.aithinker();
    camera.brownout.disable();
    camera.resolution.qqvga();

    // init camera
    while (!camera.begin().isOk())
        Serial.println(camera.exception.toString());

    // connect to WiFi
    while (!wifi.connect().isOk())
      Serial.println(wifi.exception.toString());

    // init FOMO stream http server
    while (!fomoStream.begin().isOk())
        Serial.println(fomoStream.exception.toString());

    Serial.println("Camera OK");
    Serial.println("FOMO Stream Server OK");
    Serial.println(fomoStream.address());
    Serial.println("Put object in front of camera");
}

/**
 *
 */
void loop() {
    // HTTP server runs in a task, no need to do anything here
}


