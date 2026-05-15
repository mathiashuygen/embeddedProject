#pragma once
#include <Arducam_Mega.h>
#include <SPI.h>
 
#define CAM_CS 7
static const uint32_t BAUD  = 921600;
static const uint8_t CHUNK = 64;
 
extern Arducam_Mega myCamera;
 
void takePicArdu();
