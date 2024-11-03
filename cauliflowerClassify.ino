#include <TinyMLShield.h>
#include <tensorflow/lite/c/common.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_error_reporter.h> // Add this line
#include "cauliflowerModel1.h"  // Ensure this file correctly defines cauliflower_model_data

// Declare global variables
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* tflite_model = nullptr; // Renamed to avoid conflict
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input_tensor;
TfLiteTensor* output_tensor;

constexpr int kTensorArenaSize = 136 * 1024; // Adjust size as necessary
uint8_t tensor_arena[kTensorArenaSize];
uint8_t image[128 * 128 * 3]; // Buffer for image (width * height * channels)
bool liveFlag = true; // Control flag for live processing
int bytesPerFrame; // Will be set in setup()

void initializeModel() {
  // Correct usage of Error Reporter
  static tflite::MicroErrorReporter micro_error_reporter; // Ensure this type is defined
  error_reporter = &micro_error_reporter;

  // Ensure cauliflower_model_data is defined in cauliflowerModel1.h
  tflite_model = tflite::GetModel(model_data); // Reference to model data
  if (tflite_model == nullptr) {
    error_reporter->Report("Failed to get model data");
    return; // Exit if model data is invalid
  }

  static tflite::MicroMutableOpResolver<10> resolver;
  resolver.AddConv2D();
  resolver.AddSoftmax();
  resolver.AddFullyConnected();

  // Initialize interpreter
  static tflite::MicroInterpreter static_interpreter(
      tflite_model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    error_reporter->Report("Failed to allocate tensors!");
    return; // Exit if tensor allocation fails
  }

  input_tensor = interpreter->input(0);
  output_tensor = interpreter->output(0);
}

// Preprocess the image for the model
void preprocessImage(uint8_t* image) {
  int height = 128;
  int width = 128;
  int channels = 3;

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      uint16_t rgb565 = ((uint16_t*)image)[y * width + x];
      uint8_t r = (rgb565 >> 11) & 0x1F;  // Extract red
      uint8_t g = (rgb565 >> 5) & 0x3F;   // Extract green
      uint8_t b = rgb565 & 0x1F;          // Extract blue

      // Convert to 8-bit values
      r = (r << 3) | (r >> 2);
      g = (g << 2) | (g >> 4);
      b = (b << 3) | (b >> 2);

      int index = (y * width + x) * channels;
      input_tensor->data.uint8[index] = r;
      input_tensor->data.uint8[index + 1] = g;
      input_tensor->data.uint8[index + 2] = b;
    }
  }
}

// Get classification result
int getClassification() {
  int category = 0;
  float max_prob = output_tensor->data.f[0];
  for (int i = 1; i < output_tensor->dims->data[1]; i++) {
    if (output_tensor->data.f[i] > max_prob) {
      max_prob = output_tensor->data.f[i];
      category = i;
    }
  }
  return category;
}

void setup() {
  Serial.begin(9600);
  while (!Serial); // Wait for serial monitor to open

  initializeShield();
  initializeModel();

  if (!Camera.begin(QCIF, RGB565, 1, OV7675)) {
    Serial.println("Failed to initialize camera");
    while (1); // Halt if camera fails to initialize
  }
  bytesPerFrame = Camera.width() * Camera.height() * Camera.bytesPerPixel();

  Serial.println("OV7675 camera test with live classification\n");
  Serial.println("Press the button on the shield to capture an image.");
}

void loop() {
  if (liveFlag) {
    Camera.readFrame(image);
    preprocessImage(image);
    interpreter->Invoke();

    int category = getClassification();
    Serial.print("Classified as category: ");
    Serial.println(category);
    delay(1000); // Adjust delay as needed

    if (readShieldButton()) {
      Serial.println("Button pressed! Capturing image...");
      // Additional image handling code if necessary
    }
  }
}
