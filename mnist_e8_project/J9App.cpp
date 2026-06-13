// J9App.cpp — MNIST inference on Alif E8 (Cortex-M55 + Ethos-U85)
// Runtime: TensorFlow Lite for Microcontrollers (TFLM)
// NPU:     Ethos-U85 via TFLM custom-op integration

#include <cstdint>
#include <cstdio>
#include <cstring>

// TFLM headers
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Ethos-U core driver
#include "ethosu_driver.h"

// Vela-compiled model (model_data.cpp)
#include "model_data.h"

// ── Tensor arena: 128 KB ─────────────────────────────────────────────────────
// Increase to 256*1024 if AllocateTensors() fails at runtime.
constexpr size_t kTensorArenaSize = 128 * 1024;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

// ── Ethos-U85 NPU base address on Alif E8 ────────────────────────────────────
// Verify against your Alif BSP / device header (ETHOS_U_NPU_BASE).
#ifndef ETHOS_U_BASE_ADDRESS
#define ETHOS_U_BASE_ADDRESS  reinterpret_cast<void*>(0x400E1000UL)
#endif

static struct ethosu_driver ethosu_drv;

// NPU interrupt handler — must match the vector table entry name in your BSP
extern "C" void ETHOSU_IRQHandler(void)
{
    ethosu_irq_handler(&ethosu_drv);
}

// ── A single MNIST test image embedded directly ──────────────────────────────
// This is MNIST test[0] (true label = 7), stored as raw uint8 [0-255].
// Replace with your own image bytes if needed.
static const uint8_t kTestImage[784] = {
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,84,185,159,151,60,36,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,222,254,254,254,254,241,198,
    198,198,198,198,198,198,198,170,52,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,67,114,72,114,163,227,254,225,254,254,254,250,
    229,254,254,140,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,17,66,
    14,67,67,67,59,21,236,254,106,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,83,253,209,18,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,22,233,255,83,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,129,254,238,44,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,59,249,254,62,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,133,254,187,5,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,9,205,248,58,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,126,254,182,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,75,251,240,57,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,19,221,254,166,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,203,254,219,35,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,38,254,254,77,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,31,224,254,115,1,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,133,254,254,52,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,61,242,254,254,52,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,121,254,254,219,40,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,121,254,207,18,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
};

// ── main ─────────────────────────────────────────────────────────────────────
int main(void)
{
    // Step 1: Initialise Ethos-U85 NPU hardware driver
    MicroPrintf("Initialising Ethos-U85...\r\n");
    if (ethosu_init(&ethosu_drv,
                    ETHOS_U_BASE_ADDRESS,
                    nullptr,   // no dedicated fast SRAM
                    0,
                    1,         // secure
                    1))        // privileged
    {
        MicroPrintf("ERROR: ethosu_init() failed\r\n");
        return 1;
    }
    MicroPrintf("Ethos-U85 ready.\r\n");

    // Step 2: Load model
    const tflite::Model* model = tflite::GetModel(g_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        MicroPrintf("ERROR: Model schema mismatch\r\n");
        return 1;
    }

    // Step 3: Op resolver — AddEthosU() is the only op needed for a
    // fully Vela-compiled model.  The entire graph is one ethos-u custom op.
    static tflite::MicroMutableOpResolver<1> resolver;
    resolver.AddEthosU();

    // Step 4: Build interpreter and allocate tensors
    static tflite::MicroInterpreter interpreter(
        model, resolver, tensor_arena, kTensorArenaSize);

    if (interpreter.AllocateTensors() != kTfLiteOk) {
        MicroPrintf("ERROR: AllocateTensors() failed — "
                    "try increasing kTensorArenaSize\r\n");
        return 1;
    }

    // Step 5: Fill input tensor from test image
    TfLiteTensor* input  = interpreter.input(0);
    TfLiteTensor* output = interpreter.output(0);

    const float scale      = input->params.scale;
    const int   zero_point = input->params.zero_point;

    if (input->type == kTfLiteInt8) {
        for (int i = 0; i < 784; ++i) {
            float norm = kTestImage[i] / 255.0f;
            input->data.int8[i] =
                static_cast<int8_t>(norm / scale + zero_point);
        }
    } else {
        for (int i = 0; i < 784; ++i)
            input->data.f[i] = kTestImage[i] / 255.0f;
    }

    // Step 5 — INVOKE (NPU offload via TFLM Ethos-U custom op)
    MicroPrintf("Running inference on Ethos-U85 NPU...\r\n");
    if (interpreter.Invoke() != kTfLiteOk) {
        MicroPrintf("ERROR: Invoke() failed\r\n");
        return 1;
    }

    // Step 6: Parse output and print result
    float max_score = -1e9f;
    int   max_index = 0;

    const float out_scale = output->params.scale;
    const int   out_zp    = output->params.zero_point;

    for (int i = 0; i < 10; ++i) {
        float score = (output->type == kTfLiteInt8)
            ? (static_cast<float>(output->data.int8[i]) - out_zp) * out_scale
            : output->data.f[i];
        if (score > max_score) { max_score = score; max_index = i; }
    }

    MicroPrintf("Predicted Digit = %d\r\n", max_index);
    MicroPrintf("Confidence      = %.3f\r\n", static_cast<double>(max_score));

    // Bare-metal: spin forever
    while (true) {}
    return 0;
}
