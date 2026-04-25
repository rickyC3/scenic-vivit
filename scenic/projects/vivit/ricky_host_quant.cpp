/**********
Copyright (c) 2018, Xilinx, Inc.
All rights reserved.
Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**********/

/*******************************************************************************
** HOST Code
*******************************************************************************/

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <cstring>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>
#include <chrono>

using namespace std;

#include <CL/cl.h>
#include "help_functions.h"

#define ALL_MESSAGES

constexpr int SEQ_LEN = 14 * 14 + 1; // 14x14 patches + 1 CLS token
constexpr int EMBED_DIM = 768;
constexpr int NUM_HEADS = 12;
constexpr int FRAME = 16;
constexpr int HEAD_DIM = EMBED_DIM / NUM_HEADS;
constexpr int MLP_DIM = 3072;
constexpr int NUM_ENCODER_BLOCKS = 12;
static_assert(SEQ_LEN == 197, "spatial encoder kernels expect 196 patches plus one CLS token");
static_assert(HEAD_DIM == 64, "spatial encoder attention kernels expect 64 channels per head");
static_assert(EMBED_DIM % NUM_HEADS == 0, "embedding dimension must split evenly across heads");
static_assert(MLP_DIM == 4 * EMBED_DIM, "spatial encoder MLP expansion must be 4x embedding dimension");

// -----------------------------host side helper functions--------------------------------------
// TODO: modify to load weight from quantized model
static void init_random_int(int *data, size_t count, int min_value, int max_value)
{
    int range = max_value - min_value + 1;
    for (size_t i = 0; i < count; i++)
    {
        data[i] = min_value + (rand() % range);
    }
}
// Modified from `posix_memalign`
// allocate 4096-byte aligned host memory
static void *host_aligned_alloc(size_t size_in_bytes)
{
    void *ptr = nullptr;
    if (posix_memalign(&ptr, 4096, size_in_bytes))
    {
        return nullptr;
    }
    return ptr;
}

static bool load_txt_values_as_int(const string &filename, int *dst, size_t count)
{
    ifstream fin(filename);
    if (!fin.is_open())
    {
        return false;
    }

    for (size_t i = 0; i < count; i++)
    {
        float value = 0.0f;
        if (!(fin >> value))
        {
            cerr << "HOST-Error: Failed to parse value " << i << " from " << filename << endl;
            return false;
        }
        dst[i] = static_cast<int>(roundf(value));
    }

    return true;
}

static bool load_txt_values_as_float(const string &filename, float *dst, size_t count)
{
    ifstream fin(filename);
    if (!fin.is_open())
    {
        return false;
    }

    for (size_t i = 0; i < count; i++)
    {
        if (!(fin >> dst[i]))
        {
            cerr << "HOST-Error: Failed to parse value " << i << " from " << filename << endl;
            return false;
        }
    }

    return true;
}

static bool load_txt_values_as_int8(const string &filename, int8_t *dst, size_t count)
{
    ifstream fin(filename);
    if (!fin.is_open())
    {
        return false;
    }

    for (size_t i = 0; i < count; i++)
    {
        float value = 0.0f;
        if (!(fin >> value))
        {
            cerr << "HOST-Error: Failed to parse value " << i << " from " << filename << endl;
            return false;
        }
        dst[i] = static_cast<int8_t>(roundf(value));
    }

    return true;
}

// TODO: not implement here
static int gelu_int(int x)
{
    float xf = static_cast<float>(x);
    float y = 0.5f * xf * (1.0f + tanhf(0.79788456f * (xf + 0.044715f * xf * xf * xf)));
    return static_cast<int>(roundf(y));
}

//  TODO: not implement here
static void layernorm_host(int in[SEQ_LEN][EMBED_DIM], int out[SEQ_LEN][EMBED_DIM])
{
    const float eps = 1.0e-5f;
    const float output_scale = 16.0f;

    for (int i = 0; i < SEQ_LEN; i++)
    {
        float mean = 0.0f;
        for (int d = 0; d < EMBED_DIM; d++)
            mean += static_cast<float>(in[i][d]);
        mean /= static_cast<float>(EMBED_DIM);

        float var = 0.0f;
        for (int d = 0; d < EMBED_DIM; d++)
        {
            float diff = static_cast<float>(in[i][d]) - mean;
            var += diff * diff;
        }
        var /= static_cast<float>(EMBED_DIM);

        float inv_std = 1.0f / sqrtf(var + eps);
        for (int d = 0; d < EMBED_DIM; d++)
        {
            float norm = (static_cast<float>(in[i][d]) - mean) * inv_std;
            out[i][d] = static_cast<int>(roundf(norm * output_scale));
        }
    }
}

// [12][197][64] -> [197][768]
static void concat_attention_heads(int attention_heads[NUM_HEADS][SEQ_LEN][HEAD_DIM],
                                   int attention_concat[SEQ_LEN][EMBED_DIM])
{
    for (int h = 0; h < NUM_HEADS; h++)
        for (int i = 0; i < SEQ_LEN; i++)
            for (int d = 0; d < HEAD_DIM; d++)
                attention_concat[i][h * HEAD_DIM + d] = attention_heads[h][i][d];
}

//  TODO: not implement here?
static void residual_add(int a[SEQ_LEN][EMBED_DIM], int b[SEQ_LEN][EMBED_DIM], int out[SEQ_LEN][EMBED_DIM])
{
    for (int i = 0; i < SEQ_LEN; i++)
        for (int d = 0; d < EMBED_DIM; d++)
            out[i][d] = a[i][d] + b[i][d];
}

static void residual_add(float a[SEQ_LEN][EMBED_DIM], int b[SEQ_LEN][EMBED_DIM], int out[SEQ_LEN][EMBED_DIM])
{
    for (int i = 0; i < SEQ_LEN; i++)
        for (int d = 0; d < EMBED_DIM; d++)
            out[i][d] = static_cast<int>(roundf(a[i][d] + static_cast<float>(b[i][d])));
}

// block with int convert into float
static void copy_int_to_float(int in[SEQ_LEN][EMBED_DIM], float out[SEQ_LEN][EMBED_DIM])
{
    for (int i = 0; i < SEQ_LEN; i++)
        for (int d = 0; d < EMBED_DIM; d++)
            out[i][d] = static_cast<float>(in[i][d]);
}

// TODO: delete this?
static void linear_quant_host_int_input(
    int8_t in[SEQ_LEN][EMBED_DIM],
    float scale_input[SEQ_LEN],
    int8_t weight[EMBED_DIM][EMBED_DIM],
    float bias[EMBED_DIM],
    float scale_weight[EMBED_DIM],
    int out[SEQ_LEN][EMBED_DIM])
{
    for (int i = 0; i < SEQ_LEN; i++)
    {
        for (int o = 0; o < EMBED_DIM; o++)
        {
            float acc = bias[o];
            const float input_scale = scale_input[i] * scale_weight[o];
            for (int d = 0; d < EMBED_DIM; d++)
            {
                acc += static_cast<float>(in[i][d]) * static_cast<float>(weight[d][o]) * input_scale;
            }
            out[i][o] = static_cast<int>(roundf(acc));
        }
    }
}

// attention output projection
static void linear_quant_host_fp32_input(
    int in[SEQ_LEN][EMBED_DIM],
    int8_t weight[EMBED_DIM][EMBED_DIM],
    float bias[EMBED_DIM],
    float scale_weight[EMBED_DIM],
    int out[SEQ_LEN][EMBED_DIM])
{
    for (int i = 0; i < SEQ_LEN; i++)
    {
        for (int o = 0; o < EMBED_DIM; o++)
        {
            float acc = bias[o];
            const float weight_scale = scale_weight[o];
            for (int d = 0; d < EMBED_DIM; d++)
            {
                acc += static_cast<float>(in[i][d]) * static_cast<float>(weight[d][o]) * weight_scale;
            }
            out[i][o] = static_cast<int>(roundf(acc));
        }
    }
}

static void mlp_quant_host(
    int8_t in[SEQ_LEN][EMBED_DIM],
    float scale_input[SEQ_LEN],
    int8_t W1[EMBED_DIM][MLP_DIM],
    float b1[MLP_DIM],
    float scale_w1[MLP_DIM],
    int8_t W2[MLP_DIM][EMBED_DIM],
    float b2[EMBED_DIM],
    float scale_w2[EMBED_DIM],
    int out[SEQ_LEN][EMBED_DIM])
{
    vector<float> hidden(MLP_DIM);
    for (int i = 0; i < SEQ_LEN; i++)
    {
        for (int h = 0; h < MLP_DIM; h++)
        {
            float acc = b1[h];
            const float scale = scale_input[i] * scale_w1[h];
            for (int d = 0; d < EMBED_DIM; d++)
            {
                acc += static_cast<float>(in[i][d]) * static_cast<float>(W1[d][h]) * scale;
            }
            float gelu = 0.5f * acc * (1.0f + tanhf(0.79788456f * (acc + 0.044715f * acc * acc * acc)));
            hidden[h] = gelu;
        }

        for (int d = 0; d < EMBED_DIM; d++)
        {
            float acc = b2[d];
            const float scale = scale_w2[d];
            for (int h = 0; h < MLP_DIM; h++)
            {
                acc += hidden[h] * static_cast<float>(W2[h][d]) * scale;
            }
            out[i][d] = static_cast<int>(roundf(acc));
        }
    }
}

// TODO: not implement here, not used?
static void mlp_host(int in[SEQ_LEN][EMBED_DIM],
                     int W1[EMBED_DIM][MLP_DIM], int b1[MLP_DIM],
                     int W2[MLP_DIM][EMBED_DIM], int b2[EMBED_DIM],
                     int out[SEQ_LEN][EMBED_DIM])
{
    vector<int> hidden(MLP_DIM);
    for (int i = 0; i < SEQ_LEN; i++)
    {
        for (int h = 0; h < MLP_DIM; h++)
        {
            long long acc = b1[h];
            for (int d = 0; d < EMBED_DIM; d++)
                acc += static_cast<long long>(in[i][d]) * W1[d][h];
            hidden[h] = gelu_int(static_cast<int>(acc >> 8));
        }
        for (int d = 0; d < EMBED_DIM; d++)
        {
            long long acc = b2[d];
            for (int h = 0; h < MLP_DIM; h++)
                acc += static_cast<long long>(hidden[h]) * W2[h][d];
            out[i][d] = static_cast<int>(acc >> 8);
        }
    }
}
//-------------------------------------------------------------------------------------

// ----------------------------------Same as previous version of host.cpp----------------------------------
// ********************************************************************************** //
// ---------------------------------------------------------------------------------- //
//                          M A I N    F U N C T I O N                                //
// ---------------------------------------------------------------------------------- //
// ********************************************************************************** //

int main(int argc, char *argv[])
{
    cout << endl;

// ============================================================================
// Step 1: Check Command Line Arguments
// ============================================================================
//    o) argv[1] Platfrom Vendor
//    o) argv[2] Device Name
//    o) argv[3] XCLBIN file
// ============================================================================
// INFO: sample usage: ./host Xilinx xilinx_u280_xdma_201920_3 spatial_encoder.xclbin
#ifdef ALL_MESSAGES
    cout << "HOST-Info: ============================================================= " << endl;
    cout << "HOST-Info: (Step 1) Check Command Line Arguments                      " << endl;
    cout << "HOST-Info: ============================================================= " << endl;
#endif

    if (argc != 4)
    {
        cout << "HOST-Error: Incorrect command line syntax " << endl;
        cout << "HOST-Info:  Usage: " << argv[0] << " <Platform_Vendor> <Device_Name> <XCLBIN_File>  <Test Vectors Size>" << endl
             << endl;
        return EXIT_FAILURE;
    }

    const char *Target_Platform_Vendor = argv[1];
    const char *Target_Device_Name = argv[2];
    const char *xclbinFilename = argv[3];
    cout << "HOST-Info: Platform_Vendor   : " << Target_Platform_Vendor << endl;
    cout << "HOST-Info: Device_Name       : " << Target_Device_Name << endl;
    cout << "HOST-Info: XCLBIN_file       : " << xclbinFilename << endl;

    // ============================================================================
    // Step 2: Detect Target Platform and Target Device in a system.
    //         Create Context and Command Queue.
    // ============================================================================
    // Variables:
    //   o) Target_Platform_Vendor[] - defined as main() input argument
    //   o) Target_Device_Name[]     - defined as main() input argument
    //
    // After that
    //   o) Create a Context
    //   o) Create a Command Queue
    // ============================================================================
    cout << endl;
#ifdef ALL_MESSAGES
    cout << "HOST-Info: ============================================================= " << endl;
    cout << "HOST-Info: (Step 2) Detect Target Platform and Target Device in a system " << endl;
    cout << "HOST-Info:          Create Context and Command Queue                     " << endl;
    cout << "HOST-Info: ============================================================= " << endl;
#endif

    cl_uint ui;

    cl_platform_id *Platform_IDs;
    cl_uint Nb_Of_Platforms;
    cl_platform_id Target_Platform_ID;
    bool Platform_Detected;
    char *platform_info;

    cl_device_id *Device_IDs;
    cl_uint Nb_Of_Devices;
    cl_device_id Target_Device_ID;
    bool Device_Detected;
    char *device_info;

    cl_context Context;
    cl_command_queue Command_Queue;

    cl_int errCode;
    size_t size;

    // ------------------------------------------------------------------------------------
    // Step 2.1: Get All PLATFORMS, then search for Target_Platform_Vendor (CL_PLATFORM_VENDOR)
    // ------------------------------------------------------------------------------------

    // Get the number of platforms
    // ..................................................
    errCode = clGetPlatformIDs(0, NULL, &Nb_Of_Platforms);
    if (errCode != CL_SUCCESS || Nb_Of_Platforms <= 0)
    {
        cout << endl
             << "HOST-Error: Failed to get the number of available platforms" << endl
             << endl;
        return EXIT_FAILURE;
    }

#ifdef ALL_MESSAGES
    cout << "HOST-Info: Number of detected platforms : " << Nb_Of_Platforms << endl;
#endif

    // Allocate memory to store platforms
    // ..................................................
    Platform_IDs = new cl_platform_id[Nb_Of_Platforms];
    if (!Platform_IDs)
    {
        cout << endl
             << "HOST-Error: Out of Memory during memory allocation for Platform_IDs" << endl
             << endl;
        return EXIT_FAILURE;
    }

    // Get and store all PLATFORMS
    // ..................................................
    errCode = clGetPlatformIDs(Nb_Of_Platforms, Platform_IDs, NULL);
    if (errCode != CL_SUCCESS)
    {
        cout << endl
             << "HOST-Error: Failed to get the available platforms" << endl
             << endl;
        return EXIT_FAILURE;
    }

    // Search for Platform (ex: Xilinx) using: CL_PLATFORM_VENDOR = Target_Platform_Vendor
    // ....................................................................................
    Platform_Detected = false;
    for (ui = 0; ui < Nb_Of_Platforms; ui++)
    {

        errCode = clGetPlatformInfo(Platform_IDs[ui], CL_PLATFORM_VENDOR, 0, NULL, &size);
        if (errCode != CL_SUCCESS)
        {
            cout << endl
                 << "HOST-Error: Failed to get the size of the Platofrm parameter " << "CL_PLATFORM_VENDOR" << " value " << endl
                 << endl;
            return EXIT_FAILURE;
        }

        platform_info = new char[size];
        if (!platform_info)
        {
            cout << endl
                 << "HOST-Error: Out of Memory during memory allocation for Platform Parameter " << "CL_PLATFORM_VENDOR" << endl
                 << endl;
            return EXIT_FAILURE;
        }

        errCode = clGetPlatformInfo(Platform_IDs[ui], CL_PLATFORM_VENDOR, size, platform_info, NULL);
        if (errCode != CL_SUCCESS)
        {
            cout << endl
                 << "HOST-Error: Failed to get the " << "CL_PLATFORM_VENDOR" << " platform info" << endl
                 << endl;
            return EXIT_FAILURE;
        }

        // Check if the current platform matches Target_Platform_Vendor
        // .............................................................
        if (strcmp(platform_info, Target_Platform_Vendor) == 0)
        {
            Platform_Detected = true;
            Target_Platform_ID = Platform_IDs[ui];
#ifdef ALL_MESSAGES
            cout << "HOST-Info: Selected platform            : " << Target_Platform_Vendor << endl
                 << endl;
#endif
        }
    }

    if (Platform_Detected == false)
    {
        cout << endl
             << "HOST-Error: Failed to get detect " << Target_Platform_Vendor << " platform" << endl
             << endl;
        return EXIT_FAILURE;
    }

    // ------------------------------------------------------------------------------------
    // Step 2.2:  Get All Devices for selected platform Target_Platform_ID
    //            then search for Xilinx platform (CL_DEVICE_NAME = Target_Device_Name)
    // ------------------------------------------------------------------------------------

    // Get the Number of Devices
    // ............................................................................
    errCode = clGetDeviceIDs(Target_Platform_ID, CL_DEVICE_TYPE_ALL, 0, NULL, &Nb_Of_Devices);
    if (errCode != CL_SUCCESS)
    {
        cout << endl
             << "HOST-Error: Failed to get the number of available Devices" << endl
             << endl;
        return EXIT_FAILURE;
    }
#ifdef ALL_MESSAGES
    cout << "HOST-Info: Number of available devices  : " << Nb_Of_Devices << endl;
#endif

    Device_IDs = new cl_device_id[Nb_Of_Devices];
    if (!Device_IDs)
    {
        cout << endl
             << "HOST-Error: Out of Memory during memory allocation for Device_IDs" << endl
             << endl;
        return EXIT_FAILURE;
    }

    errCode = clGetDeviceIDs(Target_Platform_ID, CL_DEVICE_TYPE_ALL, Nb_Of_Devices, Device_IDs, NULL);
    if (errCode != CL_SUCCESS)
    {
        cout << endl
             << "HOST-Error: Failed to get available Devices" << endl
             << endl;
        return EXIT_FAILURE;
    }

    // Search for CL_DEVICE_NAME = Target_Device_Name
    // ............................................................................
    Device_Detected = false;
    for (ui = 0; ui < Nb_Of_Devices; ui++)
    {
        errCode = clGetDeviceInfo(Device_IDs[ui], CL_DEVICE_NAME, 0, NULL, &size);
        if (errCode != CL_SUCCESS)
        {
            cout << endl
                 << "HOST-Error: Failed to get the size of the Device parameter value " << "CL_DEVICE_NAME" << endl
                 << endl;
            return EXIT_FAILURE;
        }

        device_info = new char[size];
        if (!device_info)
        {
            cout << endl
                 << "HOST-Error: Out of Memory during memory allocation for Device parameter " << "CL_DEVICE_NAME" << " value " << endl
                 << endl;
            return EXIT_FAILURE;
        }

        errCode = clGetDeviceInfo(Device_IDs[ui], CL_DEVICE_NAME, size, device_info, NULL);
        if (errCode != CL_SUCCESS)
        {
            cout << endl
                 << "HOST-Error: Failed to get the " << "CL_DEVICE_NAME" << " device info" << endl
                 << endl;
            return EXIT_FAILURE;
        }

        // Check if the current device matches Target_Device_Name
        // ............................................................................
        if (strcmp(device_info, Target_Device_Name) == 0)
        {
            Device_Detected = true;
            Target_Device_ID = Device_IDs[ui];
        }
    }

    if (Device_Detected == false)
    {
        cout << endl
             << "HOST-Error: Failed to get detect " << Target_Device_Name << " device" << endl
             << endl;
        return EXIT_FAILURE;
    }
    else
    {
#ifdef ALL_MESSAGES
        cout << "HOST-Info: Selected device              : " << Target_Device_Name << endl
             << endl;
#endif
    }

// ------------------------------------------------------------------------------------
// Step 2.3: Create Context
// ------------------------------------------------------------------------------------
#ifdef ALL_MESSAGES
    cout << "HOST-Info: Creating Context ... " << endl;
#endif
    Context = clCreateContext(0, 1, &Target_Device_ID, NULL, NULL, &errCode);
    if (errCode != CL_SUCCESS)
    {
        cout << endl
             << "HOST-Error: Failed to create a Context" << endl
             << endl;
        return EXIT_FAILURE;
    }

// ------------------------------------------------------------------------------------
// Step 2.4: Create Command Queue (commands are executed in-order)
// ------------------------------------------------------------------------------------
#ifdef ALL_MESSAGES
    cout << "HOST-Info: Creating Command Queue ... " << endl;
#endif
    Command_Queue = clCreateCommandQueue(Context, Target_Device_ID, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE, &errCode);
    if (errCode != CL_SUCCESS)
    {
        cout << endl
             << "HOST-Error: Failed to create a Command Queue" << endl
             << endl;
        return EXIT_FAILURE;
    }

// ============================================================================
// Step 3: Create Program and Kernel
// ============================================================================
//   o) Create a Program from a Binary File and Build it
//   o) Create a Kernel
// ============================================================================
#ifdef ALL_MESSAGES
    cout << endl;
    cout << "HOST-Info: ============================================================= " << endl;
    cout << "HOST-Info: (Step 3) Create Program and Kernels                           " << endl;
    cout << "HOST-Info: ============================================================= " << endl;
#endif

    // ------------------------------------------------------------------
    // Step 3.1: Load Binary File from a disk to Memory
    // ------------------------------------------------------------------
    unsigned char *xclbin_Memory;
    int program_length;

#ifdef ALL_MESSAGES
    cout << "HOST-Info: Loading " << xclbinFilename << " binary file to memory ..." << endl;
#endif

    program_length = loadFile2Memory(xclbinFilename, (char **)&xclbin_Memory);
    if (program_length < 0)
    {
        cout << endl
             << "HOST-Error: Failed to load " << xclbinFilename << " binary file to memory" << endl
             << endl;
        return EXIT_FAILURE;
    }

    // ------------------------------------------------------------
    // Step 3.2: Create a program using a Binary File
    // ------------------------------------------------------------
    size_t Program_Length_in_Bytes;
    cl_program Program;
    cl_int Binary_Status;

#ifdef ALL_MESSAGES
    cout << "HOST-Info: Creating Program with Binary ..." << endl;
#endif
    Program_Length_in_Bytes = program_length;
    Program = clCreateProgramWithBinary(Context, 1, &Target_Device_ID, &Program_Length_in_Bytes,
                                        (const unsigned char **)&xclbin_Memory, &Binary_Status, &errCode);
    if (errCode != CL_SUCCESS)
    {
        cout << endl
             << "HOST-Error: Failed to create a Program from a Binary" << endl
             << endl;
        return EXIT_FAILURE;
    }

// ----------------------------------------------------------------------
// Step 3.3: Build (compiles and links) a program executable from binary
// ----------------------------------------------------------------------
#ifdef ALL_MESSAGES
    cout << "HOST-Info: Building the Program ..." << endl;
#endif

    errCode = clBuildProgram(Program, 1, &Target_Device_ID, NULL, NULL, NULL);
    if (errCode != CL_SUCCESS)
    {
        cout << endl
             << "HOST-Error: Failed to build a Program Executable" << endl
             << endl;
        return EXIT_FAILURE;
    }
    // ------------------------Start to be defferent from sample host.cpp------------------------------
    // -------------------------------------------------------------
    // Step 3.4: Create Kernels
    // -------------------------------------------------------------
    cl_kernel K_layernorm, K_linear_qkv_multihead, K_attention_opt[NUM_HEADS];

#ifdef ALL_MESSAGES
    cout << "HOST-Info: Creating a Kernel: layernorm ..." << endl;
#endif
    K_layernorm = clCreateKernel(Program, "layernorm", &errCode);
    if (errCode != CL_SUCCESS)
    {
        cout << endl
             << "HOST-Error: Failed to create K_layernorm" << endl
             << endl;
        return EXIT_FAILURE;
    }

#ifdef ALL_MESSAGES
    cout << "HOST-Info: Creating a Kernel: linear_qkv_multihead ..." << endl;
#endif
    K_linear_qkv_multihead = clCreateKernel(Program, "linear_qkv_multihead", &errCode);
    if (errCode != CL_SUCCESS)
    {
        cout << endl
             << "HOST-Error: Failed to create K_linear_qkv_multihead" << endl
             << endl;
        return EXIT_FAILURE;
    }

#ifdef ALL_MESSAGES
    cout << "HOST-Info: Creating Kernels: attention_opt[0.." << (NUM_HEADS - 1) << "] ..." << endl;
#endif
    for (int h = 0; h < NUM_HEADS; h++)
    {
        K_attention_opt[h] = clCreateKernel(Program, "attention_opt", &errCode);
        if (errCode != CL_SUCCESS)
        {
            cout << endl
                 << "HOST-Error: Failed to create K_attention_opt[" << h << "]" << endl
                 << endl;
            return EXIT_FAILURE;
        }
    }
    // TODO: Add mlp kernel

    // ================================================================
    // Step 4: Prepare Data to Run Kernel
    // ================================================================
    //   o) Allocate and initialize spatial encoder input and weights
    //   o) Allocate Q/K/V, attention, MLP, and final output arrays
    //   o) Create Buffers in Global Memory to store FPGA data
    // ================================================================
    // allocate host memory for spatial encoder input, weights, and outputs
    float (*X)[EMBED_DIM];
    float *LayerNorm_gamma;
    float *LayerNorm_beta;
    int8_t (*X_norm)[EMBED_DIM];
    float *LayerNorm_scaler;
    int8_t (*Wq)[EMBED_DIM];
    float *bq;
    float *scale_wq;
    int8_t (*Wk)[EMBED_DIM];
    float *bk;
    float *scale_wk;
    int8_t (*Wv)[EMBED_DIM];
    float *bv;
    float *scale_wv;
    int8_t (*Wout)[EMBED_DIM]; // attention ouput projection weight
    float *bout;
    float *scale_wout;
    float *LayerNorm1_gamma;
    float *LayerNorm1_beta;
    float (*Q)[SEQ_LEN][HEAD_DIM];
    float (*K_out)[SEQ_LEN][HEAD_DIM];
    float (*V)[SEQ_LEN][HEAD_DIM];
    int (*Attention_Out)[SEQ_LEN][HEAD_DIM];
    int (*Attention_Concat)[EMBED_DIM];
    int (*Attention_Projected)[EMBED_DIM];
    int (*After_Attention_Residual)[EMBED_DIM];
    int (*MLP_Input)[EMBED_DIM];
    int (*MLP_Output)[EMBED_DIM];
    int (*Spatial_Encoder_Output)[EMBED_DIM];
    int8_t (*MLP_W1)[MLP_DIM];
    float *MLP_b1;
    float *MLP_scale_w1;
    int8_t (*MLP_W2)[EMBED_DIM];
    float *MLP_b2;
    float *MLP_scale_w2;

#ifdef ALL_MESSAGES
    cout << endl;
    cout << "HOST-Info: ============================================================= " << endl;
    cout << "HOST-Info: (Step 4) Prepare Data to Run Kernels                           " << endl;
    cout << "HOST-Info: ============================================================= " << endl;
#endif

    // ------------------------------------------------------------------
    // Step 4.1: Allocate and initialize spatial encoder arrays
    // ------------------------------------------------------------------
    void *ptr = nullptr;
    // config memory
    X = reinterpret_cast<float (*)[EMBED_DIM]>(host_aligned_alloc(FRAME * SEQ_LEN * EMBED_DIM * sizeof(float)));
    LayerNorm_gamma = reinterpret_cast<float *>(host_aligned_alloc(EMBED_DIM * sizeof(float)));
    LayerNorm_beta = reinterpret_cast<float *>(host_aligned_alloc(EMBED_DIM * sizeof(float)));
    X_norm = reinterpret_cast<int8_t (*)[EMBED_DIM]>(host_aligned_alloc(FRAME * SEQ_LEN * EMBED_DIM * sizeof(int8_t)));
    LayerNorm_scaler = reinterpret_cast<float *>(host_aligned_alloc(SEQ_LEN * sizeof(float)));
    Wq = reinterpret_cast<int8_t (*)[EMBED_DIM]>(host_aligned_alloc(EMBED_DIM * EMBED_DIM * sizeof(int8_t)));
    bq = reinterpret_cast<float *>(host_aligned_alloc(EMBED_DIM * sizeof(float)));
    scale_wq = reinterpret_cast<float *>(host_aligned_alloc(EMBED_DIM * sizeof(float)));
    Wk = reinterpret_cast<int8_t (*)[EMBED_DIM]>(host_aligned_alloc(EMBED_DIM * EMBED_DIM * sizeof(int8_t)));
    bk = reinterpret_cast<float *>(host_aligned_alloc(EMBED_DIM * sizeof(float)));
    scale_wk = reinterpret_cast<float *>(host_aligned_alloc(EMBED_DIM * sizeof(float)));
    Wv = reinterpret_cast<int8_t (*)[EMBED_DIM]>(host_aligned_alloc(EMBED_DIM * EMBED_DIM * sizeof(int8_t)));
    bv = reinterpret_cast<float *>(host_aligned_alloc(EMBED_DIM * sizeof(float)));
    scale_wv = reinterpret_cast<float *>(host_aligned_alloc(EMBED_DIM * sizeof(float)));
    Wout = reinterpret_cast<int8_t (*)[EMBED_DIM]>(host_aligned_alloc(EMBED_DIM * EMBED_DIM * sizeof(int8_t)));
    bout = reinterpret_cast<float *>(host_aligned_alloc(EMBED_DIM * sizeof(float)));
    scale_wout = reinterpret_cast<float *>(host_aligned_alloc(EMBED_DIM * sizeof(float)));
    LayerNorm1_gamma = reinterpret_cast<float *>(host_aligned_alloc(EMBED_DIM * sizeof(float)));
    LayerNorm1_beta = reinterpret_cast<float *>(host_aligned_alloc(EMBED_DIM * sizeof(float)));
    Q = reinterpret_cast<float (*)[SEQ_LEN][HEAD_DIM]>(host_aligned_alloc(NUM_HEADS * SEQ_LEN * HEAD_DIM * sizeof(float)));
    K_out = reinterpret_cast<float (*)[SEQ_LEN][HEAD_DIM]>(host_aligned_alloc(NUM_HEADS * SEQ_LEN * HEAD_DIM * sizeof(float)));
    V = reinterpret_cast<float (*)[SEQ_LEN][HEAD_DIM]>(host_aligned_alloc(NUM_HEADS * SEQ_LEN * HEAD_DIM * sizeof(float)));
    Attention_Out = reinterpret_cast<int (*)[SEQ_LEN][HEAD_DIM]>(host_aligned_alloc(FRAME * NUM_HEADS * SEQ_LEN * HEAD_DIM * sizeof(int)));
    Attention_Concat = reinterpret_cast<int (*)[EMBED_DIM]>(host_aligned_alloc(SEQ_LEN * EMBED_DIM * sizeof(int)));
    Attention_Projected = reinterpret_cast<int (*)[EMBED_DIM]>(host_aligned_alloc(SEQ_LEN * EMBED_DIM * sizeof(int)));
    After_Attention_Residual = reinterpret_cast<int (*)[EMBED_DIM]>(host_aligned_alloc(SEQ_LEN * EMBED_DIM * sizeof(int)));
    MLP_Input = reinterpret_cast<int (*)[EMBED_DIM]>(host_aligned_alloc(SEQ_LEN * EMBED_DIM * sizeof(int)));
    MLP_Output = reinterpret_cast<int (*)[EMBED_DIM]>(host_aligned_alloc(SEQ_LEN * EMBED_DIM * sizeof(int)));
    Spatial_Encoder_Output = reinterpret_cast<int (*)[EMBED_DIM]>(host_aligned_alloc(SEQ_LEN * EMBED_DIM * sizeof(int)));
    MLP_W1 = reinterpret_cast<int8_t (*)[MLP_DIM]>(host_aligned_alloc(EMBED_DIM * MLP_DIM * sizeof(int8_t)));
    MLP_b1 = reinterpret_cast<float *>(host_aligned_alloc(MLP_DIM * sizeof(float)));
    MLP_scale_w1 = reinterpret_cast<float *>(host_aligned_alloc(MLP_DIM * sizeof(float)));
    MLP_W2 = reinterpret_cast<int8_t (*)[EMBED_DIM]>(host_aligned_alloc(MLP_DIM * EMBED_DIM * sizeof(int8_t)));
    MLP_b2 = reinterpret_cast<float *>(host_aligned_alloc(EMBED_DIM * sizeof(float)));
    MLP_scale_w2 = reinterpret_cast<float *>(host_aligned_alloc(EMBED_DIM * sizeof(float)));

    if (!X || !LayerNorm_gamma || !LayerNorm_beta || !X_norm || !LayerNorm_scaler ||
        !Wq || !bq || !scale_wq || !Wk || !bk || !scale_wk || !Wv || !bv || !scale_wv ||
        !Wout || !bout || !scale_wout || !LayerNorm1_gamma || !LayerNorm1_beta ||
        !Q || !K_out || !V || !Attention_Out || !Attention_Concat || !Attention_Projected ||
        !After_Attention_Residual || !MLP_Input || !MLP_Output || !Spatial_Encoder_Output ||
        !MLP_W1 || !MLP_b1 || !MLP_scale_w1 || !MLP_W2 || !MLP_b2 || !MLP_scale_w2)
    {
        cout << endl
             << "HOST-Error: Out of Memory during spatial encoder allocation" << endl
             << endl;
        return EXIT_FAILURE;
    }

    // Keep the original random initialization flow for fallback/debug reference.
    /*
    srand(0);
    init_random_int(&X[0][0], SEQ_LEN * EMBED_DIM, -2, 2);
    init_random_int(&Wq[0][0], EMBED_DIM * EMBED_DIM, -1, 1);
    init_random_int(bq, EMBED_DIM, -1, 1);
    init_random_int(&Wk[0][0], EMBED_DIM * EMBED_DIM, -1, 1);
    init_random_int(bk, EMBED_DIM, -1, 1);
    init_random_int(&Wv[0][0], EMBED_DIM * EMBED_DIM, -1, 1);
    init_random_int(bv, EMBED_DIM, -1, 1);
    init_random_int(&MLP_W1[0][0], EMBED_DIM * MLP_DIM, -1, 1);
    init_random_int(MLP_b1, MLP_DIM, -1, 1);
    init_random_int(&MLP_W2[0][0], MLP_DIM * EMBED_DIM, -1, 1);
    init_random_int(MLP_b2, EMBED_DIM, -1, 1);
    */

    const string x_input_file = "input_x.txt";

    if (!load_txt_values_as_float(x_input_file, &X[0][0], FRAME * SEQ_LEN * EMBED_DIM))
    {
        cout << endl
             << "HOST-Error: Failed to load encoder input X from " << x_input_file << endl
             << endl;
        return EXIT_FAILURE;
    }
    cout << "HOST-Info: Loaded spatial encoder input X from " << x_input_file << endl;

    cout << endl;

// ------------------------------------------------------------------
// Step 4.2: Create Buffers in Global Memory to store data
//             o) GlobMem_BUF_X                 - stores fp32 attention input X (R)
//             o) GlobMem_BUF_LayerNorm_Gamma   - stores LayerNorm gamma (R)
//             o) GlobMem_BUF_LayerNorm_Beta    - stores LayerNorm beta (R)
//             o) GlobMem_BUF_X_norm            - stores int8 LayerNorm output (R/W)
//             o) GlobMem_BUF_LayerNorm_Scaler  - stores LayerNorm scaler[SEQ_LEN] (R/W)
//             o) GlobMem_BUF_Wq/bq/scale_wq    - stores Q quantized weights, bias, and scaler (R)
//             o) GlobMem_BUF_Wk/bk/scale_wk    - stores K quantized weights, bias, and scaler (R)
//             o) GlobMem_BUF_Wv/bv/scale_wv    - stores V quantized weights, bias, and scaler (R)
//             o) GlobMem_BUF_Q/K/V             - stores fp32 Q/K/V outputs (R/W)
//             o) GlobMem_BUF_Attention_Out     - stores attention output from all heads (R/W)
// ------------------------------------------------------------------
// TODO: Add mlp weights and bias buffer
#ifdef ALL_MESSAGES
    cout << "HOST-Info: Allocating buffers in Global Memory to store Input and Output Data ..." << endl;
#endif

    // buffer relative to X
    cl_mem GlobMem_BUF_X[FRAME], GlobMem_BUF_X_norm[FRAME], GlobMem_BUF_Attention_Out[FRAME];

    cl_mem GlobMem_BUF_LayerNorm_Gamma, GlobMem_BUF_LayerNorm_Beta,
        GlobMem_BUF_LayerNorm_Scaler,
        GlobMem_BUF_Wq, GlobMem_BUF_bq, GlobMem_BUF_scale_wq,
        GlobMem_BUF_Wk, GlobMem_BUF_bk, GlobMem_BUF_scale_wk,
        GlobMem_BUF_Wv, GlobMem_BUF_bv, GlobMem_BUF_scale_wv,
        GlobMem_BUF_Q, GlobMem_BUF_K, GlobMem_BUF_V;
    
    // GlobMem_BUF_X = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, SEQ_LEN * EMBED_DIM * sizeof(float), X, &errCode);
    // if (errCode != CL_SUCCESS)
    // {
    //     cout << endl
    //          << "Host-Error: Failed to allocate GlobMem_BUF_X" << endl
    //          << endl;
    //     return EXIT_FAILURE;
    // }

    size_t X_ofs = 0;
    for (int f = 0; f < FRAME; f++)
    {
        GlobMem_BUF_X[f] = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, SEQ_LEN * EMBED_DIM * sizeof(float), (X + X_ofs), &errCode);
        if (errCode != CL_SUCCESS)
        {
            cout << endl
                << "Host-Error: Failed to allocate GlobMem_BUF_X at frame[" << f << "] " << endl
                << endl;
            return EXIT_FAILURE;
        }

        GlobMem_BUF_X_norm[f] = clCreateBuffer(Context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, SEQ_LEN * EMBED_DIM * sizeof(int8_t), (X_norm + X_ofs), &errCode);
        if (errCode != CL_SUCCESS)
        {
            cout << endl
                << "Host-Error: Failed to allocate GlobMem_BUF_X_norm" << endl
                << endl;
            return EXIT_FAILURE;
        }

        GlobMem_BUF_Attention_Out[f] = clCreateBuffer(Context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, NUM_HEADS * SEQ_LEN * HEAD_DIM * sizeof(int), (Attention_Out + X_ofs * HEAD_DIM * NUM_HEADS), &errCode);
        if (errCode != CL_SUCCESS)
        {
            cout << endl
                << "Host-Error: Failed to allocate GlobMem_BUF_Attention_Out" << endl
                << endl;
            return EXIT_FAILURE;
        }

        X_ofs += SEQ_LEN;
    }



    
    // ref: clCreateBuffer method
    // https://www.cnblogs.com/willhua/p/9463515.html
    // https://zhuanlan.zhihu.com/p/684133601
    // don't care loading data at later
    GlobMem_BUF_LayerNorm_Gamma = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, EMBED_DIM * sizeof(float), LayerNorm_gamma, &errCode);
    if (errCode != CL_SUCCESS)
    {
        cout << endl
             << "Host-Error: Failed to allocate GlobMem_BUF_LayerNorm_Gamma" << endl
             << endl;
        return EXIT_FAILURE;
    }
    GlobMem_BUF_LayerNorm_Beta = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, EMBED_DIM * sizeof(float), LayerNorm_beta, &errCode);
    if (errCode != CL_SUCCESS)
    {
        cout << endl
             << "Host-Error: Failed to allocate GlobMem_BUF_LayerNorm_Beta" << endl
             << endl;
        return EXIT_FAILURE;
    }
    
    GlobMem_BUF_LayerNorm_Scaler = clCreateBuffer(Context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, SEQ_LEN * sizeof(float), LayerNorm_scaler, &errCode);
    if (errCode != CL_SUCCESS)
    {
        cout << endl
             << "Host-Error: Failed to allocate GlobMem_BUF_LayerNorm_Scaler" << endl
             << endl;
        return EXIT_FAILURE;
    }
    GlobMem_BUF_Wq = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, EMBED_DIM * EMBED_DIM * sizeof(int8_t), Wq, &errCode);
    if (errCode != CL_SUCCESS)
    {
        cout << endl
             << "Host-Error: Failed to allocate GlobMem_BUF_Wq" << endl
             << endl;
        return EXIT_FAILURE;
    }
    GlobMem_BUF_bq = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, EMBED_DIM * sizeof(float), bq, &errCode);
    if (errCode != CL_SUCCESS)
    {
        cout << endl
             << "Host-Error: Failed to allocate GlobMem_BUF_bq" << endl
             << endl;
        return EXIT_FAILURE;
    }
    GlobMem_BUF_scale_wq = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, EMBED_DIM * sizeof(float), scale_wq, &errCode);
    if (errCode != CL_SUCCESS)
    {
        cout << endl
             << "Host-Error: Failed to allocate GlobMem_BUF_scale_wq" << endl
             << endl;
        return EXIT_FAILURE;
    }
    GlobMem_BUF_Wk = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, EMBED_DIM * EMBED_DIM * sizeof(int8_t), Wk, &errCode);
    if (errCode != CL_SUCCESS)
    {
        cout << endl
             << "Host-Error: Failed to allocate GlobMem_BUF_Wk" << endl
             << endl;
        return EXIT_FAILURE;
    }
    GlobMem_BUF_bk = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, EMBED_DIM * sizeof(float), bk, &errCode);
    if (errCode != CL_SUCCESS)
    {
        cout << endl
             << "Host-Error: Failed to allocate GlobMem_BUF_bk" << endl
             << endl;
        return EXIT_FAILURE;
    }
    GlobMem_BUF_scale_wk = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, EMBED_DIM * sizeof(float), scale_wk, &errCode);
    if (errCode != CL_SUCCESS)
    {
        cout << endl
             << "Host-Error: Failed to allocate GlobMem_BUF_scale_wk" << endl
             << endl;
        return EXIT_FAILURE;
    }
    GlobMem_BUF_Wv = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, EMBED_DIM * EMBED_DIM * sizeof(int8_t), Wv, &errCode);
    if (errCode != CL_SUCCESS)
    {
        cout << endl
             << "Host-Error: Failed to allocate GlobMem_BUF_Wv" << endl
             << endl;
        return EXIT_FAILURE;
    }
    GlobMem_BUF_bv = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, EMBED_DIM * sizeof(float), bv, &errCode);
    if (errCode != CL_SUCCESS)
    {
        cout << endl
             << "Host-Error: Failed to allocate GlobMem_BUF_bv" << endl
             << endl;
        return EXIT_FAILURE;
    }
    GlobMem_BUF_scale_wv = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, EMBED_DIM * sizeof(float), scale_wv, &errCode);
    if (errCode != CL_SUCCESS)
    {
        cout << endl
             << "Host-Error: Failed to allocate GlobMem_BUF_scale_wv" << endl
             << endl;
        return EXIT_FAILURE;
    }
    GlobMem_BUF_Q = clCreateBuffer(Context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, NUM_HEADS * SEQ_LEN * HEAD_DIM * sizeof(float), Q, &errCode);
    if (errCode != CL_SUCCESS)
    {
        cout << endl
             << "Host-Error: Failed to allocate GlobMem_BUF_Q" << endl
             << endl;
        return EXIT_FAILURE;
    }
    GlobMem_BUF_K = clCreateBuffer(Context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, NUM_HEADS * SEQ_LEN * HEAD_DIM * sizeof(float), K_out, &errCode);
    if (errCode != CL_SUCCESS)
    {
        cout << endl
             << "Host-Error: Failed to allocate GlobMem_BUF_K" << endl
             << endl;
        return EXIT_FAILURE;
    }
    GlobMem_BUF_V = clCreateBuffer(Context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, NUM_HEADS * SEQ_LEN * HEAD_DIM * sizeof(float), V, &errCode);
    if (errCode != CL_SUCCESS)
    {
        cout << endl
             << "Host-Error: Failed to allocate GlobMem_BUF_V" << endl
             << endl;
        return EXIT_FAILURE;
    }
    

    // ============================================================================
    // Step 5: Run 12 Spatial Transformer encoder blocks in sequence
    // ============================================================================
#ifdef ALL_MESSAGES
    cout << "HOST-Info: ============================================================= " << endl;
    cout << "HOST-Info: (Step 5) Run 12 Spatial Encoder Blocks                        " << endl;
    cout << "HOST-Info: ============================================================= " << endl;
#endif

    double Host_Post_Time_ms = 0.0;
    int Total_Kernel_Events = 0;
    int Total_Memory_Events = 0;
    vector<cl_event> Profiling_Kernel_Events;
    vector<cl_event> Profiling_Memory_Events;
    vector<string> Profiling_Kernel_Names;



    for (int block = 0; block < NUM_ENCODER_BLOCKS; block++)
    {
        for (int frame = 0; frame < FRAME; frame++)
        {
            const string spatial_encoder_dir = "params/SpatialTransformer/encoderblock_" + to_string(block) + "/";
            cout << "HOST-Info: Loading parameters for encoder block " << block << " from " << spatial_encoder_dir << endl;

            if (!load_txt_values_as_float(spatial_encoder_dir + "LayerNorm_0_scale.txt", LayerNorm_gamma, EMBED_DIM) ||
                !load_txt_values_as_float(spatial_encoder_dir + "LayerNorm_0_bias.txt", LayerNorm_beta, EMBED_DIM) ||
                !load_txt_values_as_int8(spatial_encoder_dir + "MultiHeadDotProductAttention_0_query_kernel_qvalue.txt", &Wq[0][0], EMBED_DIM * EMBED_DIM) ||
                !load_txt_values_as_float(spatial_encoder_dir + "MultiHeadDotProductAttention_0_query_bias.txt", bq, EMBED_DIM) ||
                !load_txt_values_as_float(spatial_encoder_dir + "MultiHeadDotProductAttention_0_query_kernel_scale.txt", scale_wq, EMBED_DIM) ||
                !load_txt_values_as_int8(spatial_encoder_dir + "MultiHeadDotProductAttention_0_key_kernel_qvalue.txt", &Wk[0][0], EMBED_DIM * EMBED_DIM) ||
                !load_txt_values_as_float(spatial_encoder_dir + "MultiHeadDotProductAttention_0_key_bias.txt", bk, EMBED_DIM) ||
                !load_txt_values_as_float(spatial_encoder_dir + "MultiHeadDotProductAttention_0_key_kernel_scale.txt", scale_wk, EMBED_DIM) ||
                !load_txt_values_as_int8(spatial_encoder_dir + "MultiHeadDotProductAttention_0_value_kernel_qvalue.txt", &Wv[0][0], EMBED_DIM * EMBED_DIM) ||
                !load_txt_values_as_float(spatial_encoder_dir + "MultiHeadDotProductAttention_0_value_bias.txt", bv, EMBED_DIM) ||
                !load_txt_values_as_float(spatial_encoder_dir + "MultiHeadDotProductAttention_0_value_kernel_scale.txt", scale_wv, EMBED_DIM) ||
                !load_txt_values_as_int8(spatial_encoder_dir + "MultiHeadDotProductAttention_0_out_kernel_qvalue.txt", &Wout[0][0], EMBED_DIM * EMBED_DIM) ||
                !load_txt_values_as_float(spatial_encoder_dir + "MultiHeadDotProductAttention_0_out_bias.txt", bout, EMBED_DIM) ||
                !load_txt_values_as_float(spatial_encoder_dir + "MultiHeadDotProductAttention_0_out_kernel_scale.txt", scale_wout, EMBED_DIM) ||
                !load_txt_values_as_float(spatial_encoder_dir + "LayerNorm_1_scale.txt", LayerNorm1_gamma, EMBED_DIM) ||
                !load_txt_values_as_float(spatial_encoder_dir + "LayerNorm_1_bias.txt", LayerNorm1_beta, EMBED_DIM) ||
                !load_txt_values_as_int8(spatial_encoder_dir + "MlpBlock_0_Dense_0_kernel_qvalue.txt", &MLP_W1[0][0], EMBED_DIM * MLP_DIM) ||
                !load_txt_values_as_float(spatial_encoder_dir + "MlpBlock_0_Dense_0_bias.txt", MLP_b1, MLP_DIM) ||
                !load_txt_values_as_float(spatial_encoder_dir + "MlpBlock_0_Dense_0_kernel_scale.txt", MLP_scale_w1, MLP_DIM) ||
                !load_txt_values_as_int8(spatial_encoder_dir + "MlpBlock_0_Dense_1_kernel_qvalue.txt", &MLP_W2[0][0], MLP_DIM * EMBED_DIM) ||
                !load_txt_values_as_float(spatial_encoder_dir + "MlpBlock_0_Dense_1_bias.txt", MLP_b2, EMBED_DIM) ||
                !load_txt_values_as_float(spatial_encoder_dir + "MlpBlock_0_Dense_1_kernel_scale.txt", MLP_scale_w2, EMBED_DIM))
            {
                cout << endl
                    << "HOST-Error: Failed to load one or more parameter files for encoder block " << block << endl
                    << endl;
                return EXIT_FAILURE;
            }

            cl_mem LayerNorm_Input_Buffers[] = {
                GlobMem_BUF_X[frame], GlobMem_BUF_LayerNorm_Gamma, GlobMem_BUF_LayerNorm_Beta};
            cl_mem LayerNorm_Output_Buffers[] = {
                GlobMem_BUF_X_norm[frame], GlobMem_BUF_LayerNorm_Scaler};
            cl_mem QKV_Input_Buffers[] = {
                GlobMem_BUF_Wq, GlobMem_BUF_bq, GlobMem_BUF_scale_wq,
                GlobMem_BUF_Wk, GlobMem_BUF_bk, GlobMem_BUF_scale_wk,
                GlobMem_BUF_Wv, GlobMem_BUF_bv, GlobMem_BUF_scale_wv};

            const int Nb_Of_Mem_Events = 8;
            const int Nb_Of_Exe_Events = 3 + NUM_HEADS;
            cl_event Mem_op_event[Nb_Of_Mem_Events];
            cl_event K_exe_event[Nb_Of_Exe_Events];
            memset(Mem_op_event, 0, sizeof(Mem_op_event));
            memset(K_exe_event, 0, sizeof(K_exe_event));

            // ----------------------------------------
            // Step 5.1: Set Kernel Arguments
            // ----------------------------------------
            errCode = false;
            errCode |= clSetKernelArg(K_layernorm, 0, sizeof(cl_mem), &GlobMem_BUF_X[frame]);
            errCode |= clSetKernelArg(K_layernorm, 1, sizeof(cl_mem), &GlobMem_BUF_LayerNorm_Gamma);
            errCode |= clSetKernelArg(K_layernorm, 2, sizeof(cl_mem), &GlobMem_BUF_LayerNorm_Beta);
            errCode |= clSetKernelArg(K_layernorm, 3, sizeof(cl_mem), &GlobMem_BUF_X_norm[frame]);
            errCode |= clSetKernelArg(K_layernorm, 4, sizeof(cl_mem), &GlobMem_BUF_LayerNorm_Scaler);
            errCode |= clSetKernelArg(K_linear_qkv_multihead, 0, sizeof(cl_mem), &GlobMem_BUF_X_norm[frame]);
            errCode |= clSetKernelArg(K_linear_qkv_multihead, 1, sizeof(cl_mem), &GlobMem_BUF_Wq);
            errCode |= clSetKernelArg(K_linear_qkv_multihead, 2, sizeof(cl_mem), &GlobMem_BUF_bq);
            errCode |= clSetKernelArg(K_linear_qkv_multihead, 3, sizeof(cl_mem), &GlobMem_BUF_Wk);
            errCode |= clSetKernelArg(K_linear_qkv_multihead, 4, sizeof(cl_mem), &GlobMem_BUF_bk);
            errCode |= clSetKernelArg(K_linear_qkv_multihead, 5, sizeof(cl_mem), &GlobMem_BUF_Wv);
            errCode |= clSetKernelArg(K_linear_qkv_multihead, 6, sizeof(cl_mem), &GlobMem_BUF_bv);
            errCode |= clSetKernelArg(K_linear_qkv_multihead, 7, sizeof(cl_mem), &GlobMem_BUF_LayerNorm_Scaler);
            errCode |= clSetKernelArg(K_linear_qkv_multihead, 8, sizeof(cl_mem), &GlobMem_BUF_scale_wq);
            errCode |= clSetKernelArg(K_linear_qkv_multihead, 9, sizeof(cl_mem), &GlobMem_BUF_scale_wk);
            errCode |= clSetKernelArg(K_linear_qkv_multihead, 10, sizeof(cl_mem), &GlobMem_BUF_scale_wv);
            errCode |= clSetKernelArg(K_linear_qkv_multihead, 11, sizeof(cl_mem), &GlobMem_BUF_Q);
            errCode |= clSetKernelArg(K_linear_qkv_multihead, 12, sizeof(cl_mem), &GlobMem_BUF_K);
            errCode |= clSetKernelArg(K_linear_qkv_multihead, 13, sizeof(cl_mem), &GlobMem_BUF_V);
            if (errCode != CL_SUCCESS)
            {
                cout << endl
                    << "Host-ERROR: Failed to set LN0/QKV kernel arguments for block " << block << endl
                    << endl;
                return EXIT_FAILURE;
            }

            // ------------------------------------------------------
            // Step 5.2: Copy Input Data from Host to Global Memory
            // ------------------------------------------------------
            errCode = clEnqueueMigrateMemObjects(Command_Queue, 3, LayerNorm_Input_Buffers, 0, 0, NULL, &Mem_op_event[0]);
            errCode |= clEnqueueMigrateMemObjects(Command_Queue, 2, LayerNorm_Output_Buffers, CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED, 0, NULL, &Mem_op_event[1]);
            errCode |= clEnqueueMigrateMemObjects(Command_Queue, 9, QKV_Input_Buffers, 0, 0, NULL, &Mem_op_event[2]);
            errCode |= clEnqueueMigrateMemObjects(Command_Queue, 1, &GlobMem_BUF_Attention_Out[frame], CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED, 0, NULL, &Mem_op_event[3]);
            if (errCode != CL_SUCCESS)
            {
                cout << endl
                    << "Host-Error: Failed to migrate LN0/QKV buffers for block " << block << endl
                    << endl;
                return EXIT_FAILURE;
            }

            // ----------------------------------------
            // Step 5.3: Submit Kernels for Execution
            // ----------------------------------------
            cl_event layernorm_wait_list[] = {Mem_op_event[0], Mem_op_event[1]};
            errCode = clEnqueueTask(Command_Queue, K_layernorm, 2, layernorm_wait_list, &K_exe_event[0]);
            if (errCode != CL_SUCCESS)
            {
                cout << endl
                    << "HOST-Error: Failed to submit LN0 layernorm for block " << block << endl
                    << endl;
                return EXIT_FAILURE;
            }

            cl_event qkv_wait_list[] = {K_exe_event[0], Mem_op_event[2]};
            errCode = clEnqueueTask(Command_Queue, K_linear_qkv_multihead, 2, qkv_wait_list, &K_exe_event[1]);
            if (errCode != CL_SUCCESS)
            {
                cout << endl
                    << "HOST-Error: Failed to submit linear_qkv_multihead for block " << block << endl
                    << endl;
                return EXIT_FAILURE;
            }

            cl_mem GlobMem_BUF_head_Q[NUM_HEADS] = {NULL};
            cl_mem GlobMem_BUF_head_K[NUM_HEADS] = {NULL};
            cl_mem GlobMem_BUF_head_V[NUM_HEADS] = {NULL};
            cl_mem GlobMem_BUF_head_Out[NUM_HEADS] = {NULL};
            const size_t HEAD_BYTES = SEQ_LEN * HEAD_DIM * sizeof(float);

            for (int h = 0; h < NUM_HEADS; h++)
            {
                cl_buffer_region head_region;
                head_region.origin = static_cast<size_t>(h) * HEAD_BYTES;
                head_region.size = HEAD_BYTES;


                GlobMem_BUF_head_Q[h] = clCreateSubBuffer(GlobMem_BUF_Q, CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &head_region, &errCode);
                if (errCode != CL_SUCCESS)
                    return EXIT_FAILURE;
                GlobMem_BUF_head_K[h] = clCreateSubBuffer(GlobMem_BUF_K, CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &head_region, &errCode);
                if (errCode != CL_SUCCESS)
                    return EXIT_FAILURE;
                GlobMem_BUF_head_V[h] = clCreateSubBuffer(GlobMem_BUF_V, CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &head_region, &errCode);
                if (errCode != CL_SUCCESS)
                    return EXIT_FAILURE;
                GlobMem_BUF_head_Out[h] = clCreateSubBuffer(GlobMem_BUF_Attention_Out[frame], CL_MEM_WRITE_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &head_region, &errCode);
                if (errCode != CL_SUCCESS)
                    return EXIT_FAILURE;

                errCode = false;
                errCode |= clSetKernelArg(K_attention_opt[h], 0, sizeof(cl_mem), &GlobMem_BUF_head_Q[h]);
                errCode |= clSetKernelArg(K_attention_opt[h], 1, sizeof(cl_mem), &GlobMem_BUF_head_K[h]);
                errCode |= clSetKernelArg(K_attention_opt[h], 2, sizeof(cl_mem), &GlobMem_BUF_head_V[h]);
                errCode |= clSetKernelArg(K_attention_opt[h], 3, sizeof(cl_mem), &GlobMem_BUF_head_Out[h]);
                if (errCode != CL_SUCCESS)
                {
                    cout << endl
                        << "Host-ERROR: Failed to set attention kernel arguments for block " << block << ", head " << h << endl
                        << endl;
                    return EXIT_FAILURE;
                }

                cl_event attention_wait_list[] = {K_exe_event[1], Mem_op_event[3]};
                errCode = clEnqueueTask(Command_Queue, K_attention_opt[h], 2, attention_wait_list, &K_exe_event[2 + h]);
                if (errCode != CL_SUCCESS)
                {
                    cout << endl
                        << "HOST-Error: Failed to submit attention for block " << block << ", head " << h << endl
                        << endl;
                    return EXIT_FAILURE;
                }
            }

            // ---------------------------------------------------------
            // Step 5.4: Copy Results from Device to Host
            // ---------------------------------------------------------
            errCode = clEnqueueMigrateMemObjects(Command_Queue, 1, &GlobMem_BUF_Attention_Out[frame], CL_MIGRATE_MEM_OBJECT_HOST, NUM_HEADS, &K_exe_event[2], &Mem_op_event[4]);
            if (errCode != CL_SUCCESS)
            {
                cout << endl
                    << "Host-Error: Failed to copy attention output to host for block " << block << endl
                    << endl;
                return EXIT_FAILURE;
            }
            clWaitForEvents(1, &Mem_op_event[4]);

            for (int h = 0; h < NUM_HEADS; h++)
            {
                clReleaseMemObject(GlobMem_BUF_head_Q[h]);
                clReleaseMemObject(GlobMem_BUF_head_K[h]);
                clReleaseMemObject(GlobMem_BUF_head_V[h]);
                clReleaseMemObject(GlobMem_BUF_head_Out[h]);
            }

            auto Host_Post_Start = chrono::high_resolution_clock::now();
            concat_attention_heads(Attention_Out, Attention_Concat);
            linear_quant_host_fp32_input(Attention_Concat, Wout, bout, scale_wout, Attention_Projected);
            residual_add(X, Attention_Projected, After_Attention_Residual);
            copy_int_to_float(After_Attention_Residual, X);
            auto Host_Post_Mid = chrono::high_resolution_clock::now();
            Host_Post_Time_ms += chrono::duration<double, std::milli>(Host_Post_Mid - Host_Post_Start).count();

            memcpy(LayerNorm_gamma, LayerNorm1_gamma, EMBED_DIM * sizeof(float));
            memcpy(LayerNorm_beta, LayerNorm1_beta, EMBED_DIM * sizeof(float));

            errCode = false;
            errCode |= clSetKernelArg(K_layernorm, 0, sizeof(cl_mem), &GlobMem_BUF_X[frame]);
            errCode |= clSetKernelArg(K_layernorm, 1, sizeof(cl_mem), &GlobMem_BUF_LayerNorm_Gamma);
            errCode |= clSetKernelArg(K_layernorm, 2, sizeof(cl_mem), &GlobMem_BUF_LayerNorm_Beta);
            errCode |= clSetKernelArg(K_layernorm, 3, sizeof(cl_mem), &GlobMem_BUF_X_norm[frame]);
            errCode |= clSetKernelArg(K_layernorm, 4, sizeof(cl_mem), &GlobMem_BUF_LayerNorm_Scaler);
            if (errCode != CL_SUCCESS)
            {
                cout << endl
                    << "Host-ERROR: Failed to set LN1 kernel arguments for block " << block << endl
                    << endl;
                return EXIT_FAILURE;
            }

            errCode = clEnqueueMigrateMemObjects(Command_Queue, 3, LayerNorm_Input_Buffers, 0, 0, NULL, &Mem_op_event[5]);
            errCode |= clEnqueueMigrateMemObjects(Command_Queue, 2, LayerNorm_Output_Buffers, CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED, 0, NULL, &Mem_op_event[6]);
            if (errCode != CL_SUCCESS)
            {
                cout << endl
                    << "Host-Error: Failed to migrate LN1 buffers for block " << block << endl
                    << endl;
                return EXIT_FAILURE;
            }

            cl_event layernorm1_wait_list[] = {Mem_op_event[5], Mem_op_event[6]};
            errCode = clEnqueueTask(Command_Queue, K_layernorm, 2, layernorm1_wait_list, &K_exe_event[2 + NUM_HEADS]);
            if (errCode != CL_SUCCESS)
            {
                cout << endl
                    << "HOST-Error: Failed to submit LN1 layernorm for block " << block << endl
                    << endl;
                return EXIT_FAILURE;
            }

            cl_event ln1_to_host_wait_list[] = {K_exe_event[2 + NUM_HEADS]};
            errCode = clEnqueueMigrateMemObjects(Command_Queue, 2, LayerNorm_Output_Buffers, CL_MIGRATE_MEM_OBJECT_HOST, 1, ln1_to_host_wait_list, &Mem_op_event[7]);
            if (errCode != CL_SUCCESS)
            {
                cout << endl
                    << "Host-Error: Failed to copy LN1 output to host for block " << block << endl
                    << endl;
                return EXIT_FAILURE;
            }
            clWaitForEvents(1, &Mem_op_event[7]);

            mlp_quant_host(X_norm, LayerNorm_scaler, MLP_W1, MLP_b1, MLP_scale_w1, MLP_W2, MLP_b2, MLP_scale_w2, MLP_Output);
            residual_add(After_Attention_Residual, MLP_Output, Spatial_Encoder_Output);
            copy_int_to_float(Spatial_Encoder_Output, X);
            auto Host_Post_End = chrono::high_resolution_clock::now();
            Host_Post_Time_ms += chrono::duration<double, std::milli>(Host_Post_End - Host_Post_Mid).count();

            for (int i = 0; i < Nb_Of_Mem_Events; i++)
            {
                if (Mem_op_event[i] != nullptr)
                {
                    Profiling_Memory_Events.push_back(Mem_op_event[i]);
                    Total_Memory_Events++;
                }
            }
            Profiling_Kernel_Events.push_back(K_exe_event[0]);
            Profiling_Kernel_Names.push_back("encoderblock_" + to_string(block) + "_layernorm_0");
            Profiling_Kernel_Events.push_back(K_exe_event[1]);
            Profiling_Kernel_Names.push_back("encoderblock_" + to_string(block) + "_linear_qkv");
            for (int h = 0; h < NUM_HEADS; h++)
            {
                Profiling_Kernel_Events.push_back(K_exe_event[2 + h]);
                Profiling_Kernel_Names.push_back("encoderblock_" + to_string(block) + "_attention_head_" + to_string(h));
            }
            Profiling_Kernel_Events.push_back(K_exe_event[2 + NUM_HEADS]);
            Profiling_Kernel_Names.push_back("encoderblock_" + to_string(block) + "_layernorm_1");
            Total_Kernel_Events += Nb_Of_Exe_Events;
        }
    }

    cout << endl
         << "HOST_Info: Waiting for application to be completed ..." << endl;
    clFinish(Command_Queue);

// ============================================================================
// Step 6: Processing Output Results
//         o) Store spatial encoder output results to a file
// ============================================================================
#ifdef ALL_MESSAGES
    cout << endl;
    cout << "HOST-Info: ============================================================= " << endl;
    cout << "HOST-Info: (Step 6) Store and Check the Output Results                   " << endl;
    cout << "HOST-Info: ============================================================= " << endl;
#endif

    // ------------------------------------------------------
    // Step 6.1: Store output Result to the spatial encoder output file
    // ------------------------------------------------------
    char Output_File_Name[] = "spatial_encoder_output.txt";
    cout << "HOST_Info: Store output results in: " << Output_File_Name << endl;

    fstream Output_File;
    Output_File.open(Output_File_Name, ios::out);
    if (!Output_File.is_open())
    {
        cout << endl
             << "HOST-Error: Failed to open the " << Output_File_Name << " file for write" << endl
             << endl;
        return EXIT_FAILURE;
    }

    for (int i = 0; i < SEQ_LEN; i++)
    {
        for (int d = 0; d < EMBED_DIM; d++)
        {
            Output_File << Spatial_Encoder_Output[i][d] << " ";
        }
        Output_File << endl;
    }
    Output_File.close();
    cout << "HOST-Info: Host post-processing time (concat + residual + MLP) = "
         << Host_Post_Time_ms << " ms" << endl;
    // ============================================================================
    // Step 7: Custom Profiling
    // ============================================================================
    cout << "HOST-Info: ============================================================= " << endl;
    cout << "HOST-Info: (Step 7) Custom Profiling                                     " << endl;
    cout << "HOST-Info: ============================================================= " << endl;
    int Nb_Of_Kernels = static_cast<int>(Profiling_Kernel_Events.size());
    int Nb_Of_Memory_Tranfers = static_cast<int>(Profiling_Memory_Events.size());
    if (Nb_Of_Kernels > 0 && Nb_Of_Memory_Tranfers > 0)
    {
        run_custom_profiling(
            Nb_Of_Kernels,
            Nb_Of_Memory_Tranfers,
            Profiling_Kernel_Events.data(),
            Profiling_Memory_Events.data(),
            Profiling_Kernel_Names.data());
    }

    // ============================================================================
    // Step 8: Release Allocated Resources
    // ============================================================================
    clReleaseDevice(Target_Device_ID); // Only available in OpenCL >= 1.2

    for (cl_event evt : Profiling_Memory_Events)
        clReleaseEvent(evt);
    for (cl_event evt : Profiling_Kernel_Events)
        clReleaseEvent(evt);

    for (int f = 0; f < FRAME; f++)
    {
        clReleaseMemObject(GlobMem_BUF_X[f]);
        clReleaseMemObject(GlobMem_BUF_X_norm[f]);
        clReleaseMemObject(GlobMem_BUF_Attention_Out[f]);
    }
    

    clReleaseMemObject(GlobMem_BUF_LayerNorm_Gamma);
    clReleaseMemObject(GlobMem_BUF_LayerNorm_Beta);
    
    clReleaseMemObject(GlobMem_BUF_LayerNorm_Scaler);
    clReleaseMemObject(GlobMem_BUF_Wq);
    clReleaseMemObject(GlobMem_BUF_bq);
    clReleaseMemObject(GlobMem_BUF_scale_wq);
    clReleaseMemObject(GlobMem_BUF_Wk);
    clReleaseMemObject(GlobMem_BUF_bk);
    clReleaseMemObject(GlobMem_BUF_scale_wk);
    clReleaseMemObject(GlobMem_BUF_Wv);
    clReleaseMemObject(GlobMem_BUF_bv);
    clReleaseMemObject(GlobMem_BUF_scale_wv);
    clReleaseMemObject(GlobMem_BUF_Q);
    clReleaseMemObject(GlobMem_BUF_K);
    clReleaseMemObject(GlobMem_BUF_V);

    
    clReleaseKernel(K_layernorm);
    clReleaseKernel(K_linear_qkv_multihead);
    for (int h = 0; h < NUM_HEADS; h++)
        clReleaseKernel(K_attention_opt[h]);

    clReleaseProgram(Program);
    clReleaseCommandQueue(Command_Queue);
    clReleaseContext(Context);

    free(Platform_IDs);
    free(Device_IDs);
    free(X);
    free(LayerNorm_gamma);
    free(LayerNorm_beta);
    free(X_norm);
    free(LayerNorm_scaler);
    free(Wq);
    free(bq);
    free(scale_wq);
    free(Wk);
    free(bk);
    free(scale_wk);
    free(Wv);
    free(bv);
    free(scale_wv);
    free(Wout);
    free(bout);
    free(scale_wout);
    free(LayerNorm1_gamma);
    free(LayerNorm1_beta);
    free(Q);
    free(K_out);
    free(V);
    free(Attention_Out);
    free(Attention_Concat);
    free(Attention_Projected);
    free(After_Attention_Residual);
    free(MLP_Input);
    free(MLP_Output);
    free(Spatial_Encoder_Output);
    free(MLP_W1);
    free(MLP_b1);
    free(MLP_scale_w1);
    free(MLP_W2);
    free(MLP_b2);
    free(MLP_scale_w2);

    cout << endl
         << "HOST-Info: DONE" << endl
         << endl;

    return EXIT_SUCCESS;
}
