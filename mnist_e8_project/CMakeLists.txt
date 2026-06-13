cmake_minimum_required(VERSION 3.16)
project(J9App C CXX ASM)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

set(TFLM_ROOT /home/spoorthy_dayanand/project/tflite-micro)
set(ETHOSU_ROOT /home/spoorthy_dayanand/project/ethos-u-driver)
set(CMSIS_ROOT /home/spoorthy_dayanand/project/CMSIS_5)

set(CPU_FLAGS -mcpu=cortex-m55 -mthumb -mfloat-abi=hard)

add_executable(J9App.elf
    J9App.cpp
    model_data.cpp
    syscalls.c
)

target_include_directories(J9App.elf PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${TFLM_ROOT}
    ${TFLM_ROOT}/tensorflow/lite/micro/tools/make/downloads/flatbuffers/include
    ${TFLM_ROOT}/tensorflow/lite/micro/tools/make/downloads/gemmlowp
    ${TFLM_ROOT}/tensorflow/lite/micro/tools/make/downloads/cmsis/CMSIS/Core/Include
    ${ETHOSU_ROOT}/include
    ${CMSIS_ROOT}/CMSIS/Core/Include
)

target_compile_options(J9App.elf PRIVATE
    ${CPU_FLAGS}
    -O2
    -ffunction-sections
    -fdata-sections
    -fno-exceptions
    -fno-rtti
    -Wall
    -Wno-unused-parameter
    -DETHOSU_ARCH_U85
    -DCMSIS_NN
)

target_link_options(J9App.elf PRIVATE
    ${CPU_FLAGS}
    -specs=nano.specs
    -specs=nosys.specs
    -Wl,--gc-sections
    -Wl,--print-memory-usage -Wl,-Map=J9App.map
)

target_link_libraries(J9App.elf PRIVATE
    ${TFLM_ROOT}/gen/cortex_m_generic_cortex-m55_default_ethos_u_gcc/lib/libtensorflow-microlite.a
    ${ETHOSU_ROOT}/build/libethosu_core_driver.a
    m c stdc++
)

add_custom_command(TARGET J9App.elf POST_BUILD
    COMMAND arm-none-eabi-size J9App.elf
    COMMENT "Firmware size:"
)
