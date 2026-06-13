cmake_minimum_required(VERSION 3.20)

project(J9App C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

#CPU compile flags
set(MCPU_FLAGS
    -mcpu=cortex-m85
    -mthumb
    -O3
    -ffunction-sections
    -fdata-sections
)

#Application executable
add_executable(J9App.elf
    J9App.cpp
    model_data.cpp
    syscalls.c
)

#Include directories
target_include_directories(J9App.elf PRIVATE
    CMSIS-NN/Include
    tflite-micro
    ethos-u-driver
)

#Compiler flags
target_compile_options(J9App.elf PRIVATE
    ${MCPU_FLAGS}
)

#Linker flags
target_link_options(J9App.elf PRIVATE
    ${MCPU_FLAGS}
    -Wl,--gc-sections
)

#Size output
add_custom_command(TARGET J9App.elf POST_BUILD
    COMMAND arm-none-eabi-size J9App.elf
)

