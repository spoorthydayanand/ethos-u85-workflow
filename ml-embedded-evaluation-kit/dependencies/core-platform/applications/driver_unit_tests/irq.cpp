/*
 * SPDX-FileCopyrightText: Copyright 2021, 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/****************************************************************************
 * Includes
 ****************************************************************************/

#include "command_stream.hpp"

#include <ethosu_log.h>

#include <stdio.h>

using namespace std;
using namespace EthosU::CommandStream;

/****************************************************************************
 * Data
 ****************************************************************************/
// clang-format off
__attribute__((section(".sram.data"), aligned(16))) char commandStream[] = {
    DRIVER_ACTION_MAGIC()
    DRIVER_ACTION_NOP()
    DRIVER_ACTION_NOP()
    DRIVER_ACTION_COMMAND_STREAM(1)

    NPU_OP_STOP(0)
}; // clang-format on

/****************************************************************************
 * Functions
 ****************************************************************************/

int main() {
    CommandStream cs(DataPointer(commandStream, sizeof(commandStream)),
                     BasePointers(),
                     PmuEvents({ETHOSU_PMU_CYCLE, ETHOSU_PMU_NPU_IDLE, ETHOSU_PMU_NPU_ACTIVE}));

    const size_t repeat = 1000;

    // Clear PMU
    cs.getPmu().clear();

    // Run inference
    int ret             = cs.run(repeat);
    uint64_t cycleCount = cs.getPmu().getCycleCount();

    // Print PMU counters
    cs.getPmu().print();
    LOG("cycleCount=%llu, cycleCountPerJob=%llu\n", cycleCount, cycleCount / repeat);

    return ret;
}
