/*
 * SPDX-FileCopyrightText: Copyright 2022-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#if defined(ETHOSU)

#include <ethosu_driver.h>
#include <ethosu_log.h>

#include <FreeRTOS.h>
#include <semphr.h>

#include <stdio.h>

extern "C" {
void *ethosu_mutex_create(void) {
    return xSemaphoreCreateMutex();
}

void ethosu_mutex_destroy(void *mutex) {
    SemaphoreHandle_t handle = reinterpret_cast<SemaphoreHandle_t>(mutex);
    vSemaphoreDelete(handle);
}

int ethosu_mutex_lock(void *mutex) {
    SemaphoreHandle_t handle = reinterpret_cast<SemaphoreHandle_t>(mutex);
    if (xSemaphoreTake(handle, portMAX_DELAY) != pdTRUE) {
        LOG_ERR("Failed to lock mutex");
        return -1;
    }
    return 0;
}

int ethosu_mutex_unlock(void *mutex) {
    SemaphoreHandle_t handle = reinterpret_cast<SemaphoreHandle_t>(mutex);
    if (xSemaphoreGive(handle) != pdTRUE) {
        LOG_ERR("Failed to unlock mutex");
        return -1;
    }
    return 0;
}

void *ethosu_semaphore_create(void) {
    return xSemaphoreCreateCounting(255, 0);
}

void ethosu_semaphore_destroy(void *sem) {
    SemaphoreHandle_t handle = reinterpret_cast<SemaphoreHandle_t>(sem);
    vSemaphoreDelete(handle);
}

int ethosu_semaphore_take(void *sem, uint64_t timeout) {
    SemaphoreHandle_t handle = reinterpret_cast<SemaphoreHandle_t>(sem);
    if (xSemaphoreTake(handle, (TickType_t)timeout) != pdTRUE) {
        // Semaphore take timed out
        return -1;
    }
    return 0;
}

int ethosu_semaphore_give(void *sem) {
    SemaphoreHandle_t handle = reinterpret_cast<SemaphoreHandle_t>(sem);
    if (xPortIsInsideInterrupt()) {
        if (xSemaphoreGiveFromISR(handle, NULL) != pdTRUE) {
            LOG_ERR("Failed to give semaphore from ISR");
            return -1;
        }
    } else {
        if (xSemaphoreGive(handle) != pdTRUE) {
            LOG_ERR("Failed to give semaphore");
            return -1;
        }
    }
    return 0;
}
}

#endif
