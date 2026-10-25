// Stub header for compilation without PDK
#ifndef NVMEDIA_CORE_H
#define NVMEDIA_CORE_H
#include <stdint.h>
// NvMediaDevice already forward-declared in dw/core/Config.h as struct
typedef uint32_t NvMediaBool;
typedef struct { uint32_t major; uint32_t minor; } NvMediaVersion;
typedef enum { NVMEDIA_STATUS_OK = 0 } NvMediaStatus;
#endif
