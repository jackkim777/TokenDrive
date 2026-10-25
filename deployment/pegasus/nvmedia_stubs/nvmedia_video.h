// Stub
#ifndef NVMEDIA_VIDEO_H
#define NVMEDIA_VIDEO_H
#include "nvmedia_core.h"
#include "nvmedia_surface.h"
typedef void NvMediaIPPManager;
typedef void NvMediaIPPComponent;
typedef int NvMediaIPPComponentType;
typedef int NvMediaIPPEventType;
typedef struct { struct { uint32_t frameSequenceNumber; uint64_t frameCaptureGlobalTimeStamp; } imageInformation; } NvMediaIPPEventData;
#define NvMediaIPPManagerSetEventCallback(mgr, ctx, cb)
#define NVMEDIA_IPP_EVENT_WARNING_CAPTURE_FRAME_DROP 0
#define NVMEDIA_IPP_EVENT_WARNING_CSI_FRAME_DROP 1
#define NVMEDIA_IPP_EVENT_ERROR_CSI_INPUT_STREAM_FAILURE 2
#define NVMEDIA_IPP_EVENT_INFO_PROCESSING_DONE 3
#endif
