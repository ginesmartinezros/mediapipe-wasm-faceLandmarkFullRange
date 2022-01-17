

#include "mediapipe/framework/port/status.h"


#ifdef __cplusplus
extern "C" {
#endif


FILE * popen(const char *command, const char *type) {
    LOG(ERROR) << "popen should never be called on web.";
    return NULL;
}

#ifdef __cplusplus
} // end extern "C"
#endif