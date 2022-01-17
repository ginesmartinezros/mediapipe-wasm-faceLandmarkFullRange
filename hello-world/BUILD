load("@rules_cc//cc:defs.bzl", "cc_binary")
load("@emsdk//emscripten_toolchain:wasm_rules.bzl", "wasm_cc_binary")

package(default_visibility = ["//visibility:public"])

# notes
# warning: Link with `-s LLD_REPORT_UNDEFINED` to get more information on undefined symbols

cc_binary(
    name = "hello-world-simple",
    srcs = [
        "main.cpp",
    ],
    linkopts = [
        "-s USE_PTHREADS=0",
        "-s ALLOW_MEMORY_GROWTH=1",
        "-s ERROR_ON_UNDEFINED_SYMBOLS=1",
        "-s USE_WEBGL2=1",
        "--export-all",
        # "--no-entry",
        "-s ASSERTIONS=1",
        "--bind",
        "-s STANDALONE_WASM=0", # generates only wasm file, not .js file? generates independent wasm, but missing utilities of js
        # "--preload-file ../mediapipe/assets@assets",
        "--preload-file ../mediapipe/mediapipe/modules/face_detection/face_detection_short_range.tflite@mediapipe/modules/face_detection/face_detection_short_range.tflite",
        # "--preload-file ../mediapipe/mediapipe/@mediapipe/",
        "--preload-file ../mediapipe/mediapipe/modules/selfie_segmentation/selfie_segmentation.tflite@mediapipe/modules/selfie_segmentation/selfie_segmentation.tflite",
        "--preload-file ../mediapipe/mediapipe/modules/selfie_segmentation/selfie_segmentation_landscape.tflite@mediapipe/modules/selfie_segmentation/selfie_segmentation_landscape.tflite",
        "--preload-file ../mediapipe/mediapipe/modules/face_landmark/face_landmark.tflite@mediapipe/modules/face_landmark/face_landmark.tflite",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//hello-world/cpp:popen",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/formats:location_data_cc_proto",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/gpu:gpu_buffer",
        "//mediapipe/gpu:gpu_shared_data_internal",
        "//mediapipe/gpu:gl_calculator_helper",
        "//mediapipe/gpu:image_frame_to_gpu_buffer_calculator",
        "//mediapipe/gpu:gl_simple_calculator",
        "//mediapipe/gpu:gl_simple_shaders",
        "//mediapipe/framework/formats:landmark_cc_proto",
        # face detection
        "//mediapipe/modules/face_detection:face_detection_short_range_gpu",
        # selfie segmentation
        "//mediapipe/modules/selfie_segmentation:selfie_segmentation_gpu",
        "//mediapipe/graphs/selfie_segmentation:selfie_segmentation_gpu_deps",
        # # face mesh
        "//mediapipe/calculators/core:constant_side_packet_calculator",
        "//mediapipe/graphs/face_mesh/subgraphs:face_renderer_gpu",
        "//mediapipe/modules/face_landmark:face_landmark_front_gpu",
    ],
)
