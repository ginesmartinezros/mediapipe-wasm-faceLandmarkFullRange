
<!-- ![GSOC](docs/images/gsoc.png) -->
![MediaPipe](docs/images/mediapipe_small.png) 

# MediaPipe GSOC 2021: Web Video Effects App
- Built on-top of the MediaPipe project.
- The Emscripten mapping and the BUILD files for the project are in `hello-world`.
- Summary of the work done during GSOC 2021 is [here](https://prantoran.me/2021/08/22/gsoc-mediapipe-video-effects-app).


## How to run the project
- Clone this repo, which is a fork of MediaPipe set up with Emscripten emsdk import in the `WORKSPACE` file.
- Install node modules for the server:
    - `cd hello-server && npm i`
- From the root directory of the project, run the command in the terminal:
    - `make build && make run`
    - The scripts that are executed are in `scripts/...` and `Makefile`.
    - `make build` compiles the WASM binaries:
        - `bazel build -c opt //hello-world:hello-world-simple --config=wasm`
    - `make run` copies the required outputs and runs a NodeJS server to run the WASM binaries locally.
        - Output files are copied from `bazel-out/wasm-opt/bin/hello-world/...`

## Notes
- The files important in `hello-server` are `hello-server/public/{index.html, renderer.js}`. The rest of the contents in the `hello-server` folder are for running the server and serving web app files efficiently.
- The symbol definition for `popen` in `hello-world/cpp` are not required in the WASM runtime but I kept it for debugging purposes. 

## Live Demo
- WASM versions of face detection, selfie segmentation and face mesh are demonstated in the following [web app](https://prantoran.me/diapipe).
