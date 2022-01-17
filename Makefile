
build:
	bazel build -c opt //hello-world:hello-world-simple --config=wasm
run:
	rm -f -r hello-server/public/hello-world-simple.js hello-server/public/hello-world-simple.wasm hello-server/public/hello-world-simple.data
	cp -r bazel-out/wasm-opt/bin/hello-world/hello-world-simple.js hello-server/public/
	cp -r bazel-out/wasm-opt/bin/hello-world/hello-world-simple.wasm hello-server/public/
	cp -r bazel-out/wasm-opt/bin/hello-world/hello-world-simple.data hello-server/public/
	./scripts/runserver.sh
	