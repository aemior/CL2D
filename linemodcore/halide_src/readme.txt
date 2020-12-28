To get Halide quant_ori.a and quant_ori.h

Use
g++ quant_ori_gen.cpp /path/to/Halide/install/tools/GenGen.cpp -I /path/to/halide/install/include -L ../path/to/halide/bin -lHalide -lpthread -ldl -o quant_ori_gen

Then
./quant_ori_gen -o . -g quant_ori_generator -f quant_ori -e static_library,h,schedule target=host auto_schedule=true