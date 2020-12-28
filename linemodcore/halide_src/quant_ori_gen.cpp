// Halide tutorial lesson 15: Generators part 1

// This lesson demonstrates how to encapsulate Halide pipelines into
// reusable components called generators.

// On linux, you can compile and run it like so:
// g++ lesson_15*.cpp ../tools/GenGen.cpp -g -std=c++11 -fno-rtti -I ../include -L ../bin -lHalide -lpthread -ldl -o lesson_15_generate
// bash lesson_15_generators_usage.sh

// On os x:
// g++ lesson_15*.cpp ../tools/GenGen.cpp -g -std=c++11 -fno-rtti -I ../include -L ../bin -lHalide -o lesson_15_generate
// bash lesson_15_generators_usage.sh

// If you have the entire Halide source tree, you can also build it by
// running:
//    make tutorial_lesson_15_generators
// in a shell with the current directory at the top of the halide
// source tree.

#include "Halide.h"
#include <stdio.h>

using namespace Halide;

// Generators are a more structured way to do ahead-of-time
// compilation of Halide pipelines. Instead of writing an int main()
// with an ad-hoc command-line interface like we did in lesson 10, we
// define a class that inherits from Halide::Generator.
class QuantOriGenerator : public Halide::Generator<QuantOriGenerator> {
public:
    Input<double_t> thr{"thr"};
    Input<Buffer<uint8_t>> input{"input", 3};

    Output<Buffer<uint8_t>> quantori{"quantori", 2};
    Output<Buffer<_Float32>> mag_out{"mag_out", 2};

    Var x, y, c;

    void generate() {
        input.dim(0).set_stride(3);  // stride in dimension 0 (x) is three
        input.dim(2).set_stride(1); 
        input.dim(2).set_bounds(0, 3);
        Func in_bounded_m= BoundaryConditions::repeat_edge(input);
        Func in_bounded;
        in_bounded(x,y,c) = cast<_Float32>(in_bounded_m(x,y,c));

        Func blur_y("blur_y");
        blur_y(x,y,c) = (0.28125f * in_bounded(x, y, c) +
                        0.21875f * (in_bounded(x, y-1, c) +
                                        in_bounded(x, y+1, c)) +
                        0.109375f * (in_bounded(x, y-2, c) +
                                        in_bounded(x, y+2, c)) +
                        0.03125f * (in_bounded(x, y-3, c) +
                                    in_bounded(x, y+3, c)));
        Func blur_x("blur_x");
        blur_x(x,y,c) = (0.28125f * blur_y(x, y, c) +
                        0.21875f * (blur_y(x-1, y, c) +
                                        blur_y(x+1, y, c)) +
                        0.109375f * (blur_y(x-2, y, c) +
                                        blur_y(x+2, y, c)) +
                        0.03125f * (blur_y(x-3, y, c) +
                                    blur_y(x+3, y, c)));

        Func sobel_x("soble_x");
        sobel_x(x,y,c) = blur_x(x+1,y-1,c) - blur_x(x-1,y-1,c)+
                        blur_x(x+1,y+1,c) - blur_x(x-1,y+1,c)+
                        2*(blur_x(x+1,y,c) - blur_x(x-1,y,c));
        Func sobel_y("soble_y");
        sobel_y(x,y,c) = blur_x(x-1,y+1,c) - blur_x(x-1,y-1,c)+
                        blur_x(x+1,y+1,c) - blur_x(x+1,y-1,c)+
                        2*(blur_x(x,y+1,c) - blur_x(x,y-1,c));

        Func mag3("mag3c");
        Func mag_max_idx("mag_max_idx");
        Func mag_max("mag_max");
        Func sobel_phase("soble_phase");
        RDom max_chann(0,3);

        mag3(x,y,c) = sobel_x(x,y,c)*sobel_x(x,y,c) + sobel_y(x,y,c)*sobel_y(x,y,c);
        Tuple arg_res = argmax(mag3(x,y,max_chann));
        mag_max_idx(x,y) = clamp(arg_res[0], 0,2);
        mag_max(x,y) = arg_res[1];

        Expr phase_ori = atan2(sobel_y(x,y,mag_max_idx(x,y)), sobel_x(x,y,mag_max_idx(x,y)));
        sobel_phase(x,y) = (6.283185307179586f+phase_ori)%6.283185307179586f;

        Func quant_ori("quant_res");
   
        quant_ori(x,y) = cast<uint8_t>(sobel_phase(x,y)/0.39269908169872414f)&7;

        //hist quant
        Func hist_quant("hist");
        hist_quant(x,y,c) = 0;
        RDom r(-1, 3, -1, 3);
        RDom mc(0, 8);

        Expr bin = cast<uint8_t>(clamp(quant_ori(x+r.x, y+r.y), 0, 8));
        hist_quant(x,y,bin) += 1;
    
        Tuple arg_max = argmax(hist_quant(x,y,mc));
        Func quant_index, quant_num;
        quant_index(x,y) = cast<uint8_t>(arg_max[0]);
        quant_num(x,y) = arg_max[1];

        Func quant_filter;
        quant_filter(x,y) = select(mag_max(x,y)>thr, select(quant_num(x,y)>=5, 1 << quant_index(x,y), 0), 0);

        
        mag_out(x,y) = mag_max(x,y);
        quantori(x,y) = cast<uint8_t>(quant_filter(x,y));
        if(auto_schedule){
            input.set_estimates({{0, 3000}, {0, 3000}, {0, 3}});
            thr.set_estimate(10.0f);
            quantori.set_estimates({{0, 3000}, {0, 3000}});
            mag_out.set_estimates({{0, 3000}, {0, 3000}});
        }
        else{
        //schedule============
        blur_x.compute_root().vectorize(x, 8).parallel(y);
        blur_y.compute_at(blur_x, y).vectorize(x, 8);
        sobel_y.compute_root().vectorize(x,8).parallel(y);
        sobel_x.compute_root().vectorize(x,8).parallel(y);
        quant_ori.compute_root().parallel(y).vectorize(x,8);
        hist_quant.compute_root().parallel(y,4);
        //==================
        }
    }
};

// We compile this file along with tools/GenGen.cpp. That file defines
// an "int main(...)" that provides the command-line interface to use
// your generator class. We need to tell that code about our
// generator. We do this like so:
HALIDE_REGISTER_GENERATOR(QuantOriGenerator, quant_ori_generator)


