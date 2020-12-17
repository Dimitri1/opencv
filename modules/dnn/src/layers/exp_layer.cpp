/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Copyright (C) 2017, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "../precomp.hpp"
#include "layers_common.hpp"
#include "opencv2/core/hal/intrin.hpp"
#include "../op_inf_engine.hpp"
#include <float.h>
#include <algorithm>
#include <numeric>
using std::max;
using std::min;

namespace cv
{
namespace dnn
{

class ExpLayerImpl CV_FINAL : public ExpLayer
{
public:
    ExpLayerImpl(const LayerParams& params)
    {
    }

    void finalize(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr) CV_OVERRIDE
    {
        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        CV_Assert(!inputs.empty());

        std::vector<int> inp;
        std::vector<int> out;
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        return (backendId == DNN_BACKEND_DEFAULT);
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

    }


    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() != 0);

        std::vector<int> inpShape;
        for (int i= 0; i < inputs[0].size(); i++)
	  inpShape.push_back(inputs[0][i]);

        int numOutputs = 1;
	std::vector<int> outShape = {numOutputs};
        outputs.assign(numOutputs, outShape);
	return false;
    }

    bool updateMemoryShapes(const std::vector<MatShape> &inputs) CV_OVERRIDE
    {
        int dims = inputs[0].size();
        CV_Assert(inputs[0][dims - 1] > 0 && inputs[0][dims - 2] > 0);
        shapesInitialized = true;
        return true;
    }

    virtual int64 getFLOPS(const std::vector<MatShape> &inputs,
                           const std::vector<MatShape> &outputs) const CV_OVERRIDE
    {
        long flops = 0;
        for(int i = 0; i < outputs.size(); i++)
        {
            flops += total(outputs[i]);
        }
        return flops;
    }
private:
    bool hasDynamicShapes;
    bool shapesInitialized;
};

Ptr<ExpLayer> ExpLayer::create(const LayerParams& params)
{
    return Ptr<ExpLayer>(new ExpLayerImpl(params));
}

}
}
