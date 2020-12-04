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

class ArgMaxLayerImpl CV_FINAL : public ArgMaxLayer
{
public:
    ArgMaxLayerImpl(const LayerParams& params)
    {
        computeMaxIdx = true;
        hasDynamicShapes = params.get<bool>("has_dynamic_shapes", false);
	shapesInitialized = !hasDynamicShapes;
        String sense = toLowerCase(params.get<String>("sense", "max"));

        if (sense == "max")
		type = MAX;

        //if (params.has("pool") || params.has("kernel_size") ||
        //    params.has("kernel_w") || params.has("kernel_h"))
        //{
        //    String pool = toLowerCase(params.get<String>("pool", "max"));
        //    if (pool == "max")
        //        type = MAX;
        //    else if (pool == "ave")
        //        type = AVE;
        //    else if (pool == "stochastic")
        //        type = STOCHASTIC;
        //    else if (pool == "sum")
        //        type = SUM;
        //    else
        //        CV_Error(Error::StsBadArg, "Unknown pooling type \"" + pool + "\"");

        //    getPoolingKernelParams(params, kernel_size, isGlobalPooling, pads_begin, pads_end, strides, padMode);
        //    globalPooling = isGlobalPooling[0] || isGlobalPooling[1] || isGlobalPooling[2];
        //}
        //else if (params.has("pooled_w") || params.has("pooled_h"))
        //{
        //    type = ROI;
        //    pooledSize.width = params.get<uint32_t>("pooled_w", 1);
        //    pooledSize.height = params.get<uint32_t>("pooled_h", 1);
        //}
        //else if (params.has("output_dim") && params.has("group_size"))
        //{
        //    type = PSROI;
        //    pooledSize.width = params.get<int>("group_size");
        //    pooledSize.height = pooledSize.width;
        //    psRoiOutChannels = params.get<int>("output_dim");
        //}
        //else
        //    CV_Error(Error::StsBadArg, "Cannot determine pooling type");
        //setParamsFrom(params);
        //ceilMode = params.get<bool>("ceil_mode", true);
        //spatialScale = params.get<float>("spatial_scale", 1);
        //avePoolPaddedArea = params.get<bool>("ave_pool_padded_area", true);
    }

    void finalize(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr) CV_OVERRIDE
    {
        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        CV_Assert(!inputs.empty());

        std::vector<int> inp;
        std::vector<int> out;
        //for (int i = 2; i < inputs[0].dims; i++) {
        //    inp.push_back(inputs[0].size[i]);
        //    out.push_back(outputs[0].size[i]);
        //}
        //if (globalPooling) {
        //    std::vector<size_t> finalKernel;
        //    for (int i = 0; i < inp.size(); i++) {
        //        int idx = isGlobalPooling.size() - inp.size() + i;
        //        finalKernel.push_back(isGlobalPooling[idx] ? inp[i] : kernel_size[idx]);
        //     }
        //     kernel_size = finalKernel;
        // }

        //getConvPoolPaddings(inp, kernel_size, strides, padMode, pads_begin, pads_end);

        //if (inputs[0].dims == 3)
        //{
        //    //Pool1D
        //    kernel_size.erase(kernel_size.begin() + 1);
        //    strides.erase(strides.begin() + 1);
        //    pads_begin.erase(pads_begin.begin() + 1);
        //    pads_end.erase(pads_end.begin() + 1);
        //}

        //computeMaxIdx = type == MAX && outputs.size() == 2;
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        return (backendId == DNN_BACKEND_DEFAULT);
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        //if (type == MAX || type == MIN)
        //{
        //    CV_OCL_RUN(IS_DNN_OPENCL_TARGET(preferableTarget),
        //               forward_ocl(inputs_arr, outputs_arr, internals_arr))
        //}
        //if (inputs_arr.depth() == CV_16S)
        //{
        //    forward_fallback(inputs_arr, outputs_arr, internals_arr);
        //    return;
        //}

        //std::vector<Mat> inputs, outputs;
        //inputs_arr.getMatVector(inputs);
        //outputs_arr.getMatVector(outputs);

        //switch (type)
        //{
        //    case MAX:
        //    {
        //        CV_Assert_N(inputs.size() == 1, !computeMaxIdx || outputs.size() == 2);
        //        Mat mask = computeMaxIdx ? outputs[1] : Mat();
        //        maxPooling(inputs[0], outputs[0], mask);
        //        break;
        //    }
        //    case AVE: case SUM:
        //        CV_Assert_N(inputs.size() == 1, outputs.size() == 1);
        //        avePooling(inputs[0], outputs[0]);
        //        break;
        //    case ROI: case PSROI:
        //        CV_Assert_N(inputs.size() == 2, outputs.size() == 1);
        //        roiPooling(inputs[0], inputs[1], outputs[0]);
        //        break;
        //    default:
        //        CV_Error(Error::StsNotImplemented, "Not implemented");
        //        break;
        //}
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
    enum Type
    {
        MAX,
        MIN
    };
    bool hasDynamicShapes;
    bool shapesInitialized;
};

Ptr<ArgMaxLayer> ArgMaxLayer::create(const LayerParams& params)
{
    return Ptr<ArgMaxLayer>(new ArgMaxLayerImpl(params));
}

}
}
