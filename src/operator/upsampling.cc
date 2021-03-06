/*!
 * Copyright (c) 2015 by Contributors
 * \file upsampling_nearest.cc
 * \brief
 * \author Bing Xu
*/

#include <nnvm/op_attr_types.h>
#include "./deconvolution-inl.h"
#include "./upsampling-inl.h"

namespace mxnet {
namespace op {
template <>
Operator *CreateOp<cpu>(UpSamplingParam param, int dtype) {
    Operator *op = NULL;
    MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
        if (param.sample_type == up_enum::kNearest) {
            op = new UpSamplingNearestOp<cpu, DType>(param);
        } else if (param.sample_type == up_enum::kBilinear) {
            DeconvolutionParam p = DeconvolutionParam();
            int kernel = 2 * param.scale - param.scale % 2;
            int stride = param.scale;
            int pad = static_cast<int>(ceil((param.scale - 1) / 2.));
            p.workspace = param.workspace;
            p.num_group = param.num_filter;
            p.num_filter = param.num_filter;
            p.no_bias = true;
            int shape[] = {1, 1};
            p.dilate = TShape(shape, shape + 2);
            shape[0] = shape[1] = kernel;
            p.kernel = TShape(shape, shape + 2);
            shape[0] = shape[1] = stride;
            p.stride = TShape(shape, shape + 2);
            shape[0] = shape[1] = pad;
            p.pad = TShape(shape, shape + 2);
            op = new DeconvolutionOp<cpu, DType>(p);
        } else {
            LOG(FATAL) << "Unknown sample type";
        }
    });
    return op;
}

Operator *UpSamplingProp::CreateOperatorEx(Context ctx,
                                           std::vector<TShape> *in_shape,
                                           std::vector<int> *in_type) const {
    std::vector<TShape> out_shape, aux_shape;
    std::vector<int> out_type, aux_type;
    CHECK(InferType(in_type, &out_type, &aux_type));
    CHECK(InferShape(in_shape, &out_shape, &aux_shape));
    DO_BIND_DISPATCH(CreateOp, param_, in_type->at(0));
}

DMLC_REGISTER_PARAMETER(UpSamplingParam);

MXNET_REGISTER_OP_PROPERTY(UpSampling, UpSamplingProp)
    .describe("Performs nearest neighbor/bilinear up sampling to inputs.")
    .add_argument("data", "NDArray-or-Symbol[]", "Array of tensors to upsample")
    .add_arguments(UpSamplingParam::__FIELDS__())
    .set_key_var_num_args("num_args");

NNVM_REGISTER_OP(UpSampling)
    .set_attr<nnvm::FSetInputVarAttrOnCompose>(
        "FSetInputVarAttrOnCompose",
        [](const nnvm::NodeAttrs &attrs, nnvm::NodePtr var, const int index) {
            if (var->attrs.dict.find("__init__") != var->attrs.dict.end())
                return;
            if (index == 1) {
                var->attrs.dict["__init__"] = "[\"bilinear\", {}]";
            }
        });
}  // namespace op
}  // namespace mxnet
