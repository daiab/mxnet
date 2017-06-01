/*!
 *  Copyright (c) 2014 by Contributors
 * \file tensor_blob.h
 * \brief TBlob class that holds common representation of
 *  arbirary dimension tensor, can be used to transformed
 *  to normal fixed dimenson tensor
 * \author Tianqi Chen
 */
#ifndef MXNET_TENSOR_BLOB_H_
#define MXNET_TENSOR_BLOB_H_

#include <dlpack/dlpack.h>
#include <dmlc/json.h>
#include <dmlc/logging.h>
#include <algorithm>
#include <iostream>
#include <utility>
#include <vector>
#include "./base.h"
#if MXNET_USE_MKL2017 == 1
#include <mkl_memory.h>
#endif
namespace mxnet {

/*!
 * \brief tensor blob class that can be used to hold tensor of any dimension,
 *  any device and any data type,
 *  This is a weak type that can be used to transfer data through interface
 *  TBlob itself do not involve any arithmentic operations,
 *  but it can be converted to tensor of fixed dimension for further operations
 *
 *  Like tensor, this data structure is like a pointer class and do not
 *  implicit allocated, de-allocate space.
 *  This data structure can be helpful to hold tensors of different dimensions
 *  and wait for further processing
 */
class TBlob {
   public:
    /*! \brief pointer to the data */
    void *dptr_;
    /*! \brief shape of the tensor */
    TShape shape_;
    /*! \brief type flag of the tensor blob */
    int type_flag_;

/*! \brief storing mkl chunk buffer blob, use for experimental only */
#if MKL_EXPERIMENTAL == 1
    std::shared_ptr<MKLMemHolder> Mkl_mem_;
#endif
    /*! \brief default constructor, default copy assign will work */
    TBlob(void) : dptr_(NULL), type_flag_(mshadow::DataType<real_t>::kFlag) {
#if MKL_EXPERIMENTAL == 1
        Mkl_mem_ = NULL;
#endif
        SetDLTensor(cpu::kDevMask, 0);
    }
    /*!
     * \brief constructor that construct TBlob from contiguous memory
     * \param dptr the pointer to the memory
     * \param shape the shape of the data
     * \param dev_mask the device mask, can be cpu::kDevMask or gpu::kDevMask
     * \param dev_id the device id
     */
    template <typename DType>
    TBlob(DType *dptr, const TShape &shape, int dev_mask, int dev_id = -1)
        : dptr_(dptr),
          shape_(shape),
          type_flag_(mshadow::DataType<DType>::kFlag) {
#if MKL_EXPERIMENTAL == 1
        Mkl_mem_ = NULL;
#endif
        SetDLTensor(dev_mask, dev_id);
    }
#if MKL_EXPERIMENTAL == 1
    /*!
     * \brief constructor that construct TBlob from contiguous memory
     * \param dptr the pointer to the memory
     * \param shape the shape of the data
     * \param dev_mask the device mask, can be cpu::kDevMask or gpu::kDevMask
     * \param dev_id the device id
     * \param Mkl_mem the mkl memory
     */
    template <typename DType>
    TBlob(DType *dptr, const TShape &shape, int dev_mask, int dev_id,
          std::shared_ptr<MKLMemHolder> Mkl_mem)
        : dptr_(dptr),
          shape_(shape),
          type_flag_(mshadow::DataType<DType>::kFlag),
          Mkl_mem_(Mkl_mem) {
        SetDLTensor(dev_mask, dev_id);
    }
#endif
    /*!
     * \brief constructor that construct TBlob from contiguous memory
     * \param dptr the pointer to the memory
     * \param shape the shape of the data
     * \param dev_mask the device mask, can be cpu::kDevMask or gpu::kDevMask
     * \param type_flag the type flag. Can be one of enum mshadow::dtype
     * \param dev_id the device id
     */
    TBlob(void *dptr, const TShape &shape, int dev_mask, int type_flag,
          int dev_id = -1)
        : dptr_(dptr), shape_(shape), type_flag_(type_flag) {
#if MKL_EXPERIMENTAL == 1
        Mkl_mem_ = NULL;
#endif
        SetDLTensor(dev_mask, dev_id);
    }
    /*!
     * \brief constructor from tensor
     * \param src source tensor
     * \tparam Device which device the tensor is on
     * \tparam dim tensor dimension
     * \tparam DType the type of elements in the tensor
     */
    template <typename Device, int dim, typename DType>
    TBlob(const mshadow::Tensor<Device, dim, DType> &src) {  // NOLINT(*)
        *this = src;
    }
    /*!
     * \brief assignment from tensor
     * \param src source tensor
     * \tparam Device which device the tensor is on
     * \tparam dim tensor dimension
     * \tparam DType the type of elements in the tensor
     * \return reference of self
     */
    template <typename Device, int dim, typename DType>
    inline TBlob &operator=(const mshadow::Tensor<Device, dim, DType> &src) {
        dptr_ = src.dptr_;
        shape_ = src.shape_;
        type_flag_ = mshadow::DataType<DType>::kFlag;
        SetDLTensor(Device::kDevMask, -1);
#if MKL_EXPERIMENTAL == 1
        Mkl_mem_ = NULL;
#endif
        return *this;
    }
    /*!
     * \return whether the tensor's memory is continuous
     */
    inline bool CheckContiguous(void) const { return true; }
    /*!
     * \brief reshape to shape
     * \param shape desired shape
     * \return reshaped blob
     */
    inline TBlob reshape(const TShape &shape) const {
        CHECK_EQ(this->shape_.Size(), shape.Size())
            << "Shape size mismatch " << this->shape_.Size() << " v.s. "
            << shape.Size();
        TBlob ret(this->dptr_, shape, this->dev_mask(), this->type_flag_,
                  this->dev_id());
        return ret;
    }
    /*!
     * \brief flatten the tensor to 2 dimension, collapse the higher dimensions
     * together
     * \param stream the possible stream target tensor should reside on
     * \tparam Device which device the tensor is on
     * \tparam DType the type of elements in the tensor
     * \return tensor after flatten
     */
    template <typename Device, typename DType>
    inline mshadow::Tensor<Device, 2, DType> FlatTo2D(
        mshadow::Stream<Device> *stream = NULL) const {
        CHECK(Device::kDevMask == this->dev_mask())
            << "TBlob.get: device type do not match specified type";
        CHECK(mshadow::DataType<DType>::kFlag == type_flag_)
            << "TBlob.get_with_shape: data type do not match specified type."
            << "Expected: " << type_flag_ << " v.s. given "
            << mshadow::DataType<DType>::kFlag;
#if MKL_EXPERIMENTAL == 1
        if (Mkl_mem_ != nullptr) {
            Mkl_mem_->check_and_prv_to_cpu(dptr_);
        }
#endif
        return mshadow::Tensor<Device, 2, DType>(
            static_cast<DType *>(dptr_), shape_.FlatTo2D(),
            shape_[shape_.ndim() - 1], stream);
    }
    /*!
     * \brief flatten the tensor to 1 dimension, collapse all the dimensions
     * together.
     * \param stream the possible stream target tensor should reside on
     * \tparam Device which device the tensor is on
     * \tparam DType the type of elements in the tensor
     * \return tensor after flatten
     */
    template <typename Device, typename DType>
    inline mshadow::Tensor<Device, 1, DType> FlatTo1D(
        mshadow::Stream<Device> *stream = NULL) const {
        return this->get_with_shape<Device, 1, DType>(
            mshadow::Shape1(shape_.Size()), stream);
    }
    /*! \brief return number of dimension of the tensor inside */
    inline int ndim(void) const { return shape_.ndim(); }
    /*!
     * \brief return size of i-th dimension, start counting from highest
     * dimension
     * \param idx the dimension count from the highest dimensin
     * \return the size
     */
    inline index_t size(index_t idx) const { return shape_[idx]; }
    /*! \brief total number of elements in the tensor */
    inline index_t Size(void) const { return shape_.Size(); }
    /*! \brief get pointer in dtype */
    template <typename DType>
    inline DType *dptr() const {
        CHECK(mshadow::DataType<DType>::kFlag == type_flag_)
            << "TBlob.get_with_shape: data type do not match specified type."
            << "Expected: " << type_flag_ << " v.s. given "
            << mshadow::DataType<DType>::kFlag;
#if MKL_EXPERIMENTAL == 1
        if (Mkl_mem_ != nullptr) {
            Mkl_mem_->check_and_prv_to_cpu(dptr_);
        }
#endif
        return static_cast<DType *>(dptr_);
    }
    /*! \brief device mask of the corresponding device */
    inline int dev_mask() const { return dltensor_.ctx.device_type; }
    /*! \brief device index of the corresponding device */
    inline int dev_id() const { return dltensor_.ctx.device_id; }
    /*!
     * \brief return the corresponding DLTensor
     * \return the address of internal DLTensor
     */
    inline const DLTensor &dltensor() { return dltensor_; }

    /*!
     * \brief fetch the tensor, with respect to specific dimension
     * if dim do not match the stored dimension, an error will be issued
     * \return the tensor requested
     * \param stream the possible stream target tensor should reside on
     * \tparam Device which device the tensor is on
     * \tparam dim dimension of the tensor
     * \tparam DType the type of elements in the tensor
     */
    template <typename Device, int dim, typename DType>
    inline mshadow::Tensor<Device, dim, DType> get(
        mshadow::Stream<Device> *stream = NULL) const {
        CHECK(Device::kDevMask == this->dev_mask())
            << "TBlob.get: device type do not match specified type";
        return mshadow::Tensor<Device, dim, DType>(
            dptr<DType>(), shape_.get<dim>(), shape_[shape_.ndim() - 1],
            stream);
    }
    /*!
     * \brief fetch a tensor in given shape
     *  If size do not match the stored size, an error will be issued
     * \return the tensor requested
     * \param shape the shape required
     * \param stream the possible stream target tensor should reside on
     * \tparam Device which device the tensor is on
     * \tparam dim dimension of the tensor
     * \tparam DType the type of elements in the tensor
     */
    template <typename Device, int dim, typename DType>
    inline mshadow::Tensor<Device, dim, DType> get_with_shape(
        const mshadow::Shape<dim> &shape,
        mshadow::Stream<Device> *stream = NULL) const {
        CHECK(Device::kDevMask == this->dev_mask())
            << "TBlob.get: device type do not match specified type";
        CHECK_EQ(this->CheckContiguous(), true)
            << "TBlob.get_reshape: must be contiguous";
        CHECK_EQ(this->shape_.Size(), shape.Size())
            << "TBlob.get_with_shape: new and old shape do not match total "
               "elements";
        return mshadow::Tensor<Device, dim, DType>(dptr<DType>(), shape,
                                                   shape[dim - 1], stream);
    }
    /*!
     * \brief flatten the tensor to 3 dimension,
     *  collapse the dimension before and after specified axis.
     * \param axis The axis specified.
     * \param stream the possible stream target tensor should reside on
     * \tparam Device which device the tensor is on
     * \tparam DType the type of elements in the tensor
     * \return tensor after flatten
     */
    template <typename Device, typename DType>
    inline mshadow::Tensor<Device, 3, DType> FlatTo3D(
        int axis, mshadow::Stream<Device> *stream = NULL) const {
        return this->get_with_shape<Device, 3, DType>(
            this->shape_.FlatTo3D(axis), stream);
    }
    /*!
     * \brief flatten the tensor to 3 dimension,
     *  collapse the dimension: [0, axis_begin), [axis_begin, axis_end],
     * (axis_end, ndim).
     * \param axis_begin The beginning axis specified.
     * \param axis_end The ending axis specified.
     * \param stream the possible stream target tensor should reside on
     * \tparam Device which device the tensor is on
     * \tparam DType the type of elements in the tensor
     * \return tensor after flatten
     */
    template <typename Device, typename DType>
    inline mshadow::Tensor<Device, 3, DType> FlatTo3D(
        int axis_begin, int axis_end,
        mshadow::Stream<Device> *stream = NULL) const {
        return this->get_with_shape<Device, 3, DType>(
            this->shape_.FlatTo3D(axis_begin, axis_end), stream);
    }

   private:
    static DLDataType DTypeTransform(int type_flag) {
        static std::unordered_map<int, DLDataType>
            MSHADOW_DTYPE_TO_DLPACK_DTYPE = {
                {0, {2, 32, 1}},  // Float32
                {1, {2, 64, 1}},  // Float64
                {2, {2, 16, 1}},  // Float16
                {3, {1, 8, 1}},   // UInt8
                {4, {0, 32, 1}},  // Int32
                {5, {0, 8, 1}}    // Int8
            };
        return MSHADOW_DTYPE_TO_DLPACK_DTYPE[type_flag];
    }

    inline void SetDLTensor(int dev_mask, int dev_id) {
        dltensor_.data = dptr_;
        dltensor_.ctx = DLContext{static_cast<DLDeviceType>(dev_mask), dev_id};
        dltensor_.ndim = shape_.ndim();
        dltensor_.dtype = DTypeTransform(type_flag_);
        dltensor_.shape = shape_.data();
        dltensor_.strides = NULL;
        dltensor_.byte_offset = 0;
    }

   private:
    /*! \brief corresponding DLTensor of this TBlob */
    DLTensor dltensor_;
};
}  // namespace mxnet

namespace dmlc {
// Add a few patches to support TShape in dmlc/parameter.
DMLC_DECLARE_TYPE_NAME(mxnet::TShape, "Shape(tuple)");
DMLC_DECLARE_TYPE_NAME(nnvm::Tuple<int>, "Shape(tuple)");
DMLC_DECLARE_TYPE_NAME(nnvm::Tuple<dmlc::optional<int>>, "Shape(tuple)");

namespace parameter {

template <>
class FieldEntry<mxnet::TShape>
    : public FieldEntryBase<FieldEntry<mxnet::TShape>, mxnet::TShape> {
   public:
    FieldEntry() : enforce_nonzero_(false), expect_ndim_(0) {}
    // parent class
    typedef FieldEntryBase<FieldEntry<mxnet::TShape>, mxnet::TShape> Parent;

    virtual void Check(void *head) const {
        Parent::Check(head);
        mxnet::TShape &v = this->Get(head);
        if (expect_ndim_ != 0 && v.ndim() != expect_ndim_) {
            std::ostringstream os;
            os << "value " << v << "for Parameter " << this->key_
               << " has wrong dimensions, expected dimension=" << expect_ndim_;
            throw dmlc::ParamError(os.str());
        }
        if (enforce_nonzero_) {
            for (mxnet::index_t i = 0; i < v.ndim(); ++i) {
                if (v[i] == 0U) {
                    std::ostringstream os;
                    os << "value " << v << "for Parameter " << this->key_
                       << " is invalid, the input shape must be nonzero in all "
                          "dimensions";
                    throw dmlc::ParamError(os.str());
                }
            }
        }
    }
    inline FieldEntry<mxnet::TShape> &enforce_nonzero() {
        this->enforce_nonzero_ = true;
        return this->self();
    }
    inline FieldEntry<mxnet::TShape> &set_expect_ndim(mxnet::index_t ndim) {
        expect_ndim_ = ndim;
        return this->self();
    }

   private:
    // whether all the entries need to be nonzero
    bool enforce_nonzero_;
    // expected number of dimension, default = 0 means no restriction.
    mxnet::index_t expect_ndim_;
};

}  // namespace parameter
}  // namespace dmlc

#endif  // MXNET_TENSOR_BLOB_H_
