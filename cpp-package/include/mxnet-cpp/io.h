/*!
*  Copyright (c) 2016 by Contributors
* \file operator.h
* \brief definition of io, such as DataIter
* \author Zhang Chen
*/
#ifndef CPP_PACKAGE_INCLUDE_MXNET_CPP_IO_H_
#define CPP_PACKAGE_INCLUDE_MXNET_CPP_IO_H_

#include <map>
#include <sstream>
#include <string>
#include <vector>
#include "dmlc/logging.h"
#include "mxnet-cpp/base.h"
#include "mxnet-cpp/ndarray.h"

namespace mxnet {
namespace cpp {
/*!
* \brief Default object for holding a mini-batch of data and related
* information.
*/
class DataBatch {
   public:
    NDArray data;
    NDArray label;
    int pad_num;
    std::vector<int> index;
};
class DataIter {
   public:
    virtual void BeforeFirst(void) = 0;
    virtual bool Next(void) = 0;
    virtual NDArray GetData(void) = 0;
    virtual NDArray GetLabel(void) = 0;
    virtual int GetPadNum(void) = 0;
    virtual std::vector<int> GetIndex(void) = 0;

    DataBatch GetDataBatch() {
        return DataBatch{GetData(), GetLabel(), GetPadNum(), GetIndex()};
    }
    void Reset() { BeforeFirst(); }
};

class MXDataIterMap {
   public:
    inline MXDataIterMap() {
        mx_uint num_data_iter_creators = 0;
        DataIterCreator *data_iter_creators = nullptr;
        int r = MXListDataIters(&num_data_iter_creators, &data_iter_creators);
        CHECK_EQ(r, 0);
        for (mx_uint i = 0; i < num_data_iter_creators; i++) {
            const char *name;
            const char *description;
            mx_uint num_args;
            const char **arg_names;
            const char **arg_type_infos;
            const char **arg_descriptions;
            r = MXDataIterGetIterInfo(data_iter_creators[i], &name,
                                      &description, &num_args, &arg_names,
                                      &arg_type_infos, &arg_descriptions);
            CHECK_EQ(r, 0);
            mxdataiter_creators_[name] = data_iter_creators[i];
        }
    }
    inline DataIterCreator GetMXDataIterCreator(const std::string &name) {
        return mxdataiter_creators_[name];
    }

   private:
    std::map<std::string, DataIterCreator> mxdataiter_creators_;
};

struct MXDataIterBlob {
   public:
    MXDataIterBlob() : handle_(nullptr) {}
    explicit MXDataIterBlob(DataIterHandle handle) : handle_(handle) {}
    ~MXDataIterBlob() { MXDataIterFree(handle_); }
    DataIterHandle handle_;

   private:
    MXDataIterBlob &operator=(const MXDataIterBlob &);
};

class MXDataIter : public DataIter {
   public:
    explicit MXDataIter(const std::string &mxdataiter_type);
    MXDataIter(const MXDataIter &other) {
        creator_ = other.creator_;
        params_ = other.params_;
        blob_ptr_ = other.blob_ptr_;
    }
    void BeforeFirst();
    bool Next();
    NDArray GetData();
    NDArray GetLabel();
    int GetPadNum();
    std::vector<int> GetIndex();
    MXDataIter CreateDataIter();
    /*!
     * \brief set config parameters
     * \param name name of the config parameter
     * \param value value of the config parameter
     * \return reference of self
     */
    template <typename T>
    MXDataIter &SetParam(const std::string &name, const T &value) {
        std::string value_str;
        std::stringstream ss;
        ss << value;
        ss >> value_str;

        params_[name] = value_str;
        return *this;
    }

   private:
    DataIterCreator creator_;
    std::map<std::string, std::string> params_;
    std::shared_ptr<MXDataIterBlob> blob_ptr_;
    static MXDataIterMap *&mxdataiter_map();
};
}  // namespace cpp
}  // namespace mxnet

#endif  // CPP_PACKAGE_INCLUDE_MXNET_CPP_IO_H_
