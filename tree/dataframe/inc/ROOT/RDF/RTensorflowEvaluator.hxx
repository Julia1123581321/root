// Author: Julia Mathe, CERN  08/2022

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RTENSORFLOWEVALUATOR
#define ROOT_RTENSORFLOWEVALUATOR

#pragma once

#include <string>
#include <vector>
#include <stdexcept>

#include "tensorflow/c/c_api.h"

namespace ROOT {
namespace RDF {
namespace Experimental {

class TensorflowCEvaluator {
public:
  using output_t = std::vector<float>;

  TensorflowCEvaluator(const std::string& modelFile,
      const std::vector<std::string>& inputNames,
      const std::vector<std::string>& outputNames);
  ~TensorflowCEvaluator();

  template<class... Types>
  output_t evaluate(Types&&... inputsPerNode) const {
    InputTensors inputValues;
    inputValues.reserve(m_inputs.size());
    inputValues.load(m_inputInfo, std::forward<Types>(inputsPerNode)...);
    if ( inputValues.size() != m_inputInfo.size() ) {
      throw std::runtime_error("Not enough inputs, model has "+std::to_string(m_inputInfo.size())+" but passed only "+std::to_string(inputValues.size()));
    }
    return evaluate_tensors(inputValues);
  }

private:
  TF_Graph* m_graph;
  TF_Session* m_session;
  TF_SessionOptions* m_sessionOpts;
  std::vector<TF_Output> m_inputs;
  std::vector<TF_Output> m_outputs;

  struct InputTensorInfo {
    std::vector<int64_t> shape;
    std::size_t size;
    TF_DataType dataType;
    InputTensorInfo(const std::vector<int64_t>& shape_, std::size_t size_, TF_DataType dataType_)
      : shape(shape_), size(size_), dataType(dataType_) {}
  };
  std::vector<InputTensorInfo> m_inputInfo;

  class InputTensors {
  public:
    InputTensors() = default;
    ~InputTensors() { for ( auto t : m_tensors ) { TF_DeleteTensor(t); } }
    void reserve(std::size_t n) { m_tensors.reserve(n); }
    TF_Tensor* const* data() const { return m_tensors.data(); }
    std::size_t size() const { return m_tensors.size(); }
  private:
    std::vector<TF_Tensor*> m_tensors;
  public:
    template<typename NodeInputs, typename ...NextInputs>
    void load(const std::vector<InputTensorInfo>& inputInfo, NodeInputs&& nodeInputs, NextInputs&&... nextInputs) {
      load(inputInfo, std::forward<NodeInputs>(nodeInputs));
      load(inputInfo, std::forward<NextInputs>(nextInputs)...);
    }
    template<typename NodeInputs>
    void load(const std::vector<InputTensorInfo>& inputInfo, NodeInputs&& nodeInputs) {
      if ( m_tensors.size() == inputInfo.size() ) {
        throw std::runtime_error("Too many inputs, model has only "+std::to_string(inputInfo.size()));
      }
      const auto& nodeInfo = inputInfo[m_tensors.size()];
      if ( nodeInfo.size != nodeInputs.size() ) {
        throw std::runtime_error("Incorrect number of inputs for node "+
            std::to_string(m_tensors.size())+
            " : expected "+std::to_string(nodeInfo.size)+
            " values, but received "+std::to_string(nodeInputs.size()));
      }
      auto inTensor = TF_AllocateTensor(nodeInfo.dataType, nodeInfo.shape.data(), nodeInfo.shape.size(), nodeInfo.size*TF_DataTypeSize(nodeInfo.dataType));
      m_tensors.push_back(inTensor);
      switch (nodeInfo.dataType) {
        case TF_FLOAT:
          std::copy(std::begin(nodeInputs), std::end(nodeInputs),
              static_cast<float*>(TF_TensorData(inTensor)));
          break;
        case TF_DOUBLE:
          std::copy(std::begin(nodeInputs), std::end(nodeInputs),
              static_cast<double*>(TF_TensorData(inTensor)));
          break;
        case TF_INT32:
          std::copy(std::begin(nodeInputs), std::end(nodeInputs),
              static_cast<std::int32_t*>(TF_TensorData(inTensor)));
          break;
        case TF_UINT8:
          std::copy(std::begin(nodeInputs), std::end(nodeInputs),
              static_cast<std::uint8_t*>(TF_TensorData(inTensor)));
          break;
        case TF_INT16:
          std::copy(std::begin(nodeInputs), std::end(nodeInputs),
              static_cast<std::int16_t*>(TF_TensorData(inTensor)));
          break;
        case TF_INT8:
          std::copy(std::begin(nodeInputs), std::end(nodeInputs),
              static_cast<std::int8_t*>(TF_TensorData(inTensor)));
          break;
        // skipped STRING (7), COMPLEX64 (8)
        case TF_INT64:
          std::copy(std::begin(nodeInputs), std::end(nodeInputs),
              static_cast<std::int64_t*>(TF_TensorData(inTensor)));
        // skipped more
        default:
          break;
      }
    }
  };

  output_t evaluate_tensors(InputTensors& inputValues) const;
};

} // end of NS Experimental
} // end of NS RDF
} // end of NS ROOT

#endif
