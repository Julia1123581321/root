//#include <gtest/gtest.h>

#include <tensorflow/c/c_api.h>
#include "tensorflowevaluator.h"
#include "tensorflowevaluator.cxx"

//#include <ROOT/TestSupport.hxx>
#include "ROOT/RDataFrame.hxx"
#include "ROOT/RVec.hxx"
#include "ROOT/RDFHelpers.hxx"
#include "TRandom.h"

using namespace ROOT::VecOps;
using cRVecF = const ROOT::RVecF &;


int main() {


  // This could be a starting point: tests/test_tensorflowceval_nn.py? --> use nn1.pb file
  // Anyways, to make the evaluator work, we need three things:
  // A model, an input and an output (names)
  // --> C test from python test

  const std::string& modelfile = "nn1.pb";
  const std::vector<std::string>&  input = {"dense_1_input"};
  const std::vector<std::string>&  output = {"dense_3/Softmax"};
  TensorflowCEvaluator tfc_nn1(modelfile, input, output); 

/*
  // Test: std::vector
  std::vector<float> v_in = {.7, .7, .7, .7, .7};
  std::vector<float> v_out = tfc_nn1.evaluate(v_in);
  std::cout << v_out[0] << "\n";
  std::cout << v_out[1] << "\n";
  std::cout << std::accumulate(v_out.begin(), v_out.end(), 0.0f) << "\n";
  //EXPECT_FLOAT_EQ(std::accumulate(v_out.begin(), v_out.end(), 0.0f), 1.0f);

  // Next, we create a dataframe with an input vector
  ROOT::RDataFrame d(1);
  auto i = 0.;
  auto df = d.Define("x", [&i, &v_in]() {return v_in; });
  std::cout << df.Display("")->AsString() << "\n";
  std::cout << df.GetColumnType("x") << "\n";

  auto d1 = df.Define("y", [&tfc_nn1](const std::vector<float> x){
    // The input for the model needs to be a vector
    return tfc_nn1.evaluate(x);}, {"x"});
  std::cout << d1.Display("")->AsString() << "\n";

  // Does it also work with RVecs?
  cRVecF v_in = {.7, .7, .7, .7, .7};
  //cRVecF v_in = {.7, .7, .7, .7}; //, .7}; Here, the input vector must have 5 components "nodes"
  auto v_out = tfc_nn1.evaluate(v_in);
  std::cout << std::accumulate(v_out.begin(), v_out.end(), 0.0f) << "\n";
*/
  ///Make it work with two vectors:

  //Now, we load a dataframe with given columns:
  ROOT::RDataFrame d_data("mini;1", "/home/jmathe/root_again/root_src/tutorials/dataframe/vary_tutorial/data_A.4lep.root");

  //std::cout << d_data.Describe() << "\n";
  // --> Found float to test on
  //std::cout << d_data.Display("lep_pt")->AsString() << "\n";

  //Working example:
  auto d1_data = d_data.Define("z", [&tfc_nn1](cRVecF lep_pt){
      std::cout << lep_pt << "\n";
      // Stupidly enough, the input vectors need five components.
      // Let's copy and add a zero.
      auto lep_pt_eval = lep_pt;
      lep_pt_eval.push_back(0.0f);
      return tfc_nn1.evaluate(lep_pt_eval);
    }, {"lep_pt"});
  std::cout << d1_data.Display("z")->AsString() << "\n";

  //auto d1_data = d_data.Define("z", EvaluateModel, {"lep_pt"});

  return 0;
}

/* // This is what it should look like:
auto myTensorflowEvaluator = TensorflowCEvaluator("mymodel.tf", ...);
auto df = ROOT::RDataFrame(10).Define("a",...).Define("b",...)...;
df.Define("nn_output",
          [&myTensorflowEvaluator](double a, double b, double c) { return myTensorflowEvaluator.evaluate(a, b, c); },
          {"input1", "input2", "input3"});
df.Display().Print();
*/