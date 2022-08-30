//#include <gtest/gtest.h>
#include <tensorflow/c/c_api.h>
#include "TFCEval.hxx"
#include "TFCEval.cxx"

//#include <ROOT/TestSupport.hxx>
#include "ROOT/RDataFrame.hxx"
#include "ROOT/RVec.hxx"
#include "ROOT/RDFHelpers.hxx"
#include "TRandom.h"

using namespace ROOT::VecOps;
using cRVecF = const ROOT::RVecF &;

int main() {

    // Test the TensorflowCEvaluator with an easy XOR model
    const std::string& modelfile = "frozen_graph.pb";
    const std::vector<std::string>&  input = {"x"};
    const std::vector<std::string>&  output = {"Identity"};
    TensorflowCEvaluator model (modelfile, input, output); 

    std::vector<std::vector<int>> v_in{{0, 0}, {0, 1}, {1, 0}, {1, 1}}; 
    std::vector<int> v_check{0, 1, 1, 0};
    //int i = 0; int k = 0;

    auto d = ROOT::RDataFrame(4)
        .Define("x", [&v_in](ULong64_t e) { return v_in[e]; }, {"rdfentry_"})
        .Define("output", [&model, &v_in](ULong64_t e){
            auto output = model.evaluate(v_in[e]);
            int y = std::round(output[0]);
            return y;}, {"rdfentry_"})
        .Define("control", [&v_check](ULong64_t e) { return v_check[e]; }, {"rdfentry_"});
    //EXPECT_EQ(d.Filter("output != control").Count().GetValue(), 0);
    
    if (d.Filter("output != control").Count().GetValue() == 0){
        std::cout << "Passed" << "\n";
    }
    d.Display({"x", "output", "control"})->Print();  
}
