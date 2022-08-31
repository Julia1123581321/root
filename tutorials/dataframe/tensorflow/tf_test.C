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

    // We must tell the TensorflowCEvaluator the input and output nodes
    // so that it can correctly read the model's graph
    const std::vector<std::string>&  input = {"x"};
    const std::vector<std::string>&  output = {"Identity"};

    // If you don't know the names of the input and output nodes of your model, 
    // use this code to read them from your pb model:
    /*
    from google.protobuf import text_format
    import tensorflow as tf
    import sys

    graph = tf.Graph()
    with graph.as_default():
        graph_def = graph.as_graph_def()

        with tf.io.gfile.GFile(sys.argv[1], "rb") as f:
            graph_def.ParseFromString(f.read())

        tf.import_graph_def(graph_def)

    print('\n'.join([op.name for op in graph.get_operations()]))
    */

    TensorflowCEvaluator model (modelfile, input, output); 

    // Make data to test the model
    std::vector<std::vector<int>> v_in{{0, 0}, {0, 1}, {1, 0}, {1, 1}}; 
    std::vector<int> v_check{0, 1, 1, 0}; // reference

    // Store the data into the columns of a dataframe and compute another
    // column with the model output 
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
