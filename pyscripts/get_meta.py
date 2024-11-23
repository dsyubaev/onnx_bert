from onnx import load, shape_inference

model = load("./data/rubert-tiny2-russian-sentiment-onnx/model.onnx")

inferred_model = shape_inference.infer_shapes(model)
print(inferred_model.graph.output)

output = [node.name for node in model.graph.output]

input_all = [node.name for node in model.graph.input]
input_initializer = [node.name for node in model.graph.initializer]
net_feed_input = list(set(input_all) - set(input_initializer))

print("Inputs: ", net_feed_input)
print("Outputs: ", output)
print(model.graph.output[0])
"""
name: "logits"
type {
  tensor_type {
    elem_type: 1
    shape {
      dim {
        dim_param: "batch_size"
      }
      dim {
        dim_value: 3
      }
    }
  }
}
"""
