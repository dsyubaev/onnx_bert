package main

import (
	"fmt"
	"math"

	"github.com/daulet/tokenizers"
	ort "github.com/yalue/onnxruntime_go"
)

func makeTensor(x []uint32) *ort.Tensor[int64] {

	inputShape := ort.NewShape(1, int64(len(x)))
	inputData := make([]int64, len(x))
	for i, v := range x {
		inputData[i] = int64(v)
	}

	inputTensor, err := ort.NewTensor(inputShape, inputData)
	if err != nil {
		panic(err)
	}
	return inputTensor
}

func SoftMax(array []float32) ([]float64, error) {

	input := make([]float64, len(array))
	for i, x := range array {
		input[i] = float64(x)
	}

	s := 0.0
	c := math.Inf(-1)
	for _, e := range input {
		c = math.Max(e, c)
	}
	for _, e := range input {
		s += math.Exp(e - c)
	}

	sm := make([]float64, len(input))
	for i, v := range input {
		sm[i] = math.Exp(v-c) / s
	}

	return sm, nil
}

func main() {

	text := "Привет, ты мне нравишься!"
	tk, err := tokenizers.FromFile("/data/rubert-tiny2-russian-sentiment-onnx/tokenizer.json")
	if err != nil {
		panic(err)
	}
	defer tk.Close()

	encodeOptions := []tokenizers.EncodeOption{
		tokenizers.WithReturnTypeIDs(),
		tokenizers.WithReturnAttentionMask(),
	}
	encodingResponse := tk.EncodeWithOptions(text, true, encodeOptions...)

	fmt.Printf("IDs=%v\n", encodingResponse.IDs)
	fmt.Printf("AttentionMask=%v\n", encodingResponse.AttentionMask)
	fmt.Printf("TypeIDs=%v\n", encodingResponse.TypeIDs)

	ort.SetSharedLibraryPath("/usr/lib/libonnxruntime.so")

	err = ort.InitializeEnvironment()
	if err != nil {
		panic(err)
	}
	defer ort.DestroyEnvironment()

	// input, output, err := ort.GetInputOutputInfo("/data/rubert-tiny2-russian-sentiment-onnx/model.onnx")
	// fmt.Println(input)
	// fmt.Println(output)

	session, err := ort.NewDynamicAdvancedSession(
		"/data/rubert-tiny2-russian-sentiment-onnx/model.onnx",
		[]string{"input_ids", "token_type_ids", "attention_mask"},
		[]string{"logits"},
		nil,
	)
	if err != nil {
		panic(err)
	}
	defer session.Destroy()

	outputTensor, err := ort.NewEmptyTensor[float32](ort.NewShape(1, 3))
	defer outputTensor.Destroy()
	if err != nil {
		panic(err)
	}

	err = session.Run([]ort.Value{
		makeTensor(encodingResponse.IDs),
		makeTensor(encodingResponse.TypeIDs),
		makeTensor(encodingResponse.AttentionMask),
	}, []ort.Value{outputTensor})
	if err != nil {
		panic(err)
	}

	outputData := outputTensor.GetData()
	fmt.Println(outputData)
	fmt.Println(SoftMax(outputData))

}
