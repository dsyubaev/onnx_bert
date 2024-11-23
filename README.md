Репо к статье про ONNX и go

Скрипт для генерации бинарника модели в папку ./data
```bash
conda create -n py38_test python=3.8
conda activate py38_test
pip install 'optimum[exporters]'
python pyscripts/run_export.py 
```

Билдем приложение на го и монтинруем папку с моделью
```
```bash
docker build -t goonnx .
docker run -it --rm -v ./data/:/data goonnx
```
