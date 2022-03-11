```
docker pull "odsai/vtb22-data-fusion:0.0.1"

# docker run -v /Users/a17537911/py_projects/vtb/nn_distance:/workspace \
#     -it "odsai/vtb22-data-fusion:0.0.1" \
#     python -u nn_distance_inference.py

docker build -t ivkireev86/data_fusion2022:0.0.2 .

# docker image ls

docker login

docker image push ivkireev86/data_fusion2022:0.0.2


# docker run -v /Users/a17537911/py_projects/vtb/nn_distance:/workspace \
#     -it "ivkireev86/data_fusion2022:0.0.2" \
#     python -u nn_distance_inference.py

```
