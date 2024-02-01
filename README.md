running on 101 machine
git pull && sudo docker stop worker-0 | true && sudo docker rm worker-0 | true && sudo docker rmi --force my_tensorflow_app && sudo docker build -t my_tensorflow_app . && sudo docker run -a stdout -a stderr --gpus all -p 2222:2222 -e TF_CONFIG='{"cluster": {"worker": ["192.168.1.101:2222", "192.168.1.102:2222"]}, "task": {"type": "worker", "index": 0}}' --name worker-0  my_tensorflow_app


Running on 102 machine
git pull && sudo docker stop worker-1 | true && sudo docker rm worker-1 | true && sudo docker rmi --force my_tensorflow_app && sudo docker build -t my_tensorflow_app . && sudo docker run --gpus all -p 2222:2222 -e TF_CONFIG='{"cluster": {"worker": ["192.168.1.101:2222", "192.168.1.102:2222"]}, "task": {"type": "worker", "index": 1}}' --name worker-1  my_tensorflow_app
