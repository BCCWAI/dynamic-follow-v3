Here's how you can convert your TensorFlow graph to a DLC model file:

`(venv) shane@shane-VirtualBox:~/snpe-1.19.2/bin/x86_64-linux-clang$ python snpe-tensorflow-to-dlc --graph /home/shane/models/live_tracksv6.pb --input_dim dense_1_input "1,55" --out_node "dense_7/BiasAdd" --dlc /home/shane/models/live_tracksv6.dlc --allow_unconsumed_nodes`
