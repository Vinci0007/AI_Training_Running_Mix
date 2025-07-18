
name = "MOE"
training_type = "expert"
numbers_expert = 8
numbers_agent = 1
function_calling = True

device_type = "GPU"
device_training_type = "single"
device_number_to_use = 1
device_index = 0

batch_size = 128
learning_rate = 0.001
epoch = 100
top_k = 2
input_dim = 128
output_dim = 256
hidden_dim = 512
expert_capacity = 32

data_type = "MNIST"
data_path = "data/mnist"

model_type = "CNN"
model_path = "models/mnist_cnn.pth"

log_path = "logs"
log_level = "INFO"

# Below is the code 
# This is a test config file for testing the config module.

