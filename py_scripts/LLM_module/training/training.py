from . import read_config as rc
from . import moe_training as moe

def train():
    if rc.name == "MOE":
        moe.main_train(
            rc.numbers_expert, 
            rc.numbers_agent, 
            rc.function_calling, 
            rc.device_type, 
            rc.device_training_type, 
            rc.device_number_to_use, 
            rc.device_index,
            rc.batch_size, 
            rc.learning_rate, 
            rc.epoch, 
            rc.top_k, 
            rc.expert_capacity,
            rc.model_type, 
            rc.model_path, 
            rc.input_dim, 
            rc.output_dim, 
            rc.hidden_dim, 
            rc.data_type, 
            rc.data_path,
            rc.log_path, 
            rc.log_level)
    elif rc.name == "PPO":
        ...
    else:
        # raise ValueError("Unsupport training with the AI architecture.")
        return "Unsupport training with the AI architecture."

