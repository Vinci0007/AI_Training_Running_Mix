import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except ImportError:
    torch_npu = None
    transfer_to_npu = None


class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        return self.net(x)
    
class MoE(nn.Module):
    def __init__(self, input_dim, num_experts, top_k, expert_capacity, hidden_dim, output_dim):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_capacity = expert_capacity
        
        # 路由网络
        self.gate = nn.Linear(input_dim, num_experts)
        
        # 专家集合
        self.experts = nn.ModuleList(
            [Expert(input_dim, hidden_dim, output_dim) for _ in range(num_experts)])
        
    def forward(self, x):
        batch_size, input_dim = x.shape
        device = x.device
        
        # 路由计算
        logits = self.gate(x)
        probs = torch.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, self.top_k, dim=-1)
        
        # 辅助损失计算
        if self.training:
            # 重要性损失（专家利用率均衡）
            importance = probs.sum(0)
            importance_loss = torch.var(importance) / (self.num_experts ** 2)
            
            # 负载均衡损失（样本分配均衡）
            mask = torch.zeros_like(probs, dtype=torch.bool)
            mask.scatter_(1, topk_indices, True)
            routing_probs = probs * mask
            expert_usage = mask.float().mean(0)
            routing_weights = routing_probs.mean(0)
            load_balance_loss = self.num_experts * (expert_usage * routing_weights).sum()
            
            aux_loss = importance_loss + load_balance_loss
        else:
            aux_loss = 0.0

        # 专家分配逻辑
        flat_indices = topk_indices.view(-1)
        flat_probs = topk_probs.view(-1)
        sample_indices = torch.arange(batch_size, device=device)[:, None]\
                            .expand(-1, self.top_k).flatten()

        # 初始化输出
        outputs = torch.zeros(batch_size, self.experts[0].net[-1].out_features, 
                            device=device)

        # 处理每个专家
        for expert_idx in range(self.num_experts):
            # 获取分配给当前专家的样本
            expert_mask = flat_indices == expert_idx
            expert_samples = sample_indices[expert_mask]
            expert_weights = flat_probs[expert_mask]

            # 容量控制
            if len(expert_samples) > self.expert_capacity:
                expert_samples = expert_samples[:self.expert_capacity]
                expert_weights = expert_weights[:self.expert_capacity]

            if len(expert_samples) == 0:
                continue

            # 处理专家计算
            expert_input = x[expert_samples]
            expert_output = self.experts[expert_idx](expert_input)
            weighted_output = expert_output * expert_weights.unsqueeze(-1)
            
            # 累加输出
            outputs.index_add_(0, expert_samples, weighted_output)

        return outputs, aux_loss
    
def generate_simulation_data(batch_size, input_dim):
    # 生成高斯分布的输入数据
    data = torch.randn(batch_size, input_dim)
    # 生成随机标签
    labels = torch.randn(batch_size, input_dim)
    return data, labels

class MoEEP(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts, hidden_dim, top_k=2,
                 capacity_factor=1.0, dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        
        # 专家网络分布在不同设备
        self.experts = nn.ModuleList([
            Expert(input_dim, hidden_dim, output_dim).to(f'cuda:{i}') 
            for i in range(num_experts)
        ])
        
        # 路由器
        self.router = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        batch_size = x.size(0)
        orig_shape = x.shape
        x = x.view(-1, orig_shape[-1])
        
        # 路由计算
        logits = self.router(x)
        probs = F.softmax(logits, dim=-1)
        expert_weights, expert_indices = torch.topk(probs, self.top_k, dim=-1)
        
        # 分布式计算设置
        world_size = dist.get_world_size()
        capacity = int(self.capacity_factor * batch_size / (self.top_k * world_size))
        capacity = max(capacity, 1)
        
        # 跨设备通信
        expert_mask = F.one_hot(expert_indices, self.num_experts).sum(dim=1)
        expert_counts = expert_mask.sum(dim=0)
        dist.all_reduce(expert_counts, op=dist.ReduceOp.SUM)
        
        # 分布式负载均衡损失
        density = probs.mean(dim=0)
        usage = expert_counts / (batch_size * world_size)
        balance_loss = (density * usage).sum() * self.num_experts
        
        # 分布式专家计算
        outputs = []
        for expert_id in range(self.num_experts):
            # 获取该专家对应的设备
            device = f'cuda:{expert_id % torch.cuda.device_count()}'
            print(f"Current device:{device}")

            # 获取该专家对应的样本
            idx_mask = (expert_indices == expert_id).any(dim=-1)
            if idx_mask.sum() == 0:
                continue
                
            # 容量截断
            selected = torch.nonzero(idx_mask).flatten()
            print("selected: ", selected)
            if selected.numel() == 0:
                continue

            selected = selected[:capacity]
            if selected.numel() == 0:
                continue  # 如果容量限制后仍然为空，跳过当前循环

            # 跨设备传输
            expert_input = x[selected].to(device)
            expert_output = self.experts[expert_id](expert_input)
            
            # 加权输出传回原设备
            weights = expert_weights[selected, (expert_indices[selected] == expert_id).nonzero()[:,1]]
            weighted_output = (expert_output * weights.unsqueeze(-1)).to(x.device)
            outputs.append((selected, weighted_output))
        
        # 合并结果
        final_output = torch.zeros_like(x)
        for selected, out in outputs:
            final_output[selected] += out
            
        # 重要性损失
        importance = probs.sum(dim=0)
        dist.all_reduce(importance, op=dist.ReduceOp.SUM)
        importance_loss = (importance ** 2).mean()
        aux_loss = balance_loss + importance_loss
        
        return final_output.view(*orig_shape), aux_loss
    
class StandaloneExpert(nn.Module):
    def __init__(self, expert):
        super().__init__()
        self.expert = expert  # 直接复用原专家结构
        
    def forward(self, x):
        return self.expert(x)  # 无需门控逻辑

def single_train(
        device_type,
        device_index,
        input_dim, 
        num_experts, 
        top_k, 
        epoch,
        expert_capacity, 
        hidden_dim, 
        output_dim,
        batch_size):
    torch_use = None
    device = None
    export_type = None
    activities = []
    moe_result_path = ""
    experimental_config = None
    if device_type.lower() == 'npu':
        torch_use = torch_npu
        if torch_use is None:
            print ("torch_npu not found or python version is not supported, please install it first.")
            return 
        device = torch.device(f"npu:{device_index}" if torch.npu.is_available() else "cpu")
        activities=[
                torch_use.profiler.ProfilerActivity.CPU,
                torch_use.profiler.ProfilerActivity.NPU
            ]
        moe_result_path = "./result/ai_model/moe_stand_npu_result"
        experimental_config = torch_npu.profiler._ExperimentalConfig(
            export_type=torch_npu.profiler.ExportType.Text,
            profiler_level=torch_npu.profiler.ProfilerLevel.Level0,
            msprof_tx=False,
            aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
            l2_cache=False,
            op_attr=False,
            data_simplification=False,
            record_op_args=False,
            gc_detect_threshold=None
        )
    elif device_type.lower() == 'gpu':
        torch_use = torch 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        activities=[
                torch_use.profiler.ProfilerActivity.CPU,
                torch_use.profiler.ProfilerActivity.CUDA
            ]
        moe_result_path = "./result/ai_model/moe_stand_gpu_result"
        experimental_config = torch_use.profiler._ExperimentalConfig(
            profiler_metrics=[],
            profiler_measure_per_kernel=True,
            verbose=True,
            performance_events=[],
            enable_cuda_sync_events=True)
        # export_type=torch_use.profiler.export_chrome_trace(f"{moe_result_path}/moe_stand_gpu_result.json", prof)
    else:
        ...
    # device_index = f"npu:{device_index}"
    moe = MoE(
        input_dim, 
        num_experts, 
        top_k, 
        expert_capacity, 
        hidden_dim, 
        output_dim).to(device)
    x = torch.randn(batch_size, input_dim).to(device)

    with torch_use.profiler.profile(
            activities=activities,
            schedule=torch_use.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=1),
            on_trace_ready=torch_use.profiler.tensorboard_trace_handler(moe_result_path),
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            with_flops=False,
            with_modules=False,
            experimental_config=experimental_config,
            execution_trace_observer=None,
            acc_events=False) as prof:
            # # deprecated:
            # use_cuda: Optional[bool] = None,
            # custom_trace_id_callback: Optional[Callable[[], str]] = None

        # 训练模式
        # for _ in range(10):
        for _ in range(epoch):
            moe.train()
            output, loss = moe(x)
            print(f"Using device: {x.device}")
            print(f"Training output shape: {output.shape}")      # torch.Size([64, 256])
            print(f"Training auxiliary loss: {loss.item():.4f}")     # 示例值，如0.1234
            prof.step()

    print("=" * 80)

    # 推理模式
    moe.eval()
    output, _ = moe(x)
    print(f"Eval output shape: {output.shape}")     # torch.Size([64, 256])

    standalone_expert = StandaloneExpert(moe.experts[1])
    torch.save(standalone_expert.state_dict(), "./result/ai_model/expert_1_standalone.pth")

    ...

def multi_train(
        device_type,
        device_index,
        device_number_to_use,
        input_dim, 
        num_experts, 
        top_k, 
        epoch,
        expert_capacity, 
        hidden_dim, 
        output_dim,
        batch_size):
    
    torch_use = None
    device = None
    activities = []
    moe_result_path = ""
    if device_type.lower() == 'npu':
        torch_use = torch_npu
        if torch_use is None:
            print ("torch_npu not found or python version is not supported, please install it first.")
            return 
        device = torch.device(f"npu:{device_index}" if torch.npu.is_available() else "cpu")
        activities=[
                torch_use.profiler.ProfilerActivity.CPU,
                torch_use.profiler.ProfilerActivity.NPU
            ]
        moe_result_path = "./result/ai_model/moe_nulti_npu_result"
    elif device_type.lower() == 'gpu':
        torch_use = torch 
        device = torch.device(f"gpu:{device_index}" if torch.cuda.is_available() else "cpu")
        activities=[
                torch_use.profiler.ProfilerActivity.CPU,
                torch_use.profiler.ProfilerActivity.CUDA
            ]
        moe_result_path = "./result/ai_model/moe_multi_gpu_result"
    else:
        ...

    # 初始化分布式训练
    def setup(rank, world_size):
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)

    # 训练循环
    def train(rank, world_size, batch_size, input_dim, output_dim, hidden_dim, top_k, num_experts):
        setup(rank, world_size)

        model = MoEEP(
            input_dim=input_dim,
            output_dim=output_dim,
            num_experts=num_experts,
            hidden_dim=hidden_dim,
            top_k=top_k,
            capacity_factor=1.2
        ).to(rank)
        
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

        # 数据加载器，生成模拟数据
        data, labels = generate_simulation_data(batch_size, input_dim)
        dataset = list(zip(torch.tensor(data), torch.tensor(labels)))

        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        loader = DataLoader(dataset, batch_size=32, sampler=sampler)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        experimental_config = torch_use.profiler._ExperimentalConfig(
            export_type=torch_use.profiler.ExportType.Text,
            profiler_level=torch_use.profiler.ProfilerLevel.Level0,
            msprof_tx=False,
            aic_metrics=torch_use.profiler.AiCMetrics.AiCoreNone,
            l2_cache=False,
            op_attr=False,
            data_simplification=False,
            record_op_args=False,
            gc_detect_threshold=None
        )

        with torch_use.profiler.profile(
                activities=activities,
                schedule=torch_use.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=1),
                on_trace_ready=torch_use.profiler.tensorboard_trace_handler("./result"),
                record_shapes=False,
                profile_memory=False,
                with_stack=False,
                with_modules=False,
                with_flops=False,
                experimental_config=experimental_config) as prof:
            for epoch in range(10):
                sampler.set_epoch(epoch)
                for x, y in loader:
                    x = x.to(rank)
                    y = y.to(rank)

                    outputs, aux_loss = model(x)
                    main_loss = F.mse_loss(outputs, y)
                    total_loss = main_loss + 0.01 * aux_loss

                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()
                    prof.step()
    # world_size = 8
    world_size = device_number_to_use
    torch.multiprocessing.spawn(train, args=(world_size, batch_size, input_dim, output_dim, hidden_dim, top_k, num_experts), nprocs=world_size)
    ...

def main_train(
        numbers_expert,
        numbers_agent,
        function_calling,
        device_type,
        device_training_type,
        device_number_to_use,
        device_index,
        batch_size,
        learning_rate,
        epoch,
        top_k,
        expert_capacity,
        model_type,
        model_path,
        input_dim,
        output_dim,
        hidden_dim,
        data_type,
        data_path,
        log_path,
        log_level
        ):


    # if device_type == "npu":
    #     ...

    if device_training_type == 'single':
        single_train(
            device_type,
            device_index,
            input_dim, 
            numbers_expert, 
            top_k, 
            epoch,
            expert_capacity, 
            hidden_dim, 
            output_dim,
            batch_size)
    elif device_training_type == 'multi':
        multi_train(
            device_type,
            device_index,
            device_number_to_use,
            input_dim, 
            numbers_expert, 
            top_k, 
            epoch,
            expert_capacity, 
            hidden_dim, 
            output_dim,
            batch_size)

    else:
        ...