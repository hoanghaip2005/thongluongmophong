"""
Mô phỏng Frame Slotted ALOHA (FSA) với cơ chế Binary Exponential Backoff (BEB)
trong điều kiện không bão hòa.

Đặc điểm:
- Các gói tin mới đến theo phân phối Poisson với tốc độ lambda
- Sử dụng cơ chế Binary Exponential Backoff (BEB) cho các gói bị xung đột
- Giới hạn số lần truyền lại để tránh trễ vô hạn
- Theo dõi và vẽ đồ thị thông lượng và tỷ lệ rớt gói
"""

import numpy as np
import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm

class FSA_BEB_Simulator:
    def __init__(self, num_nodes, frame_length, num_frames, cw_exp_max=10, max_retries=15):
        """
        Khởi tạo bộ mô phỏng FSA với BEB.

        Parameters:
        -----------
        num_nodes : int
            Số lượng node trong mạng
        frame_length : int
            Số khe thời gian trong một khung
        num_frames : int
            Số khung mô phỏng
        cw_exp_max : int
            Số mũ tối đa cho cửa sổ cạnh tranh (CW = 2^exp - 1)
        max_retries : int
            Số lần thử lại tối đa cho một gói tin
        """
        self.N = num_nodes
        self.L = frame_length
        self.num_frames = num_frames
        self.CW_EXP_MAX = cw_exp_max
        self.MAX_RETRIES = max_retries
        
        # Khởi tạo trạng thái các node
        self.reset_nodes()

    def reset_nodes(self):
        """Khởi tạo lại trạng thái của tất cả các node."""
        self.nodes = [{
            'buffer': 0,              # Số gói tin trong buffer
            'collision_count': 0,     # Số lần xung đột của gói tin hiện tại
            'backoff_counter': 0      # Bộ đếm thời gian backoff còn lại
        } for _ in range(self.N)]

    def handle_new_arrivals(self, arrival_rate):
        """
        Xử lý các gói tin mới đến theo phân phối Poisson.
        
        Parameters:
        -----------
        arrival_rate : float
            Tốc độ đến trung bình của các gói tin mới (gói/khung)
            
        Returns:
        --------
        int : Số gói tin mới đến
        """
        new_packets = np.random.poisson(arrival_rate)
        if new_packets > 0:
            # Phân phối ngẫu nhiên các gói tin mới đến các node
            nodes_getting_packets = np.random.randint(0, self.N, size=new_packets)
            for node_idx in nodes_getting_packets:
                self.nodes[node_idx]['buffer'] += 1
        return new_packets

    def get_ready_nodes(self):
        """
        Xác định các node sẵn sàng truyền (có gói tin và không trong thời gian backoff).
        
        Returns:
        --------
        list : Danh sách chỉ số của các node sẵn sàng
        """
        return [i for i, node in enumerate(self.nodes)
                if node['buffer'] > 0 and node['backoff_counter'] == 0]

    def apply_beb(self, node_idx):
        """
        Áp dụng thuật toán Binary Exponential Backoff cho một node.
        
        Parameters:
        -----------
        node_idx : int
            Chỉ số của node cần áp dụng BEB
        """
        node = self.nodes[node_idx]
        exp = min(node['collision_count'], self.CW_EXP_MAX)
        window_size = (2**exp) - 1
        node['backoff_counter'] = np.random.randint(0, window_size + 1)

    def run_simulation(self, arrival_rate):
        """
        Chạy mô phỏng với một tốc độ đến cụ thể.
        
        Parameters:
        -----------
        arrival_rate : float
            Tốc độ đến trung bình của các gói tin mới
            
        Returns:
        --------
        tuple : (Thông lượng chuẩn hóa, Tỷ lệ rớt gói)
        """
        # Khởi tạo các biến đếm
        total_successful = 0
        total_dropped = 0
        total_new_packets = 0
        
        # Chạy mô phỏng qua từng khung
        for _ in range(self.num_frames):
            # Giảm bộ đếm backoff
            for node in self.nodes:
                if node['backoff_counter'] > 0:
                    node['backoff_counter'] -= 1
            
            # Xử lý gói tin mới đến
            total_new_packets += self.handle_new_arrivals(arrival_rate)
            
            # Xác định các node sẵn sàng truyền
            ready_nodes = self.get_ready_nodes()
            if not ready_nodes:
                continue
                
            # Mô phỏng quá trình truyền
            slot_usage = np.zeros(self.L, dtype=int)
            node_slots = {}  # Lưu khe thời gian được chọn bởi mỗi node
            
            # Các node chọn khe thời gian
            for node_idx in ready_nodes:
                chosen_slot = np.random.randint(0, self.L)
                slot_usage[chosen_slot] += 1
                node_slots[node_idx] = chosen_slot
            
            # Xử lý kết quả truyền
            for node_idx, slot in node_slots.items():
                if slot_usage[slot] == 1:  # Truyền thành công
                    self.nodes[node_idx]['buffer'] -= 1
                    self.nodes[node_idx]['collision_count'] = 0
                    total_successful += 1
                else:  # Xung đột xảy ra
                    self.nodes[node_idx]['collision_count'] += 1
                    if self.nodes[node_idx]['collision_count'] > self.MAX_RETRIES:
                        # Hủy gói tin
                        self.nodes[node_idx]['buffer'] -= 1
                        self.nodes[node_idx]['collision_count'] = 0
                        total_dropped += 1
                    else:
                        # Áp dụng BEB
                        self.apply_beb(node_idx)
        
        # Tính toán các chỉ số hiệu năng
        throughput = total_successful / (self.num_frames * self.L)
        drop_rate = total_dropped / total_new_packets if total_new_packets > 0 else 0
        
        return throughput, drop_rate

def run_experiment():
    """Chạy thực nghiệm và vẽ đồ thị kết quả."""
    # Tham số mô phỏng
    NUM_NODES = 50
    FRAME_LENGTH = 20
    NUM_FRAMES = 5000
    CW_EXP_MAX = 10
    MAX_RETRIES = 15
    
    # Dải giá trị tốc độ đến để khảo sát
    arrival_rates = np.linspace(0.1, FRAME_LENGTH * 2.5, 40)
    
    # Khởi tạo bộ mô phỏng
    simulator = FSA_BEB_Simulator(NUM_NODES, FRAME_LENGTH, NUM_FRAMES,
                                CW_EXP_MAX, MAX_RETRIES)
    
    # Mảng lưu kết quả
    throughputs = []
    drop_rates = []
    
    # Chạy mô phỏng
    print(f"Bắt đầu mô phỏng với N={NUM_NODES}, L={FRAME_LENGTH}")
    start_time = time()
    
    for rate in tqdm(arrival_rates, desc="Mô phỏng"):
        S, D = simulator.run_simulation(rate)
        throughputs.append(S)
        drop_rates.append(D)
    
    print(f"Hoàn thành mô phỏng sau {time() - start_time:.2f} giây")
    
    # Tính đường cong lý thuyết
    G = arrival_rates / FRAME_LENGTH  # Tải trọng cung cấp trên mỗi khe
    S_theory = G * np.exp(-G)        # Thông lượng lý thuyết
    
    # Vẽ đồ thị
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Đồ thị thông lượng
    ax1.plot(G, throughputs, 'o-', color='royalblue',
             label=f'FSA-BEB (N={NUM_NODES}, L={FRAME_LENGTH})')
    ax1.plot(G, S_theory, '--', color='red',
             label='Lý thuyết (S = G·e^{-G})')
    ax1.set_xlabel('Tải trọng cung cấp (G = λ/L)')
    ax1.set_ylabel('Thông lượng chuẩn hóa (S)')
    ax1.grid(True)
    ax1.legend()
    
    # Đồ thị tỷ lệ rớt gói
    ax2.plot(G, drop_rates, 'o-', color='seagreen')
    ax2.set_xlabel('Tải trọng cung cấp (G = λ/L)')
    ax2.set_ylabel('Tỷ lệ rớt gói')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_experiment()