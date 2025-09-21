# WITH BEB
# -*- coding: utf-8 -*-
"""
Mô phỏng Frame Slotted ALOHA (FSA) với cơ chế Backoff Luỹ thừa Nhị phân (BEB).
- Các gói tin mới đến hệ thống theo phân phối Poisson.
- Các node chỉ truyền khi có gói tin và không trong trạng thái backoff.
- Các gói tin va chạm sẽ được truyền lại sau một khoảng thời gian chờ (backoff)
  được xác định bởi thuật toán BEB.
"""

import numpy as np
import matplotlib.pyplot as plt
import time

# ==============================================================================
# THAM SỐ CHO THUẬT TOÁN BEB
# ==============================================================================
# Số mũ tối đa cho cửa sổ cạnh tranh (CW). CW = 2^c - 1.
# Giới hạn này (ví dụ: 10) tương đương với CW_max = 2^10 - 1 = 1023.
# Giúp ngăn chặn thời gian chờ quá dài.
CW_EXP_MAX = 10
# Số lần thử lại tối đa trước khi một gói tin bị hủy bỏ.
MAX_RETRIES = 15


def run_fsa_beb_simulation(arrival_rate_lambda, num_nodes, frame_length, num_frames):
    """
    Hàm chính để chạy một lần mô phỏng FSA với cơ chế BEB.

    Args:
        arrival_rate_lambda (float): Tốc độ đến trung bình của các gói tin mới trong toàn hệ thống (gói/khung).
        num_nodes (int): Tổng số node trong mạng (N).
        frame_length (int): Số khe trong một khung (L).
        num_frames (int): Tổng số khung để chạy mô phỏng.

    Returns:
        tuple: (Thông lượng chuẩn hóa, Tỷ lệ rớt gói)
    """
    # Khởi tạo trạng thái cho mỗi node. Mỗi node là một dictionary.
    nodes = [{'buffer': 0, 'collision_count': 0, 'backoff_counter': 0} for _ in range(num_nodes)]

    # Biến đếm
    total_successful_packets = 0
    total_dropped_packets = 0
    total_new_packets = 0
    total_attempted_transmissions = 0  # Đếm tổng số lần cố gắng truyền

    # Bắt đầu vòng lặp mô phỏng qua từng khung
    for frame in range(num_frames):
        # 0. QUẢN LÝ TRẠNG THÁI BACKOFF
        # Trước mỗi khung, giảm bộ đếm backoff cho các node đang phải chờ.
        for node in nodes:
            if node['backoff_counter'] > 0:
                node['backoff_counter'] -= 1

        # 1. GÓI TIN MỚI ĐẾN (POISSON ARRIVAL)
        new_packets_count = np.random.poisson(lam=arrival_rate_lambda)
        total_new_packets += new_packets_count

        if new_packets_count > 0:
            nodes_getting_packets = np.random.randint(0, num_nodes, size=new_packets_count)
            for node_idx in nodes_getting_packets:
                nodes[node_idx]['buffer'] += 1

        # 2. XÁC ĐỊNH CÁC NODE "SẴN SÀNG"
        # Node sẵn sàng là node có gói tin VÀ không trong thời gian backoff.
        ready_nodes_indices = [
            i for i, node in enumerate(nodes)
            if node['buffer'] > 0 and node['backoff_counter'] == 0
        ]

        if not ready_nodes_indices:
            continue

        # 3. QUÁ TRÌNH TRUYỀN DẪN
        slot_usage = np.zeros(frame_length, dtype=int)
        node_to_slot_map = {}

        for node_idx in ready_nodes_indices:
            chosen_slot = np.random.randint(0, frame_length)
            slot_usage[chosen_slot] += 1
            node_to_slot_map[node_idx] = chosen_slot
            total_attempted_transmissions += 1  # Đếm mỗi lần cố gắng truyền

        # 4. XỬ LÝ KẾT QUẢ VÀ ÁP DỤNG BEB
        for node_idx, chosen_slot in node_to_slot_map.items():
            if slot_usage[chosen_slot] == 1:
                # TRUYỀN THÀNH CÔNG
                nodes[node_idx]['buffer'] -= 1
                nodes[node_idx]['collision_count'] = 0 # Reset bộ đếm xung đột
                total_successful_packets += 1
            else:
                # XUNG ĐỘT
                nodes[node_idx]['collision_count'] += 1

                if nodes[node_idx]['collision_count'] > MAX_RETRIES:
                    # Gói tin bị hủy bỏ do vượt quá số lần thử lại
                    nodes[node_idx]['buffer'] -= 1
                    nodes[node_idx]['collision_count'] = 0 # Reset cho gói tin tiếp theo
                    total_dropped_packets += 1
                else:
                    # Áp dụng thuật toán BEB
                    # Giới hạn số mũ để tránh cửa sổ quá lớn
                    exp = min(nodes[node_idx]['collision_count'], CW_EXP_MAX)
                    window_size = (2**exp) - 1

                    # Chọn một khoảng backoff ngẫu nhiên (tính bằng số khung)
                    backoff_frames = np.random.randint(0, window_size + 1)
                    nodes[node_idx]['backoff_counter'] = backoff_frames

    # 5. TÍNH TOÁN KẾT QUẢ CUỐI CÙNG
    # Thông lượng chuẩn hóa theo định nghĩa lý thuyết: số gói thành công / tổng số khe
    normalized_throughput = total_successful_packets / (num_frames * frame_length)
    
    # Offered load thực tế từ mô phỏng (để so sánh)
    actual_offered_load = total_attempted_transmissions / (num_frames * frame_length)
    
    packet_drop_rate = total_dropped_packets / total_new_packets if total_new_packets > 0 else 0

    return normalized_throughput, packet_drop_rate, actual_offered_load

# ==============================================================================
# THAM SỐ MÔ PHỎNG
# ==============================================================================
NUM_NODES = 50
FRAME_LENGTH = 20
NUM_FRAMES = 5000 # Tăng số khung để kết quả ổn định hơn với BEB

ARRIVAL_RATES = np.linspace(0.1, FRAME_LENGTH * 2.5, 40)

# ==============================================================================
# CHẠY MÔ PHỎNG VÀ THU THẬP KẾT QUẢ
# ==============================================================================
throughputs_beb = []
drop_rates_beb = []
actual_loads = []  # Lưu offered load thực tế từ mô phỏng

start_time = time.time()
print("Bắt đầu mô phỏng FSA với BEB...")
print(f"Tham số: N={NUM_NODES}, L={FRAME_LENGTH}, Số khung={NUM_FRAMES}")
print(f"BEB params: CW_EXP_MAX={CW_EXP_MAX}, MAX_RETRIES={MAX_RETRIES}")
print("-" * 60)

for rate in ARRIVAL_RATES:
    print(f"Đang mô phỏng với Tải trọng cung cấp (λ) = {rate:.2f} gói/khung...")
    throughput, drop_rate, actual_load = run_fsa_beb_simulation(rate, NUM_NODES, FRAME_LENGTH, NUM_FRAMES)
    throughputs_beb.append(throughput)
    drop_rates_beb.append(drop_rate)
    actual_loads.append(actual_load)

end_time = time.time()
print("-" * 60)
print(f"Mô phỏng hoàn tất sau {end_time - start_time:.2f} giây.")

# ==============================================================================
# TÍNH TOÁN THÔNG LƯỢNG LÝ THUYẾT (S = G * e^-G)
# ==============================================================================
# Theo lý thuyết FSA: G là số lượng gói tin trung bình được cố gắng truyền đi trong 1 khe thời gian
# G = tổng gói tin cố gắng truyền / tổng số khe
# Với arrival rate λ (gói/khung), trong 1 khung có L khe, thì G = λ/L
offered_load_G = ARRIVAL_RATES / FRAME_LENGTH
theoretical_throughputs = offered_load_G * np.exp(-offered_load_G)

# ==============================================================================
# VẼ ĐỒ THỊ KẾT QUẢ
# ==============================================================================
plt.style.use('seaborn-v0_8-whitegrid')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

# Đồ thị 1: So sánh Throughput với lý thuyết
ax1.set_xlabel('Offered Load G (packets/slot)', fontsize=12)
ax1.set_ylabel('Throughput S', fontsize=12, color='royalblue')

# Vẽ đường cong mô phỏng theo offered load thực tế
ax1.plot(actual_loads, throughputs_beb, 'o-', color='royalblue', 
         label=f'FSA-BEB Simulation (N={NUM_NODES}, L={FRAME_LENGTH})', markersize=4)

# Vẽ đường cong lý thuyết
G_theory = np.linspace(0.01, 3, 100)
S_theory = G_theory * np.exp(-G_theory)
ax1.plot(G_theory, S_theory, '--', color='red', linewidth=2, label='Theory: S = G × e^(-G)')

# Đánh dấu điểm tối ưu lý thuyết (G=1, S≈0.368)
ax1.axvline(x=1, color='green', linestyle=':', alpha=0.7, label='Optimal G = 1')
ax1.axhline(y=1/np.e, color='green', linestyle=':', alpha=0.7, label='Max S ≈ 0.368')

ax1.set_xlim([0, 2.5])
ax1.set_ylim([0, 0.4])
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_title('Throughput vs Offered Load', fontsize=14)

# Đồ thị 2: Throughput và Drop rate theo Arrival rate
ax2.set_xlabel('λ (packets/frame)', fontsize=12)
ax2.set_ylabel('Throughput', fontsize=12, color='royalblue')
ax2.plot(ARRIVAL_RATES, throughputs_beb, 'o-', color='royalblue', label='Throughput')
ax2.tick_params(axis='y', labelcolor='royalblue')
ax2.set_ylim([0, 0.4])

# Trục y2 cho Packet Drop Rate
ax2_twin = ax2.twinx()
ax2_twin.set_ylabel('Packet Drop Rate', fontsize=12, color='seagreen')
ax2_twin.plot(ARRIVAL_RATES, drop_rates_beb, 's--', color='seagreen', 
              markersize=5, alpha=0.7, label='Drop Rate')
ax2_twin.tick_params(axis='y', labelcolor='seagreen')
ax2_twin.set_ylim([0, 1])

ax2.set_title('Performance vs Arrival Rate', fontsize=14)
ax2.grid(True, alpha=0.3)

# Tổng hợp legend cho đồ thị 2
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_twin.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.tight_layout()
plt.suptitle('Frame Slotted ALOHA with Binary Exponential Backoff Analysis', 
             fontsize=16, y=0.98)

# Lưu đồ thị và hiển thị
plt.savefig('FSA_BEB_Analysis.png', dpi=300, bbox_inches='tight')
print("Đồ thị đã được lưu vào file: FSA_BEB_Analysis.png")
plt.show()

print("\n" + "="*70)
print("THÔNG TIN PHÂN TÍCH:")
print("="*70)
print(f"Throughput tối đa đạt được: {max(throughputs_beb):.4f}")
max_idx = np.argmax(throughputs_beb)
print(f"Tại arrival rate λ = {ARRIVAL_RATES[max_idx]:.2f} packets/frame")
print(f"Offered load G thực tế = {actual_loads[max_idx]:.4f}")
print(f"Lý thuyết tối ưu: G = 1.0, S_max ≈ {1/np.e:.4f}")
print("="*70)