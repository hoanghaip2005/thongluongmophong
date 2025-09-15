import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def simulate_single_run(L, N, num_trials):
    """
    Hàm lõi: Mô phỏng cho một cặp giá trị (L, N) duy nhất.
    Đây là mô hình bão hòa, nơi N nút luôn có gói tin để gửi.

    Args:
        L (int): Kích thước khung (số khe).
        N (int): Số lượng nút cạnh tranh.
        num_trials (int): Số lượng frame mô phỏng để lấy trung bình.

    Returns:
        float: Giá trị thông lượng S tính được.
    """
    total_successful_slots = 0

    # Lặp qua tất cả các frame (trials) để tính trung bình
    for _ in range(num_trials):
        # Bước 1: N nút ngẫu nhiên chọn một khe từ 0 đến L-1.
        # Xây dựng mảng size N, mỗi phần tử trong mảng là 1 số ngẫu nhiên từ 0 : L-1.
        choices = np.random.randint(0, L, size=N)

        # Bước 2: Đếm số lần mỗi khe được chọn.
        # np.histogram là một cách hiệu quả để đếm.
        slot_counts, _ = np.histogram(choices, bins=np.arange(L + 1))

        # Bước 3: Một khe được coi là thành công NẾU chỉ có đúng 1 nút chọn nó.
        successes_in_frame = np.sum(slot_counts == 1)

        # Cộng dồn số khe thành công của frame này vào tổng số
        total_successful_slots += successes_in_frame

    # Bước 4: Tính thông lượng trung bình S
    # S = (Tổng số thành công) / (Tổng số khe có thể có)
    throughput = total_successful_slots / (L * num_trials)

    return throughput

def run_experiment(L, n_range, num_trials):
    """
    Hàm điều khiển: Chạy toàn bộ thí nghiệm qua nhiều giá trị N.
    """
    n_results = []
    s_results = []

    # Sử dụng tqdm để tạo thanh tiến trình đẹp mắt
    for n in tqdm(n_range, desc=f"Simulating for L={L}"):
        # Gọi hàm mô phỏng lõi cho mỗi giá trị n
        s = simulate_single_run(L, n, num_trials)

        # Lưu kết quả
        n_results.append(n)
        s_results.append(s)

    return n_results, s_results

def plot_results(n_values, s_values, L):
    """
    Hàm vẽ đồ thị kết quả.
    """
    plt.style.use('seaborn-v0_8-whitegrid') # Sử dụng style cho đồ thị đẹp hơn
    plt.figure(figsize=(12, 7)) # Kích thước đồ thị

    # Vẽ đường cong S-N
    plt.plot(n_values, s_values, marker='.', linestyle='-', label='Throughput (S)')

    # Tìm và đánh dấu điểm thông lượng cực đại
    max_s = np.max(s_values)
    optimal_n_index = np.argmax(s_values)
    optimal_n = n_values[optimal_n_index]

    plt.axvline(x=optimal_n, color='r', linestyle='--', label=f'Optimal Number of Nodes ≈ {optimal_n}')
    plt.axhline(y=max_s, color='r', linestyle='--')

    # Ghi chú giá trị cực đại
    annotation_text = f'S_max ≈ {max_s:.4f}\nOptimal N = {optimal_n}'
    plt.annotate(annotation_text,
                 xy=(optimal_n, max_s),
                 xytext=(optimal_n + 5, max_s - 0.05),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 fontsize=12)

    # Thiết lập các thông tin cho đồ thị
    plt.title(f'Throughput of Framed Slotted ALOHA (L = {L})', fontsize=16)
    plt.xlabel('Number of Nodes (N)', fontsize=12)
    plt.ylabel('Throughput (S)', fontsize=12)
    plt.legend(fontsize=12)
    plt.ylim(bottom=0)
    plt.xlim(left=0)

    # Hiển thị đồ thị
    plt.show()

# --- KHỐI ĐIỀU KHIỂN CHÍNH ---
if __name__ == '__main__':
    # 1. Thiết lập các tham số cho kịch bản
    L_PARAM = 20       # Kích thước khung
    NUM_TRIALS = 10000 # Số frame mô phỏng cho mỗi điểm dữ liệu
    N_RANGE = range(1, 61) # Quét N từ 1 đến 60

    # 2. Chạy thí nghiệm
    print("--- Bắt đầu môi trường mô phỏng ---")
    n_data, s_data = run_experiment(L_PARAM, N_RANGE, NUM_TRIALS)
    print("--- Mô phỏng hoàn tất ---")

    # 3. Vẽ và hiển thị kết quả
    print("--- Đang vẽ đồ thị kết quả ---")
    plot_results(n_data, s_data, L_PARAM)