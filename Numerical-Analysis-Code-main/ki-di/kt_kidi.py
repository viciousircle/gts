import numpy as np

def power_method(A, tol=1e-8, max_iter=1000):
    n = A.shape[0]
    
    # Chọn vector bắt đầu ngẫu nhiên có độ dài 1
    v = np.random.rand(n)
    v = v / np.linalg.norm(v)
    
    # Lặp tối đa max_iter lần
    for iter in range(max_iter):
        # Bước 1: Tính A * v^(k-1)
        w = np.dot(A, v)
        
        # Bước 2: Tính λ_k = v^(k-1)T * w^(k)
        lambda_ = np.dot(v, w)
        
        # Bước 3: Chuẩn hóa v^(k) = w^(k) / ||w^(k)|| để tính toán tiếp
        v = w / np.linalg.norm(w)
        
        # Bước 4: Kiểm tra điều kiện dừng
        if iter > 0 and np.abs(lambda_ - lambda_old) < tol:
            break
        
        # Lưu lại giá trị λ_k của vòng lặp trước để so sánh
        lambda_old = lambda_
    
    return lambda_, v

def find_anomalous_expansion(A, lambda_1):
    m, n = A.shape
    
    # Bước 1: Tính giá trị riêng lớn nhất và vector riêng tương ứng
    # lambda_1, v_1 = power_method(A)  # lambda_1 được truyền từ bên ngoài
    
    # Bước 2: Xây dựng ma trận đặc trưng B = lambda_1 * v_1 * v_1^T
    v_1 = power_method(A)[1]
    B = lambda_1 * np.outer(v_1, v_1)
    
    # Bước 3: Tính ma trận khai triển kỳ dị A_anomalous = A - B
    A_anomalous = A - B
    
    return A_anomalous, lambda_1, v_1

if __name__ == "__main__":
    # Ma trận cấp 5x5 đầu vào cho ví dụ
    A = np.array([[2, 1, 0, 0, 0],
                  [1, 3, 1, 0, 0],
                  [0, 1, 4, 1, 0],
                  [0, 0, 1, 5, 1],
                  [0, 0, 0, 1, 6]])
    
    try:
        # Tìm giá trị riêng lớn nhất và ma trận khai triển kỳ dị của ma trận A
        lambda_1, v_1 = power_method(A)
        A_anomalous, _, _ = find_anomalous_expansion(A, lambda_1)
        
        # In kết quả
        print("Ma trận A:")
        print(A)
        print("\nBước 1: Tính giá trị riêng và vector riêng:")
        print(f"Giá trị riêng lớn nhất lambda_1 = {lambda_1}")
        print(f"Vector riêng tương ứng v_1 = {v_1}")
        print("\nBước 2: Xây dựng ma trận đặc trưng B = lambda_1 * v_1 * v_1^T:")
        print("Ma trận đặc trưng B:")
        print(lambda_1 * np.outer(v_1, v_1))
        print("\nBước 3: Tính ma trận khai triển kỳ dị A_anomalous = A - B:")
        print("Ma trận khai triển kỳ dị A_anomalous:")
        print(A_anomalous)
    
    except ValueError as e:
        print(f"Lỗi: {e}")
