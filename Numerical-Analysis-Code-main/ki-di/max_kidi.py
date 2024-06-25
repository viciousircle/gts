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
        
        # In từng bước trong quá trình lặp
        print(f"Bước {iter + 1}:")
        print(f"λ_{iter + 1} = {lambda_}")
        print(f"v_{iter + 1} = {v}")
        print("------------------------")
    
    return lambda_, v

def find_max_anomalous_value(A):
    n = A.shape[0]
    
    # Bước 1: Tìm giá trị riêng lớn nhất và vector riêng tương ứng
    lambda_1, v_1 = power_method(A)
    
    # In thông tin về giá trị riêng lớn nhất và vector riêng tương ứng
    print("\nKết quả phương pháp lũy thừa:")
    print(f"Giá trị riêng lớn nhất λ_1 = {lambda_1}")
    print(f"Vector riêng tương ứng v_1 = {v_1}")
    print("------------------------")
    
    # Bước 2: Tính giá trị kỳ dị lớn nhất
    max_anomalous_value = 0.0
    
    for i in range(n):
        for j in range(i+1, n):
            anomalous_value = np.abs(lambda_1 - A[i,j])
            if anomalous_value > max_anomalous_value:
                max_anomalous_value = anomalous_value
    
    # In thông tin về giá trị kỳ dị lớn nhất
    print("\nKết quả tìm giá trị kỳ dị lớn nhất:")
    print(f"Giá trị kỳ dị lớn nhất của ma trận A là: {max_anomalous_value}")
    
    return max_anomalous_value

if __name__ == "__main__":
    # Ma trận cấp 6 đầu vào cho ví dụ
    A = np.array([[4, 1, 2, 0, 0, 0],
                  [1, 5, 3, 0, 0, 0],
                  [2, 3, 6, 0, 0, 0],
                  [0, 0, 0, 2, 1, 0],
                  [0, 0, 0, 1, 3, 0],
                  [0, 0, 0, 0, 0, 5]])
    
    # Tìm giá trị kỳ dị lớn nhất của ma trận A
    max_anomalous_value = find_max_anomalous_value(A)
