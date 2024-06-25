import numpy as np

def power_method(A, tol=1e-10, max_iter=10000):
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
        
        # Bước 3: Chuẩn hóa v^(k) = w^(k) / ||w^(k)||
        v = w / np.linalg.norm(w)
        
        # Bước 4: Kiểm tra điều kiện dừng
        if iter > 0 and np.abs(lambda_ - lambda_old) < tol:
            break
        
        # Lưu lại giá trị λ_k của vòng lặp trước để so sánh
        lambda_old = lambda_
    
    return lambda_, v

# Ma trận đối xứng 8x8 cụ thể cho ví dụ
A = np.array([[4, 1, 2, 0, 0, 0, 0, 0],
              [1, 5, 3, 0, 0, 0, 0, 0],
              [2, 3, 6, 0, 0, 0, 0, 0],
              [0, 0, 0, 2, 1, 0, 0, 0],
              [0, 0, 0, 1, 3, 0, 0, 0],
              [0, 0, 0, 0, 0, 5, 0, 0],
              [0, 0, 0, 0, 0, 0, 3, 1],
              [0, 0, 0, 0, 0, 0, 1, 2]])

if __name__ == "__main__":
    # In thông tin về ma trận A
    print("Ma trận đối xứng A:\n", A)
    
    # Kiểm tra tính đối xứng của ma trận A
    if not np.allclose(A, A.T):
        print("Lỗi: Ma trận không phải là ma trận đối xứng.")
    else:
        # Áp dụng phương pháp lũy thừa để tính giá trị riêng lớn nhất và vector riêng tương ứng
        lambda_1, v_1 = power_method(A, tol=1e-12, max_iter=20000)
        
        # In kết quả
        print("\nKết quả:")
        print(f"Giá trị riêng lớn nhất λ_1 = {lambda_1}")
        print(f"Vector riêng tương ứng v_1 = {v_1}")
        
        # Kiểm tra tính chất A * v_1 = λ_1 * v_1
        Av = np.dot(A, v_1)
        check_eigenvector = np.allclose(Av, lambda_1 * v_1)
        
        # In kiểm tra
        print("\nKiểm tra tính chất vector riêng và giá trị riêng:")
        print(f"A * v_1 = {Av}")
        print(f"λ_1 * v_1 = {lambda_1} * v_1 = {lambda_1 * v_1}")
        print(f"Phát hiện vector riêng: {'Đúng' if check_eigenvector else 'Sai'}")
