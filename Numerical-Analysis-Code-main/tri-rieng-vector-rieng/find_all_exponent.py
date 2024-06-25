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
        
        # Bước 3: Chuẩn hóa v^(k) = w^(k) / ||w^(k)||
        v = w / np.linalg.norm(w)
        
        # Bước 4: Kiểm tra điều kiện dừng
        if iter > 0 and np.abs(lambda_ - lambda_old) < tol:
            break
        
        # Lưu lại giá trị λ_k của vòng lặp trước để so sánh
        lambda_old = lambda_
    
    return lambda_, v

def power_method_second_largest(A, tol=1e-8, max_iter=1000):
    n = A.shape[0]
    
    # Bước 1: Tìm giá trị riêng lớn nhất và vector riêng tương ứng
    lambda_1, v_1 = power_method(A, tol, max_iter)
    
    # Bước 2: Loại bỏ thành phần của λ_1 và v_1 để tìm giá trị riêng lớn thứ hai
    B = A - lambda_1 * np.outer(v_1, v_1)
    
    # Bước 3: Áp dụng phương pháp lũy thừa lên ma trận đã giảm kích thước
    lambda_2, v_2 = power_method(B, tol, max_iter)
    
    return lambda_2, v_2

def find_eigenpairs(A, tol=1e-8, max_iter=1000):
    n = A.shape[0]
    eigenvalues = []
    eigenvectors = []
    
    # Duyệt qua từng giá trị riêng
    for i in range(n):
        # Bước 1: Tìm giá trị riêng lớn nhất và vector riêng tương ứng
        lambda_i, v_i = power_method(A, tol, max_iter)
        
        # Lưu giá trị riêng và vector riêng vào danh sách
        eigenvalues.append(lambda_i)
        eigenvectors.append(v_i)
        
        # Loại bỏ thành phần của λ_i và v_i để tìm giá trị riêng tiếp theo
        A = A - lambda_i * np.outer(v_i, v_i)
    
    return eigenvalues, eigenvectors

if __name__ == "__main__":
    # Ma trận đối xứng 6x6 đầu vào cho ví dụ
    A = np.array([[4, 1, 2, 0, 0, 0],
                  [1, 5, 3, 0, 0, 0],
                  [2, 3, 6, 0, 0, 0],
                  [0, 0, 0, 2, 1, 0],
                  [0, 0, 0, 1, 3, 0],
                  [0, 0, 0, 0, 0, 5]])
    
    # Đường dẫn đến file output
    output_file = "/Users/a8888/Documents/doc uni/gts/Numerical-Analysis-Code-main/tri-rieng-vector-rieng/output.txt"
    
    # Kiểm tra tính đối xứng của ma trận A
    if not np.allclose(A, A.T):
        with open(output_file, "w") as file:
            file.write("Lỗi: Ma trận không phải là ma trận đối xứng.")
    else:
        # Tìm tất cả các giá trị riêng và vector riêng tương ứng
        eigenvalues, eigenvectors = find_eigenpairs(A)
        
        # Ghi kết quả ra file
        with open(output_file, "w") as file:
            file.write(f"Ma trận đối xứng A:\n{A}\n\n")
            for i in range(len(eigenvalues)):
                file.write(f"Giá trị riêng λ_{i+1} = {eigenvalues[i]}\n")
                file.write(f"Vector riêng tương ứng v_{i+1} = {eigenvectors[i]}\n\n")
                
                # Kiểm tra tính chất A * v_i = λ_i * v_i cho từng cặp giá trị riêng và vector riêng
                Av = np.dot(A, eigenvectors[i])
                check_eigenvector = np.allclose(Av, eigenvalues[i] * eigenvectors[i])
                file.write(f"Kiểm tra tính chất vector riêng và giá trị riêng λ_{i+1}: {'Đúng' if check_eigenvector else 'Sai'}\n")
