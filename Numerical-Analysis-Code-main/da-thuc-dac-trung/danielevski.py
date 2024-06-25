import numpy as np

def jordan_form(A):
    """
    Tính dạng Jordan chuẩn của ma trận A.
    Trả về ma trận P và ma trận chuẩn tắc Jordan J.
    """
    eigenvalues, P = np.linalg.eig(A)
    J = np.diag(eigenvalues)  # Ma trận Jordan ban đầu
    return P, J

def characteristic_polynomial(A):
    """
    Tính đa thức đặc trưng của ma trận A bằng phương pháp Danielevski.
    """
    P, J = jordan_form(A)
    n = A.shape[0]
    poly = np.poly1d([1.0])
    
    for i in range(n):
        lambda_i = J[i, i]
        block_size = 1
        
        # Xác định kích thước khối Jordan
        if i < n - 1 and J[i, i+1] == 1:
            block_size += 1
        
        # Tính đa thức đặc trưng cho khối Jordan
        lambda_i_poly = np.poly1d([1, -lambda_i])
        for _ in range(1, block_size):
            lambda_i_poly = np.polymul(lambda_i_poly, np.poly1d([1, -lambda_i]))
        
        # Nhân với đa thức tích của các khối Jordan
        poly = np.polymul(poly, lambda_i_poly)
    
    return poly

# Ví dụ minh họa cho ma trận 5x5
if __name__ == "__main__":
    A = np.array([[4, 1, 2, 0, 0],
                  [1, 5, 3, 0, 0],
                  [2, 3, 6, 0, 0],
                  [0, 0, 0, 2, 1],
                  [0, 0, 0, 1, 3]])
    
    print("Ma trận A:")
    print(A)
    
    print("\nBước 1: Tính ma trận chuẩn tắc Jordan của ma trận A")
    P, J = jordan_form(A)
    print("   Ma trận P:")
    print(P)
    print("   Ma trận chuẩn tắc Jordan J:")
    print(J)
    
    print("\nBước 2: Xây dựng đa thức đặc trưng từ các khối Jordan")
    char_poly = characteristic_polynomial(A)
    print("   Đa thức đặc trưng của A:")
    print(char_poly)
