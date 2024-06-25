import math
from numpy.lib import scimath
import os
"""
Điền num_decimal là số chữ số thập phân sau dấu phẩy bạn muốn hiển thị
Điền f(x) là hàm số bạn muốn giải
"""
num_decimal = 5  # Số chữ số thập phân hiển thị

# Nhập phương trình cần giải
def f(x):
    # return 2 * x**5 - 12 * x**4 + 3 * x**2 - 15
    return math.log(x) - 1
    # return math.sin(x)
    # return x**5 - 17
    # return math.exp(x) - math.cos(2 * x)

def sign_f(x):
    if f(x) < 0:
        return -1
    elif f(x) > 0:
        return 1
    elif f(x) == 0:
        return 0

def checkInput(a, b):
    if sign_f(a) * sign_f(b) > 0:
        print("Giá trị hàm tại hai đầu mút cùng dấu --> Kiểm tra lại đầu vào")
        return 0
    elif sign_f(a) == 0:
        print("Hàm số có nghiệm là đầu mút a = ", a)
        return 0
    elif sign_f(b) == 0:
        print("Hàm số có nghiệm là đầu mút b = ", b)
        return 0
    return 1

def tien_nghiem(a, b):
    n = math.ceil(scimath.log2((b - a) / eps))
    print("Số lần lặp: ", n)
    #print("_"*110)
    #print("|{0:^9}|{1:^25}|{2:^30}|{3:^20}|{4:^20}|".format("Lần lặp", "Nghiệm x", "f(x)", "Đầu mút a", "Đầu mút b"))
    #print("-" * 110)
    fout.write("\n" + "_" * 110)
    fout.write("\n|{0:^9}|{1:^25}|{2:^30}|{3:^20}|{4:^20}|".format("Lần lặp", "Nghiệm x", "f(x)", "Đầu mút a", "Đầu mút b"))
    fout.write("\n" + "_" * 110)
    for i in range(n):
        x = (a + b) / 2
        if f(x) == 0:
            #print("|{0:^9}|{1:^25}|{2:^30}|{3:^20}|{4:^20}|".format(i + 1, round(x, num_decimal), round(f(x), num_decimal), round(a, num_decimal), round(b, num_decimal)))
            print("Phương trình đạt nghiệm đúng sau ", n, "lần lặp")
            fout.write("\n|{0:^9}|{1:^25}|{2:^30}|{3:^20}|{4:^20}|".format(i + 1, round(x, num_decimal), round(f(x), num_decimal), round(a, num_decimal), round(b, num_decimal)))
            break
        elif f(a) * f(x) < 0:
            b = x
            #print("|{0:^9}|{1:^25}|{2:^30}|{3:^20}|{4:^20}|".format(i + 1, round(x, num_decimal), round(f(x), num_decimal), round(a, num_decimal), round(b, num_decimal)))
            fout.write("\n|{0:^9}|{1:^25}|{2:^30}|{3:^20}|{4:^20}|".format(i + 1, round(x, num_decimal), round(f(x), num_decimal), round(a, num_decimal), round(b, num_decimal)))
        elif f(x) * f(b) < 0:
            a = x
            #print("|{0:^9}|{1:^25}|{2:^30}|{3:^20}|{4:^20}|".format(i + 1, round(x, num_decimal), round(f(x), num_decimal), round(a, num_decimal), round(b, num_decimal)))
            fout.write("\n|{0:^9}|{1:^25}|{2:^30}|{3:^20}|{4:^20}|".format(i + 1, round(x, num_decimal), round(f(x), num_decimal), round(a, num_decimal), round(b, num_decimal)))
    fout.write("\n" + "_" * 110)
    return x

if __name__ == "__main__":
    x0: float
    print("Nhập khoảng tìm nghiệm: ")
    a = float(input("Nhập a: "))
    b = float(input("Nhập b: "))
    eps = float(input("Nhập sai số: "))
    print()
    if checkInput(a, b) == 1:
        try:
            # Dynamically determine the path for the output file
            current_dir = os.path.dirname(__file__)
            output_path = os.path.join(current_dir, "output_tn_cd.txt")
            
            fout = open(output_path, mode='w', encoding='utf-8')
            if not (f(a) * f(b) > 0):
                x0 = tien_nghiem(a, b)
                print("Nghiệm gần đúng của phương trình là: ")
                print(x0)
                fout.write("\nNghiệm gần đúng của phương trình là: " + str(x0))
            else:
                print("Khoảng (", a, ",", b, ") không là khoảng phân ly nghiệm")
        finally:
            fout.close()
