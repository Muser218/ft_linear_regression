# 파일 열기
with open('theta.csv', 'r') as file:
    lines = file.readlines()

theta = [float(i) for i in lines[0].split(',')]

input_km = float(input('주행거리(km)를 입력하세요: '))
print(f"해당 주행 거리에 대한 예상 가격: {theta[0] + theta[1] * input_km}")
