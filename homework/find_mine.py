'''
200908 지뢰찾기 문제
표준 입력으로 2차원 리스트의 가로(col)와 세로(row)가 입력되고 그 다음 줄부터 리스트의
요소로 들어갈 문자가 입력됩니다. 이때 2차원 리스트 안에서 *는 지뢰이고 .은 지뢰가 아닙니다.
지뢰가 아닌 요소에는 인접한 지뢰의 개수를 출력하는 프로그램을 만드세요.
(input에서 안내 문자열은 출력하지 않아야 합니다)

여러 줄을 입력 받으려면 다음과 같이 for 반복문에서 input을 호출한 뒤 append로 각 줄을
추가하면 됩니다(list 안에 문자열을 넣으면 문자열이 문자 리스트로 변환됩니다).

matrix = []
for i in range(row):
    matrix.append(list(input()))
'''

# input.txt 텍스트 파일에 정보를 미리 입력하여 받아오는 방법으로 구현
# 첫 번째 줄에 rows cols 를 입력하고, 두 번째 줄부터는 지뢰 정보를 입력
import sys

sys.stdin = open("input.txt", "r")

matrix = []
rows, cols = map(int, input().split())

for i in range(rows):
    matrix.append(list(input()))

'''
# 받은 지뢰 정보 출력
for i in matrix:
    for j in i:
        print(j, end=" ")
    print("")
print("")
'''

# 해당 요소가 지뢰면 바로 출력, 아니면 상하좌우 대각선의 지뢰 갯수 세서 출력
for i in range(rows):
    for j in range(cols):
        if matrix[i][j] == "*":
            print(matrix[i][j], end=" ")
        else:
            cnt = 0
            # 가로 +-1, 세로 +-1 인덱스에 있는 요소에 지뢰가 있는지 비교
            for r in range(i - 1, i + 2):
                for c in range(j - 1, j + 2):
                    # 배열 범위를 벗어나거나 자기 자신은 체크하지 않음
                    if r < 0 or r > rows - 1 or c < 0 or c > cols - 1 or (r == i and c == j):
                        continue
                    elif matrix[r][c] == "*":
                        cnt += 1
                    else:
                        continue
            print(cnt, end=" ")
    print()
