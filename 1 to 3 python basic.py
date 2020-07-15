#2강 LIST설명
a = [10,20,30,40,50]
print("a=[0]",a[0],"a=[1]",a[1],"a=[4]",a[4])
#list안에 list
b = [10,20,"hello",[True, 3.14]]
print("b=[0]",b[0],"b=[1]",b[1],"b=[-1]",b[-1])
#append 메소드
c= []
c.append(100),c.append(200),c.append(300)
print(c)
#콜론을 이용한 슬라이싱 기능
print("a[0:2]=",a[0:2],"a[1:]=",a[0:2],"a[:-2]=",a[:-2])
print("a[:]=",a[:])
print(a)
#TUPLE 데이터 타입 immutable
a=(10,20,30,40,50)
print("a[0] == ", a[0],"a[-2] == ", a[-2])
#DICTIONARY 데이터 타입
score = {'KIM':90,'LEE':85,'JUN':95,}
print("score['KIM] == ",score['KIM'])
#DICTIONARY 타입은 입력한 순서대로 데이터가 들어가지 않음!
score['HAN'] = 100
print(score)
#딕셔너리 key, value, (key,value)
print("score key ==", score.keys())
print("score value ==", score.values())
print("score items ==", score.items())
#STRING 데이터 타입
a = 'A73,CD'
print(a[1])
a = a+",EFG"
print(a)
#여러 문자열 함수들중 split 가장많이사용 ,로 나눠 list로 리턴
b = a.split(',')
print(b)

#추가로 자주 쓰이는 함수들 type(),len(),size(),list(),str(),int()
a = [10,20,30,40,50]
b = (10,20,30,40,50)
c = {"KIM":90, "LEE":80}
d = 'seoul, Korea'
e = [[100,200],[300,400],[500,600]]

print(type(a),type(b),type(c),type(d),type(e))
print(len(a),len(b),len(c),len(d),len(e))
#print(size(e)) 에서 size함수가 인식안됨 사이즈는 모든 원소의 개수를 나타내줌 결과값 6

a = 'Hello'
b = {"KIM":90, "LEE":80}
print(list(a),list(b.keys()),list(b.values()),list(b.items()))
print(str(3.14)+str('100')+str([1,2,3]))
print(int('100'),int(3.14))





#### 3강 조건문, 반복문
# 파이썬은 코딩블럭을 표시할때 들여쓰기 사용
# if문 예시
a = 1
if a>0:
    print("a == ",a)
elif a==0:
    print("a ==",a)
    print("zero")

list_data = [10,20,300,40,50]
dict_data = {"k1":1,"k2":2}

if 45 in list_data:
    print("45 is in list_data")
else:
    print("45 is not in list_data")

if 'k1' in dict_data:
    print("k1 is in dict_data")
else:
    print("k1 is not in dict_data")

#for문 예시
for data in range(10):
    print(data," ") #end=""는 같은행에서 출력하기위함
for data in range(0,10):
    print(data," ") #증감값의 default인 1씩증가된 값들을 표시
for data in range(0,10,2): #start,end,증감값
    print(data," ")

list_data = [10,20,30,40,50]
for data in list_data:
    print(data," ")
dict_data = {'key1':1,"key2":2}
for data in dict_data:
    print(data," ")
for key,value in dict_data.items():
    print(key," ", value)    

#list comprehension 기법
list_data = [x**2 for x in range(5)]
print(list_data)
raw_data = [[1,10],[2,15],[3,30],[4,55]]

all_data = [x for x in raw_data]
x_data = [x[0] for x in raw_data]
y_data = [x[1] for x in raw_data]

print("all_data", all_data)
print("x_data", x_data)
print("y_data", y_data)

#list comprehention 사용예시
even_number = []

for data in range(10):
    if data %2 == 0:
        even_number.append(data)
print(even_number)

#while, break, continue 예시
data = 5
while data >= 0:
    print("data ==", data)
    data -= 1
data = 5
while data >= 0:
    print("data ==", data)
    data -= 1

    if data ==2:
        print("break here")
        break
    else:
        print("continue here")
        continue
