####4강 함수와 람다####

#funtion 예시 데이터타입 명시 필요없음 유연함
def sum(x,y):
    s = x+y
    return s
result = sum(10, 20)
print(result)
#함수 반환값 여러개 가능
def multi_ret_func(x):
    return x+1, x+2, x+3
x=100
y1,y2,y3 = multi_ret_func(x)
print(y1,y2,y3)
# default 파라미터의 예시
def print_name(name, count=2):
    for name in range(count):
        print("name==",name)
print_name("Dave") #print_name("Dave",5)하면 5번 출력함
# mutable/immutable parameter 예시
# mutable>>>list,dict,numpy등의 데이터형일 경우 원래 데이터의 변형이 일어남
# immutable>>>반대로 숫자,문자,tuple등은 데이터의 변형이 일어나지 않음
def mutable_immutable_func(int_x, input_list):
    int_x +=1
    input_list.append(100)
x=1
test_list = [1,2,3]
mutable_immutable_func(x,test_list)
print("x ==",x ,", test+list ==", test_list)
#lambda function 예시
#f(x) = x+100 과 같은것
f = lambda x : x+100
for i in range(3):
    print(f(i))
#lambda 함수 추가예시
def print_hello():
    print("hello")
def test_lambda(s,t):
    print("input1=",s,"input2=",t)
s=100
t=200
fx = lambda x,y : test_lambda(s,t)
fy = lambda x,y : print_hello()

fx(500, 1000)
fy(300, 600)

#class 예시
class Person:
    
    count = 0

    def __init__(self, name):
        self.name = name
        Person.count +=1
        print(self.name + "is initialized")

    def work(self, company):
        print(self.name + "is working in "+company)
    
    def sleep(self):
        print(self.name + "is sleeping")
    
    @classmethod #클래스 메소드 생성시 반드시 표시해줘야함
    def getCount(cls):
        return cls.count
#Person instance 생성
obj = Person("PARK")
#메소드콜
obj.work("ABCDEF")
obj.sleep()

print("current person object is", obj.name)

#클래스변수와 클래스 메소드
#클래스 메소드는 객체 인스턴스를 의미하는 self대신 cls라는 클래스를 의마하는 파라미터를 인수로 전달받음
obj1 = Person("PARK")
obj2 = Person("KIM")

obj1.work("ABCDEF")
obj2.sleep()

print("current person object is", obj1.name,", ",obj2.name)
print("Person count ==", Person.getCount())
print(Person.count)

#파이썬은 기본적으로 public 인데 __언더바 두개를 통해 private화 가능
#멤버변수, 멤버메소드를 __멤버변수, __멤버메소드 형태로 선언한다면 private화 가능
class PrivateMemberTest:
    def __init__(self, name1, name2):
        self.name1 = name1
        self.__name2 = name2
        print("initialized with "+ name1 + " ,"+name2)

    def getNames(self):
        self.__printNames()
        return self.name1, self.__name2

    def __printNames(self):
        print(self.name1, self.__name2)
 
obj = PrivateMemberTest("PARK","KIM")

print(obj.name1)
print(obj.getNames())
# print(obj.__printNames()) __는 private이라 호출불가
# print(obj.__name2) __는 private이라 호출불가

#외부함수와 클래스 method name이 같은 경우

def print_name2(name):
    print("[def]",name)

class SameTest:
    def __init__(self):
        pass
    def print_name2(self, name):
        print("[SameTest]",name)
    def call_test(self):
        #외부 함수 호출
        print_name2("KIM")
        #클래스 내부 method 호출
        self.print_name2("KIM")

obj = SameTest()
print_name2("LEE")
obj.print_name2("LEE")
obj.call_test()
        
#예외처리 - exception
#try블럭에서 에러가발생하면 except 문으로 이동하여 예외처리 수행
def calc(list_data):
    sum = 0
    try:
        sum = list_data[0] + list_data[1] + list_data[2]
        if sum <0:
            raise Exception("Sum is minus")

    except IndexError as err:
        print(str(err))
    except Exception as err:
        print(str(err))
    finally:
        print(sum)

calc([1,2])#index error 발생
calc([1,2,-100])#인위적인 exception 발생

#with 구문예시
#일반적으로 file이나 session을 사용하는 순서는 다음과같음
#open => read() 또는 write() => close()
#그러나 파이썬에서는 with 구문을 사용하면 명시적으로 리소스 close()를 해주지 않아도 자동으로 해줌
#일반적인 방법은
f= open("./file_test","w")
f.write("hello!!")
f.close()

#with 구문을 사용한 방법
#with 블록을 벗어나는 순간 파일 객체 f가 자동으로 close 됨.
with open("./file_test","w") as f:
    f.write("hello!!")
#딥러닝 프레임워크인 TensorFlow의 session 사용시 자주 이용됨
