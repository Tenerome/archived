### course

#### 模块管理

##### 内置模块

```python
import math
dir(math)
```

```
['__doc__',
 '__loader__',
 '__name__',
 '__package__',
 '__spec__',
 'acos',
 'acosh',
 'asin',
 'asinh',
 'atan',
 'atan2',
 'atanh',
 'ceil',
 'comb',
 'copysign',
 'cos',
 'cosh',
 'degrees',
 'dist',
 'e',
 'erf',
 'erfc',
 'exp',
 'expm1',
 'fabs',
 'factorial',
 'floor',
 'fmod',
 'frexp',
 'fsum',
 'gamma',
 'gcd',
 'hypot',
 'inf'
```

```python
print(math.e)
```

```
2.718281828459045
```

```python
#用type()查看类型
print(type(math.e))
```

```
<class 'float'>
```

```python
#使用help()查看帮助文档
help(math)
```

```
Help on built-in module math:

NAME
    math

DESCRIPTION
    This module provides access to the mathematical functions
    defined by the C standard.

FUNCTIONS
    acos(x, /)
        Return the arc cosine (measured in radians) of x.

        The result is between 0 and pi.

    acosh(x, /)
        Return the inverse hyperbolic cosine of x.

    asin(x, /)
        Return the arc sine (measured in radians) of x.

        The result is between -pi/2 and pi/2.

    asinh(x, /)
        Return the inverse hyperbolic sine of x.

    atan(x, /)
        Return the arc tangent (measured in radians) of x.

        The result is between -pi/2 and pi/2.

    atan2(y, x, /)
        Return the arc tangent (measured in radians) of y/x.

        Unlike atan(y/x), the signs of both x and y are considered.

    atanh(x, /)
        Return the inverse hyperbolic tangent of x.

    ceil(x, /)
        Return the ceiling of x as an Integral.

        This is the smallest integer >= x.

    comb(n, k, /)
        Number of ways to choose k items from n items without repetition and without order.

        Evaluates to n! / (k! * (n - k)!) when k <= n and evaluates
        to zero when k > n.

        Also called the binomial coefficient because it is equivalent
        to the coefficient of k-th term in polynomial expansion of the
        expression (1 + x)**n.

        Raises TypeError if either of the arguments are not integers.
        Raises ValueError if either of the arguments are negative.

    copysign(x, y, /)
        Return a float with the magnitude (absolute value) of x but the sign of y.

        On platforms that support signed zeros, copysign(1.0, -0.0)
        returns -1.0.

    cos(x, /)
        Return the cosine of x (measured in radians).

    cosh(x, /)
        Return the hyperbolic cosine of x.

    degrees(x, /)
        Convert angle x from radians to degrees.

    dist(p, q, /)
        Return the Euclidean distance between two points p and q.

        The points should be specified as sequences (or iterables) of
        coordinates.  Both inputs must have the same dimension.

        Roughly equivalent to:
            sqrt(sum((px - qx) ** 2.0 for px, qx in zip(p, q)))

    erf(x, /)
        Error function at x.

    erfc(x, /)
        Complementary error function at x.

    exp(x, /)
        Return e raised to the power of x.

    expm1(x, /)
        Return exp(x)-1.

        This function avoids the loss of precision involved in the direct evaluation of exp(x)-1 for small x.

    fabs(x, /)
        Return the absolute value of the float x.

    factorial(x, /)
        Find x!.

        Raise a ValueError if x is negative or non-integral.

    floor(x, /)
        Return the floor of x as an Integral.

        This is the largest integer <= x.

    fmod(x, y, /)
        Return fmod(x, y), according to platform C.

        x % y may differ.

    frexp(x, /)
        Return the mantissa and exponent of x, as pair (m, e).

        m is a float and e is an int, such that x = m * 2.**e.
        If x is 0, m and e are both 0.  Else 0.5 <= abs(m) < 1.0.

    fsum(seq, /)
        Return an accurate floating point sum of values in the iterable seq.

        Assumes IEEE-754 floating point arithmetic.

    gamma(x, /)
        Gamma function at x.

    gcd(*integers)
        Greatest Common Divisor.

    hypot(...)
        hypot(*coordinates) -> value

        Multidimensional Euclidean distance from the origin to a point.

        Roughly equivalent to:
            sqrt(sum(x**2 for x in coordinates))

        For a two dimensional point (x, y), gives the hypotenuse
        using the Pythagorean theorem:  sqrt(x*x + y*y).

        For example, the hypotenuse of a 3/4/5 right triangle is:

            >>> hypot(3.0, 4.0)
            5.0

    isclose(a, b, *, rel_tol=1e-09, abs_tol=0.0)
        Determine whether two floating point numbers are close in value.

          rel_tol
            maximum difference for being considered "close", relative to the
            magnitude of the input values
          abs_tol
            maximum difference for being considered "close", regardless of the
            magnitude of the input values

        Return True if a is close in value to b, and False otherwise.

        For the values to be considered close, the difference between them
        must be smaller than at least one of the tolerances.

        -inf, inf and NaN behave similarly to the IEEE 754 Standard.  That
        is, NaN is not close to anything, even itself.  inf and -inf are
        only close to themselves.

    isfinite(x, /)
        Return True if x is neither an infinity nor a NaN, and False otherwise.

    isinf(x, /)
        Return True if x is a positive or negative infinity, and False otherwise.

    isnan(x, /)
        Return True if x is a NaN (not a number), and False otherwise.

    isqrt(n, /)
        Return the integer part of the square root of the input.

    lcm(*integers)
        Least Common Multiple.

    ldexp(x, i, /)
        Return x * (2**i).

        This is essentially the inverse of frexp().

    lgamma(x, /)
        Natural logarithm of absolute value of Gamma function at x.

    log(...)
        log(x, [base=math.e])
        Return the logarithm of x to the given base.

        If the base not specified, returns the natural logarithm (base e) of x.

    log10(x, /)
        Return the base 10 logarithm of x.

    log1p(x, /)
        Return the natural logarithm of 1+x (base e).

        The result is computed in a way which is accurate for x near zero.

    log2(x, /)
        Return the base 2 logarithm of x.

    modf(x, /)
        Return the fractional and integer parts of x.

        Both results carry the sign of x and are floats.

    nextafter(x, y, /)
        Return the next floating-point value after x towards y.

    perm(n, k=None, /)
        Number of ways to choose k items from n items without repetition and with order.

        Evaluates to n! / (n - k)! when k <= n and evaluates
        to zero when k > n.

        If k is not specified or is None, then k defaults to n
        and the function returns n!.

        Raises TypeError if either of the arguments are not integers.
        Raises ValueError if either of the arguments are negative.

    pow(x, y, /)
        Return x**y (x to the power of y).

    prod(iterable, /, *, start=1)
        Calculate the product of all the elements in the input iterable.

        The default start value for the product is 1.

        When the iterable is empty, return the start value.  This function is
        intended specifically for use with numeric values and may reject
        non-numeric types.

    radians(x, /)
        Convert angle x from degrees to radians.

    remainder(x, y, /)
        Difference between x and the closest integer multiple of y.

        Return x - n*y where n*y is the closest integer multiple of y.
        In the case where x is exactly halfway between two multiples of
        y, the nearest even value of n is used. The result is always exact.

    sin(x, /)
        Return the sine of x (measured in radians).

    sinh(x, /)
        Return the hyperbolic sine of x.

    sqrt(x, /)
        Return the square root of x.

    tan(x, /)
        Return the tangent of x (measured in radians).

    tanh(x, /)
        Return the hyperbolic tangent of x.

    trunc(x, /)
        Truncates the Real x to the nearest Integral toward 0.

        Uses the __trunc__ magic method.

    ulp(x, /)
        Return the value of the least significant bit of the float x.

DATA
    e = 2.718281828459045
    inf = inf
    nan = nan
    pi = 3.141592653589793
    tau = 6.283185307179586

FILE
    (built-in)
```

```python
#也可以查看函数
help(math.pow)
```

```
Help on built-in function pow in module math:

pow(x, y, /)
    Return x**y (x to the power of y).
```

```python
math.pow(2,5)
```

```
32.0
```

```python
help(math.floor)
```

```
Help on built-in function floor in module math:

floor(x, /)
    Return the floor of x as an Integral.

    This is the largest integer <= x.
```

```python
math.floor(5.9)
```

```
5
```

##### 自定义模块

###### 导入其他文件

```python
#fun.py
def shout():
    print("hello")
shout()
```

在另一个py文件中定义函数,导入文件可以调用另一个文件的函数

```python
#test.py
import fun
fun.shout()
```

输出的结果是:
hello
hello
因为导入的时候会先执行一遍,为了避免fun中主函数的执行,可以添加判断条件

```python
if __name__=='_main_':
    shout()
```

函数执行时有个参数__name__,指代函数名,在本身里面它的值是__main__,其他文件调用时不是,用这个来区分.

另外一种导入方法:直接导入库函数,调用的时候不需文件名.库函数,可以直接用函数名

```python
from fun import shout
shout()
```

```
hello
```

还可以给导入库函数命名,然后用新命名调用

```python
from fun import shout as f
f()
```

```
hello
```

还可以用*导入,调用时也时直接用函数名

```python
from fun import *
shout()
```

```
hello
```

###### 包的概念

python中的包,就是一个文件夹,但是需要一个__init__.py文件来标志成包,且需要将包放在sys.path路径中

```python
import sys
sys.path
```

```
['c:\\Users\\Tenerome\\Desktop\\python',
 'c:\\environment\\python\\python39.zip',
 'c:\\environment\\python\\DLLs',
 'c:\\environment\\python\\lib',
 'c:\\environment\\python',
 '',
 'C:\\Users\\Tenerome\\AppData\\Roaming\\Python\\Python39\\site-packages',
 'C:\\Users\\Tenerome\\AppData\\Roaming\\Python\\Python39\\site-packages\\win32',
 'C:\\Users\\Tenerome\\AppData\\Roaming\\Python\\Python39\\site-packages\\win32\\lib',
 'C:\\Users\\Tenerome\\AppData\\Roaming\\Python\\Python39\\site-packages\\Pythonwin',
 'c:\\environment\\python\\lib\\site-packages']
```

sys.path路径就是拥有__pycache__文件夹的路径

```python
#导入调用
from bao import fun
fun.shout()
```

```
Moximoxi
```

#### 快速入门

![](https://img2022.cnblogs.com/blog/2629720/202210/2629720-20221010125526925-1111997528.png)

##### 1.Python标识符

在 Python 里，标识符有字母、数字、下划线组成。

在 Python 中，所有标识符可以包括英文、数字以及下划线(_)，但不能以数字开头。

Python 中的标识符是区分大小写的。

以下划线开头的标识符是有特殊意义的。以单下划线开头 _foo 的代表不能直接访问的类属性，需通过类提供的接口进行访问，不能用 from xxx import * 而导入；

以双下划线开头的 foo 代表类的私有成员；以双下划线开头和结尾的 foo 代表 Python 里特殊方法专用的标识，如 init__() 代表类的构造函数。

##### 2.Python有五个标准的数据类型

Numbers（数字）  
String（字符串）  
List（列表）  
Tuple（元组）  
Dictionary（字典）  

Python支持四种不同的数字类型：

int（有符号整型）  
long（长整型[也可以代表八进制和十六进制]）  
float（浮点型）  
complex（复数）  

python的字串列表有2种取值顺序:

从左到右索引默认0开始的，最大范围是字符串长度少1  
从右到左索引默认-1开始的，最大范围是字符串开头  

List（列表） 是 Python 中使用最频繁的数据类型。

列表可以完成大多数集合类的数据结构实现。它支持字符，数字，字符串甚至可以包含列表（即嵌套）。  

列表用 [ ] 标识，是 python 最通用的复合数据类型。  

列表中值的切割也可以用到变量 [头下标:尾下标] ，就可以截取相应的列表，从左到右索引默认 0 开始，从右到左索引默认 -1 开始，下标可以为空表示取到头或尾。  

加号 + 是列表连接运算符，星号 * 是重复操作。  

元组是另一个数据类型，类似于List（列表）

。

元组用”()”标识。内部元素用逗号隔开。但是元组不能二次赋值，相当于只读列表。  

字典(dictionary)是除列表以外python之中最灵活的内置数据结构类型。

列表是有序的对象结合，字典是无序的对象集合。两者之间的区别在于：字典当中的元素是通过键来存取的，而不是通过偏移存取。  

字典用”{ }”标识。字典由索引(key)和它对应的值value组成。

##### 3.Python数据类型转换

有时候，我们需要对数据内置的类型进行转换，数据类型的转换，你只需要将数据类型作为函数名即可。

以下几个内置的函数可以执行数据类型之间的转换。这些函数返回一个新的对象，表示转换的值。

int(x [,base]): 将x转换为一个整数

long(x [,base] ): 将x转换为一个长整数

float(x): 将x转换到一个浮点数

complex(real [,imag]): 创建一个复数

str(x):将对象 x 转换为字符串

repr(x): 将对象 x 转换为表达式字符串

eval(str): 用来计算在字符串中的有效Python表达式,并返回一个对象

tuple(s): 将序列 s 转换为一个元组

list(s): 将序列 s 转换为一个列表

set(s): 转换为可变集合

dict(d): 创建一个字典。d 必须是一个序列 (key,value)元组。

frozenset(s): 转换为不可变集合

chr(x) :将一个整数转换为一个字符

unichr(x): 将一个整数转换为Unicode字符

ord(x): 将一个字符转换为它的整数值

hex(x): 将一个整数转换为一个十六进制字符串

oct(x): 将一个整数转换为一个八进制字符串

##### 4.Python 运算符

算术运算符  
比较（关系）运算符  
赋值运算符  
逻辑运算符  
位运算符  
成员运算符  
身份运算符  
运算符优先级  

#### 序列

##### 列表

列表,类似某些语言的Arraylist,但不是数组.用[]表示

定义列表

```python
x=[1,2,3,4,5]
x2=list(range(1,9))
print(x)
print(x2)
```

```
[1, 2, 3, 4, 5]
[1, 2, 3, 4, 5, 6, 7, 8]
```

output:
[1, 2, 3, 4, 5]
[1, 2, 3, 4, 5, 6, 7, 8]

列表可以像数组一样按下标取值,自左向右从0开始,逆序从-1开始

```python
print(x[2])
print(x2[-3])
```

output:
3
6

列表支持切片操作,[a:b],取x,a≤x＜b

```python
x[1:4]
```

output:
[2, 3, 4]

列表元素可以是不同类型,还可以嵌套子列表

```python
x3=[1,'a',True,[1,2,3]]
x3[3]
```

output:[1, 2, 3]

访问子列表的元素

```python
x3[3][1]
```

output:2

追加列表的几种方式

```python
a=[1,2,3]+[4,5]
print(a)
a.append(6)
print(a)
a.extend([100,200])
print(a)
```

output:[1, 2, 3, 4, 5]
[1, 2, 3, 4, 5, 6]
[1, 2, 3, 4, 5, 6, 100, 200]

插入列表:可以按下表插入,也可以逆序

```python
a.insert(0,'c')
print(a)
a.insert(-2,'mc')
print(a)
```

output:
['c', 1, 2, 3, 4, 5, 6, 100, 200]
['c', 1, 2, 3, 4, 5, 6, 'mc', 100, 200]

列表可以直接修改某个元素,按下标

```python
a[0]=300
print(a)
```

output:[300, 1, 2, 3, 4, 5, 6, 'mc', 100, 200]

如果想切片修改,比如把a0,a1修改成4,不能直接赋值4,会报错:can only assign an iterable,只能分配一个迭代器.所以要用列表赋值

```python
# a[0:2]=4  会报错
a[0:2]=[4]
print(a)
```

output: [4, 2, 3, 4, 5, 6, 'mc', 100, 200]

删除操作
del :支持删除指定下标元素,支持切片,删除是真正删除了这个元素

```python
a1=[1,2,3,4,'a','c']
del a1[2]
print(a1)
del a1[0:2]
print(a1)
```

output: [1, 2, 4, 'a', 'c']
[4, 'a', 'c']

pop():出栈,没有参数时,默认出栈栈顶元素,可以加参数指定位置,但不支持切片.相比于del,它有返回值,出栈时返回出栈元素

```python
a2=[1,2,3,4,5,6,7]
a2.pop()
print(a2)
b=a2.pop(0)
print(b,a2)
```

output:[1, 2, 3, 4, 5, 6]
1 [2, 3, 4, 5, 6]

列表支持*操作,重复元素

```python
a2=a2 * 3
print(a2)
```

output: [2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6]

按值删除,remove(value)函数,参数value会在列表中查找等于value的第一个值,并删除之

```python
a2.remove(3)
print(a2)
```

output:[2, 4, 5, 6, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6]

清空列表:clear()

```python
a2.clear()
print(a2)
```

output: []

列表排序

```python
x=[2,5,3,100,45,77]
sorted(x)
```

output: [2, 3, 5, 45, 77, 100]

sorted()只是排序一下并输出,列表x本身次序并未改变,可以用列表排序函数

```python
x.sort()
print(x)
```

output: [2, 3, 5, 45, 77, 100]

列表推导式/列表解析式

①x=[1,2,3,4,5,6,7,8,9,10],求每个元素平方后输出

```python
#如果使用for循环<br>
x=[1,2,3,4,5,6,7,8,9,10]
for i in x: 
    print(i**2)
```

使用列表推导式

```python
[i ** 2 for i in x]
```

output: [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]

还可以继续添加条件,循环嵌套

```python
[i ** 2 for i in x if i%2 ==0]
[(i,j) for i in range(2) for j in range(3)]
```

output: [4, 16, 36, 64, 100]
[(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

②求两个列表对应位置元素的和,x从1到100,y为100个1

```python
#不用列表推导式
x=range(1,101)
y=[1]*100
z=list()
for i in range(100):
    z.append(x[i]+y[i])
print(z)
```

列表推导式

```python
[x[i] + y[i] for i in range(100)]
#或者
print([i+j for i,j in zip(x,y)],end=" ")
```

output: [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,  
18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,  
34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,  
51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66,  
67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83,  
84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101]

class zip(object)
 | zip(*iterables) --> A zip object yielding tuples until an input is exhausted.

| >>> list(zip('abcdefg', range(3), range(4)))
 | [('a', 0, 0), ('b', 1, 1), ('c', 2, 2)]
 | The zip object yields n-length tuples, where n is the number of iterables passed as positional arguments to zip(). The i-th element in every tuple comes from the i-th iterable argument to zip(). This continues until the shortest argument is exhausted.

使用向量表达式

```python
import numpy as np
x=np.array(x)
y=np.array(y)
z=x+y
print(z)
```

使用映射map()

```python
list(map(lambda x:x[0]+x[1],zip(x,y)))
```

##### 元组tuple

元组是不可变类型,比支持删除和添加操作

元组用()表示,可以用来作为函数多返回值类型

```python
def f():
    a=1
    b=2
    return a, b
x,y=f()
print(x,y)
```

```
1 2
```

查看返回值类型

```python
t=f()
print(type(t))
```

```
<class 'tuple'>
```

python 支持同时多赋值

```python
x,y,z=1,2,3
```

也可以使用元组

```python
x,y,z=(1,2,3)
```

如果元组中只有一个元素,要用','标志

```python
x=(1)
print(type(x))
x=(1,)
print(type(x))
```

```
<class 'int'>
<class 'tuple'>
```

元组也支持按下标取值还有切片

```python
x=(1,2,3,4,5)
print(x[1])
print(x[0:2])
```

```
2
(1, 2)
```

##### 集合set

用{}表示,不支持按下标取值,不支持切片

集合最大的特点:集合中的元素不能重复,类似数学中的集合的概念,也有交集,并集等性质

```python
x={1,2,2,3,4,4,5,5,5,6}
print(x)
```

```
{1, 2, 3, 4, 5, 6}
```

所以可以利用集合去重其他类型

```python
l=[1,2,3,4,4,4,5,5,6,6]
s=set(l)
print(s)
```

```
{1, 2, 3, 4, 5, 6}
```

交集

```python
a={1,2,3,4}
b={3,4,5}
a & b
```

```
{3, 4}
```

并集

```python
a | b
```

```
{1, 2, 3, 4, 5}
```

异或

```python
a ^ b
```

```
{1, 2, 5}
```

##### 字典dict

字典也用{}表示,区别于集合,它在初始化时,用键值对表示

```python
dic={'a':100,'b':200,'c':300}
```

添加update,添加的键值对用{}

```python
dic.update({'d':400})
print(dic)
```

```
{'a': 100, 'b': 200, 'c': 300, 'd': 400}
```

python中的字典类似与其他语言的hash表,按键取值

```python
dic['d']
```

```
400
```

如果无对应键,会报异常

```python
dic['e']
```

```
---------------------------------------------------------------------------

KeyError                                  Traceback (most recent call last)

c:\Users\Tenerome\Desktop\python\lesson.ipynb Cell 35 in <cell line: 1>()
----> <a href='vscode-notebook-cell:/c%3A/Users/Tenerome/Desktop/python/lesson.ipynb#Y220sZmlsZQ%3D%3D?line=0'>1</a> dic['e']


KeyError: 'e'
```

可以用get()方法try to get value

```python
dic.get('e','none')
```

```
'none'
```

get()有两个参数,第一个参数为键,若字典有对应键,返回键值.若无对用键,返回第二个参数

遍历

```python
for i in dic:
    print(i,end=" ")
```

```
a b c d 
```

直接对dic遍历返回的是键(还可以用dic.keys()),遍历键值,使用dic.values()

```python
for i in dic.values():
    print(i,end=" ")
```

```
100 200 300 400 
```

输出键值对

```python
print(dic)
print(dic.items())
```

```
{'a': 100, 'b': 200, 'c': 300, 'd': 400}
dict_items([('a', 100), ('b', 200), ('c', 300), ('d', 400)])
```

区别:dic是字符串,dic.items()是元组

```python
for i in dic:
    print(type(i),end=" ")
print()
for i in dic.items():
    print(type(i),end=" ")
```

```
<class 'str'> <class 'str'> <class 'str'> <class 'str'> 
<class 'tuple'> <class 'tuple'> <class 'tuple'> <class 'tuple'> 
```

字典的键必须是不可变类型:数字(numbers),字符串(string),元组(tuple)

```python
dic1={(1,2):'苹果','a':'橘子',3:'香蕉'}
print(dic1)
```

```
{(1, 2): '苹果', 'a': '橘子', 3: '香蕉'}
```

字典的键不能重复,其他语言中hash键重复会报异常,python虽然不报异常,但重复的键会覆盖之前的键

```python
dic2={'a':100,'b':200,'a':300}
print(dic2)
```

```
{'a': 300, 'b': 200}
```

#### 程序流程控制

选择结构

①编写程序,判断某个日期是该年的第几天

```python
import time
def demo(year,month,day):
    day_month=[31,28,30,31,30,31,31,30,31,30,31]#十二个月的天数
    pass#pass关键字用来做占位符,表示待实现
print((demo(2021,1,15)))
print(demo(2021,3,26))
```

```python
import time
def demo(year,month,day):
    day_month=[31,28,30,31,30,31,31,30,31,30,31]
    if year % 400 ==0 or (year %4==0 and year%100!=0):
        day_month[1]=29
    # if month==1:
    #     return day
    # else:
    return sum(day_month[:month-1],day)
```

```python
print(demo(2021,3,26))
```

```
85
```

获取现在的时间,并调用demo()

```python
date=time.localtime()
print(type(date))
print(date)
print(date[:3])
```

```
<class 'time.struct_time'>
time.struct_time(tm_year=2022, tm_mon=9, tm_mday=22, tm_hour=14, tm_min=45, tm_sec=39, tm_wday=3, tm_yday=265, tm_isdst=0)
(2022, 9, 22)
```

time.localtime()获取当前的时间,返回值为一个struct

struct切片后返回的是一个元组

第一种调用方法:赋值给三个变量

```python
year,month,day=date[:3]
demo(year,month,day)
```

```
264
```

第二种:使用序列解包方式传参

```python
demo(*date[:3])
```

```
264
```

year % 100 !=0 可以简写为year % 100.因为在python条件判断时,只要表达式不为0,False,None,空(序列)时判断结果都是True

三元表达式

```python
w=100
h=1 if w>=100 else 2
print(h)
```

```
1
```

python中没有Switch结构,C语言中的Switch其实使用hash表实现跳转的,在python中可以用字典代替

```python
def func1():
    pass
def func2():
    pass
def func3():
    pass
d={'key1':func1(),'key2':func2(),'key3':fun3()}
```

##### 循环结构

编写程序,打印100以内的所有素数

仿C结构

```python
for i in range(2,101):
    flag=True
    for j in range(2,int(i**0.5)+1):
        if i%j ==0:
            flag=False
            break
    if flag:
        print(i,end=" ")
```

```
2 3 5 7 11 13 17 19 23 29 31 37 41 43 47 53 59 61 67 71 73 79 83 89 97 
```

python提供了for...else结构.若内循环正常结束(而不是遇到break跳出)或没有执行for循环,则执行else语句.如果是由break语句跳出的,则不执行else语句

用for...esle结构

```python
for i in range(2,101):
    for j in range(2,int(i**0.5)+1):
        if i%j ==0:
            break
    else:
        print(i,end=" ")
```

```
2 3 5 7 11 13 17 19 23 29 31 37 41 43 47 53 59 61 67 71 73 79 83 89 97 
```

使用列表推导式

```python
print([i for i in range(2,101) if 0 not in [i%j for j in range(2,int(i**0.5)+1)]])
```

```
[2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
```

##### 异常处理

③创建txt文本文件,内容三行文字

```python
f=open('2.txt',encoding='utf-8')
```

```
---------------------------------------------------------------------------

FileNotFoundError                         Traceback (most recent call last)

c:\Users\Tenerome\Desktop\python\lesson.ipynb Cell 31 in <cell line: 1>()
----> <a href='vscode-notebook-cell:/c%3A/Users/Tenerome/Desktop/python/lesson.ipynb#X42sZmlsZQ%3D%3D?line=0'>1</a> f=open('2.txt',encoding='utf-8')


FileNotFoundError: [Errno 2] No such file or directory: '2.txt'
```

如果没有创建,就会报出异常

对可能产生异常的程序,可以用try...except Exception...else...finally

```python
try:
    f=open('test.txt',encoding='utf-8')
except Exception:
    print('文件不存在')
else:
    print(f.read())
finally:
    f.close()
```

```
Hi
Hello
CCVV
```

try:尝试运行后面的程序
如果有异常,except Exception捕获,并做个性化处理
若无异常,esle
不管有没有异常,finally

④断言语句assert,用户输入一个数字,判断是否在20-120之间,否则产生异常,并提示用户重新输入

```python
while True:
    x=int(input('请输入一个在20-120之间的数字'))
    try:
        assert x>=20 and x<=120
    except AssertionError:
        print('输入数字不合法')
    else:
        print('OK')
        break
```

```
输入数字不合法
输入数字不合法
OK
```

#### 函数与函数式编程

##### 函数的定义与调用

返回斐波那契数列中小于整数n的所有值

```python
def fib(n):
    pass
fib(1000)
```

python中的函数,用def关键字定义.函数不需要返回值类型,不需要参数类型

##### 函数的参数

函数参数的集中类型:  
位置参数(positional arguments):形参和实参数量相同,位置一致.  
缺省参数(default arguments):形参有默认值,有缺省参数的形参要在位置参数之后.  
关键字参数(keyword arguments):实参顺序可以和形参顺序不一致,避免了用户需要牢记位置参数顺序的麻烦  
可变参数(variable-length arguments):*parameter用来接受多个实参并将其放在元组中.**parameter接收多个关键参数并存放到字典中

位置参数:

```python
def func(a,b,c):
    print(a,b,c)
func(1,2)
```

```
---------------------------------------------------------------------------

TypeError                                 Traceback (most recent call last)

c:\Users\Tenerome\Desktop\python\lesson.ipynb Cell 45 in <cell line: 3>()
      <a href='vscode-notebook-cell:/c%3A/Users/Tenerome/Desktop/python/lesson.ipynb#X63sZmlsZQ%3D%3D?line=0'>1</a> def func(a,b,c):
      <a href='vscode-notebook-cell:/c%3A/Users/Tenerome/Desktop/python/lesson.ipynb#X63sZmlsZQ%3D%3D?line=1'>2</a>     print(a,b,c)
----> <a href='vscode-notebook-cell:/c%3A/Users/Tenerome/Desktop/python/lesson.ipynb#X63sZmlsZQ%3D%3D?line=2'>3</a> func(1,2)


TypeError: func() missing 1 required positional argument: 'c'
```

这个就是positional 参数,缺少时报错

缺省参数:

```python
def func1(a,b,c=1):
    print(a,b,c)
func1(2,3)
```

```
2 3 1
```

定义函数时就对形参赋值,就是缺省参数,后面传实参会覆盖

关键字参数:

```python
def func2(a,b,c):
    print(a,b,c)
func2(c=3,a=1,b=2)
```

```
1 2 3
```

可以看到调用的时候没有按形参顺序来,但是用关键字(形参名)来调用的,就可以忽略顺序

```python
def stu(country='中国',name):
    print("%s,%s" %(name,country))
stu('美国','Tom')
```

```
  Input In [7]
    def stu(country='中国',name):
                             ^
SyntaxError: non-default argument follows default argument
```

有缺省值的参数必须放位置参数后面

```python
def stu(name,country='中国'):
    print("%s ,%s" %(name,country))
stu('美国','Tom')
```

```
美国 ,Tom
```

可变参数:

##### 变量的作用域

函数体外部的变量:全局变量  
内部:局部变量

函数内可以调用全局变量

```python
x=100
def f():
    print(x)
f()
```

```
100
```

在函数内部的变量赋值操作,该变量就是局部变量.

```python
x=100
def f():
    print(x)
    x=200
    print(x)
f()
```

```
---------------------------------------------------------------------------

UnboundLocalError                         Traceback (most recent call last)

c:\Users\Tenerome\Desktop\python\lesson.ipynb Cell 63 in <cell line: 6>()
      <a href='vscode-notebook-cell:/c%3A/Users/Tenerome/Desktop/python/lesson.ipynb#Y116sZmlsZQ%3D%3D?line=3'>4</a>     x=200
      <a href='vscode-notebook-cell:/c%3A/Users/Tenerome/Desktop/python/lesson.ipynb#Y116sZmlsZQ%3D%3D?line=4'>5</a>     print(x)
----> <a href='vscode-notebook-cell:/c%3A/Users/Tenerome/Desktop/python/lesson.ipynb#Y116sZmlsZQ%3D%3D?line=5'>6</a> f()


c:\Users\Tenerome\Desktop\python\lesson.ipynb Cell 63 in f()
      <a href='vscode-notebook-cell:/c%3A/Users/Tenerome/Desktop/python/lesson.ipynb#Y116sZmlsZQ%3D%3D?line=1'>2</a> def f():
----> <a href='vscode-notebook-cell:/c%3A/Users/Tenerome/Desktop/python/lesson.ipynb#Y116sZmlsZQ%3D%3D?line=2'>3</a>     print(x)
      <a href='vscode-notebook-cell:/c%3A/Users/Tenerome/Desktop/python/lesson.ipynb#Y116sZmlsZQ%3D%3D?line=3'>4</a>     x=200
      <a href='vscode-notebook-cell:/c%3A/Users/Tenerome/Desktop/python/lesson.ipynb#Y116sZmlsZQ%3D%3D?line=4'>5</a>     print(x)


UnboundLocalError: local variable 'x' referenced before assignment
```

也就是说,在函数内,一旦引用了全局变量,就不能在函数内对全局变量赋值

但是可以先赋值再引用,这时两个x并不是同一个,相当于两个变量,一个全局,一个局部,但是有相同的变量名

```python
x=100
def f():
    x=200
    print(id(x))
f()
id(x)
```

```
2413391733392
2413391730128
```

想要在函数内真正调用全局变量,可以用global声明

```python
x=100
def f():
    global x
    x=200
    print(id(x))
f()
id(x)
```

```
2413391733392
2413391733392
```

python支持使用nonlocal关键字定义一宗介于全局和局部之间的变量,成为闭包作用域变量.关键字nonlocal声明的变量会引用距离最近的非全局作用域的变量(闭包变量或局部变量).要求声明的变量已经存在.关键字nonlocal不会创建新变量

##### lambda表达式

```python
def fsum(x,y):
    return x+y
```

像上面的函数,只是很简单的功能,就定义了函数.lambda表达式就是用来代替简单的函数的

基本结构lambda left : right.其中left是参数,right是返回值.

```python
f=lambda x,y : x+y
f(3,5)
```

```
8
```

lambda表达式就是把简单的函数改成无函数体的形式,只有参数和返回值.返回值用表达式的形式.只要能改成这样的结构的函数,都可以写成lambda表达式的形式

##### 函数式编程

函数式编程就是把函数作用于一个或多个序列,来实现一些功能

① 内置函数map()可以将一个函数作用到一个或多个序列或迭代器对象上,返回可迭代的map对象

![](https://img2022.cnblogs.com/blog/2629720/202210/2629720-20221010125528334-644694626.png)

map()会把seq中的元素一个个作用在func()上,并将返回值组成一个新的可迭代对象mapped_seq

结构:map(func, *iterables) --> map object
第一个参数是函数名,也可以是lambda表达式,第二个参数是一个可迭代的对象,返回值为map object

```python
import math
map(lambda x:math.pow(x,2),[1,2,3,4,5,6,7])
```

```
<map at 0x231ffcec3a0>
```

返回结果是map object 是可迭代的

```python
list(map(lambda x:math.pow(x,2),[1,2,3,4,5,6,7]))
```

```
[1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0]
```

②标准库functools中的reduce(function,sequence)函数可以接收一个有两个参数的函数function(),以迭代的方式从左到右依次作用到一个序列sequence上实现类huffman树的聚集操作(求和,求连乘积等)

Huffman树聚集 

![pic](https://img2022.cnblogs.com/blog/2629720/202210/2629720-20221010125528932-607149729.png)

```python
from functools import reduce
reduce(lambda i,j:i+j,range(1,101))
```

```
5050
```

③内置函数filter将一个函数作用到一个序列上,返回该序列中式函数返回值为True的元素组成的filter对象,filter对象也是可迭代的 

![](https://img2022.cnblogs.com/blog/2629720/202210/2629720-20221010125529454-1653390707.png)

如:输出从1到1000所有的素数

```python
import math
def is_prime(n):
    if n<2:
        return False
    if n==2:
        return True
    if n%2==0:
        return False
    sqrt_n=math.floor(math.sqrt(n))
    for i in range(3,sqrt_n+1,2):
        if n%i==0:
            return False
    return True
```

```python
print(list(filter(is_prime,range(1,101))),end=" ")
```

```
[2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97] 
```

④内置函数zip

zip函数像一个拉链,把两个序列连接返回一个zip对象,可迭代.其元素是两个序列元素组成的元组 

![](https://img2022.cnblogs.com/blog/2629720/202210/2629720-20221010125530005-566645114.png)

```python
list(zip(range(10),'abcdefghij'))
```

```
[(0, 'a'),
 (1, 'b'),
 (2, 'c'),
 (3, 'd'),
 (4, 'e'),
 (5, 'f'),
 (6, 'g'),
 (7, 'h'),
 (8, 'i'),
 (9, 'j')]
```

#### 面对对象设计

##### 类的定义和实例化

①结构

class 类名:
    类成员

关键字为class,类名首字母要大写,如果是自定义类的派生类,则需要用类名+(object)形式.类中的方法和属性不需要使用Pascal法则命名,即首字母不用大写

*例:定义Sheep类,内部定义一个name变量,一个shout方法,实例化一个对象sheep,调用shout方法,喊出它的name*

```python
class Sheep:
    name='Shown'
    def shout(name):
        print('My name is:',name)

sheep=Sheep()
sheep.shout()
```

```
My name is: <__main__.Sheep object at 0x000002715D415640>
```

运行提示<__main__.Sheep object>很明显这不是想传进去的name,而是实例化的sheep对象

```python
print(sheep)
```

```
<__main__.Sheep object at 0x000002713C772BE0>
```

类中的方法,分类方法和实例方法.类方法作用于类,实例方法作用域实例.python中的实例方法必须用self来做第一个参数

调用类中的全局变量时用self.变量名

```python
class Sheep:
    name='Shown'
    def shout(self):
        print('My name is:',self.name)

sheep=Sheep()
sheep.shout()
```

```
My name is: Shown
```

也可以用类名调用

```python
class Sheep:
    name='Shown'
    def shout(self):
        print('My name is:',Sheep.name)#用Sheep.name
    def run(self):
        self.shout()
        print('I am running')
sheep=Sheep()
sheep.run()
```

```
My name is: Shown
I am running
```

同样的,类中的实例方法互相调用时也需要用self.方法名,但是不能用类名

```python
class Sheep:
    name='Shown'
    def shout(self):
        print('My name is:',self.name)
    def run(self):
        self.shout()
        print('I am running')
sheep=Sheep()
sheep.run()
```

```
My name is: Shown
I am running
```

如果想在实例化的时候对对象赋初值,则会用到构造方法.python中的构造方法叫做初始化方法,用__init__表示.python中的类,通过__new__实例化,通过__init__来初始化

```python
print(dir(sheep),end=" ")
```

```
['__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', 'name', 'run', 'shout'] 
```

类中带双下划线的方法或属性如__new__叫做魔法方法/属性,它是类中的已经定义好的方法,一般有类自己调用,也可以通过编程调用或者重写.后面的name,run,shout才是自已定义的方法/属性

重写__init__,添加参数,用来实例化的时候赋初值

```python
class Sheep:
    def __init__(self,name):
        self.name=name
    def shout(self):
        print('my name is:',self.name)
sheep=Sheep('Tom')
sheep.shout()
```

```
my name is: Tom
```

这里的self相当于实例后的独有的空间  
self{  
    name;  
    shout()  
}
self和类class的关系就相当于 

![pic](https://img2022.cnblogs.com/blog/2629720/202210/2629720-20221010125530455-35739422.png)

```python
class Panda:
    attr1='Pandas'
    def func1():
        pass
    def func2(self):
        self.attr2='cats'
    def func3(self):
        print(self.attr2)

        a=3
        print(a)
        pass
pan=Panda()
print('class:',[i for i in dir(Panda) if i[0]!='_'])
print('self:',[i for i in dir(pan) if i[0]!='_'])
```

```
class: ['attr1', 'func1', 'func2', 'func3']
self: ['attr1', 'func1', 'func2', 'func3']
```

可以看到class和self共有attr1,func1,func2,func3,也就是说,在没有调用实例化方法时,self中的属性并不会创建

```python
pan.func2()
print('class:',[i for i in dir(Panda) if i[0]!='_'])
print('self:',[i for i in dir(pan) if i[0]!='_'])
```

```
class: ['attr1', 'func1', 'func2', 'func3']
self: ['attr1', 'attr2', 'func1', 'func2', 'func3']
```

调用pan.func2()后,self区创建了attr2

```python
pan.func3()
print('class:',[i for i in dir(Panda) if i[0]!='_'])
print('self:',[i for i in dir(pan) if i[0]!='_'])
```

```
cats
3
class: ['attr1', 'func1', 'func2', 'func3']
self: ['attr1', 'attr2', 'func1', 'func2', 'func3']
```

可以看到,虽然attr2是在func2()中创建的,但在func3()中仍能调用.但是func3()中的局部变量3并不在pan中.

但是不同的实例之间并不能共用self区

```python
class Panda:
    attr1='Pandas'
    def func1():
        pass
    def func2(self):
        self.attr2='cats'
    def func3(self):
        print(self.attr2)

        a=3
        print(a)
        pass
pan=Panda()
pan1=Panda()
pan.func2()
pan1.func2()
#pan修改attr2不影响pan1
pan.attr2='dogs'
print('pan1.attr2:',pan1.attr2)
```

```
pan1.attr2: cats
```

所以不同实例之间的关系是这样的 

![](https://img2022.cnblogs.com/blog/2629720/202210/2629720-20221010125530893-1186474302.png)

如果形参,self变量,类变量名字相同就这样区分

```python
class Panda:
    name='Aa'
    def func(self):
        self.name='Bb'
        name='Cc'
        print(Panda.name)
        print(self.name)
        print(name)
pan=Panda()
pan.func()
print('class:',[i for i in dir(Panda) if i[0]!='_'])
print('self:',[i for i in dir(pan) if i[0]!='_'])
```

```
Aa
Bb
Cc
class: ['func', 'name']
self: ['func', 'name']
```

可以看到三个name值不同.而class和self虽然在内部空间存在相同的属性名name,但他们是两个值

再继续看实例调用类变量的情况

```python
class Panda:
    name='Aa'
    def func(self):
        self.name='Bb'
        name='Cc'
pan=Panda()
print(pan.name)
```

```
Aa
```

未调用func(),实例可以访问类变量

```python
pan.func()
print(pan.name)
```

```
Bb
```

调用func()后,self区的name就被覆盖了

但如果是不同的变量名

```python
class Panda:
    cname='Aa'
    def func(self):
        self.name='Bb'
        name='Cc'
pan=Panda()
print(pan.cname)
pan.func()
print(pan.cname)
print('self:',[i for i in dir(pan) if i[0]!='_'])
```

```
Aa
Aa
self: ['cname', 'func', 'name']
```

就不会覆盖,self此时既有cname,又有name.所以如果类属性和实例属性名字相同,调用函数后,就不能再通过实例访问类属性

sum up:  
类中的变量分类变量(class),实例变量(self)和局部变量  
类变量直接定义在类中,可以被类名和实例化的对象调用  
实例变量定义在方法中,且由self修饰,只有在调用方法时才创建,能被所有实例调用,不能通过类名访问  
如果类变量和实例变量名相同,实例调用函数后,会覆盖  
局部变量也定义在函数中,使用后销毁,并不占用实例空间

python中的self就是java中的this,唯一区别就是self必须是实例方法的参数的第一个.其实,self只是规定了第一个参数的位置,命名什么都行,也可以用this

```python
class Sheep:
    def __init__(this,name):
        this.name=name
    def shout(this):
        print('my name is:',this.name)
sheep=Sheep('Tom')
sheep.shout()
```

```
my name is: Tom
```

②python类型是动态性的,可以动态为自定义类及对象改变为新的属性和行为,俗称混入(mixin)机制

```python
a=1
print(type(a))
a='hello'
print(type(a))
```

```
<class 'int'>
<class 'str'>
```

但是实际a的引用地址已经发生了改变

```python
a=1
print(id(a))
a='hello'
print(id(a))
```

```
2734448994608
2734528578672
```

利用这个机制,就可以给已经创建好实例,在不修改类,不重新实例化的前提下添加新属性

```python
class Sheep:
    name=''
    def shout():
        pass
shp=Sheep()
print('shp:',[i for i in dir(shp) if i[0]!='_'])
shp.age=12
print('shp:',[i for i in dir(shp) if i[0]!='_'])
```

```
shp: ['name', 'shout']
shp: ['age', 'name', 'shout']
```

而且也可以给类添加新属性

```python
Sheep.type='sheep'
print('Sheep:',[i for i in dir(Sheep) if i[0]!='_'])
```

```
Sheep: ['name', 'shout', 'type']
```

并且,再新的实例也会有type属性

```python
shp1=Sheep()
print('shp2:',[i for i in dir(shp1) if i[0]!='_'])
```

```
shp2: ['name', 'shout', 'type']
```

还有方法也可以添加,但是需要导入type库

```python
import types
def setSpeed(self,s):
    self.speed=s
shp1.setSpeed=types.MethodType(setSpeed,shp1)
#绑定后,实例不但有了setSpeed方法,而且有了speed属性
shp1.setSpeed(5)
print(shp1.speed)
```

```
5
```

③私有不私有特点:

*XXX:一个下划线开始,表示受保护的成员,不能用from .. import导入  
__XXX_\*:系统定义的特殊成员,如__init__  
__xxx:私有成员,只有类对象自己能访问,子对象不能直接访问.但在对象外部可以通过'对象名._类名__xxx'来访问,所以python中的私有并不是真正的私有

##### 方法

python中的方法分为:实例方法,类方法,静态方法

实例方法第一个参数名为self  
类方法第一个形参为cls  
静态方法没有规定必须的参数

类方法用@classmethod修饰,静态方法用@staticmethod修饰

python中的静态方法不能使用类或实例中的任何属性和方法,相当于一个独立的空间,常作为工具方法使用.python中的类方法更类似于其他语言的静态方法.

共同点:  
类方法和静态方法不能直接访问属于对象的成员  
类方法和静态方法都可以用对象或类名来调用

```python
class Cat:
    @classmethod
    def setName(cls,name='Tom'):
        cls.name=name
cat=Cat()
cat1=Cat()
cat.setName('Jerry')
print(cat1.name)
print (Cat.name)
```

```
Jerry
Jerry
```

如上,python中的类方法的不同实例是共用空间的

python的静态方法是一个独立的空间,一般用于和类对象以及实例对象无关的代码。

```python
class Game():
    @staticmethod
    def menu():
        print('开始游戏')
        print('设置')
        print('退出游戏')
    def func1():
        pass
game=Game()
game.menu()
```

```
开始游戏
设置
退出游戏
```

![](https://img2022.cnblogs.com/blog/2629720/202210/2629720-20221010125531367-2087856156.png)

##### 属性

```python
class Sheep:
    def __init__(self,name):
        self.__name=name
    def name(self):
        return self.__name
sheep=Sheep('Shown')
print(sheep.name())
print(sheep.__name)
```

```
Shown



---------------------------------------------------------------------------

AttributeError                            Traceback (most recent call last)

c:\Users\Tenerome\Desktop\python\lesson.ipynb Cell 73 in <cell line: 8>()
      <a href='vscode-notebook-cell:/c%3A/Users/Tenerome/Desktop/python/lesson.ipynb#Y134sZmlsZQ%3D%3D?line=5'>6</a> sheep=Sheep('Shown')
      <a href='vscode-notebook-cell:/c%3A/Users/Tenerome/Desktop/python/lesson.ipynb#Y134sZmlsZQ%3D%3D?line=6'>7</a> print(sheep.name())
----> <a href='vscode-notebook-cell:/c%3A/Users/Tenerome/Desktop/python/lesson.ipynb#Y134sZmlsZQ%3D%3D?line=7'>8</a> print(sheep.__name)


AttributeError: 'Sheep' object has no attribute '__name'
```

想得到sefl.__name私有变量,只能通过函数name的返回值(或用类名+私有变量名),python中的@property装饰器,用它装饰的方法可以对类的私有变量进行读写操作,将来可以通过方法名,不需要加()来对属性读写

```python
class Sheep:
    def __init__(self,name):
        self.__name=name
    @property
    def name(self):
        return self.__name
sheep=Sheep('Shown')
print(sheep.name)
```

```
Shown
```

上面是只读的例子,下面看可读可写的

```python
class Sheep:
    def __init__(self,name):
        self.__name=name
    def __get(self):
        return self.__name
    def __set(self,value):
        self.__name=value
    name=property(__get,__set)
sheep=Sheep('Shown')
print(sheep.name)
sheep.name='Tom'
print(sheep.name)
```

```
Shown
Tom
```

上面是第一种形式,除了getter和setter,还有del

```python
'''class C(object):
    def getx(self): return self._x
    def setx(self, value): self._x = value
    def delx(self): del self._x  
    x = property(getx, setx, delx, "I'm the 'x' property.")'''
```

如果不定义del的话是不能直接删除的

```python
class Sheep:
    def __init__(self,name):
        self.__name=name
    def __get(self):
        return self.__name
    def __set(self,value):
        self.__name=value
    def __del(self):
        del self.__name
    name=property(__get,__set,__del)
```

另一种形式:

```python
'''class C(object):
    @property def x(self):
        "I am the 'x' property." return self._x
    @x.setter def x(self, value):
        self._x = value
    @x.deleter def x(self):
        del self._x'''
```

```python
class Sheep:
    @property 
    def name(self):
        return self.__name
    @name.setter
    def name(self,name):
        self.__name=name
    @name.deleter
    def name(self):
        del self.__name
sheep=Sheep()
sheep.name='Shawn'
print(sheep.name)
# del sheep.name
# print(sheep.name)
```

```
Shawn
```

##### 析构方法

实例化出来的对象都是有声明周期的,当声明周期结束时,会自动调用析构函数,默认无动作.可以在析构函数中自定义语句.析构函数用__func__表示

```python
#在jupyter 中会直接结束对象的声明周期,在py文件中运行这段代码
class Dog:
    def __eat(self):
        print('i am going to eat')
    def can_bite(self,condition):
        if condition==1:
            self.__eat()
        else:
            return "I can't eat"
    def __del__(self):
        print('Over')
dog =Dog()
con=int(input('input one number:'))
dog.can_bite(con)
del(dog)
```

```
i am going to eat
Over
```

##### 继承和多态

①继承:派生类继承基类的属性和方法.

python中的派生类用class 派生类名(基类名):来定义

```python
class A:
    def hello(self):
        print('hello')
class B(A):
    pass
b=B()
b.hello()
```

```
hello
```

如果要在派生类中调用基类的方法,可以使用内置函数super().方法名或者通过基类名.方法名()来调用.私有方法在派生类中不能直接访问

```python
class A:
    def __init__(self):
        __name='shawn'
        self.age=12
    def hello(self):
        print('hello')
class B(A):
    def hi(self):
        super().hello()
b=B()
b.hi()
print(b.age)
print(b.__name)
```

```
hello
12



---------------------------------------------------------------------------

AttributeError                            Traceback (most recent call last)

c:\Users\Tenerome\Desktop\python\lesson.ipynb Cell 93 in <cell line: 13>()
     <a href='vscode-notebook-cell:/c%3A/Users/Tenerome/Desktop/python/lesson.ipynb#Y160sZmlsZQ%3D%3D?line=10'>11</a> b.hi()
     <a href='vscode-notebook-cell:/c%3A/Users/Tenerome/Desktop/python/lesson.ipynb#Y160sZmlsZQ%3D%3D?line=11'>12</a> print(b.age)
---> <a href='vscode-notebook-cell:/c%3A/Users/Tenerome/Desktop/python/lesson.ipynb#Y160sZmlsZQ%3D%3D?line=12'>13</a> print(b.__name)


AttributeError: 'B' object has no attribute '__name'
```

python也支持多继承(C++也支持,java不支持)

②多态(重载)

多态:指基类的同一个方法在不同派生类对象中具有不同的表现和行为.

```python
class Animal:
    def eat(self):
        print('吃')
    def run(self):
        print('跑')

class Dog(Animal):
    def run(self):   #子类重写父类方法
        print('用四个腿跑')#如果想继续调用父类的方法,可以使用super().方法()

dog=Dog()
dog.run()
```

```
用四个腿跑
```

#### 文本

##### 文本文件操作

对文本文件操作,用with open() as f. open是python的内置函数,有三个参数open(filename,mode,encoding),mode是操作模式,有读,写,读写三种.

```python
with open('1.txt','r') as f:
    print(f.read())
```

    Hello world
    Nice to meet you

'r':读模式,read()函数,读取文件内容,返回一个str<br>
'w':写模式<br>
'r+'或'w+':读写,不能用rw<br>
'a':追加

```python
with open('1.txt','r+') as f:
    f.write('哈尔滨商业大学')
    print(f.read())
print(f.read())
#不能同时读写?
```

##### 文件的补充

with open('1.txt','r/w') as f:<br>
这句执行后,r模式会把游标移动到最前面,w模式会清空内容<br>
如果文件不存在,r模式会报错.w模式会自动创建新文件<br>
f.read()会从当前游标位置读取到文档end<br>
f.write()会从当前位置,向后写

①r模式,游标在最前面,先写后读,会覆盖前面的字符,原文档:
![pic](https://img2022.cnblogs.com/blog/2629720/202210/2629720-20221010130207268-1206120483.png)

r+模式,先写后读

```python
with open('1.txt','r+') as f:
    f.write('Hello')
    print(f.read())
```

    6789

f.write()从最前面开始覆盖写,游标停止在o后,开始读,所以读到的是6789

![pic](https://img2022.cnblogs.com/blog/2629720/202210/2629720-20221010130207599-1579630915.png)

r+模式,先读后写

```python
with open('1.txt','r+') as f:
   print(f.read())
   f.write('nnnn')
```

    Hello6789

r+模式,开始游标在head,读所有的字符,读完游标在最后,写的内容追加到end
![pic](https://img2022.cnblogs.com/blog/2629720/202210/2629720-20221010130207899-1880948872.png)

w+模式,先把内容清空

```python
with open('1.txt','w+') as f:
    pass
```

![pic](https://img2022.cnblogs.com/blog/2629720/202210/2629720-20221010130208202-1019685978.png)

先写后读

```python
with open('1.txt','w+') as f:
    f.write('MMMM')
    print(f.read())
```

![pic](https://img2022.cnblogs.com/blog/2629720/202210/2629720-20221010130208490-1462750678.png)

能写进去,但写完之后游标在end,自然就读取不到,想要读内容,需要用f.seek()移动游标

```python
with open('1.txt','w+') as f:
    f.write('MMMM')
    f.seek(0)
    print(f.read())
```

    MMMM

w+模式,先读后写

```python
with open('1.txt','w+') as f:
   print(f.read())
   f.write('nnnn')
```

![pic](https://img2022.cnblogs.com/blog/2629720/202210/2629720-20221010130208812-662516089.png)

w+模式先清空,所以什么都读不到,然后把内容写进去

a+模式打开时游标默认也在末尾,且文件不存在也新建

##### 二进制文件操作

使用pickle库序列化对象,就是把二进制数据存储在磁盘上

```python
import pickle
i=133000
a=99.23
s='Hello'
lst=[[1,2,3],[4,5,6]]
tu=(2,3,6)
coll={2,4,7}
dic={'a':'apple','b':'banana'}
data=[i,a,s,lst,tu,coll,dic]
```

想把这些数据存下来

```python
with open('sample.dat','wb') as f:
    try:
        pickle.dump(len(data),f)
        for item in data:
            pickle.dump(item,f)
    except:
        print('写文件异常')
```

读写二进制文件,就是在w/r/a后面加b,pickle.dump(obj,file_writeable),dump()函数可以把任意对象写入一个可写的文件对象中

使用pickle库反序列化对象

```python
with open('sample.dat','rb') as f:
    n=pickle.load(f)
    for i in range(n):
        x=pickle.load(f)
        print(x)
        print(type(x))
```

    133000
    <class 'int'>
    99.23
    <class 'float'>
    Hello
    <class 'str'>
    [[1, 2, 3], [4, 5, 6]]
    <class 'list'>
    (2, 3, 6)
    <class 'tuple'>
    {2, 4, 7}
    <class 'set'>
    {'a': 'apple', 'b': 'banana'}
    <class 'dict'>

使用pickle.load()读取二进制文件,返回any,根据不同的类型,x的类型动态变化

##### OS和OS.path模块

open()用来读写单个文件,如果想对目录中的文件读写,需要OS,OS.path库

OS库:<br>
getcwd():  获取当前目录的path<br>
mkdir():   创建目录<br>
listdir(path): 返回path目录下的文件和子目录列表<br>
walk(top,topdown=True,onerror=None): 遍历目录树,就是所有目录及其子目录的所有文件<br>
remove(path):删除指定文件,要求用户拥有删除文件的权限,并且文件没有只读或其他特殊属性<br>
rename(src,dst):重命名文件或目录,可以实现文件的移动<br>
startfile(filepath[,operation] ):使用关联的应用程序打开指定文件或启动指定应用程序

OS.path库:<br>
os.path

```python
import os
import os.path
for i in os.listdir():
    if os.path.isfile(i) and i.endswith('.py'):
        print(i)
```

    test.py

os.path.isfile(path):判断是否为文件,返回bool<br>
S.endswith('suffix'):Return True if S ends with the specified suffix<br>
os.path.exist(path):判断当前路径是否存在指定的目录或文件<br>
os.path.dirname('1.txt'):获取文件的目录部分,是相对路径<br>
os.path.splittext('1.png'):分隔文件名和扩展名,运行结果:('1','png')<br>
os.path.join(path,*path),连接两个或多个path

#### python高阶

##### python的可变类型和不可变类型

可变类型:

- 列表[]

- 集合

- 字典

不可变类型

- 数字(整型,浮点型等)

- 字符串

- 元组:()

<mark>可变和不可变的实质:</mark> 可变数据和不可变数据是相对于**引用地址**来说的，不可变数据类型不允许变量的值发生变化，<u>如果改变了的变量的值，相当于新建了一个对象</u>，而对于相同的值的对象，内部会有一个引用计数来记录有多少个变量引用了这个对象。可变数据类型允许变量的值发生变化。对变量进行修改操作只会改变变量的值，不会新建对象，变量引用的地址也不会发生变化，不过对于相同的值的不同对象，在内存中则会存在不同的对象，即每个对象都有自己的地址，相当于内存中对于同值的对象保存了多份，这里不存在引用计数，是实实在在的对象。

简单地讲，可变数据和不可变数据的“变”是相对于引用地址来说的，不是不能改变其数据，而是改变数据的时候会不会改变变量的引用地址。

比如:

python中可以使用id()函数查看变量的引用地址

![](https://img2022.cnblogs.com/blog/2629720/202210/2629720-20221010125531827-857069376.png)

可以看到,字符串虽然值改变了,但它的引用地址也发生了变化,在内存中实际是

```python
a='China'
a->str1->'China'
a+='hello'
a->str2->'China hello'
```

而可变类型如列表不需要开辟新空间,还是曾经的引用地址

小数据池

```python
a=3
b=3
print(id(a)==id(b))
```

    True

python将-5~256的整数进行了缓存,当将这些整数赋值给变量时并不会创建新的对象

```python
a=300
b=300
print(id(a)==id(b))
```

    False

对于字符串变量,当字符串变量的长度为0或1时,也缓存

```python
a=' '
b=' '
print(id(a)==id(b))
```

    True

```python
a='a'
b='a'
print(id(a)==id(b))
```

    True

当字符串长度>1,且字符串中只有字母数字和下划线时,也会缓存

```python
a='adawFF1__aawd3444___1ffa'
b='adawFF1__aawd3444___1ffa'
print(id(a)==id(b))
```

    True

最后还有bool变量也适用小数据池

```python
a=True
b=True
print(id(a)==id(b))
```

    True

其他情况不缓存

```python
a='a '
b='a '
print(id(a)==id(b))
```

    False

##### 迭代器和生成器

生成器generator:也叫生成器函数,含有yield修饰的表达式.每次执行到yield语句时会返回一个值然后暂停或挂起后面代码的执行.下次通过生成器对象的__next__()方法,内置函数next(),for循环遍历生成器对象元素或其他方法显示'索要'数据时恢复执行.

普通函数

```python
def fun():
    i=0
    while i<10:
        print(i,end=" ")
        i=i+1
fun()
```

    0 1 2 3 4 5 6 7 8 9 

添加yield修饰变量,变成生成器

```python
def fun():
    i=0
    while i<10:
        yield i
        print('i=',i,end=" ")
        i=i+1
a=fun()
print(a)
```

    <generator object fun at 0x000001DCA95B5900>

a就是一个生成器对象

通过内置函数next()生成下一个a

```python
next(a)
next(a)
next(a)
next(a)
next(a)
```

    i= 0 i= 1 i= 2 i= 3 

​    
​    
​    
    4

可以看到前三个是i=,最后一个是直接一个数字,也就是说,程序顺序执行,执行到yield时,后面的print()不执行,返回i的值,就是直接输出i,下次执行时先执行print(),所以出现了三个print()

断点调试:
![gif](https://img2022.cnblogs.com/blog/2629720/202210/2629720-20221010125532341-1616192905.png)

所以yield并不是return那种返回值,而是输出

或使用生成器对象方法__next__()

```python
a.__next__()
```

    i=: 4 

​    
​    
​    
    5

```python
def fun():
    i=0
    while i<10:
        yield i
        i=i+1
a=fun()
```

而且可以用for遍历出来

```python
[i for i in a]
```

    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

迭代器Iterator:是一种可作用于next()函数的对象,表示一个惰性计算的序列.
惰性计算就是执行到yield就暂停,每调用一次next,就再执行一次,不像其他函数直接执行完

可迭代类型Iterable:凡是可以作用于for循环的对象都是可迭代的,其内部实现了__iter__()函数或__getitem()__函数.list,dict,str等都是Iterable的,但不是Iterator.可以通过iter()函数来获得一个Iterator

range()对象也不是Iterator,range()实际是一个Sequence

```python
x=range(10)
type(x)
```

    range

```python
next(x)
```

    ---------------------------------------------------------------------------
    
    TypeError                                 Traceback (most recent call last)
    
    c:\Users\Tenerome\Desktop\python\lesson.ipynb Cell 35 in <cell line: 1>()
    ----> <a href='vscode-notebook-cell:/c%3A/Users/Tenerome/Desktop/python/lesson.ipynb#X61sZmlsZQ%3D%3D?line=0'>1</a> next(x)

​    
    TypeError: 'range' object is not an iterator

在python中for语句真正的含义是迭代,while语句才是真正的循环.for类似于java中的foreach语句<br>
循环可以通过增加条件跳过不需要的元素,而迭代只能一个一个往后取数.迭代有固定的格式:for ... in ....

在for语句内部,会通过iter()方法将可迭代对象转换成迭代器,然后重复调用next()方法实现.for语句会自动捕捉StopIteration异常,并在捕捉异常后终止迭代

##### 闭包和装饰器

万物皆为对象,函数也是

```python
def fun():
    pass
print(type(fun))
```

    <class 'function'>

所有的函数都是function类的实例

①内嵌函数<br>
如果一个函数的内部还有一个函数,该如何去调用?

```python
def outer():
    print('this is outer function')
    def inner():
        print('this is inner function')
```

这样的结构是不完整的,想要调用内函数,必须在外函数内return内函数名

```python
def outer():
    print('this is outer function')
    def inner():
        print('this is inner function')
    return inner
```

```python
outer()()
```

    this is outer function
    this is inner function

这里outer()=inner
即f=outer(),f()=inner()

②闭包

如果内嵌函数使用了外函数的资源,则称内嵌函数和它引用的资源为闭包

```python
def line_conf(a):
    b=1
    def line(x):
        return a*x+b
    return line
```

```python
a=line_conf(2)
print(a(3))
```

    7

a=line_conf(2),就相当于:<br>
a(x){<br>
    return 2*x +1}

就是把外函数的参数和用到的资源都传到内函数中,返回这个内函数

如果想要在内内函数中修改用到的外函数的资源,就要用nonlocal声明

```python
def line_conf(a):
    b=1
    def line(x):
        nonlocal b
        b=3
        return a*x+b
    return line
```

```python
a=line_conf(2)
print(a(3))
```

    9

③装饰器(decorator):<br>
封闭-开放原则:封闭已经实现的功能代码块,开放对扩展开发

如现有一个函数f(),编写一个装饰器,计算f()运行时间

```python
import time
def f():
    print('Hello')
    time.sleep(1)
    print('world')
```

装饰器函数

```python
def deco(f):
    def wrapper():
        start_time=time.time()
        f()
        end_time=time.time()
        exec_time=(end_time-start_time)*1000
        print(exec_time)
    return wrapper
```

要在f()前一行添加@deco,装饰器函数名

```python
@deco
def f():
    print('Hello')
    time.sleep(1)
    print('world')
```

运行的时候可以直接用f(),系统会自动加载装饰器函数

```python
f()
```

    Hello
    world
    1004.9920082092285

带参数且有返回值装饰器,被装饰的函数有几个参数,装饰器内部的inner函数就有几个.返回值先在内函数中返回给内函数inner(),再由return inner返回给outer()

```python
def yanzheng(func):
    def inner(a,b):#func有两个参数,inner就有两个参数
        print('开始验证')
        if type(a)==int and type(b)==int:
            return func(a,b)#返回给inner()
        else:
            print('必须输入整数')
    return inner#返回给yanzheng()
@yanzheng
def add_num(a,b):
    return a+b
print(add_num(4,5))
```

    开始验证
    9

##### 赋值,深拷贝和浅拷贝

浅拷贝:copy.copy()或者用切片切出来的也是浅拷贝<br>
深拷贝:copy.deepcopy()

①不可变类型

赋值:

python中的所有赋值,都是引用赋值,类似c语言中的指针的赋值

```python
a=100
b=a
print(id(a),id(b))
```

    2491268879824 2491268879824

此时a和b指向同一个内存

```python
b=200
print(id(a),id(b))
```

    2491268879824 2491268883088

因为a,b是number类,不可原地改变,b=200,是把一个新的内存赋给了b,所以a is not b

对于不可变类型,赋值,浅拷贝,深拷贝的结果都一样:原变量改变,复制变量不变

拷贝需要导入库 copy

浅拷贝

```python
import copy
a=100
b=copy.copy(a)
print(id(a),id(b))
b=200
print(id(a),id(b))
```

    2491268879824 2491268879824
    2491268879824 2491268883088

深拷贝:

```python
import copy
a=100
b=copy.deepcopy(a)
print(id(a),id(b))
b=200
print(id(a),id(b))
```

    2491268879824 2491268879824
    2491268879824 2491268883088

总结:对于不可变类型,赋值,浅拷贝,深拷贝后指向同一个内存空间,但是改变值会指向新的空间,所以两个值互补影响

②可变类型的内部只包含简单的数据类型

赋值

```python
a=[100,200,300]
b=a
print(id(a),id(b))
a.append(400)
print(id(a),id(b))
```

    2491907542016 2491907542016
    2491907542016 2491907542016

浅拷贝

```python
import copy
a=[100,200,300]
b=copy.copy(a)
print(id(a),id(b))
a.append(400)
print(id(a),id(b))
```

    2491907815104 2491907524864
    2491907815104 2491907524864

深拷贝

```python
import copy
a=[100,200,300]
b=copy.deepcopy(a)
print(id(a),id(b))
a.append(400)
print(id(a),id(b))
```

    2491907555136 2491907692544
    2491907555136 2491907692544

总结:可变类型中只包含简单的数据类型的话,对其赋值操作,两个变量指向同一个空间,一起改变.浅拷贝和深拷贝相同都是创建了新的空间

③可变类型中还包含复杂的可变类型

赋值

```python
a=[[1,2,3],[4,5,6],7,8]
b=a
a.append(9)
print(id(a),id(b))
a[3]=100
print(id(a[3]),id(b[3]))
a[0].append(200)
print(id(a[0]),id(b[0]))
```

    2491907952960 2491907952960
    2491268879824 2491268879824
    2491907955584 2491907955584

浅拷贝

```python
a=[[1,2,3],[4,5,6],7,8]
b=copy.copy(a)
a.append(9)
print(id(a),id(b))
a[2]=100
print(id(a[2]),id(b[2]))
a[0].append(200)
print(id(a[0]),id(b[0]))
```

    2491907954624 2491908235328
    2491268879824 2491268688368
    2491907619776 2491907619776

深拷贝:

```python
a=[[1,2,3],[4,5,6],7,8]
b=a
a.append(9)
print(id(a),id(b))
a[2]=100
print(id(a[2]),id(b[2]))
a[0].append(200)
print(id(a[0]),id(b[0]))
```

    2491908213376 2491908213376
    2491268879824 2491268879824
    2491907955584 2491907955584

总结:可变类型含有可变类型时,赋值操作后指向同一块内存,深拷贝操作指向不同的内存.浅拷贝特殊,其中的不可变类型指向不同的空间,可变类型指向相同的空间

总结:<br>
赋值:不管怎么操作都指向同一空间<br>
深拷贝:不可变类型指向同一空间,可变类型指向不同空间<br><br>
浅拷贝:不可变类型指向同一空间<br>
可变类型中的不可变类型:指向不同的空间<br>
可变类型中的可变类型:指向相同的空间

记忆:赋值操作,不可变类型 和 浅拷贝可变类型中的可变类型 指向同一块空间<br>
其他都是不同的空间

##### 正则表达式

RE语法:
![pic](https://img2022.cnblogs.com/blog/2629720/202210/2629720-20221010125532916-1200344870.png)

```python
import re
```

①,提取邮编

```python
s='学校邮编是:150028,请牢记150028'
p='[1-9]\d{5}'
re.search(p,s).group(0)
```

    '150028'

p:[1-9]第一个字符,是1-9的任意一个字符,因为邮编不能以0开始<br>
\d:表示任意数字,0-9任意一个字符<br>
{5}:重复5次<br>
所以它匹配的是以1-9开始的六位数字

re.search(pattern,string,flag=0),pattern正则语法.<br>
re.search()方法扫描整个string,并返回第一个成功匹配的对象.如果匹配失败,则返回none.成功匹配后返回一个re.match对象,可以通过对象的group()获得匹配结果,如果pattern指定要返回多个结果,可以通过group(num)获取不同的匹配位置

re.match和re.search的区别:<br>
re.search并不要求从字符串开头进行匹配,re.match必须从开头匹配,若开头匹配失败,整个就失败了,返回none

```python
print(re.match(p,s))
```

    None

②,提取python,cython,jython

```python
s='python的解释器包括cython,jython等'
p='.*([cjp]ython).*([cjp]ython).*([cjp]ython).*'
m=re.search(p,s)
print(m.groups())
for i in range(1,4):
    print(m.group(i))
```

    ('python', 'cython', 'jython')
    python
    cython
    jython

p='.\*([cjp]ython).\*([cjp]ython).\*([cjp]ython).\*'<br>
.表示任意字符,*表示任意个,所以就把句子中的所有ython提取出来了<br>
groups()返回一个元组,group(num)返回第num个匹配字符串

第二种方法:使用re.findall(p,s):

```python
s='python的解释器包括cython,jython等'
p='[cjp]ython'
l=re.findall(p,s)
print(l)
```

    ['python', 'cython', 'jython']

re.findall()和search的区别就是search只返回第一个匹配结果,所以想匹配多个要重复多次模式.findall()就是返回所有的匹配对象,且返回结果是一个列表

用r''形式的字符串,中的\不转义

③,提取姓名和电话

```python
s='''
张三,手机号码13945678823
李四,手机号码18943215634
王五,手机号码12456236671
'''
p=re.compile(r'^(.+),.+(\d{11})',re.M)
for i in p.findall(s):
    print(i)
```

    ('张三', '13945678823')
    ('李四', '18943215634')
    ('王五', '12456236671')

re.compile(pattern,flag)方法将正则表达式编译为re.pattern对象,flag参数指定匹配方式,常见标志有:<br>
1,re.I:忽略大小写<br>
2,re.M多行匹配:当pattern中出现^和$时,默认匹配第一行最开始和最后一行的末尾,如果想让^和&匹配每一行的开始和末尾,则需要加上re.M<br>
3,re.S :使pattern中.匹配任意字符,包括换行,常配合(.*?)

python用''' .....'''定义多行字符串

r'^(.+),.+(\d{11})':<br>
^表示从开头,re.M表示每一行都从开头匹配<br>
():表示要匹配的内容
.:任意字符,+表示任意1和或多个字符,*表示0个或多个<br>
所以^(.+), :表示从开头到,之间的字符,且不能为空<br>
接着.+表示任意字符,(\d)表示要匹配的是数字,{11}表示重复11次

re.I忽略大小写

```python
s='abc,Abc,aBC,ABC,aabc'
re.findall('abc',s,re.I)
```

    ['abc', 'Abc', 'aBC', 'ABC', 'abc']

④,非贪婪匹配(.*?)

```python
import re
s='''
abc,c
Abc,abc,ABC
Harbin University of Commerce
'''
print(re.findall('a(.*)c',s))
print(re.findall('a(.*?)c',s))
```

    ['bc,', 'b', 'rbin University of Commer']
    ['b', 'b', 'rbin University of Commer']

a(.*)c默认贪婪匹配,当第一行abc,c的a匹配到时,它会从后面匹配c,使中间的.\*尽可能更长.<br>
?表示非贪婪匹配,匹配到a后继续向后搜索,遇到第一个c就返回

re.findall()默认从第一行第1个字符到最后一行最后一个字符进行全文搜索,不需要用re.M

⑤,去除字符串中的重复项

```python
s="It's a very good good goood idea"
#可以用set集合的不重复性
' '.join(set(s.split()))
```

    "idea It's goood a good very"

但是集合还有一个无序性,所以输出是无序的

用re.sub(pat,repl,string):将字符串中所有与pat匹配的项用repl替换,返回新的字符串.

```python
re.sub(r'(\b\w+\s)\1+',r'\1',s)
```

    "It's a very good goood idea"

\b表示边界,\w任意字母数字下划线,\s表示空格,(\b\w\s)\1,这个\1是重复前面()的内容,+表示一次或多次,所以第一个参数就是用来匹配重复的项的:good good good,然后用第二个参数替换.

\num表示第num个括号匹配的内容,如'.\*([hv]ello).\*([cb]nmk).*'<br>
\1就表示[hv]ello,\2表示[cb]nmk

⑥匹配字符串中的汉字

```python
s="not 404 found 33.4 张三 99 哈尔滨"
s=re.sub('\d+\.?\d*|[a-zA-Z]+','',s)
s=s.split()
res=' '.join(s)
print(res)
```

    张三 哈尔滨

用sub匹配到所有数字和单词,并用null,""替换.<br>
\d任意数字,+正闭包,一个或多个;  .?0个或1个点,就能匹配整数和小数;\d*,0个或多个数字,就能匹配所有的整数和小数<br>
然后|上所有字母组成的单词,就只剩汉字了

string.join(iterable),将string插入序列s除了第一个和最后一个元素之间,然后返回join之后的字符串.

⑦分割字符串

```python
s="info:zhangsan 22 Harbin"
re.split(':|\s',s)
```

    ['info', 'zhangsan', '22', 'Harbin']

re.split()的第一个参数是分隔的符号,默认空格,这里指定:或空格

⑧列表元素的匹配

```python
email_list=['xiaodaaw@163.com','xiddddaodaaw@163.comaaa','xiaodaaw@qq.com','xiaodaaw@163.ccom']
p='\w+@163\.com$'
def email_check(x):
    if re.match(p,x):
        return True
res=filter(email_check,email_list)
print(list(res))
```

    ['xiaodaaw@163.com']

或用lambda表达式

```python
email_list=['xiaodaaw@163.com','xiddddaodaaw@163.comaaa','xiaodaaw@qq.com','xiaodaaw@163.ccom']
p='\w+@163\.com$'
res=filter(lambda x: re.match(p,x),email_list)
print(list(res))
```

    ['xiaodaaw@163.com']

⑨匹配中文

```python
s='你好,hello,2022,好的'
p=re.compile('[\u4e00-\u9fa5]+')
res=p.findall(s)
print(res)
```

    ['你好', '好的']

\u4e00-\u8fa5是unicode所有的中文

##### 全局解释器锁GIL

GIL:全局解释器锁,当多个线程并行执行时,每个线程在执行前都要先获取GIL锁,保证同一时刻只有一个python线程在运行,目的时解决多线程同时竞争程序中的全局变量而出现的线程安全问题.

所以在python中的多线程执行效率会被GIL大大限制

在过去单核及其上,不能真正的并行,而是并发处理的,所以会有资源竞争问题,也就有了GIL锁,现在的机器都是多线程的了,但GIL仍存在与cpython解释器中.想要使用python多线程,就要想办法绕过GIL

释放GIL的情况:<br>
1,在IO操作等可能会引起阻塞的system call之前,可以暂时释放GIL锁,但在执行完毕后,必须重新获取锁<br>
2,python 3.x使用计时器,执行时间达到阈值后,当前线程释放GIL锁.或python2中,tickets计数达到100时,释放GIL锁

解决GIL问题的方案:<br>
1,使用其他语言:c,java等<br>
2,使用其他解释器,如java的jython<br>
3,使用多进程,python中的多进程是可以利用cpu资源的

虽然有GIL限制,多线程爬取还是比单线程性能高的,因为遇到IO阻塞时会自动释放GIL锁

##### 线程

进程,线程和协程的区别:<br>
(1)进程是资源分配的单位,线程是操作系统系统任务调度的单位,协程是1个线程内的多任务<br>
(2)进程之间不共享全局变量,线程和协程可以共享全局变量<br>
(3)计算密集型任务适合多进程,I/O密集型任务适合多线程或多协程<br>
(4)进程消耗资源大,线程消耗资源少,协程消耗资源最少<br>
(5)使用线程池或进程池可以提高效率减少资源消耗

①单线程-顺序执行方式

```python
import time 
def sing():
    for i in range(3):
        print("唱歌")
        time.sleep(1)
def dance():
    for i in range(3):
        print("跳舞")
        time.sleep(1)
if __name__=='__main__':
    start_time=time.perf_counter()
    sing()
    dance()
    end_time=time.perf_counter()
    print('total time is {}'.format(end_time-start_time))
```

    唱歌
    唱歌
    唱歌
    跳舞
    跳舞
    跳舞
    total time is 6.0794796

计算程序执行时间的几种方式:  
time.time()会将sleep()的时间也算进去  
time.perf_counter() 具有最高可用分辨率的时钟,包含sleep()时间,适合测量短持续时间  
time.process_time()不包括sleep()时间,其他与time.perf_counter类似  
此外python3.7后提供了三个方法精确到纳秒的计时:  
time.perf_counter_ns()  
time.time_ns()  
time.clock()  

②多线程

```python
import time 
import threading
def sing():
    for i in range(3):
        print("唱歌")
        time.sleep(1)
def dance():
    for i in range(3):
        print("跳舞")
        time.sleep(1)
if __name__=='__main__':
    start_time=time.perf_counter()
    t1=threading.Thread(target=sing)
    t2=threading.Thread(target=dance)
    t1.start()
    t2.start()
    end_time=time.perf_counter()
    print(f'total time is {end_time-start_time}')
```

    唱歌
    跳舞
    total time is 0.0035195000000385335
    跳舞唱歌
    
    跳舞唱歌

t1=threading.Thread(),在主线程下创建子线程t1  
target=sing,为子线程t1分配任务sing  
t1.start()开启子线程
运行程序,主线程开始执行,执行到t1,t2start(),创建子线程,接着主线程继续执行,输出了执行时间,主线程的代码执行完毕,但程序并不结束,而是等待子线程t1,t2执行,t1,t2执行完剩下的迭代,程序结束  
如图示

![pic](https://img2022.cnblogs.com/blog/2629720/202210/2629720-20221010125533466-406403022.png)

主线程执行完,若子线程未执行完,主线程会等待子线程结束后才退出程序  
子线程是因为sleep()才晚于主线程,如果没有slee()

```python
import time 
import threading
def sing():
    for i in range(3):
        print("唱歌")
        # time.sleep(1)
def dance():
    for i in range(3):
        print("跳舞")
        # time.sleep(1)
if __name__=='__main__':
    start_time=time.perf_counter()
    t1=threading.Thread(target=sing)
    t2=threading.Thread(target=dance)
    t1.start()
    t2.start()
    end_time=time.perf_counter()
    print(f'total time is {end_time-start_time}')
```

    唱歌
    唱歌
    唱歌
    跳舞
    跳舞
    跳舞
    total time is 0.00368129999969824

或者给主线程加上等待时间,也能看到并发执行完

```python
import time 
import threading
def sing():
    for i in range(3):
        print("唱歌")
        time.sleep(1)
def dance():
    for i in range(3):
        print("跳舞")
        time.sleep(1)
if __name__=='__main__':
    start_time=time.perf_counter()
    t1=threading.Thread(target=sing)
    t2=threading.Thread(target=dance)
    t1.start()
    t2.start()
    time.sleep(4)
    end_time=time.perf_counter()
    print(f'total time is {end_time-start_time}')
```

    唱歌
    跳舞
    跳舞唱歌
    
    跳舞唱歌
    
    total time is 4.0054009999998925

 统计多线程运行时间的正确方法:  

```python
import time 
import threading
def sing():
    for i in range(3):
        print("唱歌")
        time.sleep(1)
def dance():
    for i in range(3):
        print("跳舞")
        time.sleep(1)
if __name__=='__main__':
    start_time=time.perf_counter()
    t1=threading.Thread(target=sing)
    t2=threading.Thread(target=dance)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    end_time=time.perf_counter()
    print(f'total time is {end_time-start_time}')
```

    唱歌
    跳舞
    跳舞
    唱歌
    唱歌跳舞
    
    total time is 3.041722899999968

join函数会等待子线程结束,或者说要判断汇合点,都汇合后主线程才继续执行

③统计线程数 (在独立文件中运行)  

```python
import threading
import time

#查看线程数
def sing():
    for i in range(3):
        print("唱歌")
        time.sleep(1)
def dance():
    for i in range(3):
        print("跳舞")
        time.sleep(1)
if __name__=='__main__':
    sing_t=threading.Thread(target=sing)
    dance_t=threading.Thread(target=dance)
    sing_t.start()
    dance_t.start()

    while True:
        length=len(threading.enumerate())
        print('当前运行的线程数为:%d' %length)
        time.sleep(2)
        if length <=1:
            print("主线程运行结束")
            input()
            break
```

![gif](https://img2022.cnblogs.com/blog/2629720/202210/2629720-20221010125534064-1156201357.png)

④线程类  
除了通过Threading.Thread()函数创建线程对象外,话可以通过继承threading.Thread类在类中创建多个函数

Thread派生类中必须由一个run()函数

```python
class MyThread(threading.Thread):
    def run(self):
        for i in range(5):
            time.sleep(1)
            print("线程名:%s" %(self.name + "-"+str(i)))
if __name__=="__main__":
    threads=[]
    for i in range(3):
        t=MyThread()
        threads.append(t)
    for t in threads:
        t.start()
```

    线程名:Thread-69-0
    线程名:Thread-68-0
    线程名:Thread-70-0
    线程名:Thread-70-1
    线程名:Thread-68-1
    线程名:Thread-69-1
    线程名:Thread-70-2线程名:Thread-69-2
    线程名:Thread-68-2
    
    线程名:Thread-69-3线程名:Thread-70-3
    线程名:Thread-68-3
    
    线程名:Thread-69-4线程名:Thread-70-4
    线程名:Thread-68-4

三个线程分别是Thread-41,40,39.分别做输出0-4的任务,三个线程并发执行,看着是同时执行的,实际是cpu快速切换,如果给线程执行后添加sleep(),就能看到实际还是一次只有一个线程执行

```python
class MyThread(threading.Thread):
    def run(self):
        for i in range(5):
            time.sleep(1)
            print("线程名:%s" %(self.name + "-"+str(i)))
if __name__=="__main__":
    threads=[]
    for i in range(3):
        t=MyThread()
        threads.append(t)
    for t in threads:
        t.start()
        time.sleep(6)#要让线程等待的时间大于每个线程的总执行时间,即6>5*1
```

    线程名:Thread-79-0
    线程名:Thread-79-1
    线程名:Thread-79-2
    线程名:Thread-79-3
    线程名:Thread-79-4
    线程名:Thread-80-0
    线程名:Thread-80-1
    线程名:Thread-80-2
    线程名:Thread-80-3
    线程名:Thread-80-4
    线程名:Thread-81-0
    线程名:Thread-81-1
    线程名:Thread-81-2
    线程名:Thread-81-3
    线程名:Thread-81-4

⑤守护线程  
如果一个线程别设置为守护线程,那么意味着这个线程是不重要的.如果主线程结束了,守护线程还没有运行完,那么它会被强制结束,在python中可以使用setDaemon方法来将某个线程设置为守护线程  
(复制到独立文件运行)

```python
from threading import Thread
import time
def sayhi(name):
    time.sleep(2)
    print("%s say hello" %name)
    input()
if __name__=="__main__":
    t=Thread(target=sayhi,args=('Tom',))#传参数是用元组传的
    t.setDaemon(True)
    t.start()
    print("主线程")
    print(t.is_alive())
```

![pic](https://img2022.cnblogs.com/blog/2629720/202210/2629720-20221010125534508-280175265.png)

可以看到,程序开始执行,创建子线程t,子线程t启动并打印begin,然后休眠2秒,同时主线程打印"主线程",并输出子线程的状态仍alive,主线程结束,子线程也被结束掉了

⑥线程共享全局变量

```python
from concurrent.futures import thread
import threading
import time
num=100
def test1():
    global num
    num+=100
def test2():
    print("num=",num)
if __name__=="__main__":
    t1=threading.Thread(target=test1)
    t2=threading.Thread(target=test2)
    t1.start()
    time.sleep(2)
    t2.start()
    time.sleep(1)
    print("num=",num)
```

    num= 200
    num= 200

⑦实例

```python
import threading
import time
import requests
#spyder
url_start='http://xinwen.hrbcu.edu.cn/jxky.htm'
urls=[f'http://xinwen.hrbcu.edu.cn/jxky{i}.htm' for i in range(7,0,-1)]
urls.insert(0,url_start)

def craw(url):
    r=requests.get(url)
    print(url,len(r.text))
#单线程
def single_t():
    print("单线称爬取")
    for url in urls:
        craw(url)
#多线程爬取
def multi_t():
    print("多线程爬取")
    threads=[]
    for url in urls:
        threads.append(threading.Thread(target=craw,args=(url,)))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
#main
if __name__=="__main__":
    start_time=time.time()
    single_t()
    end_time=time.time()
    print("单线程爬取总时间:",end_time-start_time)

    start_time=time.time()
    multi_t()
    end_time=time.time()
    print("多线程爬取总时间:",end_time-start_time)
```

    单线称爬取
    http://xinwen.hrbcu.edu.cn/jxky.htm 28180
    http://xinwen.hrbcu.edu.cn/jxky7.htm 1693
    http://xinwen.hrbcu.edu.cn/jxky6.htm 1693
    http://xinwen.hrbcu.edu.cn/jxky5.htm 1693
    http://xinwen.hrbcu.edu.cn/jxky4.htm 1693
    http://xinwen.hrbcu.edu.cn/jxky3.htm 1693
    http://xinwen.hrbcu.edu.cn/jxky2.htm 1693
    http://xinwen.hrbcu.edu.cn/jxky1.htm 1693
    单线程爬取总时间: 0.7621386051177979
    多线程爬取
    http://xinwen.hrbcu.edu.cn/jxky6.htm 1693
    http://xinwen.hrbcu.edu.cn/jxky5.htm 1693
    http://xinwen.hrbcu.edu.cn/jxky7.htm 1693
    http://xinwen.hrbcu.edu.cn/jxky4.htm 1693
    http://xinwen.hrbcu.edu.cn/jxky3.htm 1693
    http://xinwen.hrbcu.edu.cn/jxky1.htm 1693
    http://xinwen.hrbcu.edu.cn/jxky2.htm 1693
    http://xinwen.hrbcu.edu.cn/jxky.htm 28180
    多线程爬取总时间: 0.18289875984191895

##### 进程

```python
#在独立文件中运行
import multiprocessing
import time
def sing():
    for i in range(3):
        print("唱歌")
        time.sleep(1)
def dance():
    for i in range(3):
        print("跳舞")
        time.sleep(1)
if __name__=="__main__":
    start_time=time.perf_counter()
    processes=[]
    p1=multiprocessing.Process(target=sing)
    p2=multiprocessing.Process(target=dance)
    processes.extend([p1,p2])
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    end_time=time.perf_counter()
    print("total time is ",end_time-start_time)
    input()
```

![pic](https://img2022.cnblogs.com/blog/2629720/202210/2629720-20221010125534913-1842263356.png)

①通过多进程演示计算密集型任务以及进程池

语法方面和线程差不多

```python
#判断是否为素数
import math
import time
import numpy as np

def run_time(f):
    def inner():
        start=time.time()
        f()
        end=time.time()
        print('执行时间为%.3f ms'%((end-start)*1000))
    return inner

def is_prime(n):
    if n<2:
        return False
    if n==2:
        return True
    if n%2==0:
        return False
    sqrt_n=math.floor(math.sqrt(n))
    for i in range(3,sqrt_n+1,2):
        if n%i==0:
            return False
    return True

data=np.random.randint(1e8,1e9,100)

#单线程
@run_time
def single_thread():
    for i in data:
        is_prime(i)
print("单线程",end="")
single_thread()

#多线程
import threading
@run_time
def multi_threads():
    threads=[]
    for i in data:
        t=threading.Thread(target=is_prime,args=(i,))
        threads.append(t)
    for t in threads:
        t.start()
    for t in threads:
        t.join()
print("多线程",end="")
multi_threads()

#多进程
import multiprocessing
@run_time
def multi_processes():
    processes=[]
    for i in data:
        p=multiprocessing.Process(target=is_prime,args=(i,))
        processes.append(p)
    for p in processes:
        p.start()
    for p in processes:
        p.join()

if __name__=='__main__':
    print("多进程",end="")
    multi_processes()
```

    单线程执行时间为7.996 ms
    多线程执行时间为38.205 ms
    多进程执行时间为1405.988 ms

计算密集型任务,多线程比单线慢,因为多线程会交换使用GIL锁,而多进程更慢,因为进程的资源开销比线程大很多

传统进程池

```python
#在独立文件中运行
from multiprocessing import Pool
@run_time
def pool_multi_processes():
    pool=Pool(processes=4)
    for i in data:
        pool.apply_async(is_prime,(i,))
    pool.close()
    pool.join()

if __name__=='__main__':
    print("进程池",end="")
    pool_multi_processes()
```

进程池执行时间为437.055 ms

进程池的定义用Pool(),并可以用参数processes指定核心数.然后使用apply_async(func,args)来分配任务.进程池使用完需要关闭close(),最后一步的join()是等待pool中的所有进程执行完毕,才能继续后面的操作,它必须放在close()后面

新版进程池

```python
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor
@run_time
def multi_process():
    with ProcessPoolExecutor(max_workers=4) as pool:
        pool.map(is_prime,data)
print("新版进程池",end="")
multi_process()
```

    新版进程池执行时间为198.224 ms

新版线程池

```python
@run_time
def multi_thread():
    with ThreadPoolExecutor() as pool:
        pool.map(is_prime,data)
print("线程池",end="")
multi_thread()
```

    线程池执行时间为18.909 ms

##### 协程Coroutine

协程:可以在一个子程序中 中断,去执行其他子程序.这和函数调用不同,因为不会用到栈,而是类似cpu中断  
比如A,B两个函数

```python
def A():
    print('1')
    print('2')
    print('3')
def B():
    print('x')
    print('y')
    print('z')
```

假设由协程执行,在执行A的过程中,可以随时中断去执行B,B也可能在执行中随时中断,去执行A,就可能是这样的结果  
1 2 x y 3 z

这样看起来有点像多线程并发的结果,但协程是在一个线程中实现的.协程的资源消耗很少,而且没有多线程下的锁机制

在python中,协程是通过生成器实现的

①非协程方式的函数调用

```python
import time
def test1():
    print("1",end=" ")
    time.sleep(1)
def test2():
    print("2",end=" ")
    time.sleep(1)
def main():
    for i in range(5):
        test1()
        test2()
main()
```

    1 2 1 2 1 2 1 2 1 2 

②使用生成器实现协程,函数内部通过yield语句实现"让出执行权"操作.与多线程相比,线程遇到I/O操作时会自动进行线程切换,线程切换是通过yield实现的.所以生成器实现的协程是一种半自动的协程

```python
import time
def test1():
    while True:
        print("1",end=" ")
        yield()
def test2():
    while True:
        print("2",end=" ")
        yield()
def main():
    t1=test1()
    t2=test2()
    for i in range(5):
        next(t1)
        next(t2)
main()
```

    1 2 1 2 1 2 1 2 1 2 

此时还是手动切换的程序,对yield进一步封装,可以使用第三方库greenlet实现协程,也可以对greenlet进一步封装,使用gevent库  
使用gevent库实现多协程

②

gevent:  
通过gevent实现coroutine.创建,调度的开销比线程小,执行效率高.gevent实现了python标准库中一些阻塞库的非阻塞版本:socket,os,select等.基本思想是:当一个greenlet遇到IO操作时,比如访问网络,就自动切换其他greenlet,等到IO操作完成,再适当的时候切回来继续执行.

gevent常用方法:  
gevent.spawn()创建一个greenlet对象  
gevent.getcurrent()返回当前正在执行的greenlet对象  
monkey.patch_all(),猴子补丁,会自动将python的一些标准模块替换成gevent的非阻塞版本  
gevent.join() 汇合语句,等待协程结束
gevent.joinall(greenlets):将多个greenlet对象放入列表中批量执行,相当于调用多次spawn和join

```python
import time
def run_time(f):
    def inner():
        start=time.time()
        f()
        end=time.time()
        print('执行时间为%.3f ms'%((end-start)*1000))
    return inner
```

```python
import gevent
def f(n):
    for i in range(n):
        print(gevent.getcurrent(),i)
        gevent.sleep(0.5)
```

gevent.sleep()执行到这自动进行协程切换

```python
@run_time
def gevent_test():
    print('1')
    g1=gevent.spawn(f,5)
    print('2')
    g2=gevent.spawn(f,5)
    print('3')
    g3=gevent.spawn(f,5)
    g1.join()
    g2.join()
    g3.join()
if __name__=="__main__":
    gevent_test()
```

    1
    2
    3
    <Greenlet at 0x26cf1556bf0: f(5)> 0
    <Greenlet at 0x26cf1556e10: f(5)> 0
    <Greenlet at 0x26cf19da040: f(5)> 0
    <Greenlet at 0x26cf1556bf0: f(5)> 1
    <Greenlet at 0x26cf1556e10: f(5)> 1
    <Greenlet at 0x26cf19da040: f(5)> 1
    <Greenlet at 0x26cf1556bf0: f(5)> 2
    <Greenlet at 0x26cf1556e10: f(5)> 2
    <Greenlet at 0x26cf19da040: f(5)> 2
    <Greenlet at 0x26cf1556bf0: f(5)> 3
    <Greenlet at 0x26cf1556e10: f(5)> 3
    <Greenlet at 0x26cf19da040: f(5)> 3
    <Greenlet at 0x26cf1556bf0: f(5)> 4
    <Greenlet at 0x26cf1556e10: f(5)> 4
    <Greenlet at 0x26cf19da040: f(5)> 4
    执行时间为2549.958 ms

正常执行至少7.5s,使用协程后只需2.5s

③猴子补丁

猴子补丁:运行时打补丁,对于某模块,使用时像修改其中某几个功能,在不修改其源码和调用方式的前提下,把这几个功能替换为使用者自定义的功能,这就叫打上猴子补丁  
例如开发初期使用的时import json,后来发现ujson性能更高,如果把每个文件的import json都改成import ujson as json成本较高,只需要在进程入口加上  

```python
import json
import ujson

def monkey_path_json():
    json.__name__='ujson'
    json.dumps=ujson.dumps
    json.loads=ujson.loads
```

给程序打补丁monkey.path_all(),猴子补丁可以让gevent替换那些标准库中造成阻塞调用的标准库,(比如socket,os,select等)替换为gevent自己实现,不需要改动原来的代码即可变为非阻塞

```python
import gevent
import time
from gevent import monkey
monkey.patch_all()

def run_time(f):
    def inner():
        start=time.time()
        f()
        end=time.time()
        print('执行时间为%.3f ms'%((end-start)*1000))
    return inner

def f(n):
    for i in range(n):
        print(gevent.getcurrent(),i)
        time.sleep(0.5)#time.sleep()会被自动替换成gevent.sleep()

@run_time
def gevent_test():
    print('1')
    g1=gevent.spawn(f,5)
    print('2')
    g2=gevent.spawn(f,5)
    print('3')
    g3=gevent.spawn(f,5)
    g1.join()
    g2.join()
    g3.join()
if __name__=="__main__":
    gevent_test()
```

#### python机器学习

##### 用KNN算法实现鸢尾花分类

###### 数据处理

导库

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris#鸢尾花数据集
import seaborn as sns#也用来绘图
from sklearn.model_selection import train_test_split#用来划分训练集和测试你集
from sklearn.preprocessing import StandardScaler#标准化处理
from sklearn.neighbors import KNeighborsClassifier#KNN算法
```

导入数据,分析数据

```python
iris=load_iris()
data=pd.DataFrame(iris.data)#得到data和target,都是一个numpy二维数组
target=pd.DataFrame(iris.target)
df=pd.concat((data,target),axis=1)#把data和target合并
df.columns=['sl','sw','pl','pw','target']#设置列名
df
```

|     | sl  | sw  | pl  | pw  | target |
| --- | --- | --- | --- | --- | ------ |
| 0   | 5.1 | 3.5 | 1.4 | 0.2 | 0      |
| 1   | 4.9 | 3.0 | 1.4 | 0.2 | 0      |
| 2   | 4.7 | 3.2 | 1.3 | 0.2 | 0      |
| 3   | 4.6 | 3.1 | 1.5 | 0.2 | 0      |
| 4   | 5.0 | 3.6 | 1.4 | 0.2 | 0      |
| ... | ... | ... | ... | ... | ...    |
| 145 | 6.7 | 3.0 | 5.2 | 2.3 | 2      |
| 146 | 6.3 | 2.5 | 5.0 | 1.9 | 2      |
| 147 | 6.5 | 3.0 | 5.2 | 2.0 | 2      |
| 148 | 6.2 | 3.4 | 5.4 | 2.3 | 2      |
| 149 | 5.9 | 3.0 | 5.1 | 1.8 | 2      |

150 rows × 5 columns

```python
target_names=iris.target_names#获取target标签,用于标志
target_names
```

```
array(['setosa', 'versicolor', 'virginica'], dtype='<U10')
```

```python
df.corr()#列出比尔森相关系数
```

|        | sl        | sw        | pl        | pw        | target    |
| ------ | --------- | --------- | --------- | --------- | --------- |
| sl     | 1.000000  | -0.117570 | 0.871754  | 0.817941  | 0.782561  |
| sw     | -0.117570 | 1.000000  | -0.428440 | -0.366126 | -0.426658 |
| pl     | 0.871754  | -0.428440 | 1.000000  | 0.962865  | 0.949035  |
| pw     | 0.817941  | -0.366126 | 0.962865  | 1.000000  | 0.956547  |
| target | 0.782561  | -0.426658 | 0.949035  | 0.956547  | 1.000000  |

###### 数据可视化

①饼图

```python
plt.rcParams['font.sans-serif'] = ['SimHei']#用plt.rcParams设置字体
counts=df['target'].value_counts() #返回结果按频数从大到小排列
x=[counts.iloc[0],counts.iloc[1],counts.iloc[2]]
labels=[target_names[0],target_names[1],target_names[2]]
colors=['yellowgreen','gold','#FF0000']
explode=(0,0,0.1)
plt.pie(x,explode=explode,labels=labels,colors=colors,autopct="%1.2f%%",startangle=90)
plt.title('各种鸢尾花所占比例')#fontpreperties已弃用
plt.axis('equal')
plt.show()
```

![](https://img2022.cnblogs.com/blog/2629720/202210/2629720-20221010125535285-221493812.png)

②用seaborn库绘图

盒图

iloc切换,第一个参数表示行,:表示所有行,第二个参数表示列,:4=0:4也就是第0-3列

```python
sns.boxplot(data=df.iloc[:,:4])
plt.show()
sns.boxplot(x=df['target'],y=df['sl'])
plt.show()
```

![](https://img2022.cnblogs.com/blog/2629720/202210/2629720-20221010125535613-968405273.png)

![](https://img2022.cnblogs.com/blog/2629720/202210/2629720-20221010125535934-1255499301.png)

小提琴图

```python
ax=sns.violinplot(data=df.iloc[:,:4])
ax.set_title('四个特征值的分布情况')
plt.show()
```

![](https://img2022.cnblogs.com/blog/2629720/202210/2629720-20221010125536320-443683803.png)

###### 数据预处理

①分割训练集和测试集

```python
x_train,x_test,y_train,y_test=train_test_split(data,target,test_size=0.25 ,
random_state=0)
```

有时还会分割出验证集  

![](https://img2022.cnblogs.com/blog/2629720/202210/2629720-20221010125536750-1667701269.png) test_size=0.25 :指定测试集占25%,训练集就占75%  
random_state=0 :指定了随机种子,这样每次产生的训练集和测试集元素固定  
由于data,target均为DAtaFrame,所以x_train,x_test,y_train,y_test都是DataFrame,且y_train和y_test的秩为2

②对训练集和测试的特征数据进行标准化(标准化不一定是必须的)

```python
ss=StandardScaler()
x_train=ss.fit_transform(x_train)#求x_train的均值和标准差,然后转换数据
x_test=ss.transform(x_test)
```

ss.transform 用上一步算得的x_train的均值和标准差,来对x_test数据进行标准化.

③定义模型,进行训练

```python
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train,y_train)
y_predict=knn.predict(x_test)
print(y_predict)
```

```
[2 1 0 2 0 2 0 1 1 1 2 1 1 1 1 0 2 1 0 0 2 1 0 0 2 0 0 1 1 0 2 1 0 2 2 1 0
 2]


c:\environment\python\lib\site-packages\sklearn\neighbors\_classification.py:207: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
  return self._fit(X, y)
```

提示了knnfit()的第二个参数要求是1个1维numpy数组,但传入的y_train是一个2维数组的DataFrame,可以使用y_train.values.ravel()

```python
knn.fit(x_train,y_train.values.ravel())
y_predict=knn.predict(x_test)
print(y_predict)
```

```
[2 1 0 2 0 2 0 1 1 1 2 1 1 1 1 0 2 1 0 0 2 1 0 0 2 0 0 1 1 0 2 1 0 2 2 1 0
 2]
```

④结果评估

方法1:使用模型自带的评估函数进行准确性测评

```python
print(knn.score(x_test,y_test))
```

```
0.9473684210526315
```

方法2:使用sklearn.metrics李的classification_report模块对预测结果做更详细的分析

```python
from sklearn.metrics import classification_report
print(classification_report(y_test,y_predict,target_names=target_names))
```

```
              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        13
  versicolor       1.00      0.88      0.93        16
   virginica       0.82      1.00      0.90         9

    accuracy                           0.95        38
   macro avg       0.94      0.96      0.94        38
weighted avg       0.96      0.95      0.95        38
```

方法3:使用混淆矩阵查看分类结果

```python
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_predict)
print(cm)
```

```
[[13  0  0]
 [ 0 14  2]
 [ 0  0  9]]
```

把y_test,y_predict合并在一块观察

```python
np.hstack((y_test.values,y_predict.reshape(-1,1)))
```

```
array([[2, 2],
       [1, 1],
       [0, 0],
       [2, 2],
       [0, 0],
       [2, 2],
       [0, 0],
       [1, 1],
       [1, 1],
       [1, 1],
       [2, 2],
       [1, 1],
       [1, 1],
       [1, 1],
       [1, 1],
       [0, 0],
       [1, 2],
       [1, 1],
       [0, 0],
       [0, 0],
       [2, 2],
       [1, 1],
       [0, 0],
       [0, 0],
       [2, 2],
       [0, 0],
       [0, 0],
       [1, 1],
       [1, 1],
       [0, 0],
       [2, 2],
       [1, 1],
       [0, 0],
       [2, 2],
       [2, 2],
       [1, 1],
       [0, 0],
       [1, 2]])
```

##### 泰坦尼克号生还测试

###### 导入数据

```python
import pandas as pd
titanic=pd.read_csv(r'./titanic.txt')
titanic
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>row.names</th>
      <th>pclass</th>
      <th>survived</th>
      <th>name</th>
      <th>age</th>
      <th>embarked</th>
      <th>home.dest</th>
      <th>room</th>
      <th>ticket</th>
      <th>boat</th>
      <th>sex</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1st</td>
      <td>1</td>
      <td>Allen, Miss Elisabeth Walton</td>
      <td>29.0000</td>
      <td>Southampton</td>
      <td>St Louis, MO</td>
      <td>B-5</td>
      <td>24160 L221</td>
      <td>2</td>
      <td>female</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1st</td>
      <td>0</td>
      <td>Allison, Miss Helen Loraine</td>
      <td>2.0000</td>
      <td>Southampton</td>
      <td>Montreal, PQ / Chesterville, ON</td>
      <td>C26</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>female</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1st</td>
      <td>0</td>
      <td>Allison, Mr Hudson Joshua Creighton</td>
      <td>30.0000</td>
      <td>Southampton</td>
      <td>Montreal, PQ / Chesterville, ON</td>
      <td>C26</td>
      <td>NaN</td>
      <td>(135)</td>
      <td>male</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1st</td>
      <td>0</td>
      <td>Allison, Mrs Hudson J.C. (Bessie Waldo Daniels)</td>
      <td>25.0000</td>
      <td>Southampton</td>
      <td>Montreal, PQ / Chesterville, ON</td>
      <td>C26</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>female</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>1st</td>
      <td>1</td>
      <td>Allison, Master Hudson Trevor</td>
      <td>0.9167</td>
      <td>Southampton</td>
      <td>Montreal, PQ / Chesterville, ON</td>
      <td>C22</td>
      <td>NaN</td>
      <td>11</td>
      <td>male</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1308</th>
      <td>1309</td>
      <td>3rd</td>
      <td>0</td>
      <td>Zakarian, Mr Artun</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>male</td>
    </tr>
    <tr>
      <th>1309</th>
      <td>1310</td>
      <td>3rd</td>
      <td>0</td>
      <td>Zakarian, Mr Maprieder</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>male</td>
    </tr>
    <tr>
      <th>1310</th>
      <td>1311</td>
      <td>3rd</td>
      <td>0</td>
      <td>Zenn, Mr Philip</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>male</td>
    </tr>
    <tr>
      <th>1311</th>
      <td>1312</td>
      <td>3rd</td>
      <td>0</td>
      <td>Zievens, Rene</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>female</td>
    </tr>
    <tr>
      <th>1312</th>
      <td>1313</td>
      <td>3rd</td>
      <td>0</td>
      <td>Zimmerman, Leo</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>male</td>
    </tr>
  </tbody>
</table>
<p>1313 rows × 11 columns</p>
</div>

查看数据shape大小

```python
titanic.shape
```

    (1313, 11)

查看前几行数据,可以发现,类型各异,有数值型,类别型,甚至还有缺失的数据  
head(n):可以查看n行数据,缺省时默认为5行

```python
titanic.head(10)
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>row.names</th>
      <th>pclass</th>
      <th>survived</th>
      <th>name</th>
      <th>age</th>
      <th>embarked</th>
      <th>home.dest</th>
      <th>room</th>
      <th>ticket</th>
      <th>boat</th>
      <th>sex</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1st</td>
      <td>1</td>
      <td>Allen, Miss Elisabeth Walton</td>
      <td>29.0000</td>
      <td>Southampton</td>
      <td>St Louis, MO</td>
      <td>B-5</td>
      <td>24160 L221</td>
      <td>2</td>
      <td>female</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1st</td>
      <td>0</td>
      <td>Allison, Miss Helen Loraine</td>
      <td>2.0000</td>
      <td>Southampton</td>
      <td>Montreal, PQ / Chesterville, ON</td>
      <td>C26</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>female</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1st</td>
      <td>0</td>
      <td>Allison, Mr Hudson Joshua Creighton</td>
      <td>30.0000</td>
      <td>Southampton</td>
      <td>Montreal, PQ / Chesterville, ON</td>
      <td>C26</td>
      <td>NaN</td>
      <td>(135)</td>
      <td>male</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1st</td>
      <td>0</td>
      <td>Allison, Mrs Hudson J.C. (Bessie Waldo Daniels)</td>
      <td>25.0000</td>
      <td>Southampton</td>
      <td>Montreal, PQ / Chesterville, ON</td>
      <td>C26</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>female</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>1st</td>
      <td>1</td>
      <td>Allison, Master Hudson Trevor</td>
      <td>0.9167</td>
      <td>Southampton</td>
      <td>Montreal, PQ / Chesterville, ON</td>
      <td>C22</td>
      <td>NaN</td>
      <td>11</td>
      <td>male</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>1st</td>
      <td>1</td>
      <td>Anderson, Mr Harry</td>
      <td>47.0000</td>
      <td>Southampton</td>
      <td>New York, NY</td>
      <td>E-12</td>
      <td>NaN</td>
      <td>3</td>
      <td>male</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>1st</td>
      <td>1</td>
      <td>Andrews, Miss Kornelia Theodosia</td>
      <td>63.0000</td>
      <td>Southampton</td>
      <td>Hudson, NY</td>
      <td>D-7</td>
      <td>13502 L77</td>
      <td>10</td>
      <td>female</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>1st</td>
      <td>0</td>
      <td>Andrews, Mr Thomas, jr</td>
      <td>39.0000</td>
      <td>Southampton</td>
      <td>Belfast, NI</td>
      <td>A-36</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>male</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>1st</td>
      <td>1</td>
      <td>Appleton, Mrs Edward Dale (Charlotte Lamson)</td>
      <td>58.0000</td>
      <td>Southampton</td>
      <td>Bayside, Queens, NY</td>
      <td>C-101</td>
      <td>NaN</td>
      <td>2</td>
      <td>female</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>1st</td>
      <td>0</td>
      <td>Artagaveytia, Mr Ramon</td>
      <td>71.0000</td>
      <td>Cherbourg</td>
      <td>Montevideo, Uruguay</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>(22)</td>
      <td>male</td>
    </tr>
  </tbody>
</table>
</div>

使用info()查看各列的名称,非NaN数量,数据类型

```python
titanic.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1313 entries, 0 to 1312
    Data columns (total 11 columns):
     #   Column     Non-Null Count  Dtype  
    ---  ------     --------------  -----  
     0   row.names  1313 non-null   int64  
     1   pclass     1313 non-null   object 
     2   survived   1313 non-null   int64  
     3   name       1313 non-null   object 
     4   age        633 non-null    float64
     5   embarked   821 non-null    object 
     6   home.dest  754 non-null    object 
     7   room       77 non-null     object 
     8   ticket     69 non-null     object 
     9   boat       347 non-null    object 
     10  sex        1313 non-null   object 
    dtypes: float64(1), int64(2), object(8)
    memory usage: 113.0+ KB

使用describe()查看数值列的统计描述,如果对非数值列也统计,则加上include="all"参数

```python
titanic.describe()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>row.names</th>
      <th>survived</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1313.000000</td>
      <td>1313.000000</td>
      <td>633.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>657.000000</td>
      <td>0.341965</td>
      <td>31.194181</td>
    </tr>
    <tr>
      <th>std</th>
      <td>379.174762</td>
      <td>0.474549</td>
      <td>14.747525</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.166700</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>329.000000</td>
      <td>0.000000</td>
      <td>21.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>657.000000</td>
      <td>0.000000</td>
      <td>30.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>985.000000</td>
      <td>1.000000</td>
      <td>41.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1313.000000</td>
      <td>1.000000</td>
      <td>71.000000</td>
    </tr>
  </tbody>
</table>
</div>

查看列标签

```python
titanic.columns
```

    Index(['row.names', 'pclass', 'survived', 'name', 'age', 'embarked',
           'home.dest', 'room', 'ticket', 'boat', 'sex'],
          dtype='object')

查看行索引

```python
titanic.index
```

    RangeIndex(start=0, stop=1313, step=1)

###### 数据可视化

导入matplotlib库,用rcParams设置字体和负号

```python
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
```

###### 1,年龄和生还的关系

①观察所有乘客的年龄分布,绘制直方图

```python
titanic['age'].plot(kind='hist',bins=14)#hist:类型为直方图,bins:设置分割的份
plt.show()
```

![png](https://img2022.cnblogs.com/blog/2629720/202210/2629720-20221010125537104-1003112942.png)

②绘制生还和遇难的比例饼图

```python
x=titanic['survived'].value_counts()
plt.pie(x,labels=['生还','遇难'],autopct='%1.1f%%',startangle=90,explode=[0,0.1])
plt.axis('equal')#设置成正圆
plt.show()
```

![png](https://img2022.cnblogs.com/blog/2629720/202210/2629720-20221010125537468-215741194.png)

③绘制生还和遇难年龄分布的盒图

取满足条件的列的两种方法:  
x_0=titanic[titanic['survived']==0]['age'].dropna()  
[titanic['survived']==0]是满足survived==0的,['age']是取的列,.dropna():删除NaN  

x_1=titanic.loc[titanic['survived']==1,'age'].dropna()  
loc:用行列的名称 截取DataFrame数据  
iloc:用矩阵下标 截取DataFrame数据

```python
x_0=titanic[titanic['survived']==0]['age'].dropna()
x_1=titanic.loc[titanic['survived']==1,'age'].dropna()
plt.boxplot([x_0,x_1],labels=['遇难','生还'])
plt.show()
```

![png](https://img2022.cnblogs.com/blog/2629720/202210/2629720-20221010125537809-42632669.png)

④所有生还者中年龄分布情况

```python
x=titanic[titanic['survived']==1]['age'].dropna()
plt.hist(x,edgecolor='red',facecolor='orange',bins=30)
plt.show()
```

![png](https://img2022.cnblogs.com/blog/2629720/202210/2629720-20221010125538145-1483961505.png)

⑤15岁以下生还人员年龄分布

pandas和python不同,没有and not or,要用& | !

```python
x=titanic[(titanic['survived']==1)&(titanic['age']<=15)]['age'].dropna()
plt.hist(x,edgecolor='pink',facecolor='yellow',bins=30)
plt.show()
```

![png](https://img2022.cnblogs.com/blog/2629720/202210/2629720-20221010125538468-1736797508.png)

⑥绘制对比15岁以下遇难和生还的条形图

DataFrame.query()类似sql语句,且有and not or

plt.bar和plt.xtick设置刻度和刻度标签

```python
x0=titanic.query('survived==0 and age<=15')['age']
x1=titanic.query('survived==1 and age<=15')['age']
plt.bar(x=0,height=len(x0),color='red')
plt.bar(x=1,height=len(x1),color='green')
plt.xticks([0,1],['遇难','生还'])
plt.show()
```

![png](https://img2022.cnblogs.com/blog/2629720/202210/2629720-20221010125538791-734742102.png)

###### 2.性别与生还的关系

groupby分析

```python
titanic.groupby(['sex','survived']).count()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>row.names</th>
      <th>pclass</th>
      <th>name</th>
      <th>age</th>
      <th>embarked</th>
      <th>home.dest</th>
      <th>room</th>
      <th>ticket</th>
      <th>boat</th>
    </tr>
    <tr>
      <th>sex</th>
      <th>survived</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">female</th>
      <th>0</th>
      <td>156</td>
      <td>156</td>
      <td>156</td>
      <td>44</td>
      <td>54</td>
      <td>48</td>
      <td>5</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>307</td>
      <td>307</td>
      <td>307</td>
      <td>199</td>
      <td>258</td>
      <td>243</td>
      <td>40</td>
      <td>35</td>
      <td>182</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">male</th>
      <th>0</th>
      <td>708</td>
      <td>708</td>
      <td>708</td>
      <td>308</td>
      <td>405</td>
      <td>358</td>
      <td>17</td>
      <td>20</td>
      <td>75</td>
    </tr>
    <tr>
      <th>1</th>
      <td>142</td>
      <td>142</td>
      <td>142</td>
      <td>82</td>
      <td>104</td>
      <td>105</td>
      <td>15</td>
      <td>13</td>
      <td>86</td>
    </tr>
  </tbody>
</table>
</div>

遇难者中性别统计

```python
x0=titanic[titanic['survived']==0]['sex'].value_counts()
x0
```

    male      708
    female    156
    Name: sex, dtype: int64

生还者,遇难者性别统计饼图

plt.subplot()设置多个图排列方式

```python
x1=titanic[titanic['survived']==1]['sex'].value_counts()
plt.subplot(121)
plt.pie(x1,labels=x1.index,autopct='%1.1f%%',startangle=90)
plt.axis('equal')
plt.title('生还者中性别比例')
plt.subplot(122)
plt.pie(x0,labels=x0.index,autopct='%1.1f%%',startangle=90)
plt.title('遇难者中性别比例')
plt.axis('equal')
plt.show()
```

![png](https://img2022.cnblogs.com/blog/2629720/202210/2629720-20221010125539138-1826268750.png)

###### 3,船舱等级和生还关系

创建交叉表,margins=True表示显示汇总信息

```python
x=pd.crosstab(titanic.pclass,titanic.survived,margins=True)
x
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>survived</th>
      <th>0</th>
      <th>1</th>
      <th>All</th>
    </tr>
    <tr>
      <th>pclass</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1st</th>
      <td>129</td>
      <td>193</td>
      <td>322</td>
    </tr>
    <tr>
      <th>2nd</th>
      <td>161</td>
      <td>119</td>
      <td>280</td>
    </tr>
    <tr>
      <th>3rd</th>
      <td>574</td>
      <td>137</td>
      <td>711</td>
    </tr>
    <tr>
      <th>All</th>
      <td>864</td>
      <td>449</td>
      <td>1313</td>
    </tr>
  </tbody>
</table>
</div>

```python
plt.subplot(131)
plt.pie(x.iloc[0, :2], labels = [' ',' '], autopct = '%1.1f%%')
plt.axis('equal')
plt.title('一等舱生还情况')
plt.subplot(132)
plt.pie(x.iloc[1, :2], labels = [' ',' '], autopct = '%1.1f%%')
plt.axis('equal')
plt.title('二等舱生还情况')
plt.subplot(133)
plt.pie(x.iloc[2, :2], labels = [' ',' '], autopct = '%1.1f%%')
plt.axis('equal')
plt.title('三等舱生还情况')
plt.show()
```

![png](https://img2022.cnblogs.com/blog/2629720/202210/2629720-20221010125539513-980164487.png)

###### 第二步,数据预处理,包括特征选择,填充nan值,数据特征化等

根据我们对事故的了解,sex,age,pclass这些都可能是决定幸免与否的关键因素

```python
x=titanic[['pclass','age','sex']]
y=titanic['survived']
x.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1313 entries, 0 to 1312
    Data columns (total 3 columns):
     #   Column  Non-Null Count  Dtype  
    ---  ------  --------------  -----  
     0   pclass  1313 non-null   object 
     1   age     633 non-null    float64
     2   sex     1313 non-null   object 
    dtypes: float64(1), object(2)
    memory usage: 30.9+ KB

由info()的输出,拟设计如下几个数据预处理  

1) age这个数据列只有633个non-null值,需要对null值进行补完  
2) sex与pclass 两个数据列的值都是非数值型的,需要转为数值,用整数0/1 1/2/

首先补充age里的数据,使用平均数mean()或者中位数median()都是对模型偏离造成最小的策略

```python
x=x.copy()
```

复制一份数据,否则会报错:A value is trying to be set on a copy of a slice from a DataFrame

```python
x['age'].fillna(x['age'].mean(),inplace=True)
```

inplace=True表示原地修改

对补完的数据重新探查

```python
x.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1313 entries, 0 to 1312
    Data columns (total 3 columns):
     #   Column  Non-Null Count  Dtype  
    ---  ------  --------------  -----  
     0   pclass  1313 non-null   object 
     1   age     1313 non-null   float64
     2   sex     1313 non-null   object 
    dtypes: float64(1), object(2)
    memory usage: 30.9+ KB

接着客舱等级,性别分别进行特征化

```python
x.loc[x.sex=='male','sex']=0
x.loc[x.sex=='female','sex']=1
x.loc[x.pclass=='1st','pclass']=1
x.loc[x.pclass=='2nd','pclass']=2
x.loc[x.pclass=='3rd','pclass']=3
```

这里也可以用map函数  

```python
sex_dict={'male':0,'female':1}
x['sex']=x['sex'].map(sex_dict)
```

```python
x.columns
```

    Index(['pclass', 'age', 'sex'], dtype='object')

将数据分割为训练集和测试集

```python
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=33)
```

导入warning库,过滤可能的警告

```python
import warnings
warnings.filterwarnings("ignore")
```

###### 第三步,对模型进行训练,并预测,基本上是:导库-初始化-fit-predict四步

knn法

```python
from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier()
model.fit(x_train,y_train)
y_predict=model.predict(x_test)
```

其他实现算法: 

```python
#决策树
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion = 'entropy' )
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
#随机森林法
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
#线性分类器-逻辑回归分析
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
#分类回归树CART
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
#支持向量机SVM
from sklearn.svm import LinearSVC
model = LinearSVC()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
#朴素贝叶斯
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
```

###### 第四步结果评估

方法1:使用模型自带的评估函数

```python
print(model.score(x_test,y_test))
```

    0.7811550151975684

方法2:使用sklearn.metric里面的classification_report模块对预测结果做更详细的分析

```python
from sklearn.metrics import classification_report
print(classification_report(y_test,y_predict,target_names=['died','survived']))
```

                  precision    recall  f1-score   support
    
            died       0.78      0.90      0.83       202
        survived       0.79      0.59      0.68       127
    
        accuracy                           0.78       329
       macro avg       0.78      0.75      0.76       329
    weighted avg       0.78      0.78      0.77       329

方法3:使用混淆矩阵查看分类效果

```python
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,y_predict))
```

    [[182  20]
     [ 52  75]]

##### 一元线性回归

```python
import numpy as np
import matplotlib.pyplot as plt
import mglearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
```

```python
X, y = mglearn.datasets.make_wave(n_samples=40)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
print(X.shape) 
```

    (40, 1)

拟合x,y

```python
lr = LinearRegression()
lr.fit(X_train, y_train)
```

<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LinearRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">LinearRegression</label><div class="sk-toggleable__content"><pre>LinearRegression()</pre></div></div></div></div></div>

输出模型参数

w are stored in coef_, b is stored in intercept_

```python
print(lr.coef_) #coef_的类型为numpy.ndarray
print(lr.intercept_) #intercept_的类型为numpy.float64
```

    [0.47954524]
    -0.09847983994403892

分别用模型的score函数求训练集和测试集的拟合优度

```python
print('Training set score:{:.3f}'.format(lr.score(X_train, y_train)))
print('test set score:{:.3f}'.format(lr.score(X_test, y_test)))
```

    Training set score:0.653
    test set score:0.773

绘制样本和拟合直线

```python
line = np.linspace(-3,3,1000).reshape(-1,1)
plt.plot(line[:,0], lr.predict(line))
plt.plot(X_train[:,0], y_train, 'o', c='r')
plt.plot(X_test[:,0], y_test, 'v', c='b')
plt.xlabel('Feature')
plt.ylabel('target')
plt.show()
```

![png](https://img2022.cnblogs.com/blog/2629720/202210/2629720-20221010125539901-1698307811.png)

##### 波士顿房价预测

导库导数据

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge
from sklearn.metrics import mean_squared_error
data = load_boston()
```

    c:\environment\python\lib\site-packages\sklearn\utils\deprecation.py:87: FutureWarning: Function load_boston is deprecated; `load_boston` is deprecated in 1.0 and will be removed in 1.2.
    
        The Boston housing prices dataset has an ethical problem. You can refer to
        the documentation of this function for further details.
    
        The scikit-learn maintainers therefore strongly discourage the use of this
        dataset unless the purpose of the code is to study and educate about
        ethical issues in data science and machine learning.
    
        In this special case, you can fetch the dataset from the original
        source::
    
            import pandas as pd
            import numpy as np
    
            data_url = "http://lib.stat.cmu.edu/datasets/boston"
            raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
            data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
            target = raw_df.values[1::2, 2]
    
        Alternative datasets include the California housing dataset (i.e.
        :func:`~sklearn.datasets.fetch_california_housing`) and the Ames housing
        dataset. You can load the datasets as follows::
    
            from sklearn.datasets import fetch_california_housing
            housing = fetch_california_housing()
    
        for the California housing dataset and::
    
            from sklearn.datasets import fetch_openml
            housing = fetch_openml(name="house_prices", as_frame=True)
    
        for the Ames housing dataset.
      warnings.warn(msg, category=FutureWarning)

提示了波士顿房价的数据已经失效了,用新的数据:

```python
import pandas as pd
import numpy as np
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
x = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
y = raw_df.values[1::2, 2]
```

```python
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge
from sklearn.metrics import mean_squared_error
```

```python
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)
```

绘图函数

```python
def plot_diff(y_test, y_pred):
 plt.figure(figsize=(8,5))
 plt.plot(y_test, 'b-', label='y_true')
 plt.plot(y_pred, 'r:', label='y_predict')
 plt.legend()
 plt.show()
```

(1)LinearRegression,线性回归(最小二乘法,用正规方程求解)

```python
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print(' MSE ', mean_squared_error(y_test, y_pred))
plot_diff(y_test, y_pred)
print(lr.coef_, lr.intercept_) 
print('Training set score:{:.3f}'.format(lr.score(X_train, y_train)))
print('test set score:{:.3f}'.format(lr.score(X_test, y_test)))
```

     MSE  29.78224509230234

![png](https://img2022.cnblogs.com/blog/2629720/202210/2629720-20221010125540344-665439963.png)

    [-0.97100092  1.04667838 -0.04044753  0.59408776 -1.80876877  2.60991991
     -0.19823317 -3.00216551  2.08021582 -1.93289037 -2.15743759  0.75199122
     -3.59027047] 22.6087071240106
    Training set score:0.770
    test set score:0.635

(2)SGDRegressor随机梯度下降回归,sklearn建议当样本数大于10万条时推荐用该算法

```python
sg = SGDRegressor()
sg.fit(X_train, y_train)
y_pred = sg.predict(X_test)
print('SGD MSE ', mean_squared_error(y_test, y_pred))
plot_diff(y_test[:30], y_pred[:30])
print(sg.coef_, sg.intercept_) #输出模型参数
#分别计算训练集和测试集的拟合优度
print('Training set score:{:.3f}'.format(sg.score(X_train, y_train)))
print('test score:{:.3f}'.format(sg.score(X_test, y_test)))
```

    SGD MSE  30.259089282498678

![png](https://img2022.cnblogs.com/blog/2629720/202210/2629720-20221010125540796-1523434037.png)

    [-0.92911005  0.94412689 -0.27009504  0.65094279 -1.67117324  2.67070413
     -0.2352616  -2.90654782  1.52022938 -1.33845261 -2.13156707  0.75031598
     -3.57717671] [22.61254595]
    Training set score:0.769
    test score:0.630

###### 什么时过拟合 和 欠拟合?

学习器把训练样本学得太好了的时候,可能把训练样本自身的一些特点当作了所有潜在的样本都会具有的一般性质,这样就会导致泛化能力下降,这种现象在机器学习中被称为"过拟合"(overfitting).  
欠拟合(underfitting)是指学习能力低下,对训练样本的一般性质尚未学好

岭回归在线性回归的基础上添加了额外的约束条件,让w1,w2.....wn这些系数,尽可能变小,使每一个特征尽可能少地影响输出结果,即w系数都接近0,但不是0  
约束条件通常是正则化,约束一个模型以避免过拟合,岭回归采用L2正则法,就是在损失函数后面添加L2范数

(3)岭回归Ridge--带有L2正则化的线性回归

```python
rd = Ridge()
rd.fit(X_train, y_train)
y_pred = rd.predict(X_test)
print(' MSE ', mean_squared_error(y_test, y_pred))
plot_diff(y_test[:30], y_pred[:30])
print(rd.coef_, rd.intercept_)
print('Training set score:{:.3f}'.format(rd.score(X_train, y_train)))
print('test set score:{:.3f}'.format(rd.score(X_test, y_test)))
```

     MSE  29.853763334547615

![png](https://img2022.cnblogs.com/blog/2629720/202210/2629720-20221010125541176-870144643.png)

    [-0.96187481  1.02775462 -0.06861144  0.59814087 -1.77318401  2.6205672
     -0.20466821 -2.96504904  2.00091047 -1.85840697 -2.14955893  0.75175979
     -3.57350065] 22.6087071240106
    Training set score:0.770
    test set score:0.635

##### 过拟合和岭回归

```python
from mglearn.datasets import load_extended_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Ridge
X, y = load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=40)
```

(1)线性回归LinearRegression

```python
lr = LinearRegression()
lr.fit(X_train, y_train)
print(lr.coef_, lr.intercept_) 
print('Training set score:{:.3f}'.format(lr.score(X_train, y_train)))
print('test set score:{:.3f}'.format(lr.score(X_test, y_test)))
```

    [-1.24333702e+02  8.51109172e-01 -8.78824412e+01  7.81825048e+00
     -4.06861380e+01  2.61890125e+01  5.47007747e+01 -6.90846361e+01
      5.01049890e+01  2.81858069e+01 -1.29429225e+01  4.81643468e+01
      6.04965107e+01  5.50775611e+01  1.81667676e+03  2.51073616e+02
      4.21555936e+02 -8.23756618e+01  2.68876268e+02 -1.07656749e+02
     -3.77037674e+02 -3.83383088e+02  3.51523319e+02 -7.35748233e+00
      2.67286847e+01  5.65099476e+01 -5.50667199e+00 -1.17926994e+01
     -1.28473035e+01 -2.76838844e+01 -2.87669962e+00 -3.85586219e-01
     -4.86621419e+00 -1.85513649e+01  2.19279307e+01 -8.67842854e+00
      1.76334139e+01 -2.61834116e+01  2.16356003e+01  2.99410370e+00
      1.04966826e+01  4.01265710e+01  1.63439867e+01  5.61180816e+01
     -1.91007517e+01  1.71622607e+01  4.10508880e+00  2.18656074e+01
      2.81273665e+00  7.81825048e+00 -1.18525934e+01 -3.62608172e+01
      3.05044951e+00  3.65981310e+01 -3.33277779e+00 -9.46859265e+00
     -9.73239134e+00  9.48538941e+00 -2.02269638e+01  2.06984374e+01
      2.40062595e+01 -2.30341997e+01  1.10903548e+02 -6.00556451e+00
      1.84622075e+01 -2.29697796e+01  2.29081600e+00  2.38457928e+01
      2.72633112e+01 -3.62438600e+01  2.99700977e+01 -2.37983636e+01
     -3.88143747e+01 -6.99662703e+00  1.07015056e+01 -7.44225841e+01
      4.91085514e+00 -6.78709407e+00  2.87845206e+01 -2.59067397e+01
     -2.42083666e+00 -3.87810818e+01 -2.28957639e+01  6.78634328e+01
     -2.08310147e+01 -2.57250801e+01 -5.43081961e+00 -3.02197012e+01
      4.93010816e+01 -7.63875125e+01  9.45133507e+01 -2.58006003e+01
     -7.40603658e+00 -1.00712059e+01 -1.24819910e+01  2.21603357e+01
     -2.13920836e+01 -2.88530760e+01  1.45291424e+00  1.89124871e+01
      5.70082642e+00 -9.67164307e+00 -2.30418042e+01 -1.06648201e+01] -17.07185865961382
    Training set score:0.927
    test set score:0.176

(2)岭回归Ridge

```python
rd = Ridge()
rd.fit(X_train, y_train)
print(rd.coef_, rd.intercept_) 
print('Training set score:{:.3f}'.format(rd.score(X_train, y_train)))
print('test set score:{:.3f}'.format(rd.score(X_test, y_test)))
```

    [-1.68841751e+00 -1.21756532e+00 -2.27229851e+00  8.29280456e-01
     -7.80743588e-02  8.01618677e+00 -2.42717943e-03 -4.69537735e+00
      3.64484835e+00 -1.77450946e+00 -1.84658519e+00  2.34935270e+00
     -2.79784030e+00 -1.03828380e+00  5.46354029e-03 -1.04992699e+00
      1.58455850e+00 -1.63046007e+00 -1.31404896e+00 -1.22931685e+00
     -3.40603690e-01 -1.96170505e+00 -1.69996969e+00 -1.54520138e+00
     -1.24905930e+00 -1.16408825e+00  1.86922579e+00 -1.84135671e+00
      2.15195000e+00  3.35085730e-01  3.44036564e+00 -1.95818731e+00
     -4.40322283e-01 -1.85312908e-01  3.52398000e-01  6.53546832e-01
     -1.00762321e+00 -1.77017868e+00  3.42121918e+00  1.40858792e+00
      8.78498660e-01 -3.91280661e+00  2.17738798e+00 -3.71752448e+00
      1.09951744e+00  2.54048858e+00 -1.16310771e+00 -5.71331629e-01
     -1.92919163e+00  8.29280456e-01 -4.89291341e+00 -3.66126587e+00
      1.33975269e+00 -1.17914224e+00  2.83125614e+00  3.21978680e+00
      3.94302777e-01  1.81255267e+00 -2.30766974e+00 -1.67414246e+00
     -3.53533236e+00 -2.17067231e+00 -9.25952604e-01 -2.49153483e+00
     -1.56089350e+00 -2.84270976e+00  1.55929616e+00 -1.87680529e+00
      1.49432569e+01 -3.73266046e-01  1.24079331e+00 -6.62933260e+00
     -7.51009416e+00 -4.29159411e+00  1.07477261e+01 -6.26321565e+00
      1.33172263e+00 -1.83931263e+00  3.61427066e+00  4.13554858e-01
     -1.69295163e+00 -1.99290831e+00 -3.25029990e+00  1.10791584e+00
     -1.59379257e+00 -2.74743398e+00 -7.72288993e-01 -3.81884579e+00
      3.77400765e-01  7.07738434e-01  7.60386326e-01  2.49785438e+00
      2.58562952e+00 -6.27464245e+00  1.51479963e+00  1.18250473e+00
     -1.16152937e+00 -6.38485861e+00  6.47776806e-01 -1.79770156e+00
     -1.71496657e+00 -1.00265130e+00 -5.90324030e+00  6.40531651e+00] 21.86084571054877
    Training set score:0.834
    test set score:0.860

Lasso回归是使用L1正则化的线性回归,相当于最小二乘法+L1范数

(3)Lasso回归

```python
from sklearn.linear_model import Lasso
lasso = Lasso()
lasso.fit(X_train, y_train)
print(lasso.coef_, lasso.intercept_) 
print('Training set score:{:.3f}'.format(lasso.score(X_train, y_train)))
print('test set score:{:.3f}'.format(lasso.score(X_test, y_test)))
```

    [-0.          0.         -0.          0.         -0.          0.
     -0.          0.         -0.         -0.         -0.          0.
     -0.32441098 -0.          0.         -0.          0.         -0.
     -0.         -0.         -0.         -0.         -0.         -0.
     -0.         -0.          0.          0.          0.          0.
      0.          0.          0.          0.          0.          0.
      0.          0.         -0.          0.         -0.         -0.
     -0.         -0.         -0.         -0.         -0.         -0.
     -0.          0.          0.          0.          0.          0.
      0.          0.          0.          0.          0.         -0.
     -0.         -0.         -0.         -0.         -0.         -0.
     -0.         -0.          0.          0.          0.         -0.
     -0.         -0.          0.         -0.         -0.         -0.
     -0.         -2.73694035 -0.73382069 -0.         -0.          0.
     -0.         -0.         -0.          0.         -0.         -0.
     -0.         -0.         -0.         -0.         -0.         -0.
     -0.         -0.         -0.         -0.         -0.          0.
     -0.         -0.        ] 23.777825319453992
    Training set score:0.120
    test set score:0.102

可以看到模型出现了欠拟合,实例化时Lasso()有正则化参数alpha,默认值alpha=1,alpha值越大则将w推向0的力度越大,而更低的alpha值将拟合出更复杂的模型,将alpha设置成0.01,max_iter=10000,iter_是迭代器迭代次数.Lasso因损失函数不是连续可导的,因此无法使用梯度下降法,常用坐标轴下降法或最小角回归法迭代求解

```python
lasso = Lasso(alpha=0.01, max_iter=10000)
lasso.fit(X_train, y_train)
print(lasso.coef_, lasso.intercept_) 
print('Training set score:{:.3f}'.format(lasso.score(X_train, y_train)))
print('test set score:{:.3f}'.format(lasso.score(X_test, y_test)))
```

    [ -0.          -0.          -0.           0.          -0.
       0.          -0.          -7.83175845   7.72112291   0.
      -0.           0.          -0.          -0.          -0.
      -0.           0.          -0.          -0.          -0.
      -0.          -7.57528476  -0.          -0.          -0.
      -0.           1.01619811  -0.           0.          -0.
       0.          -0.77056513   0.          -0.           0.
       0.          -0.          -0.           0.           0.
       0.          -0.           0.          -8.5544616    0.
       6.49229073  -0.          -0.          -0.           0.
      -4.86695189  -3.15397475   0.          -0.           0.
       7.74739138   0.           3.37036655  -0.          -1.01945969
      -4.67711816  -1.05909637  -0.          -0.          -0.
      -4.79229223   0.          -0.          26.4330005   -0.64282018
      -0.          -8.48483347 -11.5520677   -9.7055464   14.92670387
      -6.29599732  -0.          -0.29677602   4.57239306   0.
      -0.27696565  -2.111057    -0.96785033   0.          -0.
      -0.           0.          -0.           0.           0.
       0.           0.           0.45673343  -6.53072993   1.99027172
       0.           0.         -16.54235847   0.           0.
      -0.          -0.          -8.50513944   7.61285648] 20.64902255263751
    Training set score:0.841
    test set score:0.882

##### 信用卡欺诈检测

logistic回归是机器学习中常用的二分类方法之一.虽然被称为回归模型,但是处理分类问题,它的本质是一个线性模型加上一个映射函数Sigmoid,将任意的输入映射到了[0,1]区间,从而完成了由值到概率的转换,概率值大于0.5的数据被分入1类

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
```

(1)读取数据,搜索数据

```python
data = pd.read_csv("./creditcard.csv")
data.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Time</th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>...</th>
      <th>V21</th>
      <th>V22</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Amount</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>-1.359807</td>
      <td>-0.072781</td>
      <td>2.536347</td>
      <td>1.378155</td>
      <td>-0.338321</td>
      <td>0.462388</td>
      <td>0.239599</td>
      <td>0.098698</td>
      <td>0.363787</td>
      <td>...</td>
      <td>-0.018307</td>
      <td>0.277838</td>
      <td>-0.110474</td>
      <td>0.066928</td>
      <td>0.128539</td>
      <td>-0.189115</td>
      <td>0.133558</td>
      <td>-0.021053</td>
      <td>149.62</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>1.191857</td>
      <td>0.266151</td>
      <td>0.166480</td>
      <td>0.448154</td>
      <td>0.060018</td>
      <td>-0.082361</td>
      <td>-0.078803</td>
      <td>0.085102</td>
      <td>-0.255425</td>
      <td>...</td>
      <td>-0.225775</td>
      <td>-0.638672</td>
      <td>0.101288</td>
      <td>-0.339846</td>
      <td>0.167170</td>
      <td>0.125895</td>
      <td>-0.008983</td>
      <td>0.014724</td>
      <td>2.69</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>-1.358354</td>
      <td>-1.340163</td>
      <td>1.773209</td>
      <td>0.379780</td>
      <td>-0.503198</td>
      <td>1.800499</td>
      <td>0.791461</td>
      <td>0.247676</td>
      <td>-1.514654</td>
      <td>...</td>
      <td>0.247998</td>
      <td>0.771679</td>
      <td>0.909412</td>
      <td>-0.689281</td>
      <td>-0.327642</td>
      <td>-0.139097</td>
      <td>-0.055353</td>
      <td>-0.059752</td>
      <td>378.66</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>-0.966272</td>
      <td>-0.185226</td>
      <td>1.792993</td>
      <td>-0.863291</td>
      <td>-0.010309</td>
      <td>1.247203</td>
      <td>0.237609</td>
      <td>0.377436</td>
      <td>-1.387024</td>
      <td>...</td>
      <td>-0.108300</td>
      <td>0.005274</td>
      <td>-0.190321</td>
      <td>-1.175575</td>
      <td>0.647376</td>
      <td>-0.221929</td>
      <td>0.062723</td>
      <td>0.061458</td>
      <td>123.50</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.0</td>
      <td>-1.158233</td>
      <td>0.877737</td>
      <td>1.548718</td>
      <td>0.403034</td>
      <td>-0.407193</td>
      <td>0.095921</td>
      <td>0.592941</td>
      <td>-0.270533</td>
      <td>0.817739</td>
      <td>...</td>
      <td>-0.009431</td>
      <td>0.798278</td>
      <td>-0.137458</td>
      <td>0.141267</td>
      <td>-0.206010</td>
      <td>0.502292</td>
      <td>0.219422</td>
      <td>0.215153</td>
      <td>69.99</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 31 columns</p>
</div>

```python
data.isnull().values.sum()
```

    0

绘制条形图

```python
count_classes = pd.value_counts(data['Class'], sort = True).sort_index()
print(count_classes)
```

    0    284315
    1       492
    Name: Class, dtype: int64

```python
count_classes.plot(kind = 'bar', title='Fraud class histogram', rot=360) 
plt.show()
```

![png](https://img2022.cnblogs.com/blog/2629720/202210/2629720-20221010125541503-1315835844.png)

可以看到,由于数据量差距特别大,导致1的条形图无法显示

(2)数据的预处理  

采样,确定建模数据.由于交易类型分布不平衡,不能简单粗暴地将原始数据切分成训练集和测试集.一种想法是:按标签为1的样本的样本数对标签为0的样本集进行1:1随机采样确定将来参与建模的全部样本,即随机下采用法

```python
n = len(data[data.Class == 1])
indices_of_1 = np.array(data[data.Class == 1].index) 
indices_of_0 = data[data.Class == 0].index 
indices_of_0 = np.random.choice(indices_of_0, n, replace = False) 
indices = np.concatenate([indices_of_1,indices_of_0], axis=0) 
under_sample_data = data.iloc[indices,:] 
X_undersample = under_sample_data.loc[:,under_sample_data.columns != 'Class']
y_undersample = under_sample_data['Class']
```

划分训练集和测试集

```python
X_train, X_test, y_train, y_test = train_test_split(X_undersample, y_undersample,
test_size=0.3, random_state=0) 
```

对Amount特征进行标准化

```python
X_train = X_train.copy()
X_test = X_test.copy()
ss = StandardScaler()
X_train['normAmount'] = ss.fit_transform(X_train['Amount'].values.reshape(-1,1))
X_test['normAmount'] = ss.transform(X_test['Amount'].values.reshape(-1,1))
X_train = X_train.drop(['Time', 'Amount'], axis=1)
X_test = X_test.drop(['Time', 'Amount'], axis=1)
print(X_train)
```

                  V1        V2        V3        V4        V5        V6        V7  \
    6870   -1.863756  3.442644 -4.468260  2.805336 -2.118412 -2.332285 -4.261237   
    150791  2.214237 -0.792902 -1.473319 -0.910354 -0.099301 -0.165655 -0.668552   
    29464   1.213589 -0.741135  0.406263 -0.483767 -1.292866 -1.229874 -0.366938   
    214775 -0.395582 -0.751792 -1.984666 -0.203459  1.903967 -1.430289 -0.076548   
    149145 -2.405580  3.738235 -2.317843  1.367442  0.394001  1.919938 -3.106942   
    ...          ...       ...       ...       ...       ...       ...       ...   
    218882 -6.650873 -5.166471  0.630071  1.977026  7.226284 -4.745045 -2.538954   
    79536  -0.264869  3.386140 -3.454997  4.367629  3.336060 -2.053918  0.256890   
    193748  2.313852 -1.447755 -1.107011 -1.572196 -1.228872 -0.615661 -1.194617   
    84981  -0.775805  0.699239  2.156244 -0.505637 -0.285467 -0.753034  0.425022   
    130031  1.485722 -0.329184 -0.650416 -1.049783  0.040445 -0.400925 -0.113134   
    
                   V8        V9       V10  ...       V20        V21       V22  \
    6870     1.701682 -1.439396 -6.999907  ...  0.360924   0.667927 -0.516242   
    150791  -0.078229  0.966791  0.590159  ... -0.135084  -0.194810 -0.403193   
    29464   -0.299762 -0.834289  0.599360  ...  0.261264   0.380608  0.852311   
    214775  -0.992260  0.756307  0.217630  ... -1.027716   1.377515  2.151787   
    149145 -10.764403  3.353525  0.369936  ... -2.140874  10.005998 -2.454964   
    ...           ...       ...       ...  ...       ...        ...       ...   
    218882  -1.327611  1.026117  1.740803  ... -4.302659  -1.376437  0.010657   
    79536   -2.957235 -2.855797 -2.808456  ...  0.482513  -1.394504 -0.166029   
    193748  -0.075333 -0.943735  1.627065  ... -0.559194  -0.198388 -0.157568   
    84981    0.071048 -0.089360 -0.633079  ...  0.138920  -0.085608 -0.291283   
    130031  -0.223569 -1.421063  0.839325  ...  0.153764   0.172964  0.429903   
    
                 V23       V24       V25       V26       V27       V28  normAmount  
    6870   -0.012218  0.070614  0.058504  0.304883  0.418012  0.208858   -0.437003  
    150791  0.201311 -0.011091 -0.079658 -0.375434 -0.079145 -0.079314   -0.377512  
    29464  -0.194582  0.768330  0.573438 -0.071189 -0.012877  0.032184   -0.024683  
    214775  0.189225  0.772943 -0.872443 -0.200612  0.356856  0.032113   -0.438240  
    149145  1.684957  0.118263 -1.531380 -0.695308 -0.152502 -0.138866   -0.413087  
    ...          ...       ...       ...       ...       ...       ...         ...  
    218882 -3.912938  0.437714  0.691167 -0.355594  0.612076 -0.184440   -0.284843  
    79536  -1.452081 -0.251815  1.243461  0.452787  0.132218  0.424599   -0.437003  
    193748  0.293418  0.600750 -0.222054 -0.181151 -0.004996 -0.050555   -0.401069  
    84981  -0.099654  0.393402  0.298088  0.259466  0.167930  0.043202   -0.334112  
    130031 -0.329438 -0.778603  0.937728 -0.031974 -0.030576 -0.018621   -0.381106  
    
    [688 rows x 29 columns]

```python
print(X_test)
```

                  V1        V2        V3        V4        V5        V6         V7  \
    102782  1.232604 -0.548931  1.087873  0.894082 -1.433055 -0.356797  -0.717492   
    15452  -1.957589 -1.730984  2.629301 -1.206348  0.527344 -1.590407  -1.547321   
    8335   -1.426623  4.141986 -9.804103  6.666273 -4.749527 -2.073129 -10.089931   
    203421 -0.276322  0.847457  0.897515 -0.775907  0.036237 -0.803819   0.404633   
    215492  1.801819 -1.049583 -2.711297 -0.449081  0.285518 -0.942887   0.727762   
    ...          ...       ...       ...       ...       ...       ...        ...   
    201098  1.176633  3.141918 -6.140445  5.521821  1.768515 -1.727186  -0.932429   
    125794 -2.202432 -0.954751 -0.297493 -1.543780  0.285192 -0.660060   1.350899   
    138819 -0.237746  1.334937  0.131449 -0.042921  0.606135 -1.761389   1.159858   
    121044  1.082993 -0.006890  0.711442  0.938848 -0.592040 -0.521089   0.016720   
    184455  2.002334 -1.326313  0.091723 -0.359515 -1.276627  0.922769  -1.629738   
    
                  V8        V9        V10  ...       V20       V21       V22  \
    102782  0.003167 -0.100397   0.543187  ... -0.576274 -0.448671 -0.517568   
    15452   0.339491 -0.416022   0.095897  ...  0.644220  0.445863  0.825875   
    8335    2.791345 -3.249516 -11.420451  ...  1.410678  1.865679  0.407809   
    203421  0.102078  0.695189  -1.298455  ... -0.023836 -0.301246 -0.720473   
    215492 -0.515704 -1.505695   1.087313  ... -0.155287  0.081905  0.348247   
    ...          ...       ...        ...  ...       ...       ...       ...   
    201098  0.292797 -3.156827  -3.898240  ...  0.329568  0.129372 -0.803021   
    125794 -0.060076  0.279917  -1.016425  ... -0.194427 -0.045001  0.065485   
    138819 -0.371936 -0.514705  -1.083563  ... -0.142624  0.082596  0.237615   
    121044 -0.089600  0.294218  -0.279309  ... -0.043723 -0.138192 -0.222806   
    184455  0.377938  0.592580   0.753866  ... -0.495763 -0.390187 -0.384427   
    
                 V23       V24       V25       V26       V27       V28  normAmount  
    102782  0.012833  0.699217  0.527258 -0.322607  0.080805  0.035427   -0.362779  
    15452  -0.050376  0.410972  0.588094 -0.110504  0.241444  0.161709   -0.402586  
    8335    0.605809 -0.769348 -1.746337  0.502040  1.977258  0.711607   -0.437003  
    203421 -0.065469 -0.272522 -0.180217 -0.243681  0.281665  0.103496   -0.435166  
    215492 -0.248497  0.797162  0.406421  0.945664 -0.165443 -0.057660    0.444252  
    ...          ...       ...       ...       ...       ...       ...         ...  
    201098 -0.074098 -0.031084  0.375366  0.065897  0.488258  0.325872   -0.440995  
    125794 -0.435514 -0.310231 -0.506112  0.989468  0.238861 -0.560751    0.722300  
    138819  0.043939  0.600666 -0.976644 -0.013702  0.050118  0.270443   -0.437961  
    121044  0.066565  0.662879  0.324867  0.259481 -0.006393  0.022817   -0.271468  
    184455  0.323623  0.365166 -0.568674  0.497966  0.022897 -0.037061   -0.319419  
    
    [296 rows x 29 columns]

(3)建模--信用卡欺诈检测

```python
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_predict = lr.predict(X_test)
```

(4)评估

准确率

```python
print('Accuracy of LogisticRegression is:', lr.score(X_test, y_test))
```

    Accuracy of LogisticRegression is: 0.9391891891891891

分类报告:

```python
print(classification_report(y_test, y_predict, target_names=['Normal','Fraud']))
```

                  precision    recall  f1-score   support
    
          Normal       0.93      0.95      0.94       149
           Fraud       0.95      0.93      0.94       147
    
        accuracy                           0.94       296
       macro avg       0.94      0.94      0.94       296
    weighted avg       0.94      0.94      0.94       296

#### python深度学习

##### 用神经网络预测酸奶销量

①预测酸奶日销量1

```python
import numpy as np
import tensorflow as tf
```

自造数据集

```python
np.random.seed(50)
x = np.random.rand(32, 2)
np.random.seed(50)
y_ = [[x1 + x2 + (np.random.rand()/10 - 0.05)] for (x1, x2) in x]
w = tf.Variable(tf.random.normal([2,1], stddev=1, seed=1))
x = tf.cast(x, dtype=tf.float32)
epochs = 15000 #迭代次数
lr = 0.001 #学习率
for epoch in range(epochs):
    with tf.GradientTape() as tape:
        y = tf.matmul(x, w)
        loss_mse = tf.reduce_mean(tf.square(y_ - y)) 
    grads = tape.gradient(loss_mse, w)
    w.assign_sub(lr*grads)
    if epoch % 500 == 0:
        print('After %d training steps, w is' %epoch)
        print(w)
print('Final w is:', w)
```

    After 0 training steps, w is
    <tf.Variable 'Variable:0' shape=(2, 1) dtype=float32, numpy=
    array([[-0.8101696],
           [ 1.485255 ]], dtype=float32)>
    After 500 training steps, w is
    <tf.Variable 'Variable:0' shape=(2, 1) dtype=float32, numpy=
    array([[-0.36370227],
           [ 1.7091383 ]], dtype=float32)>
    After 1000 training steps, w is
    <tf.Variable 'Variable:0' shape=(2, 1) dtype=float32, numpy=
    array([[-0.09182037],
           [ 1.7943552 ]], dtype=float32)>
    After 1500 training steps, w is
    <tf.Variable 'Variable:0' shape=(2, 1) dtype=float32, numpy=
    array([[0.08338203],
           [1.8075756 ]], dtype=float32)>
    After 2000 training steps, w is
    <tf.Variable 'Variable:0' shape=(2, 1) dtype=float32, numpy=
    array([[0.20414291],
           [1.7845508 ]], dtype=float32)>
    After 2500 training steps, w is
    <tf.Variable 'Variable:0' shape=(2, 1) dtype=float32, numpy=
    array([[0.29343665],
           [1.7443693 ]], dtype=float32)>
    After 3000 training steps, w is
    <tf.Variable 'Variable:0' shape=(2, 1) dtype=float32, numpy=
    array([[0.3638279],
           [1.6971394]], dtype=float32)>
    After 3500 training steps, w is
    <tf.Variable 'Variable:0' shape=(2, 1) dtype=float32, numpy=
    array([[0.4222504],
           [1.6481372]], dtype=float32)>
    After 4000 training steps, w is
    <tf.Variable 'Variable:0' shape=(2, 1) dtype=float32, numpy=
    array([[0.47258648],
           [1.6000549 ]], dtype=float32)>
    After 4500 training steps, w is
    <tf.Variable 'Variable:0' shape=(2, 1) dtype=float32, numpy=
    array([[0.51706064],
           [1.5541908 ]], dtype=float32)>
    After 5000 training steps, w is
    <tf.Variable 'Variable:0' shape=(2, 1) dtype=float32, numpy=
    array([[0.55699277],
           [1.5111096 ]], dtype=float32)>
    After 5500 training steps, w is
    <tf.Variable 'Variable:0' shape=(2, 1) dtype=float32, numpy=
    array([[0.59320575],
           [1.4709893 ]], dtype=float32)>
    After 6000 training steps, w is
    <tf.Variable 'Variable:0' shape=(2, 1) dtype=float32, numpy=
    array([[0.6262438],
           [1.433812 ]], dtype=float32)>
    After 6500 training steps, w is
    <tf.Variable 'Variable:0' shape=(2, 1) dtype=float32, numpy=
    array([[0.65649307],
           [1.3994584 ]], dtype=float32)>
    After 7000 training steps, w is
    <tf.Variable 'Variable:0' shape=(2, 1) dtype=float32, numpy=
    array([[0.6842485],
           [1.3677669]], dtype=float32)>
    After 7500 training steps, w is
    <tf.Variable 'Variable:0' shape=(2, 1) dtype=float32, numpy=
    array([[0.70974827],
           [1.3385594 ]], dtype=float32)>
    After 8000 training steps, w is
    <tf.Variable 'Variable:0' shape=(2, 1) dtype=float32, numpy=
    array([[0.73319244],
           [1.3116561 ]], dtype=float32)>
    After 8500 training steps, w is
    <tf.Variable 'Variable:0' shape=(2, 1) dtype=float32, numpy=
    array([[0.75475615],
           [1.286884  ]], dtype=float32)>
    After 9000 training steps, w is
    <tf.Variable 'Variable:0' shape=(2, 1) dtype=float32, numpy=
    array([[0.7745957],
           [1.2640779]], dtype=float32)>
    After 9500 training steps, w is
    <tf.Variable 'Variable:0' shape=(2, 1) dtype=float32, numpy=
    array([[0.79285145],
           [1.2430849 ]], dtype=float32)>
    After 10000 training steps, w is
    <tf.Variable 'Variable:0' shape=(2, 1) dtype=float32, numpy=
    array([[0.8096513],
           [1.2237617]], dtype=float32)>
    After 10500 training steps, w is
    <tf.Variable 'Variable:0' shape=(2, 1) dtype=float32, numpy=
    array([[0.82511204],
           [1.2059764 ]], dtype=float32)>
    After 11000 training steps, w is
    <tf.Variable 'Variable:0' shape=(2, 1) dtype=float32, numpy=
    array([[0.83934104],
           [1.1896064 ]], dtype=float32)>
    After 11500 training steps, w is
    <tf.Variable 'Variable:0' shape=(2, 1) dtype=float32, numpy=
    array([[0.85243684],
           [1.1745398 ]], dtype=float32)>
    After 12000 training steps, w is
    <tf.Variable 'Variable:0' shape=(2, 1) dtype=float32, numpy=
    array([[0.8644893],
           [1.1606735]], dtype=float32)>
    After 12500 training steps, w is
    <tf.Variable 'Variable:0' shape=(2, 1) dtype=float32, numpy=
    array([[0.875582 ],
           [1.1479106]], dtype=float32)>
    After 13000 training steps, w is
    <tf.Variable 'Variable:0' shape=(2, 1) dtype=float32, numpy=
    array([[0.88579136],
           [1.1361648 ]], dtype=float32)>
    After 13500 training steps, w is
    <tf.Variable 'Variable:0' shape=(2, 1) dtype=float32, numpy=
    array([[0.89518756],
           [1.1253542 ]], dtype=float32)>
    After 14000 training steps, w is
    <tf.Variable 'Variable:0' shape=(2, 1) dtype=float32, numpy=
    array([[0.90383554],
           [1.1154041 ]], dtype=float32)>
    After 14500 training steps, w is
    <tf.Variable 'Variable:0' shape=(2, 1) dtype=float32, numpy=
    array([[0.9117948],
           [1.106246 ]], dtype=float32)>
    Final w is: <tf.Variable 'Variable:0' shape=(2, 1) dtype=float32, numpy=
    array([[0.91910625],
           [1.0978338 ]], dtype=float32)>

迭代15000次后,w的值array[\[0.91910625\],[1.0978338 ]]

②预测酸奶日销量2---自定义损失函数

```python
import numpy as np
import tensorflow as tf
np.random.seed(50) 
x = np.random.rand(32, 2)
np.random.seed(50) 
y_ = [[x1 + x2 + (np.random.rand()/10-0.05)] for (x1, x2) in x]
w = tf.Variable(tf.random.normal([2,1], stddev=1,seed=1)) 
x = tf.cast(x, dtype=tf.float32)
epochs = 15000
lr = 0.001
for epoch in range(epochs):
    with tf.GradientTape() as tape:
        y = tf.matmul(x, w) 
        loss_zdy = tf.reduce_sum(tf.where(tf.greater(y, y_), (y-y_)*1, (y_-y)*4)) #
    grads = tape.gradient(loss_zdy, w) 
    w.assign_sub(lr*grads) 
    if epoch % 500 == 0:
        print('After %d training steps, w is' %epoch)
        print(w)
print('Final w is:', w)
```

    After 0 training steps, w is
    <tf.Variable 'Variable:0' shape=(2, 1) dtype=float32, numpy=
    array([[1.370413 ],
           [0.5765818]], dtype=float32)>
    After 500 training steps, w is
    <tf.Variable 'Variable:0' shape=(2, 1) dtype=float32, numpy=
    array([[1.0180757],
           [1.0245619]], dtype=float32)>
    After 1000 training steps, w is
    <tf.Variable 'Variable:0' shape=(2, 1) dtype=float32, numpy=
    array([[1.0155951],
           [1.0255655]], dtype=float32)>
    After 1500 training steps, w is
    <tf.Variable 'Variable:0' shape=(2, 1) dtype=float32, numpy=
    array([[1.021677 ],
           [1.0315355]], dtype=float32)>
    After 2000 training steps, w is
    <tf.Variable 'Variable:0' shape=(2, 1) dtype=float32, numpy=
    array([[1.0152975],
           [1.0235618]], dtype=float32)>
    After 2500 training steps, w is
    <tf.Variable 'Variable:0' shape=(2, 1) dtype=float32, numpy=
    array([[1.0181403],
           [1.0245162]], dtype=float32)>
    After 3000 training steps, w is
    <tf.Variable 'Variable:0' shape=(2, 1) dtype=float32, numpy=
    array([[1.0163339],
           [1.0265371]], dtype=float32)>
    After 3500 training steps, w is
    <tf.Variable 'Variable:0' shape=(2, 1) dtype=float32, numpy=
    array([[1.0176959],
           [1.0252694]], dtype=float32)>
    After 4000 training steps, w is
    <tf.Variable 'Variable:0' shape=(2, 1) dtype=float32, numpy=
    array([[1.0168865],
           [1.0268477]], dtype=float32)>
    After 4500 training steps, w is
    <tf.Variable 'Variable:0' shape=(2, 1) dtype=float32, numpy=
    array([[1.0151399],
           [1.0238928]], dtype=float32)>
    After 5000 training steps, w is
    <tf.Variable 'Variable:0' shape=(2, 1) dtype=float32, numpy=
    array([[1.0149177],
           [1.0242693]], dtype=float32)>
    After 5500 training steps, w is
    <tf.Variable 'Variable:0' shape=(2, 1) dtype=float32, numpy=
    array([[1.0161328],
           [1.0257585]], dtype=float32)>
    After 6000 training steps, w is
    <tf.Variable 'Variable:0' shape=(2, 1) dtype=float32, numpy=
    array([[1.0190191],
           [1.0278224]], dtype=float32)>
    After 6500 training steps, w is
    <tf.Variable 'Variable:0' shape=(2, 1) dtype=float32, numpy=
    array([[1.0135769],
           [1.0243819]], dtype=float32)>
    After 7000 training steps, w is
    <tf.Variable 'Variable:0' shape=(2, 1) dtype=float32, numpy=
    array([[1.0164632],
           [1.0264457]], dtype=float32)>
    After 7500 training steps, w is
    <tf.Variable 'Variable:0' shape=(2, 1) dtype=float32, numpy=
    array([[1.0147166],
           [1.0234908]], dtype=float32)>
    After 8000 training steps, w is
    <tf.Variable 'Variable:0' shape=(2, 1) dtype=float32, numpy=
    array([[1.0183642],
           [1.0287908]], dtype=float32)>
    After 8500 training steps, w is
    <tf.Variable 'Variable:0' shape=(2, 1) dtype=float32, numpy=
    array([[1.0166177],
           [1.0258359]], dtype=float32)>
    After 9000 training steps, w is
    <tf.Variable 'Variable:0' shape=(2, 1) dtype=float32, numpy=
    array([[1.0163954],
           [1.0262125]], dtype=float32)>
    After 9500 training steps, w is
    <tf.Variable 'Variable:0' shape=(2, 1) dtype=float32, numpy=
    array([[1.0169363],
           [1.0266844]], dtype=float32)>
    After 10000 training steps, w is
    <tf.Variable 'Variable:0' shape=(2, 1) dtype=float32, numpy=
    array([[1.0166706],
           [1.0259515]], dtype=float32)>
    After 10500 training steps, w is
    <tf.Variable 'Variable:0' shape=(2, 1) dtype=float32, numpy=
    array([[1.0157741],
           [1.0253109]], dtype=float32)>
    After 11000 training steps, w is
    <tf.Variable 'Variable:0' shape=(2, 1) dtype=float32, numpy=
    array([[1.0155519],
           [1.0256875]], dtype=float32)>
    After 11500 training steps, w is
    <tf.Variable 'Variable:0' shape=(2, 1) dtype=float32, numpy=
    array([[1.0153732],
           [1.0271735]], dtype=float32)>
    After 12000 training steps, w is
    <tf.Variable 'Variable:0' shape=(2, 1) dtype=float32, numpy=
    array([[1.0166917],
           [1.0247964]], dtype=float32)>
    After 12500 training steps, w is
    <tf.Variable 'Variable:0' shape=(2, 1) dtype=float32, numpy=
    array([[1.016513 ],
           [1.0262824]], dtype=float32)>
    After 13000 training steps, w is
    <tf.Variable 'Variable:0' shape=(2, 1) dtype=float32, numpy=
    array([[1.0156165],
           [1.0256418]], dtype=float32)>
    After 13500 training steps, w is
    <tf.Variable 'Variable:0' shape=(2, 1) dtype=float32, numpy=
    array([[1.0223292],
           [1.0315195]], dtype=float32)>
    After 14000 training steps, w is
    <tf.Variable 'Variable:0' shape=(2, 1) dtype=float32, numpy=
    array([[1.0166693],
           [1.0225317]], dtype=float32)>
    After 14500 training steps, w is
    <tf.Variable 'Variable:0' shape=(2, 1) dtype=float32, numpy=
    array([[1.0148629],
           [1.0245526]], dtype=float32)>
    Final w is: <tf.Variable 'Variable:0' shape=(2, 1) dtype=float32, numpy=
    array([[1.0177649],
           [1.0252362]], dtype=float32)>

迭代15000次后,w的最后值array[\[1.0177649],[1.0252362]],与第一个例子相比,模型趋向于将w的值变大

##### 用神经网络实现鸢尾花分类

对鸢尾花数据集使用tensorflow进行预测,可视化acc,loss曲线

导入所需模块

```python
import tensorflow as tf
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
```

①获取数据,分割训练集和测试集

```python
X = load_iris().data
y = load_iris().target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 0)
```

②搭建模型

```python
model = tf.keras.models.Sequential() 
model.add(tf.keras.layers.Dense(10, activation='relu')) 
model.add(tf.keras.layers.Dense(3, activation='softmax')) 
```

③编译模型,迭代500次

```python
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
metrics=['sparse_categorical_accuracy'])
```

④训练模型

```python
history = model.fit(X_train, y_train, batch_size=16, epochs=500)
```

    Epoch 1/500
    7/7 [==============================] - 1s 3ms/step - loss: 1.8005 - sparse_categorical_accuracy: 0.3661
    Epoch 2/500
    7/7 [==============================] - 0s 3ms/step - loss: 1.6582 - sparse_categorical_accuracy: 0.3661
    Epoch 3/500
    7/7 [==============================] - 0s 4ms/step - loss: 1.5358 - sparse_categorical_accuracy: 0.3661
    Epoch 4/500
    7/7 [==============================] - 0s 3ms/step - loss: 1.4318 - sparse_categorical_accuracy: 0.3661
    Epoch 5/500
    7/7 [==============================] - 0s 4ms/step - loss: 1.3399 - sparse_categorical_accuracy: 0.3661
    Epoch 6/500
    7/7 [==============================] - 0s 4ms/step - loss: 1.2528 - sparse_categorical_accuracy: 0.3661
    Epoch 7/500
    7/7 [==============================] - 0s 4ms/step - loss: 1.1853 - sparse_categorical_accuracy: 0.3661
    Epoch 8/500
    7/7 [==============================] - 0s 4ms/step - loss: 1.1038 - sparse_categorical_accuracy: 0.3661
    Epoch 9/500
    7/7 [==============================] - 0s 4ms/step - loss: 1.0455 - sparse_categorical_accuracy: 0.3661
    Epoch 10/500
    7/7 [==============================] - 0s 4ms/step - loss: 0.9959 - sparse_categorical_accuracy: 0.3661
    Epoch 11/500
    7/7 [==============================] - 0s 4ms/step - loss: 0.9488 - sparse_categorical_accuracy: 0.3661
    Epoch 12/500
    7/7 [==============================] - 0s 3ms/step - loss: 0.9152 - sparse_categorical_accuracy: 0.3661
    Epoch 13/500
    7/7 [==============================] - 0s 3ms/step - loss: 0.8913 - sparse_categorical_accuracy: 0.4196
    Epoch 14/500
    7/7 [==============================] - 0s 3ms/step - loss: 0.8702 - sparse_categorical_accuracy: 0.6161
    Epoch 15/500
    7/7 [==============================] - 0s 3ms/step - loss: 0.8582 - sparse_categorical_accuracy: 0.6875
    Epoch 16/500
    7/7 [==============================] - 0s 3ms/step - loss: 0.8431 - sparse_categorical_accuracy: 0.6964
    Epoch 17/500
    7/7 [==============================] - 0s 3ms/step - loss: 0.8303 - sparse_categorical_accuracy: 0.6964
    Epoch 18/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.8184 - sparse_categorical_accuracy: 0.6964
    Epoch 19/500
    7/7 [==============================] - 0s 3ms/step - loss: 0.8061 - sparse_categorical_accuracy: 0.6964
    Epoch 20/500
    7/7 [==============================] - 0s 3ms/step - loss: 0.7946 - sparse_categorical_accuracy: 0.6964
    Epoch 21/500
    7/7 [==============================] - 0s 3ms/step - loss: 0.7832 - sparse_categorical_accuracy: 0.6964
    Epoch 22/500
    7/7 [==============================] - 0s 3ms/step - loss: 0.7740 - sparse_categorical_accuracy: 0.6964
    Epoch 23/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.7617 - sparse_categorical_accuracy: 0.6964
    Epoch 24/500
    7/7 [==============================] - 0s 3ms/step - loss: 0.7516 - sparse_categorical_accuracy: 0.6964
    Epoch 25/500
    7/7 [==============================] - 0s 3ms/step - loss: 0.7426 - sparse_categorical_accuracy: 0.6964
    Epoch 26/500
    7/7 [==============================] - 0s 3ms/step - loss: 0.7321 - sparse_categorical_accuracy: 0.6964
    Epoch 27/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.7223 - sparse_categorical_accuracy: 0.6964
    Epoch 28/500
    7/7 [==============================] - 0s 3ms/step - loss: 0.7134 - sparse_categorical_accuracy: 0.6964
    Epoch 29/500
    7/7 [==============================] - 0s 3ms/step - loss: 0.7052 - sparse_categorical_accuracy: 0.6964
    Epoch 30/500
    7/7 [==============================] - 0s 3ms/step - loss: 0.6961 - sparse_categorical_accuracy: 0.6964
    Epoch 31/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.6870 - sparse_categorical_accuracy: 0.6964
    Epoch 32/500
    7/7 [==============================] - 0s 3ms/step - loss: 0.6788 - sparse_categorical_accuracy: 0.6964
    Epoch 33/500
    7/7 [==============================] - 0s 3ms/step - loss: 0.6717 - sparse_categorical_accuracy: 0.6964
    Epoch 34/500
    7/7 [==============================] - 0s 3ms/step - loss: 0.6630 - sparse_categorical_accuracy: 0.6964
    Epoch 35/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.6569 - sparse_categorical_accuracy: 0.6964
    Epoch 36/500
    7/7 [==============================] - 0s 3ms/step - loss: 0.6478 - sparse_categorical_accuracy: 0.6964
    Epoch 37/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.6411 - sparse_categorical_accuracy: 0.6964
    Epoch 38/500
    7/7 [==============================] - 0s 3ms/step - loss: 0.6337 - sparse_categorical_accuracy: 0.6964
    Epoch 39/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.6271 - sparse_categorical_accuracy: 0.6964
    Epoch 40/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.6205 - sparse_categorical_accuracy: 0.6964
    Epoch 41/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.6137 - sparse_categorical_accuracy: 0.6964
    Epoch 42/500
    7/7 [==============================] - 0s 3ms/step - loss: 0.6076 - sparse_categorical_accuracy: 0.6964
    Epoch 43/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.6011 - sparse_categorical_accuracy: 0.6964
    Epoch 44/500
    7/7 [==============================] - 0s 3ms/step - loss: 0.5951 - sparse_categorical_accuracy: 0.6964
    Epoch 45/500
    7/7 [==============================] - 0s 3ms/step - loss: 0.5893 - sparse_categorical_accuracy: 0.6964
    Epoch 46/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.5835 - sparse_categorical_accuracy: 0.6964
    Epoch 47/500
    7/7 [==============================] - 0s 3ms/step - loss: 0.5780 - sparse_categorical_accuracy: 0.6964
    Epoch 48/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.5728 - sparse_categorical_accuracy: 0.6964
    Epoch 49/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.5672 - sparse_categorical_accuracy: 0.6964
    Epoch 50/500
    7/7 [==============================] - 0s 3ms/step - loss: 0.5624 - sparse_categorical_accuracy: 0.6964
    Epoch 51/500
    7/7 [==============================] - 0s 3ms/step - loss: 0.5567 - sparse_categorical_accuracy: 0.6964
    Epoch 52/500
    7/7 [==============================] - 0s 4ms/step - loss: 0.5528 - sparse_categorical_accuracy: 0.7054
    Epoch 53/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.5470 - sparse_categorical_accuracy: 0.7054
    Epoch 54/500
    7/7 [==============================] - 0s 3ms/step - loss: 0.5423 - sparse_categorical_accuracy: 0.7054
    Epoch 55/500
    7/7 [==============================] - 0s 3ms/step - loss: 0.5383 - sparse_categorical_accuracy: 0.7054
    Epoch 56/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.5331 - sparse_categorical_accuracy: 0.7054
    Epoch 57/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.5300 - sparse_categorical_accuracy: 0.7054
    Epoch 58/500
    7/7 [==============================] - 0s 3ms/step - loss: 0.5244 - sparse_categorical_accuracy: 0.7143
    Epoch 59/500
    7/7 [==============================] - 0s 3ms/step - loss: 0.5204 - sparse_categorical_accuracy: 0.7143
    Epoch 60/500
    7/7 [==============================] - 0s 3ms/step - loss: 0.5160 - sparse_categorical_accuracy: 0.7143
    Epoch 61/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.5117 - sparse_categorical_accuracy: 0.7143
    Epoch 62/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.5076 - sparse_categorical_accuracy: 0.7143
    Epoch 63/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.5040 - sparse_categorical_accuracy: 0.7143
    Epoch 64/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.5003 - sparse_categorical_accuracy: 0.7143
    Epoch 65/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.4963 - sparse_categorical_accuracy: 0.7143
    Epoch 66/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.4924 - sparse_categorical_accuracy: 0.7143
    Epoch 67/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.4896 - sparse_categorical_accuracy: 0.7321
    Epoch 68/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.4850 - sparse_categorical_accuracy: 0.7500
    Epoch 69/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.4814 - sparse_categorical_accuracy: 0.7411
    Epoch 70/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.4782 - sparse_categorical_accuracy: 0.7321
    Epoch 71/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.4750 - sparse_categorical_accuracy: 0.7500
    Epoch 72/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.4710 - sparse_categorical_accuracy: 0.7500
    Epoch 73/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.4676 - sparse_categorical_accuracy: 0.7500
    Epoch 74/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.4647 - sparse_categorical_accuracy: 0.7500
    Epoch 75/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.4612 - sparse_categorical_accuracy: 0.7589
    Epoch 76/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.4579 - sparse_categorical_accuracy: 0.7768
    Epoch 77/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.4559 - sparse_categorical_accuracy: 0.7589
    Epoch 78/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.4519 - sparse_categorical_accuracy: 0.7768
    Epoch 79/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.4484 - sparse_categorical_accuracy: 0.8304
    Epoch 80/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.4455 - sparse_categorical_accuracy: 0.8304
    Epoch 81/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.4429 - sparse_categorical_accuracy: 0.8304
    Epoch 82/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.4396 - sparse_categorical_accuracy: 0.8304
    Epoch 83/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.4368 - sparse_categorical_accuracy: 0.8304
    Epoch 84/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.4345 - sparse_categorical_accuracy: 0.8304
    Epoch 85/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.4309 - sparse_categorical_accuracy: 0.8482
    Epoch 86/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.4279 - sparse_categorical_accuracy: 0.8482
    Epoch 87/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.4251 - sparse_categorical_accuracy: 0.8393
    Epoch 88/500
    7/7 [==============================] - 0s 3ms/step - loss: 0.4233 - sparse_categorical_accuracy: 0.8393
    Epoch 89/500
    7/7 [==============================] - 0s 3ms/step - loss: 0.4201 - sparse_categorical_accuracy: 0.8393
    Epoch 90/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.4167 - sparse_categorical_accuracy: 0.8393
    Epoch 91/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.4147 - sparse_categorical_accuracy: 0.8393
    Epoch 92/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.4115 - sparse_categorical_accuracy: 0.8393
    Epoch 93/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.4090 - sparse_categorical_accuracy: 0.8482
    Epoch 94/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.4061 - sparse_categorical_accuracy: 0.8661
    Epoch 95/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.4036 - sparse_categorical_accuracy: 0.8750
    Epoch 96/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.4014 - sparse_categorical_accuracy: 0.8661
    Epoch 97/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.3987 - sparse_categorical_accuracy: 0.8839
    Epoch 98/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.3958 - sparse_categorical_accuracy: 0.8839
    Epoch 99/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.3939 - sparse_categorical_accuracy: 0.8839
    Epoch 100/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.3912 - sparse_categorical_accuracy: 0.8839
    Epoch 101/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.3882 - sparse_categorical_accuracy: 0.9018
    Epoch 102/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.3859 - sparse_categorical_accuracy: 0.9196
    Epoch 103/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.3835 - sparse_categorical_accuracy: 0.9196
    Epoch 104/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.3815 - sparse_categorical_accuracy: 0.9196
    Epoch 105/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.3785 - sparse_categorical_accuracy: 0.9196
    Epoch 106/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.3769 - sparse_categorical_accuracy: 0.9107
    Epoch 107/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.3751 - sparse_categorical_accuracy: 0.9286
    Epoch 108/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.3715 - sparse_categorical_accuracy: 0.9286
    Epoch 109/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.3691 - sparse_categorical_accuracy: 0.9286
    Epoch 110/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.3668 - sparse_categorical_accuracy: 0.9196
    Epoch 111/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.3651 - sparse_categorical_accuracy: 0.9196
    Epoch 112/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.3622 - sparse_categorical_accuracy: 0.9196
    Epoch 113/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.3603 - sparse_categorical_accuracy: 0.9196
    Epoch 114/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.3581 - sparse_categorical_accuracy: 0.9286
    Epoch 115/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.3561 - sparse_categorical_accuracy: 0.9464
    Epoch 116/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.3533 - sparse_categorical_accuracy: 0.9554
    Epoch 117/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.3516 - sparse_categorical_accuracy: 0.9375
    Epoch 118/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.3486 - sparse_categorical_accuracy: 0.9286
    Epoch 119/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.3465 - sparse_categorical_accuracy: 0.9286
    Epoch 120/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.3444 - sparse_categorical_accuracy: 0.9286
    Epoch 121/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.3420 - sparse_categorical_accuracy: 0.9464
    Epoch 122/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.3407 - sparse_categorical_accuracy: 0.9464
    Epoch 123/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.3377 - sparse_categorical_accuracy: 0.9554
    Epoch 124/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.3356 - sparse_categorical_accuracy: 0.9554
    Epoch 125/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.3338 - sparse_categorical_accuracy: 0.9554
    Epoch 126/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.3324 - sparse_categorical_accuracy: 0.9464
    Epoch 127/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.3291 - sparse_categorical_accuracy: 0.9554
    Epoch 128/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.3272 - sparse_categorical_accuracy: 0.9554
    Epoch 129/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.3262 - sparse_categorical_accuracy: 0.9375
    Epoch 130/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.3225 - sparse_categorical_accuracy: 0.9464
    Epoch 131/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.3206 - sparse_categorical_accuracy: 0.9554
    Epoch 132/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.3187 - sparse_categorical_accuracy: 0.9554
    Epoch 133/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.3168 - sparse_categorical_accuracy: 0.9554
    Epoch 134/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.3147 - sparse_categorical_accuracy: 0.9554
    Epoch 135/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.3126 - sparse_categorical_accuracy: 0.9554
    Epoch 136/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.3104 - sparse_categorical_accuracy: 0.9554
    Epoch 137/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.3084 - sparse_categorical_accuracy: 0.9554
    Epoch 138/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.3065 - sparse_categorical_accuracy: 0.9554
    Epoch 139/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.3050 - sparse_categorical_accuracy: 0.9554
    Epoch 140/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.3026 - sparse_categorical_accuracy: 0.9554
    Epoch 141/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.3006 - sparse_categorical_accuracy: 0.9554
    Epoch 142/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.2984 - sparse_categorical_accuracy: 0.9554
    Epoch 143/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.2975 - sparse_categorical_accuracy: 0.9643
    Epoch 144/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.2947 - sparse_categorical_accuracy: 0.9643
    Epoch 145/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.2926 - sparse_categorical_accuracy: 0.9643
    Epoch 146/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.2908 - sparse_categorical_accuracy: 0.9643
    Epoch 147/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.2891 - sparse_categorical_accuracy: 0.9554
    Epoch 148/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.2869 - sparse_categorical_accuracy: 0.9643
    Epoch 149/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.2852 - sparse_categorical_accuracy: 0.9643
    Epoch 150/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.2831 - sparse_categorical_accuracy: 0.9643
    Epoch 151/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.2819 - sparse_categorical_accuracy: 0.9643
    Epoch 152/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.2793 - sparse_categorical_accuracy: 0.9643
    Epoch 153/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.2778 - sparse_categorical_accuracy: 0.9643
    Epoch 154/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.2758 - sparse_categorical_accuracy: 0.9643
    Epoch 155/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.2737 - sparse_categorical_accuracy: 0.9643
    Epoch 156/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.2724 - sparse_categorical_accuracy: 0.9643
    Epoch 157/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.2712 - sparse_categorical_accuracy: 0.9643
    Epoch 158/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.2685 - sparse_categorical_accuracy: 0.9643
    Epoch 159/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.2666 - sparse_categorical_accuracy: 0.9643
    Epoch 160/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.2654 - sparse_categorical_accuracy: 0.9643
    Epoch 161/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.2630 - sparse_categorical_accuracy: 0.9643
    Epoch 162/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.2616 - sparse_categorical_accuracy: 0.9643
    Epoch 163/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.2598 - sparse_categorical_accuracy: 0.9643
    Epoch 164/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.2590 - sparse_categorical_accuracy: 0.9643
    Epoch 165/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.2562 - sparse_categorical_accuracy: 0.9643
    Epoch 166/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.2545 - sparse_categorical_accuracy: 0.9643
    Epoch 167/500
    7/7 [==============================] - 0s 3ms/step - loss: 0.2532 - sparse_categorical_accuracy: 0.9643
    Epoch 168/500
    7/7 [==============================] - 0s 3ms/step - loss: 0.2520 - sparse_categorical_accuracy: 0.9643
    Epoch 169/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.2497 - sparse_categorical_accuracy: 0.9821
    Epoch 170/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.2478 - sparse_categorical_accuracy: 0.9732
    Epoch 171/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.2470 - sparse_categorical_accuracy: 0.9643
    Epoch 172/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.2445 - sparse_categorical_accuracy: 0.9643
    Epoch 173/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.2431 - sparse_categorical_accuracy: 0.9643
    Epoch 174/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.2415 - sparse_categorical_accuracy: 0.9643
    Epoch 175/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.2399 - sparse_categorical_accuracy: 0.9643
    Epoch 176/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.2383 - sparse_categorical_accuracy: 0.9643
    Epoch 177/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.2375 - sparse_categorical_accuracy: 0.9643
    Epoch 178/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.2358 - sparse_categorical_accuracy: 0.9643
    Epoch 179/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.2338 - sparse_categorical_accuracy: 0.9643
    Epoch 180/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.2322 - sparse_categorical_accuracy: 0.9732
    Epoch 181/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.2333 - sparse_categorical_accuracy: 0.9732
    Epoch 182/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.2290 - sparse_categorical_accuracy: 0.9732
    Epoch 183/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.2276 - sparse_categorical_accuracy: 0.9732
    Epoch 184/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.2267 - sparse_categorical_accuracy: 0.9643
    Epoch 185/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.2250 - sparse_categorical_accuracy: 0.9643
    Epoch 186/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.2242 - sparse_categorical_accuracy: 0.9643
    Epoch 187/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.2220 - sparse_categorical_accuracy: 0.9732
    Epoch 188/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.2206 - sparse_categorical_accuracy: 0.9732
    Epoch 189/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.2205 - sparse_categorical_accuracy: 0.9821
    Epoch 190/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.2180 - sparse_categorical_accuracy: 0.9732
    Epoch 191/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.2163 - sparse_categorical_accuracy: 0.9732
    Epoch 192/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.2151 - sparse_categorical_accuracy: 0.9732
    Epoch 193/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.2139 - sparse_categorical_accuracy: 0.9643
    Epoch 194/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.2129 - sparse_categorical_accuracy: 0.9732
    Epoch 195/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.2116 - sparse_categorical_accuracy: 0.9732
    Epoch 196/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.2108 - sparse_categorical_accuracy: 0.9732
    Epoch 197/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.2086 - sparse_categorical_accuracy: 0.9643
    Epoch 198/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.2072 - sparse_categorical_accuracy: 0.9732
    Epoch 199/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.2067 - sparse_categorical_accuracy: 0.9821
    Epoch 200/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.2052 - sparse_categorical_accuracy: 0.9821
    Epoch 201/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.2037 - sparse_categorical_accuracy: 0.9821
    Epoch 202/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.2021 - sparse_categorical_accuracy: 0.9732
    Epoch 203/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.2010 - sparse_categorical_accuracy: 0.9732
    Epoch 204/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1997 - sparse_categorical_accuracy: 0.9732
    Epoch 205/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1993 - sparse_categorical_accuracy: 0.9821
    Epoch 206/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1973 - sparse_categorical_accuracy: 0.9821
    Epoch 207/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1962 - sparse_categorical_accuracy: 0.9732
    Epoch 208/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1950 - sparse_categorical_accuracy: 0.9732
    Epoch 209/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1938 - sparse_categorical_accuracy: 0.9732
    Epoch 210/500
    7/7 [==============================] - 0s 3ms/step - loss: 0.1931 - sparse_categorical_accuracy: 0.9732
    Epoch 211/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1920 - sparse_categorical_accuracy: 0.9732
    Epoch 212/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1904 - sparse_categorical_accuracy: 0.9821
    Epoch 213/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1899 - sparse_categorical_accuracy: 0.9732
    Epoch 214/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1881 - sparse_categorical_accuracy: 0.9732
    Epoch 215/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1888 - sparse_categorical_accuracy: 0.9732
    Epoch 216/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1861 - sparse_categorical_accuracy: 0.9821
    Epoch 217/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1851 - sparse_categorical_accuracy: 0.9732
    Epoch 218/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1843 - sparse_categorical_accuracy: 0.9821
    Epoch 219/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1830 - sparse_categorical_accuracy: 0.9821
    Epoch 220/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1821 - sparse_categorical_accuracy: 0.9732
    Epoch 221/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1806 - sparse_categorical_accuracy: 0.9821
    Epoch 222/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1796 - sparse_categorical_accuracy: 0.9821
    Epoch 223/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1785 - sparse_categorical_accuracy: 0.9821
    Epoch 224/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1778 - sparse_categorical_accuracy: 0.9732
    Epoch 225/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1767 - sparse_categorical_accuracy: 0.9732
    Epoch 226/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1762 - sparse_categorical_accuracy: 0.9821
    Epoch 227/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1743 - sparse_categorical_accuracy: 0.9821
    Epoch 228/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1741 - sparse_categorical_accuracy: 0.9732
    Epoch 229/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1729 - sparse_categorical_accuracy: 0.9732
    Epoch 230/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1716 - sparse_categorical_accuracy: 0.9821
    Epoch 231/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1707 - sparse_categorical_accuracy: 0.9821
    Epoch 232/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1699 - sparse_categorical_accuracy: 0.9821
    Epoch 233/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1688 - sparse_categorical_accuracy: 0.9821
    Epoch 234/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1680 - sparse_categorical_accuracy: 0.9821
    Epoch 235/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1669 - sparse_categorical_accuracy: 0.9821
    Epoch 236/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1661 - sparse_categorical_accuracy: 0.9732
    Epoch 237/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1652 - sparse_categorical_accuracy: 0.9732
    Epoch 238/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1641 - sparse_categorical_accuracy: 0.9821
    Epoch 239/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1635 - sparse_categorical_accuracy: 0.9821
    Epoch 240/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1627 - sparse_categorical_accuracy: 0.9732
    Epoch 241/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1617 - sparse_categorical_accuracy: 0.9732
    Epoch 242/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1612 - sparse_categorical_accuracy: 0.9821
    Epoch 243/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1601 - sparse_categorical_accuracy: 0.9821
    Epoch 244/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1590 - sparse_categorical_accuracy: 0.9732
    Epoch 245/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1585 - sparse_categorical_accuracy: 0.9732
    Epoch 246/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1573 - sparse_categorical_accuracy: 0.9821
    Epoch 247/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1569 - sparse_categorical_accuracy: 0.9821
    Epoch 248/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1558 - sparse_categorical_accuracy: 0.9732
    Epoch 249/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1552 - sparse_categorical_accuracy: 0.9821
    Epoch 250/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1543 - sparse_categorical_accuracy: 0.9821
    Epoch 251/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1534 - sparse_categorical_accuracy: 0.9821
    Epoch 252/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1527 - sparse_categorical_accuracy: 0.9821
    Epoch 253/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1519 - sparse_categorical_accuracy: 0.9732
    Epoch 254/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1511 - sparse_categorical_accuracy: 0.9821
    Epoch 255/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1502 - sparse_categorical_accuracy: 0.9821
    Epoch 256/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1496 - sparse_categorical_accuracy: 0.9821
    Epoch 257/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1509 - sparse_categorical_accuracy: 0.9732
    Epoch 258/500
    7/7 [==============================] - 0s 3ms/step - loss: 0.1479 - sparse_categorical_accuracy: 0.9732
    Epoch 259/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1471 - sparse_categorical_accuracy: 0.9821
    Epoch 260/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1463 - sparse_categorical_accuracy: 0.9821
    Epoch 261/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1457 - sparse_categorical_accuracy: 0.9821
    Epoch 262/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1450 - sparse_categorical_accuracy: 0.9821
    Epoch 263/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1443 - sparse_categorical_accuracy: 0.9821
    Epoch 264/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1435 - sparse_categorical_accuracy: 0.9821
    Epoch 265/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1429 - sparse_categorical_accuracy: 0.9821
    Epoch 266/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1422 - sparse_categorical_accuracy: 0.9821
    Epoch 267/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1420 - sparse_categorical_accuracy: 0.9732
    Epoch 268/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1409 - sparse_categorical_accuracy: 0.9732
    Epoch 269/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1401 - sparse_categorical_accuracy: 0.9821
    Epoch 270/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1395 - sparse_categorical_accuracy: 0.9821
    Epoch 271/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1390 - sparse_categorical_accuracy: 0.9821
    Epoch 272/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1384 - sparse_categorical_accuracy: 0.9821
    Epoch 273/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1382 - sparse_categorical_accuracy: 0.9821
    Epoch 274/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1368 - sparse_categorical_accuracy: 0.9821
    Epoch 275/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1367 - sparse_categorical_accuracy: 0.9821
    Epoch 276/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1359 - sparse_categorical_accuracy: 0.9821
    Epoch 277/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1349 - sparse_categorical_accuracy: 0.9821
    Epoch 278/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1343 - sparse_categorical_accuracy: 0.9821
    Epoch 279/500
    7/7 [==============================] - 0s 3ms/step - loss: 0.1337 - sparse_categorical_accuracy: 0.9821
    Epoch 280/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1330 - sparse_categorical_accuracy: 0.9821
    Epoch 281/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1324 - sparse_categorical_accuracy: 0.9821
    Epoch 282/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1319 - sparse_categorical_accuracy: 0.9821
    Epoch 283/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1313 - sparse_categorical_accuracy: 0.9821
    Epoch 284/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1307 - sparse_categorical_accuracy: 0.9821
    Epoch 285/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1299 - sparse_categorical_accuracy: 0.9821
    Epoch 286/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1298 - sparse_categorical_accuracy: 0.9821
    Epoch 287/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1290 - sparse_categorical_accuracy: 0.9821
    Epoch 288/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1283 - sparse_categorical_accuracy: 0.9821
    Epoch 289/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1279 - sparse_categorical_accuracy: 0.9821
    Epoch 290/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1279 - sparse_categorical_accuracy: 0.9732
    Epoch 291/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1266 - sparse_categorical_accuracy: 0.9821
    Epoch 292/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1262 - sparse_categorical_accuracy: 0.9821
    Epoch 293/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1256 - sparse_categorical_accuracy: 0.9821
    Epoch 294/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1250 - sparse_categorical_accuracy: 0.9821
    Epoch 295/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1244 - sparse_categorical_accuracy: 0.9821
    Epoch 296/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1238 - sparse_categorical_accuracy: 0.9821
    Epoch 297/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1237 - sparse_categorical_accuracy: 0.9732
    Epoch 298/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1228 - sparse_categorical_accuracy: 0.9821
    Epoch 299/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1224 - sparse_categorical_accuracy: 0.9821
    Epoch 300/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1219 - sparse_categorical_accuracy: 0.9821
    Epoch 301/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1221 - sparse_categorical_accuracy: 0.9821
    Epoch 302/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1208 - sparse_categorical_accuracy: 0.9821
    Epoch 303/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1206 - sparse_categorical_accuracy: 0.9821
    Epoch 304/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1205 - sparse_categorical_accuracy: 0.9821
    Epoch 305/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1195 - sparse_categorical_accuracy: 0.9821
    Epoch 306/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1189 - sparse_categorical_accuracy: 0.9821
    Epoch 307/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1184 - sparse_categorical_accuracy: 0.9821
    Epoch 308/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1177 - sparse_categorical_accuracy: 0.9821
    Epoch 309/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1181 - sparse_categorical_accuracy: 0.9821
    Epoch 310/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1174 - sparse_categorical_accuracy: 0.9821
    Epoch 311/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1168 - sparse_categorical_accuracy: 0.9821
    Epoch 312/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1164 - sparse_categorical_accuracy: 0.9821
    Epoch 313/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1159 - sparse_categorical_accuracy: 0.9821
    Epoch 314/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1150 - sparse_categorical_accuracy: 0.9821
    Epoch 315/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1147 - sparse_categorical_accuracy: 0.9821
    Epoch 316/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1140 - sparse_categorical_accuracy: 0.9821
    Epoch 317/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1141 - sparse_categorical_accuracy: 0.9821
    Epoch 318/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1137 - sparse_categorical_accuracy: 0.9821
    Epoch 319/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1129 - sparse_categorical_accuracy: 0.9821
    Epoch 320/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1124 - sparse_categorical_accuracy: 0.9821
    Epoch 321/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.1121 - sparse_categorical_accuracy: 0.9821
    Epoch 322/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1121 - sparse_categorical_accuracy: 0.9732
    Epoch 323/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1112 - sparse_categorical_accuracy: 0.9732
    Epoch 324/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1108 - sparse_categorical_accuracy: 0.9821
    Epoch 325/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1102 - sparse_categorical_accuracy: 0.9821
    Epoch 326/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1101 - sparse_categorical_accuracy: 0.9821
    Epoch 327/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1098 - sparse_categorical_accuracy: 0.9821
    Epoch 328/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1096 - sparse_categorical_accuracy: 0.9821
    Epoch 329/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1091 - sparse_categorical_accuracy: 0.9821
    Epoch 330/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1082 - sparse_categorical_accuracy: 0.9821
    Epoch 331/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1077 - sparse_categorical_accuracy: 0.9821
    Epoch 332/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1074 - sparse_categorical_accuracy: 0.9821
    Epoch 333/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1072 - sparse_categorical_accuracy: 0.9821
    Epoch 334/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1070 - sparse_categorical_accuracy: 0.9821
    Epoch 335/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1072 - sparse_categorical_accuracy: 0.9821
    Epoch 336/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1057 - sparse_categorical_accuracy: 0.9821
    Epoch 337/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1057 - sparse_categorical_accuracy: 0.9821
    Epoch 338/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1050 - sparse_categorical_accuracy: 0.9821
    Epoch 339/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1048 - sparse_categorical_accuracy: 0.9821
    Epoch 340/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1044 - sparse_categorical_accuracy: 0.9821
    Epoch 341/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1038 - sparse_categorical_accuracy: 0.9821
    Epoch 342/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.1036 - sparse_categorical_accuracy: 0.9821
    Epoch 343/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.1033 - sparse_categorical_accuracy: 0.9821
    Epoch 344/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1033 - sparse_categorical_accuracy: 0.9821
    Epoch 345/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1024 - sparse_categorical_accuracy: 0.9821
    Epoch 346/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1027 - sparse_categorical_accuracy: 0.9821
    Epoch 347/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.1017 - sparse_categorical_accuracy: 0.9821
    Epoch 348/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.1014 - sparse_categorical_accuracy: 0.9821
    Epoch 349/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1016 - sparse_categorical_accuracy: 0.9821
    Epoch 350/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.1007 - sparse_categorical_accuracy: 0.9821
    Epoch 351/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.1010 - sparse_categorical_accuracy: 0.9821
    Epoch 352/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.1001 - sparse_categorical_accuracy: 0.9821
    Epoch 353/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.0996 - sparse_categorical_accuracy: 0.9821
    Epoch 354/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0992 - sparse_categorical_accuracy: 0.9821
    Epoch 355/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0991 - sparse_categorical_accuracy: 0.9821
    Epoch 356/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.0997 - sparse_categorical_accuracy: 0.9821
    Epoch 357/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.0987 - sparse_categorical_accuracy: 0.9821
    Epoch 358/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0989 - sparse_categorical_accuracy: 0.9821
    Epoch 359/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0980 - sparse_categorical_accuracy: 0.9732
    Epoch 360/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0978 - sparse_categorical_accuracy: 0.9821
    Epoch 361/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.0972 - sparse_categorical_accuracy: 0.9821
    Epoch 362/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.0969 - sparse_categorical_accuracy: 0.9821
    Epoch 363/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0964 - sparse_categorical_accuracy: 0.9821
    Epoch 364/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0960 - sparse_categorical_accuracy: 0.9821
    Epoch 365/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0960 - sparse_categorical_accuracy: 0.9821
    Epoch 366/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0960 - sparse_categorical_accuracy: 0.9821
    Epoch 367/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0951 - sparse_categorical_accuracy: 0.9821
    Epoch 368/500
    7/7 [==============================] - 0s 4ms/step - loss: 0.0954 - sparse_categorical_accuracy: 0.9821
    Epoch 369/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.0949 - sparse_categorical_accuracy: 0.9821
    Epoch 370/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0945 - sparse_categorical_accuracy: 0.9821
    Epoch 371/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0941 - sparse_categorical_accuracy: 0.9821
    Epoch 372/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0942 - sparse_categorical_accuracy: 0.9821
    Epoch 373/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0935 - sparse_categorical_accuracy: 0.9821
    Epoch 374/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0931 - sparse_categorical_accuracy: 0.9821
    Epoch 375/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0928 - sparse_categorical_accuracy: 0.9821
    Epoch 376/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.0926 - sparse_categorical_accuracy: 0.9821
    Epoch 377/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.0929 - sparse_categorical_accuracy: 0.9821
    Epoch 378/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.0935 - sparse_categorical_accuracy: 0.9821
    Epoch 379/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0917 - sparse_categorical_accuracy: 0.9821
    Epoch 380/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.0921 - sparse_categorical_accuracy: 0.9821
    Epoch 381/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0921 - sparse_categorical_accuracy: 0.9821
    Epoch 382/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0913 - sparse_categorical_accuracy: 0.9821
    Epoch 383/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.0908 - sparse_categorical_accuracy: 0.9821
    Epoch 384/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0906 - sparse_categorical_accuracy: 0.9821
    Epoch 385/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0903 - sparse_categorical_accuracy: 0.9821
    Epoch 386/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0901 - sparse_categorical_accuracy: 0.9821
    Epoch 387/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.0897 - sparse_categorical_accuracy: 0.9821
    Epoch 388/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.0894 - sparse_categorical_accuracy: 0.9821
    Epoch 389/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.0893 - sparse_categorical_accuracy: 0.9821
    Epoch 390/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.0889 - sparse_categorical_accuracy: 0.9821
    Epoch 391/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0885 - sparse_categorical_accuracy: 0.9821
    Epoch 392/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0884 - sparse_categorical_accuracy: 0.9821
    Epoch 393/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0882 - sparse_categorical_accuracy: 0.9821
    Epoch 394/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0884 - sparse_categorical_accuracy: 0.9821
    Epoch 395/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.0877 - sparse_categorical_accuracy: 0.9821
    Epoch 396/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0887 - sparse_categorical_accuracy: 0.9821
    Epoch 397/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.0871 - sparse_categorical_accuracy: 0.9821
    Epoch 398/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0874 - sparse_categorical_accuracy: 0.9821
    Epoch 399/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0870 - sparse_categorical_accuracy: 0.9821
    Epoch 400/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0866 - sparse_categorical_accuracy: 0.9821
    Epoch 401/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.0865 - sparse_categorical_accuracy: 0.9821
    Epoch 402/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0866 - sparse_categorical_accuracy: 0.9821
    Epoch 403/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0860 - sparse_categorical_accuracy: 0.9821
    Epoch 404/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.0854 - sparse_categorical_accuracy: 0.9821
    Epoch 405/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0852 - sparse_categorical_accuracy: 0.9821
    Epoch 406/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0853 - sparse_categorical_accuracy: 0.9821
    Epoch 407/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.0849 - sparse_categorical_accuracy: 0.9821
    Epoch 408/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.0854 - sparse_categorical_accuracy: 0.9821
    Epoch 409/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.0842 - sparse_categorical_accuracy: 0.9821
    Epoch 410/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.0842 - sparse_categorical_accuracy: 0.9821
    Epoch 411/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.0839 - sparse_categorical_accuracy: 0.9821
    Epoch 412/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0837 - sparse_categorical_accuracy: 0.9821
    Epoch 413/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0834 - sparse_categorical_accuracy: 0.9821
    Epoch 414/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0832 - sparse_categorical_accuracy: 0.9821
    Epoch 415/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0830 - sparse_categorical_accuracy: 0.9821
    Epoch 416/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.0830 - sparse_categorical_accuracy: 0.9821
    Epoch 417/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0825 - sparse_categorical_accuracy: 0.9821
    Epoch 418/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0825 - sparse_categorical_accuracy: 0.9821
    Epoch 419/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0841 - sparse_categorical_accuracy: 0.9821
    Epoch 420/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0822 - sparse_categorical_accuracy: 0.9821
    Epoch 421/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0854 - sparse_categorical_accuracy: 0.9821
    Epoch 422/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0819 - sparse_categorical_accuracy: 0.9732
    Epoch 423/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0812 - sparse_categorical_accuracy: 0.9821
    Epoch 424/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0812 - sparse_categorical_accuracy: 0.9821
    Epoch 425/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0810 - sparse_categorical_accuracy: 0.9821
    Epoch 426/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0811 - sparse_categorical_accuracy: 0.9821
    Epoch 427/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0805 - sparse_categorical_accuracy: 0.9821
    Epoch 428/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0804 - sparse_categorical_accuracy: 0.9821
    Epoch 429/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0806 - sparse_categorical_accuracy: 0.9821
    Epoch 430/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0799 - sparse_categorical_accuracy: 0.9821
    Epoch 431/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0798 - sparse_categorical_accuracy: 0.9821
    Epoch 432/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0800 - sparse_categorical_accuracy: 0.9821
    Epoch 433/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.0795 - sparse_categorical_accuracy: 0.9821
    Epoch 434/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0791 - sparse_categorical_accuracy: 0.9821
    Epoch 435/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0790 - sparse_categorical_accuracy: 0.9821
    Epoch 436/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0789 - sparse_categorical_accuracy: 0.9821
    Epoch 437/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0788 - sparse_categorical_accuracy: 0.9821
    Epoch 438/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0784 - sparse_categorical_accuracy: 0.9821
    Epoch 439/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0783 - sparse_categorical_accuracy: 0.9821
    Epoch 440/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.0780 - sparse_categorical_accuracy: 0.9821
    Epoch 441/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0779 - sparse_categorical_accuracy: 0.9821
    Epoch 442/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0780 - sparse_categorical_accuracy: 0.9821
    Epoch 443/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0774 - sparse_categorical_accuracy: 0.9821
    Epoch 444/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0774 - sparse_categorical_accuracy: 0.9821
    Epoch 445/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0773 - sparse_categorical_accuracy: 0.9821
    Epoch 446/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.0770 - sparse_categorical_accuracy: 0.9821
    Epoch 447/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.0768 - sparse_categorical_accuracy: 0.9821
    Epoch 448/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0767 - sparse_categorical_accuracy: 0.9821
    Epoch 449/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.0764 - sparse_categorical_accuracy: 0.9821
    Epoch 450/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.0764 - sparse_categorical_accuracy: 0.9821
    Epoch 451/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.0768 - sparse_categorical_accuracy: 0.9821
    Epoch 452/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0762 - sparse_categorical_accuracy: 0.9821
    Epoch 453/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0757 - sparse_categorical_accuracy: 0.9821
    Epoch 454/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.0755 - sparse_categorical_accuracy: 0.9821
    Epoch 455/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0757 - sparse_categorical_accuracy: 0.9821
    Epoch 456/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0761 - sparse_categorical_accuracy: 0.9821
    Epoch 457/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0757 - sparse_categorical_accuracy: 0.9821
    Epoch 458/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0750 - sparse_categorical_accuracy: 0.9821
    Epoch 459/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.0752 - sparse_categorical_accuracy: 0.9821
    Epoch 460/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.0752 - sparse_categorical_accuracy: 0.9821
    Epoch 461/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0751 - sparse_categorical_accuracy: 0.9821
    Epoch 462/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0742 - sparse_categorical_accuracy: 0.9821
    Epoch 463/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0742 - sparse_categorical_accuracy: 0.9821
    Epoch 464/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.0743 - sparse_categorical_accuracy: 0.9821
    Epoch 465/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0740 - sparse_categorical_accuracy: 0.9821
    Epoch 466/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.0737 - sparse_categorical_accuracy: 0.9821
    Epoch 467/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0735 - sparse_categorical_accuracy: 0.9821
    Epoch 468/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0732 - sparse_categorical_accuracy: 0.9821
    Epoch 469/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.0732 - sparse_categorical_accuracy: 0.9821
    Epoch 470/500
    7/7 [==============================] - 0s 3ms/step - loss: 0.0731 - sparse_categorical_accuracy: 0.9821
    Epoch 471/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.0730 - sparse_categorical_accuracy: 0.9821
    Epoch 472/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0727 - sparse_categorical_accuracy: 0.9821
    Epoch 473/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.0726 - sparse_categorical_accuracy: 0.9821
    Epoch 474/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.0729 - sparse_categorical_accuracy: 0.9821
    Epoch 475/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0731 - sparse_categorical_accuracy: 0.9821
    Epoch 476/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0727 - sparse_categorical_accuracy: 0.9821
    Epoch 477/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0721 - sparse_categorical_accuracy: 0.9821
    Epoch 478/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.0721 - sparse_categorical_accuracy: 0.9821
    Epoch 479/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0720 - sparse_categorical_accuracy: 0.9821
    Epoch 480/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0716 - sparse_categorical_accuracy: 0.9821
    Epoch 481/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0714 - sparse_categorical_accuracy: 0.9821
    Epoch 482/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.0714 - sparse_categorical_accuracy: 0.9821
    Epoch 483/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0712 - sparse_categorical_accuracy: 0.9821
    Epoch 484/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.0712 - sparse_categorical_accuracy: 0.9821
    Epoch 485/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.0712 - sparse_categorical_accuracy: 0.9821
    Epoch 486/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0708 - sparse_categorical_accuracy: 0.9821
    Epoch 487/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0705 - sparse_categorical_accuracy: 0.9821
    Epoch 488/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0704 - sparse_categorical_accuracy: 0.9821
    Epoch 489/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0704 - sparse_categorical_accuracy: 0.9821
    Epoch 490/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0703 - sparse_categorical_accuracy: 0.9821
    Epoch 491/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0701 - sparse_categorical_accuracy: 0.9821
    Epoch 492/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0703 - sparse_categorical_accuracy: 0.9821
    Epoch 493/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0699 - sparse_categorical_accuracy: 0.9821
    Epoch 494/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.0697 - sparse_categorical_accuracy: 0.9821
    Epoch 495/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0700 - sparse_categorical_accuracy: 0.9821
    Epoch 496/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.0704 - sparse_categorical_accuracy: 0.9821
    Epoch 497/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0696 - sparse_categorical_accuracy: 0.9821
    Epoch 498/500
    7/7 [==============================] - 0s 2ms/step - loss: 0.0692 - sparse_categorical_accuracy: 0.9821
    Epoch 499/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0689 - sparse_categorical_accuracy: 0.9821
    Epoch 500/500
    7/7 [==============================] - 0s 1ms/step - loss: 0.0690 - sparse_categorical_accuracy: 0.9821

⑤结果评估

1,训练过程可视化

```python
import matplotlib.pyplot as plt
pd.DataFrame(history.history).plot()
plt.show()
```

![png](https://img2022.cnblogs.com/blog/2629720/202210/2629720-20221010125541894-552692443.png)

2,整体准确率

```python
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(' %.2f%%' %(test_acc*100))
```

    2/2 - 0s - loss: 0.1298 - sparse_categorical_accuracy: 0.9737 - 271ms/epoch - 135ms/step
     97.37%

3,预测样本

```python
predictions = model.predict(X_test, verbose=2)
print(predictions[0])
print(' %d %.2f%%' %(tf.argmax(predictions[0]), max(predictions[0])*100))
```

    2/2 - 0s - 104ms/epoch - 52ms/step
    [8.3498879e-09 1.8262751e-03 9.9817371e-01]
     2 99.82%

```python
print(predictions)
```

    [[8.3498879e-09 1.8262751e-03 9.9817371e-01]
     [3.2518408e-03 9.8306483e-01 1.3683353e-02]
     [9.9939382e-01 6.0616154e-04 4.0472346e-14]
     [5.1105918e-08 1.5814977e-02 9.8418498e-01]
     [9.9613428e-01 3.8657226e-03 1.1263443e-11]
     [5.0845561e-10 4.7897341e-04 9.9952102e-01]
     [9.9715734e-01 2.8426135e-03 4.8997234e-12]
     [7.0876861e-04 9.6024650e-01 3.9044708e-02]
     [4.2222903e-04 9.3484229e-01 6.4735457e-02]
     [4.0413835e-03 9.8746198e-01 8.4966142e-03]
     [1.2660873e-06 4.3164853e-02 9.5683390e-01]
     [1.3402370e-03 9.7123307e-01 2.7426608e-02]
     [8.4431336e-04 9.0923429e-01 8.9921355e-02]
     [4.3927005e-04 9.0233898e-01 9.7221702e-02]
     [4.7115324e-04 8.2901543e-01 1.7051351e-01]
     [9.9828017e-01 1.7199020e-03 1.7988909e-12]
     [5.4972956e-04 8.3909136e-01 1.6035888e-01]
     [8.9530699e-04 8.0865324e-01 1.9045144e-01]
     [9.9046409e-01 9.5359245e-03 1.5190581e-10]
     [9.9846578e-01 1.5342035e-03 4.7513046e-13]
     [5.6102238e-07 1.9395353e-02 9.8060405e-01]
     [5.0135364e-04 7.1416974e-01 2.8532892e-01]
     [9.9001676e-01 9.9832211e-03 2.1078256e-10]
     [9.9051952e-01 9.4804186e-03 3.4175113e-10]
     [2.6071793e-05 2.8323361e-01 7.1674031e-01]
     [9.9886900e-01 1.1309125e-03 1.0174007e-12]
     [9.9197525e-01 8.0247913e-03 6.7378651e-11]
     [2.3211068e-03 9.8043609e-01 1.7242851e-02]
     [1.7626515e-02 9.7543079e-01 6.9427858e-03]
     [9.9145693e-01 8.5430704e-03 8.4697659e-11]
     [2.2322536e-06 7.6802179e-02 9.2319554e-01]
     [3.9311437e-04 5.9891444e-01 4.0069246e-01]
     [9.9688905e-01 3.1109049e-03 4.3471437e-12]
     [2.7264292e-05 2.6483223e-01 7.3514050e-01]
     [8.7741050e-09 2.8957480e-03 9.9710423e-01]
     [2.0027638e-03 8.8426864e-01 1.1372862e-01]
     [9.9712950e-01 2.8704719e-03 1.3951041e-12]
     [7.7986178e-06 1.2152320e-01 8.7846905e-01]]

4,输出混淆矩阵

```python
from sklearn.metrics import confusion_matrix, classification_report
y_true = y_test
y_pred = np.argmax(predictions, axis=1)
print(confusion_matrix(y_true, y_pred))
```

    [[13  0  0]
     [ 0 15  1]
     [ 0  0  9]]

分类报告

```python
print(classification_report(y_true, y_pred))
```

                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00        13
               1       1.00      0.94      0.97        16
               2       0.90      1.00      0.95         9
    
        accuracy                           0.97        38
       macro avg       0.97      0.98      0.97        38
    weighted avg       0.98      0.97      0.97        38

模型结构

```python
print(model.summary())
```

    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     dense (Dense)               (None, 10)                50        
    
     dense_1 (Dense)             (None, 3)                 33        
    
    =================================================================
    Total params: 83
    Trainable params: 83
    Non-trainable params: 0
    _________________________________________________________________
    None

##### 用tensorflow实现fashion-mnist服饰分类

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
```

①导数据集

```python
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
```

    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-labels-idx1-ubyte.gz
    29515/29515 [==============================] - 0s 6us/step
    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-images-idx3-ubyte.gz
    26421880/26421880 [==============================] - 18s 1us/step
    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-labels-idx1-ubyte.gz
    5148/5148 [==============================] - 0s 0s/step
    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-images-idx3-ubyte.gz
    4422102/4422102 [==============================] - 4s 1us/step

②探索数据

1,基本描述信息

```python
print( "x_train shape:", x_train.shape, "y_train shape:", y_train.shape)
print( "x_test shape:", x_test.shape, "y_test shape:", y_test.shape)
print(y_train[:20]) #前20个label
print(len(x_train)) #训练集样本个数
```

    x_train shape: (60000, 28, 28) y_train shape: (60000,)
    x_test shape: (10000, 28, 28) y_test shape: (10000,)
    [9 0 0 3 0 2 7 2 5 5 0 9 5 5 7 9 1 0 6 4]
    60000

2,显示单张图片

```python
plt.imshow(x_train[0], cmap = 'gray')
plt.colorbar()
plt.grid(False)
plt.show()
```

![png](https://img2022.cnblogs.com/blog/2629720/202210/2629720-20221010125542432-683157394.png)

3,显示多张图片

```python
class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[y_train[i]])
plt.show()
```

![png](https://img2022.cnblogs.com/blog/2629720/202210/2629720-20221010125539513-980164487.png)

4,标准化

```python
x_train = x_train / 255.0
x_test = x_test / 255.0
```

③构建模型

1,定义层

```python
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape = (28, 28)),
 tf.keras.layers.Dense(128, activation ='relu'),
 tf.keras.layers.Dropout(0.05),
 tf.keras.layers.Dense(10, activation ='softmax')
])
```

2,编译模型

```python
model.compile(optimizer='adam',
 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
 metrics=['accuracy'])
```

3,查看模型结构

```python
model.summary()
```

    Model: "sequential_1"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     flatten (Flatten)           (None, 784)               0         
    
     dense_2 (Dense)             (None, 128)               100480    
    
     dropout (Dropout)           (None, 128)               0         
    
     dense_3 (Dense)             (None, 10)                1290      
    
    =================================================================
    Total params: 101,770
    Trainable params: 101,770
    Non-trainable params: 0
    _________________________________________________________________

④训练模型

1,训练集训练,迭代10次

```python
history = model.fit(x_train, y_train, epochs=10)
```

    Epoch 1/10
    1875/1875 [==============================] - 6s 3ms/step - loss: 0.5066 - accuracy: 0.8221
    Epoch 2/10
    1875/1875 [==============================] - 4s 2ms/step - loss: 0.3800 - accuracy: 0.8615
    Epoch 3/10
    1875/1875 [==============================] - 4s 2ms/step - loss: 0.3409 - accuracy: 0.8751
    Epoch 4/10
    1875/1875 [==============================] - 4s 2ms/step - loss: 0.3187 - accuracy: 0.8837
    Epoch 5/10
    1875/1875 [==============================] - 4s 2ms/step - loss: 0.3024 - accuracy: 0.8886
    Epoch 6/10
    1875/1875 [==============================] - 3s 2ms/step - loss: 0.2891 - accuracy: 0.8931
    Epoch 7/10
    1875/1875 [==============================] - 4s 2ms/step - loss: 0.2781 - accuracy: 0.8962
    Epoch 8/10
    1875/1875 [==============================] - 3s 2ms/step - loss: 0.2672 - accuracy: 0.9004
    Epoch 9/10
    1875/1875 [==============================] - 4s 2ms/step - loss: 0.2577 - accuracy: 0.9037
    Epoch 10/10
    1875/1875 [==============================] - 4s 2ms/step - loss: 0.2508 - accuracy: 0.9056

显示每个epoch loss和accuracy的历史值

```python
history.history
```

    {'loss': [0.506623387336731,
      0.37999778985977173,
      0.3408657908439636,
      0.3187374770641327,
      0.3024260997772217,
      0.2890622913837433,
      0.27812787890434265,
      0.26724332571029663,
      0.2576807737350464,
      0.25076183676719666],
     'accuracy': [0.8220666646957397,
      0.8614500164985657,
      0.8751000165939331,
      0.8837000131607056,
      0.888616681098938,
      0.8930500149726868,
      0.8961833119392395,
      0.9004499912261963,
      0.9036833047866821,
      0.9056166410446167]}

图形化训练过程

```python
def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize = (8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0,1)
    plt.show()
plot_learning_curves(history)
```

![png](https://img2022.cnblogs.com/blog/2629720/202210/2629720-20221010125542820-667310230.png)

2,用测试集评估准确率

```python
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2) 
print('\nTest accuracy:', test_acc) 
predictions = model.predict(x_test) 
print(predictions[0]) 
print(' %d , %.2f%%' %(np.argmax(predictions[0]), max((predictions[0]))*100)) 
```

    313/313 - 1s - loss: 0.3281 - accuracy: 0.8826 - 1s/epoch - 4ms/step
    
    Test accuracy: 0.8826000094413757
    313/313 [==============================] - 1s 2ms/step
    [2.8944891e-07 9.0106245e-10 3.1913419e-08 1.7882961e-09 3.8579850e-07
     4.8054244e-02 1.7083392e-08 1.9777216e-01 1.4086240e-06 7.5417143e-01]
     9 , 75.42%

##### 用卷积神经网络实现fashion-mnist服饰分类

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix, classification_report
import os
```

①读取数据,理解图像

```python
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data() 
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32')
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') 
input_shape = (28, 28, 1)
```

②标准化

```python
x_train = x_train / 255.0
x_test = x_test / 255.0
```

③搭建模型

1,第一个卷积层

```python
model = keras.models.Sequential() 
model.add(keras.layers.Conv2D(filters=128, kernel_size=(3,3), input_shape=input_shape))
model.add(keras.layers.BatchNormalization()) 
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=2))
```

2,第二个卷积层

```python
model.add(keras.layers.Conv2D(filters=64, kernel_size=(3,3)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation('relu')) 
model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=2))
model.add(keras.layers.Dropout(0.2)) 
```

3,全连接层

```python
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation = 'relu'))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(64, activation = 'relu')) 
model.add(keras.layers.Dropout(0.2)) #
model.add(keras.layers.Dense(10, activation = 'softmax')) 
```

④编译模型

```python
model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics =['sparse_categorical_accuracy'])
model.summary()
```

    Model: "sequential_2"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     conv2d (Conv2D)             (None, 26, 26, 128)       1280      
    
     batch_normalization (BatchN  (None, 26, 26, 128)      512       
     ormalization)                                                   
    
     activation (Activation)     (None, 26, 26, 128)       0         
    
     max_pooling2d (MaxPooling2D  (None, 13, 13, 128)      0         
     )                                                               
    
     conv2d_1 (Conv2D)           (None, 11, 11, 64)        73792     
    
     batch_normalization_1 (Batc  (None, 11, 11, 64)       256       
     hNormalization)                                                 
    
     activation_1 (Activation)   (None, 11, 11, 64)        0         
    
     max_pooling2d_1 (MaxPooling  (None, 5, 5, 64)         0         
     2D)                                                             
    
     dropout_1 (Dropout)         (None, 5, 5, 64)          0         
    
     flatten_1 (Flatten)         (None, 1600)              0         
    
     dense_4 (Dense)             (None, 128)               204928    
    
     dropout_2 (Dropout)         (None, 128)               0         
    
     dense_5 (Dense)             (None, 64)                8256      
    
     dropout_3 (Dropout)         (None, 64)                0         
    
     dense_6 (Dense)             (None, 10)                650       
    
    =================================================================
    Total params: 289,674
    Trainable params: 289,290
    Non-trainable params: 384
    _________________________________________________________________

⑤训练

1,断点续训

```python
checkpoint_save_path = 'checkpoint_CNN/mnist.ckpt'
if os.path.exists(checkpoint_save_path + '.index'):
    print('----------load the model----------')
    model.load_weights(checkpoint_save_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path, save_weights_only=True,
save_best_only=True)
```

2,训练

迭代30次,一次需要80s,进40mins

```python
history = model.fit(x_train, y_train, epochs = 30, validation_data = (x_test, y_test),
callbacks=[cp_callback]) 
```

    Epoch 1/30
    1875/1875 [==============================] - 84s 44ms/step - loss: 0.5458 - sparse_categorical_accuracy: 0.8037 - val_loss: 0.4283 - val_sparse_categorical_accuracy: 0.8342
    Epoch 2/30
    1875/1875 [==============================] - 83s 44ms/step - loss: 0.3664 - sparse_categorical_accuracy: 0.8682 - val_loss: 0.3810 - val_sparse_categorical_accuracy: 0.8411
    Epoch 3/30
    1875/1875 [==============================] - 87s 47ms/step - loss: 0.3224 - sparse_categorical_accuracy: 0.8842 - val_loss: 0.2943 - val_sparse_categorical_accuracy: 0.8939
    Epoch 4/30
    1875/1875 [==============================] - 85s 45ms/step - loss: 0.2976 - sparse_categorical_accuracy: 0.8929 - val_loss: 0.3043 - val_sparse_categorical_accuracy: 0.8864
    Epoch 5/30
    1875/1875 [==============================] - 87s 46ms/step - loss: 0.2754 - sparse_categorical_accuracy: 0.8986 - val_loss: 0.2799 - val_sparse_categorical_accuracy: 0.8988
    Epoch 6/30
    1875/1875 [==============================] - 77s 41ms/step - loss: 0.2570 - sparse_categorical_accuracy: 0.9059 - val_loss: 0.2563 - val_sparse_categorical_accuracy: 0.9066
    Epoch 7/30
    1875/1875 [==============================] - 79s 42ms/step - loss: 0.2430 - sparse_categorical_accuracy: 0.9106 - val_loss: 0.2726 - val_sparse_categorical_accuracy: 0.9034
    Epoch 8/30
    1875/1875 [==============================] - 87s 46ms/step - loss: 0.2316 - sparse_categorical_accuracy: 0.9151 - val_loss: 0.2493 - val_sparse_categorical_accuracy: 0.9125
    Epoch 9/30
    1875/1875 [==============================] - 85s 45ms/step - loss: 0.2187 - sparse_categorical_accuracy: 0.9195 - val_loss: 0.2665 - val_sparse_categorical_accuracy: 0.9034
    Epoch 10/30
    1875/1875 [==============================] - 77s 41ms/step - loss: 0.2126 - sparse_categorical_accuracy: 0.9221 - val_loss: 0.2390 - val_sparse_categorical_accuracy: 0.9137
    Epoch 11/30
    1875/1875 [==============================] - 81s 43ms/step - loss: 0.2038 - sparse_categorical_accuracy: 0.9250 - val_loss: 0.2509 - val_sparse_categorical_accuracy: 0.9136
    Epoch 12/30
    1875/1875 [==============================] - 78s 42ms/step - loss: 0.1945 - sparse_categorical_accuracy: 0.9277 - val_loss: 0.2683 - val_sparse_categorical_accuracy: 0.9055
    Epoch 13/30
    1875/1875 [==============================] - 78s 42ms/step - loss: 0.1883 - sparse_categorical_accuracy: 0.9289 - val_loss: 0.2644 - val_sparse_categorical_accuracy: 0.9032
    Epoch 14/30
    1875/1875 [==============================] - 79s 42ms/step - loss: 0.1838 - sparse_categorical_accuracy: 0.9315 - val_loss: 0.2439 - val_sparse_categorical_accuracy: 0.9133
    Epoch 15/30
    1875/1875 [==============================] - 78s 42ms/step - loss: 0.1743 - sparse_categorical_accuracy: 0.9349 - val_loss: 0.2571 - val_sparse_categorical_accuracy: 0.9130
    Epoch 16/30
    1875/1875 [==============================] - 79s 42ms/step - loss: 0.1710 - sparse_categorical_accuracy: 0.9359 - val_loss: 0.2388 - val_sparse_categorical_accuracy: 0.9177
    Epoch 17/30
    1875/1875 [==============================] - 78s 42ms/step - loss: 0.1683 - sparse_categorical_accuracy: 0.9381 - val_loss: 0.2838 - val_sparse_categorical_accuracy: 0.9087
    Epoch 18/30
    1875/1875 [==============================] - 78s 42ms/step - loss: 0.1604 - sparse_categorical_accuracy: 0.9409 - val_loss: 0.2425 - val_sparse_categorical_accuracy: 0.9172
    Epoch 19/30
    1875/1875 [==============================] - 80s 43ms/step - loss: 0.1576 - sparse_categorical_accuracy: 0.9401 - val_loss: 0.2568 - val_sparse_categorical_accuracy: 0.9161
    Epoch 20/30
    1875/1875 [==============================] - 84s 45ms/step - loss: 0.1572 - sparse_categorical_accuracy: 0.9409 - val_loss: 0.2672 - val_sparse_categorical_accuracy: 0.9076
    Epoch 21/30
    1875/1875 [==============================] - 82s 44ms/step - loss: 0.1506 - sparse_categorical_accuracy: 0.9434 - val_loss: 0.2643 - val_sparse_categorical_accuracy: 0.9124
    Epoch 22/30
    1875/1875 [==============================] - 82s 44ms/step - loss: 0.1496 - sparse_categorical_accuracy: 0.9449 - val_loss: 0.2574 - val_sparse_categorical_accuracy: 0.9160
    Epoch 23/30
    1875/1875 [==============================] - 79s 42ms/step - loss: 0.1448 - sparse_categorical_accuracy: 0.9457 - val_loss: 0.2454 - val_sparse_categorical_accuracy: 0.9162
    Epoch 24/30
    1875/1875 [==============================] - 79s 42ms/step - loss: 0.1396 - sparse_categorical_accuracy: 0.9473 - val_loss: 0.2552 - val_sparse_categorical_accuracy: 0.9171
    Epoch 25/30
    1875/1875 [==============================] - 81s 43ms/step - loss: 0.1417 - sparse_categorical_accuracy: 0.9478 - val_loss: 0.2597 - val_sparse_categorical_accuracy: 0.9180
    Epoch 26/30
    1875/1875 [==============================] - 82s 44ms/step - loss: 0.1379 - sparse_categorical_accuracy: 0.9479 - val_loss: 0.2551 - val_sparse_categorical_accuracy: 0.9192
    Epoch 27/30
    1875/1875 [==============================] - 81s 43ms/step - loss: 0.1347 - sparse_categorical_accuracy: 0.9493 - val_loss: 0.3071 - val_sparse_categorical_accuracy: 0.8981
    Epoch 28/30
    1875/1875 [==============================] - 81s 43ms/step - loss: 0.1282 - sparse_categorical_accuracy: 0.9520 - val_loss: 0.2555 - val_sparse_categorical_accuracy: 0.9149
    Epoch 29/30
    1875/1875 [==============================] - 81s 43ms/step - loss: 0.1286 - sparse_categorical_accuracy: 0.9523 - val_loss: 0.2935 - val_sparse_categorical_accuracy: 0.9121
    Epoch 30/30
    1875/1875 [==============================] - 82s 44ms/step - loss: 0.1264 - sparse_categorical_accuracy: 0.9542 - val_loss: 0.2639 - val_sparse_categorical_accuracy: 0.9153

3,绘制acc和loss曲线

```python
def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.grid(True)
    plt.ylim(0, 1)
    plt.show()
plot_learning_curves(history)
```

![png](https://img2022.cnblogs.com/blog/2629720/202210/2629720-20221010125539901-1698307811.png)

4,保存模型

```python
model.save('fashion_model.h5')
```

⑥评估和预测

1,评估evaluate

```python
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('\nTest accuracy:', test_acc)
```

    313/313 - 4s - loss: 0.2639 - sparse_categorical_accuracy: 0.9153 - 4s/epoch - 12ms/step
    
    Test accuracy: 0.9153000116348267

2,预测predict

```python
predictions = model.predict(x_test)
print(predictions[0])
print(' %d , %.2f%%' %(np.argmax(predictions[0]), max((predictions[0]))*100))
```

    313/313 [==============================] - 4s 12ms/step
    [1.9348605e-17 2.0841505e-16 7.7804646e-13 1.9641095e-16 8.0456529e-13
     1.6399678e-08 3.9137113e-13 1.6209668e-06 2.5761291e-14 9.9999833e-01]
     9 , 100.00%

⑦查看模型分类效果

1,绘制混淆矩阵

```python
y_true = y_test
y_pred = np.argmax(predictions, axis=1)
pd.crosstab(y_true, y_pred)
confusion_matrix(y_true, y_pred)
```

    array([[800,   1,  18,  27,   1,   0, 145,   0,   8,   0],
           [  1, 982,   0,  10,   1,   0,   3,   0,   3,   0],
           [ 13,   0, 850,   7,  53,   0,  76,   0,   1,   0],
           [  9,   3,  13, 929,  16,   0,  29,   0,   1,   0],
           [  0,   0,  33,  18, 868,   0,  80,   0,   1,   0],
           [  0,   0,   0,   0,   0, 980,   0,  13,   0,   7],
           [ 57,   2,  44,  27,  44,   0, 819,   0,   7,   0],
           [  0,   0,   0,   0,   0,   6,   0, 981,   0,  13],
           [  1,   0,   2,   3,   0,   1,   5,   1, 987,   0],
           [  0,   0,   0,   0,   0,   5,   0,  38,   0, 957]], dtype=int64)

2,打印分类报告

```python
print(classification_report(y_true, y_pred))
```

                  precision    recall  f1-score   support
    
               0       0.91      0.80      0.85      1000
               1       0.99      0.98      0.99      1000
               2       0.89      0.85      0.87      1000
               3       0.91      0.93      0.92      1000
               4       0.88      0.87      0.88      1000
               5       0.99      0.98      0.98      1000
               6       0.71      0.82      0.76      1000
               7       0.95      0.98      0.97      1000
               8       0.98      0.99      0.98      1000
               9       0.98      0.96      0.97      1000
    
        accuracy                           0.92     10000
       macro avg       0.92      0.92      0.92     10000
    weighted avg       0.92      0.92      0.92     10000
