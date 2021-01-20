# MATLAB笔记

![sdfgf](//qiniu.ajstudio.cn/2021-01-20-8041e52b02394f6199e5f1fe7a654445.png)

2222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222


## 常数

exp(n);以e的n次方，n为矩阵则分别运算返回矩阵

expm(x);矩阵x的指数，矩阵只能是方阵

log(n);以e为底部的对数函数

logm(n);以m对底数的对数函数

特殊字符：pi

## 基本运算

#### 常见函数

finder('a','b') 从字符串a中找b

fix():取整数

rem:求余数

exp(n):e的n次方

log(2):数学上的ln2

zeros（）构造零向量

whos:查看内存占用

disp:输出,这样可以输出多个数据

sqrt():开平方

sind():采用角度制，sin()采用弧度制

std():标准差,方差

median()；中位数

mean():平均分，求均值

det（）行列式的值

rank（）秩

trace（）迹

company（）;伴随

norm（）范数

cond（）条件数

conv()：两个或多个多项式乘积

deconv():两个多项式相除

flipud();上下翻转

fliplr();左右翻转

rot90();逆时针旋转90

注意：mean(矩阵)为按列取平均值组成的向量，mean(矩阵(:))为所有值的平均数

	- ｛a,b｝a,b为单元矩阵
	-   (a,b)a,b构成一个矩阵

#### 输入输出

disp();显示某个值，相当于c的print。参数仅为一个

input:输入

```
a=input('这里必须要参数');凄凄切切群群群群群群群群群群群群群群群群群群群群群群群群群群群群群群群群群群群群群群群群群群群群群群群群群群群群群群群群群群群群凄凄切切群群群群群群群群群群
必须输入数字

a=input('这里必须要参数','s');
输入的当做字符串
```

clc:清屏

clear:清除工作区的变量

keyboard;暂停程序的运行，将控制权交给键盘

##### 数据格式

format long 

format long  e

#### 字符串

"":两个这样的字符串连接为两个字符串数组，为字符数组

'':两个这样的字符串可以连接为新字符串，为字符向量




##### 创建

```
str0 = 'hello world';  % 单引号引起
str1 = 'I''m a student';  % 字符串中单引号写两遍
str3 = ['I''m' 'a' 'student'];  % 方括号链接多字符串
str4 = strcat(str0, str1);  % strcat连接字符串函数
str5 = strvcat(str0, str1);  % strvcat连接产生多行字符串
str7 = char(str6);  % 把ASCII转为字符串

```

##### 比较

```
strcmp(str0, str1);  % 相等为1，不等为0
strncmp(str0, str1, 3);  % 比较前3个是否相等(n)
strcmpi(str0, str1);  % 忽略大小写比较(i)
strncmpi(str0, str1, 3);  % 忽略大小写比较前3个是否相等
```

upper();字符串转为大写

lower();字符串转为小写

isstrprop(var,'digit');判断是否为数字

#### 查找与替换

```
strfind(str0, str1);  % 在str0找到str1的位置
strmatch(str1, str0);  % 在str0字符串数组中找到str1开头的行数
strtok(str0);  % 截取str0第一个分隔符（空格，tab，回车）前的部分
strrep(str0, str1, str2);  % 在str0中用str2替换str1
```

##### 转换

```
%转换
% ___2___  -->  如num2str，将数字转字符串； dec2hex，将十进制转十六进制
str_b = num2str(b);
% abs，double取ASCII码；char把ASCII转字符串
abs_str = abs('aAaA');  
```



##### 其他

```
upper(str0);  % 转大写，lower转小写
strjust(str0, 'right');  % 将str0右对齐，left左对齐，center中间对齐
strtrim(str0);  % 删除str0开头结尾空格
eval(str0);  % 将str0作为代码执行
```

#### 产生矩阵

a=[1,2;3,4];

a=[1 2;3 4];

a=1:10等价于a=1:1:10

a=linspace(0,10,5);生成[0,5]间共10个均分元素的矩阵

zeros();全零矩阵

ones(r,s):幺矩阵

eye(r,s):单位矩阵

size():返回矩阵行,列数

##### rand

```markdown
# 1.randi()函数生成均匀分布的伪随机整数，范围为imin--imax，如果没指定imin，则默认为1。
r = randi(imax,n):生成n*n的矩阵
r = randi(imax,m,n)：生成m*n的矩阵
r = randi(imax,[m,n]):同上
r = randi(imax,m,n,p,...)：生成m*n*p*...的矩阵
r = randi(imax,[m,n,p,...])同上
r = randi(imax)：1*1的矩阵
r = randi(imax,size(A))：和size(A）同维的矩阵
r = randi([imin,imax],...)

# 2.rand ()函数生成0-1的矩阵
r = rand(a,b):生成a*b的0-1(1以内)的矩阵 
r = rand(3)：生成3*3的0-1(1以内)的矩阵 
r = rand(size(A))：和size(A）同维的矩阵
r = rand(a，b,'single/double');指定精度的矩阵
r = rand(RandSteam,a,b);Rand Steam要生成的限制，生成什么样子的种子。a*b

# 3.randn（）函数,生成的是均值为0方差为1的正态分布形状的矩阵

# 4.
y=a+(b-a)*rand(n，m),(a,b)的随机矩阵
y=a+(b-a+1)*rand(n，m),[a,b]的随机矩阵
y=a+sqrt(b)*rand(n，m)：均值为a，方差为b
```

zeros():零矩阵

##### 稀疏矩阵

sparse(a)

只储存非零元素的位置与值

完全矩阵：按列储存

#### 矩阵运算

```
矩阵*矩阵：矩阵相乘
矩阵.*矩阵:点乘

矩阵/矩阵
矩阵\矩阵
矩阵./矩阵
矩阵.\矩阵

矩阵^数字：矩阵的数字次方
矩阵.^矩阵：矩阵元素的数字次方

矩阵':转置
矩阵.':共轭转置
```







##### 特征值与特征向量

A必须为方阵

e=eig(A)，e为特征向量

[v,k]=eig(A)，v特征向量，k特征值对角阵





a=diag(A,k):提取主对角线元素，生成列向量，A为矩阵，k为向量位置，不写k为主对角线

a=diag(向量);构造对角线为该向量的矩阵，其他为0

a=tril(A,k);偏移主对角线k的对角线右上方全变为0

diag(diag(a)):提取后还要构造对角阵



**同时存在inv与pinv用pinv**

```
inv():求逆矩阵 

pinv(A)：伪逆阵/广义逆矩阵,与原阵相乘不一定是单位阵
(A不是方阵，求广义逆矩阵，A是可逆矩阵，求逆矩阵)
pinv(A,tol),其中tol为误差

```

#### 矩阵修改

```matlab
% 矩阵的修改
    %部分替换
        chg_a = a;
        chg_a(2,3) = 4;  % (行，列)元素替换
        chg_a(1,:) = [2,2,2];  % (行,:)替换行，为[]删除该行
        chg_a(:,1) = [];  % (:,列)替换列，为[]删除该列
    % 指定维数拼接
        c1_a = cat(1,a,a);  % 垂直拼接
        c2_a = cat(2,a,a);  % 水平拼接
    % *变维
        rs_a = reshape(a,1,9);  % 元素个数不变，矩阵变为m*n
```

#### 信息获取

  [row_a, col_a] = size(a);  % [行数，列数]
  len_a = length(a);矩阵的列数或者行数的最大的一个/行列数的总元素数

length(x) = max(size(x))

find();寻找矩阵中大于0的数并返回其序号（上到下，左到右）

```
A=[16 4 18;19 3 -5;-7 8 31];
A>=15&A<=20
find(A>=15&A<=20)


ans:
   1   0   1
   1   0   0
   0   0   0

ans =

     1
     2
     7
```



##### 多维数组

```
% 直接法
mul_1(:,:,1) = [1,2,3;2,3,4];
mul_1(:,:,2) = [3,4,5;4,5,6];
% *扩展法
mul_2 = [1,2,3;2,3,4];
mul_2(:,:,2) = [3,4,5;4,5,6];  % 若不赋值第一页，第一页全为0
% cat法
mul_31 = [1,2,3;2,3,4];
mul_32 = [3,4,5;4,5,6];
mul_3 = cat(3,mul_31,mul_32);  % 把a1a2按照“3”维连接
```

多维数字按照从左到右，从上到下的方式储存数据

### 逻辑与结构

#### 判断

通过缩进描述关系

```matlab
%
if a>0
​	asdf
elseif case2
​	adsf
else case3
​	disp(a);
end


%字符串用格式'字符串'，数字直接写
switch a
	case '字符串'
		asdf
	case 2
		d
	otherwise
		sdaf
end

%
try
	z=x*y
catch
	disp(x);
end
```



num2cell：将数值矩阵转换为单元矩阵

#### 循环

```matlab
for a=0:1:10  %此处填数据，按照列分别循环输出
	disp(a);
end


while a>2
	disp(a);
	a=a-1;
end

%程序控制关键字
continue;
break;
return;
```

#### 匿名函数

​	a=@(x,y) x+y 



#### 导函数

p=polyder(P):求多项式的导函数

p=polyder(P,Q):多项式QP乘的导数

[p,q]=polyder(P,Q):P/Q的倒数，分子p,分母q



### 脚本文件与函数文件

```
脚本
新建脚本x.m,控制台输入x.m就执行之
```



```matlab
%主函数名与函数文件名保持一致，子函数无所谓不能被外部调用

function [outputArg1,outputArg2] = werg(inputArg1,inputArg2)
outputArg1 = subFun(inputArg1);
outputArg2 = inputArg2;
end

function [a]= subFun(aa)
a=aa*2;
end




%参数不定的情况

%*函数输入输出参数可以不定
%nargin：输入参数个数，nargout：输出参数个数
%varargin：输入参数内容的元胞数组，varargout：输出参数
%以下是函数文件
function varargout = idk(varargin)
x = length(varargin);
varargout{1} = x;
varargout{2} = x+1;
end

```



### 求解方程组

 ax=b

- x=inv(a)*b;（注意维度问题，b注意有时换成列向量）
- x=a\b也行

A\B:A*X=B

A/B:B*X=A

#### 求方程组的根

已知
$$
P(x)=x^4+2*x^3+5*x^2+3*x+10
$$
p = [1 2 5 3 10]

x=roots(p)

### 绘图

```matlab
% 示例数据
x = 0:0.1:2*pi;
y1 = sin(x);
y2 = cos(x);
y = sin(x); 
z = cos(x);
```

figure;% 开启新绘图窗口，下一次绘图在新窗口。无该函数，则只绘最后一个绘图函数

hold on;  % 本次绘图中，在之后所有绘图里，在原有窗口曲线上增加绘制下一个图形

hold off; %取消hold on

grid on;显示网格线

grid off;不显示网格线

view(a,e)；视角函数，e表示仰角，a表示方位角

impulse(sys);系统单位脉冲响应

compass();绘制复数矢量图

#### 二维绘图

> 先plot，再设置，如legend...
>
> 特例subplot，hold on;要在plot之前

```matlab
%一个窗口绘制一条
plot(y1);纵坐标为y1的值，横坐标自动取值
plot(x,y1);纵坐标为y1的值，横坐标为x


%一个窗口绘制多条
plot(矩阵);一个窗口绘制列数个曲线，曲线的y轴等于该列所有行元素，x轴随机生成
plot(x,y1,x,y2);

%设置线条
plot(x, y1, 'b:o',x,y2,'r-+'); 
% b蓝 g绿 r红 c青 m紫 y黄 k黑 w白
% -实线 :点线 --虚线 -.点画线
% .实点 o圆圈 x叉 +十字 *星号 s方块 d钻石 v下三角 ^上三角 <左三角 >右三角 p五角星 h六角星

%设置x,y轴区域
plot(x, y1);
axis([-1*pi, 3*pi, -1.5, 1.5]);  % 规定横纵坐标范围-x,x,-y,y。注意，传入的是数组

%设置标签
plot(x, y1);
title('a title');% 图像标题
xlabel('this is x');% x轴标记，同理还有ylabel，zlabel
xlabel('x','fontsize',20);%设置字大小


%设置图例
%%1
plot(x, y1);
hold on;
plot(x, y2);
legend('ha','haha');
%%2
plot(x, y1);
legend('hahaha', 'location', 'best');  % str的顺序与绘图顺序一致; 'best'指图例放置的位置最佳化，还有其他位置


%分割区域，计数是从左到右边，再从上到下
subplot(2, 2, 1);  % 分割成2列x2行区域，在第一块区域绘制下一个图形
plot(x, y1);  % y1被绘制在4块区域的第一块
subplot(2, 2, 2);  % 分割方法相同，区域改变
plot(x, y2);  % y2在第二块区域


```



##### 二维特殊图形绘制

```matlab
%柱状图
bar(x, y, width, '参数');  % x横坐标向量，m个元素; y为向量时，每个x画一竖条共m条，矩阵mxn时，每个x画n条;
% width宽度默认0.8，超过1各条会重叠;
% 参数有grouped分组式，stacked堆栈式; 默认grouped
% bar垂直柱状图,barh水平柱状图,bar3三维柱状图,barh3水平三维柱状图(三维多一个参数detached, 且为默认)
%饼形图
pie(x, explode, 'lable');  % x为向量显示每个元素占总和百分比, 为矩阵显示每个元素占所有总和百分比
% explode向量与x同长度，为1表示该元素被分离突出显示，默认全0不分离
% pie3绘制三维饼图
%直方图
hist(y, n);  % y为向量，把横坐标分为n段绘制
hist(y, x);  % x为向量，用于指定每段中间值, 若取N = hist(y, x), N为每段元素个数
%离散数据图
stairs(x, y, 'b-o');  % 阶梯图，参数同plot
stem(x, y, 'fill');  % 火柴杆图，参数fill是填充火柴杆，或定义线形
candle(HI, LO, CL, OP);  % 蜡烛图:HI为最高价格向量,LO为最低价格向量,CL为收盘价格向量,OP为开盘价格向量
%向量图
compass(u, v, 'b-o');  % 罗盘图横坐标u纵坐标v
compass(Z, 'b-o');  % 罗盘图复向量Z
feather(u, v, 'b-o');  % 羽毛图横坐标u纵坐标v
feather(Z, 'b-o');  % 羽毛图复向量Z
quiver(x, y, u, v);  % 以(x, y)为起点(u, v)为终点向量场图
%极坐标图
% polar(theta, rho, 'b-o');  % 极角theta, 半径rho
theta = -pi:0.01:pi;
rho = sin(theta);
polar(theta, rho, 'b')
%对数坐标图
semilogx(x1, y1, 'b-o');  % 把x轴对数刻度表示, semilogy是y轴对数刻度表示，loglog是两个坐标都用对数表示
%双纵坐标
plotyy(x1, y1, x2, y2, 'fun1', 'fun2');  % fun规定了两条条线的绘制方式，如plot,semilogx,semilogy,loglog,stem等
%函数绘图
f = 'sin(2*x)';
ezplot(f, [0, 2*pi]);  % 绘制f并规定横坐标范围，也有[xmin, xmax, ymin, ymax]
x = '2*cos(t)';
y = '4*sin(t)';
ezplot(x, y);  % 绘制x(t),y(t)在[0, 2*pi]图像, 也可以在最后用[tmin, tmax]规定t的范围

```



plot3(x1,y1,z1):三维曲线

meshgrid(x,y):生成平面网格坐标

xlabel('x轴','fontsize',16);:同理还有，y,zlabel

title('标题','fontsize',16);

axis equal：横纵坐标刻度一致

axis square:将当前坐标系图形设置为方形。横轴及纵轴比例是1：1

axis(1,1,2,2):限制x轴在1->1,y轴在2->2

[x,y,z]=sphere(n);n代表环数

[x,y,z]=ctkubder();

[x,y,z]=peaks();

 ezplot（<隐函数表达式>,[x最小值，x最大值，y最小值，y最大值])

ez...一般是绘制隐匿函数





#### 三维曲面

```matlab
%三维曲线
plot3(x, y, z);
%三维曲面
	%三维网格
    x = -5:0.1:5;% 规定了x轴采样点，也规定了x轴范围
    y = -4:0.1:4;% 规定了y轴采样点，也规定了y轴范围
    [X, Y] = meshgrid(x, y);% 得到了xoy面网格点
    Z = X.^2+Y.^2;
    mesh(X, Y, Z);% XY是meshgrid得到的网格点，Z是网格顶点，c是用色矩阵可省略
    %三维表面图
    x = -5:0.1:5;
    y = -4:0.1:4;
    [X, Y] = meshgrid(x, y);
    Z = X.^2+Y.^2;% 以上部分同上
    surf(X, Y, Z);% 与上一个类似
```

mesh(x,y,z);%补面无色

surf(x,y,z);%同mesh,补面有色

#### 其他三维图形

- 三维条形图 bar3(x,y)
- 三维饼图 pie3(x,explode)
- 三维实心图 fill3(x,y,z,e)
- 三维散点图 scatter3(x,y,z,e)
- 三维杆图 stem3(x,y,z)
- 三维箭头图 quiver3(x,y,z,u,v,w)

#### 曲面剪裁

[x,y,z]=sphere(n)

z(:,1:4)=NaN %NaN是预定义变量，非数字

surf(x,y,z)

#### 三维图形着色

## 多项式

### 多项式

#### 创建

```
%创建
p = [1, 2, 3, 4];  % 系数向量，按x降幂排列，最右边是常数
f1 = poly2str(p, 'x');%生成字符串 f1 = x^3 + 2 x^2 + 3 x + 4，不被认可的运算式
f2 = poly2sym(p);%生成可用的符号函数 f2 = x^3 + 2*x^2 + 3*x + 4
```

#### 运算

```
%求值
x = 4;
y1 = polyval(p, x);  % 代入求值；若x为矩阵，则对每个值单独求值
%求根
r = roots(p); % p同上，由系数求根，结果为根植矩阵
%求系数
p0 = poly(r);  % 由根求系数，结果为系数矩阵
```

```
%数据统计
%矩阵最大最小值
y = max(X);  % 求矩阵X的最大值，min最小值
[y, k] = max(X);  % 求最大值，k为该值的角标
[y, k] = max(A, [], 2);  % A是矩阵，'2'时返回y每一行最大元素构成的列向量，k元素所在列；'1'时与上述相同
%均值和中值
y = mean(X);  % 均值
y = median(X);  % 中值
y = mean(A, 2);  % '2'时返回y每一行均值构成的列向量；'1'时与上述相同
y = median(A, 2);  % '2'时返回y每一行中值构成的列向量；'1'时与上述相同
%排序
Y = sort(A, 1, 'ascend');  % sort(矩阵, dim, 'method')dim为1按列排序，2按行排序；ascend升序，descend降序
[Y, I] = sort(A, 1, 'ascend');  % I保留了元素之前在A的位置
%求和求积累加累乘
y = sum(X);  % 求和
y = prod(X);  % 求积
y = cumsum(X);  % 累加
y = cumprod(X);  % 累乘
%%
%*数值计算
%最(极)值
%多元函数在给定初值附近找最小值点
x = fminsearch(fun, x0);
%函数零点
x = fzero(fun, x0);  % 在给定初值x0附近找零点
```





### 数据统计

```
X = [2, 3, 9, 15, 6, 7, 4];
A = [1, 7, 2; 9, 5, 3; 8, 4 ,6];
B = [1, 7, 3; 9, 5, 3; 8, 4 ,6];
Z=[134,1,345,23,4,5234;23,6,43,45,4,35;3,54,2345,6783456,234,673];
```

#### 最值,和，积

y=max(矩阵);求出每列的最大值的行向量

[y,k]=max(向量)：最大值y,序号k

[Y,U]=max(矩阵)：Y为最大值向量，U为角标向量，记录每列的最大值的向量

max(max(A)),max(A(:)):求矩阵最大值

U=max(X，Y)：将同形矩阵x,y对应位置大的结果放入u





min同上

sum()

prod()

cum+prod/sum...:得出一个每个元素是前n项累...的和

#### 排序

[Y,I]=sort(A,dim,mode):Y是排序后矩阵，I是序号.dim是行(1)或列(2)排，mode是升降(descend)序

###  数据插值

```
method:
    linear：赋值中点（默认）
    nearest：赋值邻点
    pchip：曲线拟合
    spline：三次样条插值法，曲线拟合，比上面要求更高
```

####  一维插值

%yi = interp1(X, Y, xi, 'method')

```
X = [-3, -1, 0, 1, 3];
Y = [9, 1, 0, 1, 9];
y2 = interp1(X, Y, 2)
% 用spline方法(三次样条)差值预估在x=2的y的值
y2m = interp1(X, Y, 2, 'spline')
```

#### 二维插值

%zi = interp2(X, Y, Z, xi, yi, 'method')

```
X = [2:6];
Y = [1:5];
Z = [1:5;1:5;1:5;1:5;1:5;];
x1=1:0.5:5;
y1=1:0.5:5;
z = interp2(X,Y,Z,x1,y2)
```

### 曲线拟合

polyfit

polyval

## 注意

1. A(5):使用括号索引向量及其转置，索引矩阵时，是从上到下，从左到右地索引

   A(1,3);索引矩阵

2. "dddd''表示带单引号的字符串"，‘ddd’代表字符串

3. 注释用%

4. 函数不能大写

5. :占位则表示所有

6. ；用于控制输入

7. MATLAB序号从1开始

8. i=2; a=2i；则a=2.0000i。因为此时i置换定义为虚数

9. MATLAB ~等同于c的！，优先级几乎最高

10. int为8字节，int64类型不能参与任何运算





```
%%
%*数值计算
%最(极)值
%多元函数在给定初值附近找最小值点
x = fminsearch(fun, x0);
%函数零点
x = fzero(fun, x0);  % 在给定初值x0附近找零点




%%
%符号对象创建
%sym函数
p = sin(pi/3);
P = sym(p, 'r');  % 用数值p创建符号常量P；'d'浮点数'f'有理分式的浮点数'e'有理数和误差'r'有理数
%syms函数
syms x;  % 声明符号变量
f = 7*x^2 + 2*x+9;  % 创建符号函数
%符号运算
% 加减乘除外
% '转置 ； ==相等 ； ~=不等
% sin, cos, tan; asin, acos, atan 三角反三角
% sinh, cosh, tanh; asinh, acosh, atanh 双曲反双曲
% conj复数共轭；real复数实部；imag复数虚部；abs复数模；angle复数幅角
% diag矩阵对角；triu矩阵上三角；tril矩阵下三角；inv逆矩阵；det行列式；rank秩；poly特征多项式；
% |----expm矩阵指数函数；eig矩阵特征值和特征向量；svd奇异值分解；
%符号对象精度转换
digits;  % 显示当前用于计算的精度
digits(16);  % 将计算精度改为16位，降低精度有时可以加快程序运算速度或减少空间占用
a16 = vpa(sqrt(2));  % vpa括起的运算使sqrt(2)运算按照规定的精度执行
a8 = vpa(sqrt(2), 8);  % 在vpa内控制精度，离开这一步精度恢复

%%
%符号多项式函数运算
%*符号表达形式与相互转化
%多项式展开整理
g = expand(f);  % 展开
h = collect(g);  % 整理(默认按x整理)
h1 = collect(f, x);  % 按x整理（降幂排列）
%因式分解展开质因数
fac = factor(h);  % 因式分解
factor(12);  % 对12分解质因数
%符号多项式向量形式与计算
syms a b c;
n = [a, b, c];
roots(n);  % 求符号多项式ax^2+bx+c的根
n = [1, 2, 3];
roots(n);  % 求符号多项式带入a=1, b=2, c=3的根
%*反函数
fi = finverse(f, x);  % 对f中的变量x求反函数

%%
%符号微积分
%函数的极限和级数运算
% 常量a，b
%极限
limit(f, x, 4);  % 求f(x), x->4
limit(f, 4);  % 默认变量->4
limit(f);  % 默认变量->0
limit(f, x, 4, 'right');  % 求f(x), x->4+, 'left' x->4-
%*基本级数运算
%求和
symsum(s, x, 3, 5);  % 计算表达式s变量x从3到5的级数和，或symsum(s, x, [a b])或symsum(s, x, [a;b])
symsum(s, 3, 5);  % 计算s默认变量从3到5的级数和
symsum(s);  % 计算s默认变量从0到n-1的级数和
%一维泰勒展开
taylor(f, x, 4);  % f在x=4处展开为五阶泰勒级数
taylor(f, x);  % f在x=0处展开为五阶泰勒级数
taylor(f);  % f在默认变量=0处展开为五阶泰勒级数
%符号微分
%单变量求导（单变量偏导）
n = 1;  % 常量n
fn = diff(f, x, n);  % f对x的n阶导
f1 = diff(f, x);  % f对x的1阶导
diff(f, n);  % f对默认变量的n阶导
diff(f);  % 默认变量1阶导
%多元偏导
fxy = diff(f, x, y);  % 先求x偏导，再求y偏导
fxyz = diff(f, x, y, z);  % 先求x偏导，再求y偏导,再求z偏导
%符号积分
%积分命令
int(f, x, 1, 2);  % 函数f变量x在1~2区间定积分
int(f, 1, 2);  % 函数f默认变量在ab区间定积分
int(f, x);  % 函数f变量x不定积分
int(f);  % 函数f默认变量不定积分
% 傅里叶，拉普拉斯，Z变换

%%
%*符号方程求解
%符号代数方程
%一元方程
eqn1 = a*x==b;
S = solve(eqn1);  % 返回eqn符号解
%多元方程组
eqn21 = x-y==a;
eqn22 = 2*x+y==b;
[Sx, Sy] = solve(eqn21, eqn22, x, y);  % [Svar1,...SvarN]=solve(eqn1,...eqnM, var1,...varN),MN不一定相等
[Sxn, Syn] = solve(eqn21, eqn22, x, y, 'ReturnCondition', true);  % 加上参数ReturnCondition可返回通解及解的条件
% 其他参数(参数加上true生效)
% IgnoreProperty，忽略变量定义时一些假设
% IgnoreAnalyticConstraints，忽略分析限制；
% MaxDegree，大于3解显性解；
% PrincipleValue，仅主值
% Real，仅实数解
%非线性fsolve
X = fsolve(fun, X0, optimset(option));  % fun函数.m文件名；X0求根初值；option选项如('Display','off')不显示中间结果等

```

