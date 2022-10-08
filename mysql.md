# 1. Mysql

概念：数据仓库， **软件**，安装在操作系统上。

作用：存储数据，管理数据



## 1.1 数据库分类

关系型数据库： 

- MySQL， Oracle，DB2
- 通过表和表之间，行和列之间的关系进行数据的存储， 学员信息表，考勤表

非关系型数据库：

- Redis， MongDB
- 对象存储，通过对象的自身属性来决定

==DBMS(数据库管理系统)==

- 数据库的管理软件，科学有效的管理我们的数据，维护和获取数据
- MySQL

## 1.2 连接数据库

命令行连接

```sql
mysql -uroot -p123456  --连接数据库
flush priviles;  --刷新权限
--------------------------------
-- 所有的语句都使用; 结尾
show database; -- 查看所有的数据库
use school -- 切换数据库 use 数据库名
show tables; -- 查看数据库中所有表
describe student; --显示数据库中所有的表的信息
create database westos; -- 创建一个数据库

exit; --推出连接
-- 当行注释 （SQL 本来的注释）
/*

*/ （sql 多行注释）

```



数据库xxx语言

- DDL 	定义

- DML	操作

- DQL	查询

- DCL	控制



# 2. 操作数据库

操作数据库 》 操作数据库中的表 》 操作数据库中表的数据

不区分大小写

## 2.1 操作数据库

- 创建数据库

- 删除数据库

- 使用数据库

create database if not exists westos

drop database if exists westos

-- 如果表名或字段名是特殊字符， 需要加`` 

````
 use `school`
````



show databases  -- 查看所有的数据库

## 2.2 数据库列类型

> 数值

- tinyint     十分小的数据	1个字节
- smallint  较小的数据 
- mediumint
- **int**
- bigint
- float    4 个字节
- double 8 个字节
- decimal 字符串形式的浮点数  金融计算的时候，一般使用decimal

> 字符串

- char  	字符串   0- 255
- **varchar 可变字符串  0 - 65535**
- tinytext  微型文本     2^8 -1
- text        文本串      2^16 -1

> 时间日期

java.util.Date

- date YYYY-MM-DD, 日期格式
- time HH: mm: ss 时间格式
- datetime  YYYY-MM-DD HH: mm: ss
- timestamp  时间戳, 1970.1.1到现在的毫秒数
- year 年份表示

> null

- 没有值， 未知
- ==不要使用NULL进行运算，结果为NULL==



## 2.3 数据库的字段属性

==unsigned==

- 无符号的整数
- 声明了该列不能声明负数

zerofill：

- 0 填充
- 不足的位数，使用0来填充  

自增：

- 通常理解为自增，自动在上一条记录的基础上+1
- 通常用来设计唯一的主键 ~index，必须是整数类型
- 可以自定义设计主键自增的起始值和步长

非空 NULL not null

- 假设设置为 not null， 如果不赋值，就会报错
- NULL， 默认为null！

默认：

- 设置默认值

```
id 	主键
`version`  	乐观锁
is_delete  	伪删除
gmt_create  创建时间
gmt_update 	修改时间
```

一般一个表只有一个唯一的主键

```sql
CREATE TABLE IF NOT EXISTS `student` (
	`id` INT(4) NOT NULL AUTO_INCREMENT COMMENT '学号',
	`sex` VARCHAR(20) NOT NULL DEFAULT '女' COMMENT '性别',
	PRIMARY KEY(`id`)
) ENGINE  = INNODB DEFAULT CHARSET = utf8

```

格式

```sql
	'字段名' 列类型 [属性] [索引] [注释],
	'字段名' 列类型 [属性] [索引] [注释],
    .....
	'字段名' 列类型 [属性] [索引] [注释]
)[表类型][字符串设置][注释]
```



##  2.4 数据表的类型

```sql
-- 关于数据库引擎
/*
INNODB 默认使用
MYISAM 早些年使用
*/
```



|            | MYISAM | INNODB        |
| ---------- | ------ | ------------- |
| 事务支持   | 不支持 | 支持          |
| 数据行锁定 | 不支持 | 支持          |
| 外键约束   | 不支持 | 支持          |
| 全文索引   | 支持   | 不支持        |
| 表空间大小 | 较小   | 较大，约为2倍 |

常规使用操作：

- MYISAM 节约空间， 速度较快
- INNODB 安全性高，事务处理，多表多用户操作



> 设置数据库表的字符集编码

```sql
CHARSET = utf8
```

可以在my.ini  中配置默认编码

```sql
character-set-server =utf8
```



```sql
-- 修改表名
alter table teacher rename as teacher1
-- 增加表的字段
alter table teacher1 add age int(11)
-- 修改表的字段（重命名， 修改约束！）
alter table teacher1 modify age varchar(11)   -- 修改约束
alter table teacher1 change age age1 int(1)  -- 字段重命名
-- 删除表的字段
alter table teacher1 drop age1
```

==所有的创建和删除操作尽量加上判断，以免报错~==

# 3. MySQL 数据管理

## 3.1 外键（了解）

```sql
-- alter table 表 add constraint 约束名 foreign key (`gradeid`) refrences `grade`(`gradeid`)
```

==最佳实践==

- 数据库就是单纯的表，只用来存数据
- 想使用外键（程序去实现）
- 不得使用外键与级联，一切外键概念必须在应用层解决





## 3.2 DML语言（全部记住）

==数据库意义==：数据存储，数据管理

DML语言：数据操作语言

- insert
- update
- delete



## 3.3 添加

```sql
-- 插入数据
-- insert into 表名([字段名1, 字段名2, 字段名3]) values('值1'), ('值2'),('值3')

-- 由于主键自增，我们可以省略（如果不写表的字段，他就会一一对应）
insert into `grade`(`gradename`) values('大二'),('大一')
```



## 3.4 修改

```sql
-- 修改学员名字，带了简介
update `student` set `name` = `Lee` where id = 1;
-- 不指定条件的情况下，会改动所有表！
update `student` set `name` = `Lee`
-- 修改多个属性，逗号隔开
update `student` set `name` = `Lee`, email = '2754642243@qq.com' where id = 1;
-- 通过多个条件定位数据

```



| 操作符             | 含义         | 范围 | 结果 |
| ------------------ | ------------ | ---- | ---- |
| between... and ... | 在某个范围内 |      |      |
| and                |              |      |      |
| or                 |              |      |      |
| <>   !=            | 不等于       |      |      |

注意

- column_name 是数据库的列，尽量带上``
- 条件，筛选的条件，如果没有指定，会修改所有的列
- value， 可以是一个具体的值，也可以是一个变量



## 3.5 删除

> delete 命令

```sql
-- 删除数据（避免这样写，会全部删除）
delete from `student`

-- 删除指定数据
delete from `student` where id = 1;
```

> truncate 命令

作用： 完全清空一个数据库表，表的结构和索引约束不会变！

> delete 和 truncate 区别

- 相同点： 都能删除数据，都不会删除表结构
- 不同：
  - truncate 重新设置 自增列 计数器会归零
  - truncate 不会影响事务



`delete 删除的问题` ， 重启数据库，现象

- innoDB 自增列会从1开始（存在内存中，断电即失）
- myisam 继续从上一个自增量开始（存在文件中，不会丢失）



# 4. DQL 查询数据（最重点）

（Data Query Language）

- 所有的查询操作都用它 select
- 简单的查询，复杂的查询都能做
- ==数据库中最核心的语言，最重要的语句==
- 使用频率最高的语句

## 4.1 指定查询字段



```sql
-- 查询全部的学生  select 字段 from 表
select * from student
-- 查询指定字段
select 	`studentNo`, `studentname` from student
-- 别名，给结果起一个名字 as
select 	`studentNo` as 学号, `studentname` as 学生姓名 from student
-- 函数 concat(a,b)
select concat('姓名:', studentname) as 新名字 from student

```

> 有的时候，列名字不是那么见名知意。 我们起别名 





> 去重 distinct



```sql
-- 查询一下有哪些同学参加了考试，成绩
select *from result 
-- 查询有哪些同学参加了考试
select `studentno` from result
-- 发现重复数据, 去除重复数据
select distinct `studentno` from result
select version() -- 查询系统版本 (函数)
select 100*3 -1 as 计算结果  --用来计算 (表达式)
select @@auto_increment_increment -- 查询自增的步长 (变量)
select `studentno`, `studentresult`+1 as '提分后' from result
```

==数据库中的表达式： 文本值，列，null， 函数， 计算表达式， 系统变量。。。==



## 4.2 where

作用：检索数据中的==符合条件==的值

> 逻辑运算符

| 运算符    | 语法                     | 描述                             |
| --------- | ------------------------ | -------------------------------- |
| and   &&  | a and b       a && b     | 逻辑与， 两个都为真，结果为真    |
| or   \|\| | a or b          a \|\| b | 逻辑或，其中一个为真，则结果为真 |
| not  !    | not a            ! a     | 逻辑非，真为假，假为真           |

```sql
-- 查询考试成绩在 95 ~ 100 之间
select studentNo, `StudentResult` from result
where StudentResult >= 95 and StudentResult <= 100

-- 模糊查询（区间）
select studentNo, `StudentResult` from result
where StudentResult between 95 and 100
```



> 模糊查询： 比较运算符

| 运算符      | 语法              | 描述                              |
| ----------- | ----------------- | --------------------------------- |
| is null     | a is null         | 如果操作符为null， 结果为真       |
| is not null | a is not null     |                                   |
| between     | a between b and c |                                   |
| **like**    | a like b          | sql 匹配， 如果a匹配b，则结果为真 |
| **in**      | a in(a1,a2,a3...) |                                   |





```sql
-- 模糊查询
-- 查询姓刘的同学
-- like 结合  %(代表0到任意个字符)   _(一个字符)

-- 查询姓刘的同学，名字后面只有一个字的
select `StudentNo`, `StudentName` from `student`
where StudentName like '刘_'
```



## 4.3 联表查询



```sql
-- join on 连接查询
-- where 等值查询

-- 联表查询 join   inner join   left join    right join
-- 查询参加了考试的同学（学号，姓名，科目编号，分数）
select * from student
select * from result

/* 思路
1. 分析需求，分析查询的字段来自哪些表，（连接查询）
2. 确定使用哪种连接查询？ 7 种
确定交叉点（这两个表的哪些数据是相同的）
判断的条件： 学生表 studentNo = 成绩表 studentNo
*/
select s.studentNO, studentName, SubjectNo, StudentResult
from student as s
inner join result as r
where s.studentNo = r.studnetNo


-- right join
select s.studentNO, studentName, SubjectNo, StudentResult
from student as s
right join result as r
on s.studentNo = r.studnetNo

```



| 操作       | 描述                                       |
| ---------- | ------------------------------------------ |
| inner join | 如果表种至少有一个匹配，就返回行           |
| left join  | 会从左表中返回所有的值，即使右表中没有匹配 |
| right join |                                            |



> 自连接

自己的表和自己的表连接，**核心： 一张表拆成两张一样的表即可。**

父类：

| category | categoryName |
| -------- | ------------ |
| 2        | 信息技术     |
| 3        | 软件开发     |
| 5        | 美术设计     |
|          |              |

子类：

| pid  | category | categoryName |
| ---- | -------- | ------------ |
| 3    | 4        | 数据库       |
| 2    | 8        | 办公信息     |
| 3    | 6        | web开发      |
| 5    | 7        | ps设计       |

操作：

| 父类     | 子类     |
| -------- | -------- |
| 信息技术 | 办公信息 |
| 软件开发 | web开发  |
| 软件开发 | 数据库   |
| 美术设计 | ps设计   |

```sql

-- 查询父子信息

select a.`categoryName` as '父栏目', b.`categoryName` as '子栏目'
from `category` as a, `category` as b
where a.`categoryid` = b.`pid`
```



## 4.4 分页和排序



```sql
-- 分页 limit 和 排序 order by
-- 排序： 升序： asc， 降序： desc

-- 为什么分页？ 
-- 缓解数据库压力， 给人的体验更好， 瀑布流

-- 分页， 每页只显示5条数据
select s.studentNO, studentName, SubjectNo, StudentResult
from student as s
inner join result as r
on s.studentNo = r.studnetNo
inner join `subjectNo` = sub.`SubjectNo`
where subjectName = '数据库结构-1'
order by StudentResult ASC
limit 5, 5
```



语法 ==limit（起始值）（页面大小）==



# 5. MySQL函数



``` sql
-- 数学运算
select abs(-8)
select ceiling(9.4)  -- 向上取整
select floor(9.4)  -- 向下取整
select rand()  
select sign()  -- 判断一个数的符号   -1 or 1

-- 字符串函数
select char_length()   
select concat('i', 'know', 'hard-working')
select insert('i think you can do it', 1,2,'so just do')   -- 查询， 替换
select lower('Achong')
select upper('Nuli')
select REPLACE('fafda', 'fa', 'da')
select substr('faddfa', 4,6)
select reverse('fdasdf')

-- 查询姓 周的同学， 改为 邹
select replace(studentname, '周', '邹') from student
where studentname like '周%'

-- 时间和日期函数
select current_date()
select curdate()
select now()
select localtime()
select sysdate()  

select year(now())
select month(now())

-- 系统

select system_user()
select user()
select version()

```



## 5.1 聚合函数

| 函数名称 | 描述 |
| -------- | ---- |
| count()  |      |
| sum()    |      |
| max()    |      |
| ...      |      |



```sql
-- 聚合函数

-- 都能统计表中数据
 select count(studentname) from student; -- count(指定列), 会忽略所有的null
 select count(*) from student; -- 不会忽略null值
 select count(1) from student; -- 不会忽略null值
 
 
 select sum(`StudentResult`) as 总和 from result
 select avg(`StudentResult`) as 平均分 from result
 select max(`StudentResult`) as 最高分 from result
 select min(`StudentResult`) as 最低分 from result
 
 -- 查询不同课程的平均分，最高分，最低分
 -- 核心： （根据不同的课程分组）
 select SubjectName, AVG(StudentResult) as 平均分, MAX(StudentResult) as 最高分, MIN(StudentResult) as 最低分
 from result r
 inner join `subject` sub
 on r.`SubjectNo` = sub.`SubjectNo`
 where 平均分 >= 80
 group by r.SubjectNo
 having 平均分 > 80
 
 
```





# 6. 事务

==要么都成功，要么都失败==

-------

1. SQL执行， A给B转账  A 1000 - > 200  B 200
2. SQL执行， B收到A的钱，A 800 B 400

------

将一组SQL放在一个批次中执行

> 事务原则：ACID 原则： 原子性， 一致性，隔离性， 持久性  (脏读，幻读，不可重复读)

**原子性**

要么都成功，要么都失败

**一致性**

事务前后的数据完整性要保持一致，1000

**持久性** -- 事务提交

事务一旦提交则不可逆，被持久到数据库中！

**隔离性**

事物的隔离性是多个用户并发访问数据库时，数据库为没有给用户开启的事务，不能被其他事务的操作数据干扰，事务之间要相互隔离。

> 隔离所导致的一些问题

**脏读**

指一个事务读取了另一个事务未提交的数据

**不可重复读**

在一个事务内读取表中的某一行数据，多次读取结果不同（这个不一定是错误，只是某些场合不对）

**虚读（幻读）**

指一个事务内读取到了别的事务插入的数据，导致前后读取不一致。





> 执行事务



```sql
-- mysql 是默认开启事务自动提交的
set autocommit = 0 /* 关闭 */


-- 手动处理事务
set autocommit = 0 /* 关闭 */
-- 事务开启
start transaction --标记一个事务的开始，从这个之后的sql都在同一个事务内

-- 提交： 持久化（成功！）
commit
-- 回滚： 回到原来的样子
rollback
-- 事务结束
set autocommit = 1 -- 开启自动提交
--了解
savepoint 保存点名  -- 设置一个事务的保存点
rollback to savepoint 保存点名  -- 回滚到保存点
release savepoint 保存点名 -- 撤销保存点
```





斐波那契数，通常⽤ F(n) 表⽰，形成的序列称为 斐波那契数列 。该数列由 0 和 1 开始，后
⾯的每⼀项数字都是前⾯两项数字的和。也就是：
F(0) = 0，F(1) = 1
F(n) = F(n - 1) + F(n - 2)，其中 n > 1
给你n ，请计算 F(n) 。

```
# 方法一：递归

def fibonachi(n):
    if n == 0: return 0
    if n == 1: return 1
    return fibonachi(n-1) + fibonachi(n-2)

if __name__ == '__main__':
    rnt = fibonachi(8)
    print(rnt)
    
    
# 方法二： 动态规划

def fib(n):
    dp = [0]*(n+1)
    if n <= 1: return n
    dp[0] = 0
    dp[1] = 1
    for i in range(2, n+1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]

start = time.time()
rnt = fib(100)
end = time.time()
print(rnt, 'the total time is:', (end - start))

    
```

动态规划相较于递归时间复杂度小，在n值较大的时候运行较快。



假设你正在爬楼梯。需要 n 阶你才能到达楼顶。
每次你可以爬 1 或 2 个台阶。你有多少种不同的⽅法可以爬到楼顶呢？
注意：给定 n 是⼀个正整数。
⽰例 1：

- 输⼊： 2
- 输出： 2
- 解释： 有两种⽅法可以爬到楼顶。
  - 1 阶 + 1 阶
  - 2 阶

示例 2：

- 输⼊： 3
- 输出： 3
- 解释： 有三种⽅法可以爬到楼顶。
  1 阶 + 1 阶 + 1 阶
  1 阶 + 2 阶
  2 阶 + 1 阶

思路

``` python
# 爬楼梯

def climbstairs(n):
    dp = [0]*3
    dp[1] = 1
    dp[2] = 2
    for i in range(3, n+1):
        sum = dp[1]+ dp[2]
        dp[1] = dp[2]
        dp[2] = sum
    
    
    return dp[2]

start = time.time()
rnt = climbstairs(5)
end = time.time()
print(rnt, 'the total time is:', (end - start))
```



- ```
  数组的每个下标作为⼀个阶梯，第 i 个阶梯对应着⼀个⾮负数的体⼒花费值 cost[i]（下标从0 开始）。
  每当你爬上⼀个阶梯你都要花费对应的体⼒值，⼀旦⽀付了相应的体⼒值，你就可以选择向上爬⼀个阶梯或者爬两个阶梯。
  请你找出达到楼层顶部的最低花费。在开始时，你可以选择从下标为 0 或 1 的元素作为初始阶梯  
  ```

  



- 输⼊：cost = [1, 100, 1, 1, 1, 100, 1, 1, 100, 1]
- 输出：6
- 解释：最低花费⽅式是从 cost[0] 开始，逐个经过那些 1 ，跳过 cost[3] ，⼀共花费 6  