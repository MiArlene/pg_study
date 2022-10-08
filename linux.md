
# 1. Linux入门指南

## 1.1 系统目录结构

> ls /

![img](https://www.runoob.com/wp-content/uploads/2014/06/4_20.png)

![img](https://www.runoob.com/wp-content/uploads/2014/06/d0c50-linux2bfile2bsystem2bhierarchy.jpg)

- /bin :  存放最经常使用的命令
- /boot ：启动Linux的核心文件，包括一些连接文件以及镜像文件
- /dev： 存放LInux的外部设备，在Linux中访问设备的方式和访问文件的方式是相同的
- /etc ： 存放所有系统管理所需要的配置文件和子目录
- /home ： 用户的主目录， 在Linux中，每个用户都有自己的目录， 一般目录名以用户的账号命名的
- /lib： 存放系统最基本的动态连接共享库，类似于windows中的dll文件，几乎所有的应用程序都需要用到这些共享库
- /lost + found： 一般为空，当系统非法关机后，会存放一些文件
- /media:  Linux会把识别的设备挂载在这个目录下
- /root： 系统管理员，超级权限者的用户主目录
- /sbin ： 系统管理员使用的系统管理程序
- /usr： unix shared resource， 用户的许多应用程序和文件都放在这个目录下，类似于windows下的program files 目录
- /proc: 虚拟文件系统，存储当前内核运行状态的一系列特殊文件，文件内容不在硬盘上而在内存里，是系统内存的映射。









```shell
ls  	# list files
cd  	# change directory
pwd 	# print work directory
mkdir 	# make directory
rmdir 	# remove directory
cp	 	# copy file
rm 		# remove
mv 		# move file

# 文件基本属性
chown 	# change owner 修改所属用户与组
chmod 	# change mode 修改用户的权限

```





mkdir test1

ls

cd .. 

mkdir -p test2/test3/test4





rmdir test1     

==p== : parents

rmdir -p test2/test3/test4

rmdir 仅能删除空的目录，如果下面存在文件，可以使用递归删除 -p

cd /home

> cp(复制文件或者目录)

cd /home 

cp install.sh kuangstudy

ls

cd .. 

如果文件重复，选择覆盖（y）或者放弃（n）



> rm (移除文件或者目录)

-f 忽略不存在的文件，不会出现警告，强制删除

-r 递归删除目录！

-i 互动，删除询问是否删除

```
rm - rf / # 系统中所有文件都删除，删库跑路就是这么操作！
```

> mv 移动文件或者目录

-f 强制   force

-u 只替代已经更新的文件







> 看懂文件属性 root 权限最高的

实例中，boot文件的第一个属性用“d“ 表示

- 分为三个等级， root     	group    	other users
- 属主



![363003_1227493859FdXT](https://www.runoob.com/wp-content/uploads/2014/06/363003_1227493859FdXT.png)



> 修改文件属性

1. chgrp ：更改文件属性

   ```
   chgrp [-R] 属性名 文件名
   ```

   

2. ==chmod ：更改文件的9个属性==

   ```bash
   chmod [-R] xyz 文件或目录
   ```

   ```
   r: 4 			w: 2 			x: 1
   
   chmod 777 文件对所有用户可读可写可执行
   ```

   10个字母， 第一个类型 www  l：link

## 1.2 文件内容查看

Linux系统中使用以下命令来查看文件的内容：

- cat 由第一行开始显示文件内容
- tac 从最后一行开始显示，可以看出tac是cat的反转字符串
- nl 显示的时候，顺便输出行号！
- more 一页一页地显示文件内容 (空格代表翻页，只能往下翻)
- ==less 与more类似， 可以往前翻页，推出 q命令，查找字符串 /要查询的字符向下查询， ？向上查询， n继续搜寻下一个，N上一个== 
- head 只看头几行
- tail 只看最后几行

你可以使用*man [命令]* 查看各个命令的使用文档 ， 如 ： man cp。  -- 》 copy file

网络配置目录： ==`cd/etc/sysconfig/network-scripts`==  CentOs 7

ifconfig: 命令查看网络配置





> 拓展： Linux 链接概念

Linux 链接： 硬链接， 软链接！

硬链接： A---B， A和B指向同一个文件！ 允许同一个文件拥有多个路径！以防误删。

软链接：类似Win下的快捷方式， 删除了源文件，快捷方式也访问不了了！

ln ： link files

touch 命令创建文件

echo 输入字符串

# 2. Linux基本指令



## 2.1 Vim 编辑器

vim 通过一些插件可以实现和IDE一样的功能！

vim是从vi发展出来的文本编辑器。代码补全、编译及错误跳转 ==（查看内容、编辑内容、保存内容！）==

所有的Unix Like 系统都会内建 vi 文书编辑器

vim是一个程序开发工具而不是文字处理软件

![img](https://www.runoob.com/wp-content/uploads/2015/10/vi-vim-cheat-sheet-sch.gif)

> 三种使用模式

基本的vi/vim共分为三种模式，**命令模式（command mode），输入模式（insert mode）和底线命令模式（Last line）**

### 命令模式：

用户刚启动vi/vim，就进入命令模式。

此状态下敲击键盘动作会被Vim识别为命令，而非输入字符。比如我们此时按下i，并不会输入一个字符，i被当作了一个命令。

以下是常用的几个命令：

- **i** 切换到输入模式，以输入字符。
- **x** 删除当前光标所在处的字符。
- **:** 切换到底线命令模式，以在最底一行输入命令。如果是编辑模式，需要先退出编辑模式，esc。

若想要编辑文本：启动Vim，进入了命令模式，按下i，切换到输入模式。

命令模式只有一些最基本的命令，因此仍要依靠底线命令模式输入更多命令。

### 输入模式

在命令模式下按下i就进入了输入模式。

在输入模式中，可以使用以下按键：

- **字符按键以及Shift组合**，输入字符
- **ENTER**，回车键，换行
- **BACK SPACE**，退格键，删除光标前一个字符
- **DEL**，删除键，删除光标后一个字符
- **方向键**，在文本中移动光标
- **HOME**/**END**，移动光标到行首/行尾
- **Page Up**/**Page Down**，上/下翻页
- **Insert**，切换光标为输入/替换模式，光标将变成竖线/下划线
- **ESC**，退出输入模式，切换到命令模式

### 底线命令模式

在命令模式下按下:（英文冒号）就进入了底线命令模式。

底线命令模式可以输入单个或多个字符的命令，可用的命令非常多。

在底线命令模式中，基本的命令有（已经省略了冒号）：

- q 退出程序
- w 保存文件

按ESC键可随时退出底线命令模式。

![img](https://www.runoob.com/wp-content/uploads/2014/07/vim-vi-workmodel.png)

> 完整的演示说明

新建或者编辑文件，按 i 进入编辑模式，编写内容，若要退出编辑模式，按 esc





## 2.2 账号管理

一般不是root账户！

- 用户账号的添加、删除和修改
- 用户口令的管理
- 用户账号的管理

> useradd 命令 添加用户

useradd - 选项 用户名

参数说明：

- 选项:

  - -c comment 指定一段注释性描述。
  - -d 目录 指定用户主目录，如果此目录不存在，则同时使用-m选项，可以创建主目录。
  - -g 用户组 指定用户所属的用户组。
  - -G 用户组，用户组 指定用户所属的附加组。
  - -s Shell文件 指定用户的登录Shell。
  - -u 用户号 指定用户的用户号，如果同时有-o选项，则可以重复使用其他用户的标识号。

- 用户名:

  指定新账号的登录名。

-m ： 自动创建这个用户的主目录 /home/username



==本质： LInux中一切皆文件，这里的添加用户就是往某一个文件中写入用户的信息！  /etc/passwd==

> 删除用户 userdel

userdel -r 

> 修改用户 usermod

修改用户 usermod 对应修改的内容

> 切换用户！

> 锁定账户！

用户管理的一项重要内容是用户口令的管理。用户账号刚创建时没有口令，但是被系统锁定，无法使用，必须为其指定口令后才可以使用，即使是指定空口令

```
passwd -l 用户名  # 锁定之后这个用户就不能登录了！
passwd -d 用户名  # 将密码清空， 没有密码也不能登录！
```



## 2.3 用户组管理

（开发、测试、运维、root）， 不同的Linux系统对用户组的规定有所不同。

用户组的管理涉及用户组的添加、删除和修改。==组的增加、删除和修改实际上是对/etc/group文件的更新==

> 创建一个用户组  groupadd  -组名

> 删除用户组  groupdel -组名

创建完用户组可以得到一个组的id，这个id是可以指定的！ 如 ==-g 250==， 否则就自增

> 修改用户组的权限信息和名字     groupmod -g -n         n: name

> 用户如果要切换用户组

```bash
# 登录当前用户
$ newgrp root
```

```bash
passwd 选项 用户名
```

- 可使用的选项：
  - -l 锁定口令，即禁用账号。
  - -u 口令解锁。
  - -d 使账号无口令。
  - -f 强迫用户下次登录时修改口令。

登录口令： 把真正加密的密码放在了etc/shadow 目录中，但事实加密的，保证我们的密码安全性！



## 2.4 磁盘管理

> df (列出文件系统的整体使用量) du（检查磁盘空间使用量）

- df : disk free  列出文件系统的整体磁盘使用量
- du ：disk used 检查磁盘空间使用量
- fdisk： 用于磁盘分区







## 2.5 进程管理



==Linux中一切皆文件（文件： 读写执行（查看，创建，移动，复制，删除），权限（用户，用户组）， 系统（磁盘，进程））==

> 进程 和 线程

在Linux 中， 每个程序都有自己的进程，都有专属的id号

每一个进程，都有一个父进程

进程可以有两个存在方式， 前台 后台

一般的服务都是在后台运行的，基本程序都是在前台运行的。

> 命令

**ps** 查看当前系统中正在执行的各种进程信息！  process status

ps -xx ：

-  -a 显示当前终端运行的所有进程信息
-  -u 以用户的信息显示进程
-  -x 显示后来运行进程的参数！



- grep： global regular expression print

```
# ps -aux 查看所有的进程
ps -aux | grep mysql
ps -aux | grep redis

# | 在Linux这个叫管道符 A|B
# grep 查看所有符合要求的字符串！
```



 **ps -ef ： 可以查看父进程的信息**

```
# 进程树：
pstree -pu
	-p 显示父id
	-u 显示用户组
```



结束进程： 

```
kill -9 进程的id
```

# 3. 环境安装

## JDK 安装

安装软件一般有三种方式： 

- rpm （jdk， 在线发布一个SpringBoot项目）
- 解压缩（tomcat， 启动并通过外网访问，发布网站）
- yum在线安装 （docker： 直接安装运行跑起来docker 就可以！）

```
-ivh 
- i （install）
- v （view）
-h （hour 方便自己记忆 显示安装进度）
```







```
# tar -zxvf filename.tar.gz

z ：表示 tar 包是被 gzip 压缩过的 (后缀是.tar.gz)，所以解压时需要用 gunzip 解压 (.tar不需要)

x ：表示 从 tar 包中把文件提取出来

v ：表示 显示打包过程详细信息

f  ：指定被处理的文件是什么
```

## Docker 安装

yum在线安装。需要联网。

```
yum -y install 包名  # yum install 安装命令 -y 所有的提示都是yes
```



# 4. Vmware 

## 快照

保留当前系统信息为快照，随时可以恢复，以防未来系统玩坏了，归档。

平时，我们每配置一个东西就可以拍摄一个快照，保留信息！

