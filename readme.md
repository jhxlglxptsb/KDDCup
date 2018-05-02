### 训练模型
需要文件beijing_aq.json和london_aq.json, 建立文件夹files，执行如下命令
```
python process.py
python model.py
```
处理过的文件和训练的模型参数存储在files文件夹下面

### 提交模型答案
在files文件夹下新建文件夹test
首先需要运行getdata.py获取某一时间段的数据，为了避免报错，我们获取4月30日0时开始到当前时间的数据
```
python getdata.py 2018-04-30-0 [current time]
python process_test.py [beijing json file name] [london json file name]
python test.py
python submit.py beijing_output.json london_output.json
```
- process_test.py的参数[beijing json file name]和[london json file name]是getdata.py存储数据的文件名
- process_test.py处理的数据存在./files/test文件夹下
- test.py生成文件beijing_output.json和london_output.json
- submit.py 生成result文件进行提交
