import json

def loadFont():
    f = open("cars.json", encoding='utf-8')  #//设置以utf-8解码模式读取文件，encoding参数必须设置，否则默认以gbk模式读取文件，当文件中包含中文时，会报错
    setting = json.load(f)
    return setting

t = loadFont()
data=[]
y=[]
for i in range(len(t)):
    if(t[i]['Horsepower']==None or t[i]['Miles_per_Gallon']==None or t[i]['Acceleration']==None ):
        continue
    data_one=[t[i]['Horsepower'],t[i]['Miles_per_Gallon'],t[i]['Acceleration']]
    data.append(data_one)
    y.append(t[i]['Origin'])

print(y)
