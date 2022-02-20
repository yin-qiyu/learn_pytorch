[toc] 

# pytorch

jupyter notebook --allow-root

ssh -L8888:localhost:8888 root@47.99.64.50
Nyw666!!



git失效时候

git config --global http.sslVerify false 



```
pip install mediapipe opencv-python numpy matplotlib tqdm jupyter notebook -i https://pypi.tuna.tsinghua.edu.cn/simple
```



查看是否能用cuda



torch.cuda.is_available



## conda

conda atvivate pytorch

jupyter pytorch



### 安装卸载库

启动环境：conda activate (name)

卸载库：pip uninstall opencv-python



 tensorboard --logdir=logs --port=6007

tensorboard --logdir=logs  

设置端口避免冲突



![image-20220101134011080](pytorch.assets/image-20220101134011080.png)

将jpg格式转换numpy类型





<img src="pytorch.assets/image-20220101162752735.png" alt="image-20220101162752735" style="zoom:50%;" />







compose的用法

<img src="pytorch.assets/image-20220102105901763.png" alt="image-20220102105901763" style="zoom:50%;" />



![image-20220102112825242](pytorch.assets/image-20220102112825242.png)

![image-20220102112834870](pytorch.assets/image-20220102112834870.png)





![image-20220102181432669](pytorch.assets/image-20220102181432669.png)

随机抓取



卷积

<img src="pytorch.assets/image-20220102230154897.png" alt="image-20220102230154897" style="zoom:50%;" />







![image-20220103184801766](pytorch.assets/image-20220103184801766.png)

![image-20220103184855153](pytorch.assets/image-20220103184855153.png)





用gpu跑模型

方式1

![image-20220108231903953](pytorch.assets/image-20220108231903953.png)





![image-20220109120524740](pytorch.assets/image-20220109120524740.png)







方式2



![image-20220109122910190](pytorch.assets/image-20220109122910190.png)





![image-20220109134927656](pytorch.assets/image-20220109134927656.png)





issue：gpu跑的模型在cpu运行

![image-20220109141105335](pytorch.assets/image-20220109141105335.png)







![image-20220113212130257](pytorch.assets/image-20220113212130257.png)

![image-20220109163134219](pytorch.assets/image-20220109163134219.png)