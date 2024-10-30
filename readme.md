# 说明
一种基于强化学习的密集速度受限的多无人机分布式碰撞避免算法
## 效果展示
多机不同速度汇聚场景
![多机不同速度汇聚](pic\img_rl_diff_speed.gif)

多机随机场景
![多机随机场景](pic\img_rl_random.gif)

非合作无人机场景
![非合作无人机场景](pic\img_rl_circle_cross.gif)
## 1.安装依赖库
依赖库详情在requestment文件，在终端输入以下指令安装依赖。
>pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/

安装paddle框架2.3.2版本
>pip install paddlepaddle-gpu==2.3.2.post116

或者
>python -m pip install paddlepaddle-gpu==2.3.2.post116 -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html

## 2.文件详情

|文件|详情|
|------|----------|
|sac.py|SAC算法文件|
|sac_model.py|神经网络模型程序|
|sac_agent.py|智能体相关程序|
|train.py|进行训练程序|
|plot2DEnv.py|交互环境程序|
|envplot.py|渲染轨迹画面程序|
|Aircraft_math_model.py|无人机数学模型|
|inference_model|训练好的网络参数|
|ActorModelEval.py|使用训练好的参数进行实验程序|
|run_test.py|简单测试例程|


## 3.训练
在train,sac_agent,plot2DEnv程序中设置好无人机数量，运行train.py程序进行训练，训练好的参数保存，以便无人机使用。

训练会生成log文件，可以使用tensorboard查看数据曲线

>tensorboard --logdir=train_log\train

通过浏览器访问 TensorBoard 的地址（通常是 http://localhost:6006/）

## 4.实验
运行ActorModelEval.py程序即可观察训练好的网络参数的避碰效果。