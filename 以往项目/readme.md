安装Anaconda，在Anaconda Prompt中输入

`conda create -n PKU-Control python=3.6`

安装完成后，在IDE中添加配置，然后在本项目文件夹下

`pip install -r requirements.txt`

为简化操作（避免CUDA安装或跨平台显卡），我们选择安装`tensorflow`，而非`tensorflow-gpu`

待安装完成后，运行`main.py`即可