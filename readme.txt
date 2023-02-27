1.在data\下放train.json
2.直接运行run_albert.ipynb就可以训练
3.训练得到的模型会存到experiments\s_model
data_precess：处理标注数据到模型训练需要的格式，并且去除一些会报错的token train.json---train.npz
    有两种可选方法，目前用的方法慢一点但是可以处理掉脏数据；注释掉的部分会快一点，但是如果有脏数据报错
data_loader：生成训练集和验证集
