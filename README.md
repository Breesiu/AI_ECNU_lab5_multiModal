
执行你代码所需要的环境(可见requirements)
```
numpy 
torch==2.0.0
torchmetrics
torchvision
tqdm 
SentencePiece
transformers==4.29.2

```

代码文件结构

```
lab5
├─ README.md
├─ __MACOSX
│  └─ data
├─ __init__.py
├─ __pycache__
│  ├─ CustomDataset.cpython-310.pyc
│  ├─ dataset.cpython-310.pyc
│  ├─ model.cpython-310.pyc
│  ├─ trainer.cpython-310.pyc
│  └─ utils.cpython-310.pyc
├─ checkpoints
│  ├─ finetuned.pt
├─ data
│  ├─ 1.jpg
│  ├─ 1.txt
│  ├─ 10.jpg
│  ├─ 10.txt
    ...
├─ output.txt 
├─ dataset.py
├─ main.py
├─ model.py
├─ requirements.txt
├─ results
├─ run.py
├─ run.sh
├─ test.ipynb
├─ test2.ipynb
├─ test3.ipynb
├─ test_without_label.txt
├─ train.txt
├─ trainer.py
└─ utils.py

```
output.txt 为测试输出文件

执行你代码的完整流程: 直接python main.py即可, 训练， 验证， 测试可见utils.py, 通过arqparse调整参数
若分别训练， 验证， 测试， 可使用如下命令
```
训练: python main.py --do_finetune
验证: python main.py --do_eval
测试: python main.py --do_test
若需要调整超参数， 可直接在trainer.py中TrainerConfig和utils.py中调整即可
```

代码未参考其他库， 主要参考hugging face中transformers函数调用