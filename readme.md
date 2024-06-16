# Руководство пользователя:

Для запуска телеграм бота необходимо:

1) Скачать веса (файлы .pt и .pth) по ссылке https://drive.google.com/drive/folders/11iYT6JVHSNIx_IkBtgVRZFhk4XJDk9Y_?usp=sharing и положить их в папку checkpoints

2) Ввести следующие команды:
```
  git clone https://github.com/egorovivannn/mysolutionforhack.git
  
  python3.11 -m venv myvenv
  
  source activate myvenv/bin/activate
  
  pip install -r requirements.txt
  
  python python src/main.py

```


Для запуска обучения нужно:
1) Запустить ноутбук training/preprocessing.ipynb , прописать в нем пути к датасету и запустить до конца, сгенерировав df.csv
2) Запустить ноутбук training/training.ipynb , прописав в нем пути к df.csv и датасету. Обучить таким образом классификатор.
3) Запустить ноутбук training/training_yola.ipynb , прописав в нем пути к датасету.
4) Запустить ноутбук training/inference.ipynb , для инференса