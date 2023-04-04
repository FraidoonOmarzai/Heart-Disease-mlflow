<h1 align=center> Heart Disease Project `Mlflow`</h1>

**Note:** Implementing DVC along mlflow


*******************************************************************************************************************


## Steps

* Create an environment and acitvate it
```bash
conda create -n heartP python=3.7 -y
conda activate heartP
```

* create req file and run it
```bash
touch requirements.txt
pip install -r requirements.txt
```

* create template.py and specifiy all the files and directory that we are gonna create, finally run it 

```bash
touch template.py
python template.py
```

* `git init` and push the chenges to github

* get the dataset from [kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) and added to `data/` path

* we will use dvc to track our dataset

```bash
dvc init
```

* Add data into dvc for tracking
```bash
dvc add data/*.csv
```

* Add remote storage
```bash
dvc remote add -d storage gdrive://<DRIVE ID>
```

* Push the data to the remote storage
```bash
dvc push
```

* push changes to github
```bash
git add . && git commit -m ""
```

* create src/get_data.py and src/load_data.py

```bash
touch src/get_data.py src/load_data.py
```

* add parameters to params.yaml

* add code to get_data.py and load_data.py

* add stage in dvc.yaml and run the below commands

```bash
dvc repro
```


* create a split_data file, add a stage in dvc.yaml and run `dvc repro`
```bash
touch src/split_data.py
```

* visualize dvc pipeline
```bash
dvc dag
```

* create train_and_eval file, adding code to param.yaml, add a stage in dvc.yaml and run dvc repro
```bash
touch src/train_and_eval.py
```

* make dir
```bash
mkdir artifacts
```

* mlflow server command - (run in bash2)
```bash
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./artifacts \
    --host 0.0.0.0 -p 1234
```


* create log_production_model.py and add code to dvc.yaml
```bash
touch src/log_production_model.py
```


