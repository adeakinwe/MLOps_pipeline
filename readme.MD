# virtual env
python3 -m venv mlflow-env
source mlflow-env/bin/activate
pip install --upgrade pip
pip install mlflow

# virtual env 
source mlflow-env/bin/activate
pip install ipykernel
python -m ipykernel install --user --name=mlflow-env --display-name "Experiment-tracking (mlflow-env)"

#backend url
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5001

# virtual env (access to global packages)
python3 -m venv --system-site-packages mlflow-env
source mlflow-env/bin/activate
pip install --upgrade pip
pip install mlflow

# python 3.11 virtual environment
# Navigate to your project directory
cd ~/Projects/MLops_pipeline/

# Create a new virtual environment using Python 3.11
/usr/local/bin/python3.11 -m venv mlflow-311venv
source mlflow-311venv/bin/activate
