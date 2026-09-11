# Email_Intent_Classifier

# Step 1 :  
make virtual environment in Vs code \n
python -m venv .venv

# Step 2 :  
activate it \n
.venv\Scripts\activate

# Step 3 :  
install required libraries \n
pip install torch transformers datasets scikit-learn pandas numpy accelerate streamlit matplotlib seaborn

# Step 4 :  
make src folder which contains train.py and predict.py file 

# Step 5 :  
run the train.py in terminal \n
python src\train.py

# Step 6 :  
runt the predict.py in terminal \n 
python src\predict.py
