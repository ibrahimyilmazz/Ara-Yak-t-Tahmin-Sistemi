import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


from scipy import stats
from scipy.stats import norm, skew   # Istatistik işlemleri için; verinin feature engineering işlemleri için

from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, GridSearchCV   # Veriyi train ve test olarak bölebilmek için
from sklearn.metrics import mean_squared_error
from sklearn.base import clone # tüm modelleri birleştirirken kullandık


# XGBoost
import xgboost as xgb

# warning
import warnings
warnings.filterwarnings('ignore')


column_name = ["MPG", "Cylinders", "Displancement", "Horsepower", "Weight", "Acceleration", "Model Year", "Origin"] # araç isimleri ile ilgilenmiyoruz
data = pd.read_csv("auto-mpg.data", names = column_name, na_values= "?", comment="\t", sep = " ", skipinitialspace = True)

print(data.head())
print("Data Shape : ", data.shape)
data.info()

describe = data.describe() 


#missing value - eksik veriler
print(data.isna().sum())  

#horsepower içindeki eksik verileri ortalama değerler ile doldurma
data["Horsepower"] = data["Horsepower"].fillna(data["Horsepower"].mean())

#missing value - eksik veriler - tekrar kontrol için
print(data.isna().sum()) 

# Dağılıma bakmak için; NaN değer olunca göstermiyor.
sns.distplot(data.Horsepower) # Kuyruk sağa doğru



#verinin incelenmesi - EDA

#nümerik veriler için corelasyon matrisinin oluşturulması
corr_matrix = data.corr()
sns.clustermap(corr_matrix, annot = True, fmt = ".2f")
plt.title("Correlation btw features")
plt.show()

#0.75 filtresinin eklenmesi
treshold = 0.75
filtre = np.abs(corr_matrix["MPG"]) > treshold
corr_features = corr_matrix.columns[filtre].tolist()
sns.clustermap(data[corr_features].corr(), annot = True, fmt = ".2f")
plt.title("Correlation btw features")
plt.show()

"""
multicollinearlity (eş düzlem var)
yani benzer özellikte değerler, bunlar da modelimizi yaniltacaktir
"""


sns.pairplot(data, diag_kind = "kde", markers = "+")
plt.show()

"""
cylinders ve origin değerleri kategorikal olabilir. 
İlerleyen aşamalarda kategorik değişkenlere çevireceğiz (Özellik Mühendisliği)
"""



plt.figure()
sns.countplot(data["Cylinders"])
print(data["Cylinders"].value_counts())

plt.figure()
sns.countplot(data["Origin"])
print(data["Origin"].value_counts())



#box
for c in data.columns:
    plt.figure()
    sns.boxplot(x = c, data = data, orient = "v") 

"""
IQR = Q3 - Q1   |||    Q3 + 1.5 x IQR = Üst Sınır     
                |||    Q1 - 1.5 x IQR = Alt Sınır

Acceleration ve Horspower verilerinde aykırı değerlerin (outliers) fazla olduğunu görüyoruz. Onları temizleyeceğiz

"""



#Outlier
thr = 2   # Genelde alt sınır olarak 1.5 değeri alınır; veri az old. icin 2 aldim

horspower_desc = describe["Horsepower"]
q3_hp = horspower_desc[6]
q1_hp = horspower_desc[4]
IQR_hp = q3_hp - q1_hp
top_limit_hp = q3_hp + thr*IQR_hp
bottom_limit_hp = q1_hp - thr * IQR_hp
filter_hp_bottom = bottom_limit_hp < data["Horsepower"]
filter_hp_top = data["Horsepower"] < top_limit_hp
filter_hp = filter_hp_bottom & filter_hp_top

data = data[filter_hp]      # 1 aykırı değer silindi


acceleration_desc = describe["Acceleration"]
q3_acc = acceleration_desc[6]
q1_acc = acceleration_desc[4]
IQR_acc = q3_acc - q1_acc
top_limit_acc = q3_acc + thr*IQR_acc
bottom_limit_acc = q1_acc - thr * IQR_acc
filter_acc_bottom = bottom_limit_acc < data["Acceleration"]
filter_acc_top = data["Acceleration"] < top_limit_acc
filter_acc = filter_acc_bottom & filter_acc_top

data = data[filter_acc]     # 2 aykırı değer silindi



"""
Verinin 1. momenti ortalamayı
Verinin 2. momenti standart sapmayı 
Verinin 3. momenti çarpıklığı (skewed) verir

çarpıklık >  1 ise pozitif çarpıklık (kuyruk saga dogru)
çarpıklık < -1 ise negatif çarpıklık (kuyruk sola dogru)
"""
## Feature Engineering
# Skewness

# mpg dependent variable
sns.distplot(data.MPG, fit = norm)

(mu, sigma) = norm.fit(data["MPG"])
print("mu: {}, sigma = {}".format(mu, sigma))
# mu: 23.472405063291134, sigma = 7.756119546409932
# mu = ortalama           sigma = standart sapma

#qq plot
plt.figure()
stats.probplot(data["MPG"], plot = plt)
plt.show()          # veri kırmızı çizgi üzerinde olsaydı normal dağılımda olmuş olacaktı


data["MPG"] = np.log1p(data["MPG"])     # log1p kullandık
plt.figure()
sns.distplot(data.MPG, fit = norm)      # Çarpıklık azaldı

(mu, sigma) = norm.fit(data["MPG"])
print("mu: {}, sigma = {}".format(mu, sigma))   # yeni mu ve sigma değerleri daha iyi
# mu: 3.146474056830183, sigma = 0.3227569103044823

plt.figure()
stats.probplot(data["MPG"], plot = plt)
plt.show()          # yeni normallik değeri daha iyi


# feature - independent variable
skewed_feats = data.apply(lambda x: skew(x.dropna())).sort_values(ascending = False) # ascending = artan; azalan bir sıralama yaptık
skewness = pd.DataFrame(skewed_feats, columns = ["skewed"]) 
# Horspower da pozitif sekness var çünkü değeri 1 den büyük | 1.06 değeri 1'e çok yakın olduğu için düzeltmedik



# one hot encoding
"""
one hot encoding uygulayabilmek için önce kategorik hale getirmek gerekiyor
"""
data["Cylinders"] = data["Cylinders"].astype(str) 
data["Origin"] = data["Origin"].astype(str)

data = pd.get_dummies(data)



# Split 
x = data.drop(["MPG"], axis = 1)    # independent variable
y = data.MPG                        # dependent variable

test_size = 0.9 #veriyi zorlamak için az veri ile eğitim
X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size = test_size, random_state = 42)


# Standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



## Regression Models

# Lineer Regression
lr = LinearRegression()
lr.fit(X_train, Y_train)
print("LR Coef :", lr.coef_)
y_predicted_dummy = lr.predict(X_test)
mse = mean_squared_error(Y_test, y_predicted_dummy)
print("linear Regression MSE :", mse) # linear Regression MSE : 0.020632204780133005




# Ridge Regression
ridge = Ridge(random_state = 42, max_iter = 10000)

alphas = np.logspace(-4,-0.5,30)

tuned_parameters = [{'alpha': alphas}]
n_folds = 5

clf = GridSearchCV(ridge, tuned_parameters, cv = n_folds, scoring = "neg_mean_squared_error", refit = True)
clf.fit(X_train, Y_train)
scores = clf.cv_results_["mean_test_score"]
scores_std = clf.cv_results_["std_test_score"]

print("Ridge Coef: ", clf.best_estimator_.coef_)

ridge = clf.best_estimator_

print("Ridge Best Estimator: ", ridge)

y_predicted_dummy = clf.predict(X_test)
mse = mean_squared_error(Y_test, y_predicted_dummy) # Ridge MSE:  0.019725338010801185
print("Ridge MSE: ", mse)
print("--------------------------")

plt.figure()
plt.semilogx(alphas,scores)
plt.xlabel("alpha")
plt.ylabel("score")
plt.title("Ridge")



# Lasso Regression
lasso = Lasso(random_state = 42, max_iter = 10000)
alphas = np.logspace(-4,-0.5,30)

tuned_parameters = [{'alpha': alphas}]
n_folds = 5

clf = GridSearchCV(lasso, tuned_parameters, cv = n_folds, scoring = "neg_mean_squared_error", refit = True)
clf.fit(X_train, Y_train)
scores = clf.cv_results_["mean_test_score"]
scores_std = clf.cv_results_["std_test_score"]

print("Lasso Coef: ", clf.best_estimator_.coef_)
lasso = clf.best_estimator_
print("Lasso Best Estimator: ", lasso)

y_predicted_dummy = clf.predict(X_test)
mse = mean_squared_error(Y_test, y_predicted_dummy) # Lasso MSE:  0.01752159477082249
print("Lasso MSE: ", mse)
print("--------------------------")

plt.figure()
plt.semilogx(alphas,scores)
plt.xlabel("alpha")
plt.ylabel("score")
plt.title("Lasso")



# ElasticNet
parametersGrid = {"alpha": alphas,
                  "l1_ratio": np.arange(0.0, 1.0, 0.05)}
eNet = ElasticNet(random_state = 42, max_iter = 10000)
clf = GridSearchCV(eNet, tuned_parameters, cv = n_folds, scoring = "neg_mean_squared_error", refit = True)
clf.fit(X_train, Y_train)

print("ElasticNet Coef: ", clf.best_estimator_.coef_)
eNet = clf.best_estimator_
print("ElasticNet Best Estimator: ", eNet)

y_predicted_dummy = clf.predict(X_test)
mse = mean_squared_error(Y_test, y_predicted_dummy) # ElasticNet MSE:  0.017190543094566392
print("ElasticNet MSE: ", mse)



# %%

# Standardization
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



## Regression Models

# Lineer Regression
lr = LinearRegression()
lr.fit(X_train, Y_train)
print("LR Coef :", lr.coef_)
y_predicted_dummy = lr.predict(X_test)
mse = mean_squared_error(Y_test, y_predicted_dummy)
print("linear Regression MSE :", mse) # linear Regression MSE : 0.020487058280775586



# Ridge Regression
ridge = Ridge(random_state = 42, max_iter = 10000)

alphas = np.logspace(-4,-0.5,30)

tuned_parameters = [{'alpha': alphas}]
n_folds = 5

clf = GridSearchCV(ridge, tuned_parameters, cv = n_folds, scoring = "neg_mean_squared_error", refit = True)
clf.fit(X_train, Y_train)
scores = clf.cv_results_["mean_test_score"]
scores_std = clf.cv_results_["std_test_score"]

print("Ridge Coef: ", clf.best_estimator_.coef_)

ridge = clf.best_estimator_

print("Ridge Best Estimator: ", ridge)

y_predicted_dummy = clf.predict(X_test)
mse = mean_squared_error(Y_test, y_predicted_dummy) # Ridge MSE:  0.01937186998245171
print("Ridge MSE: ", mse)
print("--------------------------")

plt.figure()
plt.semilogx(alphas,scores)
plt.xlabel("alpha")
plt.ylabel("score")
plt.title("Ridge")



# Lasso Regression
lasso = Lasso(random_state = 42, max_iter = 10000)
alphas = np.logspace(-4,-0.5,30)

tuned_parameters = [{'alpha': alphas}]
n_folds = 5

clf = GridSearchCV(lasso, tuned_parameters, cv = n_folds, scoring = "neg_mean_squared_error", refit = True)
clf.fit(X_train, Y_train)
scores = clf.cv_results_["mean_test_score"]
scores_std = clf.cv_results_["std_test_score"]

print("Lasso Coef: ", clf.best_estimator_.coef_)
lasso = clf.best_estimator_
print("Lasso Best Estimator: ", lasso)

y_predicted_dummy = clf.predict(X_test)
mse = mean_squared_error(Y_test, y_predicted_dummy) # Lasso MSE:  0.018707192302889232
print("Lasso MSE: ", mse)
print("--------------------------")

plt.figure()
plt.semilogx(alphas,scores)
plt.xlabel("alpha")
plt.ylabel("score")
plt.title("Lasso")



# ElasticNet
parametersGrid = {"alpha": alphas,
                  "l1_ratio": np.arange(0.0, 1.0, 0.05)}
eNet = ElasticNet(random_state = 42, max_iter = 10000)
clf = GridSearchCV(eNet, tuned_parameters, cv = n_folds, scoring = "neg_mean_squared_error", refit = True)
clf.fit(X_train, Y_train)

print("ElasticNet Coef: ", clf.best_estimator_.coef_)
eNet = clf.best_estimator_
print("ElasticNet Best Estimator: ", eNet)

y_predicted_dummy = clf.predict(X_test)
mse = mean_squared_error(Y_test, y_predicted_dummy) # ElasticNet MSE:  0.018361024469839903
print("ElasticNet MSE: ", mse)


"""
StandartScaler
    linear Regression MSE : 0.020632204780133005
    Ridge MSE:  0.019725338010801185
    Lasso MSE:  0.01752159477082249
    ElasticNet MSE:  0.017190543094566392

RobustScaler
    linear Regression MSE : 0.020487058280775586
    Ridge MSE:  0.01937186998245171
    Lasso MSE:  0.018707192302889232
    ElasticNet MSE:  0.018361024469839903
"""


# %% XGBoost
parametersGrid = {'nthread':[4], #when use hyperthread, xgboost may become slower
              'objective':['reg:linear'],
              'learning_rate': [.03, 0.05, .07], 
              'max_depth': [5, 6, 7],
              'min_child_weight': [4],
              'silent': [1],
              'subsample': [0.7],
              'colsample_bytree': [0.7],
              'n_estimators': [500,1000]}

model_xgb = xgb.XGBRegressor()

clf = GridSearchCV(model_xgb, parametersGrid, cv = n_folds, scoring='neg_mean_squared_error', refit=True, n_jobs = 5, verbose=True)

clf.fit(X_train, Y_train)
model_xgb = clf.best_estimator_

y_predicted_dummy = clf.predict(X_test)
mse = mean_squared_error(Y_test,y_predicted_dummy)
print("XGBRegressor MSE: ",mse)         # XGBRegressor MSE:  0.017444718427058307

# %% Averaging Models

class AveragingModels():
    def __init__(self, models):
        self.models = models
        
    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self
    
    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([model.predict(X) for model in self.models_])
        return np.mean(predictions, axis=1)  


averaged_models = AveragingModels(models = (model_xgb, eNet))
averaged_models.fit(X_train, Y_train)

y_predicted_dummy = averaged_models.predict(X_test)
mse = mean_squared_error(Y_test,y_predicted_dummy)
print("Averaged Models MSE: ",mse)      # AccelerationAveraged Models MSE:  0.016551165591244896 


"""
StandartScaler
    linear Regression MSE : 0.020632204780133005
    Ridge MSE:  0.019725338010801185
    Lasso MSE:  0.01752159477082249
    ElasticNet MSE:  0.017190543094566392

RobustScaler
    linear Regression MSE : 0.020487058280775586
    Ridge MSE:  0.01937186998245171
    Lasso MSE:  0.018707192302889232
    ElasticNet MSE:  0.018361024469839903
    XGBRegressor MSE:  0.017444718427058307
    Averaged Models MSE:  0.016551165591244896
"""





# %%

# Tahmin Deneme
test = [71, 65, 1773, 19, 71, 0, 1, 0, 0, 0, 0, 0, 1]    #3.46 --> Gerçek Degerler
test = [121, 110, 2260, 14, 73, 0, 1, 0, 0, 0, 0, 1, 0]  #3.21
test = [340, 160, 3609, 8, 70, 0, 0, 0, 0, 1, 1, 0, 0]   #2.70

sutunlar = ['Displancement', 'Horsepower', 'Weight', 'Acceleration',
       'Model Year', 'Cylinders_3', 'Cylinders_4', 'Cylinders_5',
       'Cylinders_6', 'Cylinders_8', 'Origin_1', 'Origin_2', 'Origin_3']

test2 = pd.Series(data=test, index=sutunlar)
y_predicted_dummy = clf.predict(test2.to_numpy().reshape(1, -1))
y_predicted_dummy



