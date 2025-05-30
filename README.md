# Алгоритмы классического машинного обучения и Глубокого машинного обучения

Репозиторий содержит реализации классических алгоритмов машинного обучения и глубокого обучения, а также решения Kaggle-соревнований.

## Структура проекта

### 1. Classic ML

#### 1.1 Обучение без учителя (Кластеризация)
- [Clustering.ipynb](Classic%20ML/Обучение%20без%20учителя/Clustering.ipynb) - рассмотрение работы разных алгоритмов кластеризации (DBSCAN, KMeans, AgglomerativeClustering) на примере датасета данным по нефти (Проницаемость,	Общая толщина,	Нефтенасыщенная толщина,	Нефтенасыщенность)

#### 1.2 Обучение с учителем
**SVM и SVM классификатор, Логистическая регрессия:**
- [SVC, LR и калибровка вероятностей.ipynb](Classic%20ML/Обучение%20с%20учителем/SVC,%20LR%20и%20калибровка%20вероятностей.ipynb) - сравнение классификаторов SVC и Logistic Regression; Калибровка вероятностей; Методы обработки категориальных переменных (OneHot, Счетчики). Использование Логистической регрессии для максимизации прибыли по данным датасета телефонных звонков клиентам [[UCI Bank Marketing Dataset]](https://archive.ics.uci.edu/ml/datasets/bank+marketing)
- [SVM.ipynb](Classic%20ML/Обучение%20с%20учителем/SVM.ipynb) - построение нелинейных разделяющих поверхностей, подбор оптимальных параметров по выборке, уменьшение числа признаков

**Линейные модели:**
- [Линейная_регрессия.ipynb](Classic%20ML/Обучение%20с%20учителем/Линейная%20регрессия%20и%20классификация/Линейная_регрессия.ipynb) - Роль параметров, проверка адекватности модели МНК (некоррелированность остатков, нулевое мат ожидание и постоянная дисперсия), Логлинейная регрессия, предсказание выбросов по датасету [[CO2 Emissions Canada]](https://raw.githubusercontent.com/terinkov/sorces/refs/heads/main/co2_emissions_canada.csv)
- [Линейные_модели_регрессии_и_классификации.ipynb](Classic%20ML/Обучение%20с%20учителем/Линейная%20регрессия%20и%20классификация/Линейные_модели_регрессии_и_классификации.ipynb) - рассмотрение L1, L2, elastic-net регуляризаций; рассмотрение Логистической регрессии
- [boosting.ipynb](Classic%20ML/Обучение%20с%20учителем/boosting.ipynb) - Градиентный бустинг на решающих деревьях на примере библиотек: XGBoost, LightGBM, Catboost, HyperOpt 
- [KNN.ipynb](Classic%20ML/Обучение%20с%20учителем/KNN.ipynb) - K-ближайших соседей - классификация и регрессия
- [trees.ipynb](Classic%20ML/Обучение%20с%20учит  елем/trees.ipynb) - деревья решений и случайные леса (ансамбли деревьев)

### 2. Deep Learning
- [Convolutional_NN.ipynb](Deep%20Learning/Convolutional_NN.ipynb) [[Colab]](https://colab.research.google.com/drive/1R5s5xeqbXNfRUCQTm4ywtE6uvxu5SiRj#scrollTo=VRUsuZR2cQoY) - Pytorch свёрточные нейронные сети на датасете CIFAR10 с аугментацией
- [Fully_connected_networks.ipynb](Deep%20Learning/Fully_connected_networks.ipynb) [[Colab]](https://colab.research.google.com/drive/17VTfmdkMblmEJ_r5bMr6y1tG_ga4CY5W?authuser=1) - Pytorch полносвязные нейронные сети

### 3. Kaggle Competitions
- [Линейная_регрессия_NewYork_Taxi_соревнование_Kaggle.ipynb](Kaggle_competitions/Линейная_регрессия_NewYork_Taxi_соревнование_Kaggle.ipynb) [[Colab]](https://colab.research.google.com/drive/1idhz84E8cofOaQ7K1T29rUaiMTAfC28e) - решение Kaggle-соревнования по предсказанию стоимости поездок на такси в Нью-Йорке (Scikit-learn, Folium) - решено с помощью Линейной регрессии
- [Smokers_Health_Data_analysis.ipynb](Kaggle_competitions/Smokers_Health_Data_analysis.ipynb) - анализ данных о здоровье курильщиков (фильтрация данных, box-плоты, квантильный анализ, поиск корреляций и зависимостей и тд)


## Библиотеки
- Python 3.8+
- Jupyter Notebook
- Библиотеки: NumPy, Pandas, Scikit-learn, Matplotlib/Seaborn
- Для deep learning: PyTorch
