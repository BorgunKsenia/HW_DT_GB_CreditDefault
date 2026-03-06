# Credit Default Prediction (HW_DT_GB)

## Описание
Проект по бинарной классификации факта невыполнения кредитных обязательств по текущему кредиту (**Credit Default**).  
Выполнен полный ML-пайплайн: EDA, preprocessing (FE), обучение и сравнение моделей, валидация, прогноз для тестового датасета.

## Данные
- `course_project_train.csv` - обучающая выборка (содержит целевой столбец `Credit Default`)
- `course_project_test.csv` - тестовая выборка (без целевого столбца)

Целевая переменная:
- `Credit Default = 1` - дефолт
- `Credit Default = 0` - нет дефолта

## Метрики
- **F1-score**
- **Gini** = 2·AUC_ROC − 1

Для максимизации F1-score на валидации подбирался **оптимальный порог** классификации (вместо фиксированного 0.5).

## Пайплайн решения
### 1) EDA
- просмотр структуры данных, доли классов, пропусков и описательных статистик.

### 2) Preprocessing
- заполнение пропусков: median (числовые), most_frequent (категориальные);
- one-hot encoding категориальных признаков;
- scaling числовых признаков для Logistic Regression.

### 3) Feature Engineering (FE)
- преобразование признака `Years in current job` из строкового формата (`< 1 year`, `10+ years`) в числовой.

### 4) Моделирование и сравнение моделей
**Модели “из коробки” (sklearn / boosting libraries):**
- Logistic Regression
- Decision Tree
- Random Forest
- ExtraTrees (ансамбль деревьев)
- LightGBM
- XGBoost
- CatBoost

**Самописные учебные реализации:**
- `HAND_CART` — дерево решений (CART по Gini)
- `HAND_RF` — случайный лес (bagging деревьев)
- `HAND_GB_Stumps` — градиентный бустинг (логистический бустинг над stumps)

Дополнительно построены графики сравнения F1/Gini и ROC-кривые для CatBoost vs `HAND_GB_Stumps`, а также визуализация интерпретируемого дерева решений на числовых признаках.

## Результаты (validation)
Лучший результат среди “box”-моделей показал **CatBoost**:
- **F1 = 0.556**
- **AUC = 0.765**
- **Gini = 0.529**
- оптимальный порог по F1: **~0.498**

Лучший самописный алгоритм — **HAND_GB_Stumps**:
- **F1 = 0.553**
- **AUC = 0.760**
- **Gini = 0.521**

## Артефакты
- `submission.csv` - прогноз для тестового датасета (столбец `Credit Default`), порядок строк сохранён.

## Как запустить (Google Colab)
1. Открыть ноутбук `.ipynb` в Colab.
2. Загрузить `course_project_train.csv` и `course_project_test.csv` в папку `/content`.
3. Запустить ячейки сверху вниз (или `Runtime → Run all`).
4. На выходе сформируется файл `/content/submission.csv`.
