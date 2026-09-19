# Credit scoring: comparison of classification models

Проект решает задачу бинарной классификации: по социально-демографическим, кредитным и финансовым признакам оценивается риск дефолта заёмщика (`loan_status = 1`).

В репозитории сохранён исходный notebook с EDA. Воспроизводимый benchmark модей вынесен в `train_models.py`.

## Данные и схема оценки

- исходная выборка: 45 000 объектов;
- целевой класс: 10 000 дефолтов (22,2%);
- удалено 7 записей с возрастом выше 100 лет по заранее заданному правилу;
- stratified split: train 28 795, validation 7 199, test 8 999;
- test не используется для выбора модели или бизнес-порога;
- числовые признаки: median imputation + `StandardScaler`;
- категориальные признаки: most-frequent imputation + `OneHotEncoder`;
- preprocessing обучается только на train внутри `Pipeline`.

## Модели

- Logistic Regression;
- Linear SVM;
- RBF SVM;
- Decision Tree;
- Random Forest (bagging);
- Histogram Gradient Boosting.

## Метрики

ROC-AUC и Average Precision оценивают качество ранжирования риска. Дополнительно на validation выбирается порог, который максимизирует долю одобрений при ограничении:

```text
bad rate среди одобренных <= 5%
```

Одобряются клиенты с риском **ниже или равным** порогу. Это исправляет ошибку исходного notebook, где на validation проверялась противоположная часть score.

## Результаты

| Model | Validation ROC-AUC | Test ROC-AUC | Test Average Precision | Approval rate | Bad rate among approved |
|---|---:|---:|---:|---:|---:|
| Histogram Gradient Boosting | **0.9775** | **0.9769** | **0.9357** | **78.3%** | 4.76% |
| Random Forest | 0.9729 | 0.9728 | 0.9252 | 77.4% | 4.55% |
| Decision Tree | 0.9635 | 0.9661 | 0.8985 | 70.7% | 3.15% |
| RBF SVM | 0.9617 | 0.9619 | 0.8859 | 75.5% | 4.55% |
| Linear SVM | 0.9528 | 0.9522 | 0.8525 | 73.2% | 4.92% |
| Logistic Regression | 0.9527 | 0.9522 | 0.8523 | 73.2% | 4.93% |

Лучшая модель по validation ROC-AUC — Histogram Gradient Boosting. Её test ROC-AUC составил `0.9769`. Порог, подобранный только на validation, дал на test 78,3% одобрений при bad rate 4,76%.

## Запуск

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train_models.py --n-jobs 2
```

Быстрый запуск без RBF SVM:

```bash
python train_models.py --skip-rbf --n-jobs 2
```

Результаты сохраняются в:

- `artifacts/model_comparison.csv`;
- `artifacts/metrics.json`;
- `artifacts/best_model.joblib` (gitignored).

Тесты бизнес-порога:

```bash
python -m unittest discover -s tests
```

## Ограничения

- источник и условия сбора `loan_data.csv` пока не задокументированы;
- порог 5% — учебное бизнес-ограничение, а не реальная кредитная политика;
- для production-скоринга нужны temporal/out-of-time validation, калибровка, drift/fairness-анализ, мониторинг и независимая валидация.
