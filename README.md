# MedNN
Сравнение методов обучения и видов нейросетей для диагностики меланомы. 

## Введение 
Тут будет рассказано про то зачем вообще эта программа и каким образом она работает 

## 1. Скачивание базы данных 
Все изображения были взяты с сайта [ISIC](https://www.isic-archive.com).
Скаичвание базы данных делится на 2 этапа:
1. Скачивание [мета-данных ](https://github.com/gurmaaan/MedNN/blob/master/Download_meta.ipynb). Скачиваются мета-данные всей БД. Это сделано чтобы можно было фильтровать данные которые будут скачиваться в удобном формате pd.DataFrame
2. Скачивание самих [изображений](https://github.com/gurmaaan/MedNN/blob/master/Download_img.ipynb). Все изображения сохраняются в папку _img_.

## 2. Деление выборки на train и test
Берутся метаданные. Имя выступает X, класс выступает как y. Далее делается стандартный sklearn [train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
Далее создаются папки train и test (validation), а внутри них папки соответсвующие классам изображений. В [этом блокноте](https://github.com/gurmaaan/MedNN/blob/master/Sort_img_into_folders.ipynb) происходит сортировка изображений по нужным папкам соответсвующим классам изображений. Это нужно для превращения их в дальнейшем в тензоры для обучения нейросетей

## 3. Обучение нейросети 
Здеь будет описана(ы) нейронка(и) на python и что там происходит

## 4. Обертка на Qt C++ для готовой обученной модели
1. Нужно для удобства
2. Основной функционал: 
  * просмотр
  * предсказание
  * похожие 
  * рекомендация
  * фильтрация по критериям 
3. Что где лежит в проекте 
