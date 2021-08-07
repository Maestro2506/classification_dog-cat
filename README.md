Котопес на Flask

Для того, чтобы протестировать проект необходимо:
1) скачать или клонировать проект
2) загрузить все необходимые пакеты библиотек с помощью команды: "pip intstall -r requirements.txt"
3) запустить файл "app.py" , туда также можно отправить агрумент "--gpu" типа boolean, если у вас установлена CUDA и все соответсвующие программы, если же агрумент не был отправлен то программа по дефолту будет выполнятся на вашем CPU:
   1)команда для запуска на CPU: python app.py  
   2)команда для запуска на GPU: python app.py --gpu True
4) запустить файл request.py, который может принимать 3 агрумента:
   1) "--directory" типа string. Он является указателем того, с какой папки необходимо конвертировать изображения в формат base64 а затем в json объект. По дефолту этот аргумент принимает значение 'images', так как изображениz хранятся в этой папке. Если вы хотите протестировать свои изображения, то необходимо создать свою папку в папке проекта и внутри этой папки указать еще 2 папки 'cat' и 'dog' и сохранить изображения кошек в папку 'cat' и изображения собак в папку 'dog'(это важно так как из названия этих папок потом считываются правдивые значения классификации). Затем просто запустить файл request.py с указанием агрумента "--directory 'YOUR_FOLDER_NAME'"
   2) "--url" типа string. Это url адрес на котором запущен файл app.py. По дефолту значение будет 'http://127.0.0.1:5000/predict', однако вы можете указать свой url если захотите запустить файл app.py по другому адресу.
   3) "--excel_name" типа string, который необходимо указывать без указания расширения файла, то есть без '.xslx'. Этот агрумент указывает точное название файла excel который вы хотите получить при завершение программы request.py. Если вы не укажете этот агрумент по дефолту будет создан excel файл с названием 'probabilities.xlsx' в котором будет храниться таблица с ID изображений, cat_prob, dog_prob и true_label значениями, В столбце true label: 0 - кошка, а 1 - собака.

Если вы хотите указать все агрументы просто напишите их друг за другом, пример: "python request.py --directory 'new_images' --url 'http://2.1.0.1:1020/predict' --excel_name 'my_excel'"

Анализ оценки точности модели хранится в excel файле "analysis.xslx", где столбец predicted_label содержит предсказанные моделью значения. Присвоение лейбла проводилось по принципу найбольшей вероятности, то есть если значение cat_prob>dog_prob то изображение классифицируется как кошка и наоборот. При равной вероятности cat_prob и dog_prob изображение классифицируется как собака исходя из кода источника классификационномй модели. 
Таблицы и диаграммы хранятся в листе "Анализ модели" в том же файле "analysis.xslx".

Источники:
   1) Датасет с изображениями - https://www.kaggle.com/chetankv/dogs-cats-images. Я использовал первые 500 изображений кошек и собак из папки test_set (с 4001 по 4500)
   2) Open-source классификатор - https://github.com/girishkuniyal/Cat-Dog-CNN-Classifier. Сохраненная .h5 модель хранится в папке resourses.
