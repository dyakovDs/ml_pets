По известному датасету строится модель классификации для предсказания ценовой категории авто. 

Обработка данных происходит при помощи пайплайна, который сохранен в Google Drive и доступен по ссылке:
https://drive.google.com/file/d/1qKde8uzgQgB2p_OSfeRBL-qmPZHBQW1I/view?usp=sharing

Подразумевается, что модель может быть внедрена в web-сервис при помощи FastApi. Для этого реализована возможность следующих get-запросов: /status, /version; и post-запроса: /predict, где в body необходимо передать json определённого формата(примеры json файлов /model/data).

---ИНСТРУКЦИЯ ПО ЗАПУСКУ ПРОГРАММЫ---

1) Убедитесь, что используется последняя версия pip. 
Для этого в терминал вставьте команду python -m pip install --upgrade pip

2) В файле requirements.txt содержатся версии библиотек, используемых в проекте. 
Среда разработки предложит установить эти библиотеки(если этого не произошло - используйте команду pip install -r requirements.txt), для выполнения программы сделать это необходимо.

3) По ссылке из файла model.txt загрузите пайплайн и поместите файл cars_best_pipe.pkl в папку с проектом.

4) После успешной установки библиотек и загрузки пайплайна программа будет готова к работе.
