chcp 65001 > nul

@echo off

echo Создание виртуального окруженияы...

python -m venv venv

echo Активация окружения...
call venv\Scripts\activate

echo Установка numpy в виртуальное окружение...

echo Виртуальное окружение создано!


pause