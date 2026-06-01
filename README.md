# YOLO NCNN Detector for Android

Android-приложение для детекции объектов на базе пользовательских YOLO моделей через NCNN.

## Возможности

- **Поддержка YOLO v5, v6, v7, v8, v9, v10, v11** (любые NCNN модели)
- **Загрузка модели с устройства** — выбери `.param` и `.bin` файлы
- **Реального времени** — камера + детекция с отображением боксов и FPS
- **Настройка на лету** — все параметры меняются без перезапуска:
  - Confidence threshold
  - NMS threshold  
  - Input size
  - Количество классов
  - CPU потоки (1-8)
  - GPU ускорение (Vulkan)
  - Имена выходных слоёв NCNN (для нестандартных моделей)
  - Файл с именами классов (.txt)
- **Переключение камер** (фронт/тыл)

## Конвертация модели в NCNN

### YOLOv8/v9/v10/v11
```bash
# Экспорт из Ultralytics
yolo export model=yolov8n.pt format=ncnn
# Результат: model.ncnn.param + model.ncnn.bin
```

### YOLOv5
```bash
# Через onnx
python export.py --weights yolov5s.pt --include onnx
onnx2ncnn yolov5s.onnx yolov5s.param yolov5s.bin
```

## Использование

1. Скачай APK из [Actions artifacts](../../actions)
2. Установи на Android 8.0+
3. Выбери `.param` файл модели
4. Выбери `.bin` файл модели
5. Нажми **Настройки** — укажи версию YOLO, кол-во классов, имена слоёв
6. Нажми **Запустить камеру**
7. В камере нажми ⚙️ для быстрого изменения параметров

## Имена выходных слоёв

| Формат | Output 0 | Output 1 | Output 2 |
|--------|----------|----------|----------|
| YOLOv8/v9/v10/v11 | `output0` | — | — |
| YOLOv5 (стандарт) | `output` | `781` | `801` |
| YOLOv5 (custom) | Смотри в .param файле | | |

## Сборка

```bash
# Загрузи NCNN
wget https://github.com/Tencent/ncnn/releases/download/20240102/ncnn-20240102-android-vulkan.zip
unzip ncnn-20240102-android-vulkan.zip
mkdir -p app/src/main/cpp/ncnn
cp -r ncnn-20240102-android-vulkan/arm64-v8a app/src/main/cpp/ncnn/
cp -r ncnn-20240102-android-vulkan/armeabi-v7a app/src/main/cpp/ncnn/

# Собери APK
./gradlew assembleDebug
```
