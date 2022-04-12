import os

import cv2
import numpy as np

image_path = 'D:\\Pattern_Letters_HR_PAD\\BBDD\\3DMAD\\session03\\'
image_name_video = []
# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

for f in [f for f in os.listdir(image_path)]:

    if not("_C.avi" in f):  # OULU
        continue

    carpeta = os.path.join(image_path, f)
    cap = cv2.VideoCapture(carpeta)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    nFrames = cap.get(7)
    max_frames = int(nFrames)
    ruta_parcial = os.path.join('D:\\Pattern_Letters_HR_PAD\\BBDD\\3DMAD\\DeepFrames',f)
    if not(os.path.exists(ruta_parcial)):
        os.mkdir(ruta_parcial)
    ruta_parcial2 = os.path.join('D:\\Pattern_Letters_HR_PAD\\BBDD\\3DMAD\\RawFrames',f)
    if not(os.path.exists(ruta_parcial2)):
        os.mkdir(ruta_parcial2)

    L = 36
    C = []
    ka = 1

    while (cap.isOpened() and ka < max_frames):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        # rectangle around the faces
        for (x, y, w, h) in faces:
            # face = cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            face = frame[y:y + h, x:x + w]

        face = cv2.resize(face, (L, L), interpolation=cv2.INTER_AREA)
        C.append(face)
        ka += 1

    cap.release()
    cv2.destroyAllWindows()

    C = np.array(C, dtype=np.float64)

    epsilon = 0.1

    D = np.diff(C, axis=0) / (C[:-1] + C[1:])
    medias_D = np.mean(D, axis=0)
    desviaciones_D = np.std(D, axis=0)
    D = (D - medias_D) / (desviaciones_D + epsilon)
    D = np.uint8(D)

    for k, imagen in enumerate(D):
        nombre_salvar = os.path.join(ruta_parcial, str(k + 1) + '.png')
        cv2.imwrite(nombre_salvar, imagen)

    medias_C = np.mean(C, axis=0)
    desviaciones_C = np.std(C, axis=0)
    C = (C - medias_C) / (desviaciones_C + epsilon)
    C = C[:-1]
    C = np.uint8(C)

    for k, imagen in enumerate(C):
        nombre_salvar = os.path.join(ruta_parcial2, str(k + 1) + '.png')
        cv2.imwrite(nombre_salvar, imagen)


print("Exiting...")
