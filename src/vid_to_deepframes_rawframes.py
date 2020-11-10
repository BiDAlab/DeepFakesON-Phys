
import numpy as np
import os
import cv2




image_path = 'D:\\Pattern_Letters_HR_PAD\\BBDD\\3DMAD\\session03\\'
image_name_video = []
# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

for f in [f for f in os.listdir(image_path)]:
    
    if not("_C.avi" in f): #OULU
        continue
    
    carpeta= os.path.join(image_path, f)
    cap = cv2.VideoCapture(carpeta)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    nFrames = cap.get(7)
    max_frames = int(nFrames)
    ruta_parcial = os.path.join('D:\\Pattern_Letters_HR_PAD\\BBDD\\3DMAD\\DeepFrames',f) 
    if not(os.path.exists(ruta_parcial)) :
        os.mkdir(ruta_parcial);
    ruta_parcial2 = os.path.join('D:\\Pattern_Letters_HR_PAD\\BBDD\\3DMAD\\RawFrames',f) 
    if not(os.path.exists(ruta_parcial2)) :
        os.mkdir(ruta_parcial2);
    
    L = 36
    C_R=np.empty((L,L,max_frames))
    C_G=np.empty((L,L,max_frames))
    C_B=np.empty((L,L,max_frames))
    
    D_R=np.empty((L,L,max_frames))
    D_G=np.empty((L,L,max_frames))
    D_B=np.empty((L,L,max_frames))
    
    D_R2=np.empty((L,L,max_frames))
    D_G2=np.empty((L,L,max_frames))
    D_B2=np.empty((L,L,max_frames))
    
    medias_R = np.empty((L,L))
    medias_G = np.empty((L,L))
    medias_B = np.empty((L,L))
    
    desviaciones_R = np.empty((L,L))
    desviaciones_G = np.empty((L,L))
    desviaciones_B = np.empty((L,L))
    
    imagen = np.empty((L,L,3))
    
    medias_CR = np.empty((L,L))
    medias_CG = np.empty((L,L))
    medias_CB = np.empty((L,L))
    
    desviaciones_CR = np.empty((L,L))
    desviaciones_CG = np.empty((L,L))
    desviaciones_CB = np.empty((L,L))
    ka            = 1
    
    
    while(cap.isOpened() and ka< max_frames):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        #rectangle around the faces
        for (x, y, w, h) in faces:
            # face = cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            face = frame[y:y + h, x:x + w]
            
       
        face = cv2.resize(face, (L,L), interpolation = cv2.INTER_AREA)
        # cv2.imshow('img', face)
        # cv2.waitKey()
        C_R[:,:,ka] = face[:,:,0]
        C_G[:,:,ka] = face[:,:,1]
        C_B[:,:,ka] = face[:,:,2]
        
        
        if ka > 1:
            D_R[:,:,ka-1] = ( C_R[:,:,ka] - C_R[:,:,ka-1] ) / ( C_R[:,:,ka] + C_R[:,:,ka-1] );
            D_G[:,:,ka-1] = ( C_G[:,:,ka] - C_G[:,:,ka-1] ) / ( C_G[:,:,ka] + C_G[:,:,ka-1] );
            D_B[:,:,ka-1] = ( C_B[:,:,ka] - C_B[:,:,ka-1] ) / ( C_B[:,:,ka] + C_B[:,:,ka-1] );
        ka = ka+1
     
    
        
    for i in range(0,L):
        for j in range(0,L):
            medias_R[i,j]=np.mean(D_R[i,j,:]) 
            medias_G[i,j]=np.mean(D_G[i,j,:]) 
            medias_B[i,j]=np.mean(D_B[i,j,:]) 
            desviaciones_R[i,j]=np.std(D_R[i,j,:]) 
            desviaciones_G[i,j]=np.std(D_G[i,j,:]) 
            desviaciones_B[i,j]=np.std(D_B[i,j,:]) 
            
    for i in range(0,L):
        for j in range(0,L):
            medias_CR[i,j]=np.mean(C_R[i,j,:]) 
            medias_CG[i,j]=np.mean(C_G[i,j,:]) 
            medias_CB[i,j]=np.mean(C_B[i,j,:]) 
            desviaciones_CR[i,j]=np.std(C_R[i,j,:]) 
            desviaciones_CG[i,j]=np.std(C_G[i,j,:]) 
            desviaciones_CB[i,j]=np.std(C_B[i,j,:])         
            
    for k in range(0,max_frames):
        D_R2[:,:,k] = (C_R[:,:,k] - medias_CR)/(desviaciones_CR+000.1)
        D_G2[:,:,k] = (C_G[:,:,k] - medias_CG)/(desviaciones_CG+000.1)
        D_B2[:,:,k] = (C_B[:,:,k] - medias_CB)/(desviaciones_CB+000.1)
     


    for k in range(0,max_frames):
        
        imagen[:,:,0] = D_R2[:,:,k]
        imagen[:,:,1] = D_G2[:,:,k]
        imagen[:,:,2] = D_B2[:,:,k]

        imagen= np.uint8(imagen)
        
        nombre_salvar= os.path.join(ruta_parcial2,str(k)+'.png')
        cv2.imwrite(nombre_salvar, imagen)
        

    for k in range(0,max_frames):
        
        D_R[:,:,k] = (D_R[:,:,k] - medias_R)/(desviaciones_R+000.1)
        D_G[:,:,k] = (D_G[:,:,k] - medias_G)/(desviaciones_G+000.1)
        D_B[:,:,k] = (D_B[:,:,k] - medias_B)/(desviaciones_B+000.1)
        
    for k in range(0,max_frames):
        
        imagen[:,:,0] = D_R[:,:,k]
        imagen[:,:,1] = D_G[:,:,k]
        imagen[:,:,2] = D_B[:,:,k]
        
        imagen= np.uint8(imagen)

        nombre_salvar= os.path.join(ruta_parcial,str(k)+'.png')
        cv2.imwrite(nombre_salvar, imagen)            
        
        
    cap.release()
    cv2.destroyAllWindows()
print("Exiting...")
