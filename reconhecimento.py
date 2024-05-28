import cv2
import numpy as np
import tensorflow as tf

# Carregar modelo pré-treinado para detecção de faces
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Carregar modelo pré-treinado para reconhecimento facial
# Substitua 'path_para_modelo' pelo caminho para o modelo treinado
model = tf.keras.models.load_model('path_para_modelo')

# Função para fazer a detecção e reconhecimento de faces
def detect_and_recognize_faces(image):
    # Converter a imagem para escala de cinza
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detectar faces na imagem
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # Iterar sobre as faces detectadas
    for (x,y,w,h) in faces:
        # Extrair a região de interesse (ROI) da face
        face_roi = gray[y:y+h, x:x+w]
        # Redimensionar a ROI para o tamanho esperado pelo modelo
        face_roi_resized = cv2.resize(face_roi, (224, 224))
        # Normalizar os valores de pixel da ROI
        face_roi_normalized = face_roi_resized / 255.0
        # Adicionar uma dimensão adicional para o canal de cor (shape = (1, 224, 224, 1))
        face_roi_processed = np.expand_dims(face_roi_normalized, axis=0)
        # Realizar a inferência usando o modelo de reconhecimento facial
        predictions = model.predict(face_roi_processed)
        # Converter as probabilidades em rótulos
        predicted_label = np.argmax(predictions)
        # Desenhar um retângulo ao redor da face detectada
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
        # Escrever o rótulo previsto na imagem
        cv2.putText(image, 'Pessoa ' + str(predicted_label), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    
    return image

# Inicializar a captura de vídeo
cap = cv2.VideoCapture(0)

while True:
    # Capturar o frame da câmera
    ret, frame = cap.read()
    
    # Chamar a função para detectar e reconhecer faces
    frame_detected = detect_and_recognize_faces(frame)
    
    # Exibir o frame com as faces detectadas e reconhecidas
    cv2.imshow('Reconhecimento Facial', frame_detected)
    
    # Sair do loop se a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar a captura de vídeo e fechar todas as janelas
cap.release()
cv2.destroyAllWindows()
