import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


my_drawing_specs = mp_drawing.DrawingSpec(color = (0, 255, 0), thickness = 1)
cap = cv2.VideoCapture(0)

mp_face_mesh = mp.solutions.face_mesh

#setting the max faces and confidence levels
with mp_face_mesh.FaceMesh(
        max_num_faces = 1,
        refine_landmarks = True,
        min_detection_confidence = 0.5,
        min_tracking_confidence = 0.5
    ) as face_mesh:
    
    #while camera is opened read camera
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break
            
            
        results = face_mesh.process(image)

        #if there are landmarks in sight
        if results.multi_face_landmarks:
            #for each coordinate in the data
            for face_landmarks in results.multi_face_landmarks:
                #draw the landmarks
                mp_drawing.draw_landmarks(
                    image = image,
                    landmark_list = face_landmarks,
                    connections = mp_face_mesh.FACEMESH_TESSELATION,#connect the landmarks
                    landmark_drawing_spec = None,
                    connection_drawing_spec = mp_drawing_styles
                    .get_default_face_mesh_tesselation_style()
                )
                
                mp_drawing.draw_landmarks(
                    image = image,
                    landmark_list = face_landmarks,
                    connections = mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec = None,
                    connection_drawing_spec = my_drawing_specs
    #                 .get_default_face_mesh_contours_style()
                )

        #show the camera
        cv2.imshow("My video capture", cv2.flip(image, 1))

        #if q is pressed end the camera and destroy the windows
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        
cap.release()
cv2.destroyAllWindows()