import cv2
import mediapipe as mp
import csv
import copy
import itertools



# write a row to the csv file

 # open the file in the write mode
f = open('landmarks.csv', 'w', newline='')

# create the csv writer
writer = csv.writer(f)


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


my_drawing_specs = mp_drawing.DrawingSpec(color = (0, 255, 0), thickness = 1)
cap = cv2.VideoCapture(0)

mp_face_mesh = mp.solutions.face_mesh


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point.append(int(landmark_x))
        landmark_point.append(int(landmark_y))

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if isinstance(landmark_point, (list, tuple)):
            if index == 0:
                base_x, base_y = landmark_point[0], landmark_point[1]

            temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
            temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Flatten the list of lists or tuples
    temp_landmark_list_flat = []
    for point in temp_landmark_list:
        if isinstance(point, (list, tuple)):
            temp_landmark_list_flat.extend(point)
        else:
            temp_landmark_list_flat.append(point)

    # Normalization
    max_value = max(map(abs, temp_landmark_list_flat))

    def normalize_(n):
        return n / max_value

    return temp_landmark_list_flat

    

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
                # print(face_landmarks.landmark)
                #print the coorinates every landmarks
                landmark_list = calc_landmark_list(image, face_landmarks)
                landmark_list = pre_process_landmark(landmark_list)
               
                writer.writerow(landmark_list)

                # Print the coordinates
                # print(f"Landmark {idx + 1}: X={landmark_x}, Y={landmark_y}")
                #log the coordinates to a csv file



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