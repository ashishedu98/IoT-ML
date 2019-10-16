import face_recognition
import cv2
import numpy as np
import pickle
import time
# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
match_percentage=[]
start_time=time.time()
    # Grab a single frame of video
small_frame =cv2.imread("b.png")

    # Resize frame of video to 1/4 size for faster face recognition processing
    #small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        #open file to save encoding
    file_open=open("all_ppl_nameencoding","rb");
    known_face_encodings=pickle.load(file_open)
    file_open.close()
    
    
    namesfile_open=open("all_ppl_name","rb");
    known_face_names=pickle.load(namesfile_open)
    namesfile_open.close()      
    
    face_names = []
    for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
        temp_distance=[]
        temp_names=[]
        temp_matches=[]
        encodings=[]
        names=[]
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        print(matches)
        name = "*_*"
        less="0"
        i=0
        for flag in matches:
            if flag:
                filename=known_face_names[i]+"_encoding"
                namesfile_open=open(filename,"rb");
                encodings=pickle.load(namesfile_open)
                namesfile_open.close()
                mat=face_recognition.compare_faces(encodings, face_encoding)
                j=0
                for flag1 in mat:
                    if flag1:
                        enc=[]
                        enc.append(encodings[j])
                        face_distances = face_recognition.face_distance(enc,face_encoding)
                        temp_distance.append(face_distances)
                        temp_names.append(known_face_names[i])
                        j+=1
                    else:
                        j+=1
                i+=1
            else:
                i+=1
        #print(temp_names)
        #print(temp_distance)
        try:
            best_match_index = np.argmin(temp_distance)
            name = temp_names[best_match_index]
            notmatch=100*min(temp_distance)
            less="-"+str(int(100-notmatch))+"%"
            name+=less
        except:
            pass
        face_names.append(name)

process_this_frame = not process_this_frame

    # Display the results
for (top, right, bottom, left), name in zip(face_locations, face_names):
    cv2.rectangle(small_frame, (left, top), (right, bottom), (0,0,255), 2)

        # Draw a label with a name below the face
    #cv2.rectangle(small_frame, (left-15, bottom), (right+25,bottom+25), (15,214,245), cv2.FILLED)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(small_frame, name, (left-15, bottom+25), font,0.8,(0,0,255),2)
    # Display the resulting image
cv2.imshow('Video', small_frame)
print(time.time()-start_time)
if cv2.waitKey(1) & 0xFF == ord('q'):
    cv2.destroyAllWindows()























'''
temp_matches.append(flag)
                temp_encodings.append(known_face_encodings[i])
                temp_names.append(known_face_names[i])'''
