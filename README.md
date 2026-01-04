# Smile_detection
Real-time Smile Detection using Python and OpenCV.
import cv2
video=cv2.VideoCapture(0)
facescascade=cv2.CascadeClassifier("D:\smile_detection\dataset\haarcascade_frontalface_default.xml")
smilecascade=cv2.CascadeClassifier("D:\smile_detection\dataset\haarcascade_smile.xml")

while True:
    success,img=video.read()
    grayImg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=facescascade.detectMultiScale(grayImg,1.1,4)
    cnt=400
    keyPressed=cv2.waitKey(1)
    for(x,y,w,h)in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,0),3)
        smiles=smilecascade.detectMultiScale(grayImg,1.5,15)
        for(x,y,w,h)in smiles:
            cv2.rectangle(img,(x,y),(x+w,y+h),(100,100,100),5)
            print("Image "+str(cnt)+"Saved")
            path=r'./image1'+str(cnt)+'.jpg'
            cv2.imwrite(path,img)
            cnt+=1
            if(cnt>=403):
                break
    cv2.imshow('live Video',img)        
    if(keyPressed & 0xFF==ord('q')):
        break
video.release()
cv2.destroyAllWindows()            
 

