import cv2
from video import camera

stream = camera()
stream.start()
frame = stream.read()

# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
frame_width = int(frame.shape[0])
frame_height = int(frame.shape[1])

print((frame_width,frame_height))
 
# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height)) 

while True:
    frame = stream.read()
    cv2.imshow("camera",frame)

    # Write the frame into the file 'output.avi'
    out.write(frame)

    if cv2.waitKey(1) == ord('q'):
        break


stream.stop()
out.release()
cv2.destroyAllWindows()