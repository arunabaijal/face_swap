import os
import cv2

def frames_to_video(pathIn,pathOut,fps):

    frame_array = []
    files = [f for f in os.listdir(
        pathIn) if os.path.isfile(os.path.join(pathIn, f))]

    #for sorting the file names properly
    files.sort(key=lambda x: int(x[:-4]))
    print(files)

    for i in range(len(files)):
        filename = os.path.join(pathIn, files[i])
        #reading each files
        img = cv2.imread(filename)
        # cv2.imshow('img', img)
        # cv2.waitKey(0)
        height, width, layers = img.shape
        size = (width, height)
#         print(filename)
        #inserting the frames into an image array
        frame_array.append(img)

    out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()


frames_to_video('./output/', 'output_filtered_10.mp4', 25)
