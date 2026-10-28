import cv2
from pathlib import Path

folder = Path('final_testing_dataset')
output = Path('2024_Week1_ARI_BUF_30fps_3/images')
output.mkdir(parents=True, exist_ok=True)

for i in range(1):  # Adjust range for number of videos
    vid = folder / f'2024_Week1_ARI_BUF_30fps_3.mp4'
    video = cv2.VideoCapture(str(vid))

    k=0
    while(video.isOpened()):
        flag, frame = video.read()

        if flag==False:
            break

        image_path = output / f"2024_Week1_ARI_BUF_30fps_3_{k}.jpg"
        cv2.imwrite(str(image_path), frame)
        print('image', k)
        k+=1


'''import cv2
from pathlib import Path

vid_name = '2017_1.mp4'
output = Path('2017_1/images')
output.mkdir(parents=True, exist_ok=True)

video = cv2.VideoCapture(str(output.parent / vid_name))

frame_count = 0
max_frames = 306

while video.isOpened() and frame_count < max_frames:
    flag, frame = video.read()

    if not flag:
        break

    image_path = output / f"frame_{frame_count:06d}.jpg"
    cv2.imwrite(str(image_path), frame)
    print('Saved:', image_path.name)
    frame_count += 1

print('done converting')'''