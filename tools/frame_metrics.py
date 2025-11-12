import cv2, os, pandas as pd

inp = r"C:\Users\user\Creative-Pipeline\renders\Kelly\kelly_test_talk_v1.mp4"
out_csv = r"C:\Users\user\Creative-Pipeline\analytics\Kelly\kelly_test_frame_metrics.csv"

cap = cv2.VideoCapture(inp)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
frame=0; rows=[]
ret = True
prev_gray=None

while ret:
    ret, img = cap.read()
    if not ret: break
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_luma = float(gray.mean())
    motion = 0.0
    if prev_gray is not None:
        diff = cv2.absdiff(gray, prev_gray)
        motion = float(diff.mean())
    rows.append({"frame":frame,"time_s":frame/float(fps),"mean_luma":mean_luma,"motion_diff":motion})
    prev_gray = gray; frame+=1

cap.release()
pd.DataFrame(rows).to_csv(out_csv, index=False)
print("Frame metrics CSV saved:", out_csv)
