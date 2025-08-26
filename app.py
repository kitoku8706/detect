import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

@st.cache_resource
def load_model():
    return YOLO("MODEL_PATH")  # 여기에 실제 모델 경로 넣기

model = load_model()
names = model.names

st.title("YOLOv8 Object Detection 🚀")
uploaded_file = st.file_uploader("이미지 업로드", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    results = model.predict(image, conf=0.25, iou=0.7)
    r = results[0]

    class_ids = r.boxes.cls.cpu().numpy().astype(int)
    class_counts = {names[i]: (class_ids == i).sum() for i in np.unique(class_ids)}

    st.subheader("클래스별 탐지 개수")
    st.write(class_counts)

    selected_classes = st.multiselect(
        "표시할 클래스 선택",
        options=list(names.values()),
        default=list(names.values())
    )

    img_np = np.array(image)
    for box, cls_id, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
        cls_id = int(cls_id.item())
        cls_name = names[cls_id]
        if cls_name not in selected_classes:
            continue
        x1, y1, x2, y2 = map(int, box.tolist())
        cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_np, f"{cls_name} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    st.subheader("탐지 결과")
    st.image(img_np, caption="Detection Result", use_column_width=True)
