import streamlit as st
import numpy as np
from PIL import Image 
import cv2
import yolov5
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import requests

iceServers = [{"urls": ["stun:stun.l.google.com:19302"]}]
response = requests.get("https://wasteclassification.metered.live/api/v1/turn/credentials?apiKey=186a726547c4e088e331c5177d388e8aa1e6")
iceServers = response.json()

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": iceServers}
)

st.title("Waste Class Detector")
menu = ["Image","Stream"]
menu_choice = st.sidebar.selectbox("Menu",menu)


@st.cache_resource
def load_image(image_file):
	img = Image.open(image_file)
	return img 


@st.cache_resource
def load_model():
    # load model
    # model = yolov5.load('keremberke/yolov5m-garbage')
    # load pretrained model
    model = yolov5.load('best.pt')
    
    # set model parameters
    model.conf = 0.25  # NMS confidence threshold
    model.iou = 0.45  # NMS IoU threshold
    model.agnostic = False  # NMS class-agnostic
    model.multi_label = False  # NMS multiple labels per box
    model.max_det = 15  # maximum number of detections per image   
     
    return model


def draw_bbox(img, results):
    clf_img = np.array(img)     
    for box in results.xyxy[0]:
        xB = int(box[2])
        xA = int(box[0])
        yB = int(box[3])
        yA = int(box[1])
        label_value = box[5].item()
        
        if label_value == 0:
            label = 'Biodegradable'
            clr = (0,255,0)
        else:
            label = 'Recyclable'
            clr = (255,255,0)
        
        clf_img = cv2.rectangle(clf_img, (xA, yA), (xB, yB), clr, 1)
        cv2.putText(clf_img, label, (xA, yA-15), cv2.FONT_HERSHEY_SIMPLEX, 0.9, clr, 2)
    
    return clf_img
    
    
def detect_classes(img):
    model = load_model()
    # inference
    results = model(img, size=640)
    bbox_img = draw_bbox(img, results) 
    
    return bbox_img


class VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # vision processing
        flipped = img[:, ::-1, :]
        im_pil = Image.fromarray(flipped)
        clf_img = detect_classes(im_pil)
        bbox_img = np.array(clf_img)

        # model processing
        # model = load_model()
        # results = model(im_pil, size=640)
        # bbox_img = np.array(results.render()[0]) 

        return av.VideoFrame.from_ndarray(bbox_img, format="bgr24")

      

if menu_choice == "Image":  
    uploaded_img = st.file_uploader("Upload Image",type=['png','jpeg','jpg'])    
    if uploaded_img is not None:
        uimg = load_image(uploaded_img)
        col_uploaded, col_detected = st.columns(2)
        
        with col_uploaded:
            st.header("Waste Image")            
            st.image(uimg) 
            
        with col_detected:
            st.header("Classes")
            clf_img = detect_classes(uimg)
            st.image(clf_img) 
                       
else:
    st.subheader("Waste Stream")
    
    webrtc_ctx = webrtc_streamer(
    key="WYH",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
    )