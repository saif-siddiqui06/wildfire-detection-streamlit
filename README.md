# Wildfire Detection Streamlit App

This repository contains a Streamlit version of the wildfire detection project.
It uses a YOLO object detection model to detect possible fire and smoke signals
in uploaded images and sampled video frames.

## Features

- Image upload wildfire detection
- Video upload with sampled-frame analysis
- Real-time browser camera detection using WebRTC
- Camera snapshot fallback mode
- Annotated detection preview
- Fire and smoke count summary
- Risk level classification
- Confidence summary
- Downloadable annotated video output
- Streamlit Community Cloud deployment files

## Live Deployment

Try the deployed app here:

```text
https://wildfire-detection-app-dizjzodmvxcwuxlh5usrry.streamlit.app/
```

## Project Structure

```text
wildfire-detection-streamlit/
├── app.py
├── model.py
├── best.pt
├── last.pt
├── requirements.txt
├── packages.txt
├── runtime.txt
├── static/
└── README.md
```

## Run Locally

```bash
python -m venv venv
venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Cloud Settings

```text
Branch: main
Main file path: app.py
```

## Notes

- Video analysis is sampled for cloud performance.
- Live camera detection uses the user's browser camera through WebRTC.
- If WebRTC is blocked by browser/network settings, use the camera snapshot fallback.
- This project is a demonstration and should not be used as the only source for emergency decisions.
- Wildfire alerts should always be verified by humans and official emergency services.
