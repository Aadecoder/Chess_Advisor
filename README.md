# ♟️ Chess Move Predictor

A computer vision-based chess assistant that detects chessboard using YOLO, converts the board state into a FEN string, and then uses the Stockfish chess engine to suggest the top moves with visual arrows overlayed on the board.

> ⚠️ **Platform**: This project currently runs on **Windows only**.

---

## 🔍 Features

- Detects chessboard and pieces from an image using YOLO.
- Converts detected pieces into an accurate FEN string.
- Uses `python-chess` to validate the board state and interface with Stockfish.
- Renders best moves as colored arrows using OpenCV.
- Displays suggested moves and their scores from the Stockfish engine.

---

## 🧰 Requirements

- Make sure you have **Python 3.8+** installed.

- Install required libraries using pip:

```bash
pip install opencv-python python-chess supervision
```
- Download Stockfish engine for Windows from: https://stockfishchess.org/download/

- Get your Roboflow API key for the YOLO model hosted on Roboflow.

---

## 🚀 How to Run

- Replace the image path in the code with your screenshot or board image:

```python
image = cv2.imread(r"C:\path\to\your\chess_image.png")
```
- Replace the model_id and api_key in this line with your Roboflow details:

```python
model = get_model(model_id="chess.comdetection/4", api_key="your_api_key_here")
```
- Make sure the Stockfish binary path is correct:

```python
engine_path = "C:/path/to/stockfish.exe"
```
- Run the script:

```bash
python main.py
```
---

## 💡 How It Works

- Image Input: Loads the board image.

- Piece Detection: Uses YOLO model via Roboflow to detect all pieces and the board.

- Grid Mapping: Divides board into 8x8 grid and maps each piece to its respective square.

- FEN Generation: Converts this mapped data to a FEN string (the format used by chess engines).

- Stockfish Analysis: Loads Stockfish and gets top multipv=3 moves for 0.5s of analysis.

- Visualization: Draws arrows over the original image for suggested moves.

- Display: Opens a window showing the image with annotations.

---

## 📌 Known Limitations

- Works best with clear standard chess.com boards.

- Currently no support for castling, en passant detection, or move history tracking.

- Not tested on real-world photos or noisy boards.

---

## 🙏 Acknowledgements

- [Roboflow](https://roboflow.com) – for providing the YOLO model inference API.
- [Stockfish](https://stockfishchess.org) – for the powerful chess engine used to generate move suggestions.
- Python libraries: `python-chess`, `opencv-python`, `supervision`, and `inference` – for enabling the core functionality of this project.

---

### Author: Aditya Rajput