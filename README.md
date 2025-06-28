# ♟️ Chess Advisor

A computer vision-based chess assistant that detects chessboard using YOLO, converts the board state into a FEN string, and then uses the Stockfish chess engine to suggest the top moves with visual arrows overlayed on the board.

> ⚠️ **Platform**: This project currently runs on **Windows only**.

---

## 🔍 Features

- Captures a specified portion of the screen using mss.
- Detects chessboard and pieces from the captured image using YOLO.
- Converts detected pieces into an accurate FEN string.
- Uses `python-chess` to validate the board state and interface with Stockfish.
- Renders best moves as colored arrows using OpenCV.
- Displays suggested moves and their scores from the Stockfish engine.

---

## 🧰 Requirements

- Make sure you have **Python 3.8+** installed.

- Install required libraries using pip:

```bash
pip install opencv-python numpy python-chess supervision ultralytics mss keyboard
```
- Download Stockfish engine for Windows from: https://stockfishchess.org/download/

---

## 🚀 How to Run

- Make sure the Stockfish binary path is correct:

```python
ENGINE_PATH = "C:/path/to/stockfish.exe"
```

- Make sure that the monitor dimensions and placement is same as the actual chess board

```python
monitor = {"top": 143, "left": 49, "width": 870, "height": 870}
```

- To accurately align chessboard placement to monitor you can use a Power Toys tool called Screen Ruler

- top : distance of the chess board from the top of the screen in pixels 

- left : distance of the chess board from the left of the screen in pixels

- width : width of the chess board in pixels

- height : height of the chess board in pixels

- Run the script:

```bash
python main.py
```
---

## Directory Structure

```bash
Chess_Advisor/
├── weights/
│   └── chess_detection.pt
├── main.py
└── README.md
```

---

## 💡 How It Works

- Captures the Chess Board : Uses MSS model to capture the image of the chess board on the screen

- Piece Detection: Uses YOLO model to detect all pieces and the board.

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

- Only works if you play as white.

---

## 🙏 Acknowledgements

- [Stockfish](https://stockfishchess.org) – for the powerful chess engine used to generate move suggestions.
- Python libraries: `python-chess`, `numpy`, `opencv-python`, `supervision`, `mss`, `keyboard` and `ultralytics` – for enabling the core functionality of this project.

---

### Author: Aditya Rajput