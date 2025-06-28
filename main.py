from ultralytics import YOLO
import cv2 as cv
import supervision as sv
import numpy as np
import chess
import chess.engine
import mss
import keyboard

# --- CONFIGURATION ---
MODEL_PATH = r"D:\PROGRAMMING\Projects\Chess_Advisor\weights\chess_detection.pt"
ENGINE_PATH = "C:/Users/adity/Downloads/stockfish/stockfish-windows-x86-64-avx2.exe"

# Load model
model = YOLO(MODEL_PATH)
class_names = model.names  # id-to-class map

# Define screen capture region (adjust this to your chessboard)
monitor = {"top": 143, "left": 49, "width": 870, "height": 870}

# Helper to get center of a square
def square_center(square, top_left_x, top_left_y, cell_width, cell_height):
    col = chess.square_file(square)
    row = chess.square_rank(square)
    x = int(top_left_x + (col + 0.5) * cell_width)
    y = int(top_left_y + (7 - row + 0.5) * cell_height)  # Inverted Y
    return (x, y)

# Helper to draw arrow
def draw_arrow(img, start_square, end_square, color, top_left_x, top_left_y, cell_width, cell_height):
    start = square_center(start_square, top_left_x, top_left_y, cell_width, cell_height)
    end = square_center(end_square, top_left_x, top_left_y, cell_width, cell_height)
    cv.arrowedLine(img, start, end, color, 3, tipLength=0.2)

# Mapping the class names in Standard FEN Format
piece_map = {
    "black-pawn": 'p', "black-knight": 'n', "black-bishop": 'b', "black-rock": 'r', "black-queen": 'q', "black-king": 'k',
    "white-pawn": 'P', "white-knight": 'N', "white-bishop": 'B', "white-rock": 'R', "white-queen": 'Q', "white-king": 'K'
}

print("Press 's' to capture the screen and run Chess Advisor.\nPress 'q' to quit.")
with mss.mss() as sct:
    while True:
        if keyboard.is_pressed('s'):
            print("Capturing screen...")
            screenshot = sct.grab(monitor)
            frame = np.array(screenshot)
            frame = cv.cvtColor(frame, cv.COLOR_BGRA2BGR)

            # Getting the results of piece detection
            results = model(frame)[0]

            # Prepare Annotators
            detections = sv.Detections.from_ultralytics(results)

            bounding_box_annotator = sv.BoxAnnotator()
            label_annotator = sv.LabelAnnotator()

            annotated_image = bounding_box_annotator.annotate(scene=frame.copy(),detections=detections)
            annotated_image = label_annotator.annotate(scene=annotated_image,detections=detections)

            # Detecting Board
            board_box = None
            for box in results.boxes:
                cls_id = int(box.cls[0])
                if class_names[cls_id] == "board":
                    board_box = box
                    break

            if board_box is None:
                raise ValueError("No board detected in the image")

            cx, cy, w, h = board_box.xywh[0].numpy()
            top_left_x = cx - (w / 2)
            top_left_y = cy - (h / 2)
            cell_width = w / 8
            cell_height = h / 8

            # Initialize a 8x8 board matrix representing an empty chess board
            fen_board = np.full((8, 8), "", dtype=object)

            # Assign each piece to closest cell center
            for box in results.boxes:
                cls_id = int(box.cls[0])
                class_name = class_names[cls_id]
                if class_name == "board":
                    continue

                cx, cy, _, _ = box.xywh[0].cpu().numpy()
                rel_x = cx - top_left_x
                rel_y = cy - top_left_y
                col = int(rel_x / cell_width)
                row = int(rel_y / cell_height)

                # clamping to the board
                col = min(max(col, 0),7)
                row = min(max(row, 0),7)

                if class_name in piece_map:
                    fen_board[row][col] = piece_map[class_name]

                square = f"{chr(97 + col)}{8 - row}"
                cv.putText(annotated_image, square, (int(cx - 10), int(cy - 10)), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            # Convert FEN Board Matrix to FEN Notation

            def board_to_fen(fen_board):
                fen_rows = [] # This list will have the fen strings of all the rows

                # Go through each row and create a string in FEN Notation and put it in fen_rows
                for row in fen_board:
                    fen_row = "" # String containg the FEN Notation of a particular row
                    empty_count = 0 # A counter of number of empty cells on board

                    for cell in row:
                        if cell == "":
                            empty_count += 1
                        else:
                            if empty_count > 0:
                                fen_row += str(empty_count)
                                empty_count = 0
                            fen_row += cell

                    # Adding any remaining empty cells at the end
                    if empty_count > 0:
                        fen_row += str(empty_count)
                    
                    fen_rows.append(fen_row)
                
                # Join all the strings in fen_rows seperated by '/'
                board_position = "/".join(fen_rows)

                # Add Game state: active color, castling, en passant, halfmove, fullmove
                # Assuming: White to move, no castling rights, no en passant, 0 halfmoves, 1 fullmove

                fen = f"{board_position} w - - 0 1"
                return fen

            # Generate the FEN string for our board
            fen_string = board_to_fen(fen_board)
            print(f"\nGenerated FEN: {fen_string}")

            try:
                board = chess.Board(fen_string)
                print(f"FEN Validation SUCCESS")
                print(f"Board is valid: {board.is_valid()}")
                print(f"Board in ASCII:\n{board}")

                # Engine Analysis
                engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)

                # Get Multiple Best moves
                info = engine.analyse(board, chess.engine.Limit(time=0.5),multipv=3)

                colors = [(0,0,255),(255,0,0),(0,255,0)]

                if isinstance(info, list):
                    for i, line in enumerate(info):
                        if i < 3:
                            move = line['pv'][0]
                            draw_arrow(annotated_image, move.from_square, move.to_square,colors[i])
                            print(f"Move {i+1}: {move} (Score: {line['score']})")
                else:
                    move = info['pv'][0]
                    draw_arrow(annotated_image, move.from_square, move.to_square, colors[0])
                    print(f"Best Move: {move} (Score: {info['score']})")
                
                engine.quit()

            except Exception as e:
                print(f"FEN Validation: FAILED - {e}")
                print("Please Check the piece detection and positioning")
            
            cv.imshow("Chess Advisor", annotated_image)
            cv.waitKey(0)
            cv.destroyAllWindows()

            print("")
            print("------------------------------------")
            print("")

        elif keyboard.is_pressed('q'):
            print("Thanks for trying....")
            break
