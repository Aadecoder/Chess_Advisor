from inference import get_model
import cv2 as cv
import supervision as sv
import numpy as np
import chess
import chess.engine

# Import the YOLO Model for Chess Piece Detection
model = get_model(model_id="chess.comdetection/4", api_key="API_KEY")

# Image of the chess Board
image = cv.imread(r"C:\Users\adity\OneDrive\Pictures\Screenshots\Screenshot 2025-06-22 030143.png")

# Getting the results of piece detection
results = model.infer(image=image)

# Prepare Annotators
detections = sv.Detections.from_inference(results[0])

bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

annotated_image = bounding_box_annotator.annotate(scene=image.copy(),detections=detections)
annotated_image = label_annotator.annotate(scene=annotated_image,detections=detections)

# Detecting Board
board_detection = next(pred for pred in results[0].predictions if pred.class_name == "board")

board_x, board_y = board_detection.x, board_detection.y
board_w, board_h = board_detection.width, board_detection.height

top_left_x = board_x - (board_w / 2)
top_left_y = board_y - (board_h / 2)

cell_width = board_w / 8
cell_height = board_h / 8

# Initialize a 8x8 board matrix representing an empty chess board
fen_board = np.full((8, 8), "", dtype=object)

# Mapping the class names in Standard FEN Format
piece_map = {
    "black-pawn":'p', "black-knight" : 'n', "black-bishop" : 'b', "black-rock" : 'r', "black-queen" : 'q', "black-king": 'k',
    "white-pawn":'P', "white-knight" : 'N', "white-bishop" : 'B', "white-rock" : 'R', "white-queen" : 'Q', "white-king": 'K'
}

# Assign each piece to closest cell center
detected_pieces = []
for pred in results[0].predictions:
    if pred.class_name == "board":
        continue
    
    # Center coordinates of the detected piece
    piece_cx, piece_cy = pred.x, pred.y

    # Relative position of the piece with the board
    rel_x = piece_cx - top_left_x # Distance from the left corner of the board
    rel_y = piece_cy - top_left_y # Distance from the top corner of the board

    # Convert to Grid Coordinates
    col = int(rel_x / cell_width) 
    row = int(rel_y / cell_height)

    # Clamp to Board Limits
    col = min(max(col, 0), 7)
    row = min(max(row, 0), 7)

    # Actual Position in chess board in chess notation
    chess_square = f"{chr(97 + col)}{8 - row}" # for example for (0,0) it gives a8

    #store information of pieces
    detected_pieces.append({
        'piece': pred.class_name,
        'image_pos': (piece_cx, piece_cy),
        'grid_pos':(row,col),
        'chess_square': chess_square,
        'confidence': pred.confidence        
    })

    # Place pieces in FEN Board matrix
    if pred.class_name in piece_map:
        fen_board[row][col] = piece_map[pred.class_name]

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
    engine_path = "C:/Users/adity/Downloads/stockfish/stockfish-windows-x86-64-avx2.exe"
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)

    # Get Multiple Best moves
    info = engine.analyse(board, chess.engine.Limit(time=0.5),multipv=3)

    # Function to get square center coordinates
    def square_center(square):
        col = chess.square_file(square)
        row = chess.square_rank(square)

        # Converting these chess coordinates to image coordinates
        x = int(top_left_x + (col + 0.5) * cell_width)
        y = int(top_left_y + (7 - row + 0.5) * cell_height)
        return (x,y)
    
    # Function to Draw Arrows for best moves
    def draw_arrow(img, start_square, end_square, color, thickness=3):
        start = square_center(start_square)
        end = square_center(end_square)
        cv.arrowedLine(img,start,end,color,thickness,tipLength=0.1)

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

# Display Result
sv.plot_image(annotated_image)
cv.waitKey(0)
cv.destroyAllWindows()
