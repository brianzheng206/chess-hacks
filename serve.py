from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
import time
import chess
import os
import threading

from src.utils import chess_manager

app = FastAPI()

# Import main in background thread to avoid blocking server startup
# The health check endpoint should respond immediately
main_module = None
model_loading_thread = None

def load_main_module():
    """Load the main module in a background thread."""
    global main_module
    try:
        from src import main
        main_module = main
        print("Main module loaded successfully in background thread")
    except Exception as e:
        print(f"Error loading main module in background: {e}")
        import traceback
        traceback.print_exc()

# Start loading main module in background
if model_loading_thread is None:
    model_loading_thread = threading.Thread(target=load_main_module, daemon=True)
    model_loading_thread.start()

@app.get("/")
async def root_get():
    """Health check endpoint for server readiness."""
    return JSONResponse(content={"running": True, "status": "ready"})

@app.post("/")
async def root():
    """Health check endpoint for server readiness (POST)."""
    return JSONResponse(content={"running": True, "status": "ready"})

@app.get("/health")
async def health():
    """Explicit health check endpoint."""
    return JSONResponse(content={"running": True, "status": "ready"})


@app.post("/move")
async def get_move(request: Request):
    start_time = time.perf_counter()
    try:
        # Try to get JSON body, but handle empty requests gracefully
        try:
            data = await request.json()
        except Exception as json_error:
            # If request has no body or invalid JSON, check if it's a health check
            # Some platforms send empty POST requests as health checks
            content_type = request.headers.get("content-type", "")
            print(f"DEBUG: /move endpoint received request with content-type: {content_type}, error: {json_error}")
            if "application/json" not in content_type.lower():
                # Not a JSON request - might be a health check
                print("DEBUG: Treating as health check (no JSON content-type)")
                return JSONResponse(
                    content={
                        "running": True,
                        "status": "ready",
                        "error": None,
                    },
                    status_code=200,
                )
            # Otherwise, it's a real error
            time_taken = (time.perf_counter() - start_time) * 1000
            print(f"DEBUG: Returning 400 for invalid JSON: {json_error}")
            return JSONResponse(
                content={
                    "move": None,
                    "move_probs": None,
                    "time_taken": time_taken,
                    "error": f"Invalid JSON: {str(json_error)}",
                    "logs": None,
                    "exception": str(json_error),
                },
                status_code=400,
            )
    except Exception as e:
        time_taken = (time.perf_counter() - start_time) * 1000
        return JSONResponse(
            content={
                "move": None,
                "move_probs": None,
                "time_taken": time_taken,
                "error": f"Request error: {str(e)}",
                "logs": None,
                "exception": str(e),
            },
            status_code=400,
        )

    # Handle case where data might be None or empty
    if not data:
        return JSONResponse(
            content={
                "running": True,
                "status": "ready",
                "error": None,
            },
            status_code=200,
        )

    if ("pgn" not in data or "timeleft" not in data):
        time_taken = (time.perf_counter() - start_time) * 1000
        missing_fields = []
        if "pgn" not in data:
            missing_fields.append("pgn")
        if "timeleft" not in data:
            missing_fields.append("timeleft")
        print(f"DEBUG: Missing required fields: {missing_fields}, received keys: {list(data.keys()) if data else 'None'}")
        return JSONResponse(
            content={
                "move": None,
                "move_probs": None,
                "time_taken": time_taken,
                "error": f"Missing required fields: {', '.join(missing_fields)}. Received keys: {list(data.keys()) if data else 'None'}",
                "logs": None,
            },
            status_code=400,
        )

    pgn = data["pgn"]
    timeleft = data["timeleft"]  # in milliseconds

    chess_manager.set_context(pgn, timeleft)
    print("pgn", pgn)

    # Wait for main module to be loaded if it's still loading
    if main_module is None:
        if model_loading_thread and model_loading_thread.is_alive():
            print("Waiting for main module to load...")
            model_loading_thread.join(timeout=30)  # Wait up to 30 seconds
        if main_module is None:
            time_taken = (time.perf_counter() - start_time) * 1000
            return JSONResponse(
                content={
                    "move": None,
                    "move_probs": None,
                    "time_taken": time_taken,
                    "error": "Model is still loading, please try again in a moment",
                    "logs": None,
                },
                status_code=503,
            )

    try:
        move, move_probs, logs = chess_manager.get_model_move()
        end_time = time.perf_counter()
        time_taken = (end_time - start_time) * 1000
    except Exception as e:
        time_taken = (time.perf_counter() - start_time) * 1000
        print(f"Error in get_model_move: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            content={
                "move": None,
                "move_probs": None,
                "time_taken": time_taken,
                "error": "Bot raised an exception",
                "logs": None,
                "exception": str(e),
            },
            status_code=500,
        )

    # Confirm type of move_probs and validate
    if not isinstance(move_probs, dict):
        print(f"ERROR: move_probs is not a dict, type: {type(move_probs)}, value: {move_probs}")
        # Fallback: create a simple probability dict
        if move is not None:
            move_probs = {move: 1.0}
        else:
            return JSONResponse(content={"move": None, "move_probs": None, "error": "Failed to get move", "message": "Move probabilities is not a dictionary and no move available"}, status_code=500)
    
    # If move_probs is empty, create a fallback
    if not move_probs and move is not None:
        print(f"WARNING: move_probs is empty, creating fallback for move {move.uci()}")
        move_probs = {move: 1.0}
    
    # Validate move_probs contents
    validated_move_probs = {}
    for m, prob in move_probs.items():
        if isinstance(m, chess.Move) and isinstance(prob, (float, int)):
            validated_move_probs[m] = float(prob)
        else:
            print(f"WARNING: Invalid entry in move_probs: move={m} (type: {type(m)}), prob={prob} (type: {type(prob)})")
    
    # If validation removed all entries but we have a move, add it
    if not validated_move_probs and move is not None:
        print(f"WARNING: All move_probs entries were invalid, using fallback for move {move.uci()}")
        validated_move_probs = {move: 1.0}
    
    # If still no valid move_probs, return error
    if not validated_move_probs:
        return JSONResponse(content={"move": None, "move_probs": None, "error": "Failed to get move", "message": "No valid move probabilities available"}, status_code=500)
    
    move_probs = validated_move_probs

    # Translate move_probs to Dict[str, float]
    move_probs_dict = {move.uci(): prob for move, prob in move_probs.items()}

    return JSONResponse(content={"move": move.uci(), "error": None, "time_taken": time_taken, "move_probs": move_probs_dict, "logs": logs})

if __name__ == "__main__":
    port = int(os.getenv("SERVE_PORT", "5058"))
    uvicorn.run(app, host="0.0.0.0", port=port)
