
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
    return JSONResponse(content={"running": True})


@app.post("/move")
async def get_move(request: Request):
    start_time = time.perf_counter()
    try:
        data = await request.json()
    except Exception as e:
        time_taken = (time.perf_counter() - start_time) * 1000
        return JSONResponse(
            content={
                "move": None,
                "move_probs": None,
                "time_taken": time_taken,
                "error": f"Invalid JSON: {str(e)}",
                "logs": None,
                "exception": str(e),
            },
            status_code=400,
        )

    if ("pgn" not in data or "timeleft" not in data):
        time_taken = (time.perf_counter() - start_time) * 1000
        return JSONResponse(
            content={
                "move": None,
                "move_probs": None,
                "time_taken": time_taken,
                "error": "Missing pgn or timeleft",
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

    # Confirm type of move_probs
    if not isinstance(move_probs, dict):
        return JSONResponse(content={"move": None, "move_probs": None, "error": "Failed to get move", "message": "Move probabilities is not a dictionary"}, status_code=500)

    for m, prob in move_probs.items():
        if not isinstance(m, chess.Move) or not isinstance(prob, float):
            return JSONResponse(content={m: None, "move_probs": None, "error": "Failed to get move", "message": "Move probabilities is not a dictionary"}, status_code=500)

    # Translate move_probs to Dict[str, float]
    move_probs_dict = {move.uci(): prob for move, prob in move_probs.items()}

    return JSONResponse(content={"move": move.uci(), "error": None, "time_taken": time_taken, "move_probs": move_probs_dict, "logs": logs})

if __name__ == "__main__":
    port = int(os.getenv("SERVE_PORT", "5058"))
    uvicorn.run(app, host="0.0.0.0", port=port)
