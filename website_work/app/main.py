import os
import uuid
import httpx
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from typing import Annotated
from dotenv import load_dotenv
from website_work.app.fl import (
    process_job,
    train_and_save_model,
    save_global_model_weights,
)

# Load environment variables from .env file
load_dotenv()

app = FastAPI(title="UAV Trajectory Prediction Client Server")

# Get backend URL from .env, default to localhost:8000
BACKEND_SERVER_URL = os.getenv("BACKEND_SERVER_URL", "http://localhost:8000")


@app.post("/api/predict_trajectory")
async def predict_trajectory(
    uav_model: Annotated[str, Form()],
    flight_log: Annotated[UploadFile, File()],
    background_tasks: BackgroundTasks,
):
    print(f"\n[USER] New prediction request received for model: {uav_model}")
    if not flight_log.filename or not flight_log.filename.lower().endswith(".csv"):
        print("[USER] Error: Invalid file type uploaded.")
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")

    try:
        # 1. Read CSV as string to send to node
        contents = await flight_log.read()
        csv_str = contents.decode("utf-8")

        print("Data decoded")

        job_id = str(uuid.uuid4())  # Generate a unique job ID

        print("Getting the global model")
        # get the global model from the backend and save the model to the disk
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{BACKEND_SERVER_URL}/api/get_global",
                    params={"uav_model": uav_model},
                    timeout=10.0,  # Shorter timeout for better UX if backend is down
                )
                if resp.status_code == 200:
                    global_data = resp.json()
                    await save_global_model_weights(uav_model, global_data["weights"])
                    print("[CLIENT] Successfully synced with global model.")
                else:
                    print(
                        f"[CLIENT] Failed to fetch global model (Status {resp.status_code}): {resp.text}"
                    )
        except (httpx.ConnectError, httpx.TimeoutException) as conn_err:
            print(
                f"[CLIENT] Warning: Could not connect to Central Hub at {BACKEND_SERVER_URL}. Proceeding with local model. ({conn_err})"
            )
        except Exception as sync_err:
            print(f"[CLIENT] Error during global model sync: {sync_err}")

        print("Starting training the model")

        # Train and save BEFORE prediction
        new_weights = await train_and_save_model(uav_model, csv_str)

        print("Starting prediction")

        # Run prediction on the newly updated model
        results = await process_job(job_id, uav_model, csv_str)

        print(f"[HUB] Returning results for Job {job_id} to user.")

        print("Returning weights")
        # Send the federated averaging update to the background AFTER returning
        if new_weights is not None:
            from website_work.app.fl import send_weights_to_backend

            background_tasks.add_task(send_weights_to_backend, uav_model, new_weights)

        return {
            "uav_model": uav_model,
            "results": results,
            "job_id": job_id,
        }

    except Exception as e:
        import traceback

        error_details = traceback.format_exc()
        print(f"[HUB] ERROR in predict_trajectory: {e}")
        print(f"[HUB] TRACEBACK:\n{error_details}")

        error_msg = str(e) if str(e) else repr(e)
        raise HTTPException(status_code=500, detail=f"Server Error: {error_msg}")


# Mount static files (Frontend HTML/JS/CSS)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.mount(
    "/",
    StaticFiles(directory=os.path.join(BASE_DIR, "template"), html=True),
    name="static",
)

if __name__ == "__main__":
    import uvicorn

    print("\n🔍 Launching Client Uvicorn server ")
    print(f"🔗 Connected to Backend at: {BACKEND_SERVER_URL}")
    uvicorn.run(app, host="0.0.0.0", port=8001)
