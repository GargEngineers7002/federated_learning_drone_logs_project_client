# Drone Trajectory Prediction - Client Node (Frontend & Local Inference)

The client node for the drone trajectory prediction federated learning system. 

## Architecture Overview

This is the **Worker Node (Client)** of the federated learning system:
*   **Worker Node (Client):** 
    *   Hosts the web interface (HTML/JS/CSS) for end users.
    *   Accepts drone flight logs (CSV).
    *   Retrieves the latest global model from the Central Hub.
    *   Trains the local model on the newly uploaded flight log.
    *   Performs local inference and returns the predicted trajectory.
    *   Sends updated model weights back to the Central Hub for aggregation in the background.

## Project Structure

```text
website_work/
├── app/
│   ├── main.py                 # Client API & Web Server
│   ├── ml_models.py            # Model inference logic
│   ├── fl.py                   # Local training & FL synchronization logic
│   └── template/               # Frontend (HTML/JS/CSS)
└── models/                     # Local .keras models and scalers
```

## Setup & Installation

1.  **Environment Setup:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Configure Backend Connection:**
    Ensure you have a `.env` file specifying the Central Hub's URL:
    ```
    BACKEND_SERVER_URL=http://localhost:8000
    ```

3.  **Start the Client Server:**
    The client starts the Web API and UI on port 8001.
    ```bash
    export PYTHONPATH=$PYTHONPATH:.
    python website_work/app/main.py
    ```

## Usage Workflow

1.  **User Upload:** An end user uploads a drone flight log (CSV) via the web interface (`localhost:8001`).
2.  **Global Sync:** The client fetches the latest global model weights from the Central Hub (`localhost:8000`) and updates its local model.
3.  **Local Training:** The client briefly trains the model on the new data sequence.
4.  **Inference:** The client predicts the trajectory using the locally updated model and displays it to the user.
5.  **Federated Learning:** In the background, the client sends its updated weights back to the Central Hub for aggregation.
