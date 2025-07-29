# SQL-JSON Frontend

This is a React-based frontend for the SQL-JSON Chat application.

## Setup

1. Install dependencies:
   ```bash
   npm install
   ```

2. Start the development server:
   ```bash
   npm start
   ```

The app will run at [http://localhost:3000](http://localhost:3000) by default.

## Usage
- Enter your natural language query in the input box.
- The chat will display your query and the response from the backend.

## Backend
Make sure the FastAPI backend is running at `http://localhost:8000` (default) for the chat to work.

## Deployment (Vercel)
- When deploying to Vercel, set the **Root Directory** to `frontend` in your Vercel project settings.
- The build command should be `npm install && npm run build` and the output directory should be `build`. 