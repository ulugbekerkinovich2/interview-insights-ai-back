module.exports = {
  apps: [
    {
      name: "ai-interview-backend",
      script: "venv/bin/uvicorn",
      args: "main:app --host 0.0.0.0 --port 8000",
      cwd: "/var/www/workers/mentalaba/interview-insights-ai-back",
      interpreter: "none",
      watch: false,
      env: {
        NODE_ENV: "production",
        PYTHONUNBUFFERED: "1"
      },
    },
  ],
};
