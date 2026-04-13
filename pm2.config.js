module.exports = {
  apps: [
    {
      name: "ai-interview-backend",
      script: "../venv/bin/python",
      args: "main.py",
      cwd: "./",
      watch: false,
      env: {
        NODE_ENV: "production",
        PYTHONUNBUFFERED: "1"
      },
    },
  ],
};
