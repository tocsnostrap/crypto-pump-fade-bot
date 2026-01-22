import express, { type Request, Response, NextFunction } from "express";
import { registerRoutes } from "./routes";
import { serveStatic } from "./static";
import { createServer } from "http";
import { spawn, ChildProcess } from "child_process";
import * as path from "path";

const app = express();
const httpServer = createServer(app);

// Python bot process management
let pythonProcess: ChildProcess | null = null;
let pythonRestartCount = 0;
const MAX_RESTART_DELAY = 60000; // Max 1 minute between restarts

function startPythonBot() {
  // Skip if bot is managed externally (e.g., by start.sh or separate workflow)
  if (process.env.BOT_MANAGED === "1") {
    console.log(`[python-bot] Bot managed externally (BOT_MANAGED=1), skipping spawn`);
    console.log(`[python-bot] To run the bot from Express, unset BOT_MANAGED or use 'npm run dev'`);
    return;
  }
  
  const mainPyPath = path.join(process.cwd(), "main.py");
  
  // Try python3 first (common on Linux/Replit), fallback to python
  const pythonCmd = process.platform === "win32" ? "python" : "python3";
  
  console.log(`[python-bot] Starting Python trading bot with ${pythonCmd}...`);
  
  pythonProcess = spawn(pythonCmd, [mainPyPath], {
    cwd: process.cwd(),
    env: process.env,
    stdio: ["ignore", "pipe", "pipe"],
  });

  pythonProcess.stdout?.on("data", (data) => {
    const lines = data.toString().split("\n").filter((l: string) => l.trim());
    lines.forEach((line: string) => console.log(`[python-bot] ${line}`));
  });

  pythonProcess.stderr?.on("data", (data) => {
    const lines = data.toString().split("\n").filter((l: string) => l.trim());
    lines.forEach((line: string) => console.error(`[python-bot] ERROR: ${line}`));
  });

  pythonProcess.on("exit", (code, signal) => {
    console.log(`[python-bot] Process exited with code ${code}, signal ${signal}`);
    pythonProcess = null;
    
    // Auto-restart with exponential backoff
    const delay = Math.min(5000 * Math.pow(2, pythonRestartCount), MAX_RESTART_DELAY);
    pythonRestartCount++;
    
    console.log(`[python-bot] Restarting in ${delay / 1000} seconds... (attempt ${pythonRestartCount})`);
    setTimeout(() => {
      startPythonBot();
    }, delay);
  });

  pythonProcess.on("error", (err) => {
    console.error(`[python-bot] Failed to start with ${pythonCmd}: ${err.message}`);
    
    // Try fallback to 'python' if 'python3' failed
    if (pythonCmd === "python3") {
      console.log(`[python-bot] Trying fallback to 'python'...`);
      pythonProcess = spawn("python", [mainPyPath], {
        cwd: process.cwd(),
        env: process.env,
        stdio: ["ignore", "pipe", "pipe"],
      });
      
      pythonProcess.stdout?.on("data", (data) => {
        const lines = data.toString().split("\n").filter((l: string) => l.trim());
        lines.forEach((line: string) => console.log(`[python-bot] ${line}`));
      });

      pythonProcess.stderr?.on("data", (data) => {
        const lines = data.toString().split("\n").filter((l: string) => l.trim());
        lines.forEach((line: string) => console.error(`[python-bot] ERROR: ${line}`));
      });
    }
  });

  // Reset restart count after successful run of 5 minutes
  setTimeout(() => {
    if (pythonProcess) {
      pythonRestartCount = 0;
    }
  }, 300000);
}

// Cleanup on exit
process.on("SIGINT", () => {
  console.log("[python-bot] Shutting down...");
  if (pythonProcess) {
    pythonProcess.kill("SIGTERM");
  }
  process.exit(0);
});

process.on("SIGTERM", () => {
  console.log("[python-bot] Shutting down...");
  if (pythonProcess) {
    pythonProcess.kill("SIGTERM");
  }
  process.exit(0);
});

declare module "http" {
  interface IncomingMessage {
    rawBody: unknown;
  }
}

app.use(
  express.json({
    verify: (req, _res, buf) => {
      req.rawBody = buf;
    },
  }),
);

app.use(express.urlencoded({ extended: false }));

export function log(message: string, source = "express") {
  const formattedTime = new Date().toLocaleTimeString("en-US", {
    hour: "numeric",
    minute: "2-digit",
    second: "2-digit",
    hour12: true,
  });

  console.log(`${formattedTime} [${source}] ${message}`);
}

app.use((req, res, next) => {
  const start = Date.now();
  const path = req.path;
  let capturedJsonResponse: Record<string, any> | undefined = undefined;

  const originalResJson = res.json;
  res.json = function (bodyJson, ...args) {
    capturedJsonResponse = bodyJson;
    return originalResJson.apply(res, [bodyJson, ...args]);
  };

  res.on("finish", () => {
    const duration = Date.now() - start;
    if (path.startsWith("/api")) {
      let logLine = `${req.method} ${path} ${res.statusCode} in ${duration}ms`;
      if (capturedJsonResponse) {
        logLine += ` :: ${JSON.stringify(capturedJsonResponse)}`;
      }

      log(logLine);
    }
  });

  next();
});

(async () => {
  await registerRoutes(httpServer, app);

  app.use((err: any, _req: Request, res: Response, next: NextFunction) => {
    const status = err.status || err.statusCode || 500;
    const message = err.message || "Internal Server Error";

    console.error("Internal Server Error:", err);

    if (res.headersSent) {
      return next(err);
    }

    return res.status(status).json({ message });
  });

  // importantly only setup vite in development and after
  // setting up all the other routes so the catch-all route
  // doesn't interfere with the other routes
  if (process.env.NODE_ENV === "production") {
    serveStatic(app);
  } else {
    const { setupVite } = await import("./vite");
    await setupVite(httpServer, app);
  }

  // ALWAYS serve the app on the port specified in the environment variable PORT
  // Other ports are firewalled. Default to 5000 if not specified.
  // this serves both the API and the client.
  // It is the only port that is not firewalled.
  const port = parseInt(process.env.PORT || "5000", 10);
  httpServer.listen(
    {
      port,
      host: "0.0.0.0",
      reusePort: true,
    },
    () => {
      log(`serving on port ${port}`);
      
      // Start Python trading bot after server is ready
      startPythonBot();
    },
  );
})();
