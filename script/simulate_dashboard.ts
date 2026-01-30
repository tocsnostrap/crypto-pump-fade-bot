import express from "express";
import { createServer } from "http";
import { io as socketClient } from "socket.io-client";
import type { AddressInfo } from "net";
import { registerRoutes } from "../server/routes.ts";

async function runSimulation() {
  const app = express();
  app.use(express.json());
  app.use(express.urlencoded({ extended: false }));

  const httpServer = createServer(app);
  await registerRoutes(httpServer, app);

  await new Promise<void>((resolve) => {
    httpServer.listen(0, "127.0.0.1", resolve);
  });

  const address = httpServer.address() as AddressInfo;
  const baseUrl = `http://127.0.0.1:${address.port}`;

  console.log(`[simulate] server listening on ${baseUrl}`);

  const socket = socketClient(baseUrl, {
    transports: ["websocket", "polling"],
  });

  let receivedSignal = false;
  let receivedCloseTrade = false;

  const finish = (success: boolean) => {
    socket.disconnect();
    httpServer.close(() => {
      process.exit(success ? 0 : 1);
    });
  };

  const timeoutId = setTimeout(() => {
    console.error("[simulate] timeout waiting for socket events");
    finish(false);
  }, 3000);

  const checkDone = () => {
    if (receivedSignal && receivedCloseTrade) {
      clearTimeout(timeoutId);
      console.log("[simulate] received live signal + close trade events");
      finish(true);
    }
  };

  socket.on("connect", async () => {
    console.log("[simulate] socket connected");
    socket.emit("new_signal", {
      id: "test_signal_1",
      exchange: "gate",
      symbol: "BTC/USDT",
      type: "fade_watch",
      price: 42000,
      change_pct: 90,
      timestamp: new Date().toISOString(),
      message: "Test signal broadcast",
    });

    try {
      const response = await fetch(`${baseUrl}/api/close_trade/TEST-123`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ exchange: "gate", symbol: "BTC/USDT" }),
      });
      console.log(`[simulate] close_trade POST status ${response.status}`);
    } catch (error) {
      console.error("[simulate] close_trade POST failed", error);
      clearTimeout(timeoutId);
      finish(false);
    }
  });

  socket.on("new_signal", (payload) => {
    console.log("[simulate] received new_signal", payload?.id);
    receivedSignal = true;
    checkDone();
  });

  socket.on("close_trade", (payload) => {
    console.log("[simulate] received close_trade", payload?.id);
    receivedCloseTrade = true;
    checkDone();
  });

  socket.on("connect_error", (error) => {
    console.error("[simulate] socket connect error", error);
    clearTimeout(timeoutId);
    finish(false);
  });
}

runSimulation();
