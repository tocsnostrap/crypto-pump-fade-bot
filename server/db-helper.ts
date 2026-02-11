import pg from "pg";
import fs from "fs";
import path from "path";

const pool = new pg.Pool({
  connectionString: process.env.DATABASE_URL,
});

export async function queryDB(query: string, params: any[] = []): Promise<any[]> {
  const client = await pool.connect();
  try {
    const result = await client.query(query, params);
    return result.rows;
  } finally {
    client.release();
  }
}

export async function getState(key: string): Promise<any | null> {
  try {
    const rows = await queryDB("SELECT data FROM bot_state WHERE key = $1", [key]);
    if (rows.length > 0) return rows[0].data;
    return null;
  } catch {
    return null;
  }
}

export async function setState(key: string, data: any): Promise<void> {
  try {
    await queryDB(
      `INSERT INTO bot_state (key, data, updated_at)
       VALUES ($1, $2, NOW())
       ON CONFLICT (key) DO UPDATE SET data = EXCLUDED.data, updated_at = NOW()`,
      [key, JSON.stringify(data)]
    );
  } catch (err) {
    console.error(`[db-helper] setState error (${key}):`, err);
  }
}

export async function getClosedTrades(): Promise<any[]> {
  try {
    const rows = await queryDB("SELECT data FROM closed_trades ORDER BY closed_at ASC");
    return rows.map((r: any) => r.data);
  } catch {
    return [];
  }
}

export async function getSignals(limit = 100): Promise<any[]> {
  try {
    const rows = await queryDB("SELECT data FROM signals ORDER BY timestamp DESC LIMIT $1", [limit]);
    return rows.map((r: any) => r.data).reverse();
  } catch {
    return [];
  }
}

export async function getTradeFeatures(limit = 1000): Promise<any[]> {
  try {
    const rows = await queryDB("SELECT data FROM trade_features ORDER BY timestamp DESC LIMIT $1", [limit]);
    return rows.map((r: any) => r.data).reverse();
  } catch {
    return [];
  }
}

export async function getTradeJournal(): Promise<any[]> {
  try {
    const rows = await queryDB("SELECT data FROM trade_journal ORDER BY timestamp ASC");
    return rows.map((r: any) => r.data);
  } catch {
    return [];
  }
}

function writeJsonFileSafe(filePath: string, data: any): void {
  try {
    const dir = path.dirname(filePath);
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
    const tmpPath = filePath + ".tmp";
    fs.writeFileSync(tmpPath, JSON.stringify(data, null, 2));
    fs.renameSync(tmpPath, filePath);
  } catch (err) {
    console.error(`[db-helper] writeJsonFileSafe error (${filePath}):`, err);
  }
}

export async function restoreJsonFromDb(): Promise<void> {
  console.log("[db-helper] Checking if JSON files need restoration from DB...");
  let restored = 0;

  const balanceFile = "balance.json";
  if (!fs.existsSync(balanceFile)) {
    const data = await getState("balance");
    if (data && data.balance) {
      writeJsonFileSafe(balanceFile, data);
      console.log(`  Restored ${balanceFile} from DB (balance: ${data.balance})`);
      restored++;
    }
  }

  const closedTradesFile = "closed_trades.json";
  if (!fs.existsSync(closedTradesFile)) {
    const trades = await getClosedTrades();
    if (trades.length > 0) {
      writeJsonFileSafe(closedTradesFile, trades);
      console.log(`  Restored ${closedTradesFile} from DB (${trades.length} trades)`);
      restored++;
    }
  }

  const tradesLogFile = "trades_log.json";
  if (!fs.existsSync(tradesLogFile)) {
    const openTrades = await getState("open_trades");
    if (openTrades && Array.isArray(openTrades) && openTrades.length > 0) {
      writeJsonFileSafe(tradesLogFile, openTrades);
      console.log(`  Restored ${tradesLogFile} from DB (${openTrades.length} open trades)`);
      restored++;
    } else {
      writeJsonFileSafe(tradesLogFile, []);
    }
  }

  const signalsFile = "signals.json";
  if (!fs.existsSync(signalsFile)) {
    const signals = await getSignals(200);
    if (signals.length > 0) {
      writeJsonFileSafe(signalsFile, signals);
      console.log(`  Restored ${signalsFile} from DB (${signals.length} signals)`);
      restored++;
    }
  }

  const pumpStateFile = "pump_state.json";
  if (!fs.existsSync(pumpStateFile)) {
    const pumpState = await getState("pump_state");
    if (pumpState && Object.keys(pumpState).length > 0) {
      writeJsonFileSafe(pumpStateFile, pumpState);
      console.log(`  Restored ${pumpStateFile} from DB`);
      restored++;
    }
  }

  const tradeJournalFile = "trade_journal.json";
  if (!fs.existsSync(tradeJournalFile)) {
    const journal = await getTradeJournal();
    if (journal.length > 0) {
      writeJsonFileSafe(tradeJournalFile, journal);
      console.log(`  Restored ${tradeJournalFile} from DB (${journal.length} entries)`);
      restored++;
    }
  }

  const tradeFeaturesFile = "trade_features.json";
  if (!fs.existsSync(tradeFeaturesFile)) {
    const features = await getTradeFeatures(2000);
    if (features.length > 0) {
      writeJsonFileSafe(tradeFeaturesFile, features);
      console.log(`  Restored ${tradeFeaturesFile} from DB (${features.length} entries)`);
      restored++;
    }
  }

  const learningStateFile = "learning_state.json";
  if (!fs.existsSync(learningStateFile)) {
    const learningState = await getState("learning_state");
    if (learningState) {
      writeJsonFileSafe(learningStateFile, learningState);
      console.log(`  Restored ${learningStateFile} from DB`);
      restored++;
    }
  }

  const balanceHistoryFile = "balance_history.json";
  if (!fs.existsSync(balanceHistoryFile)) {
    const history = await getState("balance_history");
    if (history && Array.isArray(history) && history.length > 0) {
      writeJsonFileSafe(balanceHistoryFile, history);
      console.log(`  Restored ${balanceHistoryFile} from DB (${history.length} entries)`);
      restored++;
    }
  }

  if (restored === 0) {
    console.log("[db-helper] All JSON files present, no restoration needed");
  } else {
    console.log(`[db-helper] Restored ${restored} JSON files from DB`);
  }
}
