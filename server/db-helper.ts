import pg from "pg";

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
