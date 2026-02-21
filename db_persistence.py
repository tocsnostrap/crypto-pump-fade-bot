import os
import json
import psycopg2
import psycopg2.extras
from datetime import datetime

DATABASE_URL = os.environ.get('DATABASE_URL', '')

_pool = None

def _get_pool():
    global _pool
    if _pool is None or _pool.closed:
        try:
            from psycopg2.pool import SimpleConnectionPool
            _pool = SimpleConnectionPool(1, 10, DATABASE_URL)
        except Exception:
            _pool = None
    return _pool

def get_connection():
    pool = _get_pool()
    if pool:
        try:
            return pool.getconn()
        except Exception:
            pass
    return psycopg2.connect(DATABASE_URL)

def _return_connection(conn):
    pool = _get_pool()
    if pool and conn and not conn.closed:
        try:
            pool.putconn(conn)
            return
        except Exception:
            pass
    if conn and not conn.closed:
        try:
            _return_connection(conn)
        except Exception:
            pass

def init_tables():
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS bot_state (
                    key TEXT PRIMARY KEY,
                    data JSONB NOT NULL,
                    updated_at TIMESTAMP DEFAULT NOW()
                );
                CREATE TABLE IF NOT EXISTS closed_trades (
                    id SERIAL PRIMARY KEY,
                    ex TEXT,
                    sym TEXT,
                    entry_price DOUBLE PRECISION,
                    exit_price DOUBLE PRECISION,
                    profit DOUBLE PRECISION,
                    reason TEXT,
                    closed_at TIMESTAMP,
                    data JSONB NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_closed_trades_closed_at ON closed_trades(closed_at);
                CREATE TABLE IF NOT EXISTS signals (
                    id TEXT PRIMARY KEY,
                    exchange TEXT,
                    symbol TEXT,
                    type TEXT,
                    price DOUBLE PRECISION,
                    change_pct DOUBLE PRECISION,
                    funding_rate DOUBLE PRECISION,
                    rsi DOUBLE PRECISION,
                    timestamp TIMESTAMP,
                    message TEXT,
                    data JSONB NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp);
                CREATE TABLE IF NOT EXISTS trade_features (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP,
                    symbol TEXT,
                    exchange TEXT,
                    action TEXT,
                    data JSONB NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_trade_features_timestamp ON trade_features(timestamp);
                CREATE TABLE IF NOT EXISTS trade_journal (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP,
                    trade_id TEXT,
                    type TEXT,
                    symbol TEXT,
                    data JSONB NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_trade_journal_timestamp ON trade_journal(timestamp);
            """)
        conn.commit()
        print(f"[{datetime.now()}] DB tables initialized")
    except Exception as e:
        print(f"[{datetime.now()}] DB init error: {e}")
        conn.rollback()
    finally:
        _return_connection(conn)


def save_balance(balance, last_updated=None):
    if last_updated is None:
        last_updated = str(datetime.now())
    data = {'balance': balance, 'last_updated': last_updated}
    _upsert_state('balance', data)

def load_balance(default_balance=5000.0):
    data = _get_state('balance')
    if data:
        return data.get('balance', default_balance)
    return default_balance

def save_pump_state(prev_data):
    _upsert_state('pump_state', prev_data)

def load_pump_state():
    return _get_state('pump_state') or {}

def save_open_trades(open_trades):
    _upsert_state('open_trades', open_trades)

def load_open_trades():
    data = _get_state('open_trades')
    if data is None:
        return []
    if isinstance(data, list):
        return data
    return []


def append_closed_trade(trade_dict):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            closed_at = trade_dict.get('closed_at')
            if isinstance(closed_at, str):
                try:
                    closed_at = datetime.fromisoformat(closed_at)
                except:
                    closed_at = datetime.now()
            cur.execute("""
                INSERT INTO closed_trades (ex, sym, entry_price, exit_price, profit, reason, closed_at, data)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                trade_dict.get('ex'),
                trade_dict.get('sym'),
                trade_dict.get('entry'),
                trade_dict.get('exit'),
                trade_dict.get('profit'),
                trade_dict.get('reason'),
                closed_at,
                json.dumps(trade_dict, default=str)
            ))
        conn.commit()
    except Exception as e:
        print(f"[{datetime.now()}] DB append_closed_trade error: {e}")
        conn.rollback()
    finally:
        _return_connection(conn)

def load_closed_trades():
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT data FROM closed_trades ORDER BY closed_at ASC")
            rows = cur.fetchall()
            return [row[0] for row in rows]
    except Exception as e:
        print(f"[{datetime.now()}] DB load_closed_trades error: {e}")
        return []
    finally:
        _return_connection(conn)


def save_signal(signal_dict):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            ts = signal_dict.get('timestamp')
            if isinstance(ts, str):
                try:
                    ts = datetime.fromisoformat(ts)
                except:
                    ts = datetime.now()
            cur.execute("""
                INSERT INTO signals (id, exchange, symbol, type, price, change_pct, funding_rate, rsi, timestamp, message, data)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET data = EXCLUDED.data, timestamp = EXCLUDED.timestamp
            """, (
                signal_dict.get('id'),
                signal_dict.get('exchange'),
                signal_dict.get('symbol'),
                signal_dict.get('type'),
                signal_dict.get('price'),
                signal_dict.get('change_pct'),
                signal_dict.get('funding_rate'),
                signal_dict.get('rsi'),
                ts,
                signal_dict.get('message'),
                json.dumps(signal_dict, default=str)
            ))
        conn.commit()
    except Exception as e:
        print(f"[{datetime.now()}] DB save_signal error: {e}")
        conn.rollback()
    finally:
        _return_connection(conn)

def load_signals(limit=100):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT data FROM signals ORDER BY timestamp DESC LIMIT %s", (limit,))
            rows = cur.fetchall()
            return [row[0] for row in reversed(rows)]
    except Exception as e:
        print(f"[{datetime.now()}] DB load_signals error: {e}")
        return []
    finally:
        _return_connection(conn)


def append_trade_feature(feature_dict):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            ts = feature_dict.get('timestamp')
            if isinstance(ts, str):
                try:
                    ts = datetime.fromisoformat(ts)
                except:
                    ts = datetime.now()
            cur.execute("""
                INSERT INTO trade_features (timestamp, symbol, exchange, action, data)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                ts,
                feature_dict.get('symbol'),
                feature_dict.get('exchange'),
                feature_dict.get('action'),
                json.dumps(feature_dict, default=str)
            ))
        conn.commit()
    except Exception as e:
        print(f"[{datetime.now()}] DB append_trade_feature error: {e}")
        conn.rollback()
    finally:
        _return_connection(conn)

def load_trade_features(limit=1000):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT data FROM trade_features ORDER BY timestamp DESC LIMIT %s", (limit,))
            rows = cur.fetchall()
            return [row[0] for row in reversed(rows)]
    except Exception as e:
        print(f"[{datetime.now()}] DB load_trade_features error: {e}")
        return []
    finally:
        _return_connection(conn)


def append_trade_journal(journal_entry):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            ts = journal_entry.get('timestamp')
            if isinstance(ts, str):
                try:
                    ts = datetime.fromisoformat(ts)
                except:
                    ts = datetime.now()
            cur.execute("""
                INSERT INTO trade_journal (timestamp, trade_id, type, symbol, data)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                ts,
                journal_entry.get('trade_id'),
                journal_entry.get('type'),
                journal_entry.get('symbol'),
                json.dumps(journal_entry, default=str)
            ))
        conn.commit()
    except Exception as e:
        print(f"[{datetime.now()}] DB append_trade_journal error: {e}")
        conn.rollback()
    finally:
        _return_connection(conn)

def load_trade_journal():
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT data FROM trade_journal ORDER BY timestamp ASC")
            rows = cur.fetchall()
            return [row[0] for row in rows]
    except Exception as e:
        print(f"[{datetime.now()}] DB load_trade_journal error: {e}")
        return []
    finally:
        _return_connection(conn)


def _upsert_state(key, data):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO bot_state (key, data, updated_at)
                VALUES (%s, %s, NOW())
                ON CONFLICT (key) DO UPDATE SET data = EXCLUDED.data, updated_at = NOW()
            """, (key, json.dumps(data, default=str)))
        conn.commit()
    except Exception as e:
        print(f"[{datetime.now()}] DB upsert_state error ({key}): {e}")
        conn.rollback()
    finally:
        _return_connection(conn)

def _get_state(key):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT data FROM bot_state WHERE key = %s", (key,))
            row = cur.fetchone()
            if row:
                return row[0]
            return None
    except Exception as e:
        print(f"[{datetime.now()}] DB get_state error ({key}): {e}")
        return None
    finally:
        _return_connection(conn)


def migrate_json_to_db():
    print(f"[{datetime.now()}] Checking if JSON data needs migration to DB...")

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM closed_trades")
            ct_count = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM signals")
            sig_count = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM trade_features")
            tf_count = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM trade_journal")
            tj_count = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM bot_state")
            bs_count = cur.fetchone()[0]
    finally:
        _return_connection(conn)

    migrated = False

    if bs_count == 0:
        if os.path.exists('balance.json'):
            try:
                with open('balance.json') as f:
                    data = json.load(f)
                save_balance(data.get('balance', 5000.0), data.get('last_updated'))
                print(f"  Migrated balance.json")
                migrated = True
            except Exception as e:
                print(f"  Error migrating balance.json: {e}")

        if os.path.exists('pump_state.json'):
            try:
                with open('pump_state.json') as f:
                    data = json.load(f)
                save_pump_state(data)
                print(f"  Migrated pump_state.json")
                migrated = True
            except Exception as e:
                print(f"  Error migrating pump_state.json: {e}")

        if os.path.exists('trades_log.json'):
            try:
                with open('trades_log.json') as f:
                    data = json.load(f)
                save_open_trades(data)
                print(f"  Migrated trades_log.json ({len(data)} trades)")
                migrated = True
            except Exception as e:
                print(f"  Error migrating trades_log.json: {e}")

    if ct_count == 0 and os.path.exists('closed_trades.json'):
        try:
            with open('closed_trades.json') as f:
                trades = json.load(f)
            for t in trades:
                append_closed_trade(t)
            print(f"  Migrated closed_trades.json ({len(trades)} trades)")
            migrated = True
        except Exception as e:
            print(f"  Error migrating closed_trades.json: {e}")

    if sig_count == 0 and os.path.exists('signals.json'):
        try:
            with open('signals.json') as f:
                signals = json.load(f)
            for s in signals:
                save_signal(s)
            print(f"  Migrated signals.json ({len(signals)} signals)")
            migrated = True
        except Exception as e:
            print(f"  Error migrating signals.json: {e}")

    if tf_count == 0 and os.path.exists('trade_features.json'):
        try:
            with open('trade_features.json') as f:
                features = json.load(f)
            for f_entry in features:
                append_trade_feature(f_entry)
            print(f"  Migrated trade_features.json ({len(features)} entries)")
            migrated = True
        except Exception as e:
            print(f"  Error migrating trade_features.json: {e}")

    if tj_count == 0 and os.path.exists('trade_journal.json'):
        try:
            with open('trade_journal.json') as f:
                journal = json.load(f)
            for j in journal:
                append_trade_journal(j)
            print(f"  Migrated trade_journal.json ({len(journal)} entries)")
            migrated = True
        except Exception as e:
            print(f"  Error migrating trade_journal.json: {e}")

    if not migrated:
        print(f"[{datetime.now()}] DB already has data, skipping migration")
    else:
        print(f"[{datetime.now()}] Migration complete")
