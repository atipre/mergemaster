import { BaseCheckpointSaver } from '@langchain/langgraph';
import Database from 'better-sqlite3';
import { join, dirname } from 'path';
import { mkdir } from 'fs/promises';

export class SqliteCheckpointSaver extends BaseCheckpointSaver {
  private db: Database.Database;
  private tableName: string;
  private metadataTable: string;

  constructor(dbPath?: string) {
    super();
    const defaultPath = join(process.cwd(), '.agent', 'checkpoints.db');
    const finalPath = dbPath || defaultPath;
    
    // Ensure directory exists
    mkdir(dirname(finalPath), { recursive: true }).catch(() => {});
    
    this.db = new Database(finalPath);
    this.tableName = 'checkpoints';
    this.metadataTable = 'session_metadata';
    this.initSchema();
  }

  private initSchema() {
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS ${this.tableName} (
        thread_id TEXT NOT NULL,
        checkpoint_id TEXT NOT NULL,
        checkpoint_data TEXT NOT NULL,
        parent_checkpoint_id TEXT,
        created_at INTEGER NOT NULL,
        PRIMARY KEY (thread_id, checkpoint_id)
      );
      
      CREATE INDEX IF NOT EXISTS idx_thread_parent 
      ON ${this.tableName}(thread_id, parent_checkpoint_id);

      CREATE TABLE IF NOT EXISTS ${this.metadataTable} (
        session_id TEXT PRIMARY KEY,
        created_at INTEGER NOT NULL,
        last_activity INTEGER NOT NULL,
        name TEXT
      );
    `);
  }

  private normalizeConfig(raw: any, caller: string): any {
    let config: any = raw;
    if (config == null || typeof config !== 'object' || Array.isArray(config)) {
      config = { configurable: {} };
    }
    if (config.configurable == null || typeof config.configurable !== 'object' || Array.isArray(config.configurable)) {
      config.configurable = {};
    }
    return config;
  }

  async put(config: any, checkpoint: any, metadata: any, newVersions?: any): Promise<any> {
    const safeConfig = this.normalizeConfig(config, 'put');
    const threadId = safeConfig.configurable?.thread_id ?? 'default';
    const checkpointId = checkpoint?.id || `checkpoint-${Date.now()}-${Math.random()}`;

    // Derive parent checkpoint reference if available
    let parentId: string | null = null;
    if (checkpoint && typeof checkpoint === 'object') {
      parentId = (checkpoint as any).parent_id ?? (checkpoint as any).parentCheckpointId ?? null;
    }
    if (!parentId && metadata && typeof metadata === 'object') {
      parentId = (metadata as any).parent_checkpoint_id ?? null;
    }

    // Convert Maps to objects for JSON serialization
    const serializableCheckpoint = this.serializeCheckpoint(checkpoint);

    const stmt = this.db.prepare(`
      INSERT OR REPLACE INTO ${this.tableName} 
      (thread_id, checkpoint_id, checkpoint_data, parent_checkpoint_id, created_at)
      VALUES (?, ?, ?, ?, ?)
    `);

    stmt.run(
      threadId,
      checkpointId,
      JSON.stringify({ checkpoint: serializableCheckpoint, metadata }),
      parentId,
      Date.now()
    );

    this.recordSessionActivity(threadId);

    return safeConfig;
  }

  private serializeCheckpoint(checkpoint: any): any {
    if (!checkpoint) return checkpoint;
    
    const serialized = { ...checkpoint };
    
    if (serialized.channel_values) {
      const values = { ...serialized.channel_values };
      
      // Don't serialize openFiles to avoid checkpoint bloat cache will be rebuilt from disk on reads
      if (values.openFiles) {
        // Only store file paths for reference
        const filePaths = values.openFiles instanceof Map 
          ? Array.from(values.openFiles.keys())
          : [];
        values.openFiles = filePaths;
      }
      
      serialized.channel_values = values;
    }
    
    return serialized;
  }

  private deserializeCheckpoint(checkpoint: any): any {
    if (!checkpoint) return checkpoint;
    
    const deserialized = { ...checkpoint };
    
    if (deserialized.channel_values) {
      const values = { ...deserialized.channel_values };
      
      // Initialize empty Map - cache will be rebuilt from disk on reads
      if (values.openFiles && Array.isArray(values.openFiles)) {
        values.openFiles = new Map();
      }
      
      deserialized.channel_values = values;
    }
    
    return deserialized;
  }

  async get(config: any): Promise<any> {
    const result = await this.getTuple(config);
    return result?.checkpoint || null;
  }

  async getTuple(config: any): Promise<any> {
    if (config == null || typeof config !== 'object' || Array.isArray(config)) {
      return undefined;
    }
    const normalized = this.normalizeConfig(config, 'getTuple');
    const threadId = normalized.configurable?.thread_id ?? 'default';
    const checkpointId = normalized.configurable?.checkpoint_id;

    let stmt;
    let row;
    
    if (checkpointId) {
      stmt = this.db.prepare(`
        SELECT checkpoint_data FROM ${this.tableName}
        WHERE thread_id = ? AND checkpoint_id = ?
      `);
      row = stmt.get(threadId, checkpointId);
    } else {
      // Get latest checkpoint for thread
      stmt = this.db.prepare(`
        SELECT checkpoint_data FROM ${this.tableName}
        WHERE thread_id = ?
        ORDER BY created_at DESC
        LIMIT 1
      `);
      row = stmt.get(threadId);
    }
    
    if (!row) return undefined;
    
    const data = JSON.parse((row as any).checkpoint_data);
    
    // Deserialize Maps
    if (data.checkpoint) {
      data.checkpoint = this.deserializeCheckpoint(data.checkpoint);
    }
    
    return {
      config: normalized,
      checkpoint: data.checkpoint,
      metadata: data.metadata || {},
    };
  }

  async putWrites(config: any, writes: any, taskId: string): Promise<void> {
    // Normalize config in case LangGraph expects a return value
    const normalized = this.normalizeConfig(config, 'putWrites');
    return Promise.resolve();
  }

  async deleteThread(config: any): Promise<void> {
    if (config == null || typeof config !== 'object' || Array.isArray(config)) {
      return;
    }
    const normalized = this.normalizeConfig(config, 'deleteThread');
    const threadId = normalized.configurable?.thread_id ?? 'default';
    const stmt = this.db.prepare(`DELETE FROM ${this.tableName} WHERE thread_id = ?`);
    stmt.run(threadId);
  }

  async *list(config: any, filter?: any): AsyncGenerator<any, void, unknown> {
    if (config == null || typeof config !== 'object' || Array.isArray(config)) {
      return;
    }
    const normalized = this.normalizeConfig(config, 'list');
    const threadId = normalized.configurable?.thread_id ?? 'default';
    
    const stmt = this.db.prepare(`
      SELECT checkpoint_id, checkpoint_data, created_at
      FROM ${this.tableName}
      WHERE thread_id = ?
      ORDER BY created_at DESC
    `);
    
    const rows = stmt.all(threadId) as any[];
    for (const row of rows) {
      yield {
        ...JSON.parse(row.checkpoint_data),
        checkpoint_id: row.checkpoint_id,
        created_at: row.created_at,
      };
    }
  }

  private async getLatestCheckpointId(threadId: string): Promise<string | null> {
    const stmt = this.db.prepare(`
      SELECT checkpoint_id FROM ${this.tableName}
      WHERE thread_id = ?
      ORDER BY created_at DESC
      LIMIT 1
    `);
    const row = stmt.get(threadId) as any;
    return row?.checkpoint_id || null;
  }

  recordSessionStart(sessionId: string, name?: string) {
    const now = Date.now();
    const stmt = this.db.prepare(`
      INSERT INTO ${this.metadataTable} (session_id, created_at, last_activity, name)
      VALUES (?, ?, ?, ?)
      ON CONFLICT(session_id) DO UPDATE SET
        name = COALESCE(excluded.name, ${this.metadataTable}.name),
        last_activity = excluded.last_activity
    `);
    stmt.run(sessionId, now, now, name ?? null);
  }

  recordSessionActivity(sessionId: string) {
    const now = Date.now();
    const stmt = this.db.prepare(`
      INSERT INTO ${this.metadataTable} (session_id, created_at, last_activity)
      VALUES (?, ?, ?)
      ON CONFLICT(session_id) DO UPDATE SET
        last_activity = excluded.last_activity
    `);
    stmt.run(sessionId, now, now);
  }

  getSessionMetadata(sessionId: string): {
    session_id: string;
    created_at: number;
    last_activity: number;
    name?: string | null;
  } | null {
    const stmt = this.db.prepare(`
      SELECT session_id, created_at, last_activity, name
      FROM ${this.metadataTable}
      WHERE session_id = ?
    `);
    const row = stmt.get(sessionId) as any;
    return row || null;
  }

  listSessions(): Array<{
    session_id: string;
    created_at: number;
    last_activity: number;
    name?: string | null;
  }> {
    const stmt = this.db.prepare(`
      SELECT session_id, created_at, last_activity, name
      FROM ${this.metadataTable}
      ORDER BY last_activity DESC
    `);
    return stmt.all() as any[];
  }

  deleteSession(sessionId: string) {
    const deleteCheckpoints = this.db.prepare(`
      DELETE FROM ${this.tableName} WHERE thread_id = ?
    `);
    deleteCheckpoints.run(sessionId);

    const deleteMeta = this.db.prepare(`
      DELETE FROM ${this.metadataTable} WHERE session_id = ?
    `);
    deleteMeta.run(sessionId);
  }

  close() {
    this.db.close();
  }
}
