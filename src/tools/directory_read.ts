import { readdir, stat } from 'fs/promises';
import { join } from 'path';

export interface DirectoryEntry {
  name: string;
  type: 'file' | 'directory';
  path: string;
}

export async function directoryReadTool(dirPath: string, basePath?: string): Promise<DirectoryEntry[]> {
  try {
    const fullPath = basePath ? join(basePath, dirPath) : dirPath;
    const entries = await readdir(fullPath, { withFileTypes: true });
    
    const result: DirectoryEntry[] = await Promise.all(
      entries.map(async (entry) => {
        const entryPath = join(fullPath, entry.name);
        const stats = await stat(entryPath);
        return {
          name: entry.name,
          type: stats.isDirectory() ? 'directory' : 'file',
          path: entryPath,
        };
      })
    );
    
    return result;
  } catch (error) {
    throw new Error(`Failed to read directory ${dirPath}: ${error instanceof Error ? error.message : String(error)}`);
  }
}
