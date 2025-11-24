import { writeFile } from 'fs/promises';
import { join } from 'path';
import { mkdir } from 'fs/promises';
import { dirname } from 'path';

export async function writeFileTool(filePath: string, content: string, basePath?: string): Promise<void> {
  try {
    const fullPath = basePath ? join(basePath, filePath) : filePath;
    const dir = dirname(fullPath);
    await mkdir(dir, { recursive: true });
    await writeFile(fullPath, content, 'utf-8');
  } catch (error) {
    throw new Error(`Failed to write file ${filePath}: ${error instanceof Error ? error.message : String(error)}`);
  }
}
