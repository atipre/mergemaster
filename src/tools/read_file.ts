import { readFile } from 'fs/promises';
import { join } from 'path';

export async function readFileTool(filePath: string, basePath?: string): Promise<string> {
  try {
    const fullPath = basePath ? join(basePath, filePath) : filePath;
    const content = await readFile(fullPath, 'utf-8');
    return content;
  } catch (error) {
    throw new Error(`Failed to read file ${filePath}: ${error instanceof Error ? error.message : String(error)}`);
  }
}
