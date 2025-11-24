import { spawn } from 'child_process';

export interface SearchResult {
  path: string;
  line_number: number;
  content: string;
}

export async function directorySearchTool(
  pattern: string,
  searchPath?: string,
  options?: {
    caseSensitive?: boolean;
    fileTypes?: string[];
    maxResults?: number;
  }
): Promise<SearchResult[]> {
  const searchDir = searchPath || '.';
  const maxResults = options?.maxResults ?? 1000;
  const searchResults: SearchResult[] = [];

  const args: string[] = ['--json', '--color', 'never'];

  if (options?.caseSensitive === false) {
    args.push('--ignore-case');
  }

  const aliasGlobs = new Map<string, Set<string>>();
  const aliasOrder: string[] = [];
  if (options?.fileTypes && options.fileTypes.length > 0) {
    options.fileTypes.forEach((rawType, index) => {
      const filter = buildFileTypeFilter(rawType, index);
      if (!filter) {
        return;
      }

      if (!aliasGlobs.has(filter.alias)) {
        aliasGlobs.set(filter.alias, new Set());
        aliasOrder.push(filter.alias);
      }
      const globSet = aliasGlobs.get(filter.alias)!;
      for (const glob of filter.globs) {
        globSet.add(glob);
      }
    });
  }

  if (aliasOrder.length > 0) {
    for (const alias of aliasOrder) {
      const globs = aliasGlobs.get(alias);
      if (!globs || globs.size === 0) {
        continue;
      }
      for (const glob of globs) {
        args.push('--type-add', `${alias}:${glob}`);
      }
      args.push('--type', alias);
    }
  }

  if (maxResults > 0) {
    args.push('-m', String(maxResults));
  }

  args.push(pattern, searchDir);

  return new Promise<SearchResult[]>((resolve, reject) => {
    let terminatedEarly = false;
    const rg = spawn('rg', args, { stdio: ['ignore', 'pipe', 'pipe'] });

    let stderrBuffer = '';
    let stdoutBuffer = '';

    rg.stdout.on('data', chunk => {
      stdoutBuffer += chunk.toString();
      let newlineIndex: number;
      while ((newlineIndex = stdoutBuffer.indexOf('\n')) !== -1) {
        const line = stdoutBuffer.slice(0, newlineIndex).trim();
        stdoutBuffer = stdoutBuffer.slice(newlineIndex + 1);
        if (!line) {
          continue;
        }
        try {
          const json: any = JSON.parse(line);
          if (json.type === 'match' && json.data) {
            searchResults.push({
              path: json.data.path.text,
              line_number: json.data.line_number,
              content: json.data.lines.text.trim(),
            });
            if (maxResults > 0 && searchResults.length >= maxResults) {
              terminatedEarly = true;
              rg.kill();
            }
          }
        } catch {
          // ignore malformed JSON line
        }
      }
    });

    rg.stderr.on('data', chunk => {
      stderrBuffer += chunk.toString();
    });

    rg.on('error', error => {
      reject(new Error(`Failed to start ripgrep: ${error.message}`));
    });

    rg.on('close', (code, signal) => {
      if (stdoutBuffer.trim()) {
        try {
          const json: any = JSON.parse(stdoutBuffer.trim());
          if (json.type === 'match' && json.data) {
            searchResults.push({
              path: json.data.path.text,
              line_number: json.data.line_number,
              content: json.data.lines.text.trim(),
            });
          }
        } catch {
          // ignore trailing partial line
        }
      }

      if (signal === 'SIGTERM' && terminatedEarly && searchResults.length > 0) {
        resolve(searchResults);
        return;
      }

      if (code === 0 || (code === 1 && searchResults.length === 0)) {
        resolve(searchResults);
        return;
      }

      const errorMessage = stderrBuffer.trim() || `rg exited with code ${code}`;
      reject(new Error(`Failed to search directory: ${errorMessage}`));
    });
  });
}

interface FileTypeFilter {
  alias: string;
  globs: string[];
}

const FILE_TYPE_PRESETS: Record<string, string[]> = {
  'c++': ['*.cpp', '*.cc', '*.cxx', '*.hpp', '*.hxx', '*.hh', '*.ipp', '*.tpp'],
  cpp: ['*.cpp', '*.cc', '*.cxx', '*.hpp', '*.hxx', '*.hh', '*.ipp', '*.tpp'],
  cxx: ['*.cpp', '*.cc', '*.cxx', '*.hpp', '*.hxx', '*.hh', '*.ipp', '*.tpp'],
  cc: ['*.cc', '*.cpp', '*.cxx'],
  hpp: ['*.hpp', '*.hxx', '*.hh'],
  'c': ['*.c', '*.h'],
  'objc': ['*.m', '*.mm', '*.h'],
  'objective-c': ['*.m', '*.mm', '*.h'],
  'objectivec': ['*.m', '*.mm', '*.h'],
  'typescript': ['*.ts', '*.tsx'],
  'javascript': ['*.js', '*.jsx'],
  'jsx': ['*.jsx'],
  'tsx': ['*.tsx'],
  'ts': ['*.ts'],
  'js': ['*.js'],
  'json': ['*.json'],
  'python': ['*.py'],
  'py': ['*.py'],
  'go': ['*.go'],
  'rust': ['*.rs'],
  'rs': ['*.rs'],
  'java': ['*.java'],
  'kotlin': ['*.kt', '*.kts'],
  'kt': ['*.kt', '*.kts'],
  'swift': ['*.swift'],
  'scala': ['*.scala'],
  'ruby': ['*.rb'],
  'rb': ['*.rb'],
  'php': ['*.php'],
};

function buildFileTypeFilter(rawType: string | undefined, index: number): FileTypeFilter | null {
  if (!rawType) return null;
  const trimmed = rawType.trim();
  if (!trimmed) return null;

  let lookupKey = trimmed.toLowerCase();
  if (lookupKey.startsWith('.')) {
    lookupKey = lookupKey.slice(1);
  }

  let globs: string[] | undefined = FILE_TYPE_PRESETS[lookupKey];

  if (!globs || globs.length === 0) {
    let extension = lookupKey;
    if (/^[a-z0-9]+$/.test(extension)) {
      globs = [`*.${extension}`];
    } else {
      return null;
    }
  }

  const aliasBase = lookupKey.replace(/[^a-z0-9_]+/g, '_');
  const alias = aliasBase ? `ft_${aliasBase}` : `ft_${index}`;

  return {
    alias,
    globs,
  };
}
