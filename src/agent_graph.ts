import 'dotenv/config';
import Anthropic from '@anthropic-ai/sdk';
import OpenAI from 'openai';
import { StateGraph, END, START } from '@langchain/langgraph';
import { Output } from './output.js';
import { readFileTool } from './tools/read_file.js';
import { writeFileTool } from './tools/write_file.js';
import { directoryReadTool } from './tools/directory_read.js';
import type { DirectoryEntry } from './tools/directory_read.js';
import { directorySearchTool } from './tools/directory_search.js';
import type { SearchResult } from './tools/directory_search.js';
import {
  executeCommandTool,
  waitForCommandCompletion,
  executeServerCommandTool,
  waitForServerCommand,
  waitForContinue,
} from './tools/Execute_commands.js';
import type { CommandResult } from './tools/Execute_commands.js';
import { SqliteCheckpointSaver } from './checkpoint_store.js';

const MAX_PROMPT_TOKEN_BUDGET = 180000;
const MAX_TOOL_RESULT_CHARS = 4000;
const MAX_TOOL_RESULT_LINES = 120;

function truncateString(input: string, charLimit: number = MAX_TOOL_RESULT_CHARS): string {
  if (input.length <= charLimit) {
    return input;
  }
  const truncated = input.slice(0, charLimit);
  return `${truncated}\n… (truncated ${input.length - charLimit} more characters)`;
}

function truncateLines(input: string, lineLimit: number = MAX_TOOL_RESULT_LINES): string {
  const lines = input.split('\n');
  if (lines.length <= lineLimit) {
    return input;
  }
  const kept = lines.slice(0, lineLimit).join('\n');
  return `${kept}\n… (truncated ${lines.length - lineLimit} more lines)`;
}

function summarizeReadFileResult(content: string, filePath: string): string {
  const limitedLines = truncateLines(content);
  const limited = truncateString(limitedLines);
  const totalLines = content.split('\n').length;
  const summaryLines = limited.split('\n').length;
  if (summaryLines < totalLines) {
    return `Contents of ${filePath} (showing first ${summaryLines} of ${totalLines} lines):\n${limited}`;
  }
  return `Contents of ${filePath}:\n${limited}`;
}

function summarizeDirectoryRead(entries: DirectoryEntry[], dirPath: string): string {
  if (!entries.length) {
    return `Directory ${dirPath} is empty.`;
  }
  const lines: string[] = [];
  const limit = 50;
  for (const entry of entries.slice(0, limit)) {
    lines.push(`${entry.type === 'directory' ? '📁' : '📄'} ${entry.name}`);
  }
  if (entries.length > limit) {
    lines.push(`… (${entries.length - limit} more entries)`);
  }
  return `Directory listing for ${dirPath} (${entries.length} entries):\n${lines.join('\n')}`;
}

function summarizeDirectorySearch(result: SearchResult[], pattern: string): string {
  if (!result || result.length === 0) {
    return `No matches found for "${pattern}".`;
  }

  const topMatches = result.slice(0, 20);
  const byFile = new Map<string, Array<{ line: number; text: string }>>();

  for (const match of topMatches) {
    if (!match || typeof match !== 'object') continue;
    const path = match.path;
    const lineNumber = match.line_number;
    const text = match.content;
    if (typeof path !== 'string') continue;

    if (!byFile.has(path)) {
      byFile.set(path, []);
    }
    byFile.get(path)!.push({
      line: typeof lineNumber === 'number' ? lineNumber : 0,
      text: typeof text === 'string' ? text.trim() : '',
    });
  }

  const lines: string[] = [];
  for (const [path, matches] of byFile) {
    lines.push(`• ${path}`);
    for (const m of matches.slice(0, 3)) {
      const snippet = truncateString(m.text, 200);
      lines.push(`  - #${m.line}: ${snippet}`);
    }
    if (matches.length > 3) {
      lines.push(`  - … (+${matches.length - 3} more matches in this file)`);
    }
  }

  if (result.length > topMatches.length) {
    lines.push(`… (${result.length - topMatches.length} additional matches truncated)`);
  }

  return `Found ${result.length} matches for "${pattern}" across ${byFile.size} file(s):\n${lines.join('\n')}`;
}

function summarizeCommandResult(result: CommandResult | string | undefined, command: string): string {
  if (!result || typeof result === 'string') {
    const text = typeof result === 'string' ? result : '(no output)';
    return truncateString(`Output from "${command}":\n${truncateLines(text, 80)}`);
  }

  const stdout = result.stdout || '';
  const stderr = result.stderr || '';
  const combined = [stdout, stderr].filter(Boolean).join('\n').trim();
  const snippet = combined ? truncateString(truncateLines(combined, 80)) : '(no output)';
  const exitInfo = typeof result.exitCode === 'number' ? `Exit code: ${result.exitCode}` : 'Exit code unknown';
  return `${exitInfo}\nCommand: ${command}\nOutput:\n${snippet}`;
}

function summarizeToolResult(toolName: string, toolInput: any, result: unknown): string {
  switch (toolName) {
    case 'read_file': {
      if (typeof result === 'string') {
        return summarizeReadFileResult(result, toolInput.file_path ?? '(unknown file)');
      }
      return `Read file ${toolInput.file_path ?? ''}.`;
    }
    case 'write_file': {
      const length = typeof toolInput?.content === 'string' ? toolInput.content.length : 0;
      return `Wrote ${toolInput.file_path} (${length} characters).`;
    }
    case 'directory_read': {
      if (Array.isArray(result)) {
        return summarizeDirectoryRead(result as DirectoryEntry[], toolInput.dir_path ?? toolInput.search_path ?? '.');
      }
      return truncateString(typeof result === 'string' ? result : JSON.stringify(result));
    }
    case 'directory_search': {
      if (Array.isArray(result)) {
        return summarizeDirectorySearch(result as SearchResult[], toolInput.pattern ?? '');
      }
      return truncateString(typeof result === 'string' ? result : JSON.stringify(result));
    }
    case 'execute_command': {
      return summarizeCommandResult(result as CommandResult | string | undefined, toolInput.command ?? '');
    }
    case 'execute_server_command': {
      return summarizeCommandResult(result as CommandResult | string | undefined, toolInput.command ?? '');
    }
    default: {
      if (typeof result === 'string') {
        return truncateString(truncateLines(result));
      }
      return truncateString(JSON.stringify(result, null, 2));
    }
  }
}

function estimateTokensFromMessage(message: any): number {
  if (!message) return 0;
  const overhead = 6;
  const collectStrings = (value: any): string[] => {
    if (typeof value === 'string') {
      return [value];
    }
    if (Array.isArray(value)) {
      return value.flatMap(collectStrings);
    }
    if (value && typeof value === 'object') {
      return Object.values(value).flatMap(collectStrings);
    }
    return [];
  };

  const text = collectStrings(message);
  const totalChars = text.reduce((sum, piece) => sum + piece.length, 0);
  return Math.ceil(totalChars / 4) + overhead;
}

function summarizeMessage(message: any): string | null {
  if (!message) return null;
  const role = message.role || 'unknown';

  // Tool results
  if (Array.isArray(message.content)) {
    const toolBlocks = message.content.filter((block: any) => block.type === 'tool_result');
    if (toolBlocks.length > 0) {
      const toolSummaries = toolBlocks.map((block: any) => {
        if (block.tool_use_id) {
          return `${block.tool_use_id}: ${truncateString(block.content || '', 200)}`;
        }
        return truncateString(block.content || '', 200);
      });
      return `Tool results (${toolSummaries.length}): ${toolSummaries.join(' | ')}`;
    }

    const textBlocks = message.content.filter((block: any) => block.type === 'text');
    if (textBlocks.length > 0) {
      const combined = textBlocks.map((block: any) => block.text).join('\n');
      return `${role}: ${truncateString(combined, 200)}`;
    }
  }

  if (typeof message.content === 'string') {
    return `${role}: ${truncateString(message.content, 200)}`;
  }

  if (message.tool_calls) {
    const names = message.tool_calls.map((call: any) => call.function?.name || 'unknown');
    return `${role} tool calls: ${names.join(', ')}`;
  }

  return null;
}

function pruneMessages(
  messages: any[],
  maxMessages: number = 50,
  maxTokens: number = MAX_PROMPT_TOKEN_BUDGET
): { prunedMessages: any[]; archivedSummaries: string[] } {
  if (messages.length === 0) {
    return { prunedMessages: messages, archivedSummaries: [] };
  }

  const firstMessage = messages[0];
  let usedTokens = estimateTokensFromMessage(firstMessage);
  if (usedTokens > maxTokens) {
    return { prunedMessages: [firstMessage], archivedSummaries: [] };
  }
  const selected: any[] = [];
  const archived: string[] = [];

  // Helper to check if message has tool_result blocks
  const hasToolResults = (msg: any) => {
    if (!msg?.content || !Array.isArray(msg.content)) return false;
    return msg.content.some((block: any) => block?.type === 'tool_result');
  };

  // Helper to check if message has tool_use blocks
  const hasToolUses = (msg: any) => {
    if (!msg?.content || !Array.isArray(msg.content)) return false;
    return msg.content.some((block: any) => block?.type === 'tool_use');
  };

  for (let i = messages.length - 1; i >= 1; i--) {
    const msg = messages[i];
    const tokens = estimateTokensFromMessage(msg);
    
    // If this message has tool_results, we MUST keep the previous message too (tool_uses)
    const needsPreviousMessage = hasToolResults(msg) && i > 1;
    
    if (selected.length >= maxMessages - 1 && !needsPreviousMessage) {
      const summary = summarizeMessage(msg);
      if (summary) {
        archived.push(summary);
      }
      continue;
    }
    if (usedTokens + tokens > maxTokens && !needsPreviousMessage) {
      const summary = summarizeMessage(msg);
      if (summary) {
        archived.push(summary);
      }
      continue;
    }
    selected.push(msg);
    usedTokens += tokens;
    
    // If we kept a message with tool_results, force-keep the previous assistant message
    if (needsPreviousMessage && !selected.includes(messages[i - 1])) {
      const prevMsg = messages[i - 1];
      selected.push(prevMsg);
      usedTokens += estimateTokensFromMessage(prevMsg);
      i--; // Skip the previous message in the next iteration
    }
  }

  selected.reverse();
  let pruned = [firstMessage, ...selected];

  if (pruned.length > maxMessages) {
    const overflow = pruned.slice(0, pruned.length - maxMessages);
    for (const msg of overflow) {
      const summary = summarizeMessage(msg);
      if (summary) {
        archived.push(summary);
      }
    }
    pruned = pruned.slice(-maxMessages);
  }
  
  // Final validation: remove any orphaned tool_result blocks
  // If a message has tool_results but previous message doesn't have tool_uses, strip them
  for (let i = 1; i < pruned.length; i++) {
    const msg = pruned[i];
    const prevMsg = pruned[i - 1];
    
    if (hasToolResults(msg) && !hasToolUses(prevMsg)) {
      // Strip tool_result blocks from this message
      if (msg.content && Array.isArray(msg.content)) {
        msg.content = msg.content.filter((block: any) => block?.type !== 'tool_result');
        // If message now has no content, remove it entirely
        if (msg.content.length === 0) {
          pruned.splice(i, 1);
          i--;
        }
      }
    }
  }
  
  if (pruned.length > maxMessages) {
    return { prunedMessages: pruned.slice(-maxMessages), archivedSummaries: archived };
  }

  return { prunedMessages: pruned, archivedSummaries: archived };
}

const anthropic = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY || '',
});

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY || '',
});

let output: any = new Output();

export function setOutputHandler(handler: any) {
  output = handler;
}

export function getOutputHandler() {
  return output;
}

// Global checkpoint store instance that can be shared across TUI sessions
let globalCheckpointStore: any = undefined;

export function setCheckpointStore(store: any) {
  globalCheckpointStore = store;
}

export function getCheckpointStore() {
  return globalCheckpointStore;
}

// Undo last file change
export async function undoLastChange(filePath?: string): Promise<{ success: boolean; message: string; path?: string }> {
  // access the checkpoint store to get the current state
  if (!globalCheckpointStore) {
    return { success: false, message: 'No checkpoint store available' };
  }
  
  try {
    const config = { configurable: { thread_id: 'default' } };
    const checkpointTuple = await globalCheckpointStore.getTuple(config);
    
    if (!checkpointTuple || !checkpointTuple.checkpoint || !checkpointTuple.checkpoint.channel_values) {
      return { success: false, message: 'No file history available' };
    }
    
    const openFiles: Map<string, FileContext> = checkpointTuple.checkpoint.channel_values.openFiles;
    const recentChanges: Array<{ path: string; timestamp: number }> = checkpointTuple.checkpoint.channel_values.recentChanges || [];
    
    // Find the file to undo
    let targetPath: string;
    if (filePath) {
      targetPath = filePath;
    } else if (recentChanges.length > 0) {
      // Undo most recent change
      targetPath = recentChanges[recentChanges.length - 1].path;
    } else {
      return { success: false, message: 'No recent changes to undo' };
    }
    
    const fileContext = openFiles.get(targetPath);
    if (!fileContext || !fileContext.previousVersions || fileContext.previousVersions.length === 0) {
      return { success: false, message: `No previous version available for ${targetPath}` };
    }
    
    // Pop the last version and restore it
    const previousContent = fileContext.previousVersions.pop()!;
    await writeFileTool(targetPath, previousContent);
    
    // Update the current content in checkpoint without adding to history
    fileContext.content = previousContent;
    
    return { 
      success: true, 
      message: `Reverted ${targetPath} to previous version`,
      path: targetPath
    };
  } catch (error) {
    return { 
      success: false, 
      message: `Undo failed: ${error instanceof Error ? error.message : String(error)}`
    };
  }
}

const tools: Anthropic.Tool[] = [
  {
    name: 'read_file',
    description: 'Read the contents of a file',
    input_schema: {
      type: 'object' as const,
      properties: {
        file_path: {
          type: 'string',
          description: 'Path to the file to read',
        },
      },
      required: ['file_path'],
    },
  },
  {
    name: 'write_file',
    description: 'Write content to a file (overwrites existing content)',
    input_schema: {
      type: 'object' as const,
      properties: {
        file_path: {
          type: 'string',
          description: 'Path to the file to write',
        },
        content: {
          type: 'string',
          description: 'Content to write to the file',
        },
      },
      required: ['file_path', 'content'],
    },
  },
  {
    name: 'directory_read',
    description: 'List files and directories in a given path',
    input_schema: {
      type: 'object' as const,
      properties: {
        dir_path: {
          type: 'string',
          description: 'Path to the directory to read',
        },
      },
      required: ['dir_path'],
    },
  },
  {
    name: 'directory_search',
    description: 'Search for text patterns in files using ripgrep',
    input_schema: {
      type: 'object' as const,
      properties: {
        pattern: {
          type: 'string',
          description: 'Search pattern (regex supported)',
        },
        search_path: {
          type: 'string',
          description: 'Directory path to search in (default: current directory)',
        },
        case_sensitive: {
          type: 'boolean',
          description: 'Whether search should be case sensitive',
        },
        file_types: {
          type: 'array',
          items: { type: 'string' },
          description: 'File types to search in (e.g., ["ts", "js"])',
        },
        max_results: {
          type: 'number',
          description: 'Maximum number of results to return',
        },
      },
      required: ['pattern'],
    },
  },
  {
    name: 'execute_command',
    description: 'Propose a shell command for the user to approve and execute. The command will be shown to the user for approval before running. Fully interactive commands (like npm create, git commit with editor) are supported - the user will handle any prompts.',
    input_schema: {
      type: 'object' as const,
      properties: {
        command: {
          type: 'string',
          description: 'Shell command to execute',
        },
        cwd: {
          type: 'string',
          description: 'Working directory for the command',
        },
        timeout: {
          type: 'number',
          description: 'Timeout in milliseconds (default: 30000)',
        },
      },
      required: ['command'],
    },
  },
  {
    name: 'execute_server_command',
    description: 'Request starting a long-running server process (like npm run dev). The user will approve and the output will stream to a dedicated read-only terminal window.',
    input_schema: {
      type: 'object' as const,
      properties: {
        command: {
          type: 'string',
          description: 'Shell command to execute',
        },
        cwd: {
          type: 'string',
          description: 'Working directory for the command',
        },
        timeout: {
          type: 'number',
          description: 'Optional timeout in seconds before auto-terminating the server',
        },
      },
      required: ['command'],
    },
  },
];

// Context tracking
interface FileContext {
  path: string;
  content?: string;
  previousVersions?: string[]; // Stack of previous content for undo
  lastRead: number;
  lastModified: number;
}

interface AgentState {
  messages: any[];
  verbose: boolean;
  lastResponse?: any;
  toolUses?: Anthropic.ToolUseBlock[];
  textBlocks?: Anthropic.TextBlock[];
  toolResults?: Anthropic.ToolResultBlockParam[];
  // Context management
  openFiles: Map<string, FileContext>;
  recentChanges: Array<{ path: string; timestamp: number }>;
  recentReads: Array<{ path: string; timestamp: number }>;
  recentCommands: Array<{ command: string; output?: string; timestamp: number }>;
  workingSet: string[]; // Files currently in focus
  historicalSummaries: Array<{ timestamp: number; summary: string }>;
  // Iteration tracking
  iterationCount?: number;
}

async function executeTool(name: string, input: any): Promise<any> {
  switch (name) {
    case 'read_file':
      return await readFileTool(input.file_path);
    case 'write_file':
      return await writeFileTool(input.file_path, input.content);
    case 'directory_read':
      return await directoryReadTool(input.dir_path);
    case 'directory_search':
      return await directorySearchTool(
        input.pattern,
        input.search_path,
        {
          caseSensitive: input.case_sensitive,
          fileTypes: input.file_types,
          maxResults: input.max_results,
        }
      );
    case 'execute_command':
      return await executeCommandTool(input.command, input.cwd, {
        timeout: input.timeout,
      });
    case 'execute_server_command':
      return await executeServerCommandTool(input.command, input.cwd, {
        timeout: input.timeout,
      });
    default:
      throw new Error(`Unknown tool: ${name}`);
  }
}

// extract artifacts from command output 
function extractArtifacts(command: string, output?: string): string[] {
  const artifacts: string[] = [];
  
  if (!output) return artifacts;
  
  // Extract executable names from compilation commands
  const compileMatch = command.match(/-o\s+(\S+)/);
  if (compileMatch) {
    artifacts.push(compileMatch[1]);
  }
  
  // Extract from output messages 
  const createdMatch = output.match(/(?:created|compiled|generated|built)\s+(?:executable|file|binary)?\s*:?\s*([^\s,]+)/i);
  if (createdMatch) {
    artifacts.push(createdMatch[1]);
  }
  
  // Extract file paths from output
  const filePathMatch = output.match(/(?:^|\s)([\/\w\-\.]+\.(?:exe|out|bin|so|dylib|dll))(?:\s|$)/);
  if (filePathMatch) {
    artifacts.push(filePathMatch[1]);
  }
  
  return artifacts;
}

// Enforce cache size limit with LRU eviction
function evictOldCacheEntries(cache: Map<string, FileContext>, maxSize: number) {
  if (cache.size <= maxSize) {
    return;
  }
  
  // Sort by lastRead timestamp (oldest first)
  const entries = Array.from(cache.entries()).sort((a, b) => {
    return (a[1].lastRead || 0) - (b[1].lastRead || 0);
  });
  
  // Remove oldest entries until we're under the limit
  const toRemove = cache.size - maxSize;
  for (let i = 0; i < toRemove; i++) {
    cache.delete(entries[i][0]);
  }
}

// Generate context summary
function getContextSummary(state: AgentState): string {
  const openFilesList = Array.from(state.openFiles?.keys() || []);
  const recentChanges = state.recentChanges?.slice(-5) || [];
  const recentReads = state.recentReads?.slice(-5) || [];
  const recentCommands = state.recentCommands?.slice(-3) || []; // Last 3 commands
  const historical = state.historicalSummaries?.slice(-5) || [];
  
  if (openFilesList.length === 0 && recentChanges.length === 0 && recentCommands.length === 0) {
    if (!historical.length) {
      return '';
    }
  }

  let summary = '[Context] ';
  
  if (openFilesList.length > 0) {
    summary += `Working with files: ${openFilesList.join(', ')}. `;
  }
  
  if (recentChanges.length > 0) {
    summary += `Recent changes: ${recentChanges.map(c => c.path).join(', ')}. `;
  }
  
  // Include recent commands and their artifacts
  if (recentCommands.length > 0) {
    const commandSummaries = recentCommands.map(cmd => {
      const artifacts = extractArtifacts(cmd.command, cmd.output);
      let summary = `Ran: ${cmd.command}`;
      if (artifacts.length > 0) {
        summary += ` (created: ${artifacts.join(', ')})`;
      }
      return summary;
    });
    summary += `Recent commands: ${commandSummaries.join('; ')}. `;
  }
  
  if (state.workingSet && state.workingSet.length > 0) {
    summary += `Focus: ${state.workingSet.join(', ')}. `;
  }

  if (historical.length > 0) {
    const historySnippets = historical.map(entry => {
      const elapsed = Date.now() - entry.timestamp;
      const minutes = Math.max(1, Math.round(elapsed / 60000));
      return `${minutes}m ago: ${entry.summary}`;
    });
    summary += `Earlier work: ${historySnippets.join(' | ')}. `;
  }
  
  return summary.trim();
}

// Convert Anthropic tools to OpenAI format
function convertToolsToOpenAI(anthropicTools: Anthropic.Tool[]): any[] {
  return anthropicTools.map(tool => ({
    type: 'function',
    function: {
      name: tool.name,
      description: tool.description,
      parameters: tool.input_schema,
    },
  }));
}

// Convert Anthropic message format to OpenAI format
function convertMessagesToOpenAI(messages: any[]): any[] {
  const converted: any[] = [];
  
  for (const msg of messages) {
    if (!msg.content) {
      converted.push(msg);
      continue;
    }
    
    // If content is already a string, return as-is
    if (typeof msg.content === 'string') {
      converted.push(msg);
      continue;
    }
    
    // If content is an array (Anthropic format), convert it
    if (Array.isArray(msg.content)) {
      // Check for tool_result blocks - these need to become separate tool messages
      const toolResultBlocks = msg.content.filter((block: any) => block.type === 'tool_result');
      
      if (toolResultBlocks.length > 0) {
        // Convert each tool_result to a separate OpenAI tool message
        for (const block of toolResultBlocks) {
          converted.push({
            role: 'tool',
            tool_call_id: block.tool_use_id,
            content: block.is_error ? `Error: ${block.content}` : String(block.content),
          });
        }
        continue;
      }
      
      // Extract text blocks
      const textBlocks = msg.content.filter((block: any) => block.type === 'text');
      const text = textBlocks.map((block: any) => block.text).join('\n');
      
      // Extract tool_use blocks and convert to OpenAI tool_calls
      const toolUseBlocks = msg.content.filter((block: any) => block.type === 'tool_use');
      
      if (toolUseBlocks.length > 0) {
        converted.push({
          role: msg.role,
          content: text || null,
          tool_calls: toolUseBlocks.map((block: any) => ({
            id: block.id,
            type: 'function',
            function: {
              name: block.name,
              arguments: JSON.stringify(block.input),
            },
          })),
        });
        continue;
      }
      
      converted.push({
        role: msg.role,
        content: text || '',
      });
      continue;
    }
    
    converted.push(msg);
  }
  
  return converted;
}

// Node 1: Call LLM with streaming
async function callLLM(state: AgentState, config?: any): Promise<Partial<AgentState>> {
  // Get model from config or use default
  const model = config?.configurable?.model || 'claude-sonnet-4-20250514';
  const provider = model.startsWith('gpt-') ? 'openai' : 'anthropic';
  
  // Check if user wants to continue after hitting iteration limit
  let resetCounter = false;
  if ((state.iterationCount || 0) >= 25 && state.messages.length > 0) {
    const lastMessage = state.messages[state.messages.length - 1];
    if (lastMessage?.role === 'user') {
      const content = typeof lastMessage.content === 'string' 
        ? lastMessage.content 
        : lastMessage.content?.[0]?.text || '';
      const normalized = content.toLowerCase().trim();
      if (normalized.includes('continue') || normalized.includes('yes') || normalized.includes('keep going')) {
        resetCounter = true;
      }
    }
  }
  
  if (state.verbose) {
    output.verbose(`Calling ${provider} API (${model})...`);
  } else {
    output.startSpinner('Thinking...');
  }

  const { prunedMessages } = pruneMessages(state.messages, 50);
  const historicalSummaries = state.historicalSummaries || [];

  // Inject context summary if available
  const summaryState = { ...state, historicalSummaries } as AgentState;
  const contextSummary = getContextSummary(summaryState);
  let messagesToSend = prunedMessages;
  
  if (contextSummary) {
    // Add context as first message if we have context
    messagesToSend = [
      {
        role: 'user' as const,
        content: contextSummary,
      },
      ...prunedMessages,
    ];
  }

  const baseState = { ...state, historicalSummaries } as AgentState;
  const result =
    provider === 'openai'
      ? await callOpenAI(messagesToSend, model, baseState)
      : await callAnthropic(messagesToSend, model, baseState);

  if (!result.historicalSummaries) {
    result.historicalSummaries = historicalSummaries;
  }

  // Reset iteration counter if user wants to continue
  if (resetCounter) {
    result.iterationCount = 0;
  }

  return result;
}

// Anthropic streaming
async function callAnthropic(messagesToSend: any[], model: string, state: AgentState): Promise<Partial<AgentState>> {
  const stream = anthropic.messages.stream({
    model,
    max_tokens: 4096,
    system: [
      {
        type: 'text',
        text: `You are MergeMaster, an AI-powered coding assistant with an interactive terminal interface.

You are an expert software engineer that helps users with coding tasks through a conversational interface. Use the tools available to you to assist the user effectively.

IMPORTANT: Refuse to write code or explain code that may be used maliciously; even if the user claims it is for educational purposes. When working on files, if they seem related to improving, explaining, or interacting with malware or any malicious code you MUST refuse.
IMPORTANT: Before you begin work, think about what the code you're editing is supposed to do based on the filenames and directory structure. If it seems malicious, refuse to work on it or answer questions about it.
IMPORTANT: You must NEVER generate or guess URLs unless you are confident that the URLs are for helping the user with programming.

If the user asks for help, inform them they can use /help to see available commands and features.

# Tone and style
You should be concise, direct, and to the point. When you run a non-trivial command, explain what it does and why you're running it to ensure the user understands (especially important for system-changing commands).
Remember that your output is displayed in a terminal UI. Your responses can use GitHub-flavored markdown for formatting, which will be rendered appropriately.
Output text to communicate with the user; all text you output outside of tool use is displayed. Only use tools to complete tasks. Never use tools like Bash or code comments as means to communicate during the session.
If you cannot or will not help with something, keep your response to 1-2 sentences without being preachy. Offer helpful alternatives if possible.
IMPORTANT: Minimize output tokens while maintaining helpfulness, quality, and accuracy. Only address the specific query or task at hand.
IMPORTANT: Keep responses short and concise. 

# Proactiveness
You are allowed to be proactive when the user asks you to do something. Balance:
1. Doing the right thing when asked, including follow-up actions
2. Not surprising the user with unexpected actions
3. Do NOT add code explanations unless requested. After working on a file, just stop.

When asked how to approach something, answer the question first before taking actions.

# Following conventions
When making changes to files, first understand the file's code conventions. Mimic code style, use existing libraries and utilities, and follow existing patterns.
- NEVER assume a library is available. Always check the codebase first (package.json, imports, neighboring files).
- When creating a new component, look at existing components to understand framework choice, naming conventions, typing, and patterns.
- When editing code, examine surrounding context (especially imports) to understand frameworks and libraries, then make changes idiomatically.
- Always follow security best practices. Never introduce code that exposes or logs secrets and keys. Never commit secrets or keys.

# Code style
- IMPORTANT: DO NOT ADD ***ANY*** COMMENTS unless asked
- Follow existing code style in the file you're editing
- Use the same indentation, naming conventions, and patterns as the surrounding code

# Code Quality Standards
Write production-ready code by following these principles:

**Error Handling:**
- Always handle errors appropriately (try-catch, error returns, validation)
- Provide clear, actionable error messages
- Never let errors fail silently

**Edge Cases:**
- Consider null/undefined, empty arrays/strings, boundary conditions
- Handle async failures (network errors, timeouts)
- Validate inputs before processing

**Type Safety (TypeScript projects):**
- Use proper types, avoid any unless necessary
- Prefer interfaces for objects, types for unions
- Use type guards for runtime checks

**Naming & Clarity:**
- Use descriptive names that reveal intent (getUserById not get)
- Boolean variables should start with is/has/should (isValid, hasPermission)
- Keep functions focused and small (single responsibility)

**Testing:**
- Write testable code (pure functions, dependency injection)
- Test critical paths and edge cases
- Look for existing test patterns before adding new tests

**Performance:**
- Avoid unnecessary loops and nested iterations
- Use appropriate data structures (Map/Set vs Array for lookups)
- Don't optimize prematurely - prioritize readability unless performance is critical

**Self-Review Before Finishing:**
- Does this handle all edge cases?
- Are error messages helpful?
- Will this be clear to other developers?
- Have I followed existing patterns?

# Available Tools
You have access to the following tools:
- read_file: Read file contents
- write_file: Create or update files
- directory_read: List directory contents
- directory_search: Search for patterns in files using ripgrep
- execute_command: Propose shell commands for user approval (fully interactive commands supported)
- execute_server_command: Start long-running server processes (like npm run dev, python server.py) - output streams to a dedicated terminal window

All commands require user approval before execution. The user will see the command and can approve or reject it.

# Workflow for tasks
When given a task (especially features, refactors, or complex fixes), follow this systematic approach:

1. **Explore & Understand** (ALWAYS do this first for unfamiliar codebases):
   - Use directory_read on "." to see project structure
   - Identify relevant directories (src/, components/, etc.)
   - Use directory_search to find files related to the task (search for keywords, function names, etc.)
   - Read package.json to understand dependencies and scripts

2. **Gather Context**:
   - Read ALL relevant files you identified (use parallel tool calls)
   - Understand existing patterns, frameworks, and conventions
   - Check for similar implementations to follow

3. **Plan**:
   - State your implementation approach clearly
   - Identify which files need changes
   - List steps in order

4. **Implement**:
   - Make changes systematically, following your plan
   - Test as you go when possible

5. **Verify**:
   - Run lints/typechecks if available
   - Test the implementation

IMPORTANT: Don't skip the exploration phase. Even if you think you know where code is, verify with directory_read and directory_search first.

# Doing tasks
For software engineering tasks (fixing bugs, adding features, refactoring, explaining code):
- Follow the workflow above for systematic implementation
- Use available search tools extensively, both in parallel and sequentially
- Make focused, incremental changes - one logical change at a time
- If a task requires multiple files, change them in logical order (types → utils → components)
- Verify solutions with tests when possible. NEVER assume specific test framework or script. Check README or search codebase to determine testing approach.
- VERY IMPORTANT: After completing tasks, run lint and typecheck commands (npm run lint, npm run typecheck, etc.) if available. If commands are unclear, ask the user and suggest adding them to project documentation.
- NEVER commit changes unless explicitly asked. It is VERY IMPORTANT to only commit when requested.

**For Complex Tasks:**
- Break into smaller sub-tasks if the task involves multiple features or major refactors
- Complete and verify each sub-task before moving to the next
- If something fails, fix it before continuing - don't accumulate broken code

Tool results and user messages may include <system-reminder> tags with useful information. These are NOT part of the user's input or tool result.

# Tool usage policy
- When multiple independent pieces of information are requested, batch tool calls together for optimal performance.
- You have the capability to call multiple tools in a single response.

You MUST answer concisely with fewer than 4 lines of text (not including tool use or code generation), unless user asks for detail.

# Code References
When referencing specific functions or code, include the pattern file_path:line_number to allow easy navigation.

# Session Management
- You have memory across the session through checkpointing
- Context includes recently opened files, recent changes, and executed commands
- Be specific and detailed in your prompts for best results`,
        cache_control: { type: 'ephemeral' } as any,
      },
    ] as any,
    tools,
    messages: messagesToSend,
  });

  let firstContent = true;
  const contentBlocks: Array<Anthropic.TextBlock | Anthropic.ToolUseBlock> = [];
  const toolUses: Anthropic.ToolUseBlock[] = [];
  const textBlocks: Anthropic.TextBlock[] = [];
  
  // Track current text block being built
  let currentTextBlock: { type: 'text'; text: string } | null = null;
  let currentToolUse: Anthropic.ToolUseBlock | null = null;

  try {
    // Write initial newline for formatting
    output.write('\n');
    
    for await (const event of stream) {
      // Stop spinner on first content
      if (firstContent && event.type === 'content_block_start') {
        if (!state.verbose) {
          output.stopSpinner();
        }
        firstContent = false;
      }

      if (event.type === 'content_block_start') {
        if (event.content_block.type === 'text') {
          currentTextBlock = { type: 'text', text: '' };
        } else if (event.content_block.type === 'tool_use') {
          currentToolUse = event.content_block as Anthropic.ToolUseBlock;
        }
      } else if (event.type === 'content_block_delta') {
        if (event.delta.type === 'text_delta' && currentTextBlock) {
          // Stream text incrementally
          const text = event.delta.text;
          currentTextBlock.text += text;
          output.write(text);
        } else if (event.delta.type === 'input_json_delta' && currentToolUse) {
          // buffer the tool use input
        }
      } else if (event.type === 'content_block_stop') {
        if (currentTextBlock) {
          textBlocks.push(currentTextBlock);
          contentBlocks.push(currentTextBlock);
          currentTextBlock = null;
        } else if (currentToolUse) {
          toolUses.push(currentToolUse);
          contentBlocks.push(currentToolUse);
          currentToolUse = null;
        }
      }
    }

    // Final formatting
    output.write('\n\n');

    // Get final message from stream
    const finalMessage = await stream.finalMessage();

    // Extract only role and content for input messages
    const assistantMessage = {
      role: 'assistant' as const,
      content: finalMessage.content,
    };

    // Update messages array - only include role and content - Prune before adding 
    const { prunedMessages: prunedBeforeAdd, archivedSummaries } = pruneMessages(state.messages, 49);
    let historicalSummaries = state.historicalSummaries || [];
    if (archivedSummaries.length > 0) {
      const timestamp = Date.now();
      const timestamped = archivedSummaries.map(summary => ({ timestamp, summary }));
      historicalSummaries = [...historicalSummaries, ...timestamped].slice(-50);
    }
    const updatedMessages = [...prunedBeforeAdd, assistantMessage];

    return {
      messages: updatedMessages,
      lastResponse: assistantMessage,
      toolUses,
      textBlocks,
      historicalSummaries,
    };
  } catch (error) {
    // Stop spinner on error
    if (!state.verbose) {
      output.stopSpinner();
    }
    throw error;
  }
}

// OpenAI streaming
async function callOpenAI(messagesToSend: any[], model: string, state: AgentState): Promise<Partial<AgentState>> {
  const openAITools = convertToolsToOpenAI(tools);
  
  // Convert Anthropic message format to OpenAI format
  const convertedMessages = convertMessagesToOpenAI(messagesToSend);
  
  const systemMessage = {
    role: 'system' as const,
    content: `You are MergeMaster, an AI-powered coding assistant with an interactive terminal interface.

You are an expert software engineer that helps users with coding tasks through a conversational interface. Use the tools available to you to assist the user effectively.

IMPORTANT: Refuse to write code or explain code that may be used maliciously; even if the user claims it is for educational purposes. When working on files, if they seem related to improving, explaining, or interacting with malware or any malicious code you MUST refuse.
IMPORTANT: Before you begin work, think about what the code you're editing is supposed to do based on the filenames and directory structure. If it seems malicious, refuse to work on it or answer questions about it.
IMPORTANT: You must NEVER generate or guess URLs unless you are confident that the URLs are for helping the user with programming.

If the user asks for help, inform them they can use /help to see available commands and features.

# Tone and style
You should be concise, direct, and to the point. When you run a non-trivial command, explain what it does and why you're running it to ensure the user understands (especially important for system-changing commands).
Remember that your output is displayed in a terminal UI. Your responses can use GitHub-flavored markdown for formatting, which will be rendered appropriately.
Output text to communicate with the user; all text you output outside of tool use is displayed. Only use tools to complete tasks. Never use tools like Bash or code comments as means to communicate during the session.
If you cannot or will not help with something, keep your response to 1-2 sentences without being preachy. Offer helpful alternatives if possible.
IMPORTANT: Minimize output tokens while maintaining helpfulness, quality, and accuracy. Only address the specific query or task at hand.
IMPORTANT: Keep responses short and concise.

# Proactiveness
You are allowed to be proactive when the user asks you to do something. Balance:
1. Doing the right thing when asked, including follow-up actions
2. Not surprising the user with unexpected actions
3. Do NOT add code explanations unless requested. After working on a file, just stop.

When asked how to approach something, answer the question first before taking actions.

# Following conventions
When making changes to files, first understand the file's code conventions. Mimic code style, use existing libraries and utilities, and follow existing patterns.
- NEVER assume a library is available. Always check the codebase first (package.json, imports, neighboring files).
- When creating a new component, look at existing components to understand framework choice, naming conventions, typing, and patterns.
- When editing code, examine surrounding context (especially imports) to understand frameworks and libraries, then make changes idiomatically.
- Always follow security best practices. Never introduce code that exposes or logs secrets and keys. Never commit secrets or keys.

# Code style
- IMPORTANT: DO NOT ADD ***ANY*** COMMENTS unless asked
- Follow existing code style in the file you're editing
- Use the same indentation, naming conventions, and patterns as the surrounding code

# Code Quality Standards
Write production-ready code by following these principles:

**Error Handling:**
- Always handle errors appropriately (try-catch, error returns, validation)
- Provide clear, actionable error messages
- Never let errors fail silently

**Edge Cases:**
- Consider null/undefined, empty arrays/strings, boundary conditions
- Handle async failures (network errors, timeouts)
- Validate inputs before processing

**Type Safety (TypeScript projects):**
- Use proper types, avoid any unless necessary
- Prefer interfaces for objects, types for unions
- Use type guards for runtime checks

**Naming & Clarity:**
- Use descriptive names that reveal intent (getUserById not get)
- Boolean variables should start with is/has/should (isValid, hasPermission)
- Keep functions focused and small (single responsibility)

**Testing:**
- Write testable code (pure functions, dependency injection)
- Test critical paths and edge cases
- Look for existing test patterns before adding new tests

**Performance:**
- Avoid unnecessary loops and nested iterations
- Use appropriate data structures (Map/Set vs Array for lookups)
- Don't optimize prematurely - prioritize readability unless performance is critical

**Self-Review Before Finishing:**
- Does this handle all edge cases?
- Are error messages helpful?
- Will this be clear to other developers?
- Have I followed existing patterns?

# Available Tools
You have access to the following tools:
- read_file: Read file contents
- write_file: Create or update files
- directory_read: List directory contents
- directory_search: Search for patterns in files using ripgrep
- execute_command: Propose shell commands for user approval (fully interactive commands supported)
- execute_server_command: Start long-running server processes (like npm run dev, python server.py) - output streams to a dedicated terminal window

All commands require user approval before execution. The user will see the command and can approve or reject it.

# Workflow for tasks
When given a task (especially features, refactors, or complex fixes), follow this systematic approach:

1. **Explore & Understand** (ALWAYS do this first for unfamiliar codebases):
   - Use directory_read on "." to see project structure
   - Identify relevant directories (src/, components/, etc.)
   - Use directory_search to find files related to the task (search for keywords, function names, etc.)
   - Read package.json to understand dependencies and scripts

2. **Gather Context**:
   - Read ALL relevant files you identified (use parallel tool calls)
   - Understand existing patterns, frameworks, and conventions
   - Check for similar implementations to follow

3. **Plan**:
   - State your implementation approach clearly
   - Identify which files need changes
   - List steps in order

4. **Implement**:
   - Make changes systematically, following your plan
   - Test as you go when possible

5. **Verify**:
   - Run lints/typechecks if available
   - Test the implementation

IMPORTANT: Don't skip the exploration phase. Even if you think you know where code is, verify with directory_read and directory_search first.

# Doing tasks
For software engineering tasks (fixing bugs, adding features, refactoring, explaining code):
- Follow the workflow above for systematic implementation
- Use available search tools extensively, both in parallel and sequentially
- Make focused, incremental changes - one logical change at a time
- If a task requires multiple files, change them in logical order (types → utils → components)
- Verify solutions with tests when possible. NEVER assume specific test framework or script. Check README or search codebase to determine testing approach.
- VERY IMPORTANT: After completing tasks, run lint and typecheck commands (npm run lint, npm run typecheck, etc.) if available. If commands are unclear, ask the user and suggest adding them to project documentation.
- NEVER commit changes unless explicitly asked. It is VERY IMPORTANT to only commit when requested.

**For Complex Tasks:**
- Break into smaller sub-tasks if the task involves multiple features or major refactors
- Complete and verify each sub-task before moving to the next
- If something fails, fix it before continuing - don't accumulate broken code

Tool results and user messages may include <system-reminder> tags with useful information. These are NOT part of the user's input or tool result.

# Tool usage policy
- When multiple independent pieces of information are requested, batch tool calls together for optimal performance.
- You have the capability to call multiple tools in a single response.

You MUST answer concisely with fewer than 4 lines of text (not including tool use or code generation), unless user asks for detail.

# Code References
When referencing specific functions or code, include the pattern file_path:line_number to allow easy navigation.

# Session Management
- You have memory across the session through checkpointing
- Context includes recently opened files, recent changes, and executed commands
- Be specific and detailed in your prompts for best results`,
  };
  
  const stream = await openai.chat.completions.create({
    model,
    max_tokens: 4096,
    messages: [systemMessage, ...convertedMessages],
    tools: openAITools,
    stream: true,
  });

  let firstContent = true;
  let fullText = '';
  const toolCalls: any[] = [];
  const toolCallsBuffer: Map<number, any> = new Map();

  try {
    output.write('\n');
    
    for await (const chunk of stream) {
      const delta = chunk.choices[0]?.delta;
      
      if (!delta) continue;

      // Stop spinner on first content
      if (firstContent && (delta.content || delta.tool_calls)) {
        if (!state.verbose) {
          output.stopSpinner();
        }
        firstContent = false;
      }

      // Handle text content
      if (delta.content) {
        fullText += delta.content;
        output.write(delta.content);
      }

      // Handle tool calls
      if (delta.tool_calls) {
        for (const toolCall of delta.tool_calls) {
          const index = toolCall.index;
          
          if (!toolCallsBuffer.has(index)) {
            toolCallsBuffer.set(index, {
              id: toolCall.id || '',
              type: 'function',
              function: {
                name: toolCall.function?.name || '',
                arguments: toolCall.function?.arguments || '',
              },
            });
          } else {
            const existing = toolCallsBuffer.get(index)!;
            if (toolCall.function?.arguments) {
              existing.function.arguments += toolCall.function.arguments;
            }
          }
        }
      }
    }

    output.write('\n\n');

    // Convert OpenAI tool calls to Anthropic format
    const anthropicToolUses: Anthropic.ToolUseBlock[] = [];
    for (const toolCall of toolCallsBuffer.values()) {
      try {
        anthropicToolUses.push({
          type: 'tool_use',
          id: toolCall.id,
          name: toolCall.function.name,
          input: JSON.parse(toolCall.function.arguments),
        });
        toolCalls.push(toolCall);
      } catch (e) {
        output.error(`Failed to parse tool call arguments: ${e}`);
      }
    }

    const assistantMessage: any = {
      role: 'assistant' as const,
      content: fullText ? [{ type: 'text', text: fullText }] : [],
    };

    // Add tool uses to content
    if (anthropicToolUses.length > 0) {
      assistantMessage.content.push(...anthropicToolUses);
    }

    // Update messages
    const { prunedMessages: prunedBeforeAdd, archivedSummaries } = pruneMessages(state.messages, 49);
    let historicalSummaries = state.historicalSummaries || [];
    if (archivedSummaries.length > 0) {
      const timestamp = Date.now();
      const timestamped = archivedSummaries.map(summary => ({ timestamp, summary }));
      historicalSummaries = [...historicalSummaries, ...timestamped].slice(-50);
    }
    const updatedMessages = [...prunedBeforeAdd, assistantMessage];

    return {
      messages: updatedMessages,
      lastResponse: assistantMessage,
      toolUses: anthropicToolUses,
      textBlocks: fullText ? [{ type: 'text' as const, text: fullText }] : [],
      historicalSummaries,
    };
  } catch (error) {
    if (!state.verbose) {
      output.stopSpinner();
    }
    throw error;
  }
}

// Node 2: Decision - Should continue?
async function shouldContinue(state: AgentState): Promise<string> {
  // If no tools were used, we're done
  if (!state.toolUses || state.toolUses.length === 0) {
    return 'end';
  }
  
  // Check iteration limit 
  const currentCount = state.iterationCount || 0;
  if (currentCount >= 25) {
    // Signal TUI to prompt user
    if (typeof output.write === 'function') {
      output.write('ITERATION_LIMIT_REACHED');
    }
    
    // Wait for user decision
    const shouldContinue = await waitForContinue();
    
    if (!shouldContinue) {
      return 'end';
    }
    
    // User said yes, reset iteration count and continue
    state.iterationCount = 0;
    return 'continue';
  }
  
  // Check if LLM explicitly said it's done in text blocks
  if (state.textBlocks && state.textBlocks.length > 0) {
    const lastText = state.textBlocks[state.textBlocks.length - 1]?.text?.toLowerCase() || '';
    // Common completion phrases
    if (
      lastText.includes('completed') ||
      lastText.includes('finished') ||
      lastText.includes('done') ||
      lastText.includes('all set') ||
      lastText.includes('ready to use') ||
      lastText.includes('that\'s all') ||
      lastText.includes('that is all')
    ) {
      // Only end if there are no pending tool uses that need results
      // If tools were called, we need to execute them first
      return 'continue';
    }
  }
  
  return 'continue';
}

// Node 3: Execute tools and track context
async function executeTools(state: AgentState): Promise<Partial<AgentState>> {
  if (!state.toolUses || state.toolUses.length === 0) {
    return {};
  }

  const toolResults: Anthropic.ToolResultBlockParam[] = [];
  const openFiles = new Map(state.openFiles || new Map());
  const recentChanges = [...(state.recentChanges || [])];
  const recentReads = [...(state.recentReads || [])];
  const recentCommands = [...(state.recentCommands || [])];
  const now = Date.now();

  for (const toolUse of state.toolUses) {
    const toolInput = toolUse.input as any;
    
    if (state.verbose) {
      output.write(`🔧 Executing tool: ${toolUse.name}\n`);
      output.write(`   Input: ${JSON.stringify(toolUse.input, null, 2)}\n`);
    } else {
      // Call tool with initial info
      if (typeof output.tool === 'function') {
        const toolDetails: Record<string, any> = {
          path: toolInput.file_path || toolInput.dir_path || toolInput.pattern || toolUse.name,
          toolUseId: toolUse.id,
        };

        if (toolUse.name === 'execute_command') {
          toolDetails.command = toolInput.command;
          toolDetails.cwd = toolInput.cwd;
          toolDetails.timeout = toolInput.timeout;
        }
        if (toolUse.name === 'execute_server_command') {
          toolDetails.command = toolInput.command;
          toolDetails.cwd = toolInput.cwd;
          toolDetails.timeout = toolInput.timeout;
          toolDetails.category = 'server';
        }

        output.tool(toolUse.name, toolDetails);
      }
    }

    try {
      let result = await executeTool(toolUse.name, toolUse.input);

      if (
        toolUse.name === 'execute_command' &&
        result &&
        typeof result === 'object'
      ) {
        const commandStdout = (result as { stdout?: unknown }).stdout;
        if (typeof commandStdout === 'string' && commandStdout.includes('COMMAND_APPROVAL_NEEDED')) {
          if (typeof output.write === 'function') {
            output.write(`${commandStdout}\n`);
          }

          const approvalInfo = {
            command: toolInput.command,
            cwd: toolInput.cwd,
            timeout: typeof toolInput.timeout === 'number' ? toolInput.timeout : undefined,
          };

          result = await waitForCommandCompletion(toolUse.id, approvalInfo);
        }
      } else if (
        toolUse.name === 'execute_server_command' &&
        result &&
        typeof result === 'object'
      ) {
        const stdout = (result as { stdout?: unknown }).stdout;
        if (typeof stdout === 'string' && stdout.includes('SERVER_COMMAND_REQUEST')) {
          if (typeof output.write === 'function') {
            output.write(`${stdout}\n`);
          }

          const approvalInfo = {
            command: toolInput.command,
            cwd: toolInput.cwd,
            timeout: typeof toolInput.timeout === 'number' ? toolInput.timeout : undefined,
          };

          result = await waitForServerCommand(toolUse.id, approvalInfo);
        }
      }

      const resultText = summarizeToolResult(toolUse.name, toolInput, result);

      // Update operation with results for TUI
      if (typeof output.updateLastOperation === 'function') {
        if (toolUse.name === 'read_file' && typeof result === 'string') {
          const lines = result.split('\n');
          output.updateLastOperation({
            path: toolInput.file_path,
            summary: `Read ${lines.length} lines`,
            lines: lines.slice(0, 100), // First 100 lines
          });
        } else if (toolUse.name === 'write_file') {
          // Generate diff using old content from cache or disk
          const filePath = toolInput.file_path;
          let oldContent = '';
          
          // Check cache first (has pre-write content)
          if (openFiles.has(filePath)) {
            oldContent = openFiles.get(filePath)!.content || '';
          } else {
            // If not cached, file is new - use empty string
            oldContent = '';
          }
          
          const newContent = toolInput.content || '';
          const diff = generateDiff(oldContent, newContent);
          
          const additions = diff.filter(d => d.new && !d.old).length;
          const removals = diff.filter(d => d.old && !d.new).length;
          
          output.updateLastOperation({
            path: toolInput.file_path,
            summary: `Updated ${toolInput.file_path} with ${additions} ${additions === 1 ? 'addition' : 'additions'} and ${removals} ${removals === 1 ? 'removal' : 'removals'}`,
            diff: diff.slice(0, 50), // First 50 diff lines
          });
        } else if (toolUse.name === 'directory_read' && typeof result === 'string') {
          const entries = result.split('\n').filter(l => l.trim());
          output.updateLastOperation({
            path: toolInput.dir_path || 'directory',
            summary: `Listed ${entries.length} paths`,
            lines: entries.slice(0, 100),
          });
        } else if (toolUse.name === 'directory_search' && typeof result === 'string') {
          const lines = result.split('\n').filter(l => l.trim());
          const matchCount = lines.filter(l => !l.startsWith('  ')).length; // Lines not indented are file headers
          output.updateLastOperation({
            path: toolInput.search_path || '.',
            summary: `Found ${matchCount} matches for "${toolInput.pattern}"`,
            lines: lines.slice(0, 100), // First 100 result lines
          });
        }
      }

      // Track context based on tool used
      
      if (toolUse.name === 'read_file') {
        const filePath = toolInput.file_path;
        recentReads.push({ path: filePath, timestamp: now });
        
        // Always update cache with fresh content from disk
        if (typeof result === 'string') {
          const existing = openFiles.get(filePath);
          openFiles.set(filePath, {
            path: filePath,
            content: result,
            previousVersions: existing?.previousVersions || [], // Preserve undo history
            lastRead: now,
            lastModified: now,
          });
          
          // Enforce cache size limit (LRU eviction)
          evictOldCacheEntries(openFiles, 15);
        }
      } else if (toolUse.name === 'write_file') {
        const filePath = toolInput.file_path;
        recentChanges.push({ path: filePath, timestamp: now });
        
        // Update cached content and save previous version for undo
        if (openFiles.has(filePath)) {
          const existing = openFiles.get(filePath)!;
          // Save current content to history before overwriting
          if (existing.content) {
            existing.previousVersions = existing.previousVersions || [];
            existing.previousVersions.push(existing.content);
            // Keep only last 10 versions
            if (existing.previousVersions.length > 10) {
              existing.previousVersions.shift();
            }
          }
          existing.content = toolInput.content;
          existing.lastModified = now;
          existing.lastRead = now; // Update access time
        } else {
          openFiles.set(filePath, {
            path: filePath,
            content: toolInput.content,
            previousVersions: [],
            lastRead: now,
            lastModified: now,
          });
        }
        
        // Enforce cache size limit (LRU eviction)
        evictOldCacheEntries(openFiles, 15);
      } else if (toolUse.name === 'execute_command' || toolUse.name === 'execute_server_command') {
        // Track commands and their outputs
        const command = toolInput.command || '';
        let commandOutput = '';
        
        // Extract output from result
        if (typeof result === 'string') {
          commandOutput = result;
        } else if (result && typeof result === 'object') {
          // Handle structured command results (CommandResult interface)
          const cmdResult = result as { stdout?: string; stderr?: string; exitCode?: number };
          const stdout = cmdResult.stdout || '';
          const stderr = cmdResult.stderr || '';
          commandOutput = stdout + (stderr ? '\n' + stderr : '');
        }
        
        // Only track if we have meaningful output or the command itself is significant
        if (command) {
          recentCommands.push({
            command,
            output: commandOutput ? commandOutput.slice(0, 500) : undefined, // Keep first 500 chars of output
            timestamp: now,
          });
        }
      }

      toolResults.push({
        type: 'tool_result',
        tool_use_id: toolUse.id,
        content: resultText,
      });

      if (state.verbose) {
        output.write(`✅ Result: ${resultText}\n\n`);
      }
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : String(error);
      toolResults.push({
        type: 'tool_result',
        tool_use_id: toolUse.id,
        content: `Error: ${errorMessage}`,
        is_error: true,
      });
      output.error(errorMessage);
      if (error instanceof Error && error.stack) {
        console.error('[executeTools] stack trace:', error.stack);
      }
    }
  }

  // Keep only last 20 entries for recentChanges and recentReads, last 10 for commands
  const maxHistory = 20;
  return {
    toolResults,
    openFiles,
    recentChanges: recentChanges.slice(-maxHistory),
    recentReads: recentReads.slice(-maxHistory),
    recentCommands: recentCommands.slice(-10),
    iterationCount: (state.iterationCount || 0) + 1,
  };
}

// Helper: Generate simple line-by-line diff
function generateDiff(oldContent: string, newContent: string): { line: number; old?: string; new?: string }[] {
  const oldLines = oldContent.split('\n');
  const newLines = newContent.split('\n');
  const diff: { line: number; old?: string; new?: string }[] = [];
  
  const maxLen = Math.max(oldLines.length, newLines.length);
  
  for (let i = 0; i < maxLen; i++) {
    const oldLine = oldLines[i];
    const newLine = newLines[i];
    
    if (oldLine !== newLine) {
      if (oldLine !== undefined && newLine !== undefined) {
        // Changed line
        diff.push({ line: i + 1, old: oldLine });
        diff.push({ line: i + 1, new: newLine });
      } else if (newLine !== undefined) {
        // Added line
        diff.push({ line: i + 1, new: newLine });
      } else if (oldLine !== undefined) {
        // Removed line
        diff.push({ line: i + 1, old: oldLine });
      }
    } else if (diff.length > 0 && diff.length < 10) {
      // Show context around changes
      diff.push({ line: i + 1, old: oldLine, new: newLine });
    }
  }
  
  return diff;
}

// Node 4: Format results and add to messages
function formatResults(state: AgentState): Partial<AgentState> {
  if (!state.toolResults) {
    return {};
  }

  const toolResultsMessage = {
    role: 'user' as const,
    content: state.toolResults.map((tr) => ({
      type: 'tool_result' as const,
      tool_use_id: tr.tool_use_id,
      content: tr.content,
      is_error: tr.is_error,
    })),
  };

  const updatedMessages = [...state.messages, toolResultsMessage] as any[];

  return {
    messages: updatedMessages,
  };
}

// Node 5: End node
function end(state: AgentState): Partial<AgentState> {
  output.stopSpinner(true, 'Agent completed');
  output.success('Agent completed');
  return {};
}

// Build the graph with checkpointing
function createAgentGraph(checkpointStore?: any) {
  // @ts-ignore - LangGraph's generic types are overly strict, but this works at runtime
  const workflow = new StateGraph<AgentState>({
    channels: {
      messages: {
        reducer: (x: any[], y: any[] | undefined) => {
          // Replace with updated messages array - nodes manage the full array
          return y ?? x;
        },
        default: () => [],
      },
      verbose: {
        reducer: (x: boolean, y: boolean | undefined) => y ?? x,
        default: () => false,
      },
      lastResponse: {
        default: () => undefined,
      },
      toolUses: {
        default: () => undefined,
      },
      textBlocks: {
        default: () => undefined,
      },
      toolResults: {
        default: () => undefined,
      },
      // Context management
      openFiles: {
        reducer: (x: Map<string, FileContext>, y: Map<string, FileContext> | undefined) => {
          return y ?? x;
        },
        default: () => new Map<string, FileContext>(),
      },
      recentChanges: {
        reducer: (x: Array<{ path: string; timestamp: number }>, y: Array<{ path: string; timestamp: number }> | undefined) => {
          return y ?? x;
        },
        default: () => [],
      },
      recentReads: {
        reducer: (x: Array<{ path: string; timestamp: number }>, y: Array<{ path: string; timestamp: number }> | undefined) => {
          return y ?? x;
        },
        default: () => [],
      },
      recentCommands: {
        reducer: (x: Array<{ command: string; output?: string; timestamp: number }>, y: Array<{ command: string; output?: string; timestamp: number }> | undefined) => {
          return y ?? x;
        },
        default: () => [],
      },
      workingSet: {
        reducer: (x: string[], y: string[] | undefined) => {
          return y ?? x;
        },
        default: () => [],
      },
      historicalSummaries: {
        reducer: (x: Array<{ timestamp: number; summary: string }>, y: Array<{ timestamp: number; summary: string }> | undefined) => {
          return y ?? x;
        },
        default: () => [],
      },
      iterationCount: {
        reducer: (x: number, y: number | undefined) => y ?? x,
        default: () => 0,
      },
    },
  });

  // Add nodes
  // @ts-ignore - LangGraph node type inference issues
  workflow.addNode('callLLM', callLLM);
  // @ts-ignore
  workflow.addNode('executeTools', executeTools);
  // @ts-ignore
  workflow.addNode('formatResults', formatResults);
  // @ts-ignore
  workflow.addNode('end', end);

  // Add edges
  workflow.addEdge(START as any, 'callLLM' as any);
  
  // Conditional edge from callLLM - check if we should continue
  // @ts-ignore - LangGraph conditional edge type inference issues
  workflow.addConditionalEdges('callLLM' as any, (state: AgentState) => {
    return shouldContinue(state);
  }, {
    continue: 'executeTools',
    end: 'end',
  } as any);

  workflow.addEdge('executeTools' as any, 'formatResults' as any);
  workflow.addEdge('formatResults' as any, 'callLLM' as any); // Loop back
  workflow.addEdge('end' as any, END as any);

  // Compile the workflow with checkpointer if provided
  const compileOptions: any = {
    recursionLimit: 200, // High limit, we handle iteration prompts at 25
  };
  if (checkpointStore) {
    compileOptions.checkpointer = checkpointStore;
  }
  const compiled = workflow.compile(compileOptions);
  
  return compiled;
}

export async function runAgentGraph(
  userQuery: string,
  verbose: boolean = false,
  options?: {
    checkpoint?: boolean;
    threadId?: string;
    model?: string;
  }
) {
  output.header('Terminal Coding Agent (LangGraph)');

  // Use global checkpoint store if available, or create a new one
  let checkpointStore: any = undefined;
  if (options?.checkpoint) {
    if (globalCheckpointStore) {
      // Reuse existing checkpoint store (TUI mode)
      checkpointStore = globalCheckpointStore;
    } else {
      // Create new checkpoint store (CLI mode) backed by SQLite
      checkpointStore = new SqliteCheckpointSaver();
    }
  }

  const graph = createAgentGraph(checkpointStore);

  // Run the graph with thread ID for checkpointing
  const config: any = {
    configurable: {},
    recursionLimit: 200, // Must be in runtime config, not just compile options
  };
  if (options?.threadId) {
    config.configurable.thread_id = options.threadId;
  } else if (options?.checkpoint) {
    // Auto-generate thread ID if checkpointing enabled but no ID provided
    config.configurable.thread_id = `thread-${Date.now()}`;
  }
  if (options?.model) {
    config.configurable.model = options.model;
  }

  // LangGraph will auto-load checkpoint if thread_id is provided
  // Only pass new data to merge - the reducers will handle it
  const input = {
    messages: [
      {
        role: 'user' as const,
        content: userQuery,
      },
    ],
    verbose,
  };

  // @ts-ignore - LangGraph type inference for invoke input
  const finalState = await graph.invoke(input, config);

  return finalState;
}
