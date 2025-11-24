import React, { useState, useCallback, useRef, useEffect, useMemo } from 'react';
import { render, Text, Box, useInput, useApp, Static } from 'ink';
import Spinner from 'ink-spinner';
import { runAgentGraph, setOutputHandler, setCheckpointStore, undoLastChange } from './agent_graph.js';
import { SqliteCheckpointSaver } from './checkpoint_store.js';
import {
  CommandPTYSession,
  executeCommandWithPTY,
  resolvePendingCommand,
  rejectPendingCommand,
  resolvePendingServerCommand,
  rejectPendingServerCommand,
  resolvePendingContinue,
  rejectPendingContinue,
} from './tools/Execute_commands.js';

const ANSI_ESCAPE_REGEX = /\u001b\[[0-9;?]*[ -\/]*[@-~]/g;
const OSC_ESCAPE_REGEX = /\u001b\][^\u0007]*(?:\u0007|\u001b\\)/g;
const CONTROL_CHAR_REGEX = /[\u0000-\u0008\u000b-\u000c\u000e-\u001f\u007f]/g;
const CLEAR_SEQUENCE_REGEX = /\u001b\[[012]?J|\u001b\[H|\u001b\[2K/g;
// Unicode Braille patterns used for spinners (U+2800-U+28FF)
const SPINNER_CHAR_REGEX = /[\u2800-\u28ff]/g;

// Available models for selection
const AVAILABLE_MODELS = [
  { 
    id: 'gpt-5.1-2025-11-13', 
    name: 'GPT-5.1', 
    description: 'Latest OpenAI model with advanced reasoning',
    provider: 'openai'
  },
  { 
    id: 'gpt-4o', 
    name: 'GPT-4o', 
    description: 'Fast and capable OpenAI model',
    provider: 'openai'
  },
  { 
    id: 'gpt-4o-mini', 
    name: 'GPT-4o Mini (cheapest)', 
    description: 'Lightweight and affordable OpenAI model',
    provider: 'openai'
  },
  { 
    id: 'claude-sonnet-4-20250514', 
    name: 'Claude Sonnet 4.5', 
    description: 'Balanced performance and speed from Anthropic',
    provider: 'anthropic'
  },
  { 
    id: 'claude-opus-4-20250514', 
    name: 'Claude Opus 4.1 (Most Expensive, best performance)', 
    description: 'Most capable Anthropic model, highest quality',
    provider: 'anthropic'
  },
  { 
    id: 'claude-3-5-haiku-20241022', 
    name: 'Claude Haiku 3.5', 
    description: 'Fastest and most affordable Anthropic model',
    provider: 'anthropic'
  },
];

type MarkdownBlock =
  | { type: 'heading'; level: number; text: string }
  | { type: 'paragraph'; text: string }
  | { type: 'list'; ordered: boolean; items: Array<{ marker: string; text: string }> }
  | { type: 'code'; code: string; lang?: string }
  | { type: 'quote'; text: string };

interface InlineSegment {
  type: 'text' | 'bold' | 'italic' | 'code' | 'newline';
  content?: string;
}

function parseMarkdownBlocks(content: string): MarkdownBlock[] {
  const blocks: MarkdownBlock[] = [];
  const normalized = content.replace(/\r\n/g, '\n');
  const lines = normalized.split('\n');

  let paragraphLines: string[] = [];
  let listItems: Array<{ marker: string; text: string }> = [];
  let listType: 'unordered' | 'ordered' | null = null;
  let inCodeBlock = false;
  let codeLang = '';
  let codeLines: string[] = [];
  let quoteLines: string[] = [];

  const flushParagraph = () => {
    if (paragraphLines.length > 0) {
      const text = paragraphLines.join('\n').trim();
      if (text) {
        blocks.push({ type: 'paragraph', text });
      }
      paragraphLines = [];
    }
  };

  const flushList = () => {
    if (listItems.length > 0) {
      blocks.push({
        type: 'list',
        ordered: listType === 'ordered',
        items: listItems.map(item => ({ ...item })),
      });
      listItems = [];
      listType = null;
    }
  };

  const flushQuote = () => {
    if (quoteLines.length > 0) {
      const text = quoteLines.join('\n').trim();
      if (text) {
        blocks.push({ type: 'quote', text });
      }
      quoteLines = [];
    }
  };

  const flushCode = () => {
    if (codeLines.length > 0) {
      blocks.push({
        type: 'code',
        code: codeLines.join('\n'),
        lang: codeLang || undefined,
      });
      codeLines = [];
      codeLang = '';
    }
  };

  for (const rawLine of lines) {
    const line = rawLine.replace(/\s+$/, '');

    if (inCodeBlock) {
      if (line.trim().startsWith('```')) {
        flushCode();
        inCodeBlock = false;
        continue;
      }
      codeLines.push(rawLine);
      continue;
    }

    if (quoteLines.length > 0 && !line.trim().startsWith('>') && line.trim() !== '') {
      flushQuote();
    }

    if (line.trim().startsWith('```')) {
      flushParagraph();
      flushList();
      flushQuote();
      inCodeBlock = true;
      codeLang = line.trim().slice(3).trim();
      codeLines = [];
      continue;
    }

    if (line.trim() === '') {
      flushParagraph();
      flushList();
      flushQuote();
      continue;
    }

    const headingMatch = line.match(/^(#{1,6})\s+(.*)$/);
    if (headingMatch) {
      flushParagraph();
      flushList();
      flushQuote();
      blocks.push({
        type: 'heading',
        level: headingMatch[1].length,
        text: headingMatch[2].trim(),
      });
      continue;
    }

    const quoteMatch = line.match(/^\s*>\s?(.*)$/);
    if (quoteMatch) {
      flushParagraph();
      flushList();
      quoteLines.push(quoteMatch[1]);
      continue;
    }

    const unorderedMatch = line.match(/^\s*[-*+]\s+(.*)$/);
    if (unorderedMatch) {
      flushParagraph();
      flushQuote();
      if (listType !== 'unordered') {
        flushList();
        listType = 'unordered';
      }
      listItems.push({ marker: '•', text: unorderedMatch[1] });
      continue;
    }

    const orderedMatch = line.match(/^\s*(\d+)\.\s+(.*)$/);
    if (orderedMatch) {
      flushParagraph();
      flushQuote();
      if (listType !== 'ordered') {
        flushList();
        listType = 'ordered';
      }
      listItems.push({ marker: `${orderedMatch[1]}.`, text: orderedMatch[2] });
      continue;
    }

    flushList();
    flushQuote();
    paragraphLines.push(line.trim());
  }

  if (inCodeBlock) {
    flushCode();
  }
  flushParagraph();
  flushList();
  flushQuote();

  return blocks;
}

function parseInlineSegments(text: string): InlineSegment[] {
  const segments: InlineSegment[] = [];
  const lines = text.split('\n');

  for (let lineIndex = 0; lineIndex < lines.length; lineIndex++) {
    const line = lines[lineIndex];
    const pattern = /(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)/g;
    let lastIndex = 0;
    let match: RegExpExecArray | null;

    while ((match = pattern.exec(line)) !== null) {
      if (match.index > lastIndex) {
        segments.push({
          type: 'text',
          content: line.slice(lastIndex, match.index),
        });
      }
      const token = match[0];
      if (token.startsWith('**')) {
        segments.push({ type: 'bold', content: token.slice(2, -2) });
      } else if (token.startsWith('*')) {
        segments.push({ type: 'italic', content: token.slice(1, -1) });
      } else if (token.startsWith('`')) {
        segments.push({ type: 'code', content: token.slice(1, -1) });
      }
      lastIndex = pattern.lastIndex;
    }

    if (lastIndex < line.length) {
      segments.push({
        type: 'text',
        content: line.slice(lastIndex),
      });
    }

    if (lineIndex < lines.length - 1) {
      segments.push({ type: 'newline' });
    }
  }

  if (segments.length === 0) {
    segments.push({ type: 'text', content: text });
  }

  return segments;
}

const MarkdownText: React.FC<{ content: string }> = React.memo(({ content }) => {
  type InkTextProps = React.ComponentProps<typeof Text>;
  const blocks = useMemo(() => parseMarkdownBlocks(content), [content]);

  const renderInline = (text: string, keyPrefix: string, textProps?: InkTextProps) => {
    const segments = parseInlineSegments(text);
    return (
      <Text wrap="wrap" {...textProps}>
        {segments.map((segment, idx) => {
          const key = `${keyPrefix}-${idx}`;
          if (segment.type === 'newline') {
            return <Text key={key}>{'\n'}</Text>;
          }
          if (segment.type === 'bold') {
            return (
              <Text key={key} bold>
                {segment.content}
              </Text>
            );
          }
          if (segment.type === 'italic') {
            return (
              <Text key={key} italic>
                {segment.content}
              </Text>
            );
          }
          if (segment.type === 'code') {
            return (
              <Text key={key} backgroundColor="#303030" color="#f5f5f5">
                {' '}{segment.content}{' '}
              </Text>
            );
          }
          return (
            <Text key={key}>
              {segment.content}
            </Text>
          );
        })}
      </Text>
    );
  };

  return (
    <Box flexDirection="column">
      {blocks.map((block, idx) => {
        const isLast = idx === blocks.length - 1;
        const marginBottom = isLast ? 0 : 1;

        if (block.type === 'heading') {
          const colors = ['cyan', 'cyan', 'green', 'green', 'yellow', 'yellow'];
          const color = colors[Math.min(block.level - 1, colors.length - 1)];
          return (
            <Box key={`heading-${idx}`} marginBottom={marginBottom}>
              {renderInline(block.text, `heading-${idx}`, { color, bold: true })}
            </Box>
          );
        }

        if (block.type === 'list') {
          return (
            <Box key={`list-${idx}`} flexDirection="column" marginBottom={marginBottom}>
              {block.items.map((item, itemIdx) => (
                <Box key={`list-${idx}-${itemIdx}`} flexDirection="row">
                  <Text color="cyan">{block.ordered ? `${item.marker} ` : '• '}</Text>
                  {renderInline(item.text, `list-${idx}-${itemIdx}`)}
                </Box>
              ))}
            </Box>
          );
        }

        if (block.type === 'code') {
          const lines = block.code.split('\n');
          return (
            <Box
              key={`code-${idx}`}
              flexDirection="column"
              borderStyle="round"
              borderColor="gray"
              paddingX={1}
              paddingY={0}
              marginBottom={marginBottom}
            >
              {block.lang && (
                <Box>
                  <Text color="gray" dimColor>
                    {block.lang}
                  </Text>
                </Box>
              )}
              {lines.map((line, lineIdx) => (
                <Text key={`code-${idx}-${lineIdx}`} color="#f5f5f5">
                  {line || ' '}
                </Text>
              ))}
            </Box>
          );
        }

        if (block.type === 'quote') {
          return (
            <Box
              key={`quote-${idx}`}
              flexDirection="column"
              borderStyle="single"
              borderColor="gray"
              paddingLeft={1}
              paddingY={0}
              marginBottom={marginBottom}
            >
              <Text color="gray">{'>'}</Text>
              <Box marginLeft={1}>
                {renderInline(block.text, `quote-${idx}`, { color: 'gray' })}
              </Box>
            </Box>
          );
        }

        return (
          <Box key={`paragraph-${idx}`} marginBottom={marginBottom}>
            {renderInline(block.text, `paragraph-${idx}`)}
          </Box>
        );
      })}
    </Box>
  );
});

function isLikelyShellPrompt(line: string): boolean {
  const trimmed = line.trim();
  if (!trimmed) return false;
  if (trimmed.length > 200) return false;
  if (!/[#$%]$/.test(trimmed)) return false;

  // Ignore percentages like "100%"
  if (/^\d{1,3}%$/.test(trimmed)) return false;

  // Must have whitespace before the prompt symbol to avoid catching e.g. git percent outputs
  if (!/\s[#$%]$/.test(trimmed)) return false;

  const withoutPrompt = trimmed.slice(0, -1).trim();
  if (!withoutPrompt) return false;

  // Heuristics: typical shell prompt elements
  if (
    withoutPrompt.includes('@') ||
    withoutPrompt.includes('~') ||
    withoutPrompt.includes('/') ||
    withoutPrompt.includes('\\') ||
    withoutPrompt.includes(':')
  ) {
    return true;
  }

  // Prompts often have two tokens (user host / directory)
  const tokenCount = withoutPrompt.split(/\s+/).length;
  if (tokenCount <= 3) {
    return true;
  }

  return false;
}

function normalizeCarriageReturns(text: string): string {
  return text
    .split('\n')
    .map(segment => {
      if (segment.includes('\r')) {
        const parts = segment.split('\r');
        return parts[parts.length - 1];
      }
      return segment;
    })
    .join('\n');
}

function sanitizeCommandChunk(chunk: string): string {
  if (!chunk) return '';
  let cleaned = chunk
    .replace(/\r\n/g, '\n')
    .replace(ANSI_ESCAPE_REGEX, '')
    .replace(OSC_ESCAPE_REGEX, '')
    .replace(CONTROL_CHAR_REGEX, '')
    .replace(SPINNER_CHAR_REGEX, ''); // Remove spinner characters

  cleaned = normalizeCarriageReturns(cleaned);
  cleaned = cleaned.replace(/\r/g, '');

  const lines = cleaned.split('\n');
  const filtered = lines.filter((line, idx) => {
    if (isLikelyShellPrompt(line)) {
      return false;
    }
    if (idx === lines.length - 1 && cleaned && !cleaned.endsWith('\n')) {
      return true;
    }
    return true;
  });

  return filtered.join('\n');
}

interface Operation {
  type: 'list' | 'read' | 'write' | 'search' | 'command' | 'server';
  name: string;
  path: string;
  summary?: string;
  lines?: string[];
  diff?: { line: number; old?: string; new?: string }[];
  expanded?: boolean;
  commandState?: CommandState;
  toolUseId?: string;
  serverState?: ServerSession;
}

interface CommandState {
  command: string;
  cwd?: string;
  timeout?: number;
  status: 'pending' | 'running' | 'success' | 'error' | 'cancelled';
  output: string;
  exitCode?: number;
  error?: string;
  timedOut?: boolean;
  isServer?: boolean;
}

interface PendingCommandState {
  command: string;
  cwd?: string;
  timeout?: number;
  selected: number;
  turnIndex: number;
  eventIndex: number;
  toolUseId?: string;
  isServer?: boolean;
}

interface PendingContinueState {
  selected: number; // 0 = Yes, 1 = No
  turnIndex: number;
}

const MAX_COMMAND_OUTPUT_LENGTH = 8000;

interface TimelineEvent {
  type: 'operation' | 'text';
  timestamp: number;
  operation?: Operation;
  text?: string;
}

interface Turn {
  userMessage: string;
  timeline: TimelineEvent[];
  isProcessing: boolean;
  progressText?: string;
  timeElapsed?: number;
  tokens?: number;
}

interface ServerSession {
  id: string;
  command: string;
  cwd?: string;
  status: 'pending' | 'starting' | 'running' | 'stopped' | 'error';
  output: string;
  exitCode?: number;
  startedAt?: number;
  endedAt?: number;
  error?: string;
}

interface AgentUIProps {
  verbose?: boolean;
  checkpoint?: boolean;
  sessionId: string;
  resume?: boolean;
}

function CommandOperationItem({
  op,
  isPending,
  pendingSelection,
  isActive,
}: {
  op: Operation;
  isPending: boolean;
  pendingSelection?: number;
  isActive: boolean;
}) {
  const state: CommandState = op.commandState ?? {
    command: op.path,
    status: 'pending',
    output: '',
  };

  const status = state.status ?? 'pending';
  // Show approval UI if status is pending OR if isPending prop is true
  const showApproval = status === 'pending' || isPending;

  const statusStyles: Record<CommandState['status'], { label: string; color: string }> = {
    pending: { label: '[?] Awaiting approval', color: 'yellow' },
    running: { label: '[>] Running', color: 'cyan' },
    success: { label: '[ok] Completed', color: 'green' },
    error: { label: '[!] Failed', color: 'red' },
    cancelled: { label: '[x] Cancelled', color: 'gray' },
  };

  const { label, color } = statusStyles[status];
  const cleanedOutput = (state.output || '').replace(/\r/g, '');
  const outputLines = cleanedOutput ? cleanedOutput.split('\n') : [];

  return (
    <Box
      flexDirection="column"
      borderStyle="round"
      borderColor={color}
      paddingX={1}
      paddingY={1}
    >
      <Box>
        <Text color={color} bold>{label}</Text>
        <Text> {state.command || op.path}</Text>
      </Box>

      {state.cwd && (
        <Box marginTop={1}>
          <Text color="gray" dimColor>@ {state.cwd}</Text>
        </Box>
      )}

      {showApproval && (
        <Box marginTop={1} flexDirection="column">
          <Box>
            <Text color={pendingSelection === 0 ? 'green' : 'gray'}>
              {pendingSelection === 0 ? '[✓]' : '[ ]'} Yes, run
            </Text>
            <Text>   </Text>
            <Text color={pendingSelection === 1 ? 'red' : 'gray'}>
              {pendingSelection === 1 ? '[x]' : '[ ]'} No, cancel
            </Text>
          </Box>
          <Box marginTop={1}>
            <Text color="gray" dimColor>Use arrow keys then press enter</Text>
          </Box>
        </Box>
      )}

          {!showApproval && (
        <>
          {status === 'running' && (
            <Box marginTop={1}>
              <Text color={state.isServer ? 'green' : isActive ? 'cyan' : 'gray'} dimColor>
                {state.isServer
                  ? 'Streaming output in Server Sessions panel'
                  : isActive
                    ? '▶ Interactive — type here'
                    : 'Running...'}
              </Text>
            </Box>
          )}

          {status === 'error' && state.timedOut && (
            <Box marginTop={1}>
              <Text color="red">{state.error}</Text>
            </Box>
          )}

          {status === 'error' && !state.timedOut && state.error && (
            <Box marginTop={1}>
              <Text color="red">{state.error}</Text>
            </Box>
          )}

          {!state.isServer && outputLines.length > 0 && (
            <Box marginTop={1} flexDirection="column" marginLeft={1}>
              {outputLines.map((line, idx) => (
                <Text key={idx} color="white">
                  {line || ' '}
                </Text>
              ))}
            </Box>
          )}

          {!state.isServer && outputLines.length === 0 && status !== 'cancelled' && (
            <Box marginTop={1}>
              <Text color="gray" dimColor>No output yet.</Text>
            </Box>
          )}

          {status === 'success' && (
            <Box marginTop={1}>
              <Text color="green">
                {state.isServer ? 'Server stopped' : 'Exit code'} {state.exitCode ?? 0}
              </Text>
            </Box>
          )}

          {status === 'error' && typeof state.exitCode === 'number' && (
            <Box marginTop={1}>
              <Text color="red" dimColor>Exit code {state.exitCode}</Text>
            </Box>
          )}

          {status === 'cancelled' && (
            <Box marginTop={1}>
              <Text color="gray" dimColor>
                {state.isServer ? 'Server command not approved.' : 'Command not approved.'}
              </Text>
            </Box>
          )}
        </>
      )}
    </Box>
  );
}

function ServerSessionView({ session }: { session: ServerSession }) {
  const { command, cwd, status, output, exitCode, error } = session;
  const statusLabels: Record<ServerSession['status'], { label: string; color: string }> = {
    pending: { label: '[?] Awaiting approval', color: 'yellow' },
    starting: { label: '[>] Starting', color: 'cyan' },
    running: { label: '[▶] Running', color: 'green' },
    stopped: { label: '[■] Stopped', color: 'gray' },
    error: { label: '[!] Error', color: 'red' },
  };

  const { label, color } = statusLabels[status];
  const cleanedOutput = output.replace(/\r/g, '');
  const lines = cleanedOutput ? cleanedOutput.split('\n').slice(-200) : [];

  return (
    <Box
      flexDirection="column"
      borderStyle="round"
      borderColor={color}
      paddingX={1}
      paddingY={1}
      marginBottom={1}
    >
      <Box>
        <Text color={color} bold>{label}</Text>
        <Text> {command}</Text>
      </Box>
      {cwd && (
        <Box marginTop={1}>
          <Text color="gray" dimColor>@ {cwd}</Text>
        </Box>
      )}
      {status === 'error' && error && (
        <Box marginTop={1}>
          <Text color="red">{error}</Text>
        </Box>
      )}
      {status === 'stopped' && (
        <Box marginTop={1}>
          <Text color="gray" dimColor>Exit code {exitCode ?? 0}</Text>
        </Box>
      )}
      <Box marginTop={1} flexDirection="column">
        {lines.length === 0 ? (
          <Text color="gray" dimColor>No output yet.</Text>
        ) : (
          lines.map((line, idx) => (
            <Text key={`${session.id}-${idx}`}>{line || ' '}</Text>
          ))
        )}
      </Box>
    </Box>
  );
}

function OperationItem({
  op,
  onToggle,
  isPendingCommand,
  pendingSelection,
  isActiveCommand,
}: {
  op: Operation;
  onToggle: () => void;
  isPendingCommand?: boolean;
  pendingSelection?: number;
  isActiveCommand?: boolean;
}) {
  if (op.type === 'command' || op.type === 'server') {
    return (
      <CommandOperationItem
        op={op}
        isPending={Boolean(isPendingCommand)}
        pendingSelection={pendingSelection}
        isActive={Boolean(isActiveCommand)}
      />
    );
  }

  const getTypeColor = () => {
    switch (op.type) {
      case 'list': return 'blue';
      case 'read': return 'cyan';
      case 'write': return 'yellow';
      case 'search': return 'magenta';
      case 'command': return 'green';
      case 'server': return 'green';
      default: return 'white';
    }
  };

  const getTypeLabel = () => {
    switch (op.type) {
      case 'list': return 'List';
      case 'read': return 'Read';
      case 'write': return 'Update';
      case 'search': return 'Search';
      case 'command': return 'Execute';
      case 'server': return 'Server';
      default: return op.type;
    }
  };

  return (
    <Box flexDirection="column" marginBottom={1}>
      <Box>
        <Text color="green">• </Text>
        <Text color={getTypeColor()} bold>{getTypeLabel()}</Text>
        <Text>({op.path})</Text>
      </Box>
      
      {/* Summary for read/list - no expansion */}
      {op.summary && !op.diff && (
        <Box marginLeft={2}>
          <Text color="gray">└ {op.summary}</Text>
        </Box>
      )}
      
      {/* Diff for write/update - auto-expand if < 60 lines, otherwise show button */}
      {op.diff && op.diff.length > 0 && (() => {
        const changeCount = op.diff.filter(d => d.old || d.new).length;
        const shouldAutoExpand = changeCount < 60;
        const isExpanded = op.expanded !== undefined ? op.expanded : shouldAutoExpand;
        
        return (
          <Box flexDirection="column" marginLeft={2}>
            <Box flexDirection="row" alignItems="center">
              <Text color="gray">└ Updated <Text bold>{op.path}</Text> with <Text bold>{changeCount}</Text> {changeCount === 1 ? 'change' : 'changes'}</Text>
              {!shouldAutoExpand && (
                <Box marginLeft={2}>
                  <Text 
                    color="cyan" 
                    bold
                    inverse
                  >
                    {isExpanded ? ' Hide changes ' : ' See changes '}
                  </Text>
                </Box>
              )}
            </Box>
            {isExpanded && op.diff.map((d, i) => (
            <Box key={i} marginLeft={2}>
              {d.old && !d.new && (
                <Box>
                  <Text color="gray">{d.line} </Text>
                  <Text backgroundColor="#cc0000" color="#ffffff">- {d.old}</Text>
                </Box>
              )}
              {d.new && !d.old && (
                <Box>
                  <Text color="gray">{d.line} </Text>
                  <Text backgroundColor="#00aa00" color="#ffffff">+ {d.new}</Text>
                </Box>
              )}
              {d.old && d.new && (
                <Box flexDirection="column">
                  <Box>
                    <Text color="gray">{d.line} </Text>
                    <Text backgroundColor="#cc0000" color="#ffffff">- {d.old}</Text>
                  </Box>
                  <Box>
                    <Text color="gray">{d.line} </Text>
                    <Text backgroundColor="#00aa00" color="#ffffff">+ {d.new}</Text>
                  </Box>
                </Box>
              )}
            </Box>
            ))}
          </Box>
        );
      })()}
    </Box>
  );
}

function TurnView({
  turn,
  turnIndex,
  onToggleOperation,
  pendingCommand,
  activeCommand,
}: {
  turn: Turn;
  turnIndex: number;
  onToggleOperation: (eventIndex: number) => void;
  pendingCommand?: PendingCommandState | null;
  activeCommand?: { turnIndex: number; eventIndex: number } | null;
}) {
  return (
    <Box flexDirection="column">
      <Box marginBottom={1}>
        <Text color="#0052cc" bold>You: </Text>
        <Text>{turn.userMessage}</Text>
      </Box>

      <Box flexDirection="column">
        <Text color="#ff8800" bold>Assistant:</Text>
        
        {turn.timeline.length === 0 && turn.isProcessing && (
          <Box marginLeft={2} marginTop={1}>
            <Text color="yellow">
              <Spinner type="dots" /> Thinking...
            </Text>
          </Box>
        )}

        {turn.timeline.map((event, i) => (
          <Box key={i} flexDirection="column">
            {event.type === 'operation' && event.operation && (
              <Box marginTop={1}>
                <OperationItem
                  op={event.operation}
                  onToggle={() => onToggleOperation(i)}
                  isPendingCommand={
                    Boolean(pendingCommand && pendingCommand.turnIndex === turnIndex && pendingCommand.eventIndex === i)
                  }
                  pendingSelection={
                    pendingCommand && pendingCommand.turnIndex === turnIndex && pendingCommand.eventIndex === i
                      ? pendingCommand.selected
                      : undefined
                  }
                  isActiveCommand={
                    Boolean(activeCommand && activeCommand.turnIndex === turnIndex && activeCommand.eventIndex === i)
                  }
                />
              </Box>
            )}
            {event.type === 'text' && event.text && (
              <Box marginLeft={2} marginTop={1}>
                <MarkdownText content={event.text} />
              </Box>
            )}
          </Box>
        ))}
        
        {turn.isProcessing && turn.progressText && (
          <Box marginTop={1}>
            <Text color="yellow">
              <Spinner type="dots" /> {turn.progressText}
            </Text>
            <Text color="gray"> · esc to interrupt</Text>
          </Box>
        )}
      </Box>
    </Box>
  );
}

function AgentUI({ verbose, checkpoint, sessionId, resume }: AgentUIProps) {
  const { exit } = useApp();
  
  // Model selection state
  const [modelSelected, setModelSelected] = useState(false);
  const [selectedModel, setSelectedModel] = useState('claude-sonnet-4-20250514');
  const [modelMenuIndex, setModelMenuIndex] = useState(0);
  
  const [turns, setTurns] = useState<Turn[]>([]);
  const turnsRef = useRef<Turn[]>(turns);
  const [input, setInput] = useState<string>('');
  const [history, setHistory] = useState<string[]>([]);
  const [historyIndex, setHistoryIndex] = useState<number>(-1);
  const [isProcessing, setIsProcessing] = useState(false);
  const startTimeRef = useRef<number>(0);
  const checkpointStoreRef = useRef<SqliteCheckpointSaver | null>(null);
  const shuttingDownRef = useRef(false);
  const [shutdownMessage, setShutdownMessage] = useState<string | null>(null);
  
  // For interactive commands that need user input
  const [promptMode, setPromptMode] = useState<{
    active: boolean;
    command: string;
    prompt: string;
    collectedInputs: string[];
    prompts: string[];
    callback: (inputs: string[]) => Promise<void>;
  } | null>(null);
  
  // For command approval (like Gemini)
  const [pendingCommand, setPendingCommand] = useState<PendingCommandState | null>(null);
  const [pendingContinue, setPendingContinue] = useState<PendingContinueState | null>(null);
  const [activeCommandSession, setActiveCommandSession] = useState<{ turnIndex: number; eventIndex: number } | null>(null);
  const commandSessionRef = useRef<{
    session: CommandPTYSession;
    turnIndex: number;
    eventIndex: number;
  } | null>(null);
  const queuedTextRef = useRef<string[]>([]); // Queue text that arrives after approval needed
  const pendingCommandRef = useRef<PendingCommandState | null>(null);
  const lastToggledOperationRef = useRef<{ turnIndex: number; eventIndex: number } | null>(null);
  const activeCommandSessionRef = useRef<{ turnIndex: number; eventIndex: number } | null>(null);
  const [serverSessions, setServerSessions] = useState<ServerSession[]>([]);
  const serverSessionsRef = useRef<ServerSession[]>(serverSessions);
  const serverSessionControllersRef = useRef(new Map<string, CommandPTYSession>());
  const [serverSessionsCollapsed, setServerSessionsCollapsed] = useState(false);

  useEffect(() => {
    turnsRef.current = turns;
  }, [turns]);

  useEffect(() => {
    pendingCommandRef.current = pendingCommand;
  }, [pendingCommand]);

  useEffect(() => {
    activeCommandSessionRef.current = activeCommandSession;
  }, [activeCommandSession]);

  useEffect(() => {
    serverSessionsRef.current = serverSessions;
  }, [serverSessions]);

  useEffect(() => {
    if (serverSessions.length === 0 && serverSessionsCollapsed) {
      setServerSessionsCollapsed(false);
    }
  }, [serverSessions.length, serverSessionsCollapsed]);

  useEffect(() => {
    return () => {
      commandSessionRef.current?.session.dispose();
      for (const session of serverSessionControllersRef.current.values()) {
        session.dispose();
      }
    };
  }, []);

  const setPendingCommandState = useCallback(
    (
      value:
        | PendingCommandState
        | null
        | ((prev: PendingCommandState | null) => PendingCommandState | null)
    ) => {
      setPendingCommand(prev => {
        const next = typeof value === 'function' ? (value as (prev: PendingCommandState | null) => PendingCommandState | null)(prev) : value;
        pendingCommandRef.current = next;
        return next;
      });
    },
    []
  );

  const setActiveCommandSessionState = useCallback(
    (value: { turnIndex: number; eventIndex: number } | null) => {
      activeCommandSessionRef.current = value;
      setActiveCommandSession(value);
    },
    []
  );

  // Handle terminal resize with debounce to prevent artifacts
  useEffect(() => {
    let resizeTimer: NodeJS.Timeout | null = null;
    
    const handleResize = () => {
      // Clear any pending resize
      if (resizeTimer) {
        clearTimeout(resizeTimer);
      }
      
      // Debounce: only re-render after user stops resizing (300ms)
      resizeTimer = setTimeout(() => {
        // Resize PTY sessions if active
        if (commandSessionRef.current) {
          const cols = process.stdout.columns || 80;
          const rows = process.stdout.rows || 24;
          commandSessionRef.current.session.resize(cols, rows);
        }
        
        // Force a re-render by triggering a state update
        setInput(prev => prev);
      }, 300);
    };

    process.stdout.on('resize', handleResize);
    return () => {
      if (resizeTimer) {
        clearTimeout(resizeTimer);
      }
      process.stdout.off('resize', handleResize);
    };
  }, []);

  const updateCommandOperation = useCallback(
    (turnIndex: number, eventIndex: number, updater: (op: Operation) => Operation) => {
      setTurns(prev => {
        const currentTurn = prev[turnIndex];
        if (!currentTurn) {
          return prev;
        }
        const currentEvent = currentTurn.timeline[eventIndex];

        if (!currentEvent || currentEvent.type !== 'operation' || !currentEvent.operation) {
          return prev;
        }

        const op = currentEvent.operation;
        const clonedOp: Operation = {
          ...op,
          commandState: op.commandState ? { ...op.commandState } : undefined,
          serverState: op.serverState ? { ...op.serverState } : undefined,
          diff: op.diff ? [...op.diff] : op.diff,
          lines: op.lines ? [...op.lines] : op.lines,
        };

        const updatedOp = updater(clonedOp);

        if (updatedOp === op) {
          return prev;
        }

        const newTimeline = [...currentTurn.timeline];
        newTimeline[eventIndex] = {
          ...currentEvent,
          operation: updatedOp,
        };

        const newTurns = [...prev];
        newTurns[turnIndex] = {
          ...currentTurn,
          timeline: newTimeline,
        };

        turnsRef.current = newTurns;
        return newTurns;
      });
    },
    []
  );

  const flushQueuedText = useCallback((turnIndex: number) => {
    const queued = queuedTextRef.current;
    if (queued.length === 0) {
      return;
    }

    queuedTextRef.current = [];
    setTurns(prev => {
      const updated = [...prev];
      if (updated[turnIndex]) {
        const cleanedTexts = queued
          .join('')
          .replace(/\n{3,}/g, '\n\n')
          .trim()
          .split('\n\n')
          .filter(t => t && !t.includes('━') && !t.includes('Terminal Coding Agent') &&
                       !t.includes('💾') && !t.includes('📂'));

        if (cleanedTexts.length > 0) {
          updated[turnIndex].timeline = [
            ...updated[turnIndex].timeline,
            ...cleanedTexts.map(text => ({
              type: 'text' as const,
              timestamp: Date.now(),
              text,
            })),
          ];
        }
      }
      turnsRef.current = updated;
      return updated;
    });
  }, []);

  const updateServerSession = useCallback((id: string, updater: (session: ServerSession) => ServerSession) => {
    setServerSessions(prev => {
      const index = prev.findIndex(s => s.id === id);
      if (index === -1) {
        return prev;
      }

      const updatedSession = updater(prev[index]);
      if (updatedSession === prev[index]) {
        return prev;
      }

      const next = [...prev];
      next[index] = updatedSession;
      serverSessionsRef.current = next;
      return next;
    });
  }, []);

  const appendServerOutput = useCallback((id: string, chunk: string) => {
    if (!chunk) return;
    updateServerSession(id, session => {
      const combined = (session.output + chunk).slice(-MAX_COMMAND_OUTPUT_LENGTH);
      if (combined === session.output) {
        return session;
      }
      return {
        ...session,
        output: combined,
      };
    });
    checkpointStoreRef.current?.recordSessionActivity(sessionId);
  }, [updateServerSession, sessionId]);

  const createOrUpdateServerSession = useCallback((session: ServerSession) => {
    setServerSessions(prev => {
      const index = prev.findIndex(s => s.id === session.id);
      if (index === -1) {
        const next = [...prev, session];
        serverSessionsRef.current = next;
        checkpointStoreRef.current?.recordSessionActivity(sessionId);
        return next;
      }
      const next = [...prev];
      next[index] = {
        ...next[index],
        ...session,
      };
      serverSessionsRef.current = next;
      checkpointStoreRef.current?.recordSessionActivity(sessionId);
      return next;
    });
  }, [sessionId]);

  const stopServerSession = useCallback((id?: string) => {
    const sessions = serverSessionsRef.current;
    let target = id;

    if (!target) {
      for (let i = sessions.length - 1; i >= 0; i--) {
        if (sessions[i].status === 'running' || sessions[i].status === 'starting') {
          target = sessions[i].id;
          break;
        }
      }
    }

    if (!target) {
      return;
    }

    const controller = serverSessionControllersRef.current.get(target);
    if (controller) {
      try {
        controller.dispose();
      } catch {
        // ignore disposal errors
      }
      serverSessionControllersRef.current.delete(target);
    }

    updateServerSession(target, (sessionState) => ({
      ...sessionState,
      status: 'stopped',
      endedAt: Date.now(),
    }));

    appendServerOutput(target, '\n[Session terminated by user]\n');

    const turnsSnapshot = turnsRef.current;
    let foundTurn = -1;
    let foundEvent = -1;
    for (let t = 0; t < turnsSnapshot.length; t++) {
      const timeline = turnsSnapshot[t].timeline;
      for (let e = 0; e < timeline.length; e++) {
        const event = timeline[e];
        if (event.type === 'operation' && event.operation?.toolUseId === target) {
          foundTurn = t;
          foundEvent = e;
          break;
        }
      }
      if (foundTurn >= 0) break;
    }

    if (foundTurn >= 0 && foundEvent >= 0) {
      const turnIndex = foundTurn;
      const eventIndex = foundEvent;
      updateCommandOperation(turnIndex, eventIndex, (op) => {
        if (op.type !== 'server') {
          return op;
        }
        return {
          ...op,
          serverState: {
            ...(op.serverState ?? {
              id: target,
              command: op.commandState?.command || op.path,
              cwd: op.commandState?.cwd,
              status: 'stopped',
              output: '',
            }),
            status: 'stopped',
            exitCode: op.serverState?.exitCode ?? 0,
          },
          commandState: {
            ...(op.commandState ?? {
              command: op.path,
              cwd: op.serverState?.cwd,
              timeout: undefined,
              status: 'success',
              output: '',
              isServer: true,
            }),
            status: 'success',
            exitCode: op.commandState?.exitCode ?? 0,
            isServer: true,
          },
        };
      });

      setTurns(prev => {
        const updated = [...prev];
        if (updated[turnIndex]) {
          updated[turnIndex].progressText = undefined;
        }
        turnsRef.current = updated;
        return updated;
      });
    }
  }, [appendServerOutput, updateCommandOperation, updateServerSession]);

  const removeServerSession = useCallback((id?: string) => {
    if (!id) return;
    serverSessionControllersRef.current.delete(id);
    setServerSessions(prev => {
      const next = prev.filter(session => session.id !== id);
      serverSessionsRef.current = next;
      return next;
    });
  }, []);

  const gracefulShutdown = useCallback(() => {
    if (shuttingDownRef.current) {
      return;
    }
    shuttingDownRef.current = true;

    const pending = pendingCommandRef.current;
    if (pending) {
      setPendingCommandState(null);
      if (pending.toolUseId) {
        if (pending.isServer) {
          rejectPendingServerCommand(pending.toolUseId, 'Session terminated by user');
          removeServerSession(pending.toolUseId);
        } else {
          resolvePendingCommand(pending.toolUseId, {
            stdout: '',
            stderr: 'Command cancelled (session terminated)',
            exitCode: -1,
          });
        }
      }
    }

    if (commandSessionRef.current) {
      try {
        commandSessionRef.current.session.dispose();
      } catch {
        // ignore
      }
      commandSessionRef.current = null;
    }

    const activeServerIds = Array.from(serverSessionControllersRef.current.keys());
    for (const id of activeServerIds) {
      stopServerSession(id);
    }

    checkpointStoreRef.current?.recordSessionActivity(sessionId);

    setIsProcessing(false);
    const resumeCommand = `npm start -- --resume ${sessionId}`;
    setShutdownMessage(`Session saved. Resume with: ${resumeCommand}`);

    setTimeout(() => {
      process.stdout.write(`\nSession saved. Resume with: ${resumeCommand}\n`);
      exit();
    }, 250);
  }, [exit, removeServerSession, sessionId, setPendingCommandState, stopServerSession]);

  const startCommandSession = useCallback(
    (pending: {
      command: string;
      cwd?: string;
      timeout?: number;
      turnIndex: number;
      eventIndex: number;
      toolUseId?: string;
    }) => {
      if (pending.eventIndex < 0) {
        return;
      }

      const { command, cwd, timeout, turnIndex, eventIndex, toolUseId } = pending;

      const normalizedTimeout =
        typeof timeout === 'number'
          ? timeout > 1000
            ? Math.round(timeout / 1000)
            : timeout
          : 900;

      let aggregatedOutput = '';

      updateCommandOperation(turnIndex, eventIndex, (op) => {
        const existing = op.commandState ?? {
          command,
          cwd,
          timeout: normalizedTimeout,
          output: '',
          status: 'pending',
        };

        return {
          ...op,
          path: command || op.path,
          commandState: {
            ...existing,
            command: command || existing.command,
            cwd: cwd ?? existing.cwd,
            timeout: normalizedTimeout,
            status: 'running',
          },
        };
      });

      setTurns(prev => {
        const updated = [...prev];
        if (updated[turnIndex]) {
          updated[turnIndex].progressText = 'Running command...';
        }
        turnsRef.current = updated;
        return updated;
      });

      try {
        const session = executeCommandWithPTY(command, {
          cwd: cwd || process.cwd(),
          timeout: normalizedTimeout,
            onData: (chunk: string) => {
              // Check if this chunk has clear sequences (indicates interactive UI)
              const hasClearSequences = CLEAR_SEQUENCE_REGEX.test(chunk);
              CLEAR_SEQUENCE_REGEX.lastIndex = 0;
              
              const cleanedChunk = sanitizeCommandChunk(chunk);
              if (!cleanedChunk && !hasClearSequences) {
                return;
              }
              
              // If we detect a screen clear, start a new frame instead of appending
              // This prevents interactive UIs from showing duplicated/garbled text
              if (hasClearSequences && cleanedChunk) {
                // Replace output with new frame
                aggregatedOutput = cleanedChunk;
              } else if (cleanedChunk) {
                // Normal append for non-interactive output
                aggregatedOutput += cleanedChunk;
                if (aggregatedOutput.length > MAX_COMMAND_OUTPUT_LENGTH) {
                  aggregatedOutput = aggregatedOutput.slice(aggregatedOutput.length - MAX_COMMAND_OUTPUT_LENGTH);
                }
              }

            const finalOutput = aggregatedOutput;

            updateCommandOperation(turnIndex, eventIndex, (op) => {
              const state = op.commandState ?? {
                command,
                cwd,
                timeout: normalizedTimeout,
                status: 'running',
                output: '',
              };

              return {
                ...op,
                commandState: {
                  ...state,
                  status: 'running',
                  output: finalOutput,
                },
              };
            });
          },
          onExit: ({ exitCode, timedOut }: { exitCode: number; timedOut?: boolean }) => {
            // Clear session refs immediately
            commandSessionRef.current = null;
            setActiveCommandSessionState(null);
            
            // Clear progress text immediately
            setTurns(prev => {
              const updated = [...prev];
              if (updated[turnIndex]) {
                updated[turnIndex].progressText = undefined;
              }
              turnsRef.current = updated;
              return updated;
            });
            
            updateCommandOperation(turnIndex, eventIndex, (op) => {
              const state = op.commandState ?? {
                command,
                cwd,
                timeout: normalizedTimeout,
                output: '',
                status: 'running',
              };

              const status = exitCode === 0 && !timedOut ? 'success' : 'error';
              const errorMessage = timedOut
                ? `Command timed out after ${normalizedTimeout} seconds`
                : state.error;

              return {
                ...op,
                commandState: {
                  ...state,
                  status,
                  exitCode,
                  timedOut,
                  error: errorMessage,
                },
              };
            });

            if (toolUseId) {
              const cleanedStdout = sanitizeCommandChunk(aggregatedOutput).trimEnd();
              const result = {
                stdout: cleanedStdout,
                stderr: timedOut
                  ? `Command timed out after ${normalizedTimeout} seconds`
                  : '',
                exitCode: typeof exitCode === 'number' ? exitCode : timedOut ? 124 : 0,
              };
              resolvePendingCommand(toolUseId, result);
            }

            // Add queued text after command completes
            flushQueuedText(turnIndex);
          },
          onError: (error: Error) => {
            commandSessionRef.current?.session.dispose();
            commandSessionRef.current = null;
            setActiveCommandSessionState(null);
            
            // Clear progress text immediately
            setTurns(prev => {
              const updated = [...prev];
              if (updated[turnIndex]) {
                updated[turnIndex].progressText = undefined;
              }
              turnsRef.current = updated;
              return updated;
            });
            
            updateCommandOperation(turnIndex, eventIndex, (op) => {
              const state = op.commandState ?? {
                command,
                cwd,
                timeout: normalizedTimeout,
                output: '',
                status: 'running',
              };

              return {
                ...op,
                commandState: {
                  ...state,
                  status: 'error',
                  error: error.message,
                },
              };
            });

            if (toolUseId) {
              rejectPendingCommand(toolUseId, error);
            }

            flushQueuedText(turnIndex);

            setTurns(prev => {
              const updated = [...prev];
              if (updated[turnIndex]) {
                updated[turnIndex].progressText = undefined;
              }
              turnsRef.current = updated;
              return updated;
            });
          },
        });

        commandSessionRef.current = {
          session,
          turnIndex,
          eventIndex,
        };
        setActiveCommandSessionState({ turnIndex, eventIndex });
      } catch (error) {
        commandSessionRef.current = null;
        setActiveCommandSessionState(null);
        updateCommandOperation(turnIndex, eventIndex, (op) => {
          const state = op.commandState ?? {
            command,
            cwd,
            timeout: normalizedTimeout,
            output: '',
            status: 'pending',
          };

          return {
            ...op,
            commandState: {
              ...state,
              status: 'error',
              error: error instanceof Error ? error.message : String(error),
            },
          };
        });

        if (toolUseId) {
          rejectPendingCommand(toolUseId, error instanceof Error ? error : new Error(String(error)));
        }

        flushQueuedText(turnIndex);

        setTurns(prev => {
          const updated = [...prev];
          if (updated[turnIndex]) {
            updated[turnIndex].progressText = undefined;
          }
          turnsRef.current = updated;
          return updated;
        });
      }
    },
    [flushQueuedText, updateCommandOperation]
  );

  const startServerSession = useCallback(
    (pending: {
      command: string;
      cwd?: string;
      timeout?: number;
      turnIndex: number;
      eventIndex: number;
      toolUseId: string;
    }) => {
      const { command, cwd, timeout, turnIndex, eventIndex, toolUseId } = pending;

      const normalizedTimeout =
        typeof timeout === 'number'
          ? timeout > 1000
            ? Math.round(timeout / 1000)
            : timeout
          : undefined;

      createOrUpdateServerSession({
        id: toolUseId,
        command,
        cwd,
        status: 'starting',
        output: '',
        startedAt: Date.now(),
      });

      updateCommandOperation(turnIndex, eventIndex, (op) => {
        return {
          ...op,
          type: 'server',
          path: command || op.path,
          serverState: {
            id: toolUseId,
            command,
            cwd,
            status: 'starting',
            output: '',
          },
          commandState: {
            ...(op.commandState ?? {
              command,
              cwd,
              timeout: normalizedTimeout,
              status: 'pending',
              output: '',
              isServer: true,
            }),
            command,
            cwd,
            timeout: normalizedTimeout,
            status: 'running',
            output: '',
            isServer: true,
          },
        };
      });

      setTurns(prev => {
        const updated = [...prev];
        if (updated[turnIndex]) {
          updated[turnIndex].progressText = 'Server running in dedicated terminal';
        }
        turnsRef.current = updated;
        return updated;
      });

      try {
        const session = executeCommandWithPTY(command, {
          cwd: cwd || process.cwd(),
          timeout: normalizedTimeout,
            onData: (chunk: string) => {
              const hasClearSequences = CLEAR_SEQUENCE_REGEX.test(chunk);
              CLEAR_SEQUENCE_REGEX.lastIndex = 0;
              
              const cleaned = sanitizeCommandChunk(chunk);
              if (!cleaned && !hasClearSequences) {
                return;
              }

              updateServerSession(toolUseId, sessionState => {
                let nextOutput: string;
                
                // If clear sequences detected, replace output (new frame)
                if (hasClearSequences && cleaned) {
                  nextOutput = cleaned;
                } else if (cleaned) {
                  nextOutput = (sessionState.output + cleaned).slice(-MAX_COMMAND_OUTPUT_LENGTH);
                } else {
                  return sessionState;
                }
                
                if (nextOutput === sessionState.output) {
                  return sessionState;
                }
                return {
                  ...sessionState,
                  output: nextOutput,
                };
              });
              checkpointStoreRef.current?.recordSessionActivity(sessionId);
            },
          onExit: ({ exitCode, timedOut }: { exitCode: number; timedOut?: boolean }) => {
            serverSessionControllersRef.current.delete(toolUseId);
            updateServerSession(toolUseId, (sessionState) => ({
              ...sessionState,
              status: timedOut ? 'error' : 'stopped',
              exitCode,
              endedAt: Date.now(),
              error: timedOut
                ? `Server command timed out after ${normalizedTimeout ?? 0} seconds`
                : sessionState.error,
            }));

            updateCommandOperation(turnIndex, eventIndex, (op) => {
              if (op.type !== 'server') {
                return op;
              }
              return {
                ...op,
                serverState: {
                  ...(op.serverState ?? {
                    id: toolUseId,
                    command,
                    cwd,
                    status: 'stopped',
                    output: '',
                  }),
                  status: timedOut ? 'error' : 'stopped',
                  exitCode,
                },
              };
            });

            setTurns(prev => {
              const updated = [...prev];
              if (updated[turnIndex]) {
                updated[turnIndex].progressText = undefined;
              }
              turnsRef.current = updated;
              return updated;
            });
          },
          onError: (error: Error) => {
            serverSessionControllersRef.current.delete(toolUseId);
            updateServerSession(toolUseId, (sessionState) => ({
              ...sessionState,
              status: 'error',
              error: error.message,
              endedAt: Date.now(),
            }));

            updateCommandOperation(turnIndex, eventIndex, (op) => {
              if (op.type !== 'server') {
                return op;
              }
              return {
                ...op,
                serverState: {
                  ...(op.serverState ?? {
                    id: toolUseId,
                    command,
                    cwd,
                    status: 'error',
                    output: '',
                  }),
                  status: 'error',
                  error: error.message,
                },
              };
            });

            setTurns(prev => {
              const updated = [...prev];
              if (updated[turnIndex]) {
                updated[turnIndex].progressText = undefined;
              }
              turnsRef.current = updated;
              return updated;
            });
          },
        });

        serverSessionControllersRef.current.set(toolUseId, session);

        updateServerSession(toolUseId, (sessionState) => ({
          ...sessionState,
          status: 'running',
        }));

        updateCommandOperation(turnIndex, eventIndex, (op) => {
          if (op.type !== 'server') {
            return op;
          }
          return {
            ...op,
            serverState: {
              ...(op.serverState ?? {
                id: toolUseId,
                command,
                cwd,
                status: 'running',
                output: '',
              }),
              status: 'running',
            },
          };
        });

        resolvePendingServerCommand(toolUseId, {
          stdout: JSON.stringify({ status: 'started', command, cwd }),
          stderr: '',
          exitCode: 0,
        });

        flushQueuedText(turnIndex);
      } catch (error) {
        serverSessionControllersRef.current.delete(toolUseId);
        updateServerSession(toolUseId, (sessionState) => ({
          ...sessionState,
          status: 'error',
          error: error instanceof Error ? error.message : String(error),
          endedAt: Date.now(),
        }));

        updateCommandOperation(turnIndex, eventIndex, (op) => {
          if (op.type !== 'server') {
            return op;
          }
          return {
            ...op,
            serverState: {
              ...(op.serverState ?? {
                id: toolUseId,
                command,
                cwd,
                status: 'error',
                output: '',
              }),
              status: 'error',
              error: error instanceof Error ? error.message : String(error),
            },
          };
        });

        rejectPendingServerCommand(
          toolUseId,
          error instanceof Error ? error : new Error(String(error))
        );

        flushQueuedText(turnIndex);
      }
    },
    [appendServerOutput, createOrUpdateServerSession, flushQueuedText, updateCommandOperation, updateServerSession]
  );

  // Initialize checkpoint store once on mount
  useEffect(() => {
    if (!checkpoint) {
      return;
    }

    const store = new SqliteCheckpointSaver();
    checkpointStoreRef.current = store;
    setCheckpointStore(store);

    const existingSession = store.getSessionMetadata(sessionId);
    if (!existingSession) {
      store.recordSessionStart(sessionId);
      store.recordSessionActivity(sessionId);
    } else {
      store.recordSessionActivity(sessionId);
    }

    return () => {
      try {
        store.recordSessionActivity(sessionId);
      } catch {
        // ignore errors during shutdown
      }
      try {
        store.close();
      } catch {
        // ignore close errors
      }
      if (checkpointStoreRef.current === store) {
        checkpointStoreRef.current = null;
      }
      setCheckpointStore(undefined);
    };
  }, [checkpoint, resume, sessionId]);

  useEffect(() => {
    const handler = () => gracefulShutdown();
    process.on('SIGINT', handler);
    return () => {
      process.off('SIGINT', handler);
    };
  }, [gracefulShutdown]);

  const runQuery = useCallback(async (query: string) => {
    if (!query.trim() || isProcessing) return;

    // Handle special commands
    if (query.startsWith('/')) {
      const parts = query.slice(1).split(' ');
      const command = parts[0].toLowerCase();
      
      if (command === 'undo') {
        const filePath = parts[1]; // optional file path
        const result = await undoLastChange(filePath);
        
        // Add result as a turn
        setTurns(prev => [...prev, {
          userMessage: query,
          timeline: [{
            type: 'text',
            timestamp: Date.now(),
            text: result.success ? `✅ ${result.message}` : `❌ ${result.message}`,
          }],
          isProcessing: false,
        }]);
        
        setHistory(prev => [...prev, query]);
        setHistoryIndex(-1);
        
        return;
      } else if (command === 'rename') {
        // Example: interactive command that needs user input
        if (parts[1] && parts[2]) {
          // All args provided: /rename old.py new.py
          const oldPath = parts[1];
          const newPath = parts[2];
          
          setTurns(prev => [...prev, {
            userMessage: query,
            timeline: [{
              type: 'text',
              timestamp: Date.now(),
              text: `✅ Would rename ${oldPath} to ${newPath} (not implemented)`,
            }],
            isProcessing: false,
          }]);
          
          setHistory(prev => [...prev, query]);
          setHistoryIndex(-1);
        } else {
          // Prompt for inputs
          setPromptMode({
            active: true,
            command: 'rename',
            prompt: 'Enter file to rename:',
            prompts: ['Enter file to rename:', 'Enter new name:'],
            collectedInputs: [],
            callback: async (inputs) => {
              const [oldPath, newPath] = inputs;
              setTurns(prev => [...prev, {
                userMessage: `/rename ${oldPath} ${newPath}`,
                timeline: [{
                  type: 'text',
                  timestamp: Date.now(),
                  text: `✅ Would rename ${oldPath} to ${newPath} (not implemented)`,
                }],
                isProcessing: false,
              }]);
              
              setHistory(prev => [...prev, `/rename ${oldPath} ${newPath}`]);
              setHistoryIndex(-1);
            },
          });
        }
        
        return;
      } else if (command === 'sessions-list') {
        // List all sessions from the checkpoint store
        const sessions = checkpointStoreRef.current?.listSessions() || [];
        
        let text: string;
        if (sessions.length === 0) {
          text = '📋 No saved sessions found.';
        } else {
          const sessionLines = sessions.map((s, i) => {
            const createdDate = new Date(s.created_at).toLocaleString();
            const lastActivityDate = new Date(s.last_activity).toLocaleString();
            const name = s.name || 'Untitled';
            return `${i + 1}. ${s.session_id}\n   Name: ${name}\n   Created: ${createdDate}\n   Last Activity: ${lastActivityDate}`;
          }).join('\n\n');
          
          text = `📋 Saved Sessions (${sessions.length}):\n\n${sessionLines}`;
        }
        
        setTurns(prev => [...prev, {
          userMessage: query,
          timeline: [{
            type: 'text',
            timestamp: Date.now(),
            text,
          }],
          isProcessing: false,
        }]);
        
        setHistory(prev => [...prev, query]);
        setHistoryIndex(-1);
        
        return;
      } else if (command === 'sessions-clear') {
        // Prompt for confirmation before clearing all sessions
        setPromptMode({
          active: true,
          command: 'sessions-clear',
          prompt: 'Are you sure you want to clear ALL sessions? (yes/no)',
          prompts: ['Are you sure you want to clear ALL sessions? (yes/no)'],
          collectedInputs: [],
          callback: async (inputs) => {
            const [confirmation] = inputs;
            
            if (confirmation.toLowerCase() === 'yes' || confirmation.toLowerCase() === 'y') {
              const sessions = checkpointStoreRef.current?.listSessions() || [];
              
              for (const session of sessions) {
                checkpointStoreRef.current?.deleteSession(session.session_id);
              }
              
              setTurns(prev => [...prev, {
                userMessage: `/sessions-clear`,
                timeline: [{
                  type: 'text',
                  timestamp: Date.now(),
                  text: `✅ Cleared ${sessions.length} session(s) from the database.`,
                }],
                isProcessing: false,
              }]);
            } else {
              setTurns(prev => [...prev, {
                userMessage: `/sessions-clear`,
                timeline: [{
                  type: 'text',
                  timestamp: Date.now(),
                  text: '❌ Cancelled. No sessions were deleted.',
                }],
                isProcessing: false,
              }]);
            }
            
            setHistory(prev => [...prev, `/sessions-clear`]);
            setHistoryIndex(-1);
          },
        });
        
        return;
      } else if (command === 'help') {
        // Display help information
        const helpText = `Help
        
🎯 SLASH COMMANDS:
  /help                  Show this help message
  /undo [file]           Undo last change (optionally specify file)
  /rename [old] [new]    Rename a file (interactive if no args)
  /sessions-list         List all saved sessions
  /sessions-clear        Clear all sessions from database

⌨️  KEYBOARD SHORTCUTS:
  Ctrl+X                 Stop the most recent server session
  Ctrl+E                 Toggle server sessions panel (expand/collapse)
  Ctrl+R                 Toggle diff expansion for most recent update
  Esc / Ctrl+C           Exit the application
  ↑ / ↓                  Navigate command history

🛠️  AGENT TOOLS:
The agent can use the following tools:
  • read_file            Read file contents
  • write_file           Create or update files
  • directory_read       List directory contents
  • directory_search     Search for patterns in files
  • execute_command      Run shell commands (requires approval)

💡 TIPS:
  • The agent has memory across the session
  • Command approvals: use arrow keys to select Yes/No
  • Resume sessions: use --resume <sessionId> flag
  • Diffs auto-expand if < 60 lines, otherwise use Ctrl+R
  • Be specific and detailed - clarity improves results!

📋 SESSION INFO:
  Current Session: ${sessionId}
  Resume command: npm start -- --resume ${sessionId}`;

        setTurns(prev => [...prev, {
          userMessage: query,
          timeline: [{
            type: 'text',
            timestamp: Date.now(),
            text: helpText,
          }],
          isProcessing: false,
        }]);
        
        setHistory(prev => [...prev, query]);
        setHistoryIndex(-1);
        
        return;
      } else {
        setTurns(prev => [...prev, {
          userMessage: query,
          timeline: [{
            type: 'text',
            timestamp: Date.now(),
            text: `❌ Unknown command: /${command}. Type /help for available commands.`,
          }],
          isProcessing: false,
        }]);
        
        setHistory(prev => [...prev, query]);
        setHistoryIndex(-1);
        
        return;
      }
    }

    setIsProcessing(true);
    startTimeRef.current = Date.now();
    
    const turnIndex = turns.length;
    
    setTurns(prev => [...prev, {
      userMessage: query,
      timeline: [],
      isProcessing: true,
    }]);
    
    setHistory(prev => [...prev, query]);
    setHistoryIndex(-1);

    const customOutput = {
      write: (text: string) => {
        if (text.includes('SERVER_COMMAND_REQUEST:')) {
          const parts = text.split('SERVER_COMMAND_REQUEST:')[1].trim();

          let command = parts;
          let cwd: string | undefined;
          let timeout: number | undefined;

          const cwdMatch = parts.match(/:CWD:(.+?)(?::TIMEOUT:|$)/);
          if (cwdMatch) {
            cwd = cwdMatch[1];
            command = parts.split(':CWD:')[0];
          }

          const timeoutMatch = parts.match(/:TIMEOUT:(\d+)/);
          if (timeoutMatch) {
            timeout = parseInt(timeoutMatch[1], 10);
            if (!cwd) {
              command = parts.split(':TIMEOUT:')[0];
            }
          }

          const trimmedCommand = command.trim();
          const currentTurn = turnsRef.current[turnIndex];
          let serverEventIndex = -1;

          if (currentTurn) {
            let fallbackIndex = -1;
            for (let i = currentTurn.timeline.length - 1; i >= 0; i--) {
              const event = currentTurn.timeline[i];
              if (event.type === 'operation' && event.operation?.type === 'server') {
                const opCommand = event.operation.serverState?.command || event.operation.path;
                if (opCommand && opCommand.trim() === trimmedCommand) {
                  serverEventIndex = i;
                  break;
                }
                if (fallbackIndex < 0 && (!opCommand || opCommand === 'execute_server_command')) {
                  fallbackIndex = i;
                }
              }
            }
            if (serverEventIndex < 0 && fallbackIndex >= 0) {
              serverEventIndex = fallbackIndex;
            }
          }

          const currentSession = serverEventIndex >= 0 ? currentTurn?.timeline?.[serverEventIndex] : undefined;
          const toolUseId =
            currentSession && currentSession.type === 'operation'
              ? currentSession.operation?.toolUseId
              : undefined;

          const sessionId = toolUseId || `server-${Date.now()}`;

          if (serverEventIndex < 0 && currentTurn) {
            const newOp: Operation = {
              type: 'server',
              name: 'execute_server_command',
              path: trimmedCommand,
              serverState: {
                id: sessionId,
                command: trimmedCommand,
                cwd,
                status: 'pending',
                output: '',
              },
              commandState: {
                command: trimmedCommand,
                cwd,
                timeout,
                status: 'pending',
                output: '',
                isServer: true,
              },
              toolUseId: sessionId,
            };
            serverEventIndex = currentTurn.timeline.length;
            setTurns(prev => {
              const updated = [...prev];
              if (updated[turnIndex]) {
                updated[turnIndex].timeline = [
                  ...updated[turnIndex].timeline,
                  {
                    type: 'operation',
                    timestamp: Date.now(),
                    operation: newOp,
                  },
                ];
              }
              turnsRef.current = updated;
              return updated;
            });
          }

          createOrUpdateServerSession({
            id: sessionId,
            command: trimmedCommand,
            cwd,
            status: 'pending',
            output: '',
          });

          if (serverEventIndex >= 0) {
            updateCommandOperation(turnIndex, serverEventIndex, (op) => {
              return {
                ...op,
                type: 'server',
                toolUseId: toolUseId ?? sessionId,
                path: trimmedCommand || op.path,
                serverState: {
                  ...(op.serverState ?? {
                    id: sessionId,
                    command: trimmedCommand,
                    cwd,
                    status: 'pending',
                    output: '',
                  }),
                  command: trimmedCommand,
                  cwd: cwd ?? op.serverState?.cwd,
                  status: 'pending',
                },
                commandState: {
                  ...(op.commandState ?? {
                    command: trimmedCommand,
                    cwd,
                    timeout,
                    status: 'pending',
                    output: '',
                    isServer: true,
                  }),
                  command: trimmedCommand,
                  cwd: cwd ?? op.commandState?.cwd,
                  timeout,
                  status: 'pending',
                  output: op.commandState?.output ?? '',
                  isServer: true,
                  exitCode: undefined,
                },
              };
            });

            setPendingCommandState({
              command: trimmedCommand,
              cwd,
              timeout,
              selected: 0,
              turnIndex,
              eventIndex: serverEventIndex,
              toolUseId: toolUseId ?? sessionId,
              isServer: true,
            });
          }

          setTurns(prev => {
            const updated = [...prev];
            if (updated[turnIndex]) {
              updated[turnIndex].progressText = 'Waiting for server approval...';
            }
            turnsRef.current = updated;
            return updated;
          });

          setIsProcessing(false);
          queuedTextRef.current = [];
          return;
        }

        // Check for command approval request
        if (text.includes('COMMAND_APPROVAL_NEEDED:')) {
          const parts = text.split('COMMAND_APPROVAL_NEEDED:')[1].trim();
          
          // Parse: COMMAND:CWD:path:TIMEOUT:seconds
          let command = parts;
          let cwd: string | undefined;
          let timeout: number | undefined;
          
          // Extract CWD
          const cwdMatch = parts.match(/:CWD:(.+?)(?::TIMEOUT:|$)/);
          if (cwdMatch) {
            cwd = cwdMatch[1];
            command = parts.split(':CWD:')[0];
          }
          
          // Extract TIMEOUT
          const timeoutMatch = parts.match(/:TIMEOUT:(\d+)/);
          if (timeoutMatch) {
            timeout = parseInt(timeoutMatch[1], 10);
            // Remove timeout from command if CWD wasn't present
            if (!cwd) {
              command = parts.split(':TIMEOUT:')[0];
            }
          }

          const trimmedCommand = command.trim();

          // Locate the most recent command operation for this turn that matches this command
          // We need to check both turnsRef (for immediate updates) and do a fresh search
          const currentTurn = turnsRef.current[turnIndex];
          let commandEventIndex = -1;
          
          // Search for matching command operation
          // Priority: 1) Exact command match, 2) Most recent command without command set (just created)
          if (currentTurn) {
            let fallbackIndex = -1;
            for (let i = currentTurn.timeline.length - 1; i >= 0; i--) {
              const event = currentTurn.timeline[i];
              if (event.type === 'operation' && event.operation?.type === 'command') {
                const opCommand = event.operation.commandState?.command || event.operation.path;
                // Exact match
                if (opCommand && opCommand.trim() === trimmedCommand) {
                  commandEventIndex = i;
                  break;
                }
                // Fallback: most recent command operation without command set (just created by tool())
                if (fallbackIndex < 0 && (!opCommand || opCommand === 'execute_command' || opCommand === 'execute command')) {
                  fallbackIndex = i;
                }
              }
            }
            // Use fallback if no exact match
            if (commandEventIndex < 0 && fallbackIndex >= 0) {
              commandEventIndex = fallbackIndex;
            }
          }

          // If no matching command operation found, create one
          if (commandEventIndex < 0 && currentTurn) {
            const newOp: Operation = {
              type: 'command',
              name: 'execute_command',
              path: trimmedCommand,
              commandState: {
                command: trimmedCommand,
                cwd,
                timeout,
                status: 'pending',
                output: '',
              },
            };
            // Calculate eventIndex before state update
            commandEventIndex = currentTurn.timeline.length;
            setTurns(prev => {
              const updated = [...prev];
              if (updated[turnIndex]) {
                updated[turnIndex].timeline = [
                  ...updated[turnIndex].timeline,
                  {
                    type: 'operation',
                    timestamp: Date.now(),
                    operation: newOp,
                  }
                ];
              }
              // Update ref synchronously
              turnsRef.current = updated;
              return updated;
            });
          }

          if (commandEventIndex >= 0) {
            // Check if this command is already pending
            const alreadyPending = pendingCommandRef.current && 
              pendingCommandRef.current.turnIndex === turnIndex &&
              pendingCommandRef.current.eventIndex === commandEventIndex;
            const commandEvent = currentTurn?.timeline?.[commandEventIndex];
            const toolUseId =
              commandEvent && commandEvent.type === 'operation'
                ? commandEvent.operation?.toolUseId
                : undefined;
            
            if (!alreadyPending) {
              updateCommandOperation(turnIndex, commandEventIndex, (op) => {
                return {
                  ...op,
                  path: trimmedCommand || op.path,
                  commandState: {
                    command: trimmedCommand || op.commandState?.command || op.path,
                    cwd: cwd ?? op.commandState?.cwd,
                    timeout: timeout ?? op.commandState?.timeout,
                    status: 'pending',
                    output: op.commandState?.output ?? '',
                  },
                };
              });

              setPendingCommandState({
                command: trimmedCommand,
                cwd,
                timeout,
                selected: 0,
                turnIndex,
                eventIndex: commandEventIndex,
                toolUseId,
              });
            }
          }
          setTurns(prev => {
            const updated = [...prev];
            if (updated[turnIndex]) {
              updated[turnIndex].progressText = 'Waiting for command approval...';
            }
            turnsRef.current = updated;
            return updated;
          });
          setIsProcessing(false);
          queuedTextRef.current = []; // Clear queue
          return;
        }

        // Check for iteration limit reached
        if (text.includes('ITERATION_LIMIT_REACHED')) {
          setPendingContinue({
            selected: 0,
            turnIndex,
          });
          setTurns(prev => {
            const updated = [...prev];
            if (updated[turnIndex]) {
              updated[turnIndex].progressText = '';
              updated[turnIndex].isProcessing = false;
            }
            turnsRef.current = updated;
            return updated;
          });
          setIsProcessing(false);
          return;
        }
        
        // If waiting for command approval, queue text instead of displaying it
        if (pendingCommandRef.current && pendingCommandRef.current.turnIndex === turnIndex) {
          queuedTextRef.current.push(text);
          return;
        }
        
        const cleaned = text.replace(/\n{3,}/g, '\n\n');
        if (cleaned && !cleaned.includes('━') && !cleaned.includes('Terminal Coding Agent') && 
            !cleaned.includes('💾') && !cleaned.includes('📂')) {
          setTurns(prev => {
            const updated = [...prev];
            if (updated[turnIndex]) {
              const timeline = updated[turnIndex].timeline;
              const lastEvent = timeline[timeline.length - 1];
              
              // If last event is text, append to it (streaming)
              if (lastEvent && lastEvent.type === 'text') {
                lastEvent.text += cleaned;
              } else {
                // Otherwise create new text block
                updated[turnIndex].timeline = [
                  ...timeline,
                  {
                    type: 'text',
                    timestamp: Date.now(),
                    text: cleaned,
                  }
                ];
              }
            }
            turnsRef.current = updated;
            return updated;
          });
        }
      },
      header: () => {},
      info: (text: string) => {
        setTurns(prev => {
          const updated = [...prev];
          if (updated[turnIndex]) {
            updated[turnIndex].progressText = text;
          }
          return updated;
        });
      },
      success: () => {},
      error: (text: string) => {
        setTurns(prev => {
          const updated = [...prev];
          if (updated[turnIndex]) {
            updated[turnIndex].timeline = [
              ...updated[turnIndex].timeline,
              {
                type: 'text',
                timestamp: Date.now(),
                text: `Error: ${text}`,
              }
            ];
          }
          return updated;
        });
      },
      tool: (toolName: string, details?: any) => {
        const opType: Operation['type'] = 
          toolName === 'read_file' ? 'read' :
          toolName === 'write_file' ? 'write' :
          toolName === 'directory_read' ? 'list' :
          toolName === 'directory_search' ? 'search' :
          toolName === 'execute_command' ? 'command' :
          toolName === 'execute_server_command' ? 'server' : 'read';

        const newOp: Operation = {
          type: opType,
          name: toolName,
          path: details?.path || toolName.replace('_', ' '),
          summary: details?.summary,
          lines: details?.lines,
          diff: details?.diff,
          expanded: undefined, // Auto-expand if < 60 lines, otherwise requires ctrl+r
          toolUseId: details?.toolUseId,
        };

        if (opType === 'command') {
          const commandText = details?.command || details?.path || toolName.replace('_', ' ');
          newOp.path = commandText;
          newOp.summary = undefined;
          // If there's already a pending command, mark this one as queued
          const hasPending = pendingCommandRef.current && pendingCommandRef.current.turnIndex === turnIndex;
          newOp.commandState = {
            command: commandText,
            cwd: details?.cwd,
            timeout: details?.timeout,
            status: hasPending ? 'pending' : 'pending', // Always pending initially
            output: '',
          };
        } else if (opType === 'server') {
          const commandText = details?.command || details?.path || toolName.replace('_', ' ');
          newOp.path = commandText;
          newOp.serverState = {
            id: details?.toolUseId || `server-${Date.now()}`,
            command: commandText,
            cwd: details?.cwd,
            status: pendingCommandRef.current ? 'pending' : 'pending',
            output: '',
          };
        }
        
        setTurns(prev => {
          const updated = [...prev];
          if (updated[turnIndex]) {
            updated[turnIndex].timeline = [
              ...updated[turnIndex].timeline,
              {
                type: 'operation',
                timestamp: Date.now(),
                operation: newOp,
              }
            ];
            updated[turnIndex].progressText = `Executing ${toolName}...`;
          }
          // Update ref synchronously so write() can find it immediately
          turnsRef.current = updated;
          return updated;
        });
      },
      updateLastOperation: (details: any) => {
        setTurns(prev => {
          const updated = [...prev];
          if (updated[turnIndex]) {
            const timeline = updated[turnIndex].timeline;
            // Find last operation
            for (let i = timeline.length - 1; i >= 0; i--) {
              if (timeline[i].type === 'operation' && timeline[i].operation) {
                const op = timeline[i].operation!;
                if (details.path) op.path = details.path;
                if (details.summary) op.summary = details.summary;
                if (details.lines) op.lines = details.lines;
                if (details.diff) op.diff = details.diff;
                break;
              }
            }
          }
          // Update ref synchronously for immediate visibility
          turnsRef.current = updated;
          return updated;
        });
      },
      verbose: () => {},
      context: () => {},
      startSpinner: (text: string) => {
        setTurns(prev => {
          const updated = [...prev];
          if (updated[turnIndex]) {
            updated[turnIndex].progressText = text;
          }
          return updated;
        });
      },
      stopSpinner: () => {
        setTurns(prev => {
          const updated = [...prev];
          if (updated[turnIndex]) {
            updated[turnIndex].progressText = undefined;
          }
          return updated;
        });
      },
    };
    
    setOutputHandler(customOutput);

    try {
      await runAgentGraph(query, verbose || false, { checkpoint, threadId: sessionId, model: selectedModel });
      checkpointStoreRef.current?.recordSessionActivity(sessionId);

      setTurns(prev => {
        const updated = [...prev];
        if (updated[turnIndex]) {
          updated[turnIndex].isProcessing = false;
          updated[turnIndex].progressText = undefined;
          if (updated[turnIndex].timeline.length === 0) {
            updated[turnIndex].timeline = [{
              type: 'text',
              timestamp: Date.now(),
              text: 'Task completed successfully.',
            }];
          }
        }
        return updated;
      });
    } catch (error: any) {
      checkpointStoreRef.current?.recordSessionActivity(sessionId);
      
      // Log full error with stack trace for debugging
      const errorMessage = error?.message || String(error);
      const stackTrace = error?.stack || 'No stack trace available';
      console.error('Full error:', errorMessage);
      console.error('Stack trace:', stackTrace);
      
      setTurns(prev => {
        const updated = [...prev];
        if (updated[turnIndex]) {
          updated[turnIndex].isProcessing = false;
          updated[turnIndex].timeline = [
            ...updated[turnIndex].timeline,
            {
              type: 'text',
              timestamp: Date.now(),
              text: `Error: ${errorMessage}\n\nStack trace logged to console.`,
            }
          ];
        }
        return updated;
      });
    } finally {
      checkpointStoreRef.current?.recordSessionActivity(sessionId);
      const { Output } = await import('./output.js');
      setOutputHandler(new Output());
      setIsProcessing(false);
    }
  }, [verbose, checkpoint, sessionId, isProcessing, turns.length, updateCommandOperation]);

  const toggleLastOperation = useCallback(() => {
    setTurns(prev => {
      const updated = [...prev];
      
      // Collect all write operations with diffs
      const writeOps: Array<{ turnIndex: number; eventIndex: number; op: Operation }> = [];
      for (let i = updated.length - 1; i >= 0; i--) {
        const timeline = updated[i].timeline;
        for (let j = timeline.length - 1; j >= 0; j--) {
          if (timeline[j].type === 'operation' && timeline[j].operation) {
            const op = timeline[j].operation!;
            if (op.type === 'write' && op.diff && op.diff.length > 0) {
              writeOps.push({ turnIndex: i, eventIndex: j, op });
            }
          }
        }
      }
      
      if (writeOps.length === 0) {
        return updated;
      }
      
      // Find the next operation to toggle (cycle through them)
      let targetIndex = 0;
      if (lastToggledOperationRef.current) {
        const last = lastToggledOperationRef.current;
        // Find index of last toggled operation
        const lastIdx = writeOps.findIndex(
          w => w.turnIndex === last.turnIndex && w.eventIndex === last.eventIndex
        );
        if (lastIdx >= 0) {
          // Move to next operation (cycle back to start if at end)
          targetIndex = (lastIdx + 1) % writeOps.length;
        }
      }
      
      // Toggle the target operation
      const target = writeOps[targetIndex];
      target.op.expanded = target.op.expanded === undefined 
        ? false  // If auto-expanded (< 60 lines), collapse it
        : !target.op.expanded;  // Otherwise toggle
      
      // Remember which one we toggled
      lastToggledOperationRef.current = {
        turnIndex: target.turnIndex,
        eventIndex: target.eventIndex,
      };
      
      return updated;
    });
  }, []);

  useInput((inputChar: string, key: any) => {
    // Handle model selection first
    if (!modelSelected) {
      if (key.upArrow) {
        setModelMenuIndex(prev => Math.max(0, prev - 1));
        return;
      }
      if (key.downArrow) {
        setModelMenuIndex(prev => Math.min(AVAILABLE_MODELS.length - 1, prev + 1));
        return;
      }
      if (key.return) {
        const selectedModelData = AVAILABLE_MODELS[modelMenuIndex];
        const provider = selectedModelData.provider;
        
        // Validate API key for selected model
        if (provider === 'openai' && !process.env.OPENAI_API_KEY) {
          setShutdownMessage('❌ OPENAI_API_KEY not found. Please add it to your .env file to use GPT models.');
          setTimeout(() => exit(), 2000);
          return;
        }
        if (provider === 'anthropic' && !process.env.ANTHROPIC_API_KEY) {
          setShutdownMessage('❌ ANTHROPIC_API_KEY not found. Please add it to your .env file to use Claude models.');
          setTimeout(() => exit(), 2000);
          return;
        }
        
        setSelectedModel(selectedModelData.id);
        setModelSelected(true);
        return;
      }
      if (key.escape || (key.ctrl && inputChar === 'c')) {
        exit();
        return;
      }
      return; // Ignore all other inputs during model selection
    }
    
    const pending = pendingCommandRef.current;
    // Handle pending command approval before forwarding to PTY
    if (pending) {
      if (key.leftArrow || key.upArrow) {
        setPendingCommandState(prev => (prev ? { ...prev, selected: 0 } : null));
        return;
      }
      if (key.rightArrow || key.downArrow) {
        setPendingCommandState(prev => (prev ? { ...prev, selected: 1 } : null));
        return;
      }
      if (key.escape) {
        const context = pending;
        setPendingCommandState(null);
        if (context.eventIndex >= 0) {
          updateCommandOperation(context.turnIndex, context.eventIndex, (op) => {
            const state = op.commandState ?? {
              command: context.command,
              cwd: context.cwd,
              timeout: context.timeout,
              output: '',
              status: 'pending',
            };
            return {
              ...op,
              commandState: {
                ...state,
                status: 'cancelled',
              },
            };
          });
        }
        if (context.toolUseId) {
          if (context.isServer) {
            rejectPendingServerCommand(context.toolUseId, 'Server command approval dismissed');
            removeServerSession(context.toolUseId);
          } else {
            resolvePendingCommand(context.toolUseId, {
              stdout: '',
              stderr: 'Command approval dismissed',
              exitCode: -1,
            });
          }
        }
        flushQueuedText(context.turnIndex);
        setTurns(prev => {
          const updated = [...prev];
          if (updated[context.turnIndex]) {
            updated[context.turnIndex].progressText = undefined;
          }
          turnsRef.current = updated;
          return updated;
        });
        if (!input) {
          setInput('Please suggest a different approach');
        }
        return;
        }
        if (key.return) {
        const selection = pending.selected;
        const context = pending;
        setPendingCommandState(null);
        if (selection === 0) {
          if (context.isServer) {
            if (context.toolUseId) {
              startServerSession({
                command: context.command,
                cwd: context.cwd,
                timeout: context.timeout,
                turnIndex: context.turnIndex,
                eventIndex: context.eventIndex,
                toolUseId: context.toolUseId,
              });
            }
          } else {
            startCommandSession(context);
          }
        } else {
          if (context.eventIndex >= 0) {
            updateCommandOperation(context.turnIndex, context.eventIndex, (op) => {
              const state = op.commandState ?? {
                command: context.command,
                cwd: context.cwd,
                timeout: context.timeout,
                output: '',
                status: 'pending',
              };
              return {
                ...op,
                commandState: {
                  ...state,
                  status: 'cancelled',
                },
              };
            });
          }
          if (context.toolUseId) {
            if (context.isServer) {
              resolvePendingServerCommand(context.toolUseId, {
                stdout: JSON.stringify({ status: 'rejected', command: context.command, cwd: context.cwd }),
                stderr: '',
                exitCode: -1,
              });
              removeServerSession(context.toolUseId);
            } else {
              resolvePendingCommand(context.toolUseId, {
                stdout: '',
                stderr: 'Command not approved by user',
                exitCode: -1,
              });
            }
          }
          flushQueuedText(context.turnIndex);
          setTurns(prev => {
            const updated = [...prev];
            if (updated[context.turnIndex]) {
              updated[context.turnIndex].progressText = undefined;
            }
            turnsRef.current = updated;
            return updated;
          });
          if (!input) {
            setInput('The command was not suitable because ');
          }
        }
        return;
      }
      // Block all other input when waiting for approval
      return;
    }

    // Handle pending continue approval
    if (pendingContinue) {
      if (key.leftArrow || key.upArrow) {
        setPendingContinue(prev => (prev ? { ...prev, selected: 0 } : null));
        return;
      }
      if (key.rightArrow || key.downArrow) {
        setPendingContinue(prev => (prev ? { ...prev, selected: 1 } : null));
        return;
      }
      if (key.return) {
        const selection = pendingContinue.selected;
        const tIdx = pendingContinue.turnIndex;
        setPendingContinue(null);
        
        if (selection === 0) {
          // User chose to continue
          resolvePendingContinue(true);
        } else {
          // User chose to stop
          resolvePendingContinue(false);
          setTurns(prev => {
            const updated = [...prev];
            if (updated[tIdx]) {
              updated[tIdx].timeline = [
                ...updated[tIdx].timeline,
                {
                  type: 'text',
                  timestamp: Date.now(),
                  text: '\nStopped at iteration limit.',
                }
              ];
              updated[tIdx].isProcessing = false;
            }
            turnsRef.current = updated;
            return updated;
          });
        }
        return;
      }
      if (key.escape) {
        // Same as choosing No
        const tIdx = pendingContinue.turnIndex;
        setPendingContinue(null);
        resolvePendingContinue(false);
        setTurns(prev => {
          const updated = [...prev];
          if (updated[tIdx]) {
            updated[tIdx].timeline = [
              ...updated[tIdx].timeline,
              {
                type: 'text',
                timestamp: Date.now(),
                text: '\nStopped at iteration limit.',
              }
            ];
            updated[tIdx].isProcessing = false;
          }
          turnsRef.current = updated;
          return updated;
        });
        return;
      }
      return; // Block all other input during continue prompt
    }

    if (key.ctrl && inputChar === 'x') {
      stopServerSession();
      return;
    }

    if (key.ctrl && (inputChar === 'e' || inputChar === 'E')) {
      if (serverSessionsRef.current.length > 0) {
        setServerSessionsCollapsed(prev => !prev);
      }
      return;
    }

    // Interactive command session: forward input to PTY
    // Only forward if BOTH are set AND the command is actually running (status === 'running')
    const activeSession = activeCommandSessionRef.current;
    if (activeSession) {
      const sessionRecord = commandSessionRef.current;
      if (!sessionRecord) {
        // Session already cleaned up – ensure state reflects that
        setActiveCommandSessionState(null);
        return;
      }
      // Double-check the command is still running by checking the operation status
      const { turnIndex, eventIndex } = activeSession;
      const currentTurn = turnsRef.current[turnIndex];
      const event = currentTurn?.timeline[eventIndex];
      const isRunning = event?.type === 'operation' && 
                       event.operation?.commandState?.status === 'running';
      
      if (!isRunning) {
        // Command completed but refs not cleared yet - sync state and stop forwarding
        setActiveCommandSessionState(null);
        return;
      }
      
      const session = sessionRecord.session;
      if (key.ctrl && inputChar === 'c') {
        session.write('\x03');
        return;
      }
      if (key.return) {
        session.write('\r');
        return;
      }
      if (key.tab) {
        session.write('\t');
        return;
      }
      if (key.backspace || key.delete) {
        session.write('\x7f');
        return;
      }
      if (key.leftArrow) {
        session.write('\x1b[D');
        return;
      }
      if (key.rightArrow) {
        session.write('\x1b[C');
        return;
      }
      if (key.upArrow) {
        session.write('\x1b[A');
        return;
      }
      if (key.downArrow) {
        session.write('\x1b[B');
        return;
      }
      if (key.escape) {
        session.write('\x1b');
        return;
      }
      if (inputChar) {
        session.write(inputChar);
        return;
      }
      return;
    }

    if (key.escape || (key.ctrl && inputChar === 'c')) {
      // If in prompt mode, cancel it instead of exiting
      if (promptMode?.active) {
        setPromptMode(null);
        setInput('');
        return;
      }
      gracefulShutdown();
      return;
    }
    
    // Allow ctrl+r to toggle the most recent write operation with diff
    if (key.ctrl && inputChar === 'r') {
      toggleLastOperation();
      return;
    }
    
    if (isProcessing) {
      return;
    }
    
    if (key.return) {
      if (input.trim()) {
        // Handle prompt mode
        if (promptMode?.active) {
          const newInputs = [...promptMode.collectedInputs, input];
          
          // Check if we have more prompts
          if (newInputs.length < promptMode.prompts.length) {
            // Show next prompt
            setPromptMode({
              ...promptMode,
              collectedInputs: newInputs,
              prompt: promptMode.prompts[newInputs.length],
            });
            setInput('');
          } else {
            // All inputs collected, execute callback
            setPromptMode(null);
            setInput('');
            promptMode.callback(newInputs);
          }
        } else {
          // Normal command mode
          runQuery(input);
          setInput('');
        }
      }
    } else if (key.backspace || key.delete) {
      setInput(prev => prev.slice(0, -1));
    } else if (key.upArrow) {
      if (history.length > 0 && historyIndex < history.length - 1) {
        const newIndex = historyIndex + 1;
        setHistoryIndex(newIndex);
        setInput(history[history.length - 1 - newIndex]);
      }
    } else if (key.downArrow) {
      if (historyIndex > 0) {
        const newIndex = historyIndex - 1;
        setHistoryIndex(newIndex);
        setInput(history[history.length - 1 - newIndex]);
      } else if (historyIndex === 0) {
        setHistoryIndex(-1);
        setInput('');
      }
    } else if (!key.ctrl && !key.meta && inputChar) {
      setInput(prev => prev + inputChar);
    }
  });

  const handleToggleOperation = useCallback((turnIndex: number, eventIndex: number) => {
    setTurns(prev => {
      const updated = [...prev];
      const event = updated[turnIndex].timeline[eventIndex];
      if (event.type === 'operation' && event.operation) {
        event.operation.expanded = !event.operation.expanded;
      }
      return updated;
    });
  }, []);

  const activeCommandPosition = activeCommandSession;
  
  // Separate completed turns from the current processing turn
  const completedTurns = turns.filter((turn, index) => !turn.isProcessing);
  const currentTurn = turns.find((turn, index) => turn.isProcessing);
  const currentTurnIndex = turns.findIndex(turn => turn.isProcessing);

  // Show model selection screen if model not yet selected
  if (!modelSelected) {
    const hasOpenAI = Boolean(process.env.OPENAI_API_KEY);
    const hasAnthropic = Boolean(process.env.ANTHROPIC_API_KEY);
    
    return (
      <Box flexDirection="column">
        {/* Logo */}
        <Box flexDirection="column" marginBottom={2}>
          <Text color="red" bold>╔╦╗╔═╗╦═╗╔═╗╔═╗╔╦╗╔═╗╔═╗╔╦╗╔═╗╦═╗</Text>
          <Text color="red" bold>║║║║╣ ╠╦╝║ ╦║╣ ║║║╠═╣╚═╗ ║ ║╣ ╠╦╝</Text>
          <Text color="red" bold>╩ ╩╚═╝╩╚═╚═╝╚═╝╩ ╩╩ ╩╚═╝ ╩ ╚═╝╩╚═</Text>
        </Box>
        
        {/* Model selection */}
        <Box flexDirection="column" borderStyle="round" borderColor="yellow" paddingX={2} paddingY={1}>
          <Text color="yellow" bold>Select a model:</Text>
          <Box height={1} />
          
          {AVAILABLE_MODELS.map((model, index) => {
            const isAvailable = model.provider === 'openai' ? hasOpenAI : hasAnthropic;
            const color = !isAvailable ? 'gray' : (index === modelMenuIndex ? 'green' : 'white');
            const dimmed = !isAvailable;
            
            return (
              <Box key={model.id} marginBottom={index === AVAILABLE_MODELS.length - 1 ? 0 : 1}>
                <Text color={color} dimColor={dimmed}>
                  {index === modelMenuIndex ? '● ' : '○ '}{model.name}
                  {!isAvailable && ' (API key missing)'}
                </Text>
              </Box>
            );
          })}
          
          <Box height={1} />
          <Text color="gray" dimColor>
            {AVAILABLE_MODELS[modelMenuIndex].description}
          </Text>
          {(() => {
            const selectedModel = AVAILABLE_MODELS[modelMenuIndex];
            const isAvailable = selectedModel.provider === 'openai' ? hasOpenAI : hasAnthropic;
            if (!isAvailable) {
              return (
                <>
                  <Box height={1} />
                  <Text color="red">
                    ⚠️  {selectedModel.provider === 'openai' ? 'OPENAI_API_KEY' : 'ANTHROPIC_API_KEY'} not found in .env
                  </Text>
                </>
              );
            }
            return null;
          })()}
          <Box height={1} />
          <Text color="gray" dimColor>
            Use arrow keys ↑↓ to select, Enter to confirm
          </Text>
        </Box>
      </Box>
    );
  }
  
  return (
    <Box flexDirection="column">
      {shutdownMessage && (
        <Box marginBottom={1}>
          <Text color="red" bold>{shutdownMessage}</Text>
        </Box>
      )}

      {/* Completed turns - use Static so they scroll naturally */}
      <Static items={completedTurns}>
        {(turn, index) => {
          const originalIndex = turns.indexOf(turn);
          return (
            <Box key={originalIndex} marginBottom={2}>
              <TurnView
                turn={turn}
                turnIndex={originalIndex}
                onToggleOperation={(eventIdx) => handleToggleOperation(originalIndex, eventIdx)}
                pendingCommand={null}
                activeCommand={null}
              />
            </Box>
          );
        }}
      </Static>

      {/* Current processing turn - render dynamically */}
      {currentTurn && (
        <Box marginBottom={2}>
          <TurnView
            turn={currentTurn}
            turnIndex={currentTurnIndex}
            onToggleOperation={(eventIdx) => handleToggleOperation(currentTurnIndex, eventIdx)}
            pendingCommand={
              pendingCommand && pendingCommand.turnIndex === currentTurnIndex ? pendingCommand : null
            }
            activeCommand={
              activeCommandPosition && activeCommandPosition.turnIndex === currentTurnIndex
                ? activeCommandPosition
                : null
            }
          />
        </Box>
      )}

      {serverSessions.length > 0 && (
        <Box flexDirection="column" marginBottom={1}>
          <Box>
            <Text color="cyan" bold>Server Sessions</Text>
            <Text> </Text>
            <Text color="gray" dimColor>
              {serverSessionsCollapsed
                ? '(ctrl + e to see all server sessions)'
                : '(ctrl + e to collapse)'}
            </Text>
          </Box>
          {!serverSessionsCollapsed && (
            <>
              <Box height={1} />
              {serverSessions.map(session => (
                <ServerSessionView key={session.id} session={session} />
              ))}
              <Text color="gray" dimColor>Press Ctrl+X to stop the most recent server session.</Text>
            </>
          )}
        </Box>
      )}

      {/* Spacer */}
      <Box height={1} />
      
      {/* Input section - fixed at bottom */}
      <Box flexDirection="column">
        {/* Show prompt if in prompt mode */}
        {promptMode?.active && (
          <Box marginBottom={1}>
            <Text color="yellow">{promptMode.prompt} </Text>
          </Box>
        )}
        
        {pendingCommand && (
          <Box marginBottom={1}>
            <Text color="yellow">Waiting for command approval: Use arrow keys to select Yes/No, then press Enter</Text>
          </Box>
        )}

        {pendingContinue && (
          <Box marginBottom={1} flexDirection="column" borderStyle="round" borderColor="yellow" paddingX={1}>
            <Text color="yellow" bold>
              🔄 Reached 25 iterations
            </Text>
            <Box marginTop={1}>
              <Text>Continue with more tool calls?  </Text>
              <Text color={pendingContinue.selected === 0 ? 'green' : 'white'}>
                {pendingContinue.selected === 0 ? '[✓]' : '[ ]'} Yes
              </Text>
              <Text>  </Text>
              <Text color={pendingContinue.selected === 1 ? 'red' : 'white'}>
                {pendingContinue.selected === 1 ? '[x]' : '[ ]'} No
              </Text>
            </Box>
            <Box marginTop={1}>
              <Text dimColor>Use arrow keys to select, Enter to confirm, Esc to stop</Text>
            </Box>
          </Box>
        )}
        
        <Box
          borderStyle="round"
          borderColor="red"
          paddingX={1}
          paddingY={0}
          marginTop={0}
          width={(process.stdout.columns || 80) - 2}
        >
          <Text color="red" bold>› </Text>
          <Text>{input}</Text>
          {promptMode?.active ? (
            <Text color="yellow">█</Text>
          ) : (!isProcessing && !pendingCommand && !pendingContinue && !activeCommandSession && (
            <Text color="red">█</Text>
          ))}
          {activeCommandSession && (
            <Text color="red" dimColor> (typing goes to command)</Text>
          )}
        </Box>

        <Box marginTop={1}>
          {!promptMode?.active ? (
            <Text color="gray" dimColor>/help · Ctrl+X stop server · Ctrl+E toggle sessions · esc to quit</Text>
          ) : (
            <Text color="gray" dimColor>({promptMode.collectedInputs.length + 1}/{promptMode.prompts.length}) • esc to cancel</Text>
          )}
        </Box>
      </Box>
    </Box>
  );
}

interface RenderTUIOptions {
  verbose?: boolean;
  checkpoint?: boolean;
  sessionId: string;
  resume?: boolean;
}

export function renderTUI({
  verbose,
  checkpoint = true,
  sessionId,
  resume = false,
}: RenderTUIOptions) {
  // Print session info to stderr (persists in scrollback)
  process.stderr.write(`\x1b[1m\x1b[32mSession ID:\x1b[0m ${sessionId}\n`);
  if (resume) {
    process.stderr.write(`\x1b[33mResuming session...\x1b[0m\n`);
  } else {
    process.stderr.write(`\x1b[2mResume later with: npm start -- --resume ${sessionId}\x1b[0m\n\n`);
  }
  
  const { unmount, waitUntilExit } = render(
    <AgentUI
      verbose={verbose}
      checkpoint={checkpoint}
      sessionId={sessionId}
      resume={resume}
    />
  );
  
  waitUntilExit();
}
