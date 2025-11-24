import * as pty from 'node-pty';

export interface CommandResult {
  stdout: string;
  stderr: string;
  exitCode: number;
}

export interface CommandApproval {
  command: string;
  cwd?: string;
  timeout?: number;
}

export interface CommandPTYSession {
  write(data: string): void;
  resize(columns: number, rows: number): void;
  dispose(): void;
}

interface PendingCommandEntry {
  approval: CommandApproval;
  resolve: (result: CommandResult) => void;
  reject: (error: Error) => void;
}

const pendingCommands = new Map<string, PendingCommandEntry>();

interface PendingServerCommandEntry {
  approval: CommandApproval;
  resolve: (result: CommandResult) => void;
  reject: (error: Error) => void;
}

const pendingServerCommands = new Map<string, PendingServerCommandEntry>();

export function waitForCommandCompletion(
  toolUseId: string,
  approval: CommandApproval
): Promise<CommandResult> {
  if (!toolUseId) {
    throw new Error('toolUseId is required to track command approvals');
  }

  if (pendingCommands.has(toolUseId)) {
    throw new Error(`Command approval already pending for ${toolUseId}`);
  }

  return new Promise<CommandResult>((resolve, reject) => {
    pendingCommands.set(toolUseId, {
      approval,
      resolve,
      reject,
    });
  });
}

export function resolvePendingCommand(
  toolUseId: string,
  result: CommandResult
): boolean {
  const pending = pendingCommands.get(toolUseId);
  if (!pending) {
    return false;
  }

  pendingCommands.delete(toolUseId);
  pending.resolve(result);
  return true;
}

export function rejectPendingCommand(
  toolUseId: string,
  error: Error | string
): boolean {
  const pending = pendingCommands.get(toolUseId);
  if (!pending) {
    return false;
  }

  pendingCommands.delete(toolUseId);
  pending.reject(error instanceof Error ? error : new Error(error));
  return true;
}

export function getPendingCommand(toolUseId: string): CommandApproval | undefined {
  return pendingCommands.get(toolUseId)?.approval;
}

export function waitForServerCommand(
  toolUseId: string,
  approval: CommandApproval
): Promise<CommandResult> {
  if (!toolUseId) {
    throw new Error('toolUseId is required to track server command approvals');
  }

  if (pendingServerCommands.has(toolUseId)) {
    throw new Error(`Server command approval already pending for ${toolUseId}`);
  }

  return new Promise<CommandResult>((resolve, reject) => {
    pendingServerCommands.set(toolUseId, {
      approval,
      resolve,
      reject,
    });
  });
}

export function resolvePendingServerCommand(
  toolUseId: string,
  result: CommandResult
): boolean {
  const pending = pendingServerCommands.get(toolUseId);
  if (!pending) {
    return false;
  }

  pendingServerCommands.delete(toolUseId);
  pending.resolve(result);
  return true;
}

export function rejectPendingServerCommand(
  toolUseId: string,
  error: Error | string
): boolean {
  const pending = pendingServerCommands.get(toolUseId);
  if (!pending) {
    return false;
  }

  pendingServerCommands.delete(toolUseId);
  pending.reject(error instanceof Error ? error : new Error(error));
  return true;
}

export function getPendingServerCommand(toolUseId: string): CommandApproval | undefined {
  return pendingServerCommands.get(toolUseId)?.approval;
}

// Global state for continue approval (iteration limit)
let pendingContinueResolve: ((shouldContinue: boolean) => void) | null = null;
let pendingContinueReject: ((error: Error) => void) | null = null;

export function waitForContinue(): Promise<boolean> {
  return new Promise((resolve, reject) => {
    pendingContinueResolve = resolve;
    pendingContinueReject = reject;
  });
}

export function resolvePendingContinue(shouldContinue: boolean) {
  if (pendingContinueResolve) {
    pendingContinueResolve(shouldContinue);
    pendingContinueResolve = null;
    pendingContinueReject = null;
  }
}

export function rejectPendingContinue(error: Error) {
  if (pendingContinueReject) {
    pendingContinueReject(error);
    pendingContinueResolve = null;
    pendingContinueReject = null;
  }
}

/**
 * Execute command with PTY support - returns approval request marker
 * The TUI will handle the actual PTY execution after user approval
 */
export async function executeCommandTool(
  command: string,
  cwd?: string,
  options?: { timeout?: number }
): Promise<CommandResult> {
  // Return special marker that TUI recognizes as needing approval
  // Format: COMMAND_APPROVAL_NEEDED:COMMAND:CWD:TIMEOUT
  const approvalMarker = `COMMAND_APPROVAL_NEEDED:${command}${cwd ? `:CWD:${cwd}` : ''}${options?.timeout ? `:TIMEOUT:${options.timeout}` : ''}`;
  
  return {
    stdout: approvalMarker,
    stderr: '',
    exitCode: 0,
  };
}

/**
 * Execute long-running server command - returns special marker
 */
export async function executeServerCommandTool(
  command: string,
  cwd?: string,
  options?: { timeout?: number }
): Promise<CommandResult> {
  const marker = `SERVER_COMMAND_REQUEST:${command}${cwd ? `:CWD:${cwd}` : ''}${options?.timeout ? `:TIMEOUT:${options.timeout}` : ''}`;
  return {
    stdout: marker,
    stderr: '',
    exitCode: 0,
  };
}

/**
 * Execute command in PTY with full terminal control
 */
export function executeCommandWithPTY(
  command: string,
  {
    cwd = process.cwd(),
    timeout = 900,
    onData,
    onExit,
    onError,
  }: {
    cwd?: string;
    timeout?: number;
    onData?: (chunk: string) => void;
    onExit?: (result: { exitCode: number; signal?: number; timedOut: boolean }) => void;
    onError?: (error: Error) => void;
  } = {}
): CommandPTYSession {
  const isWindows = process.platform === 'win32';
  const defaultShell = isWindows ? process.env.COMSPEC || 'cmd.exe' : process.env.SHELL || '/bin/bash';

  const shell = defaultShell;
  const shellArgs = isWindows
    ? ['/d', '/s', '/c', command]
    : ['-lc', command];

  const ptyProcess = pty.spawn(shell, shellArgs, {
    name: 'xterm-color',
    cols: process.stdout.columns || 80,
    rows: process.stdout.rows || 24,
    cwd,
    env: process.env as { [key: string]: string },
  });

  let timedOut = false;
  let timeoutId: NodeJS.Timeout | null = null;

  if (timeout && timeout > 0) {
    timeoutId = setTimeout(() => {
      timedOut = true;
      try {
        ptyProcess.kill();
      } catch (error) {
        onError?.(error as Error);
      }
    }, timeout * 1000);
  }

  ptyProcess.onData((data: string) => {
    onData?.(data);
  });

  ptyProcess.onExit((event: { exitCode: number; signal?: number }) => {
    if (timeoutId) {
      clearTimeout(timeoutId);
    }
    onExit?.({
      exitCode: event.exitCode,
      signal: event.signal,
      timedOut,
    });
  });

  return {
    write(data: string) {
      try {
        ptyProcess.write(data);
      } catch (error) {
        onError?.(error as Error);
      }
    },
    resize(columns: number, rows: number) {
      try {
        ptyProcess.resize(columns, rows);
      } catch {
        // Ignore resize errors (e.g., when PTY already closed)
      }
    },
    dispose() {
      if (timeoutId) {
        clearTimeout(timeoutId);
      }
      try {
        ptyProcess.kill();
      } catch {
        // ignore kill errors
      }
    },
  };
}
