#!/usr/bin/env node
import 'dotenv/config';
import { randomUUID } from 'crypto';
import { renderTUI } from './tui.js';

interface CLIOptions {
  verbose: boolean;
  resumeSessionId?: string;
}

function parseCLIArgs(argv: string[]): CLIOptions {
  const options: CLIOptions = { verbose: false };

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];

    if (arg === '--verbose') {
      options.verbose = true;
      continue;
    }

    if (arg === '--resume' || arg === '-resume') {
      const value = argv[i + 1];
      if (!value || value.startsWith('-')) {
        throw new Error('Missing session ID after --resume');
      }
      options.resumeSessionId = value;
      i += 1;
      continue;
    }

    if (arg.startsWith('--resume=')) {
      const value = arg.split('=')[1];
      if (!value) {
        throw new Error('Missing session ID after --resume');
      }
      options.resumeSessionId = value;
      continue;
    }

    if (arg.startsWith('-') && arg !== '-resume') {
      throw new Error(`Unknown flag: ${arg}`);
    }
  }

  return options;
}

async function main() {
  const hasAnthropicKey = Boolean(process.env.ANTHROPIC_API_KEY);
  const hasOpenAIKey = Boolean(process.env.OPENAI_API_KEY);

  if (!hasAnthropicKey && !hasOpenAIKey) {
    console.error('❌ Error: No API keys found');
    console.error('   Create a .env file with:');
    console.error('   ANTHROPIC_API_KEY=your-anthropic-key');
    console.error('   OPENAI_API_KEY=your-openai-key');
    console.error('');
    console.error('   Or set them as environment variables:');
    console.error('   export ANTHROPIC_API_KEY=your-anthropic-key');
    console.error('   export OPENAI_API_KEY=your-openai-key');
    process.exit(1);
  }

  // Warn about missing keys
  if (!hasAnthropicKey) {
    console.warn('⚠️  Warning: ANTHROPIC_API_KEY not found - Claude models will not work');
  }
  if (!hasOpenAIKey) {
    console.warn('⚠️  Warning: OPENAI_API_KEY not found - GPT models will not work');
  }
  if (!hasAnthropicKey || !hasOpenAIKey) {
    console.log(''); // Empty line for spacing
  }

  let options: CLIOptions;

  try {
    options = parseCLIArgs(process.argv.slice(2));
  } catch (error) {
    console.error(`❌ ${error instanceof Error ? error.message : String(error)}`);
    console.error('   Usage: npm start [--verbose] [--resume <sessionId>]');
    process.exit(1);
  }

  const sessionId = options.resumeSessionId ?? randomUUID();
  const isResume = Boolean(options.resumeSessionId);

  try {
    // Always run TUI mode with command approval
    renderTUI({
      verbose: options.verbose,
      checkpoint: true,
      sessionId,
      resume: isResume,
    });
  } catch (error) {
    console.error(`❌ Error: ${error instanceof Error ? error.message : String(error)}`);
    if (options.verbose && error instanceof Error && error.stack) {
      console.error(`\nStack trace:\n${error.stack}`);
    }
    process.exit(1);
  }
}

main();
