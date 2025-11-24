import chalk from 'chalk';
import ora from 'ora';
import boxen from 'boxen';

export class Output {
  private spinner: any = null;

  write(text: string) {
    process.stdout.write(text);
  }

  header(text: string) {
    this.write(boxen(text, { 
      title: '🤖 Coding Agent',
      borderColor: 'cyan',
      padding: 1,
      margin: { top: 1, bottom: 1 },
    }) + '\n');
  }

  startSpinner(text: string) {
    if (this.spinner) {
      this.spinner.stop();
    }
    this.spinner = ora(text).start();
  }

  stopSpinner(success: boolean = true, text?: string) {
    if (this.spinner) {
      if (success) {
        this.spinner.succeed(text);
      } else {
        this.spinner.fail(text);
      }
      this.spinner = null;
    }
  }

  info(text: string) {
    this.write(chalk.blue('ℹ ') + text + '\n');
  }

  success(text: string) {
    this.write(chalk.green('✅ ') + text + '\n');
  }

  error(text: string) {
    this.write(chalk.red('❌ ') + text + '\n');
  }

  tool(text: string) {
    this.write(chalk.yellow('🔧 ') + chalk.bold(text) + '\n');
  }

  verbose(text: string) {
    this.write(chalk.gray('💭 ') + text + '\n');
  }

  context(text: string) {
    this.write(chalk.cyan('📂 ') + text + '\n');
  }
}

