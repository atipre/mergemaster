# MergeMaster 🎯
<div align="center">

![Node.js](https://img.shields.io/badge/node-%3E%3D18-brightgreen?logo=node.js&logoColor=white) ![Version](https://img.shields.io/badge/version-1.0.2-ff69b4) ![Contributions](https://img.shields.io/badge/contributions-welcome-brightgreen) ![License](https://img.shields.io/badge/license-MIT-d10000) 

</div>

> AI-powered coding assistant with an interactive terminal interface

MergeMaster is a powerful terminal-based coding agent that helps you write, understand, and refactor code directly from your command line. Built with multi-model support, session persistence, and an intuitive TUI.

## ⚙️ Features

- 🤖 **Multi-Model Support**: Choose from GPT-5.1, GPT-4o, GPT-4o Mini, Claude Sonnet 4.5, Claude Opus 4.1, Claude Haiku 3.5
- 💾 **Session Persistence**: Resume conversations with SQLite checkpointing
- 🎨 **Beautiful TUI**: Clean terminal interface with real-time streaming
- ✅ **Command Approval**: Review and approve all shell commands before execution
- 🔍 **Smart Search**: ripgrep integration for fast codebase exploration
- 📝 **Markdown Rendering**: GitHub-flavored markdown support
- 🖥️ **Server Management**: Run and monitor long-running processes (dev servers, etc.)
- ⚡ **Streaming Responses**: Real-time output as the agent thinks

## 🎥 Demo

https://github.com/user-attachments/assets/6066b9a4-f1e1-4603-8f88-ba9ce46bcb94


## 🚀 Quick Start

### Installation

**Global Install (Recommended):**
```bash
npm install -g mergemaster
```

**Or run locally:**
```bash
git clone https://github.com/atipre/mergemaster.git
cd mergemaster
npm install
npm start
```

### Prerequisites

1. **Node.js 18+** - [Download](https://nodejs.org/)
2. **ripgrep** - For code search
   ```bash
   # macOS
   brew install ripgrep
   
   # Linux
   sudo apt install ripgrep
   
   # Windows
   choco install ripgrep
   ```

3. **API Keys** - Add to `.env` file or environment:
   ```bash
   ANTHROPIC_API_KEY=your_anthropic_key_here
   OPENAI_API_KEY=your_openai_key_here
   ```
   
   Get keys from:
   - Anthropic: https://console.anthropic.com/
   - OpenAI: https://platform.openai.com/api-keys

## 💻 Usage

### Start MergeMaster
```bash
mergemaster
```

Select your preferred model, then start chatting!

### Resume a Session
```bash
mergemaster --resume <session-id>
```

### Commands

**Slash Commands:**
- `/help` - Show available commands and shortcuts
- `/sessions-list` - View all saved sessions
- `/sessions-clear` - Clear session database

**Keyboard Shortcuts:**
- `↑/↓` - Navigate command history
- `ctrl+r` - Toggle diff expansion
- `ctrl+x` - Stop server
- `ctrl+e` - Toggle server sessions
- `esc` - Quit

## 🛠️ Development

```bash
# Install dependencies
npm install

# Run in development mode
npm run dev

# Build for production
npm run build

# Type check
npx tsc --noEmit
```

## 📖 How It Works

MergeMaster uses LangGraph to orchestrate a ReAct (Reasoning and Acting) agent loop:

1. **Explore** - Lists directories, searches files
2. **Understand** - Reads relevant code and context
3. **Plan** - Outlines implementation approach
4. **Implement** - Makes focused, incremental changes
5. **Verify** - Runs lints, tests, and validates

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
