# Linux Commands Cheatsheet

## 📁 File Operations

| Command | Description | Example |
|---------|-------------|---------|
| `ls -alh` | Detailed list with human-readable sizes | `ls -alh` |
| `ls -la` | Show all files including hidden | `ls -la` |
| `mkdir` | Create single directory | `mkdir folder_name` |
| `mkdir -p` | Create nested directories | `mkdir -p path/to/folder` |
| `cp` | Copy file | `cp file1.txt file2.txt` |
| `cp -r` | Copy directory recursively | `cp -r dir1 dir2` |
| `mv` | Rename file | `mv old.txt new.txt` |
| `mv` | Move file to directory | `mv file.txt dir/` |

## 👀 File Viewing

| Command | Description | Example |
|---------|-------------|---------|
| `cat` | Display entire file | `cat file.txt` |
| `head -N` | Show first N lines | `head -10 file.csv` |
| `less` | Page through file | `less file.txt` |

## 🔍 Search & Patterns

| Command | Description | Example |
|---------|-------------|---------|
| `grep "pattern"` | Search for text | `grep "error" log.txt` |
| `grep -i` | Case-insensitive search | `grep -i "error" log.txt` |
| `grep -r` | Recursive search | `grep -r "TODO" /path/` |
| `grep "^pattern"` | Lines starting with pattern | `grep "^Name" data.csv` |
| `grep "[0-9]"` | Lines containing numbers | `grep "[0-9]" file.txt` |

## 🔄 Git Operations

| Command | Description | Example |
|---------|-------------|---------|
| `git clone` | Clone repository | `git clone https://github.com/user/repo.git` |
| `git pull` | Pull latest changes | `git pull origin main` |
| `git add` | Stage all changes | `git add .` |
| `git commit` | Commit changes | `git commit -m "message"` |
| `git push` | Push to remote | `git push origin main` |

## ⚙️ System & Permissions

| Command | Description | Example |
|---------|-------------|---------|
| `pwd` | Show current directory | `pwd` |
| `chmod +x` | Make file executable | `chmod +x script.sh` |
| `rm` | Remove file | `rm file.txt` |
| `rm -rf` | Remove directory recursively | `rm -rf directory/` |

## 🧭 Navigation in `less`

| Key | Description | Usage |
|-----|-------------|-------|
| `Spacebar` | Scroll down one page | While in `less` |
| `b` | Scroll up one page | While in `less` |
| `Enter` | Scroll down one line | While in `less` |
| `/pattern` | Search forward for pattern | `/error` |
| `?pattern` | Search backward for pattern | `?warning` |
| `n` | Go to next match | While searching |
| `N` | Go to previous match | While searching |
| `g` | Go to start of file | While in `less` |
| `G` | Go to end of file | While in `less` |
| `q` | Quit less | While in `less` |
| `ZZ` | Quit less (alternative) | While in `less` |
| `h` | Show help screen | While in `less` |

## 📝 Common Regex Patterns

| Pattern | Description | Example |
|---------|-------------|---------|
| `^pattern` | Starts with pattern | `^hello` |
| `pattern$` | Ends with pattern | `world$` |
| `[0-9]` | Any digit | `[0-9]` |
| `[A-Z]` | Any uppercase letter | `[A-Z]` |
| `.` | Any single character | `a.b` |

## 🔬 Colab Specific

| Note | Usage |
|------|-------|
| Prefix with `!` for shell commands | `!ls -la` |

---

**Pro Tip:** Save this as `linux_cheatsheet.md` in your GitHub repo for easy reference!

