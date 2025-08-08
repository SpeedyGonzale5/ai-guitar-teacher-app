# Deployment Guide

This guide walks you through deploying your AI Sports Analysis Tool to GitHub and setting up a new repository.

## Step 1: Prepare Your Repository

### Clean Up Large Files
Since you originally cloned this from another repository, you'll want to create a fresh start:

```bash
# Remove git history (if you want a fresh start)
rm -rf .git

# Initialize new git repository
git init

# Add all files
git add .

# Make initial commit
git commit -m "Initial commit: AI Sports Analysis Tool"
```

### Check File Sizes
Before committing, ensure no large video files are included:

```bash
# Check for large files
find . -size +50M -type f

# If any large files are found, add them to .gitignore
echo "large_video_file.mp4" >> .gitignore
```

## Step 2: Create GitHub Repository

### Option A: GitHub Web Interface
1. Go to [GitHub.com](https://github.com)
2. Click "New Repository"
3. Choose a descriptive name (e.g., `ai-sports-analysis`, `sports-ai-tracker`)
4. Add description: "AI-powered sports analysis tool with real-time feedback and statistics overlay"
5. Choose Public or Private
6. **Don't** initialize with README (you already have one)
7. Click "Create repository"

### Option B: GitHub CLI
```bash
# Install GitHub CLI if not already installed
# Then create repository
gh repo create ai-sports-analysis --public --description "AI-powered sports analysis tool"
```

## Step 3: Connect and Push

```bash
# Add your new repository as origin
git remote add origin https://github.com/yourusername/ai-sports-analysis.git

# Push to GitHub
git branch -M main
git push -u origin main
```

## Step 4: Repository Setup

### Add Repository Topics
In your GitHub repository settings, add relevant topics:
- `computer-vision`
- `sports-analysis`
- `opencv`
- `mediapipe`
- `python`
- `ai`
- `pose-detection`

### Enable GitHub Features

#### Issues
- Enable issues for bug reports and feature requests
- Create issue templates (optional)

#### Actions (CI/CD)
Create `.github/workflows/ci.yml` for automated testing:

```yaml
name: CI

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, '3.10', '3.11']

    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    
    - name: Lint with flake8
      run: |
        pip install flake8
        flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
    
    - name: Test imports
      run: python -c "import cv2; import mediapipe; import numpy; print('All imports successful')"
```

## Step 5: Documentation Enhancements

### Create Sample Data
Since video files are too large for GitHub, create sample JSON files:

```bash
# Create samples directory
mkdir samples

# Add sample JSON with anonymized data
cat > samples/sample_analysis.json << 'EOF'
{
  "shots": [
    {
      "timestamp_of_outcome": "0:15",
      "shot_by_player": "Player 1",
      "shot_type": "Forehand Drive",
      "point_winner": "Foreground Team", 
      "current_score": "1-0-1",
      "feedback": "Excellent shot placement! The forehand was executed with perfect timing."
    },
    {
      "timestamp_of_outcome": "0:32",
      "shot_by_player": "Player 2", 
      "shot_type": "Backhand Slice",
      "point_winner": "Far Side Team",
      "current_score": "1-1-2", 
      "feedback": "Great defensive shot with good court coverage."
    }
  ]
}
EOF
```

### Add License
Create a LICENSE file:

```bash
cat > LICENSE << 'EOF'
MIT License

Copyright (c) 2024 Your Name

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
EOF
```

## Step 6: Release Management

### Create Your First Release
```bash
# Tag your initial version
git tag -a v1.0.0 -m "Initial release: AI Sports Analysis Tool"
git push origin v1.0.0
```

### Release Notes Template
Create release notes on GitHub with:
- **Features**: Core functionality list
- **Installation**: Quick setup instructions  
- **Usage**: Basic usage examples
- **Requirements**: System requirements
- **Known Issues**: Any limitations

## Step 7: Community Setup

### Add Security Policy
Create `.github/SECURITY.md`:

```markdown
# Security Policy

## Reporting Security Vulnerabilities

If you discover a security vulnerability, please send an email to [your-email@domain.com]. 

Please do not report security vulnerabilities through public GitHub issues.
```

### Pull Request Template
Create `.github/pull_request_template.md`:

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature  
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tested locally
- [ ] Added/updated tests
- [ ] Verified with sample data

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings introduced
```

## Step 8: Promotion and Sharing

### Social Media
- Share on Twitter/X with relevant hashtags
- Post in relevant Reddit communities
- Share in computer vision Discord servers

### Documentation Sites
- Consider creating GitHub Pages for documentation
- Add to awesome lists (awesome-computer-vision, etc.)

### Package Registry
- Consider publishing to PyPI for easy installation
- Create Conda package for conda-forge

## Maintenance

### Regular Updates
- Keep dependencies updated
- Monitor for security vulnerabilities
- Respond to community issues promptly

### Backup Strategy
- GitHub provides automatic backups
- Consider additional backup of large files not in repo
- Document data sources and formats

---

Your project is now ready for the world! 🚀
