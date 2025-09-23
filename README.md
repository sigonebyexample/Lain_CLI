
# 🎨 Lain_CLI - ASCII Media System Monitor

![Python](https://img.shields.io/badge/Python-3.6%2B-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)
![Terminal](https://img.shields.io/badge/Terminal-Fancy-brightgreen)

A stunning terminal-based system monitor that transforms your media files into beautiful ASCII art while displaying real-time system performance metrics. Perfect for developers, sysadmins, and anyone who loves terminal aesthetics! ✨

## ✨ Features

- 🖼️ **Media to ASCII Conversion**: Display images and videos as beautiful ASCII art
- 📊 **Real-time System Monitoring**: CPU, memory, disk, network, and battery stats
- 🎨 **Multiple ASCII Styles**: 10+ different character sets for unique visual effects
- 🎥 **Video Support**: Play videos converted to real-time ASCII animation
- 🎮 **Interactive Controls**: Navigate media, switch styles, toggle details
- 📱 **Responsive UI**: Adapts to terminal size with detailed/compact views
- 📈 **History Graphs**: Visualize CPU and memory usage trends
- 🎪 **Visual Enhancements**: Borders, colors, and progress bars

## 🚀 Quick Start

### Installation

1. **Install dependencies**:
```bash
pip install psutil numpy Pillow opencv-python
```
# Basic Usage
bash
```
python Lain.py --folder /path/to/media --style artistic --refresh 0.5
```
# 🎯 Usage Examples

Display media from current directory:
bash
```
python Lain.py
```
# Use a specific ASCII style:
bash
```
python Lain.py --style blocks
```
# Monitor a custom media folder:
bash
```
python Lain.py --folder ~/Pictures --refresh 1.0
```
# 🎨 ASCII Styles Available
```
Style	Preview	Description
artistic	░▒▓█	Most detailed, great for photos
blocks	█▓▒░	Solid block characters
smooth	░▒▓█	Gradient-style rendering
high_contrast	@$8W	High contrast characters
minimal	#+.	Simple and clean
standard	@#S%	Balanced default style
detailed	@#S%?*	More detail than standard
redmi	■□▪▫	Geometric shapes
tiny	•·	Minimal dot patterns
small_optimized	@#*+	Optimized for small terminals
🎮 Controls
Key	Action
← →	Navigate between media files
S	Cycle through ASCII styles
D	Toggle detailed/compact view
Q	Quit application
```
# 📊 System Metrics Displayed

    CPU Usage: Overall and per-core utilization with history graph

    Memory: Used/available RAM with percentage and history

    Disk Space: Storage usage with visual progress bar

    Network: Upload/download traffic statistics

    Battery: Percentage and charging status (if available)

    System Info: Uptime and process count

# 🔧 Requirements

    Python 3.6+

    curses (usually included with Python on Unix systems)

    psutil >=5.8.0

    numpy >=1.21.0

    Pillow >=9.0.0

    opencv-python >=4.5.0 (for video support)

# Installation for Different Systems

Ubuntu/Debian:
bash
```
sudo apt-get install python3-dev
pip install psutil numpy Pillow opencv-python
```
macOS:
bash
```
brew install python
pip install psutil numpy Pillow opencv-python
```
Windows (requires Windows Terminal or similar):
bash
```
pip install windows-curses psutil numpy Pillow opencv-python
```
# 🖼️ Supported Media Formats

Images: JPG, JPEG, PNG, BMP, GIF, TIFF
Videos: MP4, AVI, MOV, MKV, WEBM, FLV
Quick Start

    First, install the required packages (if you haven't already):

bash

pip install psutil pillow numpy opencv-python

    Run the script directly:

bash

python3 system_monitor.py

If you want to use your own media files:

Option 1: Place media files in the same folder

    Just put your images (JPG, PNG, etc.) and videos (MP4, AVI, etc.) in the same folder as the Python script

    The program will automatically detect them

Option 2: Create a subfolder for media
bash

# Create a media folder
mkdir media

# Move your media files into it
mv *.jpg *.png *.mp4 media/

# Run the script pointing to the media folder
python3 system_monitor.py --folder media

Simple Test

To test if it works immediately:
bash

python3 system_monitor.py --style blocks

Available ASCII Styles to Try:

    standard - Basic characters

    artistic - Detailed artistic characters (default)

    blocks - Block characters for solid look

    smooth - Smooth gradient blocks

    minimal - Minimalist style

Example:
bash

python3 system_monitor.py --style blocks --refresh 0.3

Controls Once Running:

    Arrow keys (← →) - Switch between media files

    S - Change ASCII style

    D - Toggle detailed view

    Q - Quit

If you get errors:

Make sure all dependencies are installed:
bash

pip list | grep -E "(psutil|Pillow|numpy|opencv)"

If OpenCV fails to install:
bash

# Try this instead of opencv-python
pip install opencv-python-headless

The script should run immediately since everything is in the same folder! Just make sure you have some image or video files in the directory for the best experience.

# 📝 License

This project is licensed under the MIT License. See the LICENSE file for details.
🤝 Contributing

Contributions are welcome! Feel free to submit issues and pull requests.
