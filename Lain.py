#!/usr/bin/env python3
import curses
import time
import psutil
import os
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import argparse
import signal
import sys
import queue
import cv2
import threading

# Enhanced ASCII character sets for better visual quality
ASCII_CHARS = {
    'standard': ['@', '#', 'S', '%', '?', '*', '+', ';', ':', ',', '.'],
    'detailed': ['@', '#', 'S', '%', '?', '*', '+', ';', ':', ',', '.', "'", '"', '^', '~', '-', '_'],
    'blocks': ['█', '▓', '▒', '░', ' '],
    'minimal': ['#', '+', '.', ' '],
    'redmi': ['■', '□', '▪', '▫', ' '],
    'tiny': ['•', '·', ' '],
    'small_optimized': ['@', '#', '*', '+', ';', ',', '.', ' '],
    'smooth': [' ', '░', '▒', '▓', '█'],
    'high_contrast': [' ', '.', ':', '!', '/', 'r', '(', 'l', '1', 'Z', '4', 'H', '9', 'W', '8', '$', '@'],
    'artistic': [' ', '.', "'", '`', '^', '"', ',', ':', ';', 'I', 'l', '!', 'i', '>', '<', '~', '+', '_', '-', '?', 
                '[', ']', '{', '}', '1', ')', '(', '|', '\\', '/', 't', 'f', 'j', 'r', 'x', 'n', 'u', 'v', 'c', 'z', 
                'X', 'Y', 'U', 'J', 'C', 'L', 'Q', '0', 'O', 'Z', 'm', 'w', 'q', 'p', 'd', 'b', 'k', 'h', 'a', 'o', 
                '*', '#', 'M', 'W', '&', '8', '%', 'B', '@', '$']
}

def convert_frame_to_ascii(frame, width=80, ascii_chars=None):
    """
    Convert a frame to ASCII art using a character set based on brightness
    """
    if ascii_chars is None:
        ascii_chars = " .:-=+*#%@"
    
    height = int(frame.shape[0] * width / frame.shape[1] / 2) 
    if height == 0:
        height = 1
        
    resized_frame = cv2.resize(frame, (width, height))

    if len(resized_frame.shape) > 2:
        gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
    else:
        gray_frame = resized_frame
    
    normalized = gray_frame / 255.0
    ascii_frame = []
    
    for row in normalized:
        line = ""
        for pixel in row:
            index = int(pixel * (len(ascii_chars) - 1)) 
            line += ascii_chars[index]
        ascii_frame.append(line)
    
    return ascii_frame

class MediaManager:
    def __init__(self, media_folder=None):
        self.media_folder = media_folder or "."
        self.media_files = []
        self.current_index = 0
        self.current_frame = None
        self.original_frame = None
        self.media_type = None
        self.video_capture = None
        self.video_thread = None
        self.running = False
        self.frame_queue = queue.Queue(maxsize=2)
        
        # Discover media files
        self.discover_media_files()
        
    def discover_media_files(self):
        """Find all media files in the specified folder"""
        media_extensions = {
            '.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff',  # Images
            '.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv'    # Videos
        }
        
        self.media_files = []
        try:
            for file in os.listdir(self.media_folder):
                full_path = os.path.join(self.media_folder, file)
                if os.path.isfile(full_path):
                    if any(file.lower().endswith(ext) for ext in media_extensions):
                        self.media_files.append(full_path)
        except Exception as e:
            print(f"Error reading media folder: {e}")
        
        # Sort files for consistent navigation
        self.media_files.sort()
        
    def get_current_media(self):
        """Get current media file path"""
        if self.media_files and 0 <= self.current_index < len(self.media_files):
            return self.media_files[self.current_index]
        return None
    
    def next_media(self):
        """Move to next media file"""
        if self.media_files:
            self.stop_video()
            self.current_index = (self.current_index + 1) % len(self.media_files)
            return self.load_current_media()
        return False, "No media files"
    
    def prev_media(self):
        """Move to previous media file"""
        if self.media_files:
            self.stop_video()
            self.current_index = (self.current_index - 1) % len(self.media_files)
            return self.load_current_media()
        return False, "No media files"
    
    def load_current_media(self):
        """Load current media file"""
        media_path = self.get_current_media()
        if not media_path:
            return False, "No media file selected"
        
        self.media_type = None
        self.stop_video()
        
        # Check if it's a video file
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv')
        if media_path.lower().endswith(video_extensions):
            try:
                self.video_capture = cv2.VideoCapture(media_path)
                if not self.video_capture.isOpened():
                    return False, "Could not open video file"
                
                self.media_type = 'video'
                self.running = True
                # Start video thread
                self.video_thread = threading.Thread(target=self.video_loop, daemon=True)
                self.video_thread.start()
                return True, f"Video: {os.path.basename(media_path)}"
            except Exception as e:
                return False, f"Error loading video: {str(e)}"
        else:
            # It's an image file
            try:
                img = Image.open(media_path)
                self.media_type = 'image'
                self.current_frame = img
                self.original_frame = img.copy()
                return True, f"Image: {os.path.basename(media_path)}"
            except Exception as e:
                return False, f"Error loading image: {str(e)}"
    
    def video_loop(self):
        """Thread function for video playback using the new ASCII conversion"""
        while self.running and self.video_capture.isOpened():
            ret, frame = self.video_capture.read()
            if not ret:
                # Loop video
                self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            # Store the original frame for potential use
            self.original_frame = frame.copy()
            
            # Control playback speed based on video FPS
            video_fps = self.video_capture.get(cv2.CAP_PROP_FPS)
            frame_delay = 1.0 / video_fps if video_fps > 0 else 1.0 / 30
            time.sleep(frame_delay)
    
    def get_frame(self, original=False):
        """Get current frame - returns original image data if requested"""
        if original and self.original_frame is not None:
            return self.original_frame
        return self.current_frame
    
    def get_video_frame(self):
        """Get the current video frame for ASCII conversion"""
        if self.media_type == 'video' and self.original_frame is not None:
            return self.original_frame
        return None
    
    def stop_video(self):
        """Stop video playback"""
        self.running = False
        if self.video_thread and self.video_thread.is_alive():
            self.video_thread.join(timeout=1.0)
        if self.video_capture:
            self.video_capture.release()
        self.video_capture = None
        self.video_thread = None
    
    def stop(self):
        """Clean up resources"""
        self.stop_video()

class EnhancedSystemMonitor:
    def __init__(self, stdscr, media_folder=None, ascii_style='artistic', refresh_rate=1):
        self.stdscr = stdscr
        self.ascii_style = ascii_style
        self.refresh_rate = refresh_rate
        self.running = True
        self.ascii_art = []
        self.original_image_data = None
        self.show_help = False
        self.show_ascii = True
        self.detailed_view = False
        self.media_manager = MediaManager(media_folder)
        self.media_info = "Loading..."
        
        # System monitoring history
        self.cpu_history = []
        self.memory_history = []
        self.max_history = 20
        
        # Setup colors
        try:
            curses.start_color()
            curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
            curses.init_pair(2, curses.COLOR_CYAN, curses.COLOR_BLACK)
            curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)
            curses.init_pair(4, curses.COLOR_RED, curses.COLOR_BLACK)
            curses.init_pair(5, curses.COLOR_MAGENTA, curses.COLOR_BLACK)
            curses.init_pair(6, curses.COLOR_BLUE, curses.COLOR_BLACK)
            curses.init_pair(7, curses.COLOR_WHITE, curses.COLOR_BLACK)
        except:
            pass
        
        try:
            self.height, self.width = stdscr.getmaxyx()
        except:
            self.height, self.width = 24, 80
        
        signal.signal(signal.SIGINT, self.signal_handler)
        
        # Load first media file
        success, message = self.media_manager.load_current_media()
        self.media_info = message
        if success:
            self.prepare_display()
        
    def signal_handler(self, signum, frame):
        self.running = False
        self.media_manager.stop()
        
    def get_detailed_performance_data(self):
        """Get comprehensive system performance data"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1, percpu=True)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            network = psutil.net_io_counters()
            battery = psutil.sensors_battery()
            
            boot_time = psutil.boot_time()
            processes = len(psutil.pids())
            disk_io = psutil.disk_io_counters()
            
            # Update history
            avg_cpu = sum(cpu_percent) / len(cpu_percent)
            self.cpu_history.append(avg_cpu)
            self.memory_history.append(memory.percent)
            if len(self.cpu_history) > self.max_history:
                self.cpu_history.pop(0)
                self.memory_history.pop(0)
            
            return {
                "cpu": {
                    "percent": avg_cpu,
                    "percpu": cpu_percent,
                    "history": self.cpu_history.copy()
                },
                "memory": {
                    "total": memory.total,
                    "available": memory.available,
                    "used": memory.used,
                    "percent": memory.percent,
                    "history": self.memory_history.copy()
                },
                "disk": {
                    "total": disk.total,
                    "used": disk.used,
                    "free": disk.free,
                    "percent": disk.percent,
                    "read_bytes": disk_io.read_bytes if disk_io else 0,
                    "write_bytes": disk_io.write_bytes if disk_io else 0
                },
                "network": {
                    "sent": network.bytes_sent,
                    "received": network.bytes_recv
                },
                "battery": {
                    "percent": battery.percent if battery else None,
                    "power_plugged": battery.power_plugged if battery else None
                },
                "system": {
                    "boot_time": boot_time,
                    "processes": processes
                }
            }
        except Exception as e:
            return self.get_fallback_data()
    
    def get_fallback_data(self):
        """Fallback data in case of errors"""
        return {
            "cpu": {"percent": 0, "percpu": [0], "history": [0]},
            "memory": {"percent": 0, "history": [0]},
            "disk": {"percent": 0},
            "network": {"sent": 0, "received": 0},
            "battery": {"percent": None},
            "system": {"processes": 0}
        }
    
    def prepare_display(self):
        """Prepare the ASCII art display"""
        self.generate_ascii_art()
    
    def generate_ascii_art(self):
        """Generate high-quality ASCII art from current media frame"""
        # Handle video frames differently using the new conversion function
        if self.media_manager.media_type == 'video':
            frame = self.media_manager.get_video_frame()
            if frame is None:
                self.ascii_art = ["Loading video..."]
                return
            
            try:
                # Use the new ASCII conversion for videos
                ascii_width = max(20, min(self.width - 10, int(self.width * 0.8)))
                chars = ''.join(ASCII_CHARS.get(self.ascii_style, ASCII_CHARS['artistic']))
                self.ascii_art = convert_frame_to_ascii(frame, ascii_width, chars)
            except Exception as e:
                self.ascii_art = [f"Video error: {str(e)[:15]}"]
        
        # Handle images with the original method
        elif self.media_manager.media_type == 'image':
            frame = self.media_manager.get_frame()
            if frame is None:
                self.ascii_art = ["Loading..."]
                return
                
            try:
                # Enhance the image before converting to ASCII
                img = frame.copy()
                
                # Convert to grayscale
                if img.mode != 'L':
                    img = img.convert('L')
                
                # Enhance contrast
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(1.5)  # Increase contrast
                
                # Apply slight sharpening
                img = img.filter(ImageFilter.SHARPEN)
                
                # Calculate dimensions based on terminal size
                ascii_width = max(20, min(self.width - 10, int(self.width * 0.8)))
                aspect_ratio = img.height / img.width
                ascii_height = max(5, min(self.height - 10, int(ascii_width * aspect_ratio * 0.5)))
                
                # Resize with high-quality filter
                img = img.resize((ascii_width, ascii_height), Image.Resampling.LANCZOS)
                
                # Get pixels and character set
                pixels = np.array(img)
                chars = ASCII_CHARS.get(self.ascii_style, ASCII_CHARS['artistic'])
                
                # Generate ASCII art with improved mapping
                self.ascii_art = []
                for row in pixels:
                    line = ''.join([self.map_pixel_to_char(pixel, chars) for pixel in row])
                    self.ascii_art.append(line)
                    
            except Exception as e:
                self.ascii_art = [f"Error: {str(e)[:15]}"]
        else:
            self.ascii_art = ["No media loaded"]
    
    def map_pixel_to_char(self, pixel_value, chars):
        """Map pixel value to ASCII character with improved distribution"""
        # Normalize pixel value to 0-1 range
        normalized = pixel_value / 255.0
        
        # Apply gamma correction for better visual distribution
        gamma = 2.2
        corrected = normalized ** (1/gamma)
        
        # Map to character index
        index = min(int(corrected * len(chars)), len(chars) - 1)
        return chars[index]
    
    def draw_progress_bar(self, percent, width=20, label=""):
        filled = int(width * percent / 100)
        bar = '█' * filled + '░' * (width - filled)
        return f"{label}: [{bar}] {percent:.1f}%"
    
    def format_bytes(self, bytes):
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes < 1024.0:
                return f"{bytes:.1f}{unit}"
            bytes /= 1024.0
        return f"{bytes:.1f}PB"
    
    def safe_addstr(self, y, x, text, attr=0):
        try:
            if y < 0 or y >= self.height or x < 0 or x >= self.width:
                return False
            if x + len(text) > self.width:
                text = text[:self.width - x]
            if text:
                self.stdscr.addstr(y, x, text, attr)
                return True
        except:
            pass
        return False
    
    def draw_ascii_art(self, start_row):
        """Draw ASCII art with enhanced visual presentation"""
        if not self.ascii_art:
            return start_row
        
        # Calculate available width and center the art
        art_width = len(self.ascii_art[0]) if self.ascii_art else 0
        centered_x = max(2, (self.width - art_width) // 2)
        
        # Draw a border around the ASCII art if there's enough space
        if art_width + 4 < self.width and len(self.ascii_art) + 2 < self.height - start_row:
            border_width = min(self.width, art_width + 4)
            border_top = "╔" + "═" * (border_width - 2) + "╗"
            border_bottom = "╚" + "═" * (border_width - 2) + "╝"
            
            self.safe_addstr(start_row, (self.width - len(border_top)) // 2, border_top, curses.color_pair(6))
            start_row += 1
            
            for i, line in enumerate(self.ascii_art):
                if start_row + i < self.height - 2:
                    # Add border on the sides
                    border_left = "║" if i == 0 or i == len(self.ascii_art) - 1 else "│"
                    border_right = "║" if i == 0 or i == len(self.ascii_art) - 1 else "│"
                    
                    self.safe_addstr(start_row + i, centered_x - 2, border_left, curses.color_pair(6))
                    self.safe_addstr(start_row + i, centered_x, line, curses.color_pair(1))
                    self.safe_addstr(start_row + i, centered_x + len(line), border_right, curses.color_pair(6))
            
            self.safe_addstr(start_row + len(self.ascii_art), (self.width - len(border_bottom)) // 2, border_bottom, curses.color_pair(6))
            start_row += len(self.ascii_art) + 2
        else:
            # Not enough space for border, just draw the art
            for i, line in enumerate(self.ascii_art):
                if start_row + i < self.height - 2:
                    self.safe_addstr(start_row + i, centered_x, line, curses.color_pair(1))
            start_row += len(self.ascii_art) + 1
        
        return start_row
    
    def draw_detailed_system_info(self, data, start_row):
        """Draw detailed system information"""
        row = start_row
        
        # CPU Information with history graph
        cpu_info = f"CPU: {data['cpu']['percent']:.1f}% ({len(data['cpu']['percpu'])} cores)"
        self.safe_addstr(row, 2, cpu_info, curses.color_pair(2))
        
        # Draw CPU history graph if there's space
        if data['cpu']['history'] and self.width > 50:
            graph_width = min(20, self.width - len(cpu_info) - 6)
            graph = self.draw_history_graph(data['cpu']['history'], graph_width, max_value=100)
            self.safe_addstr(row, len(cpu_info) + 4, graph, curses.color_pair(2))
        row += 1
        
        # Memory Information
        mem_used = data['memory']['used'] / (1024**3)
        mem_total = data['memory']['total'] / (1024**3)
        mem_info = f"RAM: {mem_used:.1f}G/{mem_total:.1f}G ({data['memory']['percent']:.1f}%)"
        self.safe_addstr(row, 2, mem_info, curses.color_pair(3))
        
        # Draw memory history graph if there's space
        if data['memory']['history'] and self.width > 50:
            graph_width = min(20, self.width - len(mem_info) - 6)
            graph = self.draw_history_graph(data['memory']['history'], graph_width, max_value=100)
            self.safe_addstr(row, len(mem_info) + 4, graph, curses.color_pair(3))
        row += 1
        
        # Disk Information
        disk_used = data['disk']['used'] / (1024**3)
        disk_total = data['disk']['total'] / (1024**3)
        disk_info = f"Disk: {disk_used:.1f}G/{disk_total:.1f}G ({data['disk']['percent']:.1f}%)"
        self.safe_addstr(row, 2, disk_info, curses.color_pair(4))
        row += 1
        
        # Network Information
        net_sent = self.format_bytes(data['network']['sent'])
        net_recv = self.format_bytes(data['network']['received'])
        net_info = f"Net: ▲{net_sent} ▼{net_recv}"
        self.safe_addstr(row, 2, net_info, curses.color_pair(5))
        row += 1
        
        # Battery Information
        if data['battery']['percent'] is not None:
            battery_icon = "⚡" if data['battery']['power_plugged'] else "🔋"
            batt_info = f"Batt: {battery_icon} {data['battery']['percent']:.1f}%"
            self.safe_addstr(row, 2, batt_info, curses.color_pair(6))
            row += 1
        
        return row
    
    def draw_compact_system_info(self, data, start_row):
        """Draw compact system information"""
        row = start_row
        
        # Use a more compact layout for small terminals
        if self.width >= 60:
            info_line = f"CPU:{data['cpu']['percent']:.1f}% "
            info_line += f"RAM:{data['memory']['percent']:.1f}% "
            info_line += f"DSK:{data['disk']['percent']:.1f}%"
            
            if data['battery']['percent'] is not None:
                info_line += f" BAT:{data['battery']['percent']:.1f}%"
            
            self.safe_addstr(row, 2, info_line, curses.color_pair(2))
            row += 1
        else:
            # Very compact layout for very small terminals
            cpu_info = f"CPU:{data['cpu']['percent']:.1f}%"
            self.safe_addstr(row, 2, cpu_info, curses.color_pair(2))
            
            mem_info = f"RAM:{data['memory']['percent']:.1f}%"
            self.safe_addstr(row, self.width - len(mem_info) - 2, mem_info, curses.color_pair(3))
            row += 1
            
            if data['battery']['percent'] is not None:
                batt_info = f"BAT:{data['battery']['percent']:.1f}%"
                self.safe_addstr(row, 2, batt_info, curses.color_pair(6))
            
            disk_info = f"DSK:{data['disk']['percent']:.1f}%"
            self.safe_addstr(row, self.width - len(disk_info) - 2, disk_info, curses.color_pair(4))
            row += 1
        
        return row
    
    def draw_history_graph(self, history, width, max_value=100):
        """Draw a simple history graph"""
        if not history or width <= 0:
            return ""
        
        # Normalize values to graph height (1 line)
        values = [min(h / max_value, 1.0) for h in history[-width:]]
        chars = [' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█']
        
        graph = ""
        for value in values:
            index = min(int(value * (len(chars) - 1)), len(chars) - 1)
            graph += chars[index]
        
        return graph
    
    def draw_media_info(self, start_row):
        """Draw media file information"""
        row = start_row
        
        current_media = self.media_manager.get_current_media()
        if current_media:
            total_files = len(self.media_manager.media_files)
            current_num = self.media_manager.current_index + 1
            
            media_name = os.path.basename(current_media)
            if len(media_name) > self.width - 15:
                media_name = media_name[:self.width - 18] + "..."
            
            media_type = "Video" if self.media_manager.media_type == 'video' else "Image"
            media_info = f"{media_name} ({current_num}/{total_files}) - {media_type}"
            
            self.safe_addstr(row, 2, media_info, curses.color_pair(1))
            row += 1
        
        return row
    
    def draw_ui(self, data):
        try:
            # Get current terminal size
            self.height, self.width = self.stdscr.getmaxyx()
            
            # Check if terminal is too small
            if self.height < 10 or self.width < 30:
                try:
                    self.stdscr.clear()
                    msg = "Terminal too small. Please resize to at least 30x10."
                    self.safe_addstr(0, (self.width - len(msg)) // 2, msg, curses.A_BOLD | curses.color_pair(4))
                    self.stdscr.refresh()
                except:
                    pass
                return
            
            try:
                self.stdscr.clear()
            except:
                pass
            
            # Header with decorative elements
            header = "✨ Media System Monitor ✨"
            if len(header) > self.width:
                header = "System Monitor"
            
            self.safe_addstr(0, (self.width - len(header)) // 2, header, curses.color_pair(1) | curses.A_BOLD)
            
            # Draw media information
            row = 2
            row = self.draw_media_info(row)
            
            # Draw system information based on available space
            if self.detailed_view and self.width >= 50 and self.height >= 15:
                row = self.draw_detailed_system_info(data, row)
            else:
                row = self.draw_compact_system_info(data, row)
            
            # Draw ASCII art if there's enough space
            if self.height - row > 5:
                row = self.draw_ascii_art(row)
            
            # Status bar with controls
            status_bar = f"Style: {self.ascii_style}"
            if self.width > 50:
                status_bar += f" | Detail: {'ON' if self.detailed_view else 'OFF'}"
            
            controls = "←→:Media S:Style"
            if self.width > 40:
                controls += " D:Detail"
            controls += " Q:Quit"
            
            # Ensure controls fit on screen
            if len(controls) > self.width - 2:
                controls = "←→:Media S:Style Q:Quit"
            
            self.safe_addstr(self.height - 1, 2, status_bar, curses.color_pair(3))
            self.safe_addstr(self.height - 1, self.width - len(controls) - 2, controls, curses.color_pair(4))
            
            self.stdscr.refresh()
            
        except Exception as e:
            pass
    
    def toggle_detailed_view(self):
        self.detailed_view = not self.detailed_view
    
    def next_ascii_style(self):
        """Cycle through ASCII styles"""
        styles = list(ASCII_CHARS.keys())
        current_index = styles.index(self.ascii_style) if self.ascii_style in styles else 0
        self.ascii_style = styles[(current_index + 1) % len(styles)]
        self.prepare_display()
    
    def next_media(self):
        success, message = self.media_manager.next_media()
        self.media_info = message
        if success:
            self.prepare_display()
    
    def prev_media(self):
        success, message = self.media_manager.prev_media()
        self.media_info = message
        if success:
            self.prepare_display()
    
    def run(self):
        try:
            self.stdscr.nodelay(True)
            curses.curs_set(0)
        except:
            pass
        
        while self.running:
            try:
                key = self.stdscr.getch()
                if key == ord('q'):
                    break
                elif key == ord('d'):
                    self.toggle_detailed_view()
                elif key == ord('s'):
                    self.next_ascii_style()
                elif key == curses.KEY_RIGHT:
                    self.next_media()
                elif key == curses.KEY_LEFT:
                    self.prev_media()
                elif key == curses.KEY_RESIZE:
                    # Handle terminal resize
                    self.height, self.width = self.stdscr.getmaxyx()
                    self.prepare_display()
            except:
                pass
            
            # For videos, we need to update the display regularly
            if self.media_manager.media_type == 'video':
                self.prepare_display()
            
            try:
                data = self.get_detailed_performance_data()
                self.draw_ui(data)
            except:
                time.sleep(0.1)
            
            time.sleep(self.refresh_rate)
        
        self.media_manager.stop()

def main():
    parser = argparse.ArgumentParser(description='Beautiful ASCII System Monitor')
    parser.add_argument('--folder', '-f', type=str, default=".",
                       help='Folder containing media files')
    parser.add_argument('--style', '-s', type=str, default='artistic', 
                       choices=list(ASCII_CHARS.keys()), help='ASCII art style')
    parser.add_argument('--refresh', '-r', type=float, default=0.5,
                       help='Refresh rate in seconds')
    
    args = parser.parse_args()
    
    try:
        def monitor(stdscr):
            monitor = EnhancedSystemMonitor(stdscr, args.folder, args.style, args.refresh)
            monitor.run()
        
        curses.wrapper(monitor)
        print("Media monitor exited successfully")
    except KeyboardInterrupt:
        print("\nMonitor stopped by user")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
