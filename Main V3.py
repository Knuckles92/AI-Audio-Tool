import os
import sys
import threading
import time
import wave
import logging
import queue
import json
import shutil
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext, simpledialog

import sounddevice as sd
import numpy as np
import requests
import pygame
import yaml
import keyring
from pydub import AudioSegment
from docx import Document
from fpdf import FPDF
import pyperclip
import pystray
from PIL import Image, ImageDraw
import tkinterdnd2
import tempfile

from pynput import keyboard

# Initialize basic logging for debugging and application behavior tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AudioTranscriberApp:
    """
    AudioTranscriberApp encapsulates the functionalities of the Audio Recorder and Transcriber application.
    It handles audio recording, transcription via API, user interface interactions, profile management, and more.
    """

    def __init__(self, root):
        self.root = root
        self.root.title("Audio Recorder and Transcriber")

        # Initialize pygame for playing sounds
        pygame.mixer.init()
        
        # Initialize reading state
        self.reading_cancelled = False

        # Initialize configuration
        self.config = self.load_config()
        self.settings = self.config['settings']

        # Initialize tray icon
        self.initialize_tray_icon()

        # Initialize other attributes
        self.FORMAT = 'int16'
        self.CHUNK = 1024
        self.recording = False
        self.cancel_recording = False
        self.cancel_transcription = False
        self.WAVE_OUTPUT_FILENAME = "output.wav"
        self.transcription_queue = queue.Queue()
        self.transcription_history = []
        self.history_file = os.path.join(self.get_executable_dir(), 'transcription_history.json')
        self.backup_history_file = os.path.join(self.get_executable_dir(), 'transcription_history_backup.json')

        # Initialize paused attribute
        self.paused = False

        # Create UI components
        self.create_ui()
        self.bind_events()

        # Perform system check
        self.system_check()

        # Start authentication and API key setup
        self.authenticate_and_initialize()

    def authenticate_and_initialize(self):
        """
        Initializes the app by verifying the API key.
        """
        if self.load_and_verify_api_key():
            # Proceed with further initialization
            logging.info("API key setup successful.")
            # Load transcription history
            self.load_transcription_history()
            # Update history list in UI
            self.update_history_list()
        else:
            messagebox.showerror("API Key Missing", "API key not found or invalid. Exiting.")
            self.on_closing()


    def authenticate_user(self):
        return True

    def load_and_verify_api_key(self):
        """
        Retrieves the API key from keyring. If not found, prompts the user to enter it.
        Returns True if API key is successfully retrieved, False otherwise.
        """
        api_key = keyring.get_password("whisper_api", "api_key")
        if not api_key:
            logging.info("API key not found. Prompting user to enter it.")
            return self.prompt_for_api_key()
        else:
            self.API_KEY = api_key
            return True

    def prompt_for_api_key(self):
        """
        Prompts the user to enter their OpenAI API key and stores it securely.
        Returns True if the API key is provided, False otherwise.
        """
        while True:
            api_key = simpledialog.askstring("API Key", "Enter your OpenAI API key for transcription:", show='*')
            if api_key:
                keyring.set_password("whisper_api", "api_key", api_key)
                self.API_KEY = api_key
                logging.info("API key saved securely in keyring.")
                return True
            else:
                retry = messagebox.askretrycancel("API Key Missing", "API key cannot be empty. Retry?")
                if not retry:
                    return False


    def show_auth_window_recursively(self):
        """
        This method was causing infinite recursion in the original code.
        It has been removed to prevent such issues.
        """
        pass  # Removed to prevent recursive calls

    def read_aloud(self):
        """
        Uses OpenAI's text-to-speech API to read the transcription aloud.
        """
        # Get the text from the recording result text box
        self.recording_result_text.config(state='normal')
        text = self.recording_result_text.get(1.0, tk.END).strip()
        self.recording_result_text.config(state='disabled')

        if not text:
            self.update_status("No text to read aloud", "warning")
            return

        # Disable the read aloud button and enable cancel button
        self.read_aloud_button.config(state='disabled')
        self.cancel_button.config(state='normal', text="Cancel Reading", command=self.cancel_reading)
        self.reading_cancelled = False

        # Start the reading in a separate thread
        self.read_thread = threading.Thread(target=self._read_aloud_thread, args=(text,), daemon=True)
        self.read_thread.start()

    def _read_aloud_thread(self, text):
        """
        Thread function that handles the actual text-to-speech conversion and playback.
        """
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.API_KEY)
            
            # Create a temporary file
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, f'tts_audio_{int(time.time())}.mp3')
            
            # Call OpenAI TTS API
            response = client.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=text
            )
            
            # Write the response content to the temp file
            response.write_to_file(temp_path)
            
            if not self.reading_cancelled:
                try:
                    # Create a separate mixer channel for TTS playback
                    pygame.mixer.Channel(1).play(pygame.mixer.Sound(temp_path))
                    
                    # Wait for playback to finish or cancellation
                    while pygame.mixer.Channel(1).get_busy() and not self.reading_cancelled:
                        pygame.time.Clock().tick(10)
                        
                finally:
                    # Clean up the channel
                    pygame.mixer.Channel(1).stop()
            
            # Try to remove the temporary file
            try:
                os.remove(temp_path)
            except Exception as e:
                logging.warning(f"Could not remove temporary file {temp_path}: {e}")
            
            if not self.reading_cancelled:
                self.root.after(0, lambda: self.update_status("Reading text aloud completed", "info"))
            else:
                self.root.after(0, lambda: self.update_status("Reading cancelled", "warning"))
            
        except Exception as e:
            logging.error(f"Error reading text aloud: {e}")
            error_msg = str(e)  # Capture the error message
            self.root.after(0, lambda: self.update_status(f"Error reading text: {error_msg}", "error"))
        finally:
            # Re-enable the read aloud button and reset cancel button
            self.root.after(0, self._reset_read_aloud_buttons)

    def _reset_read_aloud_buttons(self):
        """
        Resets the read aloud and cancel buttons to their default states.
        """
        self.read_aloud_button.config(state='normal')
        self.cancel_button.config(
            state='disabled',
            text="Cancel Transcription",
            command=self.cancel_transcription_action
        )

    def cancel_reading(self):
        """
        Cancels the current text-to-speech playback.
        """
        self.reading_cancelled = True
        pygame.mixer.Channel(1).stop()
        

    def load_config(self):
        """
        Loads the configuration from config.yaml. If the file doesn't exist, creates it with default settings.
        Ensures that the API key is retrieved securely from keyring.
        """
        # Define the config filename and default configuration
        config_filename = 'config.yaml'
        default_config = {
            "settings": {
                "save_location": "",
                "dark_mode": False,
                "bitrate": "128 kbps",
                "sample_rate": 44100,
                "transcription_model": "whisper-1",
                "volume": 1.0,
                "profiles": {},
                "gpt_model": "chatgpt-4o-latest"
            }
        }

        # Get the full path to the config file
        config_path = os.path.join(self.get_executable_dir(), config_filename)

        # If the config file doesn't exist, create it with default settings
        if not os.path.exists(config_path):
            try:
                with open(config_path, 'w') as file:
                    yaml.dump(default_config, file)
                logging.info(f"Created new config file at {config_path}")
            except PermissionError:
                logging.warning(f"Unable to create config file at {config_path}. Using default settings.")
                return default_config

        try:
            # Load the existing config file
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
                
                # Merge loaded config with default config to ensure all keys exist
                config = {**default_config, **config}
                config["settings"] = {**default_config["settings"], **config.get("settings", {})}
                
                # Ensure sample_rate is an integer
                config["settings"]["sample_rate"] = int(config["settings"].get("sample_rate", 44100))
                
                # Attempt to retrieve API key from keyring
                api_key = keyring.get_password("whisper_api", "api_key")
                
                # If API key not in keyring, check config file
                if not api_key:
                    api_key = config["settings"].get("api_key")
                    if api_key:
                        # If found in config, save to keyring for future use
                        keyring.set_password("whisper_api", "api_key", api_key)
                        logging.info("API key found in config file and saved to keyring.")
                    else:
                        # If not found anywhere, prompt user to enter it
                        logging.warning("API key not found in keyring or config file. Prompting user to enter it.")
                        self.prompt_for_api_key()
                
                # Set the API key in the config and class attribute
                config["settings"]["api_key"] = api_key
                self.API_KEY = api_key
                
                return config
        except Exception as e:
            # Log any errors and return default config if loading fails
            logging.error(f"Error while loading configuration: {e}")
            return default_config


    def save_config(self):
        """
        Saves the current settings to config.yaml. Excludes the API key to ensure it remains secure.
        """
        config_path = os.path.join(self.get_executable_dir(), 'config.yaml')
        try:
            # Exclude API key and auth key from being saved in config.yaml
            settings_to_save = self.settings.copy()
            settings_to_save.pop("api_key", None)
            settings_to_save.pop("auth_key", None)
            with open(config_path, 'w') as file:
                yaml.dump({"settings": settings_to_save}, file)
            logging.info(f"Configuration saved successfully to {config_path}")
        except Exception as e:
            logging.error(f"Error while saving configuration: {e}")
            messagebox.showerror("Error", f"Failed to save configuration: {e}")


    def get_executable_dir(self):
        """
        Get the directory of the executable or script.
        """
        if getattr(sys, 'frozen', False):
            # Running as compiled executable
            return os.path.dirname(sys.executable)
        else:
            # Running as script
            return os.path.dirname(os.path.abspath(__file__))

    def resource_path(self, relative_path):
        """
        Get absolute path to resource, works for dev and for PyInstaller.
        """
        try:
            # PyInstaller creates a temp folder and stores path in _MEIPASS
            base_path = sys._MEIPASS
        except Exception:
            base_path = os.path.abspath(".")

        return os.path.join(base_path, relative_path)

    def create_ui(self):
        """
        Creates and configures the main user interface components.
        """
        # Create menu bar
        menu_bar = tk.Menu(self.root)
        self.root.config(menu=menu_bar)

        # Create File menu
        file_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Export", command=self.export_transcription)
        file_menu.add_command(label="Settings", command=self.open_settings)
        file_menu.add_command(label="Load Previous Transcriptions", command=self.load_previous_transcriptions)
        file_menu.add_command(label="Help", command=self.open_help)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_closing)

        # Set up the style for tkinter widgets to have a consistent modern look
        self.style = ttk.Style()
        self.style.theme_use('clam')

        # Define styles
        self.define_styles()

        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Create frames for each tab
        self.recording_frame = ttk.Frame(self.notebook, padding="10", style="TFrame")
        self.profiles_frame = ttk.Frame(self.notebook, padding="10", style="TFrame")
        self.history_frame = ttk.Frame(self.notebook, padding="10", style="TFrame")

        # Add frames to notebook
        self.notebook.add(self.recording_frame, text="Recording")
        self.notebook.add(self.profiles_frame, text="Profiles")
        self.notebook.add(self.history_frame, text="History")

        # Create UI for recording tab
        self.create_recording_ui()

        # Create UI for profiles tab
        self.create_profiles_ui()

        # Create UI for history tab
        self.create_history_ui()

    def define_styles(self):
        """
        Defines custom styles for various widgets to ensure a consistent and modern UI appearance.
        """
        # Define button styles
        self.style.configure("Cancel.TButton", padding=10, relief="flat", background="#f44336", foreground="white")
        self.style.map("Cancel.TButton",
                       background=[('active', '#FFA500'), ('!active', 'grey')],
                       foreground=[('!active', 'white')])

        self.style.configure("Record.TButton", padding=10, relief="flat", background="#4CAF50", foreground="white")
        self.style.map("Record.TButton", background=[('active', '#45a049')])

        self.style.configure("Stop.TButton", padding=10, relief="flat", background="#f44336", foreground="white")
        self.style.map("Stop.TButton", background=[('active', '#e53935')])

        self.style.configure("Upload.TButton", padding=10, relief="flat", background="#8A2BE2", foreground="white")
        self.style.map("Upload.TButton", background=[('active', '#7B68EE')])

        self.style.configure("Clear.TButton", padding=10, relief="flat", background="#FF0000", foreground="white")
        self.style.map("Clear.TButton", background=[('active', '#CC0000')])  # Darker red on hover

        self.style.configure("ClearAll.TButton", padding=10, relief="flat", background="#FF4500", foreground="white")
        self.style.map("ClearAll.TButton", background=[('active', '#FF6347')])

        self.style.configure("Export.TButton", padding=10, relief="flat", background="#2196F3", foreground="white")
        self.style.map("Export.TButton", background=[('active', '#1976D2')])

        self.style.configure("Toggle.TButton", padding=10, relief="flat", background="#555555", foreground="white")
        self.style.map("Toggle.TButton", background=[('active', '#777777')])

        self.style.configure("Copy.TButton", padding=10, relief="flat", background="#4CAF50", foreground="white")
        self.style.map("Copy.TButton", background=[('active', '#45a049')])

        # Define label and frame styles
        self.style.configure("TLabel", padding=6, background="#f0f0f0")
        self.style.configure("TFrame", background="#f0f0f0")
        self.style.configure("TText", background="#ffffff", foreground="#000000")

        # Define button styles for profile page
        self.style.configure("Save.TButton", padding=10, relief="flat", background="#4CAF50", foreground="white")
        self.style.map("Save.TButton", background=[('active', '#45a049')])

        self.style.configure("Delete.TButton", padding=10, relief="flat", background="#f44336", foreground="white")
        self.style.map("Delete.TButton", background=[('active', '#d32f2f')])

        # Add ReadAloud button style
        self.style.configure("ReadAloud.TButton", padding=10, relief="flat", background="#2196F3", foreground="white")
        self.style.map("ReadAloud.TButton", background=[('active', '#1976D2')])

    def create_recording_ui(self):
        """
        Creates the user interface components for the Recording tab.
        """
        # Create frames within the recording tab
        self.top_frame = ttk.Frame(self.recording_frame, padding="10", style="TFrame")
        self.top_frame.pack(fill=tk.BOTH, expand=True)

        self.middle_frame = ttk.Frame(self.recording_frame, padding="10", style="TFrame")
        self.middle_frame.pack(fill=tk.BOTH, expand=True)

        self.bottom_frame = ttk.Frame(self.recording_frame, padding="10", style="TFrame")
        self.bottom_frame.pack(fill=tk.BOTH, expand=True)

        # Button container frame with grid layout
        self.button_container = ttk.Frame(self.top_frame)
        self.button_container.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        self.button_container.grid_columnconfigure(0, weight=1)
        self.button_container.grid_columnconfigure(1, weight=1)
        self.button_container.grid_columnconfigure(2, weight=1)
        self.button_container.grid_columnconfigure(3, weight=1)

        # Record button
        self.record_button = ttk.Button(
            self.button_container,
            text="Start Recording (Alt+R)",
            command=self.toggle_recording,
            style="Record.TButton"
        )
        self.record_button.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        # Upload button
        self.upload_button = ttk.Button(
            self.button_container,
            text="Upload Audio File\n(or drag and drop)",
            command=self.select_audio_file,
            style="Upload.TButton"
        )
        self.upload_button.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        # Cancel button
        self.cancel_button = ttk.Button(
            self.button_container,
            text="Cancel",
            command=self.cancel_transcription_action,
            style="Cancel.TButton",
            state=tk.DISABLED
        )
        self.cancel_button.grid(row=0, column=2, padx=5, pady=5, sticky="ew")

        # Clear button
        self.clear_button = ttk.Button(
            self.button_container,
            text="Clear",
            command=self.clear_transcription,
            style="Clear.TButton"
        )
        self.clear_button.grid(row=0, column=3, padx=5, pady=5, sticky="ew")

        # Setup drag and drop for upload button
        self.upload_button.drop_target_register(tkinterdnd2.DND_FILES)
        self.upload_button.dnd_bind('<<Drop>>', self.on_drop)

        # Transcription frame
        self.transcription_frame = ttk.Frame(self.middle_frame, padding="10", style="TFrame")
        self.transcription_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Profile selection dropdown
        profile_label = ttk.Label(self.transcription_frame, text="Transcription Profile:")
        profile_label.pack(side=tk.TOP, anchor="w", padx=5)
        self.recording_profile_var = tk.StringVar()
        self.recording_profile_dropdown = ttk.Combobox(self.transcription_frame, textvariable=self.recording_profile_var, width=20, state="readonly")
        self.recording_profile_dropdown.pack(side=tk.TOP, anchor="w", padx=5)
        self.update_recording_profile_dropdown()

        self.result_label = ttk.Label(
            self.transcription_frame,
            text="Transcription",
            font=("Helvetica", 14, "bold")
        )
        self.result_label.pack()

        # Rename self.result_text to self.recording_result_text
        self.recording_result_text = scrolledtext.ScrolledText(
            self.transcription_frame,
            height=10,
            width=50,
            wrap=tk.WORD,
            state='disabled'
        )
        self.recording_result_text.pack(pady=5, fill=tk.BOTH, expand=True)

        # Apply the style to the internal Text widget of ScrolledText
        self.recording_result_text.configure(bg=self.style.lookup("TText", "background"))
        self.recording_result_text.configure(fg=self.style.lookup("TText", "foreground"))

        # Read Aloud Button and Auto-Read Checkbox Frame
        read_aloud_frame = ttk.Frame(self.transcription_frame)
        read_aloud_frame.pack(pady=(5, 10))

        self.read_aloud_button = ttk.Button(
            read_aloud_frame,
            text="Read Aloud",
            command=self.read_aloud,
            style="ReadAloud.TButton"
        )
        self.read_aloud_button.pack(side=tk.LEFT, padx=5)

        # Auto-read checkbox
        self.auto_read_var = tk.BooleanVar(value=False)
        self.auto_read_checkbox = ttk.Checkbutton(
            read_aloud_frame,
            text="Auto-read new transcriptions",
            variable=self.auto_read_var
        )
        self.auto_read_checkbox.pack(side=tk.LEFT, padx=5)

        # Send History checkbox
        self.send_history_var = tk.BooleanVar(value=False)
        self.send_history_checkbox = ttk.Checkbutton(
            read_aloud_frame,
            text="Send History with Prompt",
            variable=self.send_history_var
        )
        self.send_history_checkbox.pack(side=tk.LEFT, padx=5)

        # Timer frame
        self.timer_frame = ttk.Frame(self.top_frame)
        self.timer_frame.pack(fill=tk.X, pady=10)

        self.timer_label = ttk.Label(
            self.timer_frame,
            text="Recording: 00:00",
            relief="solid",
            anchor="center",
            padding=10,
            foreground="white",
            background="#4CAF50"
        )
        self.timer_label.pack(side=tk.TOP, fill=tk.X, padx=25)

        # Progress bar for transcription
        self.progress = ttk.Progressbar(self.middle_frame, orient='horizontal', mode='determinate')
        self.progress.pack(fill=tk.X, pady=5)
        self.progress['value'] = 0
        self.progress.pack_forget()  # Hide initially

        # Status label
        self.status_label = ttk.Label(
            self.bottom_frame,
            text="Status: Ready",
            relief="solid",
            anchor="center",
            padding=10,
            foreground="white",
            background="#4CAF50"
        )
        self.status_label.pack(fill=tk.X, padx=10, pady=10, side=tk.TOP)

    def clear_transcription(self):
        """
        Clears the transcription text box and resets the UI elements.
        """
        self.clear_textbox(self.recording_result_text)
        self.update_status("Transcription cleared", "info")
        self.progress['value'] = 0
        self.progress.pack_forget()

    def create_history_ui(self):
        """
        Creates the user interface components for the History tab.
        """
        # Search bar for transcription history
        search_frame = ttk.Frame(self.history_frame)
        search_frame.pack(fill=tk.X, pady=5)

        search_label = ttk.Label(search_frame, text="Search:")
        search_label.pack(side=tk.LEFT, padx=5)

        self.search_entry = ttk.Entry(search_frame)
        self.search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.search_entry.bind("<KeyRelease>", self.on_search)

        # Create a frame to hold the listbox and its scrollbar
        list_frame = ttk.Frame(self.history_frame)
        list_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # History listbox
        self.history_listbox = tk.Listbox(list_frame, height=10)
        self.history_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.history_listbox.bind('<<ListboxSelect>>', self.on_history_select)

        # Scrollbar for the listbox
        history_scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.history_listbox.yview)
        history_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.history_listbox.config(yscrollcommand=history_scrollbar.set)

        # Buttons for history management
        # Create button frame with center alignment
        button_frame = ttk.Frame(self.history_frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        # Create inner frame to center the buttons
        button_container = ttk.Frame(button_frame)
        button_container.pack(expand=True, anchor="center")

        self.clear_selected_button = ttk.Button(
            button_container,
            text="Clear Selected",
            command=self.clear_selected_history,
            style="Clear.TButton"
        )
        self.clear_selected_button.pack(side=tk.LEFT, padx=5)

        self.clear_all_history_button = ttk.Button(
            button_container,
            text="Clear All",
            command=self.clear_all_history,
            style="ClearAll.TButton"
        )
        self.clear_all_history_button.pack(side=tk.LEFT, padx=5)

        self.copy_to_clipboard_button = ttk.Button(
            button_container,
            text="Copy to Clipboard",
            command=self.copy_selected_to_clipboard,
            style="Copy.TButton"
        )
        self.copy_to_clipboard_button.pack(side=tk.LEFT, padx=5)

        # History result text
        self.history_result_text = scrolledtext.ScrolledText(
            self.history_frame,
            height=10,
            width=50,
            wrap=tk.WORD,
            state='disabled'
        )
        self.history_result_text.pack(pady=5, fill=tk.BOTH, expand=True)

    def copy_selected_to_clipboard(self):
        """
        Copies the selected transcription from history to the clipboard.
        """
        selected_indices = self.history_listbox.curselection()
        if selected_indices:
            selected_index = selected_indices[0]
            selected_text = self.transcription_history[selected_index]
            pyperclip.copy(selected_text)
            self.update_status("Transcription copied to clipboard", "info")
        else:
            self.update_status("No transcription selected", "warning")

    def create_profiles_ui(self):
        """
        Creates the user interface components for the Profiles tab.
        """
        # Profile selection dropdown
        profile_selection_label = ttk.Label(self.profiles_frame, text="Select Profile to Edit:")
        profile_selection_label.pack(pady=5)
        self.profile_selection_var = tk.StringVar()
        self.profile_selection_dropdown = ttk.Combobox(
            self.profiles_frame,
            textvariable=self.profile_selection_var,
            width=30,
            state="readonly"
        )
        self.profile_selection_dropdown.pack(pady=5, fill=tk.X)
        self.profile_selection_dropdown.bind("<<ComboboxSelected>>", self.on_profile_select)
        self.update_profile_selection_dropdown()

        # Profile name input
        profile_name_frame = ttk.Frame(self.profiles_frame)
        profile_name_frame.pack(fill=tk.X, pady=5)
        profile_name_label = ttk.Label(profile_name_frame, text="Profile Name:")
        profile_name_label.pack(side=tk.LEFT, padx=5)
        self.profile_name_entry = ttk.Entry(profile_name_frame, width=30)
        self.profile_name_entry.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)

        # Profile description input
        profile_desc_label = ttk.Label(self.profiles_frame, text="Profile Description:")
        profile_desc_label.pack(pady=5)
        self.profile_desc_text = scrolledtext.ScrolledText(self.profiles_frame, height=5, width=40)
        self.profile_desc_text.pack(pady=5, fill=tk.BOTH, expand=True)

        # Button frame
        button_frame = ttk.Frame(self.profiles_frame)
        button_frame.pack(fill=tk.X, pady=10)

        # Center the buttons by creating a container frame
        button_container = ttk.Frame(button_frame)
        button_container.pack(expand=True, anchor="center")

        # Save profile button
        self.save_profile_button = ttk.Button(
            button_container,
            text="Save Profile",
            command=self.save_profile,
            style="Save.TButton"
        )
        self.save_profile_button.pack(side=tk.LEFT, padx=5)

        # Delete profile button
        self.delete_profile_button = ttk.Button(
            button_container,
            text="Delete Profile",
            command=self.delete_profile,
            style="Delete.TButton"
        )
        self.delete_profile_button.pack(side=tk.LEFT, padx=5)

    def update_profile_selection_dropdown(self):
        """
        Updates the profile selection dropdown with available profiles.
        """
        profiles = ["New Profile"] + list(self.settings.get("profiles", {}).keys())
        self.profile_selection_dropdown['values'] = profiles
        if self.profile_selection_var.get() not in profiles:
            self.profile_selection_var.set("New Profile")

    def delete_profile(self):
        selected_profile = self.profile_selection_var.get()
        if selected_profile == "New Profile":
            messagebox.showwarning("Warning", "Cannot delete 'New Profile'.")
            return

        confirm = messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete the profile '{selected_profile}'?")
        if confirm:
            del self.settings["profiles"][selected_profile]
            self.save_config()
            self.update_profile_selection_dropdown()
            self.update_recording_profile_dropdown()
            self.profile_name_entry.delete(0, tk.END)
            self.profile_desc_text.delete("1.0", tk.END)
            messagebox.showinfo("Success", f"Profile '{selected_profile}' deleted successfully.")

    def on_profile_select(self, event):
        """
        Handles the event when a profile is selected from the dropdown.
        """
        selected_profile = self.profile_selection_var.get()
        if selected_profile == "New Profile":
            # Clear fields for new profile creation
            self.profile_name_entry.delete(0, tk.END)
            self.profile_desc_text.delete("1.0", tk.END)
        else:
            # Load the selected profile's data
            profile_description = self.settings["profiles"].get(selected_profile, "")
            self.profile_name_entry.delete(0, tk.END)
            self.profile_name_entry.insert(0, selected_profile)
            self.profile_desc_text.delete("1.0", tk.END)
            self.profile_desc_text.insert(tk.END, profile_description)

    def save_profile(self):
        """
        Saves the current profile to the settings.
        """
        profile_name = self.profile_name_entry.get().strip()
        profile_desc = self.profile_desc_text.get("1.0", tk.END).strip()

        if not profile_name:
            messagebox.showerror("Error", "Profile name cannot be empty.")
            return

        self.settings.setdefault("profiles", {})[profile_name] = profile_desc
        self.save_config()
        self.update_profile_selection_dropdown()
        self.update_recording_profile_dropdown()

        # Set the selected profile to the saved one
        self.profile_selection_var.set(profile_name)
        messagebox.showinfo("Success", f"Profile '{profile_name}' saved successfully.")

    def update_recording_profile_dropdown(self):
        """
        Updates the recording profile dropdown with available profiles.
        """
        profiles = ["No Profile"] + list(self.settings.get("profiles", {}).keys())
        self.recording_profile_dropdown['values'] = profiles
        if self.recording_profile_var.get() not in profiles:
            self.recording_profile_var.set("No Profile")

    def bind_events(self):
        """
        Binds keyboard shortcuts and initializes the transcription queue processing.
        """
        # Bind keyboard shortcuts using pynput for global hotkeys
        self.initialize_global_hotkeys()

        # Start processing transcription queue
        self.root.after(100, self.process_transcription_queue)

        # Clear search bar
        self.search_entry.delete(0, tk.END)

    def toggle_recording(self):
        """
        Toggles the recording state. Can be called from both the UI and global hotkeys.
        """
        self.root.after(0, self._toggle_recording)

    def _toggle_recording(self):
        if self.recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        """
        Initiates the audio recording process.
        """
        if not self.check_api_key():
            return

        self.recording = True
        self.cancel_recording = False
        self.start_time = time.time()
        self.record_button.config(text="Stop Recording (Alt+R)", style="Stop.TButton")
        self.cancel_button.config(state=tk.NORMAL, text="Cancel Transcription")
        self.update_status("Recording in progress", "recording")
        self.update_timer_label("Recording: 00:00", "recording")
        self.clear_textbox(self.recording_result_text)

        self.record_thread = threading.Thread(target=self.record_audio, daemon=True)
        self.record_thread.start()
        self.update_timer()
        self.play_sound('sound.mp3')

    def stop_recording(self): 
        """
        Stops the audio recording process.
        """
        self.recording = False 
        self.cancel_recording = False
        self.update_status("Recording stopped. Processing...", "info")
        self.record_button.config(text="Start Recording (Alt+R)", style="Record.TButton")
        self.cancel_button.config(state=tk.DISABLED, text="Cancel")
        self.update_timer_label("Recording: 00:00", "info")
        self.play_sound('sound.mp3')

    def record_audio(self):
        """
        Handles the audio recording logic using sounddevice.
        """
        try:
            devices = sd.query_devices()
            default_input = sd.default.device[0]
            device_info = sd.query_devices(default_input, 'input')
            self.CHANNELS = min(device_info['max_input_channels'], 2)  # Use max 2 channels
            self.RATE = int(device_info['default_samplerate'])

            with sd.InputStream(samplerate=self.RATE, channels=self.CHANNELS, dtype=self.FORMAT, blocksize=self.CHUNK) as stream:
                frames = []
                while self.recording and not self.cancel_recording:
                    if not self.paused:
                        data, _ = stream.read(self.CHUNK)
                        frames.append(data)
                    else:
                        time.sleep(0.1)

                if self.cancel_recording:
                    self.update_status("Recording cancelled.", "warning")
                    self.play_sound('cancel.wav')
                    return

                # Save recorded frames to a WAV file
                with wave.open(self.WAVE_OUTPUT_FILENAME, 'wb') as wf:
                    wf.setnchannels(self.CHANNELS)
                    wf.setsampwidth(np.dtype(self.FORMAT).itemsize)
                    wf.setframerate(self.RATE)
                    wf.writeframes(b''.join(frames))

                self.update_status("Recording finished. Transcribing...", "info")
                self.transcribe_audio(self.WAVE_OUTPUT_FILENAME)
        except sd.PortAudioError as e:
            logging.error(f"Failed to access audio device: {e}")
            self.update_status("Failed to access audio device. Check your microphone settings.", "error")
        except Exception as e:
            logging.error(f"Error during recording: {e}")
            self.update_status(f"Error during recording: {e}", "error")
        finally:
            self.recording = False
            self.cancel_recording = False
            self.root.after(0, lambda: self.record_button.config(text="Start Recording (Alt+R)", style="Record.TButton"))
            self.root.after(0, lambda: self.cancel_button.config(state=tk.DISABLED))
            self.root.after(0, lambda: self.update_timer_label("Recording: 00:00", "info"))

    def transcribe_audio(self, file_path):
        """
        Handles the transcription of the recorded audio file.
        """
        if not self.API_KEY:
            self.update_status("API key is missing. Please set it in the settings.", "error")
            return

        file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
        self.update_status(f"Processing audio file ({file_size:.2f}MB)...", "info")

        try:
            if file_size > 25:
                logging.info(f"Large file detected ({file_size:.2f}MB). Using file splitting.")
                self.update_status("Large file detected. Splitting for processing...", "info")
                transcription = self.transcribe_large_file(file_path)
            else:
                logging.info(f"Using standard transcription for {file_size:.2f}MB file.")
                self.update_status("Transcribing audio...", "info")
                transcription = self.transcribe_normal(file_path)

            if self.cancel_transcription:
                self.update_status("Transcription cancelled.", "warning")
                self.play_sound('cancel.wav')
                return

            if transcription is None:
                self.update_status("Transcription failed: No result returned", "error")
                return

            logging.info(f"Raw transcription result: {transcription}")  # Log the raw transcription

            try:
                # Check if a profile is selected
                if self.recording_profile_var.get() != "No Profile":
                    profile_name = self.recording_profile_var.get()
                    profile_description = self.settings["profiles"].get(profile_name, "")
                    self.update_status("Processing transcription with profile instructions...", "info")
                    processed_transcription = self.process_with_gpt(transcription, profile_description)
                    if processed_transcription is None:
                        raise ValueError("GPT processing returned None")
                    transcription = processed_transcription
                    self.update_status("Processing with GPT complete.", "info")

                if not isinstance(transcription, str):
                    raise TypeError(f"Expected string, got {type(transcription)}")

                logging.info("Transcription completed successfully")
                self.transcription_queue.put(("insert", transcription))
                self.transcription_history.append(transcription)
                self.update_history_list()
                self.save_transcription_history()
                pyperclip.copy(transcription)
                self.update_status("Transcription complete and copied to clipboard.", "info")
                
                # Check if auto-read is enabled and read the transcription
                if self.auto_read_var.get():
                    self.root.after(500, lambda: self.read_aloud())  # Small delay to ensure text is updated
                else:
                    self.play_sound('done.mp3')

            except Exception as e:
                logging.error(f"Error processing transcription: {e}")
                self.update_status(f"Error processing transcription: {e}", "error")
                # If there's an error in processing, still save the original transcription
                self.transcription_queue.put(("insert", transcription))
                self.transcription_history.append(transcription)
                self.update_history_list()
                self.save_transcription_history()
        except requests.exceptions.Timeout:
            logging.error("API request timed out.")
            self.update_status("API request timed out. Please try again later.", "error")
        except requests.exceptions.ConnectionError:
            logging.error("Failed to connect to the API.")
            self.update_status("Failed to connect to the API. Check your internet connection.", "error")
        except Exception as e:
            logging.error(f"An unexpected error occurred during transcription: {e}")
            self.update_status(f"An unexpected error occurred: {e}", "error")
        finally:
            self.cancel_button.config(state=tk.DISABLED, text="Cancel")
            self.progress.pack_forget()  # Hide progress bar after transcription

    def process_with_gpt(self, transcription, profile_description):
        """
        Processes the transcription text using GPT based on the selected profile.
        """
        if not self.settings.get('api_key'):
            self.update_status("API key is missing. Please set it in the settings.", "error")
            return transcription

        try:
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.API_KEY}",
                "Content-Type": "application/json"
            }
            model = self.settings.get("gpt_model", "gpt-4")

            # Prepare the messages array
            messages = [
                {"role": "system", "content": profile_description}
            ]

            # If send history is enabled and there's history, include it
            if self.send_history_var.get() and self.transcription_history:
                # Format the entire conversation history
                formatted_history = []
                for i, hist_entry in enumerate(self.transcription_history, 1):
                    formatted_history.append(f"Entry {i}:\n{hist_entry}")
                
                history_text = "\n\n".join(formatted_history)
                
                # Create a comprehensive prompt with the entire history
                prompt = (
                    "Here is the complete conversation history:\n\n"
                    f"{history_text}\n\n"
                    "--- New transcription to process ---\n\n"
                    f"{transcription}\n\n"
                    "Please process this new transcription while considering the entire conversation history above."
                )
                
                messages.append({
                    "role": "user",
                    "content": prompt
                })
            else:
                messages.append({
                    "role": "user",
                    "content": transcription
                })

            data = {
                "model": model,
                "messages": messages
            }
            
            response = requests.post(url, headers=headers, json=data, timeout=60)
            if response.status_code == 200:
                result = response.json()
                if result and 'choices' in result and len(result['choices']) > 0:
                    modified_text = result['choices'][0]['message']['content']
                    return modified_text
                else:
                    raise Exception("Invalid API response")
            else:
                raise Exception(f"GPT API call failed with status {response.status_code}: {response.text}")
        except Exception as e:
            logging.error(f"An error occurred during GPT processing: {e}")
            self.update_status(f"An error occurred during GPT processing: {e}", "error")
            return transcription  # Return original transcription if error occurs

    def transcribe_large_file(self, file_path):
        """
        Splits large audio files into chunks and transcribes each chunk individually.
        """
        self.cancel_transcription = False
        chunk_duration = 60 * 1000  # 60 seconds in milliseconds
        audio = AudioSegment.from_file(file_path)
        chunks = [audio[i:i + chunk_duration] for i in range(0, len(audio), chunk_duration)]

        transcriptions = []
        for i, chunk in enumerate(chunks):
            if self.cancel_transcription:
                raise Exception("Transcription cancelled by user")
            self.update_status(f"Transcribing chunk {i + 1} of {len(chunks)}...", "info")
            transcription = f"Transcribing chunk {i + 1}..."
            self.transcription_queue.put(("insert", transcription))
            self.progress['value'] = ((i + 1) / len(chunks)) * 100
            self.progress.pack(fill=tk.X, pady=5)
            chunk_file = f"temp_chunk_{i}.wav"
            chunk.export(chunk_file, format="wav")
            try:
                chunk_transcription = self.transcribe_normal(chunk_file)
                transcriptions.append(chunk_transcription)
            finally:
                if os.path.exists(chunk_file):
                    os.remove(chunk_file)

        return " ".join(transcriptions)

    def transcribe_normal(self, file_path):
        """
        Sends the audio file to the transcription API and returns the transcribed text.
        """
        self.cancel_transcription = False
        url = "https://api.openai.com/v1/audio/transcriptions"
        headers = {"Authorization": f"Bearer {self.API_KEY}"}
        self.FORMAT = 'int16'
        self.CHANNELS = 2
        self.RATE = int(self.settings.get('sample_rate', 44100))
        self.CHUNK = 1024
        self.WAVE_OUTPUT_FILENAME = "output.wav"

        self.update_status("Preparing audio file for upload...", "info")
        with open(file_path, 'rb') as f:
            file_data = f.read()

        self.update_status("Uploading audio file to API...", "info")
        try:
            response = requests.post(
                url,
                headers=headers,
                files={'file': ('audio.wav', file_data)},
                data={'model': self.settings.get("transcription_model", "whisper-1")},
                timeout=300  # Extended timeout for large files
            )
        except requests.exceptions.RequestException as e:
            logging.error(f"API request failed: {e}")
            self.update_status(f"API request failed: {e}", "error")
            return None

        if self.cancel_transcription:
            raise Exception("Transcription cancelled by user")

        self.update_status("Processing transcription...", "info")
        logging.info(f"API Response: {response.text}")

        if response.status_code == 200:
            result = response.json()
            if result and 'text' in result:
                return result['text']
            else:
                logging.error(f"Invalid API response: {result}")
                return f"Error: Invalid API response - {result}"
        else:
            error_message = f"API call failed with status {response.status_code}: {response.text}"
            logging.error(error_message)
            return f"Error: {error_message}"

    def select_audio_file(self):
        """
        Opens a file dialog for the user to select an audio file for transcription.
        """
        file_path = filedialog.askopenfilename(
            filetypes=[("Audio files", "*.wav;*.mp3;*.ogg;*.flac")]
        )
        if file_path:
            self.update_status("Transcribing audio file...", "info")
            threading.Thread(target=self.transcribe_audio, args=(file_path,), daemon=True).start()

    def process_transcription_queue(self):
        """
        Processes items in the transcription queue and updates the UI accordingly.
        """
        try:
            while not self.transcription_queue.empty():
                action, data = self.transcription_queue.get_nowait()
                if action == "insert":
                    self.recording_result_text.config(state='normal')
                    self.recording_result_text.insert(tk.END, data + "\n\n")
                    self.recording_result_text.config(state='disabled')
                elif action == "history":
                    self.transcription_history.append(data)
                    self.update_history_list()
                    self.save_transcription_history()  # Save after adding new transcription
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.process_transcription_queue)

    def cancel_transcription_action(self):
        """
        Cancels the ongoing transcription or recording.
        """
        # Schedule the cancellation action to be executed in the main thread
        self.root.after(0, self._cancel_transcription_action)

    def _cancel_transcription_action(self):
        """
        Internal method to handle the cancellation of transcription or recording.
        """
        # Check if the app is currently recording
        if self.recording:
            # Set the cancel flag to True
            self.cancel_recording = True
            # Update the status label to indicate that recording is being cancelled
            self.update_status("Cancelling recording...", "warning")
        else:
            # Set the cancel flag to True
            self.cancel_transcription = True
            # Update the status label to indicate that transcription is being cancelled
            self.update_status("Cancelling transcription...", "warning")

    def update_timer_label(self, message, status_type="info"):
        """
        Updates the timer label with the current recording time and status.
        """
        timer_colors = {
            "info": "#4CAF50",      # Green
            "recording": "#f44336"  # Red
        }
        self.timer_label.config(
            text=message,
            background=timer_colors.get(status_type, "#4CAF50"),
            foreground="white"
        )

    def update_status(self, message, status_type="info"):
        """
        Updates the status label with the provided message and color based on status type.
        """
        status_colors = {
            "info": "#4CAF50",    # Green
            "error": "#f44336",   # Red
            "warning": "#FFA500", # Orange
            "recording": "#f44336" # Red
        }
        self.status_label.config(
            text=message,
            background=status_colors.get(status_type, "#4CAF50"),
            foreground="white"
        )

    def transcribe_audio_threaded(self, file_path):
        """
        Initiates the transcription process in a separate thread.
        """
        threading.Thread(target=self.transcribe_audio, args=(file_path,), daemon=True).start()

    def update_timer(self):
        """
        Updates the recording timer every second.
        """
        if self.recording:
            elapsed_time = time.time() - self.start_time
            minutes, seconds = divmod(int(elapsed_time), 60)
            self.update_timer_label(f"Recording: {minutes:02}:{seconds:02}", "recording")
            self.root.after(1000, self.update_timer)
        else:
            self.update_timer_label("Recording: 00:00", "info")

    def clear_textbox(self, textbox):
        """
        Clears the specified textbox.
        """
        textbox.config(state='normal')
        textbox.delete(1.0, tk.END)
        textbox.config(state='disabled')

    def update_history_list(self, filtered_history=None):
        """
        Updates the transcription history listbox with the provided history data.
        """
        self.history_listbox.delete(0, tk.END)
        history_to_display = filtered_history if filtered_history is not None else self.transcription_history
        for item in history_to_display:
            display_text = item[:50] + "..." if len(item) > 50 else item
            self.history_listbox.insert(tk.END, display_text)

    def on_search(self, event):
        """
        Filters the transcription history based on the search query.
        """
        query = self.search_entry.get().lower()
        filtered_history = [item for item in self.transcription_history if query in item.lower()]
        self.update_history_list(filtered_history)

    def on_history_select(self, event):
        """
        Displays the selected transcription from the history in the history transcription textbox.
        """
        selected_index = self.history_listbox.curselection()
        if selected_index:
            selected_text = self.transcription_history[selected_index[0]]
            self.display_history_text(selected_text)

    def display_history_text(self, text):
        """
        Displays the provided text in the history transcription textbox.
        """
        self.clear_textbox(self.history_result_text)
        self.history_result_text.config(state='normal')
        self.history_result_text.insert(tk.END, text)
        self.history_result_text.config(state='disabled')

    def clear_selected_history(self):
        """
        Clears the selected transcription from the history.
        """
        selected_index = self.history_listbox.curselection()
        if selected_index:
            confirm = messagebox.askyesno("Confirm", "Clear selected history?")
            if confirm:
                self.transcription_history.pop(selected_index[0])
                self.update_history_list()
                self.clear_textbox(self.history_result_text)
                self.search_entry.delete(0, tk.END)  # Clear search bar
                self.save_transcription_history()  # Save after removing transcription

    def clear_all_history(self):
        """
        Clears all transcriptions from the history.
        """
        confirm = messagebox.askyesno("Confirm", "Clear all history?")
        if confirm:
            self.transcription_history.clear()
            self.update_history_list()
            self.clear_textbox(self.history_result_text)
            self.search_entry.delete(0, tk.END)  # Clear search bar
            self.save_transcription_history()  # Save the empty state

    def browse_directory(self, entry_widget):
        """
        Opens a directory selection dialog and updates the provided entry widget with the selected path.
        """
        directory = filedialog.askdirectory()
        if directory:
            entry_widget.delete(0, tk.END)
            entry_widget.insert(0, directory)
            self.settings["save_location"] = directory

    def toggle_dark_mode(self):
        """
        Toggles between light and dark modes for the application.
        """
        if self.settings.get("dark_mode", False):
            self.style.theme_use('clam')
            self.settings["dark_mode"] = False
        else:
            # For a more comprehensive dark mode, you might need to customize styles further or use a dedicated theme library
            self.style.theme_use('alt')
            self.settings["dark_mode"] = True

    def export_transcription(self):
        """
        Exports the transcription history to a file in the chosen format.
        """
        if not self.transcription_history:
            messagebox.showwarning("Warning", "No transcriptions to export.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[
                ("Text files", "*.txt"),
                ("Word documents", "*.docx"),
                ("PDF files", "*.pdf")
            ]
        )
        if file_path:
            ext = os.path.splitext(file_path)[1].lower()
            try:
                if ext == ".txt":
                    with open(file_path, "w", encoding='utf-8') as file:
                        for item in self.transcription_history:
                            file.write(item + "\n\n")
                elif ext == ".docx":
                    doc = Document()
                    for item in self.transcription_history:
                        doc.add_paragraph(item)
                    doc.save(file_path)
                elif ext == ".pdf":
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_auto_page_break(auto=True, margin=15)
                    pdf.set_font("Arial", size=12)
                    for item in self.transcription_history:
                        # Split text into lines to prevent overflow
                        for line in item.split('\n'):
                            pdf.multi_cell(0, 10, line)
                        pdf.ln()
                    pdf.output(file_path)
                self.update_status(f"Transcriptions exported to {file_path}", "info")
            except Exception as e:
                logging.error(f"Error exporting transcriptions: {e}")
                self.update_status(f"Error exporting transcriptions: {e}", "error")

    def load_transcription_history(self):
        """
        Loads the transcription history from the history file.
        """
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
                if isinstance(history, list):
                    self.transcription_history = history
                    logging.info("Transcription history loaded successfully.")
                else:
                    logging.error("Invalid transcription history format. Starting with empty history.")
                    self.transcription_history = []
            except json.JSONDecodeError:
                logging.error("Error decoding transcription history file. Starting with empty history.")
                self.transcription_history = []
            except Exception as e:
                logging.error(f"Error loading transcription history: {e}")
                self.transcription_history = []
        else:
            logging.info("No transcription history file found. Starting with empty history.")
            self.transcription_history = []

    def load_previous_transcriptions(self):
        """
        Loads previous transcriptions from the history file and updates the history list.
        """
        self.load_transcription_history()
        self.update_history_list()
        messagebox.showinfo("Load Complete", "Previous transcriptions have been loaded successfully.")

    def save_transcription_history(self):
        """
        Saves the transcription history to the history file safely using a temporary file.
        """
        try:
            temp_file = f"{self.history_file}.tmp"
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(self.transcription_history, f, ensure_ascii=False, indent=2)
            shutil.move(temp_file, self.history_file)
            logging.info("Transcription history saved successfully.")
        except Exception as e:
            logging.error(f"Error saving transcription history: {e}")
            messagebox.showerror("Save Error", f"Failed to save transcription history: {e}")

    def on_closing(self):
        """
        Handles the application closing event, ensuring all resources are cleaned up properly.
        """
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            if hasattr(self, 'tray_icon') and self.tray_icon:
                self.tray_icon.stop()
            if hasattr(self, 'hotkey_listener'):
                self.hotkey_listener.stop()
            self.root.destroy()

    def open_settings(self):
        """
        Opens the settings window where users can configure application preferences.
        """
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Settings")
        settings_window.geometry("600x600")
        settings_window.grab_set()  # Make the settings window modal

        notebook = ttk.Notebook(settings_window)
        notebook.pack(expand=True, fill="both")

        general_frame = ttk.Frame(notebook)
        audio_frame = ttk.Frame(notebook)
        api_frame = ttk.Frame(notebook)

        notebook.add(general_frame, text="General")
        notebook.add(audio_frame, text="Audio")
        notebook.add(api_frame, text="API")

        # General Settings
        ttk.Label(general_frame, text="Default Save Location:").pack(pady=5)
        save_location_entry = ttk.Entry(general_frame, width=40)
        save_location_entry.pack(pady=5)
        save_location_entry.insert(0, self.settings.get("save_location", ""))
        ttk.Button(general_frame, text="Browse", command=lambda: self.browse_directory(save_location_entry)).pack(pady=5)

        dark_mode_var = tk.BooleanVar(value=self.settings.get("dark_mode", False))
        ttk.Checkbutton(
            general_frame,
            text="Enable Dark Mode",
            variable=dark_mode_var,
            command=self.toggle_dark_mode
        ).pack(pady=5)

        # Audio Settings
        bitrate_values = ["64 kbps", "128 kbps", "256 kbps"]
        ttk.Label(audio_frame, text="Bitrate:").pack(pady=5)
        bitrate_combobox = ttk.Combobox(audio_frame, values=bitrate_values, state="readonly")
        bitrate_combobox.set(self.settings.get("bitrate", bitrate_values[1]))
        bitrate_combobox.pack(pady=5)

        sample_rate_values = [22050, 44100, 48000]
        ttk.Label(audio_frame, text="Sample Rate:").pack(pady=5)
        sample_rate_combobox = ttk.Combobox(audio_frame, values=sample_rate_values, state="readonly")
        sample_rate_combobox.set(self.settings.get("sample_rate", sample_rate_values[1]))
        sample_rate_combobox.pack(pady=5)

        # Volume Slider
        ttk.Label(audio_frame, text="Volume:").pack(pady=5)
        volume_slider = ttk.Scale(
            audio_frame,
            from_=0,
            to=1,
            orient=tk.HORIZONTAL,
            length=200,
            command=lambda val: self.update_volume(float(val))
        )
        volume_slider.set(self.settings.get("volume", 1.0))
        volume_slider.pack(pady=5)

        # API Settings
        ttk.Label(api_frame, text="API Key:").pack(pady=5)
        api_key_entry = ttk.Entry(api_frame, width=40, show="*")
        api_key_entry.pack(pady=5)
        api_key_entry.insert(0, self.API_KEY or "")
        api_key_entry.configure(state='normal')  # Enable direct editing

        def update_api_key():
            """
            Updates the API key in keyring based on user input.
            """
            new_api_key = api_key_entry.get().strip()
            if new_api_key:
                keyring.set_password("whisper_api", "api_key", new_api_key)
                self.settings["api_key"] = new_api_key
                self.API_KEY = new_api_key
                logging.info("API key updated and saved securely in keyring.")
                messagebox.showinfo("Success", "API Key updated successfully.")
            else:
                messagebox.showerror("Error", "API Key cannot be empty.")

        ttk.Button(api_frame, text="Update API Key", command=update_api_key).pack(pady=5)

        transcription_models = ["whisper-1"]
        ttk.Label(api_frame, text="Transcription Model:").pack(pady=5)
        model_combobox = ttk.Combobox(api_frame, values=transcription_models, state="readonly")
        model_combobox.set(self.settings.get("transcription_model", transcription_models[0]))
        model_combobox.pack(pady=5)

        # Profile Model Settings
        profile_models = ["chatgpt-4o-latest", "gpt-4o-mini", "gpt-4o-2024-08-06", "gpt-4o"]
        ttk.Label(api_frame, text="Profile Model:").pack(pady=5)
        profile_model_combobox = ttk.Combobox(api_frame, values=profile_models, state="readonly")
        profile_model_combobox.set(self.settings.get("gpt_model", profile_models[0]))
        profile_model_combobox.pack(pady=5)

        def save_settings():
            """
            Saves the settings from the settings window to the configuration and keyring.
            """
            self.settings["save_location"] = save_location_entry.get()
            self.settings["dark_mode"] = dark_mode_var.get()
            self.settings["bitrate"] = bitrate_combobox.get()
            self.settings["sample_rate"] = int(sample_rate_combobox.get())
            self.settings["transcription_model"] = model_combobox.get()
            self.settings["gpt_model"] = profile_model_combobox.get()
            self.settings["volume"] = volume_slider.get()

            # Update API key
            new_api_key = api_key_entry.get().strip()
            if new_api_key:
                self.settings["api_key"] = new_api_key
                self.API_KEY = new_api_key
                keyring.set_password("whisper_api", "api_key", new_api_key)

            self.save_config()
            self.update_volume(self.settings["volume"])
            settings_window.destroy()

        ttk.Button(settings_window, text="Save", command=save_settings).pack(pady=10)

    def initialize_global_hotkeys(self):
        """
        Initializes global hotkey listeners for Alt+R and Alt+C using pynput.
        """
        # Define the hotkeys and their corresponding callbacks
        hotkeys = {
            '<alt>+r': self.toggle_recording,
            '<alt>+c': self.cancel_transcription_action
        }

        # Define a listener class to handle hotkeys
        class HotkeyListener(threading.Thread):
            def __init__(self, hotkeys):
                super().__init__()
                self.hotkeys = hotkeys
                self.listener = keyboard.GlobalHotKeys(self.hotkeys)
                self.daemon = True

            def run(self):
                self.listener.start()

            def stop(self):
                self.listener.stop()

        # Start the global hotkey listener
        self.hotkey_listener = HotkeyListener(hotkeys)
        self.hotkey_listener.start()

    def run(self):
        """
        Runs the main application loop with global exception handling.
        """
        try:
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
            self.root.mainloop()
        except Exception as e:
            logging.error(f"Unhandled exception: {e}")
            messagebox.showerror("Error", f"An unexpected error occurred: {e}")

    def toggle_history_visibility(self):
        """
        Toggles the visibility of the transcription history panel.
        """
        if self.history_frame.winfo_viewable():
            self.history_frame.pack_forget()
            self.clear_history_button.pack_forget()
            self.toggle_history_button.config(text="Show History")
        else:
            self.history_frame.pack(fill=tk.BOTH, expand=True)
            self.clear_history_button.pack(side=tk.LEFT, padx=5, pady=10)
            self.toggle_history_button.config(text="Hide History")

    def open_help(self):
        """
        Opens the help window displaying the user guide.
        """
        help_window = tk.Toplevel(self.root)
        help_window.title("User Guide")
        help_window.geometry("800x500")  # Increased width from 600 to 800
        help_window.grab_set()  # Make the help window modal

        help_text = scrolledtext.ScrolledText(help_window, wrap=tk.WORD, state='normal')
        help_text.pack(expand=True, fill="both")

        help_file_path = os.path.join(self.get_executable_dir(), 'help_text.txt')
        try:
            with open(help_file_path, 'r', encoding='utf-8') as help_file:
                help_content = help_file.read()
        except FileNotFoundError:
            help_content = "Help file not found. Please check the application installation."
        except Exception as e:
            help_content = f"Error loading help content: {str(e)}"

        # Configure tags for HTML-like formatting
        help_text.tag_configure("h1", font=("Helvetica", 16, "bold"))
        help_text.tag_configure("h2", font=("Helvetica", 14, "bold"))
        help_text.tag_configure("h3", font=("Helvetica", 12, "bold"))
        help_text.tag_configure("strong", font=("Helvetica", 10, "bold"))
        help_text.tag_configure("em", font=("Helvetica", 10, "italic"))

        # Parse and insert formatted content
        self.insert_formatted_text(help_text, help_content)
        help_text.config(state='disabled')

    def insert_formatted_text(self, text_widget, content):
        """
        Parses the HTML-like tags and inserts formatted text into the text widget.
        """
        tag_stack = []
        i = 0
        while i < len(content):
            if content[i] == '<':
                end = content.find('>', i)
                if end != -1:
                    tag = content[i+1:end]
                    if tag.startswith('/'):
                        if tag[1:] in tag_stack:
                            tag_stack.remove(tag[1:])
                    else:
                        tag_stack.append(tag)
                    i = end + 1

                    # Handle special tags
                    if tag in ['table', '/table', 'tr', '/tr', 'th', '/th', 'td', '/td']:
                        text_widget.insert(tk.END, '\n')
                    elif tag == 'li':
                        text_widget.insert(tk.END, '• ')
                    elif tag == '/li':
                        text_widget.insert(tk.END, '\n')
                    elif tag in ['summary', '/summary', 'details', '/details']:
                        text_widget.insert(tk.END, '\n')

                    continue

            # Handle emoji and other special characters
            text_widget.insert(tk.END, content[i], tuple(tag_stack))
            i += 1

        # Configure additional tags
        text_widget.tag_configure("table", background="#f0f0f0", borderwidth=1, relief="solid")
        text_widget.tag_configure("tr", background="#ffffff")
        text_widget.tag_configure("th", font=("Helvetica", 10, "bold"))
        text_widget.tag_configure("td")
        text_widget.tag_configure("kbd", font=("Courier", 9), background="#e0e0e0", relief="raised", borderwidth=1)
        text_widget.tag_configure("a", foreground="blue", underline=1)
        text_widget.tag_bind("a", "<Enter>", lambda e: text_widget.config(cursor="hand2"))
        text_widget.tag_bind("a", "<Leave>", lambda e: text_widget.config(cursor=""))

    def on_drop(self, event):
        """
        Handles the drag-and-drop event for audio files.
        """
        file_path = event.data
        file_path = file_path.strip("{}").strip('"')
        _, file_extension = os.path.splitext(file_path.lower())

        if file_extension in ('.wav', '.mp3', '.ogg', '.flac'):
            self.update_status(f"Transcribing dropped audio file: {os.path.basename(file_path)}...", "info")
            self.transcribe_audio_threaded(file_path)
        else:
            self.update_status(
                f"Unsupported file type: {file_extension}. Please drop a .wav, .mp3, .ogg, or .flac file.",
                "error"
            )

    def play_sound(self, sound_file):
        """
        Plays the specified sound file using pygame.
        """
        sound_path = self.resource_path(sound_file)
        logging.info(f"Attempting to play sound from: {sound_path}")
        if os.path.exists(sound_path):
            try:
                pygame.mixer.music.load(sound_path)
                pygame.mixer.music.set_volume(self.settings.get("volume", 1.0))
                pygame.mixer.music.play()
                logging.info("Sound played successfully.")
            except pygame.error as e:
                logging.error(f"Failed to play sound: {e}")
        else:
            logging.error(f"Sound file not found: {sound_path}")

    def update_volume(self, volume):
        """
        Updates the volume for sound playback.
        """
        pygame.mixer.music.set_volume(volume)
        self.settings["volume"] = volume

    def initialize_tray_icon(self):
        """
        Initializes the system tray icon using pystray.
        """
        image = self.create_image(64, 64, "black", "white")
        self.tray_icon = pystray.Icon("name", image, "Audio Transcriber", self.create_tray_menu())
        tray_thread = threading.Thread(target=self.tray_icon.run, daemon=True)
        tray_thread.start()

    def create_tray_menu(self):
        """
        Creates the context menu for the system tray icon.
        """
        return pystray.Menu(
            pystray.MenuItem("Open", self.on_tray_click),
            pystray.MenuItem("Exit", self.on_tray_exit)
        )

    def create_image(self, width, height, color1, color2):
        """
        Creates an image for the system tray icon.
        """
        image = Image.new('RGB', (width, height), color1)
        dc = ImageDraw.Draw(image)
        dc.rectangle([(width // 2, 0), (width, height)], fill=color2)
        return image

    def on_tray_click(self, icon, item):
        """
        Restores the application window when the tray icon is clicked.
        """
        self.root.deiconify()

    def on_tray_exit(self, icon, item):
        """
        Exits the application when the 'Exit' option is selected from the tray menu.
        """
        self.tray_icon.stop()
        self.on_closing()

    def check_internet_connection(self, timeout=5):
        """
        Checks if the application has an active internet connection.
        """
        try:
            requests.get("https://www.google.com", timeout=timeout)
            return True
        except requests.ConnectionError:
            return False

    def check_api_key(self):
        """
        Validates the presence of the API key.
        """
        if not self.API_KEY:
            messagebox.showwarning("API Key Missing", "Please set your API key in the settings menu.")
            self.prompt_for_api_key()
            return False
        return True

    def system_check(self):
        """
        Performs initial system checks to ensure all dependencies and requirements are met.
        """
        checks = {
            "Internet Connection": self.check_internet_connection(),
            "Audio Devices": len(sd.query_devices()) > 0,
            "API Key": bool(self.settings.get("api_key")),
            "FFmpeg": shutil.which('ffmpeg') is not None
        }

        failed_checks = [check for check, passed in checks.items() if not passed]

        if failed_checks:
            message = "The following system checks failed:\n\n" + "\n".join(failed_checks)
            message += "\n\nSome features may not work correctly."
            messagebox.showwarning("System Check", message)

    def create_backup(self):
        """
        Creates a backup of the transcription history file.
        """
        try:
            if os.path.exists(self.history_file):
                shutil.copy2(self.history_file, self.backup_history_file)
                logging.info("Backup of transcription history created successfully.")
        except Exception as e:
            logging.error(f"Error creating backup of transcription history: {e}")

    def check_history_size(self):
        """
        Checks if the history size might exceed token limits and warns the user.
        Returns True if the user wants to proceed, False otherwise.
        """
        if len(self.transcription_history) > 10:  # Adjust this threshold as needed
            return messagebox.askyesno(
                "Large History Warning",
                "You are about to send a large conversation history which may exceed token limits "
                "or increase API costs. Do you want to proceed?"
            )
        return True

def main():
    root = tkinterdnd2.TkinterDnD.Tk()
    app = AudioTranscriberApp(root)
    app.run()

if __name__ == "__main__":
    main()
