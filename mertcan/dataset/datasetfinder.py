import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import json
import sqlite3
from datetime import datetime
from pathlib import Path
import re
from collections import defaultdict

class MRIDatasetManager:
    def __init__(self, root):
        self.root = root
        self.root.title("MRI Dataset Manager")
        self.root.geometry("1200x800")
        
        # Initialize database
        self.init_database()
        
        # Create main interface
        self.create_widgets()
        
        # Load existing data
        self.refresh_tree()
    
    def init_database(self):
        """Initialize SQLite database optimized for MRI datasets"""
        self.conn = sqlite3.connect('mri_dataset_manager.db')
        self.cursor = self.conn.cursor()
        
        # Create tables
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS mri_datasets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                root_path TEXT NOT NULL UNIQUE,
                dataset_type TEXT,
                total_size INTEGER,
                total_files INTEGER,
                centers_count INTEGER,
                patients_count INTEGER,
                created_date TEXT,
                last_modified TEXT,
                description TEXT,
                tags TEXT,
                notes TEXT
            )
        ''')
        
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS mri_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_id INTEGER,
                filename TEXT NOT NULL,
                filepath TEXT NOT NULL,
                file_type TEXT,
                size INTEGER,
                center TEXT,
                scanner TEXT,
                patient TEXT,
                sequence_type TEXT,
                data_category TEXT,
                relative_path TEXT,
                created_date TEXT,
                modified_date TEXT,
                FOREIGN KEY (dataset_id) REFERENCES mri_datasets (id)
            )
        ''')
        
        self.conn.commit()
    
    def create_widgets(self):
        """Create the main GUI elements"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Toolbar
        toolbar = ttk.Frame(main_frame)
        toolbar.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Button(toolbar, text="Add MRI Dataset", 
                  command=self.add_mri_dataset).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(toolbar, text="Refresh", 
                  command=self.refresh_tree).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(toolbar, text="Find Duplicates", 
                  command=self.find_duplicates).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(toolbar, text="Advanced Search", 
                  command=self.advanced_search).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(toolbar, text="Export Report", 
                  command=self.export_report).pack(side=tk.LEFT, padx=(0, 5))
        
        # Left panel - Dataset tree
        left_frame = ttk.Frame(main_frame)
        left_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        left_frame.columnconfigure(0, weight=1)
        left_frame.rowconfigure(0, weight=1)
        
        # Tree view with MRI-specific columns
        tree_frame = ttk.Frame(left_frame)
        tree_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        tree_frame.columnconfigure(0, weight=1)
        tree_frame.rowconfigure(0, weight=1)
        
        self.tree = ttk.Treeview(tree_frame, columns=('type', 'size', 'centers', 'patients', 'files'), show='tree headings')
        self.tree.heading('#0', text='Dataset Name')
        self.tree.heading('type', text='Type')
        self.tree.heading('size', text='Size')
        self.tree.heading('centers', text='Centers')
        self.tree.heading('patients', text='Patients')
        self.tree.heading('files', text='Files')
        
        self.tree.column('#0', width=200)
        self.tree.column('type', width=100)
        self.tree.column('size', width=100)
        self.tree.column('centers', width=80)
        self.tree.column('patients', width=80)
        self.tree.column('files', width=80)
        
        # Scrollbars for tree
        tree_scrolly = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        tree_scrollx = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(yscrollcommand=tree_scrolly.set, xscrollcommand=tree_scrollx.set)
        
        self.tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        tree_scrolly.grid(row=0, column=1, sticky=(tk.N, tk.S))
        tree_scrollx.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        # Bind tree selection
        self.tree.bind('<<TreeviewSelect>>', self.on_tree_select)
        
        # Right panel - Details with tabs
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(0, weight=1)
        
        # Notebook for tabs
        self.notebook = ttk.Notebook(right_frame)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Tab 1: Dataset Details
        self.create_details_tab()
        
        # Tab 2: File Browser
        self.create_browser_tab()
        
        # Tab 3: Statistics
        self.create_stats_tab()
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Initialize variables
        self.current_dataset_id = None
    
    def create_details_tab(self):
        """Create the dataset details tab"""
        details_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(details_frame, text="Dataset Details")
        
        details_frame.columnconfigure(0, weight=1)
        details_frame.rowconfigure(4, weight=1)
        
        # Basic info
        info_frame = ttk.LabelFrame(details_frame, text="Basic Information", padding="10")
        info_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        info_frame.columnconfigure(1, weight=1)
        
        ttk.Label(info_frame, text="Dataset Name:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.name_var = tk.StringVar()
        ttk.Entry(info_frame, textvariable=self.name_var, width=50).grid(row=0, column=1, sticky=(tk.W, tk.E), pady=2)
        
        ttk.Label(info_frame, text="Root Path:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.path_var = tk.StringVar()
        ttk.Entry(info_frame, textvariable=self.path_var, width=50, state='readonly').grid(row=1, column=1, sticky=(tk.W, tk.E), pady=2)
        
        ttk.Label(info_frame, text="Dataset Type:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.type_var = tk.StringVar()
        type_combo = ttk.Combobox(info_frame, textvariable=self.type_var, width=47)
        type_combo['values'] = ('BlackBlood', 'Cine', 'Mixed', 'Other')
        type_combo.grid(row=2, column=1, sticky=(tk.W, tk.E), pady=2)
        
        ttk.Label(info_frame, text="Tags:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.tags_var = tk.StringVar()
        ttk.Entry(info_frame, textvariable=self.tags_var, width=50).grid(row=3, column=1, sticky=(tk.W, tk.E), pady=2)
        
        # Description
        desc_frame = ttk.LabelFrame(details_frame, text="Description", padding="10")
        desc_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        desc_frame.columnconfigure(0, weight=1)
        desc_frame.rowconfigure(0, weight=1)
        
        self.description_text = tk.Text(desc_frame, height=4, width=50)
        desc_scroll = ttk.Scrollbar(desc_frame, orient=tk.VERTICAL, command=self.description_text.yview)
        self.description_text.configure(yscrollcommand=desc_scroll.set)
        
        self.description_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        desc_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Notes
        notes_frame = ttk.LabelFrame(details_frame, text="Notes", padding="10")
        notes_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        notes_frame.columnconfigure(0, weight=1)
        notes_frame.rowconfigure(0, weight=1)
        
        self.notes_text = tk.Text(notes_frame, height=4, width=50)
        notes_scroll = ttk.Scrollbar(notes_frame, orient=tk.VERTICAL, command=self.notes_text.yview)
        self.notes_text.configure(yscrollcommand=notes_scroll.set)
        
        self.notes_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        notes_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Save button
        ttk.Button(details_frame, text="Save Changes", command=self.save_dataset).grid(row=3, column=0, pady=10)
    
    def create_browser_tab(self):
        """Create the file browser tab"""
        browser_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(browser_frame, text="File Browser")
        
        browser_frame.columnconfigure(0, weight=1)
        browser_frame.rowconfigure(1, weight=1)
        
        # Filter frame
        filter_frame = ttk.Frame(browser_frame)
        filter_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(filter_frame, text="Filter:").pack(side=tk.LEFT, padx=(0, 5))
        self.filter_var = tk.StringVar()
        filter_entry = ttk.Entry(filter_frame, textvariable=self.filter_var, width=30)
        filter_entry.pack(side=tk.LEFT, padx=(0, 5))
        filter_entry.bind('<KeyRelease>', self.filter_files)
        
        ttk.Button(filter_frame, text="Clear", command=self.clear_filter).pack(side=tk.LEFT, padx=(0, 10))
        
        # File type filter
        ttk.Label(filter_frame, text="Type:").pack(side=tk.LEFT, padx=(0, 5))
        self.filetype_var = tk.StringVar()
        filetype_combo = ttk.Combobox(filter_frame, textvariable=self.filetype_var, width=15)
        filetype_combo['values'] = ('All', '.mat', '.csv', '.png', '.nii')
        filetype_combo.set('All')
        filetype_combo.pack(side=tk.LEFT, padx=(0, 5))
        filetype_combo.bind('<<ComboboxSelected>>', self.filter_files)
        
        # File treeview
        file_tree_frame = ttk.Frame(browser_frame)
        file_tree_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        file_tree_frame.columnconfigure(0, weight=1)
        file_tree_frame.rowconfigure(0, weight=1)
        
        self.file_tree = ttk.Treeview(file_tree_frame, columns=('size', 'center', 'patient', 'type', 'category'), show='tree headings')
        self.file_tree.heading('#0', text='File Name')
        self.file_tree.heading('size', text='Size')
        self.file_tree.heading('center', text='Center')
        self.file_tree.heading('patient', text='Patient')
        self.file_tree.heading('type', text='Type')
        self.file_tree.heading('category', text='Category')
        
        self.file_tree.column('#0', width=250)
        self.file_tree.column('size', width=80)
        self.file_tree.column('center', width=100)
        self.file_tree.column('patient', width=80)
        self.file_tree.column('type', width=60)
        self.file_tree.column('category', width=120)
        
        # Scrollbars for file tree
        file_tree_scrolly = ttk.Scrollbar(file_tree_frame, orient=tk.VERTICAL, command=self.file_tree.yview)
        file_tree_scrollx = ttk.Scrollbar(file_tree_frame, orient=tk.HORIZONTAL, command=self.file_tree.xview)
        self.file_tree.configure(yscrollcommand=file_tree_scrolly.set, xscrollcommand=file_tree_scrollx.set)
        
        self.file_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        file_tree_scrolly.grid(row=0, column=1, sticky=(tk.N, tk.S))
        file_tree_scrollx.grid(row=1, column=0, sticky=(tk.W, tk.E))
    
    def create_stats_tab(self):
        """Create the statistics tab"""
        stats_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(stats_frame, text="Statistics")
        
        stats_frame.columnconfigure(0, weight=1)
        stats_frame.rowconfigure(0, weight=1)
        
        # Statistics text widget
        self.stats_text = tk.Text(stats_frame, height=20, width=60, state='disabled')
        stats_scroll = ttk.Scrollbar(stats_frame, orient=tk.VERTICAL, command=self.stats_text.yview)
        self.stats_text.configure(yscrollcommand=stats_scroll.set)
        
        self.stats_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        stats_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))
    
    def add_mri_dataset(self):
        """Add an MRI dataset by selecting the root folder"""
        folder_path = filedialog.askdirectory(title="Select MRI Dataset Root Folder")
        if folder_path:
            self.scan_mri_dataset(folder_path)
    
    def scan_mri_dataset(self, root_path):
        """Scan MRI dataset and extract structure information"""
        try:
            self.status_var.set("Scanning MRI dataset...")
            self.root.update()
            
            root_path_obj = Path(root_path)
            dataset_name = root_path_obj.name
            
            # Parse the MRI dataset structure
            files_info = []
            total_size = 0
            centers = set()
            patients = set()
            
            # Walk through the directory structure
            for file_path in root_path_obj.rglob('*'):
                if file_path.is_file():
                    # Extract MRI-specific information from path
                    parts = file_path.parts
                    relative_path = file_path.relative_to(root_path_obj)
                    
                    # Parse structure: root/[BlackBlood|Cine]/TrainingSet/[FullSample|ImageShow|Mask_TaskAll]/Center###/Scanner/P###/file
                    center, scanner, patient, sequence_type, data_category = self.parse_mri_path(parts)
                    
                    if center:
                        centers.add(center)
                    if patient:
                        patients.add(patient)
                    
                    file_size = file_path.stat().st_size
                    total_size += file_size
                    
                    files_info.append({
                        'filename': file_path.name,
                        'filepath': str(file_path),
                        'relative_path': str(relative_path),
                        'file_type': file_path.suffix.lower(),
                        'size': file_size,
                        'center': center,
                        'scanner': scanner,
                        'patient': patient,
                        'sequence_type': sequence_type,
                        'data_category': data_category,
                        'created_date': datetime.fromtimestamp(file_path.stat().st_ctime).isoformat(),
                        'modified_date': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                    })
            
            # Determine dataset type
            if 'BlackBlood' in root_path and 'Cine' in root_path:
                dataset_type = 'Mixed'
            elif 'BlackBlood' in root_path:
                dataset_type = 'BlackBlood'
            elif 'Cine' in root_path:
                dataset_type = 'Cine'
            else:
                dataset_type = 'Other'
            
            # Insert dataset into database
            self.cursor.execute('''
                INSERT OR REPLACE INTO mri_datasets 
                (name, root_path, dataset_type, total_size, total_files, centers_count, patients_count, 
                 created_date, last_modified, description, tags, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (dataset_name, str(root_path), dataset_type, total_size, len(files_info), 
                  len(centers), len(patients), datetime.now().isoformat(), 
                  datetime.now().isoformat(), '', '', ''))
            
            dataset_id = self.cursor.lastrowid
            
            # Insert files into database
            for file_info in files_info:
                self.cursor.execute('''
                    INSERT INTO mri_files 
                    (dataset_id, filename, filepath, file_type, size, center, scanner, patient, 
                     sequence_type, data_category, relative_path, created_date, modified_date)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (dataset_id, file_info['filename'], file_info['filepath'], 
                      file_info['file_type'], file_info['size'], file_info['center'], 
                      file_info['scanner'], file_info['patient'], file_info['sequence_type'], 
                      file_info['data_category'], file_info['relative_path'], 
                      file_info['created_date'], file_info['modified_date']))
            
            self.conn.commit()
            self.refresh_tree()
            self.status_var.set(f"Added MRI dataset: {dataset_name} ({len(files_info)} files, {len(centers)} centers, {len(patients)} patients)")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to scan MRI dataset: {str(e)}")
            self.status_var.set("Ready")
    
    def parse_mri_path(self, path_parts):
        """Parse MRI dataset path to extract structure information"""
        center = scanner = patient = sequence_type = data_category = None
        
        # Convert to list for easier processing
        parts = list(path_parts)
        
        # Find center (Center###)
        center_pattern = re.compile(r'Center\d+')
        for part in parts:
            if center_pattern.match(part):
                center = part
                break
        
        # Find scanner (contains scanner info)
        scanner_patterns = [r'Siemens', r'UIH', r'Philips', r'GE']
        for part in parts:
            for pattern in scanner_patterns:
                if pattern in part:
                    scanner = part
                    break
        
        # Find patient (P###)
        patient_pattern = re.compile(r'P\d+')
        for part in parts:
            if patient_pattern.match(part):
                patient = part
                break
        
        # Find sequence type and data category
        for part in parts:
            if part in ['BlackBlood', 'Cine']:
                sequence_type = part
            elif part in ['FullSample', 'ImageShow', 'Mask_TaskAll']:
                data_category = part
        
        return center, scanner, patient, sequence_type, data_category
    
    def refresh_tree(self):
        """Refresh the dataset tree view"""
        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Load datasets from database
        self.cursor.execute('''
            SELECT id, name, dataset_type, total_size, centers_count, patients_count, total_files 
            FROM mri_datasets ORDER BY name
        ''')
        datasets = self.cursor.fetchall()
        
        for dataset in datasets:
            dataset_id, name, dataset_type, total_size, centers_count, patients_count, total_files = dataset
            size_str = self.format_file_size(total_size)
            self.tree.insert('', 'end', values=(dataset_type, size_str, centers_count, patients_count, total_files), 
                           text=name, tags=(dataset_id,))
    
    def on_tree_select(self, event):
        """Handle dataset selection"""
        selection = self.tree.selection()
        if selection:
            item = selection[0]
            tags = self.tree.item(item, 'tags')
            if tags:
                dataset_id = tags[0]
                self.load_dataset_details(dataset_id)
    
    def load_dataset_details(self, dataset_id):
        """Load dataset details into all tabs"""
        self.current_dataset_id = dataset_id
        
        # Load dataset info
        self.cursor.execute('SELECT * FROM mri_datasets WHERE id = ?', (dataset_id,))
        dataset = self.cursor.fetchone()
        
        if dataset:
            # Update details tab
            _, name, root_path, dataset_type, total_size, total_files, centers_count, patients_count, created, modified, description, tags, notes = dataset
            
            self.name_var.set(name)
            self.path_var.set(root_path)
            self.type_var.set(dataset_type or '')
            self.tags_var.set(tags or '')
            
            self.description_text.delete('1.0', tk.END)
            self.description_text.insert('1.0', description or '')
            
            self.notes_text.delete('1.0', tk.END)
            self.notes_text.insert('1.0', notes or '')
            
            # Update file browser
            self.load_files(dataset_id)
            
            # Update statistics
            self.update_statistics(dataset_id)
    
    def load_files(self, dataset_id):
        """Load files into the file browser"""
        # Clear existing files
        for item in self.file_tree.get_children():
            self.file_tree.delete(item)
        
        # Load files from database
        self.cursor.execute('''
            SELECT filename, size, center, patient, file_type, data_category, relative_path
            FROM mri_files WHERE dataset_id = ? ORDER BY relative_path
        ''', (dataset_id,))
        files = self.cursor.fetchall()
        
        for file_info in files:
            filename, size, center, patient, file_type, data_category, relative_path = file_info
            size_str = self.format_file_size(size)
            self.file_tree.insert('', 'end', 
                                values=(size_str, center or '', patient or '', file_type or '', data_category or ''),
                                text=filename)
    
    def filter_files(self, event=None):
        """Filter files based on current filters"""
        if not self.current_dataset_id:
            return
        
        filter_text = self.filter_var.get().lower()
        file_type_filter = self.filetype_var.get()
        
        # Clear existing files
        for item in self.file_tree.get_children():
            self.file_tree.delete(item)
        
        # Build query
        query = '''
            SELECT filename, size, center, patient, file_type, data_category, relative_path
            FROM mri_files WHERE dataset_id = ?
        '''
        params = [self.current_dataset_id]
        
        if filter_text:
            query += ' AND (filename LIKE ? OR relative_path LIKE ?)'
            params.extend([f'%{filter_text}%', f'%{filter_text}%'])
        
        if file_type_filter != 'All':
            query += ' AND file_type = ?'
            params.append(file_type_filter)
        
        query += ' ORDER BY relative_path'
        
        self.cursor.execute(query, params)
        files = self.cursor.fetchall()
        
        for file_info in files:
            filename, size, center, patient, file_type, data_category, relative_path = file_info
            size_str = self.format_file_size(size)
            self.file_tree.insert('', 'end', 
                                values=(size_str, center or '', patient or '', file_type or '', data_category or ''),
                                text=filename)
    
    def clear_filter(self):
        """Clear all filters"""
        self.filter_var.set('')
        self.filetype_var.set('All')
        if self.current_dataset_id:
            self.load_files(self.current_dataset_id)
    
    def find_duplicates(self):
        """Find duplicate file names across the dataset"""
        if not self.current_dataset_id:
            messagebox.showwarning("Warning", "Please select a dataset first")
            return
        
        # Find duplicate filenames
        self.cursor.execute('''
            SELECT filename, COUNT(*) as count, GROUP_CONCAT(relative_path) as paths
            FROM mri_files 
            WHERE dataset_id = ?
            GROUP BY filename
            HAVING count > 1
            ORDER BY count DESC, filename
        ''', (self.current_dataset_id,))
        
        duplicates = self.cursor.fetchall()
        
        # Show results in a new window
        dup_window = tk.Toplevel(self.root)
        dup_window.title("Duplicate Files")
        dup_window.geometry("800x600")
        
        # Create treeview for duplicates
        dup_tree = ttk.Treeview(dup_window, columns=('count', 'paths'), show='tree headings')
        dup_tree.heading('#0', text='Filename')
        dup_tree.heading('count', text='Count')
        dup_tree.heading('paths', text='Locations')
        
        dup_tree.column('#0', width=200)
        dup_tree.column('count', width=80)
        dup_tree.column('paths', width=500)
        
        # Scrollbars
        dup_scrolly = ttk.Scrollbar(dup_window, orient=tk.VERTICAL, command=dup_tree.yview)
        dup_scrollx = ttk.Scrollbar(dup_window, orient=tk.HORIZONTAL, command=dup_tree.xview)
        dup_tree.configure(yscrollcommand=dup_scrolly.set, xscrollcommand=dup_scrollx.set)
        
        dup_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        dup_scrolly.pack(side=tk.RIGHT, fill=tk.Y)
        dup_scrollx.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Populate duplicates
        for filename, count, paths in duplicates:
            dup_tree.insert('', 'end', values=(count, paths.replace(',', '\n')), text=filename)
        
        if not duplicates:
            ttk.Label(dup_window, text="No duplicate files found!").pack(pady=20)
    
    def advanced_search(self):
        """Open advanced search dialog"""
        search_window = tk.Toplevel(self.root)
        search_window.title("Advanced Search")
        search_window.geometry("500x400")
        
        # Search criteria
        search_frame = ttk.LabelFrame(search_window, text="Search Criteria", padding="10")
        search_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Filename
        ttk.Label(search_frame, text="Filename:").grid(row=0, column=0, sticky=tk.W, pady=2)
        filename_var = tk.StringVar()
        ttk.Entry(search_frame, textvariable=filename_var, width=30).grid(row=0, column=1, sticky=(tk.W, tk.E), pady=2)
        
        # Center
        ttk.Label(search_frame, text="Center:").grid(row=1, column=0, sticky=tk.W, pady=2)
        center_var = tk.StringVar()
        ttk.Entry(search_frame, textvariable=center_var, width=30).grid(row=1, column=1, sticky=(tk.W, tk.E), pady=2)
        
        # Patient
        ttk.Label(search_frame, text="Patient:").grid(row=2, column=0, sticky=tk.W, pady=2)
        patient_var = tk.StringVar()
        ttk.Entry(search_frame, textvariable=patient_var, width=30).grid(row=2, column=1, sticky=(tk.W, tk.E), pady=2)
        
        # File type
        ttk.Label(search_frame, text="File Type:").grid(row=3, column=0, sticky=tk.W, pady=2)
        filetype_var = tk.StringVar()
        filetype_combo = ttk.Combobox(search_frame, textvariable=filetype_var, width=27)
        filetype_combo['values'] = ('All', '.mat', '.csv', '.png', '.nii')
        filetype_combo.set('All')
        filetype_combo.grid(row=3, column=1, sticky=(tk.W, tk.E), pady=2)
        
        # Data category
        ttk.Label(search_frame, text="Category:").grid(row=4, column=0, sticky=tk.W, pady=2)
        category_var = tk.StringVar()
        category_combo = ttk.Combobox(search_frame, textvariable=category_var, width=27)
        category_combo['values'] = ('All', 'FullSample', 'ImageShow', 'Mask_TaskAll')
        category_combo.set('All')
        category_combo.grid(row=4, column=1, sticky=(tk.W, tk.E), pady=2)
        
        search_frame.columnconfigure(1, weight=1)
        
        # Results
        results_frame = ttk.LabelFrame(search_window, text="Results", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        results_tree = ttk.Treeview(results_frame, columns=('center', 'patient', 'type', 'path'), show='tree headings')
        results_tree.heading('#0', text='Filename')
        results_tree.heading('center', text='Center')
        results_tree.heading('patient', text='Patient')
        results_tree.heading('type', text='Type')
        results_tree.heading('path', text='Path')
        
        results_tree.pack(fill=tk.BOTH, expand=True)
        
        def perform_search():
            # Clear results
            for item in results_tree.get_children():
                results_tree.delete(item)
            
            # Build query
            query = 'SELECT filename, center, patient, file_type, relative_path FROM mri_files WHERE 1=1'
            params = []
            
            if filename_var.get():
                query += ' AND filename LIKE ?'
                params.append(f'%{filename_var.get()}%')
            
            if center_var.get():
                query += ' AND center LIKE ?'
                params.append(f'%{center_var.get()}%')
            
            if patient_var.get():
                query += ' AND patient LIKE ?'
                params.append(f'%{patient_var.get()}%')
            
            if filetype_var.get() != 'All':
                query += ' AND file_type = ?'
                params.append(filetype_var.get())
            
            if category_var.get() != 'All':
                query += ' AND data_category = ?'
                params.append(category_var.get())
            
            query += ' ORDER BY relative_path'
            
            self.cursor.execute(query, params)
            results = self.cursor.fetchall()
            
            for result in results:
                filename, center, patient, file_type, relative_path = result
                results_tree.insert('', 'end', 
                                  values=(center or '', patient or '', file_type or '', relative_path),
                                  text=filename)
        
        ttk.Button(search_window, text="Search", command=perform_search).pack(pady=10)
    
    def update_statistics(self, dataset_id):
        """Update statistics tab with dataset information"""
        self.stats_text.config(state='normal')
        self.stats_text.delete('1.0', tk.END)
        
        # Basic stats
        self.cursor.execute('SELECT * FROM mri_datasets WHERE id = ?', (dataset_id,))
        dataset = self.cursor.fetchone()
        
        if dataset:
            _, name, root_path, dataset_type, total_size, total_files, centers_count, patients_count, created, modified, description, tags, notes = dataset
            
            stats_text = f"DATASET STATISTICS\n{'='*50}\n\n"
            stats_text += f"Dataset Name: {name}\n"
            stats_text += f"Type: {dataset_type}\n"
            stats_text += f"Total Size: {self.format_file_size(total_size)}\n"
            stats_text += f"Total Files: {total_files:,}\n"
            stats_text += f"Centers: {centers_count}\n"
            stats_text += f"Patients: {patients_count}\n\n"
            
            # File type distribution
            self.cursor.execute('''
                SELECT file_type, COUNT(*) as count, SUM(size) as total_size
                FROM mri_files WHERE dataset_id = ?
                GROUP BY file_type ORDER BY count DESC
            ''', (dataset_id,))
            file_types = self.cursor.fetchall()
            
            stats_text += f"FILE TYPE DISTRIBUTION\n{'-'*30}\n"
            for file_type, count, size in file_types:
                pct = (count / total_files) * 100
                stats_text += f"{file_type:>6}: {count:>6,} files ({pct:>5.1f}%) - {self.format_file_size(size)}\n"
            
            # Center distribution
            self.cursor.execute('''
                SELECT center, COUNT(*) as count
                FROM mri_files WHERE dataset_id = ? AND center IS NOT NULL
                GROUP BY center ORDER BY center
            ''', (dataset_id,))
            centers = self.cursor.fetchall()
            
            stats_text += f"\nCENTER DISTRIBUTION\n{'-'*30}\n"
            for center, count in centers:
                pct = (count / total_files) * 100
                stats_text += f"{center:>12}: {count:>6,} files ({pct:>5.1f}%)\n"
            
            # Patient distribution
            self.cursor.execute('''
                SELECT patient, COUNT(*) as count
                FROM mri_files WHERE dataset_id = ? AND patient IS NOT NULL
                GROUP BY patient ORDER BY patient
            ''', (dataset_id,))
            patients = self.cursor.fetchall()
            
            stats_text += f"\nPATIENT DISTRIBUTION\n{'-'*30}\n"
            for patient, count in patients[:20]:  # Show first 20
                pct = (count / total_files) * 100
                stats_text += f"{patient:>8}: {count:>6,} files ({pct:>5.1f}%)\n"
            
            if len(patients) > 20:
                stats_text += f"... and {len(patients) - 20} more patients\n"
            
            # Data category distribution
            self.cursor.execute('''
                SELECT data_category, COUNT(*) as count
                FROM mri_files WHERE dataset_id = ? AND data_category IS NOT NULL
                GROUP BY data_category ORDER BY count DESC
            ''', (dataset_id,))
            categories = self.cursor.fetchall()
            
            stats_text += f"\nDATA CATEGORY DISTRIBUTION\n{'-'*30}\n"
            for category, count in categories:
                pct = (count / total_files) * 100
                stats_text += f"{category:>15}: {count:>6,} files ({pct:>5.1f}%)\n"
            
            self.stats_text.insert('1.0', stats_text)
        
        self.stats_text.config(state='disabled')
    
    def save_dataset(self):
        """Save dataset changes"""
        if not self.current_dataset_id:
            return
        
        try:
            description = self.description_text.get('1.0', tk.END).strip()
            notes = self.notes_text.get('1.0', tk.END).strip()
            
            self.cursor.execute('''
                UPDATE mri_datasets 
                SET name = ?, dataset_type = ?, tags = ?, description = ?, notes = ?, last_modified = ?
                WHERE id = ?
            ''', (self.name_var.get(), self.type_var.get(), self.tags_var.get(),
                  description, notes, datetime.now().isoformat(), self.current_dataset_id))
            
            self.conn.commit()
            self.refresh_tree()
            self.status_var.set("Dataset saved successfully")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save dataset: {str(e)}")
    
    def export_report(self):
        """Export comprehensive dataset report"""
        if not self.current_dataset_id:
            messagebox.showwarning("Warning", "Please select a dataset first")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                # Export dataset with all files
                self.cursor.execute('SELECT * FROM mri_datasets WHERE id = ?', (self.current_dataset_id,))
                dataset = self.cursor.fetchone()
                
                if dataset:
                    dataset_dict = {
                        'dataset_info': {
                            'id': dataset[0],
                            'name': dataset[1],
                            'root_path': dataset[2],
                            'dataset_type': dataset[3],
                            'total_size': dataset[4],
                            'total_files': dataset[5],
                            'centers_count': dataset[6],
                            'patients_count': dataset[7],
                            'created_date': dataset[8],
                            'last_modified': dataset[9],
                            'description': dataset[10],
                            'tags': dataset[11],
                            'notes': dataset[12]
                        }
                    }
                    
                    # Add files
                    self.cursor.execute('SELECT * FROM mri_files WHERE dataset_id = ?', (self.current_dataset_id,))
                    files = self.cursor.fetchall()
                    
                    dataset_dict['files'] = []
                    for file_info in files:
                        file_dict = {
                            'filename': file_info[2],
                            'filepath': file_info[3],
                            'file_type': file_info[4],
                            'size': file_info[5],
                            'center': file_info[6],
                            'scanner': file_info[7],
                            'patient': file_info[8],
                            'sequence_type': file_info[9],
                            'data_category': file_info[10],
                            'relative_path': file_info[11],
                            'created_date': file_info[12],
                            'modified_date': file_info[13]
                        }
                        dataset_dict['files'].append(file_dict)
                    
                    with open(filename, 'w') as f:
                        json.dump(dataset_dict, f, indent=2)
                    
                    self.status_var.set(f"Report exported to {filename}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export report: {str(e)}")
    
    def format_file_size(self, size):
        """Format file size in human readable format"""
        if size < 1024:
            return f"{size} B"
        elif size < 1024 * 1024:
            return f"{size / 1024:.1f} KB"
        elif size < 1024 * 1024 * 1024:
            return f"{size / (1024 * 1024):.1f} MB"
        else:
            return f"{size / (1024 * 1024 * 1024):.1f} GB"
    
    def __del__(self):
        """Close database connection"""
        if hasattr(self, 'conn'):
            self.conn.close()

if __name__ == "__main__":
    root = tk.Tk()
    app = MRIDatasetManager(root)
    root.mainloop()
