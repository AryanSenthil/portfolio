import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import pandas as pd


class StressStrainAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("Stress-Strain Analyzer")
        self.root.geometry("1400x900")
        
        # Data storage
        self.extension = []
        self.strain = []
        self.stress = []
        self.mean_extension = None
        self.uncertainty = None
        self.location = []
        
        # Setup UI
        self.setup_ui()
        
    def setup_ui(self):
        # Top control panel
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E))
        
        # File import section
        ttk.Label(control_frame, text="Data Import", font=('Arial', 12, 'bold')).grid(row=0, column=0, padx=10, pady=10)
        
        ttk.Button(control_frame, text="Import CSV File", command=self.import_file, width=20).grid(row=0, column=1, pady=10, padx=5)
        
        self.file_label = ttk.Label(control_frame, text="No file loaded", foreground="gray")
        self.file_label.grid(row=0, column=2, pady=10, padx=10)
        
        # Analysis button
        ttk.Button(control_frame, text="Find Extension", command=self.find_extension, width=20).grid(row=0, column=3, pady=10, padx=5)
        
        # Results display
        results_frame = ttk.Frame(control_frame, relief='solid', borderwidth=1, padding="5")
        results_frame.grid(row=0, column=4, columnspan=2, padx=20, pady=10)
        
        ttk.Label(results_frame, text="Mean Extension:").grid(row=0, column=0, sticky=tk.W, pady=2, padx=5)
        self.extension_label = ttk.Label(results_frame, text="---", foreground="blue", font=('Arial', 10, 'bold'))
        self.extension_label.grid(row=0, column=1, sticky=tk.W, pady=2, padx=5)
        
        ttk.Label(results_frame, text="Uncertainty:").grid(row=1, column=0, sticky=tk.W, pady=2, padx=5)
        self.uncertainty_label = ttk.Label(results_frame, text="---", foreground="blue", font=('Arial', 10, 'bold'))
        self.uncertainty_label.grid(row=1, column=1, sticky=tk.W, pady=2, padx=5)
        
        # Main content area - 2 columns
        # Left: Data preview
        data_frame = ttk.LabelFrame(self.root, text="Data Preview", padding="10")
        data_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        # Treeview for data display
        tree_scroll = ttk.Scrollbar(data_frame)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.tree = ttk.Treeview(data_frame, yscrollcommand=tree_scroll.set, height=35)
        self.tree.pack(fill=tk.BOTH, expand=True)
        tree_scroll.config(command=self.tree.yview)
        
        self.tree['columns'] = ('Extension', 'Strain', 'Stress')
        self.tree.column('#0', width=60, minwidth=50)
        self.tree.column('Extension', width=100, minwidth=80)
        self.tree.column('Strain', width=100, minwidth=80)
        self.tree.column('Stress', width=100, minwidth=80)
        
        self.tree.heading('#0', text='Index')
        self.tree.heading('Extension', text='Extension (mm)')
        self.tree.heading('Strain', text='Strain')
        self.tree.heading('Stress', text='Stress (MPa)')
        
        # Right: Stress vs Extension plot
        plot_frame = ttk.LabelFrame(self.root, text="Stress vs Extension", padding="10")
        plot_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        self.fig = Figure(figsize=(8, 8), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Configure grid weights for responsive layout
        self.root.columnconfigure(0, weight=1, minsize=450)
        self.root.columnconfigure(1, weight=2, minsize=700)
        self.root.rowconfigure(1, weight=1)
        
    def import_file(self):
        filename = filedialog.askopenfilename(
            title="Select CSV File",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
        )
        
        if not filename:
            return
        
        try:
            self.extension = []
            self.strain = []
            self.stress = []
            
            with open(filename, 'r') as file:
                csv_reader = csv.reader(file)
                
                # Skip the first 12 lines
                for _ in range(12):
                    next(csv_reader)
                
                # Read the desired columns from the 13th line onwards
                for row in csv_reader:
                    if len(row) >= 5:
                        try:
                            self.extension.append(float(row[1]))
                            self.strain.append(float(row[3]))
                            self.stress.append(float(row[4]))
                        except ValueError:
                            continue
            
            self.file_label.config(text=f"Loaded: {filename.split('/')[-1]}", foreground="green")
            self.update_data_view()
            self.plot_data()
            messagebox.showinfo("Success", f"Loaded {len(self.extension)} data points")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {str(e)}")
    
    def update_data_view(self):
        # Clear existing data
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Display first 100 rows
        for i in range(min(100, len(self.extension))):
            self.tree.insert('', tk.END, text=str(i),
                           values=(f"{self.extension[i]:.5f}",
                                  f"{self.strain[i]:.5f}",
                                  f"{self.stress[i]:.6f}"))
    
    def plot_data(self, show_extension_line=False):
        # Clear plot
        self.ax.clear()
        
        if len(self.extension) > 0:
            # Stress vs Extension plot
            self.ax.plot(self.extension, self.stress, 'b-', linewidth=1.5, label='Stress-Extension')
            
            if show_extension_line and self.mean_extension is not None:
                self.ax.axvline(self.mean_extension, color='r', linestyle='--', 
                              linewidth=2, label=f'Mean Extension: {self.mean_extension:.5f} mm')
            
            self.ax.axhline(0, color='k', linestyle=':', linewidth=1, alpha=0.5)
            self.ax.set_xlabel('Extension (mm)', fontsize=12)
            self.ax.set_ylabel('Stress (MPa)', fontsize=12)
            self.ax.set_title('Stress vs. Extension', fontsize=13, fontweight='bold')
            self.ax.grid(True, alpha=0.3)
            self.ax.legend()
            
        self.fig.tight_layout()
        self.canvas.draw()
    
    def find_extension(self):
        if len(self.extension) == 0:
            messagebox.showwarning("Warning", "Please import data first")
            return
        
        # Analysis logic
        self.location = []
        tolerance = 1e-1
        
        for i in range(2, len(self.strain) - 2):
            slopes_before = []
            for j in range(i-2, i):
                if (self.strain[j + 1] - self.strain[j]) != 0:
                    slope = (self.stress[j + 1] - self.stress[j]) / (self.strain[j + 1] - self.strain[j])
                else:
                    slope = 0
                slopes_before.append(slope)
            
            # Check if slope at i is positive
            if (self.strain[i] - self.strain[i - 1]) != 0:
                slope_i = (self.stress[i] - self.stress[i - 1]) / (self.strain[i] - self.strain[i - 1])
            else:
                slope_i = 0
            
            slopes_after = []
            for j in range(i, i+2):
                if (self.strain[j + 1] - self.strain[j]) != 0:
                    slope = (self.stress[j + 1] - self.stress[j]) / (self.strain[j + 1] - self.strain[j])
                else:
                    slope = 0
                slopes_after.append(slope)
            
            if all(slope <= tolerance for slope in slopes_before) \
                and all(slope > -tolerance for slope in slopes_after) \
                and slope_i > -tolerance and self.stress[i] >= 0 and self.strain[i] >= 0:
                self.location.append(i)
        
        if len(self.location) == 0:
            messagebox.showwarning("Warning", "No valid extension points found")
            return
        
        extension_at_location = [self.extension[index] for index in self.location]
        self.mean_extension = np.mean(extension_at_location)
        self.uncertainty = np.std(extension_at_location) / np.sqrt(len(extension_at_location))
        
        # Update results
        self.extension_label.config(text=f"{self.mean_extension:.8f} mm")
        self.uncertainty_label.config(text=f"{self.uncertainty:.8f} mm")
        
        # Update plot with red line
        self.plot_data(show_extension_line=True)
        
        messagebox.showinfo("Analysis Complete", 
                          f"Found {len(self.location)} extension points\n\n"
                          f"Mean Extension: {self.mean_extension:.8f} mm\n"
                          f"Uncertainty: {self.uncertainty:.8f} mm")


def main():
    root = tk.Tk()
    app = StressStrainAnalyzer(root)
    root.mainloop()


if __name__ == "__main__":
    main()