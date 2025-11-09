import tkinter as tk
from tkinter import messagebox, ttk, font
from predictor import CarPredictor
import pandas as pd

class CarRecommendationGUI:
    def __init__(self):
        self.predictor = CarPredictor()
        self.options = self.predictor.get_available_options()
        self.setup_window()
        self.create_widgets()
        
    def setup_window(self):
        self.root = tk.Tk()
        self.root.title("🚗 Automobile Recommendation System")
        self.root.geometry("750x850")
        self.root.configure(bg='#f5f6fa')
        self.root.resizable(False, False)
        
        # Center window
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
        
        # Fonts
        self.title_font = font.Font(family="Segoe UI", size=24, weight="bold")
        self.subtitle_font = font.Font(family="Segoe UI", size=12)
        self.label_font = font.Font(family="Segoe UI", size=11)
        self.button_font = font.Font(family="Segoe UI", size=11, weight="bold")
        self.result_font = font.Font(family="Segoe UI", size=13, weight="bold")
    
    def create_widgets(self):
        self.create_header()
        self.create_input_section()
        self.create_result_section()
        self.create_footer()
    
    def create_header(self):
        header_frame = tk.Frame(self.root, bg='#4834d4', height=120)
        header_frame.pack(fill='x')
        header_frame.pack_propagate(False)
        
        header_content = tk.Frame(header_frame, bg='#4834d4')
        header_content.pack(expand=True)
        
        tk.Label(header_content, text="🚗 Automobile Recommendation System", 
                font=self.title_font, bg='#4834d4', fg='white').pack(pady=(15, 5))
        
        accuracy_text = f"AI-Powered with {self.options['accuracy']:.1%} Accuracy | BDA Project"
        tk.Label(header_content, text=accuracy_text, 
                font=self.subtitle_font, bg='#4834d4', fg='#dfe6e9').pack()
    
    def create_input_section(self):
        input_container = tk.Frame(self.root, bg='#f5f6fa')
        input_container.pack(fill='x', padx=40, pady=20)
        
        input_card = tk.Frame(input_container, bg='white', relief='raised', bd=1)
        input_card.pack(fill='x')
        
        tk.Label(input_card, text="Enter Your Preferences", 
                font=font.Font(family="Segoe UI", size=16, weight="bold"),
                bg='white', fg='#2d3436').pack(pady=20)
        
        # Input fields
        fields_frame = tk.Frame(input_card, bg='white')
        fields_frame.pack(padx=50, pady=(0, 20))
        
        # Variables
        self.year_var = tk.StringVar()
        self.price_var = tk.StringVar()
        self.fuel_var = tk.StringVar()
        self.gear_var = tk.StringVar()
        self.manufacturer_var = tk.StringVar()
        self.color_var = tk.StringVar()
        
        # Create input rows
        self.create_input_row(fields_frame, "📅 Year (2005-2024):", self.year_var)
        self.create_input_row(fields_frame, "💰 Price ($):", self.price_var)
        self.create_input_row(fields_frame, "⛽ Fuel Type:", self.fuel_var, 
                             self.options['fuel_types'])
        self.create_input_row(fields_frame, "⚙️ Transmission:", self.gear_var, 
                             self.options['gear_types'])
        self.create_input_row(fields_frame, "🏭 Manufacturer:", self.manufacturer_var, 
                             self.options['manufacturers'])
        self.create_input_row(fields_frame, "🎨 Color:", self.color_var, 
                             self.options['colors'])
        
        # Buttons
        self.create_buttons(input_card)
    
    def create_input_row(self, parent, label, var, values=None):
        frame = tk.Frame(parent, bg='white')
        frame.pack(fill='x', pady=8)
        tk.Label(frame, text=label, font=self.label_font, bg='white', 
                width=18, anchor='w').pack(side='left')
        
        if values:
            combo = ttk.Combobox(frame, textvariable=var, values=sorted(values), 
                                 font=self.label_font, width=28, state='readonly')
            combo.pack(side='left', padx=10)
        else:
            entry = tk.Entry(frame, textvariable=var, font=self.label_font, 
                            width=30, relief='solid', bd=1)
            entry.pack(side='left', padx=10)
    
    def create_buttons(self, parent):
        buttons_frame = tk.Frame(parent, bg='white')
        buttons_frame.pack(pady=(0, 30))
        
        predict_btn = tk.Button(buttons_frame, text="🔍 Find My Car", 
                               command=self.predict_car,
                               font=self.button_font, bg='#0984e3', fg='white',
                               padx=25, pady=10, relief='flat', cursor='hand2')
        predict_btn.pack(side='left', padx=10)
        
        self.clear_btn = tk.Button(buttons_frame, text="🔄 Clear", 
                                  command=self.clear_inputs,
                                  font=self.button_font, bg='#b2bec3', fg='white',
                                  padx=25, pady=10, relief='flat', state='disabled')
        self.clear_btn.pack(side='left', padx=10)
        
        about_btn = tk.Button(buttons_frame, text="ℹ️ About", 
                             command=self.show_about,
                             font=self.button_font, bg='#a29bfe', fg='white',
                             padx=25, pady=10, relief='flat', cursor='hand2')
        about_btn.pack(side='left', padx=10)
    
    def create_result_section(self):
        result_container = tk.Frame(self.root, bg='#f5f6fa')
        result_container.pack(fill='x', padx=40, pady=(0, 20))
        
        self.result_frame = tk.Frame(result_container, bg='#dfe6e9', 
                                     relief='raised', bd=1, height=150)
        self.result_frame.pack(fill='x')
        self.result_frame.pack_propagate(False)
        
        result_content = tk.Frame(self.result_frame, bg='#dfe6e9')
        result_content.pack(expand=True)
        
        self.category_label = tk.Label(result_content, 
                                       text="Your Recommendation Will Appear Here",
                                       font=font.Font(family="Segoe UI", size=14, weight="bold"),
                                       bg='#dfe6e9', fg='#636e72')
        self.category_label.pack(pady=(20, 10))
        
        self.models_label = tk.Label(result_content, text="",
                                     font=font.Font(family="Segoe UI", size=12),
                                     bg='#dfe6e9', fg='#2d3436', justify='left')
        self.models_label.pack()
        
        self.confidence_label = tk.Label(result_content, text="",
                                         font=font.Font(family="Segoe UI", size=11),
                                         bg='#dfe6e9', fg='#2d3436')
        self.confidence_label.pack(pady=5)
    
    def create_footer(self):
        footer = tk.Frame(self.root, bg='#2d3436', height=40)
        footer.pack(fill='x', side='bottom')
        footer.pack_propagate(False)
        
        tk.Label(footer, text="Developed by Prathamesh Parab | BIG DATA ANALYTICS",
                font=font.Font(family="Segoe UI", size=10),
                bg='#2d3436', fg='#dfe6e9').pack(pady=10)
    
    def predict_car(self):
        try:
            # Get inputs
            year = int(self.year_var.get())
            price = int(self.price_var.get())
            fuel_type = self.fuel_var.get()
            gear_type = self.gear_var.get()
            manufacturer = self.manufacturer_var.get()
            color = self.color_var.get()
            
            if not all([fuel_type, gear_type, manufacturer, color]):
                messagebox.showwarning("⚠️ Input Error", "Please fill in all fields!")
                return
            
            # Make prediction
            result = self.predictor.predict(year, price, fuel_type, gear_type, 
                                          manufacturer, color)
            
            if result['success']:
                # Update result display
                self.result_frame.configure(bg='#00b894')
                self.category_label.config(text=f"Category: {result['category']}", 
                                         bg='#00b894', fg='white')
                
                # Show recommended models
                models_text = "Recommended Models:\n"
                for i, car in enumerate(result['recommended_cars'], 1):
                    models_text += f"{i}. {car.upper()}\n"
                
                self.models_label.config(text=models_text, bg='#00b894', fg='white')
                self.confidence_label.config(text=f"Confidence: {result['confidence']:.1%}", 
                                           bg='#00b894', fg='white')
                
                self.clear_btn.config(state='normal', bg='#e17055')
            else:
                # Show errors
                error_msg = "\n".join(result['errors'])
                messagebox.showerror("❌ Error", error_msg)
                
        except ValueError:
            messagebox.showerror("❌ Input Error", 
                               "Please enter valid numeric values for Year and Price.")
    
    def clear_inputs(self):
        self.year_var.set("")
        self.price_var.set("")
        self.fuel_var.set("")
        self.gear_var.set("")
        self.manufacturer_var.set("")
        self.color_var.set("")
        
        self.result_frame.configure(bg='#dfe6e9')
        self.category_label.config(text="Your Recommendation Will Appear Here", 
                                  fg='#636e72', bg='#dfe6e9')
        self.models_label.config(text="", bg='#dfe6e9')
        self.confidence_label.config(text="", bg='#dfe6e9')
        self.clear_btn.config(state='disabled', bg='#b2bec3')
    
    def show_about(self):
        about_window = tk.Toplevel(self.root)
        about_window.title("About")
        about_window.geometry("550x450")
        about_window.configure(bg='white')
        about_window.resizable(False, False)
        
        # Header
        header = tk.Frame(about_window, bg='#6c5ce7', height=80)
        header.pack(fill='x')
        header.pack_propagate(False)
        
        tk.Label(header, text="About This Project", 
                font=font.Font(family="Segoe UI", size=20, weight="bold"),
                bg='#6c5ce7', fg='white').pack(pady=25)
        
        # Content
        content = tk.Frame(about_window, bg='white', padx=30, pady=20)
        content.pack(fill='both', expand=True)
        
        about_text = f"""🚗 Automobile Management System

Developed by: Prathamesh Parab
Course: Big Data Analytics 
Institution: St. Francis Institute of Technology
Academic Year: 2023-2024
Guide: Ms. Jayshree Mittal

This system uses Machine Learning to recommend cars
based on user preferences with {self.options['accuracy']:.1%} accuracy.

The model analyzes:
• Year and Price Range
• Fuel Type and Transmission
• Manufacturer and Color Preferences

Trained on 5,000+ vehicle records using
Random Forest Classification algorithm."""
        
        tk.Label(content, text=about_text, 
                font=('Segoe UI', 11), bg='white', fg='#2d3436',
                justify='left').pack()
        
        tk.Button(about_window, text="Close", command=about_window.destroy,
                 font=self.button_font, bg='#6c5ce7', fg='white',
                 padx=30, pady=8, relief='flat').pack(pady=20)
    
    def run(self):
        """Start the GUI"""
        self.root.mainloop()

if __name__ == "__main__":
    app = CarRecommendationGUI()
    app.run()