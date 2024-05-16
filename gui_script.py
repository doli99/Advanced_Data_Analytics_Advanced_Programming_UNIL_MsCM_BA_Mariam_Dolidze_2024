import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import seaborn as sns
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*Matplotlib.*")

# Ensure Matplotlib backend is set to TkAgg
plt.switch_backend('TkAgg')

class TaxiTripEDAApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Taxi Trip EDA")

        # Load the dataset
        self.df = pd.read_parquet('sampled_taxi_dataset_v.1.parquet')

        # Identify numeric and categorical features
        self.numeric_features = ['fare_amount', 'trip_distance', 'trip_duration', 'speed_mph', 'tip_amount']
        self.categorical_features = ['pickup_time_of_day', 'pickup_day_type', 'pickup_season', 'is_holiday', 'PUcategory', 'DOcategory']

        # Set up the main frame with yellow background
        self.main_frame = ttk.Frame(root, padding="10", style="Main.TFrame")
        self.main_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure the grid to be resizable
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.rowconfigure(1, weight=1)
        
        # Create the header frame
        self.create_header_frame()

        # Create the tabbed interface
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Create the Main Page tab
        self.main_page_frame = ttk.Frame(self.notebook, style="Content.TFrame")
        self.notebook.add(self.main_page_frame, text='Main Page')
        self.create_main_page(self.main_page_frame)

        # Create the Statistics tab
        self.statistics_frame = ttk.Frame(self.notebook, style="Content.TFrame")
        self.notebook.add(self.statistics_frame, text='Statistics')
        self.create_statistics(self.statistics_frame)

        # Create the Visualizations tab
        self.visualizations_frame = ttk.Frame(self.notebook, style="Content.TFrame")
        self.notebook.add(self.visualizations_frame, text='Visualizations')
        self.create_visualizations(self.visualizations_frame)

        # Create the Correlation tab
        self.correlation_frame = ttk.Frame(self.notebook, style="Content.TFrame")
        self.notebook.add(self.correlation_frame, text='Correlation')
        self.create_correlation(self.correlation_frame)

        # Create the Modeling tab
        self.model_frame = ttk.Frame(self.notebook, style="Content.TFrame")
        self.notebook.add(self.model_frame, text='Modeling')
        self.create_model(self.model_frame)

        # Create and set up the menu bar
        self.create_menu_bar()

    def create_header_frame(self):
        header_frame = ttk.Frame(self.root, style="Header.TFrame")
        header_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))

        # Add logo
        logo_image = Image.open('logo.png')
        logo_image = logo_image.resize((100, 100), Image.ANTIALIAS)
        logo_photo = ImageTk.PhotoImage(logo_image)
        logo_label = ttk.Label(header_frame, image=logo_photo, style="Header.TLabel")
        logo_label.image = logo_photo  # Keep a reference to avoid garbage collection
        logo_label.grid(row=0, column=0, padx=10)

        # Add title text
        title_text = (
            "Taxi Trip EDA Application\n"
            "Understand key features influencing taxi fares.\n"
            "Assist policy makers to understand what influences taxi demand and pricing."
        )
        title_label = ttk.Label(header_frame, text=title_text, style="Header.TLabel")
        title_label.grid(row=0, column=1, padx=10)

    def create_menu_bar(self):
        menu_bar = tk.Menu(self.root)
        self.root.config(menu=menu_bar)
        
        # Create the File menu
        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="Introduction", command=lambda: self.notebook.select(self.main_page_frame))
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menu_bar.add_cascade(label="File", menu=file_menu)
        
        # Create the Charts menu
        charts_menu = tk.Menu(menu_bar, tearoff=0)
        charts_menu.add_command(label="Fare Amount Distribution", command=lambda: self.show_image('fare_amount_distribution.png'))
        charts_menu.add_command(label="Trip Duration Distribution", command=lambda: self.show_image('trip_duration_distribution.png'))
        charts_menu.add_command(label="Trip Distance Distribution", command=lambda: self.show_image('trip_distance_distribution.png'))
        charts_menu.add_command(label="Correlation Matrix", command=lambda: self.show_image('correlation_matrix.png'))
        menu_bar.add_cascade(label="Charts", menu=charts_menu)
        

    def create_main_page(self, parent):
        # Add main page content
        logo_image = Image.open('logo.png')
        logo_image = logo_image.resize((120, 120), Image.ANTIALIAS)
        logo_photo = ImageTk.PhotoImage(logo_image)
        logo_label = ttk.Label(parent, image=logo_photo, style="Content.TLabel")
        logo_label.image = logo_photo  # Keep a reference to avoid garbage collection
        logo_label.pack(pady=10)

        description_text = (
            "Taxi Trip EDA Application\n\n"
            "Welcome to the Taxi Trip EDA Application! This tool helps you understand key features influencing taxi fares, "
            "trends in taxi trip demand, and factors affecting fares through various visualizations and predictive models.\n\n"
            "Use this application to explore the data, visualize trends, and gain insights into the factors that influence "
            "taxi fares in NYC. This can help in creating predictive models for cities where such data is not readily available.\n\n"
            "Navigate through the tabs to access different features and visualizations. The Statistics tab provides summary "
            "statistics for selected features, while the Visualizations tab allows you to create various types of plots to "
            "analyze the data."
        )

        description_label = tk.Label(parent, text=description_text, font=("Helvetica", 12), justify="center", wraplength=600)
        description_label.pack(pady=10)

        # Display key statistics
        stats_frame = ttk.Frame(parent, style="Content.TFrame")
        stats_frame.pack(pady=20)
        
        # Calculate statistics
        num_records = len(self.df)
        avg_trip_distance = self.df['trip_distance'].mean()
        avg_trip_duration = self.df['trip_duration'].mean()
        avg_fare_amount = self.df['fare_amount'].mean()
        avg_speed = self.df['speed_mph'].mean()
        avg_tip_amount = self.df['tip_amount'].mean()

        stats = [
            (f"{num_records:,}", "Records"),
            (f"{avg_trip_distance:.2f} miles", "Avg. Trip Distance"),
            (f"{avg_trip_duration:.2f} minutes", "Avg. Trip Duration"),
            (f"${avg_fare_amount:.2f}", "Avg. Fare Amount"),
            (f"{avg_speed:.2f} mph", "Avg. Speed"),
            (f"${avg_tip_amount:.2f}", "Avg. Tip Amount")
        ]
        
        for value, text in stats:
            stat_frame = ttk.Frame(stats_frame, style="Stats.TFrame")
            stat_frame.pack(side=tk.LEFT, padx=10)
            value_label = ttk.Label(stat_frame, text=value, style="StatValue.TLabel")
            value_label.pack()
            text_label = ttk.Label(stat_frame, text=text, style="StatText.TLabel")
            text_label.pack()

    def create_statistics(self, parent):
        # Frame for selecting features
        feature_select_frame = ttk.Frame(parent, style="Content.TFrame")
        feature_select_frame.pack(pady=20)

        # Dropdown for selecting features
        self.stat_feature_var = tk.StringVar()
        stat_feature_dropdown = ttk.Combobox(feature_select_frame, textvariable=self.stat_feature_var)
        stat_feature_dropdown['values'] = list(self.df.columns)
        stat_feature_dropdown.set("Select Feature for Statistics")
        stat_feature_dropdown.pack(side=tk.LEFT, padx=10)

        # Button to display statistics
        stat_button = ttk.Button(feature_select_frame, text="Show Statistics", command=self.show_statistics)
        stat_button.pack(side=tk.LEFT, padx=10)

        # Frame for displaying statistics
        self.stats_display_frame = ttk.Frame(parent, style="Content.TFrame")
        self.stats_display_frame.pack(fill=tk.BOTH, expand=True)

    def show_statistics(self):
        feature = self.stat_feature_var.get()
        if feature:
            self.clear_frame(self.stats_display_frame)
            stats = self.df[feature].describe()
            stats_str = f"For the {feature} feature, the summary statistics are as follows:\n\n{stats.to_string()}"
            stats_label = tk.Label(self.stats_display_frame, text=stats_str, font=("Helvetica", 12), justify="center", wraplength=600)
            stats_label.pack(pady=10)

    def create_visualizations(self, parent):
        # Create interactive buttons to select features and types of visualizations
        button_frame = ttk.Frame(parent, style="Content.TFrame")
        button_frame.pack(pady=10)

        # Dropdown for selecting features
        self.feature_var = tk.StringVar()
        feature_dropdown = ttk.Combobox(button_frame, textvariable=self.feature_var)
        feature_dropdown['values'] = [""] + self.numeric_features + self.categorical_features
        feature_dropdown.set("Select Feature")
        feature_dropdown.bind("<<ComboboxSelected>>", self.update_viz_options)
        feature_dropdown.pack(side=tk.LEFT, padx=10)

        # Dropdown for selecting types of visualizations
        self.viz_type_var = tk.StringVar()
        self.viz_type_dropdown = ttk.Combobox(button_frame, textvariable=self.viz_type_var)
        self.viz_type_dropdown.pack(side=tk.LEFT, padx=10)

        # Button to generate the selected visualization
        viz_button = ttk.Button(button_frame, text="Generate Visualization", command=self.generate_visualization)
        viz_button.pack(side=tk.LEFT, padx=10)

        # Frame for displaying visualizations
        self.viz_display_frame = ttk.Frame(parent, style="Content.TFrame")
        self.viz_display_frame.pack(fill=tk.BOTH, expand=True)

    def update_viz_options(self, event):
        feature = self.feature_var.get()
        if feature in self.numeric_features:
            self.viz_type_dropdown['values'] = ["Histogram", "Box Plot", "Scatter Plot"]
        elif feature in self.categorical_features:
            self.viz_type_dropdown['values'] = ["Count Plot", "Box Plot"]
        else:
            self.viz_type_dropdown['values'] = []

    def generate_visualization(self):
        feature = self.feature_var.get()
        viz_type = self.viz_type_var.get()

        if feature and viz_type:
            # Clear previous visualizations
            self.clear_frame(self.viz_display_frame)

            # Enhance plot aesthetics
            sns.set_style("whitegrid")

            fig, ax = plt.subplots(figsize=(10, 6))

            if feature in self.numeric_features:
                if viz_type == "Histogram":
                    sns.histplot(self.df[feature], kde=True, color='#FFD700', edgecolor='#000000', ax=ax)
                    ax.set_title(f'Histogram of {feature}', fontsize=16)
                    ax.set_xlabel(feature, fontsize=14)
                    ax.set_ylabel('Frequency', fontsize=14)
                elif viz_type == "Box Plot":
                    sns.boxplot(y=self.df[feature], color='#FFD700', ax=ax)
                    ax.set_title(f'Box Plot of {feature}', fontsize=16)
                    ax.set_xlabel(feature, fontsize=14)
                elif viz_type == "Scatter Plot":
                    sns.scatterplot(x=self.df[feature], y=self.df['fare_amount'], color='#FFD700', edgecolor='#000000', ax=ax)
                    ax.set_title(f'Scatter Plot of {feature} vs Fare Amount', fontsize=16)
                    ax.set_xlabel(feature, fontsize=14)
                    ax.set_ylabel('Fare Amount', fontsize=14)
            elif feature in self.categorical_features:
                if viz_type == "Count Plot":
                    sns.countplot(y=self.df[feature], palette=['#FFD700', '#000000'], ax=ax)
                    ax.set_title(f'Count Plot of {feature}', fontsize=16)
                    ax.set_xlabel(feature, fontsize=14)
                    ax.set_ylabel('Count', fontsize=14)
                elif viz_type == "Box Plot":
                    sns.boxplot(x=self.df[feature], y=self.df['fare_amount'], palette=['#FFD700', '#000000'], ax=ax)
                    ax.set_title(f'Box Plot of {feature} vs Fare Amount', fontsize=16)
                    ax.set_xlabel(feature, fontsize=14)
                    ax.set_ylabel('Fare Amount', fontsize=14)

            # Embed the plot in the Tkinter canvas
            canvas = FigureCanvasTkAgg(fig, master=self.viz_display_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def create_correlation(self, parent):
        self.clear_frame(parent)
        sns.set_style("whitegrid")
        fig, ax = plt.subplots(figsize=(12, 10))
        corr_matrix = self.df[self.numeric_features].corr()
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax)
        ax.set_title('Correlation Matrix', fontsize=16)
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
    def show_correlation_matrix(self):
        self.notebook.select(self.correlation_frame)
        self.create_correlation(self.correlation_frame)

    def clear_frame(self, frame):
        for widget in frame.winfo_children():
            widget.destroy()

    def show_about(self):
        about_text = "This is a Taxi Trip EDA application."
        tk.messagebox.showinfo("About", about_text)

    def create_model(self, parent):
        # Introduction to the modeling section
        intro_text = (
            "Modeling Section\n\n"
            "In this section, we developed various predictive models for taxi fares using the NYC yellow taxi trip dataset from 2023. "
            "The goal was to identify key factors influencing fare amounts and create models that accurately estimate fares. "
            "This section displays the results of the models, including training and validation scores.\n"
            "The models include Linear Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost, and an Ensemble Model."
        )
        intro_label = tk.Label(parent, text=intro_text, font=("Helvetica", 12), justify="left", wraplength=600)
        intro_label.pack(pady=10)

        # Frame for model buttons
        model_button_frame = ttk.Frame(parent, style="Content.TFrame")
        model_button_frame.pack(pady=10)

        # Button for each model
        models = ['Linear Regression', 'Decision Tree', 'Random Forest', 'Gradient Boosting', 'XGBoost', 'Ensemble Model']
        for model in models:
            button = ttk.Button(model_button_frame, text=model, command=lambda m=model: self.show_model_results(m))
            button.pack(side=tk.LEFT, padx=10)

        # Button to clear model results
        clear_button = ttk.Button(model_button_frame, text="Clear Model Results", command=self.clear_model_results)
        clear_button.pack(side=tk.LEFT, padx=10)

        # Frame for displaying model results
        self.model_display_frame = ttk.Frame(parent, style="Content.TFrame")
        self.model_display_frame.pack(fill=tk.BOTH, expand=True)

    def show_model_results(self, model_name):
        self.clear_frame(self.model_display_frame)
        
        # Sample model results data
        model_results = {
            'Linear Regression': {'Validation MAE': 0.098, 'Validation R²': 0.975, 'Test MAE': 0.099, 'Test R²': 0.975},
            'Decision Tree': {'Validation MAE': 0.094, 'Validation R²': 0.976, 'Test MAE': 0.095, 'Test R²': 0.975},
            'Random Forest': {'Validation MAE': 0.088, 'Validation R²': 0.979, 'Test MAE': 0.088, 'Test R²': 0.979},
            'Gradient Boosting': {'Validation MAE': 0.094, 'Validation R²': 0.978, 'Test MAE': 0.095, 'Test R²': 0.977},
            'XGBoost': {'Validation MAE': 0.075, 'Validation R²': 0.983, 'Test MAE': 0.075, 'Test R²': 0.981},
            'Ensemble Model': {'Validation MAE': 0.086, 'Validation R²': 0.980, 'Test MAE': 0.088, 'Test R²': 0.979},
        }
        
        # Display results for the selected model
        if model_name in model_results:
            results = model_results[model_name]
            results_text = (
                f"Results for {model_name}\n\n"
                f"Validation MAE: {results['Validation MAE']:.3f}\n"
                f"Validation R²: {results['Validation R²']:.3f}\n"
                f"Test MAE: {results['Test MAE']:.3f}\n"
                f"Test R²: {results['Test R²']:.3f}\n"
            )
            results_label = tk.Label(self.model_display_frame, text=results_text, font=("Helvetica", 12), justify="left", wraplength=600)
            results_label.pack(pady=10)
            
            # Placeholder for model summary and additional details
            summary_text = (
                f"Summary for {model_name}:\n\n"
                f"{model_name} was trained using the preprocessed dataset with appropriate feature transformations. "
                f"The model was evaluated using cross-validation to ensure robustness and generalizability. "
                f"The results indicate that {model_name} performed well on both the training and validation datasets.\n"
            )
            summary_label = tk.Label(self.model_display_frame, text=summary_text, font=("Helvetica", 12), justify="left", wraplength=600)
            summary_label.pack(pady=10)

    def clear_model_results(self):
        self.clear_frame(self.model_display_frame)

    def show_image(self, filepath):
        # Switch to the Visualizations tab and display the image
        self.notebook.select(self.visualizations_frame)
        self.clear_frame(self.visualizations_frame)
        image = Image.open(filepath)
        photo = ImageTk.PhotoImage(image)
        label = ttk.Label(self.visualizations_frame, image=photo, style="Content.TLabel")
        label.image = photo  # Keep a reference to avoid garbage collection
        label.pack(fill=tk.BOTH, expand=True)

    def clear_frame(self, frame):
        for widget in frame.winfo_children():
            widget.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    
    # Add styles
    style = ttk.Style(root)
    style.configure("Main.TFrame", background="white")
    style.configure("Button.TFrame", background="black")
    style.configure("TButton", background="white", foreground="black", font=("Helvetica", 10, "bold"))
    style.configure("Content.TFrame", background="grey")
    style.configure("Content.TLabel", background="black", foreground="black", font=("Helvetica", 12))
    style.configure("Header.TFrame", background="yellow")
    style.configure("Header.TLabel", background="yellow", foreground="black", font=("Helvetica", 16, "bold"))
    style.configure("Stats.TFrame", background="yellow")
    style.configure("StatValue.TLabel", background="yellow", foreground="black", font=("Helvetica", 24, "bold"))
    style.configure("StatText.TLabel", background="yellow", foreground="black", font=("Helvetica", 12))

    app = TaxiTripEDAApp(root)
    root.mainloop()
