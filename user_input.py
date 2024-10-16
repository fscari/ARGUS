import tkinter as tk
from tkinter import messagebox


def start_experiment(input_data, root):
    try:
        input_data['experiment_nr'] = int(experiment_nr_entry.get())
    except ValueError:
        messagebox.showerror("Input Error", "Please enter a valid experiment number.")
        return

    try:
        if fog_density_var.get():
            input_data['fog_density'] = [0, 50, 100]
        else:
            input_data['fog_density'] = [int(fog_density_entry.get())]
            if int(fog_density_entry.get()) not in [0, 50, 100]:
                messagebox.showerror("Input Error", "Fog density must be 0, 50, or 100.")
                return
    except ValueError:
        messagebox.showerror("Input Error", "Please enter a valid fog density (0, 50, or 100).")
        return

    try:
        if iteration_nr_joan_global is not None and int(iteration_nr_joan_global) > 0:
            if int(iteration_nr_entry.get()) != int(iteration_nr_joan_global):
                input_data['iteration_nr'] = int(iteration_nr_entry.get())
                messagebox.showwarning(f"Using the entered iteration number: {input_data['iteration_nr']}")
            else:
                input_data['iteration_nr'] = int(iteration_nr_joan_global)
        else:
            input_data['iteration_nr'] = int(iteration_nr_entry.get())
    except ValueError:
        messagebox.showerror("Input Error", "Please enter a valid iteration number.")
        return

    control_type = control_type_var.get()
    if control_type not in ["p", "f", "pf"]:
        messagebox.showerror("Input Error", "Please select a valid control type.")
        return

    input_data['type'] = {
        "p": "power_control",
        "f": "frequency_control",
        "pf": "pf"
    }.get(control_type, "none")

    # Close the root window to end the mainloop
    root.destroy()

def update_fog_density_entry():
    if fog_density_var.get():
        # make the fog_density entry empty and disable it
        fog_density_entry.delete(0, tk.END)
        fog_density_entry.config(state='disabled')
    else:
        # enable the fog_density entry
        fog_density_entry.config(state='normal')


def user_input(iteration_nr_joan=None, next_experiment_nr=None):
    global experiment_nr_entry, fog_density_entry, iteration_nr_entry, control_type_var, fog_density_var, iteration_nr_joan_global

    if iteration_nr_joan is not None:
        iteration_nr_joan_global = iteration_nr_joan
    else:
        iteration_nr_joan_global = None
    # Dictionary to store user inputs
    input_data = {}

    # Create main window
    root = tk.Tk()
    root.title("ARGUS Experiment Setup")

    # Experiment Number
    tk.Label(root, text="Experiment Number:").grid(row=0, column=0, padx=10, pady=5)
    experiment_nr_entry = tk.Entry(root)
    experiment_nr_entry.grid(row=0, column=1, padx=10, pady=5)
    if next_experiment_nr is not None:
        experiment_nr_entry.insert(0, next_experiment_nr)

    # Fog Density
    tk.Label(root, text="Fog Density (0, 50, 100):").grid(row=1, column=0, padx=10, pady=5)
    fog_density_entry = tk.Entry(root)
    fog_density_entry.grid(row=1, column=1, padx=10, pady=5)
    # create a checkbutton for selecting all fog densities
    fog_density_var = tk.IntVar(value=0)
    tk.Checkbutton(root, text="All fog densities", variable=fog_density_var, command=update_fog_density_entry).grid(row=1, column=2, padx=10, pady=5)


    # Iteration Number
    tk.Label(root, text="Iteration Number:").grid(row=2, column=0, padx=10, pady=5)
    iteration_nr_entry = tk.Entry(root)
    iteration_nr_entry.grid(row=2, column=1, padx=10, pady=5)
    # insert the iteration number from Joan
    if iteration_nr_joan is not None and int(iteration_nr_joan) > 0:
        iteration_nr_entry.insert(0, int(iteration_nr_joan))

    # Control Type
    tk.Label(root, text="Control Type:").grid(row=3, column=0, padx=10, pady=5)
    control_type_var = tk.StringVar(value="none")  # Default value
    tk.Radiobutton(root, text="Power Control (p)", variable=control_type_var, value="p").grid(row=3, column=1, sticky="w", padx=10, pady=2)
    tk.Radiobutton(root, text="Frequency Control (f)", variable=control_type_var, value="f").grid(row=4, column=1, sticky="w", padx=10, pady=2)
    tk.Radiobutton(root, text="Both (pf)", variable=control_type_var, value="pf").grid(row=5, column=1, sticky="w", padx=10, pady=2)

    # Start Experiment Button
    start_button = tk.Button(root, text="Start Experiment", command=lambda: start_experiment(input_data, root))
    start_button.grid(row=6, column=0, columnspan=2, pady=20)

    root.mainloop()

    # Return the collected inputs
    return input_data['experiment_nr'], input_data['fog_density'], input_data['iteration_nr'], input_data['type']


# # Call the function and print the inputs to verify
# experiment_nr, fog_density, iteration_nr, control_type = user_input()
# print(f"Experiment Number: {experiment_nr}")
# print(f"Fog Density: {fog_density}")
# print(f"Iteration Number: {iteration_nr}")
# print(f"Control Type: {control_type}")
