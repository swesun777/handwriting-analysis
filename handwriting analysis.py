import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import cv2
from PIL import Image, ImageTk

def pixels_to_mm(pixels, dpi=300):
    """
    Convert pixels to millimeters based on the DPI (Dots Per Inch).
    :param pixels: Number of pixels to convert.
    :param dpi: Dots per inch of the image. Default is 300 DPI.
    :return: Equivalent length in millimeters.
    """
    mm_per_inch = 25.4  # 1 inch = 25.4 millimeters
    return (pixels / dpi) * mm_per_inch

def analyze_handwriting(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    _, binary_image = cv2.threshold(blurred_image, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    def calculate_baseline(contours):
        baselines = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            baseline_y = y + h
            cv2.line(image, (x, baseline_y), (x + w, baseline_y), (255, 0, 0), 2)
            baselines.append(baseline_y)
        return sum(baselines) / len(baselines) if baselines else None

    # Assuming `contours` is a list of contours and `image` is the image array.
    avg_baseline = calculate_baseline(contours)

    def calculate_slant(contour):
        (x, y), (MA, ma), angle = cv2.fitEllipse(contour)
        return angle

    slant_angles = [calculate_slant(contour) for contour in contours]
    average_slant = sum(slant_angles) / len(slant_angles)

    bounding_boxes = [cv2.boundingRect(contour) for contour in contours]
    heights = [box[3] for box in bounding_boxes]  # box[3] is the height
    average_height = sum(heights) / len(heights)
    average_height_mm = pixels_to_mm(average_height)

    '''thicknesses = [cv2.contourArea(contour) / cv2.arcLength(contour, True) for contour in contours]
    average_thickness = sum(thicknesses) / len(thicknesses)
    average_thickness_mm = pixels_to_mm(average_thickness)'''

    def calculate_spacing(bounding_boxes):
        spacing = []
        for i in range(1, len(bounding_boxes)):
            previous_box = bounding_boxes[i - 1]
            current_box = bounding_boxes[i]
            spacing.append(current_box[0] - (previous_box[0] + previous_box[2]))
        return spacing

    word_spacing = calculate_spacing(bounding_boxes)
    average_word_spacing = sum(word_spacing) / len(word_spacing)
    average_word_spacing_mm = pixels_to_mm(average_word_spacing)

    return average_slant, average_height_mm, average_word_spacing_mm, avg_baseline
def interpret_spacing(spacing_mm):
    if spacing_mm > 5:
        return "You are a person who values freedom and independence. person who values personal space and independence, likely more reserved or introverted."
    elif spacing_mm < 2:
        return "You are sociable and enjoy being around others. seeks closeness with others. They may also be impulsive or quick to act."
    elif 2 <= spacing_mm <= 5:
        return ('''You are balanced and adaptable. Neither too cautious nor too  impulsive."
                   You have a good sense of when to act and when to hold back.The individual enjoys both company and solitude, maintaining a healthy balance. ''')
    else:
        return "No clear interpretation."

def interpret_slant(slant_angle):
    if slant_angle >= 0 and slant_angle <= 45:
        return "You are an outgoing person who loves expressing their emotions. You are friendly, sociable and enthusiastic.Responsive and interested in others."
    elif slant_angle >45 and slant_angle<= 135:
        return "You are a very balanced person, rational and disciplined. You generally tend to control your emotions. Practical,independant, controlled and self sufficient."
    elif slant_angle > 135 and slant_angle <=180:
        return "You are generally an introverted and reserved person, entirely tending to be self reliant. You are often very cautious or skeptical about things.Tend to be observant,self reliant and non intrusive.These individuals might be more self-reliant and prefer solitude or small, close-knit social circles."
    else:
        return "No clear interpretation."

def interpret_height(height_mm):
    if height_mm >= 7 and height_mm < 10:
        return "Extroverted, outgoing, confident, ambitious, and sociable."
    elif height_mm >= 5 and height_mm <7:
        return " Balanced, practical, and adaptable. Neither too introverted nor extroverted."
    elif height_mm >=2 and height_mm < 5:
        return " Introverted, detail-oriented, and focused. Often prefers solitude and precision."
    else:
        return "No clear interpretation."

def interpret_thickness(thickness):
    if thickness > 0.5:
        return "confident, determined,assertive. They have a strong drive and high levels of energy.  Thick strokes can indicate a person who is emotionally expressive and passionate."
    elif thickness >=0.3 and thickness<0.5:
        return (" People with medium thickness handwriting often have a balanced personality, combining both assertiveness and caution.They tend to be practical, grounded, and well-adjusted to their environment."
                "Their emotional expression is often measured and controlled.")
    elif thickness < 0.3:
        return ("Thin handwriting may suggest a person who is more cautious, introverted, and reserved.These individuals are often meticulous and focus on finer details."
                "They may also be more sensitive or fragile emotionally, preferring to keep a low profile.")
    else:
        return "No clear interpretation."

def interpret_baseline(baseline):
    if baseline >= -2 and baseline <=+2:
        return ("Indicates stability, consistency, and a well-balanced personality.The writer maintains a steady emotional state ")
    elif baseline >= +3 and baseline<=+10:
        return("Reflects optimism, ambition, and a forward-looking attitude. The writer is motivated and has a positive outlook.")
    elif baseline >= -3 and baseline <= -10:
        return("Indicates pessimism, low energy, or potential feelings of discouragement. The writer may be experiencing difficulties or stress.")
    elif baseline >= +15 and baseline <=+30:
        return ("Reflects high ambition, drive, and a strong desire to succeed. However, it may also indicate impatience or overenthusiasm.")
    elif baseline >= -15 and baseline <=-30:
        return ("Suggests significant discouragement, lack of energy, or possible depression. The writer may feel overwhelmed or defeated.")
    elif baseline >=+10 or baseline>=-10:
        return(" Indicates internal conflict, inconsistency, or fluctuating emotions. The writer might struggle with maintaining focus or emotional balance.")
    else:
        return("No clear interpretation")


from transformers import BartTokenizer, BartForConditionalGeneration
import torch

# Load the pre-trained BART model and tokenizer
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

def summarize_text(text, max_length=130, min_length=100):
    # Encode the input text and add summarization-specific tokens
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)

    # Generate the summary using the model
    summary_ids = model.generate(inputs, max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4,
                                 early_stopping=True)

    # Decode the generated summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary



def load_and_analyze():
    filepath = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png")])
    if not filepath:
        return

    try:
        avg_slant, avg_height, avg_spacing, avg_baseline = analyze_handwriting(filepath)
        personality_trait =(interpret_spacing(avg_spacing) + "\n" + interpret_baseline(avg_baseline)+"\n" + interpret_height(avg_height) +  "\n" + interpret_slant(avg_slant)) #the summarized personality trait will be displayed
        summary_trait = summarize_text(personality_trait)
        result_text.set(f"Average Slant Angle: {avg_slant:.2f}°\n"
                        f"Average Height: {avg_height:.2f} mm\n"
                        f"Average Word Spacing: {avg_spacing:.2f} mm\n\n"
                        f"Average Baseline: {avg_baseline:.2f} pixels\n"
                        f"Personality Trait:\n{summary_trait}")

        # Load and display the image
        img = Image.open(filepath)
        img.thumbnail((750,750))  # Resize the image to fit the GUI
        img = ImageTk.PhotoImage(img)
        image_label.config(image=img)
        image_label.image = img

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred while processing the image: {e}")

# Set up the main Tkinter window
root = tk.Tk()
root.title("Inksight Handwriting Analysis")
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.geometry(f"{screen_width}x{screen_height}")
root.attributes("-fullscreen", True)

# Title Label
title_label = tk.Label(root, text="Handwriting Analysis", font=("Arial", 20))
title_label.pack(pady=10)
bg_image = Image.open("sb.png")

bg = ImageTk.PhotoImage(bg_image)

# Step 3: Create a canvas and set the background image
canvas = tk.Canvas(root, width=screen_width, height=screen_height)
canvas.pack(fill="both", expand=True)  # Expand the canvas to fill the window
canvas.create_image(0, 0, image=bg, anchor="nw")
# Button to load and analyze an image
analyze_button = tk.Button(root, text="Load and Analyze Image", command=load_and_analyze, font=("Verdana", 14))
analyze_button.place(x=665, y=100)

# Label to display the image
image_label = tk.Label(root)
image_label.place(x=300, y=200)

# Label to display the analysis results
result_text = tk.StringVar()
result_label = tk.Label(root, textvariable=result_text, font=("Arial", 8), justify=tk.LEFT)
result_label.place(x=90,y=300)

# Start the Tkinter main loop
root.mainloop()