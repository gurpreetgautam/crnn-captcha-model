import random
import string
import os
from PIL import Image, ImageDraw, ImageFont

WIDTH = 200
HEIGHT = 50
FONT_SIZE = 40
CHAR_SET = string.ascii_letters + string.digits

FONT_PATH = "/home/gautam/rcnn captcha/fonts/ARIAL.TTF"

# TEXT GENERATION
def generate_text(length=6):
    return ''.join(random.choices(CHAR_SET,k=length))

# =============================================================
# STATE PORTAL CAPTCHA GENERATOR
# =============================================================

def draw_text(draw, text):
    font_size = FONT_SIZE
    font = ImageFont.truetype(FONT_PATH, font_size)

    # Adjust font size to fit width
    total_char_width = sum(int(font.getlength(c)) for c in text)
    while total_char_width > WIDTH - 20 and font_size > 8:
        font_size -= 1
        font = ImageFont.truetype(FONT_PATH, font_size)
        total_char_width = sum(int(font.getlength(c)) for c in text)
    n = len(text)
    # Remaining space to distribute
    remaining_space = WIDTH - total_char_width
    # spacing between characters
    if n > 1:
        spacing = remaining_space // (n + 1)
    else:
        spacing = remaining_space // 2
    x_offset = spacing
    for char in text:
        char_width = int(font.getlength(char))
        char_img = Image.new("RGBA", (char_width + 10, HEIGHT), (255, 255, 255, 0))
        char_draw = ImageDraw.Draw(char_img)
        y_pos = (HEIGHT - font_size) // 2
        char_draw.text((5, y_pos), char, font=font, fill=(0, 0, 0))
        draw._image.paste(char_img, (int(x_offset), 0), char_img)

        # move forward with uniform spacing
        x_offset += char_width + spacing

def add_squares(draw, count=None):
    if count is None:
        count = random.randint(25, 35)
    for _ in range(count):
        side = random.randint(2, 7) 
        x = random.randint(0, WIDTH - side)
        y = random.randint(0, HEIGHT - side)
        draw.rectangle([x, y, x + side, y + side], fill=(0, 0, 0))

def add_circles(draw,color,radius,count=None):
    if count is None:
        count = random.randint(10, 20)
    for _ in range(count):
        x=random.randint(0,WIDTH)
        y=random.randint(0,HEIGHT)
        draw.ellipse([x-radius,y-radius,x+radius,y+radius],fill=color)

def add_lines(draw,count=2):
    for _ in range(count):
        x1=random.randint(0,WIDTH)
        y1=random.randint(0,HEIGHT)
        x2=random.randint(0,WIDTH)
        y2=random.randint(0,HEIGHT)
        color=tuple(random.randint(0,255) for _ in range(3))
        thickness=1
        draw.line([x1,y1,x2,y2],fill=color,width=thickness)

def state_portal_captcha_generator():
    img=Image.new("RGB",(WIDTH,HEIGHT),(255,255,255))
    draw=ImageDraw.Draw(img)
    text=generate_text()
    add_circles(draw,color=(0,0,255),radius=random.uniform(0.3,0.5),count=random.randint(30,80))
    draw_text(draw,text)
    add_circles(draw,color=(0,0,0),radius=random.randint(1,4),count=random.randint(10,20))
    add_squares(draw)
    add_lines(draw)
    return img,text


# =============================================================
# CPP CAPTCHA GENERATOR
# =============================================================

FONT_PATHS = [os.path.join("/home/gautam/rcnn captcha/fonts",x) for x in os.listdir("/home/gautam/rcnn captcha/fonts")]

def random_light_bg():
    base=random.randint(230,255)
    return (base,base+random.randint(-5,5),base)

def random_color():
    return (
        random.randint(0,120),
        random.randint(0,120),
        random.randint(0,120)
    )

def draw_character(img, char, x, y, font_size_override=None):
    font_path = random.choice(FONT_PATHS)
    fs = font_size_override or FONT_SIZE
    font = ImageFont.truetype(font_path, fs)

    pad = 2
    cell_w = fs + pad * 2
    cell_h = HEIGHT  # full image height — vertical cropping impossible

    char_img = Image.new("RGBA", (cell_w, cell_h), (0, 0, 0, 0))
    d = ImageDraw.Draw(char_img)
    color = random_color()

    # Center character vertically within the full-height cell + slight jitter
    y_in_cell = (cell_h - fs) // 2 + random.randint(-3, 3)
    y_in_cell = max(pad, min(y_in_cell, cell_h - fs - pad))

    if random.random() < 0.5:
        for dx in range(2):
            for dy in range(2):
                d.text((pad + dx, y_in_cell + dy), char, font=font, fill=color)
    else:
        d.text((pad, y_in_cell), char, font=font, fill=color)

    angle = random.uniform(-15, 15)
    char_img = char_img.rotate(angle, expand=False)  # fixed size kept

    x = max(0, min(x, WIDTH - cell_w))
    img.paste(char_img, (x, 0), char_img)  # always paste at y=0
    return cell_w

def add_noise(draw,color,count=None):
    if count is None:
        count = random.randint(20, 30)
    for _ in range(count):
        x=random.randint(0,WIDTH)
        y=random.randint(0,HEIGHT)
        draw.point((x,y),fill=color)

# GENERATOR
def cpp_captcha_generator():
    bg_color = random_light_bg()
    img = Image.new("RGB", (WIDTH, HEIGHT), bg_color)
    draw = ImageDraw.Draw(img)

    text = generate_text()

    for _ in range(random.randint(3, 7)):
        color = random_color()
        add_noise(draw, color=color, count=150)

    fs = FONT_SIZE
    target_width = int(WIDTH * 0.90)
    pad = 2
    step = target_width // len(text)
    fs = max(30, min(step - pad*2 - 2, HEIGHT - 8)) # font size tied to step, capped by height
    x_offset = max(2, (WIDTH - step * len(text)) // 2)

    for char in text:
        draw_character(img, char, x_offset, 0, font_size_override=fs)
        x_offset += step

    add_noise(draw, color=bg_color, count=100)

    return img, text


# =============================================================
# SAVE
# =============================================================

STATE_DIR = "captcha_dataset/state_portal_captchas"
os.makedirs(STATE_DIR, exist_ok=True)
CPPP_DIR = "captcha_dataset/cpp_captchas"
os.makedirs(CPPP_DIR, exist_ok=True)

NUM_IMAGES = 3

for i in range(NUM_IMAGES):
    image_state, text_state = state_portal_captcha_generator()
    image_state.save(os.path.join(STATE_DIR, f"{text_state}.png"))
    print(f"Saved CAPTCHA (state) [{i+1}]: {text_state}.png")

    image_cpp, text_cpp = cpp_captcha_generator()
    image_cpp.save(os.path.join(CPPP_DIR, f"{text_cpp}.png"))
    print(f"Saved CAPTCHA (cpp): {text_cpp}.png")

    print("-" * 30)