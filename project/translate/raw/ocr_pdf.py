import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = 'tesseract.exe'
text = pytesseract.image_to_string(Image.open('./1.jpg'))

print(text)