import cv2
import numpy as np
import pytesseract
import re

#Function to detect yellow rectangles and extract text
def detect_yellow_rectangles_and_text(frame, lower_yellow, upper_yellow):
    #Convert the frame to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #Create a mask for the yellow color
    mask = cv2.inRange(hsv_frame, lower_yellow, upper_yellow)

    #Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #Loop through contours to find rectangles
    for contour in contours:
        #Approximate the contour to get a polygon
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)

        #If the polygon has 4 vertices, it's likely a rectangle
        if len(approx) == 4:
            #Get the bounding box (x, y, width, height) of the rectangle
            x, y, w, h = cv2.boundingRect(approx)

            #Draw the rectangle (optional, for visualization)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            #Extract the region inside the rectangle
            roi = frame[y:y + h, x:x + w]

            #Apply OCR to the region to detect text
            text = pytesseract.image_to_string(roi)
            print(text)


            pattern = r"(^[A-Z]{2}[0-9]{2}[A-Z]{3}$)|(?P<Prefix>^[A-Z][0-9]{1,3}[A-Z]{3}$)|(?P<Suffix>^[A-Z]{3}[0-9]{1,3}[A-Z]$)|(?P<DatelessLongNumberPrefix>^[0-9]{1,4}[A-Z]{1,2}$)|(?P<DatelessShortNumberPrefix>^[0-9]{1,3}[A-Z]{1,3}$)|(?P<DatelessLongNumberSuffix>^[A-Z]{1,2}[0-9]{1,4}$)|(?P<DatelessShortNumberSuffix>^[A-Z]{1,3}[0-9]{1,3}$)|(?P<DatelessNorthernIreland>^[A-Z]{1,3}[0-9]{1,4}$)|(?P<DiplomaticPlate>^[0-9]{3}[DX]{1}[0-9]{3}$)"
            result = re.search(pattern, text.strip())
            #Print the detected text in the terminal
            if result != None:
                print(f"Detected Text: {text.strip()}")

            cv2.putText(frame, text.strip(), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return frame, mask

#Open the camera
cap = cv2.VideoCapture(0)

#Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

#Define the lower and upper bounds for yellow color in HSV
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])

while True:
    #Capture
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to grab frame")
        break

    #Call the function to detect yellow rectangles and extract text
    result_frame, mask = detect_yellow_rectangles_and_text(frame, lower_yellow, upper_yellow)

    #Display the result frame with detected rectangles and text
    cv2.imshow('Frame', result_frame)

    cv2.imshow('Mask', mask)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
