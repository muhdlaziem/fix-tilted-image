from skew_correction.skewer import Skewer
# image show
import cv2

skewer = Skewer(image_url="https://raw.githubusercontent.com/muhdlaziem/fix-tilted-image/main/sample.jpg")

rotated = skewer.get_rotated()
original = cv2.imread("sample.jpg")

if skewer.is_rotated(): # Returns true or false according to any skew operation
    cv2.imshow("Rotated image", rotated)
    cv2.imshow("Original image", original)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
