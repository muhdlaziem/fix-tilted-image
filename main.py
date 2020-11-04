from skew_correction.skewer import Skewer
# image show
import cv2

skewer = Skewer(image_url="sample.jpg")

rotated = skewer.get_rotated()

if skewer.is_rotated(): # Returns true or false according to any skew operation
    cv2.imshow("Rotated image", rotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
