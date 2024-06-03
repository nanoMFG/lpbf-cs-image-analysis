from . import *
import cv2 as cv
import argparse
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser("magic wand mask generator")
    parser.add_argument("image", help="path to image")
    parser.add_argument("-ns", "--nosave", action="store_false", help="toggle saving the mask (default = True)")
    parser.add_argument("-fh", "--fillholes", action="store_false", help="toggle filling holes in the mask (default = True)")
    args = parser.parse_args()

    img = cv.imread(args.image)
    if img is None or img.size == 0:
        raise Exception(f"Unable to read image {args.image}. Please check the path.")

    window = SelectionWindow(img, "Magic Wand Selector")

    print("Left click to seed a selection.")
    print("Drag to continue floodfill.")
    print("Right click to start painting circles.")
    print("Drag to continue painting.")
    print(" * [SHIFT] subtracts from the selection.")
    print(" * [ALT] starts a new selection with a new seed.")
    print(" * [SHIFT] + [ALT] intersects the selections.")
    print("Adjust sliders to change tolerance of floodfill and radius of paiting.")
    print()

    window.show()

    output = window.mask
    if args.fillholes:
        contour,hier = cv.findContours(output,cv.RETR_CCOMP,cv.CHAIN_APPROX_SIMPLE)
        for cnt in contour:
            cv.drawContours(output,[cnt],0,255,-1)
    print(f"Save: {args.nosave}")
    if args.nosave:
        output_name = os.path.splitext(args.image)[0]+'_mask.png'
        cv.imwrite(output_name, output)
        print(f"Successfully saved mask. Save location: {output_name}")

    cv.destroyAllWindows()
