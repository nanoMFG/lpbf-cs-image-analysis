import cv2 as cv
import numpy as np


SHIFT_KEY = cv.EVENT_FLAG_SHIFTKEY
ALT_KEY = cv.EVENT_FLAG_ALTKEY


def _find_exterior_contours(img):
    ret = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if len(ret) == 2:
        return ret[0]
    elif len(ret) == 3:
        return ret[1]
    raise Exception("Check the signature for `cv.findContours()`.")


class SelectionWindow:
    def __init__(self, img, name="Magic Wand Selector", connectivity=4, tolerance=20, radius=15):
        self.name = name
        h, w = img.shape[:2]
        self.img = img
        self.mask = np.zeros((h, w), dtype=np.uint8)
        self._flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        self._flood_fill_flags = (
            connectivity | cv.FLOODFILL_FIXED_RANGE | cv.FLOODFILL_MASK_ONLY | 255 << 8
        )  # 255 << 8 tells to fill with the value 255
        cv.namedWindow(self.name)
        self.tolerance = (tolerance,) * 3
        self.radius = radius
        cv.createTrackbar(
            "Tolerance", self.name, tolerance, 255, self._tolerance_trackbar_callback
        )
        cv.createTrackbar(
            "Paint radius", self.name, radius, min(h, w)//2, self._radius_trackbar_callback
        )
        cv.setMouseCallback(self.name, self._mouse_callback)
        self.filling = False
        self.painting = False

    def _tolerance_trackbar_callback(self, pos):
        self.tolerance = (pos,) * 3

    def _radius_trackbar_callback(self, rad):
        self.radius = rad

    def _mouse_callback(self, event, x, y, flags, *userdata):

        if event == cv.EVENT_LBUTTONDOWN: # LEFT CLICK: start filling
            self.filling = True
            self._flood_mask[:] = 0

        elif event == cv.EVENT_RBUTTONDOWN: # RIGHT CLICK: start painting
            self.painting = True
            self._flood_mask[:] = 0

        elif event == cv.EVENT_MOUSEMOVE:
            if self.filling == True:
                cv.floodFill(
                    self.img,
                    self._flood_mask,
                    (x, y),
                    0,
                    self.tolerance,
                    self.tolerance,
                    self._flood_fill_flags,
                )
                flood_mask = self._flood_mask[1:-1, 1:-1].copy()

                modifier = flags & (ALT_KEY + SHIFT_KEY)

                if modifier ==  SHIFT_KEY: # SHIFT+CLICK to erase
                    self.mask = cv.bitwise_and(self.mask, cv.bitwise_not(flood_mask))
                elif modifier == ALT_KEY: # ALT+CLICK to restart
                    self.mask = flood_mask
                else: # LCLICK to continue floodfill
                    self.mask = cv.bitwise_or(self.mask, flood_mask)
                
                self._update()

            if self.painting == True:
                cv.circle(
                    self._flood_mask,
                    (x, y),
                    self.radius,
                    255, 
                    -1,
                )
                flood_mask = self._flood_mask[1:-1, 1:-1].copy()

                modifier = flags & (ALT_KEY + SHIFT_KEY)

                if modifier ==  SHIFT_KEY: # SHIFT+CLICK to erase
                    self.mask = cv.bitwise_and(self.mask, cv.bitwise_not(flood_mask))
                elif modifier == ALT_KEY: # ALT+CLICK to restart
                    self.mask = flood_mask
                elif modifier == (ALT_KEY + SHIFT_KEY): # ALT+SHIFT+CLICK to get intersection
                    self.mask = cv.bitwise_and(self.mask, flood_mask)
                else: # RCLICK to continue painting
                    self.mask = cv.bitwise_or(self.mask, flood_mask)
                
                self._update()
        elif event == cv.EVENT_LBUTTONUP:
            self.filling = False
            cv.floodFill(
                self.img,
                self._flood_mask,
                (x, y),
                0,
                self.tolerance,
                self.tolerance,
                self._flood_fill_flags,
            )
            flood_mask = self._flood_mask[1:-1, 1:-1].copy()

            modifier = flags & (ALT_KEY + SHIFT_KEY)

            if modifier ==  SHIFT_KEY: # SHIFT+CLICK to erase
                self.mask = cv.bitwise_and(self.mask, cv.bitwise_not(flood_mask))
            elif modifier == ALT_KEY: # ALT+CLICK to restart
                self.mask = flood_mask
            elif modifier == (ALT_KEY + SHIFT_KEY): # ALT+SHIFT+CLICK to get intersection
                self.mask = cv.bitwise_and(self.mask, flood_mask)
            else: # LCLICK to continue floodfill
                self.mask = cv.bitwise_or(self.mask, flood_mask)
            self._update()

        elif event == cv.EVENT_RBUTTONUP:
            self.painting = False
            cv.circle(
                self._flood_mask,
                (x, y),
                self.radius,
                255, 
                -1,
            )
            flood_mask = self._flood_mask[1:-1, 1:-1].copy()

            modifier = flags & (ALT_KEY + SHIFT_KEY)

            if modifier ==  SHIFT_KEY: # SHIFT+LCLICK to erase
                self.mask = cv.bitwise_and(self.mask, cv.bitwise_not(flood_mask))
            elif modifier == ALT_KEY: # ALT+LCLICK to restart
                self.mask = flood_mask
            elif modifier == (ALT_KEY + SHIFT_KEY): # ALT+SHIFT+CLICK to get intersection
                self.mask = cv.bitwise_and(self.mask, flood_mask)
            else: # RCLICK to continue painting
                self.mask = cv.bitwise_or(self.mask, flood_mask)
            
            self._update()

        # elif event == cv.EVENT_MBUTTONDOWN: # click Middle button to save mask
        #     des = self.mask
        #     contour,hier = cv.findContours(des,cv.RETR_CCOMP,cv.CHAIN_APPROX_SIMPLE)
        #     for cnt in contour:
        #         cv.drawContours(des,[cnt],0,255,-1)
        #     savename = 'mask'+ ["".join(item) for item in self.mean.T.astype(int).astype(str)][0] +'.png'
        #     cv.imwrite(savename, des)
        #     print(f"Saved mask as `{savename}`")
        else:
            return

    def _update(self):
        """Updates an image in the already drawn window."""
        viz = self.img.copy()
        contours = _find_exterior_contours(self.mask)
        viz = cv.drawContours(viz, contours, -1, color=(255,) * 3, thickness=-1)
        viz = cv.addWeighted(self.img, 0.75, viz, 0.25, 0)
        viz = cv.drawContours(viz, contours, -1, color=(255, 0, 0) , thickness=2)

        self.mean, self.stddev = cv.meanStdDev(self.img, mask=self.mask)
        meanstr = "mean=({:.2f}, {:.2f}, {:.2f})".format(*self.mean[:, 0])
        stdstr = "std=({:.2f}, {:.2f}, {:.2f})".format(*self.stddev[:, 0])
        cv.imshow(self.name, viz)
        # cv.displayStatusBar(self.name, ", ".join((meanstr, stdstr)))

    def show(self):
        """Draws a window with the supplied image."""
        self._update()
        print("Press [q] or [esc] to close the window and save the mask.")
        while True:
            k = cv.waitKey() & 0xFF
            if k in (ord("q"), ord("\x1b")):
                cv.destroyWindow(self.name)

                break
