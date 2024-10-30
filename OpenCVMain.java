import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import org.opencv.highgui.HighGui;

import java.util.ArrayList;
import java.util.List;

public class OpenCVMain {
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static void main(String[] args) {
        // Open a connection to the default webcam (camera index 0)
        VideoCapture cap = new VideoCapture(0);

        if (!cap.isOpened()) {
            System.out.println("Error: Could not open the webcam!");
            return;
        }

        Mat frame = new Mat();

        while (true) {
            // Capture the frame from the webcam
            cap.read(frame);

            if (frame.empty()) {
                System.out.println("Error: Could not capture frame!");
                break;
            }

            // Convert the frame to HSV for easier color detection
            Mat hsvFrame = new Mat();
            Imgproc.cvtColor(frame, hsvFrame, Imgproc.COLOR_BGR2HSV);

            // Create binary masks for red, blue, and yellow
            Mat redMask = new Mat();
            Mat blueMask = new Mat();
            Mat yellowMask = new Mat();

            // HSV color ranges for red, blue, and yellow
            Scalar lowerRed1 = new Scalar(0, 170, 170);    // Reduced red sensitivity
            Scalar upperRed1 = new Scalar(10, 255, 255);
            Scalar lowerRed2 = new Scalar(170, 170, 170);
            Scalar upperRed2 = new Scalar(180, 255, 255);

            Scalar lowerBlue = new Scalar(100, 150, 0);
            Scalar upperBlue = new Scalar(140, 255, 255);

            Scalar lowerYellow = new Scalar(20, 100, 100);
            Scalar upperYellow = new Scalar(30, 255, 255);

            // Detect red (two ranges to cover red properly)
            Mat redMask1 = new Mat();
            Mat redMask2 = new Mat();
            Core.inRange(hsvFrame, lowerRed1, upperRed1, redMask1);  // First red range
            Core.inRange(hsvFrame, lowerRed2, upperRed2, redMask2);  // Second red range
            Core.add(redMask1, redMask2, redMask);                   // Combine both ranges

            // Detect blue and yellow
            Core.inRange(hsvFrame, lowerBlue, upperBlue, blueMask);
            Core.inRange(hsvFrame, lowerYellow, upperYellow, yellowMask);

            // Find contours for red, blue, and yellow areas
            List<MatOfPoint> redContours = new ArrayList<>();
            List<MatOfPoint> blueContours = new ArrayList<>();
            List<MatOfPoint> yellowContours = new ArrayList<>();

            Imgproc.findContours(redMask, redContours, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
            Imgproc.findContours(blueMask, blueContours, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
            Imgproc.findContours(yellowMask, yellowContours, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

            // Draw contours (borders) around detected areas
            Imgproc.drawContours(frame, redContours, -1, new Scalar(255, 0, 255), 2);  // Purple for red areas
            Imgproc.drawContours(frame, blueContours, -1, new Scalar(0, 255, 0), 2);   // Green for blue areas
            Imgproc.drawContours(frame, yellowContours, -1, new Scalar(0, 255, 255), 2); // Yellow for yellow areas

            // Calculate and print the area covered by each color
            double redArea = 0;
            double blueArea = 0;
            double yellowArea = 0;

            // Sum the area of all red contours
            for (MatOfPoint contour : redContours) {
                redArea += Imgproc.contourArea(contour);
            }

            // Sum the area of all blue contours
            for (MatOfPoint contour : blueContours) {
                blueArea += Imgproc.contourArea(contour);
            }

            // Sum the area of all yellow contours
            for (MatOfPoint contour : yellowContours) {
                yellowArea += Imgproc.contourArea(contour);
            }

            // Print the areas to the console
            System.out.println("Red Area: " + redArea);
            System.out.println("Blue Area: " + blueArea);
            System.out.println("Yellow Area: " + yellowArea);

            // Display the original frame with borders
            HighGui.imshow("Webcam Feed - Red, Blue, Yellow Areas with Borders", frame);

            // Break the loop if 'q' is pressed
            if (HighGui.waitKey(30) == 'q') {
                break;
            }
        }

        // Release the webcam and close windows
        cap.release();
        HighGui.destroyAllWindows();
    }
}
