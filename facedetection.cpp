#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main() {
    // Open the default camera
    VideoCapture camera(0); // 0 for default camera, you can change it if you have multiple cameras

    // Check if camera opened successfully
    if (!camera.isOpened()) {
        cout << "Error: Unable to open camera." << endl;
        return -1;
    }

    // Load the pre-trained face detection classifier
    CascadeClassifier faceCascade;
    if (!faceCascade.load("haarcascade_frontalface_default.xml")) {
        cout << "Error: Unable to load face detection classifier." << endl;
        return -1;
    }

    // Capture frames from the camera
    Mat frame;
    while (true) {
        // Read a frame from the camera
        camera.read(frame);

        // Check if the frame is empty
        if (frame.empty()) {
            cout << "Error: Unable to read frame from camera." << endl;
            break;
        }

        // Convert the frame to grayscale
        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        // Detect faces in the frame
        vector<Rect> faces;
        faceCascade.detectMultiScale(gray, faces, 1.1, 4);

        // Draw rectangles around the detected faces
        for (const Rect& face : faces) {
            rectangle(frame, face, Scalar(255, 135, 0), 2);
        }

        // Display the frame with detected faces
        imshow("Face Detection", frame);
        // putText(frame, "Face", Point(face.x, face.y - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0), 2);
        

        // Wait for 1ms and check if the user pressed ESC key
        if (waitKey(1) == 27) {
            cout << "ESC key is pressed by the user. Exiting." << endl;
            break;
        }
    }

    // Release the camera
    camera.release();

    return 0;
}
