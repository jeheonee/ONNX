#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn/dnn.hpp>

using namespace std;


// CIFAR-10 classes
std::string CLASSES[10] = { "plane", "car", "bird", "cat", "deer", 
                            "dog", "frog", "horse", "ship", "truck" };


string getClasses(const cv::Mat& output) {
    double minVal = 0., maxVal = 0.;
    int minLoc, maxLoc;

    cv::minMaxIdx(output, &minVal, &maxVal, &minLoc, &maxLoc);

    return CLASSES[maxLoc];
}

void evaluateClassifier() {
    // Get a classifier from ONNX file
    cv::dnn::Net model = cv::dnn::readNetFromONNX(".\\models\\resnet18.onnx");

    // Read a test image
    cv::Mat image = cv::imread(".\\data\\airplane1.png");

    // Convert an image to a blob
    cv::Mat input;
    input = cv::dnn::blobFromImage(image, 1 / 255., image.size());

    // Predict the label corresponding to the image
    cv::Mat output;
    string pred;

    model.setInput(input);
    output = model.forward();
    pred = getClasses(output);

    cout << "Actual: plane" << '\n';
    cout << "Predict: " << pred << '\n';
}

void getDetector() {
    // Get a detector from ONNX file
    cv::dnn::Net model = cv::dnn::readNetFromONNX(".\\models\\yolov5s.onnx");

    // Read an image
    cv::Mat image = cv::imread(".\\data\\zidane.jpg");

    // Convert an image to a blob
    cv::Mat input;

    cv::resize(image, input, cv::Size(640, 640));
    input = cv::dnn::blobFromImage(image, 1 / 255., cv::Size(640, 640), cv::Scalar(), true, false);

    // Predict the classes and bounding boxes
    vector<cv::Mat> outputs;

    model.setInput(input);
    model.forward(outputs);

    cout << outputs[0].size() << '\n';
}

int main()
{
    getDetector();

    return 0;
}