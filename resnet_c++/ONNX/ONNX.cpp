#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn/dnn.hpp>

using namespace std;


// CIFAR-10 classes
std::string CLASSES[10] = { "plane", "car", "bird", "cat", "deer", 
                            "dog", "frog", "horse", "ship", "truck" };


int getLabel(const cv::Mat& output) {
    double minVal = 0., maxVal = 0.;
    int minLoc, maxLoc;

    cv::minMaxIdx(output, &minVal, &maxVal, &minLoc, &maxLoc);

    return maxLoc;
}

int main()
{
    // Get a deep learning model from ONNX file
    cv::Mat image = cv::imread(".\\data\\airplane1.png");

    // Read a test image
    cv::dnn::Net model = cv::dnn::readNetFromONNX(".\\models\\resnet18_cifar10.onnx");
    
    // Convert an image to a blob
    cv::Mat input;
    input = cv::dnn::blobFromImage(image, 1 / 255., cv::Size(32, 32));

    // Predict the label corresponding to the image
    cv::Mat output;
    string pred;

    model.setInput(input);
    output = model.forward();
    pred = CLASSES[getLabel(output)];

    cout << "Actual: plane" << '\n';
    cout << "Predict: " << pred << '\n';

    return 0;
}