/*****************************************************************************
*   Face Recognition using Eigenfaces or Fisherfaces
******************************************************************************
*   by 龚云鹏, 21th Dec 2019
*****************************************************************************/

#include "recognition.h"     // 训练人脸识别系统,从图像中识别出一个人。

#include "ImageUtils.h"      // Shervin的OpenCV实用功能。

//从收集到的人脸开始训练
//人脸识别算法可以是下列之一，或者是更多的，这取决于你的OpenCV版本，必须至少是v2.4.1：
//“FaceRecognizer.Eigenfaces”：Eigenfaces，也称为PCA(Turk and Pentland, 1991)
//“FaceRecognizer.Fisherfaces”：Fisherfaces, 也称为LDA（Belhumeur等人，1997）
//“FaceRecognizer.LBPH”：局部二进制模式直方图（Ahonen等人，2006）

Ptr<FaceRecognizer> learnCollectedFaces(const vector<Mat> preprocessedFaces, const vector<int> faceLabels, const string facerecAlgorithm)
{
    Ptr<FaceRecognizer> model;

    cout << "Learning the collected faces using the [" << facerecAlgorithm << "] algorithm ..." << endl;

    // 确保在运行时"contrib"模块被动态加载
    // 需要OpenCVv2.4.1或更高版本（从2012年6月起），否则FaceRecognizer将无法编译或运行！
    bool haveContribModule = initModule_contrib();
    if (!haveContribModule) {
        cerr << "ERROR: The 'contrib' module is needed for FaceRecognizer but has not been loaded into OpenCV!" << endl;
        exit(1);
    }

    // 在OpenCV的"contrib"模块中使用新的FaceRecognizer类：
    // 需要OpenCVv2.4.1或更高版本（从2012年6月起），否则FaceRecognizer将无法编译或运行！
    model = Algorithm::create<FaceRecognizer>(facerecAlgorithm);
    if (model.empty()) {
        cerr << "ERROR: The FaceRecognizer algorithm [" << facerecAlgorithm << "] is not available in your version of OpenCV. Please update to OpenCV v2.4.1 or newer." << endl;
        exit(1);
    }

    // 从收集到的人脸上进行实际训练。可能需要几秒钟或几分钟，具体取决于输入！
    model->train(preprocessedFaces, faceLabels);
	cout << "train successfully" << endl;

    return model;
}

// 将矩阵行或列（浮点矩阵）转换为可显示或保存的8位矩形图像
// 将值缩放到0到255之间
Mat getImageFrom1DFloatMat(const Mat matrixRow, int height)
{
    // 使其成为矩形图像而不是单行
    Mat rectangularMat = matrixRow.reshape(1, height);
    // Scale the values to be between 0 to 255 and store them as a regular 8-bit uchar image.
    Mat dst;
    normalize(rectangularMat, dst, 0, 255, NORM_MINMAX, CV_8UC1);
    return dst;
}

// 显示内部人脸识别数据,帮助调试
void showTrainingDebugData(const Ptr<FaceRecognizer> model, const int faceWidth, const int faceHeight)
{
    try {   // 用try/catch块包围OpenCV调用，以便在某些模型参数不可用时不会崩溃

        // 显示平均人脸（收集的图像中每个像素的统计平均值）
        Mat averageFaceRow = model->get<Mat>("mean");
        printMatInfo(averageFaceRow, "averageFaceRow");
        // Convert the matrix row (1D float matrix) to a regular 8-bit image.
        Mat averageFace = getImageFrom1DFloatMat(averageFaceRow, faceHeight);
        printMatInfo(averageFace, "averageFace");
        imshow("averageFace", averageFace);

        // 得到特征向量。
        Mat eigenvectors = model->get<Mat>("eigenvectors");
        printMatInfo(eigenvectors, "eigenvectors");

        // 展现最好的20张脸
        for (int i = 0; i < min(20, eigenvectors.cols); i++) {
            // 从特征向量 #i创建列向量。
            // 请注意，clone()确保它是连续的，因此我们可以将其视为数组，否则我们无法将其重塑为矩形。
            // 注意，FaceRecognizer类已经为我们提供了L2规范化特征向量，所以我们不必自己规范化它们。
            Mat eigenvectorColumn = eigenvectors.col(i).clone();
            //printMatInfo(eigenvectorColumn, "eigenvector");

            Mat eigenface = getImageFrom1DFloatMat(eigenvectorColumn, faceHeight);
            //printMatInfo(eigenface, "eigenface");
            imshow(format("Eigenface%d", i), eigenface);
        }

        // 获得特征值
        Mat eigenvalues = model->get<Mat>("eigenvalues");
        printMat(eigenvalues, "eigenvalues");

        //int ncomponents = model->get<int>("ncomponents");
        //cout << "ncomponents = " << ncomponents << endl;

        vector<Mat> projections = model->get<vector<Mat> >("projections");
        cout << "projections: " << projections.size() << endl;
        for (int i = 0; i < (int)projections.size(); i++) {
            printMat(projections[i], "projections");
        }

        //labels = model->get<Mat>("labels");
        //printMat(labels, "labels");

    } catch (cv::Exception e) {
        //cout << "WARNING: Missing FaceRecognizer properties." << endl;
    }

}


// Generate an approximately reconstructed face by back-projecting the eigenvectors & eigenvalues of the given (preprocessed) face.
Mat reconstructFace(const Ptr<FaceRecognizer> model, const Mat preprocessedFace)
{
    // Since we can only reconstruct the face for some types of FaceRecognizer models (ie: Eigenfaces or Fisherfaces),
    // we should surround the OpenCV calls by a try/catch block so we don't crash for other models.
    try {

        // Get some required data from the FaceRecognizer model.
        Mat eigenvectors = model->get<Mat>("eigenvectors");
        Mat averageFaceRow = model->get<Mat>("mean");

        int faceHeight = preprocessedFace.rows;

        // Project the input image onto the PCA subspace.
        Mat projection = subspaceProject(eigenvectors, averageFaceRow, preprocessedFace.reshape(1,1));
        //printMatInfo(projection, "projection");

        // Generate the reconstructed face back from the PCA subspace.
        Mat reconstructionRow = subspaceReconstruct(eigenvectors, averageFaceRow, projection);
        //printMatInfo(reconstructionRow, "reconstructionRow");

        // Convert the float row matrix to a regular 8-bit image. Note that we
        // shouldn't use "getImageFrom1DFloatMat()" because we don't want to normalize
        // the data since it is already at the perfect scale.

        // Make it a rectangular shaped image instead of a single row.
        Mat reconstructionMat = reconstructionRow.reshape(1, faceHeight);
        // Convert the floating-point pixels to regular 8-bit uchar pixels.
        Mat reconstructedFace = Mat(reconstructionMat.size(), CV_8U);
        reconstructionMat.convertTo(reconstructedFace, CV_8U, 1, 0);
        //printMatInfo(reconstructedFace, "reconstructedFace");

        return reconstructedFace;

    } catch (cv::Exception e) {
        //cout << "WARNING: Missing FaceRecognizer properties." << endl;
        return Mat();
    }
}


// Compare two images by getting the L2 error (square-root of sum of squared error).
double getSimilarity(const Mat A, const Mat B)
{
    if (A.rows > 0 && A.rows == B.rows && A.cols > 0 && A.cols == B.cols) {
        // Calculate the L2 relative error between the 2 images.
        double errorL2 = norm(A, B, CV_L2);
        // Convert to a reasonable scale, since L2 error is summed across all pixels of the image.
        double similarity = errorL2 / (double)(A.rows * A.cols);
        return similarity;
    }
    else {
        //cout << "WARNING: Images have a different size in 'getSimilarity()'." << endl;
        return 100000000.0;  // Return a bad value
    }
}
