/*****************************************************************************
*   Face Recognition using Eigenfaces or Fisherfaces
******************************************************************************
*   by ������, 21th Dec 2019
*****************************************************************************/

#include "recognition.h"     // ѵ������ʶ��ϵͳ,��ͼ����ʶ���һ���ˡ�

#include "ImageUtils.h"      // Shervin��OpenCVʵ�ù��ܡ�

//���ռ�����������ʼѵ��
//����ʶ���㷨����������֮һ�������Ǹ���ģ���ȡ�������OpenCV�汾������������v2.4.1��
//��FaceRecognizer.Eigenfaces����Eigenfaces��Ҳ��ΪPCA(Turk and Pentland, 1991)
//��FaceRecognizer.Fisherfaces����Fisherfaces, Ҳ��ΪLDA��Belhumeur���ˣ�1997��
//��FaceRecognizer.LBPH�����ֲ�������ģʽֱ��ͼ��Ahonen���ˣ�2006��

Ptr<FaceRecognizer> learnCollectedFaces(const vector<Mat> preprocessedFaces, const vector<int> faceLabels, const string facerecAlgorithm)
{
    Ptr<FaceRecognizer> model;

    cout << "Learning the collected faces using the [" << facerecAlgorithm << "] algorithm ..." << endl;

    // ȷ��������ʱ"contrib"ģ�鱻��̬����
    // ��ҪOpenCVv2.4.1����߰汾����2012��6���𣩣�����FaceRecognizer���޷���������У�
    bool haveContribModule = initModule_contrib();
    if (!haveContribModule) {
        cerr << "ERROR: The 'contrib' module is needed for FaceRecognizer but has not been loaded into OpenCV!" << endl;
        exit(1);
    }

    // ��OpenCV��"contrib"ģ����ʹ���µ�FaceRecognizer�ࣺ
    // ��ҪOpenCVv2.4.1����߰汾����2012��6���𣩣�����FaceRecognizer���޷���������У�
    model = Algorithm::create<FaceRecognizer>(facerecAlgorithm);
    if (model.empty()) {
        cerr << "ERROR: The FaceRecognizer algorithm [" << facerecAlgorithm << "] is not available in your version of OpenCV. Please update to OpenCV v2.4.1 or newer." << endl;
        exit(1);
    }

    // ���ռ����������Ͻ���ʵ��ѵ����������Ҫ�����ӻ򼸷��ӣ�����ȡ�������룡
    model->train(preprocessedFaces, faceLabels);
	cout << "train successfully" << endl;

    return model;
}

// �������л��У��������ת��Ϊ����ʾ�򱣴��8λ����ͼ��
// ��ֵ���ŵ�0��255֮��
Mat getImageFrom1DFloatMat(const Mat matrixRow, int height)
{
    // ʹ���Ϊ����ͼ������ǵ���
    Mat rectangularMat = matrixRow.reshape(1, height);
    // Scale the values to be between 0 to 255 and store them as a regular 8-bit uchar image.
    Mat dst;
    normalize(rectangularMat, dst, 0, 255, NORM_MINMAX, CV_8UC1);
    return dst;
}

// ��ʾ�ڲ�����ʶ������,��������
void showTrainingDebugData(const Ptr<FaceRecognizer> model, const int faceWidth, const int faceHeight)
{
    try {   // ��try/catch���ΧOpenCV���ã��Ա���ĳЩģ�Ͳ���������ʱ�������

        // ��ʾƽ���������ռ���ͼ����ÿ�����ص�ͳ��ƽ��ֵ��
        Mat averageFaceRow = model->get<Mat>("mean");
        printMatInfo(averageFaceRow, "averageFaceRow");
        // Convert the matrix row (1D float matrix) to a regular 8-bit image.
        Mat averageFace = getImageFrom1DFloatMat(averageFaceRow, faceHeight);
        printMatInfo(averageFace, "averageFace");
        imshow("averageFace", averageFace);

        // �õ�����������
        Mat eigenvectors = model->get<Mat>("eigenvectors");
        printMatInfo(eigenvectors, "eigenvectors");

        // չ����õ�20����
        for (int i = 0; i < min(20, eigenvectors.cols); i++) {
            // ���������� #i������������
            // ��ע�⣬clone()ȷ�����������ģ�������ǿ��Խ�����Ϊ���飬���������޷���������Ϊ���Ρ�
            // ע�⣬FaceRecognizer���Ѿ�Ϊ�����ṩ��L2�淶�������������������ǲ����Լ��淶�����ǡ�
            Mat eigenvectorColumn = eigenvectors.col(i).clone();
            //printMatInfo(eigenvectorColumn, "eigenvector");

            Mat eigenface = getImageFrom1DFloatMat(eigenvectorColumn, faceHeight);
            //printMatInfo(eigenface, "eigenface");
            imshow(format("Eigenface%d", i), eigenface);
        }

        // �������ֵ
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
