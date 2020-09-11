/*****************************************************************************
*   ��using Eigenfaces �� Fisherfaces��������ʶ��
*****************************************************************************/

//�������һ��Rect����rect(100, 50, 50, 100)����ôrect�������¼������ܣ�
//rect.area();     ����rect����� 5000
//rect.size();     ����rect�ĳߴ� [50 �� 100]
 //rect.tl();      ����rect�����϶�������� [100, 50]
//rect.br();       ����rect�����¶�������� [150, 150]
 //rect.width();   ����rect�Ŀ�� 50
//rect.height();   ����rect�ĸ߶� 100
//rect.contains(Point(x, y));  ���ز����������ж�rect�Ƿ����Point(x, y)��


#include "detectObject.h"       // ���������沿���۾���ʹ��LBP��Haar������


// Search for objects such as faces in the image using the given parameters, storing the multiple cv::Rects into 'objects'.
// Can use Haar cascades or LBP cascades for Face Detection, or even eye, mouth, or car detection.
// Input is temporarily shrunk to 'scaledWidth' for much faster detection, since 200 is enough to find faces.
//ʹ�ø�����������ͼ���еĶ������棩�������cv::Rect�洢��"objects"��
//����ʹ��Haar������LBP���������沿��⣬�����۾�����ͻ��������
//������ʱ��С����scaledWidth���Ա����ؼ�⣬��Ϊ200�����ҵ�����
void detectObjectsCustom(const Mat &img, CascadeClassifier &cascade, vector<Rect> &objects, int scaledWidth, int flags, Size minFeatureSize, float searchScaleFactor, int minNeighbors)
{
    // �������ͼ���ǻҶȣ���BGR��BGRA��ɫͼ��ת��Ϊ�Ҷȡ�
    Mat gray;
    if (img.channels() == 3) {//
        cvtColor(img, gray, CV_BGR2GRAY);
    }
    else if (img.channels() == 4) {
        cvtColor(img, gray, CV_BGRA2GRAY);
    }
    else {
        // ֱ�ӷ�������ͼ����Ϊ���Ѿ��ǻҶȵ�
        gray = img;
    }

    // ���ܻ���Сͼ��,�Ը��������
    Mat inputImg;
    float scale = img.cols / (float)scaledWidth;
    if (img.cols > scaledWidth) {
        // Shrink the image while keeping the same aspect ratio.
        // ��Сͼ��ͬʱ������ͬ���ݺ�ȡ�
        int scaledHeight = cvRound(img.rows / scale);
        resize(gray, inputImg, Size(scaledWidth, scaledHeight));
    }
    else {
        // ֱ�ӷ�������ͼ��,��Ϊ���Ѿ���С��
        inputImg = gray;
    }

    // ��׼�����ȺͶԱȶ��Ը��ư�ͼ��
    Mat equalizedImg;
    equalizeHist(inputImg, equalizedImg);

    // ���С�Ҷ�ͼ���еĶ���
    cascade.detectMultiScale(equalizedImg, objects, searchScaleFactor, minNeighbors, flags, minFeatureSize);

    // ����ڼ��֮ǰͼ����ʱ��С,��Ŵ���
    if (img.cols > scaledWidth) {
        for (int i = 0; i < (int)objects.size(); i++ ) {
            objects[i].x = cvRound(objects[i].x * scale);
            objects[i].y = cvRound(objects[i].y * scale);
            objects[i].width = cvRound(objects[i].width * scale);
            objects[i].height = cvRound(objects[i].height * scale);
        }
    }

    // ȷ��������ȫ��ͼ����,�Է����ڱ߽���
    for (int i = 0; i < (int)objects.size(); i++ ) {
        if (objects[i].x < 0)
            objects[i].x = 0;
        if (objects[i].y < 0)
            objects[i].y = 0;
        if (objects[i].x + objects[i].width > img.cols)
            objects[i].x = img.cols - objects[i].width;
        if (objects[i].y + objects[i].height > img.rows)
            objects[i].y = img.rows - objects[i].height;
    }

    // ����ʱ����⵽���������δ洢��"����"��
}

//ֻ����ͼ���е�һ������������������,������洢��'largestObject'��
//����ʹ��Haar������LBP���������沿��⣬�����۾�����ͻ��������
//������ʱ��С��"scaledWidth"�Ա����ؼ�⣬��Ϊ240�����ҵ�����
//ע�⣺detectLargestObject()Ӧ�ñ�detectManyObjects()��
void detectLargestObject(const Mat &img, CascadeClassifier &cascade, Rect &largestObject, int scaledWidth)
{
    // ֻ����һ������(ͼ��������)
    int flags = CASCADE_FIND_BIGGEST_OBJECT;// | CASCADE_DO_ROUGH_SEARCH;
    // ��С�����С
    Size minFeatureSize = Size(20, 20);
    // How detailed should the search be. Must be larger than 1.0.
    // ����Ӧ���ж���ϸ.�������1.0
    float searchScaleFactor = 1.1f;
    // How much the detections should be filtered out. This should depend on how bad false detections are to your system.
    // minNeighbors=2 means lots of good+bad detections, and minNeighbors=6 means only good detections are given but some are missed.
    // Ӧ�ù��˵����ټ��������Ӧ��ȡ���ڴ������ϵͳ��Ӱ��
    // minNeighbors=2��ʾ�кܶ� �õ�+���� ���,minNeighbors=6 ��ʾֻ�кõļ�ⱻ����,����Щ����©��
    int minNeighbors = 4;

    // ִ��Ŀ����������,ֻ����1��Ŀ��(ͼ��������)
    vector<Rect> objects;
    detectObjectsCustom(img, cascade, objects, scaledWidth, flags, minFeatureSize, searchScaleFactor, minNeighbors);
    if (objects.size() > 0) {
        // ����Ψһ��⵽�Ķ���
        largestObject = (Rect)objects.at(0);
    }
    else {
        // ����һ����Ч��rect
        largestObject = Rect(-1,-1,-1,-1);
    }
}

//����ͼ���е������������е�������������洢��"objects"��
//����ʹ��Haar������LBP���������沿��⣬�����۾�����ͻ��������
//������ʱ��С��"scaledWidth"�Ա����ؼ�⣬��Ϊ240�����ҵ�����
//ע�⣺detectLargestObject()Ӧ�ñ�detectManyObjects()��
void detectManyObjects(const Mat &img, CascadeClassifier &cascade, vector<Rect> &objects, int scaledWidth)
{
    // Search for many objects in the one image.
    int flags = CASCADE_SCALE_IMAGE;

    //��С����ĳߴ�
    Size minFeatureSize = Size(20, 20);
    // ����Ӧ���ж���ϸ.�������1.0
    float searchScaleFactor = 1.1f;
    // Ӧ�ù��˵����ټ��������Ӧ��ȡ���ڴ������ϵͳ��Ӱ��
    // minNeighbors=2 means lots of good+bad detections, and minNeighbors=6 means only good detections are given but some are missed.
    // minNeighbors=2��ʾ�кܶ� �õ�+���� ���,minNeighbors=6 ��ʾֻ�кõļ�ⱻ����,����Щ����©��
    int minNeighbors = 4;

    // ִ�ж�����������,��һ��ͼ���в��Ҷ������
    detectObjectsCustom(img, cascade, objects, scaledWidth, flags, minFeatureSize, searchScaleFactor, minNeighbors);
}
