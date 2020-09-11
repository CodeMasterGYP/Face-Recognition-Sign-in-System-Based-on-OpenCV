/*****************************************************************************
*   用using Eigenfaces 或 Fisherfaces进行人脸识别
*****************************************************************************/

//如果创建一个Rect对象rect(100, 50, 50, 100)，那么rect会有以下几个功能：
//rect.area();     返回rect的面积 5000
//rect.size();     返回rect的尺寸 [50 × 100]
 //rect.tl();      返回rect的左上顶点的坐标 [100, 50]
//rect.br();       返回rect的右下顶点的坐标 [150, 150]
 //rect.width();   返回rect的宽度 50
//rect.height();   返回rect的高度 100
//rect.contains(Point(x, y));  返回布尔变量，判断rect是否包含Point(x, y)点


#include "detectObject.h"       // 检测对象，如面部或眼睛（使用LBP或Haar级联）


// Search for objects such as faces in the image using the given parameters, storing the multiple cv::Rects into 'objects'.
// Can use Haar cascades or LBP cascades for Face Detection, or even eye, mouth, or car detection.
// Input is temporarily shrunk to 'scaledWidth' for much faster detection, since 200 is enough to find faces.
//使用给定参数搜索图像中的对象（如面），将多个cv::Rect存储到"objects"中
//可以使用Haar级联或LBP级联进行面部检测，甚至眼睛、嘴巴或汽车检测
//输入暂时缩小到“scaledWidth”以便更快地检测，因为200足以找到人脸
void detectObjectsCustom(const Mat &img, CascadeClassifier &cascade, vector<Rect> &objects, int scaledWidth, int flags, Size minFeatureSize, float searchScaleFactor, int minNeighbors)
{
    // 如果输入图像不是灰度，则将BGR或BGRA彩色图像转换为灰度。
    Mat gray;
    if (img.channels() == 3) {//
        cvtColor(img, gray, CV_BGR2GRAY);
    }
    else if (img.channels() == 4) {
        cvtColor(img, gray, CV_BGRA2GRAY);
    }
    else {
        // 直接访问输入图像，因为它已经是灰度的
        gray = img;
    }

    // 可能会缩小图像,以更快地运行
    Mat inputImg;
    float scale = img.cols / (float)scaledWidth;
    if (img.cols > scaledWidth) {
        // Shrink the image while keeping the same aspect ratio.
        // 缩小图像，同时保持相同的纵横比。
        int scaledHeight = cvRound(img.rows / scale);
        resize(gray, inputImg, Size(scaledWidth, scaledHeight));
    }
    else {
        // 直接访问输入图像,因为它已经很小了
        inputImg = gray;
    }

    // 标准化亮度和对比度以改善暗图像
    Mat equalizedImg;
    equalizeHist(inputImg, equalizedImg);

    // 检测小灰度图像中的对象
    cascade.detectMultiScale(equalizedImg, objects, searchScaleFactor, minNeighbors, flags, minFeatureSize);

    // 如果在检测之前图像暂时缩小,则放大结果
    if (img.cols > scaledWidth) {
        for (int i = 0; i < (int)objects.size(); i++ ) {
            objects[i].x = cvRound(objects[i].x * scale);
            objects[i].y = cvRound(objects[i].y * scale);
            objects[i].width = cvRound(objects[i].width * scale);
            objects[i].height = cvRound(objects[i].height * scale);
        }
    }

    // 确保对象完全在图像中,以防它在边界上
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

    // 返回时将检测到的人脸矩形存储在"对象"中
}

//只搜索图像中的一个对象，例如最大的人脸,将结果存储到'largestObject'中
//可以使用Haar级联或LBP级联进行面部检测，甚至眼睛、嘴巴或汽车检测
//输入暂时缩小到"scaledWidth"以便更快地检测，因为240足以找到人脸
//注意：detectLargestObject()应该比detectManyObjects()快
void detectLargestObject(const Mat &img, CascadeClassifier &cascade, Rect &largestObject, int scaledWidth)
{
    // 只搜索一个对象(图像中最大的)
    int flags = CASCADE_FIND_BIGGEST_OBJECT;// | CASCADE_DO_ROUGH_SEARCH;
    // 最小对象大小
    Size minFeatureSize = Size(20, 20);
    // How detailed should the search be. Must be larger than 1.0.
    // 搜索应该有多详细.必须大于1.0
    float searchScaleFactor = 1.1f;
    // How much the detections should be filtered out. This should depend on how bad false detections are to your system.
    // minNeighbors=2 means lots of good+bad detections, and minNeighbors=6 means only good detections are given but some are missed.
    // 应该过滤掉多少检测结果。这应该取决于错误检测对系统的影响
    // minNeighbors=2表示有很多 好的+坏的 检测,minNeighbors=6 表示只有好的检测被给出,但有些被遗漏了
    int minNeighbors = 4;

    // 执行目标或人脸检测,只查找1个目标(图像中最大的)
    vector<Rect> objects;
    detectObjectsCustom(img, cascade, objects, scaledWidth, flags, minFeatureSize, searchScaleFactor, minNeighbors);
    if (objects.size() > 0) {
        // 返回唯一检测到的对象
        largestObject = (Rect)objects.at(0);
    }
    else {
        // 返回一个无效的rect
        largestObject = Rect(-1,-1,-1,-1);
    }
}

//搜索图像中的许多对象，如所有的人脸，将结果存储到"objects"中
//可以使用Haar级联或LBP级联进行面部检测，甚至眼睛、嘴巴或汽车检测
//输入暂时缩小到"scaledWidth"以便更快地检测，因为240足以找到人脸
//注意：detectLargestObject()应该比detectManyObjects()快
void detectManyObjects(const Mat &img, CascadeClassifier &cascade, vector<Rect> &objects, int scaledWidth)
{
    // Search for many objects in the one image.
    int flags = CASCADE_SCALE_IMAGE;

    //最小对象的尺寸
    Size minFeatureSize = Size(20, 20);
    // 搜索应该有多详细.必须大于1.0
    float searchScaleFactor = 1.1f;
    // 应该过滤掉多少检测结果。这应该取决于错误检测对系统的影响
    // minNeighbors=2 means lots of good+bad detections, and minNeighbors=6 means only good detections are given but some are missed.
    // minNeighbors=2表示有很多 好的+坏的 检测,minNeighbors=6 表示只有好的检测被给出,但有些被遗漏了
    int minNeighbors = 4;

    // 执行对象或人脸检测,在一个图像中查找多个对象
    detectObjectsCustom(img, cascade, objects, scaledWidth, flags, minFeatureSize, searchScaleFactor, minNeighbors);
}
