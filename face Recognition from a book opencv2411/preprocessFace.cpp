/*****************************************************************************
*   Face Recognition using Eigenfaces or Fisherfaces
    使用特征面或渔业面的面部识别
******************************************************************************
*   by Shervin Emami, 5th Dec 2012
*   http://www.shervinemami.info/openCV.html
******************************************************************************
*   Ch8 of the book "Mastering OpenCV with Practical Computer Vision Projects"
*   Copyright Packt Publishing 2012.
*   http://www.packtpub.com/cool-projects-with-opencv/book
*****************************************************************************/

//////////////////////////////////////////////////////////////////////////////////////
// preprocessFace.cpp, by Shervin Emami (www.shervinemami.info) on 30th May 2012.
// Easily preprocess face images, for face recognition.
//////////////////////////////////////////////////////////////////////////////////////

const double DESIRED_LEFT_EYE_X = 0.16;
// Controls how much of the face is visible after preprocessing.
//控制预处理后有多少面可见。

const double DESIRED_LEFT_EYE_Y = 0.14;
const double FACE_ELLIPSE_CY = 0.40;
const double FACE_ELLIPSE_W = 0.50;
// Should be atleast 0.5 应该至少是0.5
const double FACE_ELLIPSE_H = 0.80;
// Controls how tall the face mask is. 控制面罩的高度。


#include "detectObject.h"
// Easily detect faces or eyes (using LBP or Haar Cascades).
//易于检测面部或眼睛（使用LBP或Haar级联）。

#include "preprocessFace.h"
// Easily preprocess face images, for face recognition.
//易于对人脸图像进行预处理，用于人脸识别。

#include "ImageUtils.h"
// Shervin's handy OpenCV utility functions.
//谢尔文的OpenCV实用功能。

/*
// Remove the outer border of the face, so it doesn't include the background & hair.
// Keeps the center of the rectangle at the same place, rather than just dividing all values by 'scale'.
 //去掉脸的外边框，这样就不包括背景和头发。
 //将矩形的中心保持在同一位置，而不是将所有值除以“scale”。
Rect scaleRectFromCenter(const Rect wholeFaceRect, float scale)
{
    float faceCenterX = wholeFaceRect.x + wholeFaceRect.width * 0.5f;
    float faceCenterY = wholeFaceRect.y + wholeFaceRect.height * 0.5f;
    float newWidth = wholeFaceRect.width * scale;
    float newHeight = wholeFaceRect.height * scale;
    Rect faceRect;
    faceRect.width = cvRound(newWidth);
 // Shrink the region
    faceRect.height = cvRound(newHeight);
    faceRect.x = cvRound(faceCenterX - newWidth * 0.5f);
 // Move the region so that the center is still the same spot.
    faceRect.y = cvRound(faceCenterY - newHeight * 0.5f);
    
    return faceRect;
}
*/

// Search for both eyes within the given face image. Returns the eye centers in 'leftEye' and 'rightEye',
// or sets them to (-1,-1) if each eye was not found. Note that you can pass a 2nd eyeCascade if you
// want to search eyes using 2 different cascades. For example, you could use a regular eye detector
// as well as an eyeglasses detector, or a left eye detector as well as a right eye detector.
// Or if you don't want a 2nd eye detection, just pass an uninitialized CascadeClassifier.
// Can also store the searched left & right eye regions if desired.
//在给定的人脸图像中搜索双眼。返回“leftEye”和“rightEye”中的眼睛中心，
//如果没有找到每只眼睛，则将它们设置为（-1，-1）。注意，如果你
//想用两个不同的级联搜索眼睛。例如，你可以使用普通的眼睛检测器
//以及一个眼镜探测器，或者一个左眼探测器和一个右眼探测器。
//或者如果你不想要第二个眼睛检测，只需要通过一个未初始化的级联分类器。
//如果需要，还可以存储搜索到的左眼和右眼区域。
void detectBothEyes(const Mat &face, CascadeClassifier &eyeCascade1, CascadeClassifier &eyeCascade2, Point &leftEye, Point &rightEye, Rect *searchedLeftEye, Rect *searchedRightEye)
{
    // Skip the borders of the face, since it is usually just hair and ears, that we don't care about.
/*
    // For "2splits.xml": Finds both eyes in roughly 60% of detected faces, also detects closed eyes.
 //跳过脸的边缘，因为通常只是头发和耳朵，我们不在乎。
 /*
 //对于“2splits.xml”：在大约60%的被检测到的脸上发现双眼，同时也检测到闭着的眼睛。
    const float EYE_SX = 0.12f;
    const float EYE_SY = 0.17f;
    const float EYE_SW = 0.37f;
    const float EYE_SH = 0.36f;
*/
    

    // For mcs.xml: Finds both eyes in roughly 80% of detected faces, also detects closed eyes.
    //对于mcs.xml：在大约80%的被检测到的脸上发现双眼，也检测到闭着的眼睛。
    const float EYE_SX = 0.10f;
    const float EYE_SY = 0.19f;
    const float EYE_SW = 0.40f;
    const float EYE_SH = 0.36f;

/*
    // For default eye.xml or eyeglasses.xml: Finds both eyes in roughly 40% of detected faces, but does not detect closed eyes.
    //对于default eye.xml或eyegasses.xml：在大约40%的检测到的人脸中发现双眼，但不检测闭着的眼睛。
    const float EYE_SX = 0.16f;
    const float EYE_SY = 0.26f;
    const float EYE_SW = 0.30f;
    const float EYE_SH = 0.28f;
*/
    //左眼
    int leftX = cvRound(face.cols * EYE_SX);
    int topY = cvRound(face.rows * EYE_SY);
    int widthX = cvRound(face.cols * EYE_SW);
    int heightY = cvRound(face.rows * EYE_SH);
    int rightX = cvRound(face.cols * (1.0-EYE_SX-EYE_SW) );  // Start of right-eye corner //右眼角起点

    //左右眼在脸中部分
    Mat topLeftOfFace = face(Rect(leftX, topY, widthX, heightY));
    Mat topRightOfFace = face(Rect(rightX, topY, widthX, heightY));
    Rect leftEyeRect, rightEyeRect;

    // Return the search windows to the caller, if desired.
    //如果需要，将搜索窗口返回给调用方。  返回左右眼
    if (searchedLeftEye)
        *searchedLeftEye = Rect(leftX, topY, widthX, heightY);
    if (searchedRightEye)
        *searchedRightEye = Rect(rightX, topY, widthX, heightY);

    // Search the left region, then the right region using the 1st eye detector.
    //用第一眼探测器搜索左边区域，然后再搜索右边区域。
    detectLargestObject(topLeftOfFace, eyeCascade1, leftEyeRect, topLeftOfFace.cols);
    detectLargestObject(topRightOfFace, eyeCascade1, rightEyeRect, topRightOfFace.cols);

    // If the eye was not detected, try a different cascade classifier.
    //检测左眼
    //如果未检测到眼睛，请尝试其他级联分类器。
    if (leftEyeRect.width <= 0 && !eyeCascade2.empty()) {
        detectLargestObject(topLeftOfFace, eyeCascade2, leftEyeRect, topLeftOfFace.cols);
        //if (leftEyeRect.width > 0)
        //    cout << "2nd eye detector LEFT SUCCESS" << endl;
        //else
        //    cout << "2nd eye detector LEFT failed" << endl;
    }
    //else
    //    cout << "1st eye detector LEFT SUCCESS" << endl;

    // If the eye was not detected, try a different cascade classifier.
    //如果未检测到眼睛，请尝试其他级联分类器。
    //检测右眼
    if (rightEyeRect.width <= 0 && !eyeCascade2.empty()) {
        detectLargestObject(topRightOfFace, eyeCascade2, rightEyeRect, topRightOfFace.cols);
        //if (rightEyeRect.width > 0)
        //    cout << "2nd eye detector RIGHT SUCCESS" << endl;
        //else
        //    cout << "2nd eye detector RIGHT failed" << endl;
    }
    //else
    //    cout << "1st eye detector RIGHT SUCCESS" << endl;

    if (leftEyeRect.width > 0) {
        // Check if the eye was detected.
        //检查眼睛是否被发现。  左眼
        leftEyeRect.x += leftX;
        // Adjust the left-eye rectangle because the face border was removed.
        //调整左眼矩形，因为面边框已删除。
        leftEyeRect.y += topY;
        leftEye = Point(leftEyeRect.x + leftEyeRect.width/2, leftEyeRect.y + leftEyeRect.height/2);
    }
    else {
        leftEye = Point(-1, -1);
        // Return an invalid point
        //返回无效点
    }

    if (rightEyeRect.width > 0) {
        // Check if the eye was detected.
        //检查眼睛是否被发现。 右眼
        rightEyeRect.x += rightX;
        // Adjust the right-eye rectangle, since it starts on the right side of the image.
        //调整右眼矩形，因为它从图像的右侧开始。
        rightEyeRect.y += topY;
        // Adjust the right-eye rectangle because the face border was removed.
        ////调整右眼矩形，因为面边框已删除。
        rightEye = Point(rightEyeRect.x + rightEyeRect.width/2, rightEyeRect.y + rightEyeRect.height/2);
    }
    else {
        rightEye = Point(-1, -1);
        // Return an invalid point
        //返回无效点
    }
}

// Histogram Equalize seperately for the left and right sides of the face.
////直方图分别为脸的左侧和右侧均衡。
void equalizeLeftAndRightHalves(Mat &faceImg)
{
    // It is common that there is stronger light from one half of the face than the other. In that case,
    // if you simply did histogram equalization on the whole face then it would make one half dark and
    // one half bright. So we will do histogram equalization separately on each face half, so they will
    // both look similar on average. But this would cause a sharp edge in the middle of the face, because
    // the left half and right half would be suddenly different. So we also histogram equalize the whole
    // image, and in the middle part we blend the 3 images together for a smooth brightness transition.
    //很常见的一种情况是，一半的脸比另一半的脸发出更强的光。在这种情况下，
    //如果你只是在整个脸上做直方图均衡化，那么它会使一个半暗的和
    //一半亮。所以我们将分别对每个半张脸做直方图均衡化，这样他们就会
    //两者看起来平均相似。但这会在脸中间形成一个锐利的边缘，因为
    //左半边和右半边会突然不同。所以我们也用直方图来平衡整个
    //图像，在中间部分，我们将3个图像混合在一起，以实现平滑的亮度过渡。

    int w = faceImg.cols;
    int h = faceImg.rows;

    // 1) First, equalize the whole face.
    //1）首先，平衡整个面部。
    Mat wholeFace;
    equalizeHist(faceImg, wholeFace);

    // 2) Equalize the left half and the right half of the face separately.
    //2）分别平衡面部的左半部分和右半部分。
    int midX = w/2;
    Mat leftSide = faceImg(Rect(0,0, midX,h));
    Mat rightSide = faceImg(Rect(midX,0, w-midX,h));
    equalizeHist(leftSide, leftSide);
    equalizeHist(rightSide, rightSide);

    // 3) Combine the left half and right half and whole face together, so that it has a smooth transition.
    //3）将左、右半边和整张脸结合在一起，使其平稳过渡。
    for (int y=0; y<h; y++) {
        for (int x=0; x<w; x++) {
            int v;
            if (x < w/4) {
                // Left 25%: just use the left face.
                //左25%：只用左脸。
                v = leftSide.at<uchar>(y,x);
            }
            else if (x < w*2/4) {
                // Mid-left 25%: blend the left face & whole face.
                //左中25%：混合左脸和整张脸。
                int lv = leftSide.at<uchar>(y,x);
                int wv = wholeFace.at<uchar>(y,x);
                // Blend more of the whole face as it moves further right along the face.
                //当它沿着面进一步向右移动时，混合更多的整个面。
                float f = (x - w*1/4) / (float)(w*0.25f);
                v = cvRound((1.0f - f) * lv + (f) * wv);
            }
            else if (x < w*3/4) {
                // Mid-right 25%: blend the right face & whole face.
                //右中25%：混合右脸和整张脸。
                int rv = rightSide.at<uchar>(y,x-midX);
                int wv = wholeFace.at<uchar>(y,x);
                // Blend more of the right-side face as it moves further right along the face.
                //混合更多的右侧面，因为它沿着面进一步向右移动。
                float f = (x - w*2/4) / (float)(w*0.25f);
                v = cvRound((1.0f - f) * wv + (f) * rv);
            }
            else {
                // Right 25%: just use the right face.
                //右25%：只要用右脸。
                v = rightSide.at<uchar>(y,x-midX);
            }
            faceImg.at<uchar>(y,x) = v;
        }
        // end x loop
    }
    //end y loop
}


// Create a grayscale face image that has a standard size and contrast & brightness.
// "srcImg" should be a copy of the whole color camera frame, so that it can draw the eye positions onto.
// If 'doLeftAndRightSeparately' is true, it will process left & right sides seperately,
// so that if there is a strong light on one side but not the other, it will still look OK.
// Performs Face Preprocessing as a combination of:
//  - geometrical scaling, rotation and translation using Eye Detection,
//  - smoothing away image noise using a Bilateral Filter,
//  - standardize the brightness on both left and right sides of the face independently using separated Histogram Equalization,
//  - removal of background and hair using an Elliptical Mask.
// Returns either a preprocessed face square image or NULL (ie: couldn't detect the face and 2 eyes).
// If a face is found, it can store the rect coordinates into 'storeFaceRect' and 'storeLeftEye' & 'storeRightEye' if given,
// and eye search regions into 'searchedLeftEye' & 'searchedRightEye' if given.
//创建具有标准大小、对比度和亮度的灰度面部图像。
//“srcImg”应该是整个彩色相机框架的副本，这样它就可以把眼睛的位置吸引到上面。
//如果“doleftandrightspeparate”为真，则它将分别处理左右两侧，
//所以如果一边有强光而另一边没有，它看起来还是可以的。
//将人脸预处理作为以下操作的组合执行：
//-使用眼睛检测的几何缩放、旋转和平移，
//-使用双边滤波器平滑图像噪声，
//-使用分离的直方图均衡化独立地标准化脸部左右两侧的亮度，
//-使用椭圆口罩去除背景和头发。
//返回预处理的人脸正方形图像或空值（即：无法检测人脸和两只眼睛）。
//如果找到一个面，它可以将rect坐标存储到'storeFaceRect'和'storeLeftEye'和'storerighteeye'中，如果给定，
//眼睛搜索区域分为“searchedLeftEye”和“searchedRightEye”。
Mat getPreprocessedFace(Mat &srcImg, int desiredFaceWidth, CascadeClassifier &faceCascade, CascadeClassifier &eyeCascade1, CascadeClassifier &eyeCascade2, bool doLeftAndRightSeparately, Rect *storeFaceRect, Point *storeLeftEye, Point *storeRightEye, Rect *searchedLeftEye, Rect *searchedRightEye)
{
    // Use square faces.
    //使用正方形面。
    int desiredFaceHeight = desiredFaceWidth;

    // Mark the detected face region and eye search regions as invalid, in case they aren't detected.
    //将检测到的人脸区域和眼睛搜索区域标记为无效，以防它们未被检测到。
    if (storeFaceRect)
        storeFaceRect->width = -1;
    if (storeLeftEye)
        storeLeftEye->x = -1;
    if (storeRightEye)
        storeRightEye->x= -1;
    if (searchedLeftEye)
        searchedLeftEye->width = -1;
    if (searchedRightEye)
        searchedRightEye->width = -1;

    // Find the largest face.
    //找到最大的脸。
    Rect faceRect;
    detectLargestObject(srcImg, faceCascade, faceRect);

    // Check if a face was detected.
    //检查是否检测到面部。
    if (faceRect.width > 0) {

        // Give the face rect to the caller if desired.
        //如果需要的话，把face rect给呼叫者。
        if (storeFaceRect)
            *storeFaceRect = faceRect;

        Mat faceImg = srcImg(faceRect);
        // Get the detected face image.//获取检测到的人脸图像。

        // If the input image is not grayscale, then convert the BGR or BGRA color image to grayscale.
        //如果输入图像不是灰度，则将BGR或BGRA彩色图像转换为灰度。
        Mat gray;
        if (faceImg.channels() == 3) {
            cvtColor(faceImg, gray, CV_BGR2GRAY);
        }
        else if (faceImg.channels() == 4) {
            cvtColor(faceImg, gray, CV_BGRA2GRAY);
        }
        else {
            // Access the input image directly, since it is already grayscale.
            //直接访问输入图像，因为它已经是灰度的。
            gray = faceImg;
        }
		//imshow("gray", gray);
		//getchar();
        // Search for the 2 eyes at the full resolution, since eye detection needs max resolution possible!
        //在全分辨率下搜索2只眼睛，因为眼睛检测需要尽可能高的分辨率！
        Point leftEye, rightEye;
        detectBothEyes(gray, eyeCascade1, eyeCascade2, leftEye, rightEye, searchedLeftEye, searchedRightEye);

        // Give the eye results to the caller if desired.
        //如果需要的话，把眼睛测试结果给调用的人。
        if (storeLeftEye)
            *storeLeftEye = leftEye;
        if (storeRightEye)
            *storeRightEye = rightEye;

        // Check if both eyes were detected.
        //检查双眼是否被发现。
		cout << "leftEye.x "<<leftEye.x <<"rightEye.x "<< rightEye.x << endl;
        if (leftEye.x >= 0 && rightEye.x >= 0) {

            // Make the face image the same size as the training images.

            // Since we found both eyes, lets rotate & scale & translate the face so that the 2 eyes
            // line up perfectly with ideal eye positions. This makes sure that eyes will be horizontal,
            // and not too far left or right of the face, etc.

            // Get the center between the 2 eyes.
            //使人脸图像与训练图像大小相同。
            //既然我们找到了两只眼睛，让我们旋转、缩放和平移脸部，使两只眼睛
            //与理想的眼睛位置完美地排成一行。这样可以确保眼睛是水平的，
            //脸的左边或右边不要太远。
            //两眼中间。
            Point2f eyesCenter = Point2f( (leftEye.x + rightEye.x) * 0.5f, (leftEye.y + rightEye.y) * 0.5f );
            // Get the angle between the 2 eyes.
            //得到两眼之间的角度。
            double dy = (rightEye.y - leftEye.y);
            double dx = (rightEye.x - leftEye.x);
            double len = sqrt(dx*dx + dy*dy);
            double angle = atan2(dy, dx) * 180.0/CV_PI;
            // Convert from radians to degrees.
            //从弧度转换为度。

            // Hand measurements shown that the left eye center should ideally be at roughly (0.19, 0.14) of a scaled face image.
            //手部测量显示，理想情况下，左眼中心应该在缩放的面部图像的大约（0.19，0.14）处。
            const double DESIRED_RIGHT_EYE_X = (1.0f - DESIRED_LEFT_EYE_X);
            // Get the amount we need to scale the image to be the desired fixed size we want.
            //获取缩放图像所需的量，使其达到所需的固定大小。
            double desiredLen = (DESIRED_RIGHT_EYE_X - DESIRED_LEFT_EYE_X) * desiredFaceWidth;
            double scale = desiredLen / len;
            // Get the transformation matrix for rotating and scaling the face to the desired angle & size.
            //得到用于旋转和缩放面到所需角度和大小的变换矩阵。
            Mat rot_mat = getRotationMatrix2D(eyesCenter, angle, scale);
            // Shift the center of the eyes to be the desired center between the eyes.
            //移动眼睛中心，使其成为眼睛之间所需的中心。
            
            rot_mat.at<double>(0, 2) += desiredFaceWidth * 0.5f - eyesCenter.x;
            rot_mat.at<double>(1, 2) += desiredFaceHeight * DESIRED_LEFT_EYE_Y - eyesCenter.y;

            // Rotate and scale and translate the image to the desired angle & size & position!
            // Note that we use 'w' for the height instead of 'h', because the input face has 1:1 aspect ratio.
            //旋转和缩放并将图像转换到所需的角度、大小和位置！
            //注意，我们使用“w”而不是“h”作为高度，因为输入面的纵横比为1:1。
            Mat warped = Mat(desiredFaceHeight, desiredFaceWidth, CV_8U, Scalar(128));
            // Clear the output image to a default grey.
            //将输出图像清除为默认灰色。
            warpAffine(gray, warped, rot_mat, warped.size());
            //imshow("warped", warped);
            //imshow（“翘曲”，翘曲）；
            // Give the image a standard brightness and contrast, in case it was too dark or had low contrast.
            //给图像一个标准的亮度和对比度，以防太暗或对比度低。
            if (!doLeftAndRightSeparately) {
                // Do it on the whole face.
                //整张脸都做。
                equalizeHist(warped, warped);
            }
            else {
                // Do it seperately for the left and right sides of the face.
                //在脸的左右两边分别做。
                equalizeLeftAndRightHalves(warped);
            }
            //imshow("equalized", warped);
            //imshow（“均衡”，扭曲）；

            // Use the "Bilateral Filter" to reduce pixel noise by smoothing the image, but keeping the sharp edges in the face.
            //使用“双边滤波器”通过平滑图像来减少像素噪声，但要保持脸部的锐利边缘。
            Mat filtered = Mat(warped.size(), CV_8U);
            bilateralFilter(warped, filtered, 0, 20.0, 2.0);
            //imshow("filtered", filtered);

            // Filter out the corners of the face, since we mainly just care about the middle parts.
            // Draw a filled ellipse in the middle of the face-sized image.
            //过滤掉脸上的角，因为我们只关心中间部分。
            //在面大小的图像中间画一个填充椭圆。
            Mat mask = Mat(warped.size(), CV_8U, Scalar(0));
            // Start with an empty mask.//从一个空面具开始。
            Point faceCenter = Point( desiredFaceWidth/2, cvRound(desiredFaceHeight * FACE_ELLIPSE_CY) );
            Size size = Size( cvRound(desiredFaceWidth * FACE_ELLIPSE_W), cvRound(desiredFaceHeight * FACE_ELLIPSE_H) );
            ellipse(mask, faceCenter, size, 0, 0, 360, Scalar(255), CV_FILLED);
            //imshow("mask", mask);
            //imshow（“面具”，mask）；

            // Use the mask, to remove outside pixels.
            // 使用遮罩移除外部像素。
            Mat dstImg = Mat(warped.size(), CV_8U, Scalar(128)); // Clear the output image to a default gray.
            /*
            namedWindow("filtered");
            imshow("filtered", filtered);
            namedWindow("dstImg");
            imshow("dstImg", dstImg);
            namedWindow("mask");
            imshow("mask", mask);
            */
            // Apply the elliptical mask on the face.
            /*
             namedWindow（“过滤”）；
             imshow（“过滤”，过滤）；
             姓名（dstImg）；
             显示（“dstImg”，dstImg）；
             名称窗口（“掩码”）；
             imshow（“面具”，mask）；
             */
            //在脸上涂上椭圆面膜。
            filtered.copyTo(dstImg, mask);
            // Copies non-masked pixels from filtered to dstImg.
           // imshow("dstImg", dstImg);
 
			cout << "dstImg.rows " << dstImg.rows << " " << "dstImg.cols" << dstImg.cols << endl;
            return dstImg;
		}
		
        else {
            // Since no eyes were found, just do a generic image resize.
			resize(gray, gray, Size(desiredFaceWidth, desiredFaceHeight));
			Mat warped = gray;
			//给图像一个标准的亮度和对比度，以防太暗或对比度低。
			/*
			if (!doLeftAndRightSeparately) {
				// Do it on the whole face.
				//整张脸都做。
				equalizeHist(warped, warped);
			}
			else {
				// Do it seperately for the left and right sides of the face.
				//在脸的左右两边分别做。
				equalizeLeftAndRightHalves(warped);
			}
			*/
			equalizeHist(warped, warped);
			//imshow("equalized", warped);
			//imshow（“均衡”，扭曲）；

			// Use the "Bilateral Filter" to reduce pixel noise by smoothing the image, but keeping the sharp edges in the face.
			//使用“双边滤波器”通过平滑图像来减少像素噪声，但要保持脸部的锐利边缘。
			Mat filtered = Mat(warped.size(), CV_8U);
			bilateralFilter(warped, filtered, 0, 20.0, 2.0);
			//imshow("filtered", filtered);

			// Filter out the corners of the face, since we mainly just care about the middle parts.
			// Draw a filled ellipse in the middle of the face-sized image.
			//过滤掉脸上的角，因为我们只关心中间部分。
			//在面大小的图像中间画一个填充椭圆。
			Mat mask = Mat(warped.size(), CV_8U, Scalar(0));
			// Start with an empty mask.//从一个空面具开始。
			Point faceCenter = Point(desiredFaceWidth / 2, cvRound(desiredFaceHeight * FACE_ELLIPSE_CY));
			Size size = Size(cvRound(desiredFaceWidth * FACE_ELLIPSE_W), cvRound(desiredFaceHeight * FACE_ELLIPSE_H));
			ellipse(mask, faceCenter, size, 0, 0, 360, Scalar(255), CV_FILLED);
			//imshow("mask", mask);
			//imshow（“面具”，mask）；

			// Use the mask, to remove outside pixels.
			// 使用遮罩移除外部像素。
			Mat dstImg = Mat(warped.size(), CV_8U, Scalar(128)); // Clear the output image to a default gray.
			
			filtered.copyTo(dstImg, mask);
			// Copies non-masked pixels from filtered to dstImg.
			//imshow("dstImg", dstImg);
			cout << "else  dstImg.rows " << dstImg.rows << " " << "dstImg.cols" << dstImg.cols << endl;
			return dstImg;
		}
		
		
    }
    return Mat();
}
