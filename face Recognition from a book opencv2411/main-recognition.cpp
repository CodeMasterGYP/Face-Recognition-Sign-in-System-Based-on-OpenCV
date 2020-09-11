/*
*   Face Recognition using Eigenfaces or Fisherfaces
*   特征脸算法或基于的线性判别人脸识别算法
*/


/* Face Detection & Face Recognition from a webcam using LBP and Eigenfaces or Fisherfaces.
*使用LBP和特征脸或Fisherfaces的网络摄像头进行人脸检测和人脸识别。
*LBP:　LBP（Local Binary Pattern，局部二值模式）是一种用来描述图像局部纹理特征的算子；它具有旋转不变性和灰度不变性等显著的优点。它是首先由T.
*Ojala, M.Pietikäinen, 和D. Harwood 在1994年提出，用于纹理特征提取。而且，提取的特征是图像的局部的纹理特征；
* Requires OpenCV v2.4.1 or later (from June 2012), otherwise the FaceRecognizer will not compile or run.
* 需要OpenCVv2.4.1或更高版本（从2012年6月起），否则FaceRecognizer将无法编译或运行。
*/
// The Face Recognition algorithm can be one of these and perhaps more, depending on your version of OpenCV, which must be atleast v2.4.1:
//    "FaceRecognizer.Eigenfaces":  Eigenfaces, also referred to as PCA (Turk and Pentland, 1991).
//    "FaceRecognizer.Fisherfaces": Fisherfaces, also referred to as LDA (Belhumeur et al, 1997).
//    "FaceRecognizer.LBPH":        Local Binary Pattern Histograms (Ahonen et al, 2006).
const char *facerecAlgorithm = "FaceRecognizer.Eigenfaces";
//const char *facerecAlgorithm = "FaceRecognizer.Eigenfaces";

/*
不同算法时间不同
*/
// Sets how confident the Face Verification algorithm should be to decide if it is an unknown person or a known person.
// A value roughly around 0.5 seems OK for Eigenfaces or 0.7 for Fisherfaces, but you may want to adjust it for your
// conditions, and if you use a different Face Recognition algorithm.
// Note that a higher threshold value means accepting more faces as known people,
// whereas lower values mean more faces will be classified as "unknown".
const float UNKNOWN_PERSON_THRESHOLD = 0.8f;
//设定阈值


// Cascade Classifier file, used for Face Detection.
/*
级联分类器文件，用于人脸检测。
人脸数据
*/
const char *faceCascadeFilename = "lbpcascade_frontalface.xml";     // LBP face detector. LBP面部探测
//const char *faceCascadeFilename = "haarcascade_frontalface_alt_tree.xml";  // Haar face detector.
//const char *eyeCascadeFilename1 = "haarcascade_lefteye_2splits.xml";   // Best eye detector for open-or-closed eyes.
//const char *eyeCascadeFilename2 = "haarcascade_righteye_2splits.xml";   // Best eye detector for open-or-closed eyes.
//const char *eyeCascadeFilename1 = "haarcascade_mcs_lefteye.xml";       // Good eye detector for open-or-closed eyes.
//const char *eyeCascadeFilename2 = "haarcascade_mcs_righteye.xml";       // Good eye detector for open-or-closed eyes.
/*
能够检测睁或者闭眼的人眼检测器如下:
1、haarcascade_mcs_lefteye.xml(and haarcascade_mcs_righteye.xml)
2、haarcascade_lefteye_2splits.xml(and haarcascade_righteye_2splits.xml)
*/

/*
只能检测睁眼的人眼检测器：
1、haarcascade_eye.xml
2、haarcascade_eye_tree_eyeglasses.xml
*/
const char *eyeCascadeFilename1 = "haarcascade_mcs_lefteye.xml";               // Basic eye detector for open eyes only.
const char *eyeCascadeFilename2 = "haarcascade_mcs_righteye.xml"; // Basic eye detector for open eyes if they might wear glasses.


// Set the desired face dimensions. Note that "getPreprocessedFace()" will return a square face.
/*
设置所需的面尺寸。注意，“getPreprocessedFace（）”将返回一个方形面。
*/
const int faceWidth = 70;
const int faceHeight = faceWidth;

// Try to set the camera resolution 分辨率. Note that this only works for some cameras on
// some computers and only for some drivers, so don't rely on it to work!
//设置分辨率 实用电脑和驱动
const int DESIRED_CAMERA_WIDTH = 480;
const int DESIRED_CAMERA_HEIGHT = 640;

// Parameters controlling how often to keep new faces when collecting them. Otherwise, the training set could look to similar to each other!
/*
参数控制收集新面时保留新面的频率。否则，训练集看起来会很相似！
*/
const double CHANGE_IN_IMAGE_FOR_COLLECTION = 0.3;      // How much the facial image should change before collecting a new face photo for training.
/*
在收集新的面部照片进行训练之前，面部图像应该改变多少。
*/
const double CHANGE_IN_SECONDS_FOR_COLLECTION = 1.0;       // How much time must pass before collecting a new face photo for training.

const char *windowName = "based on web";  // Name shown in the GUI图形化用户界面 window.
const int BORDER = 8;  // Border between GUI elements to the edge of the image.
//8  边界
const bool preprocessLeftAndRightSeparately = true;
// Preprocess left & right sides of the face separately, in case there is stronger light on one side.
/*
*分别对脸部左右两侧进行预处理，以防一侧光线较强。
*/
// Set to true if you want to see many windows created, showing various debug info. Set to 0 otherwise.
bool m_debug = false;
/*
如果希望看到创建了许多窗口，并显示各种调试信息，请设置为true。否则设置为0。
bool m_debug=false；
*/

#include <stdio.h>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>  
#include <io.h>
#include <sstream> 
#include "mySQL.h"
// Include OpenCV's C++ Interface
#include "opencv2/opencv.hpp"

// Include the rest of our code!
#include "detectObject.h"//易于检测面部或眼睛（使用LBP或Haar级联）。
#include "preprocessFace.h"//易于对人脸图像进行预处理，用于人脸识别。
#include "recognition.h"     ////训练人脸识别系统，从图像中识别出一个人。
#include "ImageUtils.h"      // Shervin's handy OpenCV utility functions.工具类

using namespace cv;
using namespace std;


#if !defined VK_ESCAPE
#define VK_ESCAPE 0x1B      // Escape character (27)转义字符（27）
#endif

//初始化数据库连接
mySQL mysql;

// Running mode for the Webcam-based interactive GUI program.
//基于网络摄像头的交互式图形用户界面程序的运行模式。
enum MODES { MODE_STARTUP = 0, MODE_DETECTION, MODE_RECOGNITION, MODE_SIGN_IN, MODE_END };
//0  1  2...
const char* MODE_NAMES[] = { "Startup", "Detection", "Recognition", "SignIn", "ERROR!" };
MODES m_mode = MODE_STARTUP;

// Position of GUI buttons:
//设置界面按钮位置
Rect BtnRecFace;
Rect BtnSignIn;
int m_gui_faces_left = -1;//左上
int m_gui_faces_top = -1;//最上边
 

//整数转为字符串
string toString(int temp){
	string out;
	stringstream in;
	in << temp;
	in >> out;
	return out;
}
//从字符串转为int
int fromString(string str){
	int out;
	stringstream in(str);
	in >> out;
	return out;
}

// Load the face and 1 or 2 eye detection XML classifiers.
//加载人脸和1或2个眼睛检测XML分类器。
//传脸、传左右眼
void initDetectors(CascadeClassifier &faceCascade, CascadeClassifier &eyeCascade1, CascadeClassifier &eyeCascade2)
{
	// Load the Face Detection cascade classifier xml file.
	try {   // Surround the OpenCV call by a try/catch block so we can give a useful error message!
		//try catch报错
		faceCascade.load(faceCascadeFilename);//级联分类器文件  传脸
	}
	catch (cv::Exception &e) {}
	if (faceCascade.empty()) { //无法加载人脸检测级联分类器
		cerr << "ERROR: Could not load Face Detection cascade classifier [" << faceCascadeFilename << "]!" << endl;
		//将文件从OpenCV数据文件夹（例如：“C:\\OpenCV\\data\\lbpcascades”）复制到这个WebcamFaceRec文件夹中。
		cerr << "Copy the file from your OpenCV data folder (eg: 'C:\\OpenCV\\data\\lbpcascades') into this WebcamFaceRec folder." << endl;
		exit(1);
	}
	cout << "Loaded the Face Detection cascade classifier [" << faceCascadeFilename << "]." << endl;
	//加载了人脸检测级联分类器

	// Load the Eye Detection cascade classifier xml file.
	//加载眼睛检测级联分类器xml文件
	try {   // Surround the OpenCV call by a try/catch block so we can give a useful error message!
		eyeCascade1.load(eyeCascadeFilename1);
		//加载眼睛检测级联分类器xml文件
	}
	catch (cv::Exception &e) {}
	if (eyeCascade1.empty()) {
		cerr << "ERROR: Could not load 1st Eye Detection cascade classifier [" << eyeCascadeFilename1 << "]!" << endl;
		//第一个眼加载失败
		cerr << "Copy the file from your OpenCV data folder (eg: 'C:\\OpenCV\\data\\haarcascades') into this WebcamFaceRec folder." << endl;
		exit(1);
	}
	cout << "Loaded the 1st Eye Detection cascade classifier [" << eyeCascadeFilename1 << "]." << endl;
	//succeed
	// Load the Eye Detection cascade classifier xml file.
	try {   // Surround the OpenCV call by a try/catch block so we can give a useful error message!
		eyeCascade2.load(eyeCascadeFilename2);
	}
	catch (cv::Exception &e) {}
	if (eyeCascade2.empty()) {
		cerr << "Could not load 2nd Eye Detection cascade classifier [" << eyeCascadeFilename2 << "]." << endl;
		// Dont exit if the 2nd eye detector did not load, because we have the 1st eye detector at least.
		//exit(1);
	}
	else
		cout << "Loaded the 2nd Eye Detection cascade classifier [" << eyeCascadeFilename2 << "]." << endl;
}
//加载另外一只眼

// Get access to the webcam.
//访问网络摄像头。
//初始化 图像和编号
void initWebcam(VideoCapture &videoCapture, int cameraNumber)
//获得设备的编号 权限
{
	// Get access to the default camera.
	try {   // Surround the OpenCV call by a try/catch block so we can give a useful error message!
		videoCapture.open(cameraNumber);
	}
	catch (cv::Exception &e) {}
	if (!videoCapture.isOpened()) {
		cerr << "ERROR: Could not access the camera!" << endl;
		exit(1);
	}
	cout << "Loaded camera " << cameraNumber << "." << endl;
}
//调用相机


// Draw text into an image. Defaults to top-left-justified text, but you can give negative x coords for right-justified text,
// and/or negative y coords for bottom-justified text.
// Returns the bounding rect around the drawn text.
/*
将文本绘制到图像中。默认为左上对齐文本，但可以为右对齐文本提供负x坐标，
和或下对齐文本的负y坐标。
返回绘制文本周围的边界矩形。  设置位置
*/
//矩形
Rect drawString(Mat img, string text, Point coord, Scalar color, float fontScale = 0.6f, int thickness = 1, int fontFace = FONT_HERSHEY_COMPLEX)
{
	// Get the text size & baseline.
	//获取文本大小和基线。
	int baseline = 0; //基线
	Size textSize = getTextSize(text, fontFace, fontScale, thickness, &baseline);
	//设置大小
	baseline += thickness;

	// Adjust the coords for left/right-justified or top/bottom-justified.
	//调整左对齐/右对齐或上对齐/下对齐的坐标。
	if (coord.y >= 0) {
		// Coordinates are for the top-left corner of the text from the top-left of the image, so move down by one row.
		//坐标是从图像左上角开始的文本左上角，因此向下移动一行。
		coord.y += textSize.height;
	}
	else {
		// Coordinates are for the bottom-left corner of the text from the bottom-left of the image, so come up from the bottom.
		//坐标是从图像左下角开始的文本的左下角，所以从底部开始。
		coord.y += img.rows - baseline + 1;
	}
	// Become right-justified if desired.
	//如果需要，调整为右对齐
	if (coord.x < 0) {
		coord.x += img.cols - textSize.width + 1;
	}

	// Get the bounding box around the text.
	//获取文本周围的边框
	Rect boundingRect = Rect(coord.x, coord.y - textSize.height, textSize.width, baseline + textSize.height);

	// Draw anti-aliased text.
	//绘制反走样文本
	putText(img, text, coord, fontFace, fontScale, color, thickness, CV_AA);

	// Let the user know how big their text is, in case they want to arrange things.
	//让用户知道他们的文本有多大，以防他们想安排事情
	return boundingRect;
}

// Draw a GUI button into the image, using drawString().
// Can specify a minWidth if you want several buttons to all have the same width.
// Returns the bounding rect around the drawn button, allowing you to position buttons next to each other.
/*
使用drawString()函数在图像中绘制一个GUI按钮。
如果希望多个按钮的宽度都相同，则可以指定最小宽度。
返回绘制按钮周围的边界矩形，允许您将按钮彼此相邻放置。
*/
Rect drawButton(Mat img, string text, Point coord, int minWidth = 0)
{
	int B = BORDER;
	Point textCoord = Point(coord.x + B, coord.y + B);
	// Get the bounding box around the text.
	//获取文本周围的边框
	Rect rcText = drawString(img, text, textCoord, CV_RGB(0, 0, 0));
	// Draw a filled rectangle around the text.
	//在文本周围画一个填充矩形。
	Rect rcButton = Rect(rcText.x - B, rcText.y - B, rcText.width + 2 * B, rcText.height + 2 * B);
	// Set a minimum button width.
	//设置最小按钮宽度
	if (rcButton.width < minWidth)
		rcButton.width = minWidth;
	// Make a semi-transparent white rectangle
	//做一个半透明的白色矩形。
	//按钮文本背景框  Addperson
	Mat matButton = img(rcButton);
	matButton += CV_RGB(90, 90, 90);
	// Draw a non-transparent white border.
	//画一个不透明的白色边框。
	//检测到的人脸
	rectangle(img, rcButton, CV_RGB(200, 200, 200), 1, CV_AA);

	// Draw the actual text that will be displayed, using anti-aliasing.
	//使用反锯齿绘制将显示的实际文本。  消除边缘锯齿
	drawString(img, text, textCoord, CV_RGB(10, 55, 20));

	return rcButton;
}

//判断点是否在矩形框里
bool isPointInRect(const Point pt, const Rect rc)
{
	if (pt.x >= rc.x && pt.x <= (rc.x + rc.width - 1))
		if (pt.y >= rc.y && pt.y <= (rc.y + rc.height - 1))
			return true;

	return false;
}

// Mouse event handler. Called automatically by OpenCV when the user clicks in the GUI window.
//鼠标事件处理程序。当用户在GUI窗口中单击时，OpenCV自动调用。
void onMouse(int event, int x, int y, int, void*)
{
	// We only care about left-mouse clicks, not right-mouse clicks or mouse movement.
	//左键
	if (event != CV_EVENT_LBUTTONDOWN)
		return;

	// Check if the user clicked on one of our GUI buttons.
	//判断用户是否点击GUI按钮
	Point pt = Point(x, y);
	//添加人脸按钮
	if (isPointInRect(pt, BtnRecFace)) {
		cout << "User clicked [Recognition] button" << endl;
		m_mode = MODE_RECOGNITION;
	}
	//删除所有
	else if (isPointInRect(pt, BtnSignIn)) {
		cout << "User clicked [Sign in] button." << endl;
		if (m_mode == MODE_RECOGNITION)
			m_mode = MODE_SIGN_IN;
		else{
			cout << "please recognize before sign in" << endl;
			m_mode = MODE_RECOGNITION;
		}
			
		
	}
	else {
		cout << "Do nothing" << endl;
		m_mode = MODE_DETECTION;
	}
}
//识别和签到
void recognitionAndSignIn(VideoCapture &videoCapture, CascadeClassifier &faceCascade, CascadeClassifier &eyeCascade1, CascadeClassifier &eyeCascade2){
	//加载训练好的模型
	Ptr<FaceRecognizer> model = Algorithm::create<FaceRecognizer>(facerecAlgorithm);
	model->load("faceClass.xml");
	string name;
	string id;
	// Since we have already initialized everything, lets start in Detection mode.
	//既然我们已经初始化了所有的东西，让我们从检测模式开始。
	m_mode = MODE_DETECTION;

	// Run forever, until the user hits Escape to "break" out of this loop.
	//一直运行 直到用户退出
	while (true) {

		// Grab the next camera frame. Note that you can't modify camera frames.
		//抓住下一个摄像机镜头。请注意，不能修改相机帧。
		Mat cameraFrame;
		videoCapture >> cameraFrame;
		if (cameraFrame.empty()) {
			cerr << "ERROR: Couldn't grab the next camera frame." << endl;
			//无法获取下一个相机帧  找不到下一张图片
			exit(1);
		}

		// Get a copy of the camera frame that we can draw onto.
		//拿一份我们可以画的照相机的相框。  将照片存入对应框架
		Mat displayedFrame;
		cameraFrame.copyTo(displayedFrame);

		// Run the face recognition system on the camera image. It will draw some things onto the given image, so make sure it is not read-only memory!
		////在摄像机图像上运行人脸识别系统。它会在给定的图像上绘制一些内容，因此请确保它不是只读内存！ 写入

		// Find a face and preprocess it to have a standard size and contrast & brightness.
		//找到一张脸并对其进行预处理，使其具有标准尺寸、对比度和亮度。
		Rect faceRect;  // Position of detected face.
		//检测面位置。
		Rect searchedLeftEye, searchedRightEye; // top-left and top-right regions of the face, where eyes were searched.
		//脸的左上角和右上角，眼睛被搜索到的地方。
		Point leftEye, rightEye;    // Position of the detected eyes.
		//检测眼睛的位置。
		Mat preprocessedFace = getPreprocessedFace(displayedFrame, faceWidth, faceCascade, eyeCascade1, eyeCascade2, preprocessLeftAndRightSeparately, &faceRect, &leftEye, &rightEye, &searchedLeftEye, &searchedRightEye);

		//初始化为未检测到
		bool gotFaceAndEyes = false;
		if (preprocessedFace.data)
			//检测到人脸和眼
			gotFaceAndEyes = true;

		// Draw an anti-aliased rectangle around the detected face.
		//在检测到的面周围绘制一个抗锯齿矩形。  保证准确率 识别率
		if (faceRect.width > 0) {
			rectangle(displayedFrame, faceRect, CV_RGB(255, 255, 0), 2, CV_AA);

			// Draw light-blue anti-aliased circles for the 2 eyes.
			//为两只眼睛画浅蓝色抗锯齿圆圈。  确定两只眼的位置
			Scalar eyeColor = CV_RGB(0, 255, 255);
			if (leftEye.x >= 0) {   // Check if the eye was detected
				//检查眼睛是否被发现 左眼被检测到
				circle(displayedFrame, Point(faceRect.x + leftEye.x, faceRect.y + leftEye.y), 6, eyeColor, 1, CV_AA);
			}
			if (rightEye.x >= 0) {   // Check if the eye was detected
				//右眼被检测到
				circle(displayedFrame, Point(faceRect.x + rightEye.x, faceRect.y + rightEye.y), 6, eyeColor, 1, CV_AA);
			}
		}
		//检测
		if (m_mode == MODE_DETECTION) {
			// Don't do anything special.
			////别做什么特别的事。  
		}
		//识别
		else if (m_mode == MODE_RECOGNITION) {
			if (gotFaceAndEyes) {
				// Generate a face approximation by back-projecting the eigenvectors & eigenvalues.
				//通过反向投影特征向量和特征值来生成人脸近似。
				Mat reconstructedFace;
				reconstructedFace = reconstructFace(model, preprocessedFace);

				// Verify whether the reconstructed face looks like the preprocessed face, otherwise it is probably an unknown person.
				//验证重建的人脸是否与预处理的人脸相似，否则可能是未知的人。
				double similarity = getSimilarity(preprocessedFace, reconstructedFace);
				string outputStr;
				if (similarity < UNKNOWN_PERSON_THRESHOLD) {
					// Identify who the person is in the preprocessed face image.
					//识别预处理的人脸图像中的人。
					int identity = model->predict(preprocessedFace);
					name = mysql.selectNameByLabel(identity);
					outputStr = name;
					id = mysql.selectIdByLabel(identity);
					cout << outputStr << endl;
				}
				else {
					// Since the confidence is low, assume it is an unknown person.
					//可信度低，认为是未知的人
					outputStr = "Unknow";
					name.clear();
					id.clear();
				}
				cout << "Identity: " << outputStr << ". Similarity: " << similarity << endl;

				// Show the confidence rating for the recognition in the mid-top of the display.
				//在显示屏的中上部显示识别的置信度。
				int cx = (displayedFrame.cols - faceWidth) / 2;
				Point ptBottomRight = Point(cx - 5, BORDER + faceHeight);//右下角
				Point ptTopLeft = Point(cx - 15, BORDER);//左上角

				RNG g_rng(12345);
				Scalar color = Scalar(g_rng.uniform(0, 255), g_rng.uniform(0, 255), g_rng.uniform(0, 255));//所取的颜色任意值
				putText(displayedFrame, outputStr, ptBottomRight, FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 255));//打印文本 

				// Draw a gray line showing the threshold for an "unknown" person.
				//画一条灰线，显示一个“未知”的人的阈值。
				Point ptThreshold = Point(ptTopLeft.x, ptBottomRight.y - (1.0 - UNKNOWN_PERSON_THRESHOLD) * faceHeight);
				rectangle(displayedFrame, ptThreshold, Point(ptBottomRight.x, ptThreshold.y), CV_RGB(200, 200, 200), 1, CV_AA);
				// Crop the confidence rating between 0.0 to 1.0, to show in the bar.
				//裁剪0.0到1.0之间的置信度，以在栏中显示。
				double confidenceRatio = 1.0 - min(max(similarity, 0.0), 1.0);
				Point ptConfidence = Point(ptTopLeft.x, ptBottomRight.y - confidenceRatio * faceHeight);
				// Show the light-blue confidence bar.//显示淡蓝色的置信条。
				rectangle(displayedFrame, ptConfidence, ptBottomRight, CV_RGB(0, 255, 255), CV_FILLED, CV_AA);

				if (!name.empty() && !id.empty())
					waitKey(200);	//等待200ms，方便点击签到

			}
		}
		//签到
		else if (m_mode == MODE_SIGN_IN){
			cout << "into sign in" << endl;
			string text;
			if (!id.empty()){
				//获取当前时间
				struct tm *newtime;
				time_t t;
				time(&t);
				newtime = localtime(&t);
				int hour = newtime->tm_hour;
				char tmpbuf[128];
				strftime(tmpbuf, 128, "%Y-%m-%d %X", newtime);
				string nowtime = tmpbuf;
				cout << "now time:" + nowtime << endl;
				//签到
				if (hour < 12){
					if (hour < 6){
						text = "this is not a time to sign in";
					}
					else if (hour >= 9){
						text= "be late";
					}
					else {
						if (mysql.insertSignIn(id,nowtime)){
							text = "Sign in successfully";
						}
						else text = "Sign in unsuccessfully";
					}
				}
				//签退
				else{
					if (hour < 17){
						text="this is not a time to sign out" ;
					}
					else {
						if (mysql.insertSignIn(id, nowtime)){
							text = "Sign out successfully";
							if (mysql.hasAttendOnMorning(id, *newtime)){
								mysql.attended(id);
							}
						}
						else text = "Sign out unsuccessfully";
					}
				}
			}
			else text = "id is unknow";

			cout << text << endl;
			//显示闪光
			if (faceRect.width>0&&faceRect.height>0){
				Mat displayedFaceRegion = displayedFrame(faceRect);
				displayedFaceRegion += CV_RGB(90, 90, 90);
				int cx = (displayedFrame.cols - faceWidth) / 2;
				Point ptBottomRight = Point(cx - 5, BORDER + faceHeight);
				RNG g_rng(12345);
				Scalar color = Scalar(g_rng.uniform(0, 255), g_rng.uniform(0, 255), g_rng.uniform(0, 255));//所取的颜色任意值
				putText(displayedFrame, text, ptBottomRight, FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 255));//打印文本 
			}

			m_mode = MODE_RECOGNITION;
		}
		//错误：无效的运行模式
		else {
			cerr << "ERROR: Invalid run mode " << m_mode << endl;
			exit(1);
		}


		//显示帮助
		string help;
		Rect rcHelp;
		if (m_mode == MODE_DETECTION)//检测
			help = "Click [Recognition] to recognition faces.";
		else if (m_mode == MODE_SIGN_IN) //签到
			help = "Please wait while your sign in successfully";
		else if (m_mode == MODE_RECOGNITION) //辨认
			help = "now recognizing.";
		if (help.length() > 0) {
			// Draw it with a black background and then again with a white foreground.
			// Since BORDER may be 0 and we need a negative position, subtract 2 from the border so it is always negative.
			//用黑色背景画，然后再用白色前景画。 显示帮助
			//因为BORDER可能是0，我们需要一个负位置，所以从BORDER减去2，它总是负的。
			float txtSize = 0.4;
			drawString(displayedFrame, help, Point(BORDER, -BORDER - 2), CV_RGB(0, 0, 0), txtSize);
			// Black shadow.背景
			rcHelp = drawString(displayedFrame, help, Point(BORDER + 1, -BORDER - 1), CV_RGB(255, 255, 255), txtSize);  // White text. 前景文档
		}

		// Show the current mode.  显示当前模式。
		if (m_mode >= 0 && m_mode < MODE_END) {
			string modeStr = "MODE: " + string(MODE_NAMES[m_mode]);
			drawString(displayedFrame, modeStr, Point(BORDER, -BORDER - 2 - rcHelp.height), CV_RGB(0, 0, 0));       // Black shadow 黑阴影
			drawString(displayedFrame, modeStr, Point(BORDER + 1, -BORDER - 1 - rcHelp.height), CV_RGB(0, 255, 0)); // Green text 绿文本
		}

		// Draw the GUI buttons into the main image.
		//将GUI按钮绘制到主图像中。
		//增加 删除 调试 训练  删除单个人脸功能
		BtnRecFace = drawButton(displayedFrame, "Recognition", Point(BORDER, BORDER));
		BtnSignIn = drawButton(displayedFrame, "Sign In", Point(BtnRecFace.x, BtnRecFace.y + BtnRecFace.height), BtnRecFace.width);

		// Show the camera frame on the screen.
		//在屏幕上显示摄像机画面。
		imshow(windowName, displayedFrame);

		// IMPORTANT: Wait for atleast 20 milliseconds, so that the image can be displayed on the screen!
		// Also checks if a key was pressed in the GUI window. Note that it should be a "char" to support Linux.
		//重要提示：请至少等待20毫秒，以便在屏幕上显示图像！
		//同时检查是否在GUI窗口中按下了键。注意，它应该是一个“char”来支持Linux。
		char keypress = waitKey(20);  // This is needed if you want to see anything!

		if (keypress == VK_ESCAPE) {   // Escape Key
			// Quit the program!
			//退出按键
			break;
		}

	}//end while

}



int main(int argc, char *argv[])
{
	CascadeClassifier faceCascade;
	CascadeClassifier eyeCascade1;
	CascadeClassifier eyeCascade2;
	VideoCapture videoCapture;

	cout << "WebcamFaceRec, by Shervin Emami (www.shervinemami.info), June 2012." << endl;
	cout << "Realtime face detection + face recognition from a webcam using LBP and Eigenfaces or Fisherfaces." << endl;
	cout << "Compiled with OpenCV version " << CV_VERSION << endl << endl;

	// Load the face and 1 or 2 eye detection XML classifiers.
	//加载人脸和1或2个眼睛检测XML分类器。
	initDetectors(faceCascade, eyeCascade1, eyeCascade2);

	cout << endl;
	cout << "Hit 'Escape' in the GUI window to quit." << endl;

	// Allow the user to specify a camera number, since not all computers will be the same camera number.允许用户指定摄像机号，因为并非所有计算机都是相同的摄像机号。
	int cameraNumber = 0;   // Change this if you want to use a different camera device.
	////如果要使用其他相机设备，请更改此项。
	if (argc > 1) {
		cameraNumber = atoi(argv[1]);
	}

	// Get access to the webcam.
	////访问网络摄像头。
	initWebcam(videoCapture, cameraNumber);

	// Try to set the camera resolution. Note that this only works for some cameras on
	// some computers and only for some drivers, so don't rely on it to work!
	//尝试设置相机分辨率。注意，这只适用于
	//一些电脑和只为一些硬件驱动，所以不要依赖它来工作！
	videoCapture.set(CV_CAP_PROP_FRAME_WIDTH, DESIRED_CAMERA_WIDTH);
	videoCapture.set(CV_CAP_PROP_FRAME_HEIGHT, DESIRED_CAMERA_HEIGHT);

	// Create a GUI window for display on the screen.
	////创建一个在屏幕上显示的GUI窗口。
	namedWindow(windowName); // Resizable window, might not work on Windows.
	//可调整大小的窗口，可能无法在窗口上工作。

	// Get OpenCV to automatically call my "onMouse()" function when the user clicks in the GUI window.
	//让OpenCV在用户单击GUI窗口时自动调用我的“onMouse（）”函数。
	setMouseCallback(windowName, onMouse, 0);

	// Run Face Recogintion interactively from the webcam. This function runs until the user quits.
	//从网络摄像头以交互方式运行面部重新编码。此函数一直运行到用户退出
	recognitionAndSignIn(videoCapture, faceCascade, eyeCascade1, eyeCascade2);
	return 0;
}
