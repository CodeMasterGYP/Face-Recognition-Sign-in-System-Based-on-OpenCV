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
const float UNKNOWN_PERSON_THRESHOLD = 0.7f;
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

// Preprocess left & right sides of the face separately, in case there is stronger light on one side.
/*
*分别对脸部左右两侧进行预处理，以防一侧光线较强。
*/
const bool preprocessLeftAndRightSeparately = true;



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


//初始化数据库连接
mySQL mysql;

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



//从文件获取图片路径和图片名称
//path 目录路径
//files 存放图片的路径
//name 存放图片名
void getFiles(string path, vector<string>& files, vector<string>& name)
{
	//文件句柄	
	__int64 hFile = 0;
	int i;
	string p;
	p = path + "\\*.jpg";
	//文件信息	
	struct __finddata64_t fileinfo;   //存储文件各种信息的结构体    			         	
	hFile = _findfirst64(p.c_str(), &fileinfo);
	i = hFile;
	if (i != (-1))
	{
		//assign给string类变量赋值
		//c_str()函数返回一个指向正规C字符串的指针, 内容与本string串相同.，这是为了与c语言兼容，在c语言中没有string类型，故必须通过string类对象的成员函数c_str()把string 对象转换成c中的字符串样式。
		do{
			files.push_back(p.assign(path).append("\\").append(fileinfo.name));
			name.push_back(fileinfo.name);
		} while (_findnext64(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}
//从文件中获取人脸并收集人脸和标签
int collectFace(CascadeClassifier &faceCascade, CascadeClassifier &eyeCascade1, CascadeClassifier &eyeCascade2, vector<Mat>& preprocessedFaces,vector<int>& faceLabels){
	mysql.deleteStudentTable();
	string folder = "pictures\\origin";//文件目录
	vector<string> files;//存放图片路径
	vector<string> name;//存放图片名
	string id;//id:student's id
	string oldId="";
	getFiles(folder, files, name);
	int count = 0;//记录人数
	for (int i = 0; i < files.size(); i++)
	{
		int pos = name[i].find_first_of('_');
		id = name[i].substr(0, pos); cout << "id " << id << endl;
		//插入学生信息
		if (id != oldId){
			count++;
			mysql.insertStudent(name[i],count);
			oldId = id;
		}
		cout << count << endl;

		Mat src = imread(files[i]); cout << "files " << files[i] << endl;
		Mat preprocessedFace = getPreprocessedFace(src, faceWidth, faceCascade, eyeCascade1, eyeCascade2, preprocessLeftAndRightSeparately);//要改preprocessFace.cpp
		//如果预处理成功
		if (preprocessedFace.data){

			Mat mirroredFace;
			flip(preprocessedFace, mirroredFace, 1);
		//imshow("preprocessedFace", preprocessedFace);
		//imshow("mirroredFace", mirroredFace);
		//getchar();

		//将人脸图像添加到检测到的人脸列表中。
			preprocessedFaces.push_back(preprocessedFace);
			preprocessedFaces.push_back(mirroredFace);
			faceLabels.push_back(count);
			faceLabels.push_back(count);
			string path = "pictures\\train\\" + name[i];
			cv::imwrite(path, preprocessedFace);
		}

		
	}
	return count;

}
//训练模型
void trainFace(int count,const vector<Mat> preprocessedFaces,const vector<int> faceLabels){
	//检查是否有足够的数据进行训练。对于特征脸，如果我们愿意，我们可以只学习一个人，但是对于Fisherfaces.xml，
	//我们至少需要两个人否则它会失败！
	//是否拥有足够数据
	bool haveEnoughData = true;
	//判断是否满足fisherface所需要两个人
	if (strcmp(facerecAlgorithm, "FaceRecognizer.Fisherfaces") == 0) {
		if (count < 2) {
			cout << "Warning: Fisherfaces needs atleast 2 people, otherwise there is nothing to differentiate! Collect more data ..." << endl;
			haveEnoughData = false;
		}
	}
	//特征脸需要一个人脸数据
	if (count < 1 || preprocessedFaces.size() <= 0 || preprocessedFaces.size() != faceLabels.size()) {
		cout << "Warning: Need some training data before it can be learnt! Collect more data ..." << endl;
		haveEnoughData = false;
	}
	//如果有数据  进行特征脸算法
	if (haveEnoughData) {
		// Start training from the collected faces using Eigenfaces or a similar algorithm.
		//使用特征面或类似算法从收集的面开始训练。
		//使用训练模型
		Ptr<FaceRecognizer> model = learnCollectedFaces(preprocessedFaces, faceLabels, facerecAlgorithm);
		model->save("faceClass.xml");
		printf("训练完成,模型输出为faceClass.xml文件\n按任意键结束\n");
		getchar();
	}
}

int main(int argc, char *argv[])
{
	CascadeClassifier faceCascade;
	CascadeClassifier eyeCascade1;
	CascadeClassifier eyeCascade2;
	VideoCapture videoCapture;
	vector<Mat> preprocessedFaces;
	vector<int> faceLabels;

	cout << "Realtime face detectio using Fisherfaces." << endl;
	cout << "Compiled with OpenCV version " << CV_VERSION << endl << endl;

	// Load the face and 1 or 2 eye detection XML classifiers.
	//加载人脸和1或2个眼睛检测XML分类器。
	initDetectors(faceCascade, eyeCascade1, eyeCascade2);

	int count = 0;
	count = collectFace(faceCascade, eyeCascade1, eyeCascade2, preprocessedFaces, faceLabels);
	trainFace(count, preprocessedFaces, faceLabels);

	return 0;
}
