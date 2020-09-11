#include "mySQL.h"
#include <cstdio>
#include <mysql.h>
#include <iostream>
#include <string>
#include <vector>
#include <time.h>
#include <sstream>

using namespace std;

mySQL::mySQL(){
	ConnectDatabase();
}
mySQL::~mySQL(){
	FreeConnect();
}
//连接数据库  
bool mySQL::ConnectDatabase()
{
	//初始化mysql  
	mysql_init(&mysql);  //连接mysql，数据库  
	const char host[] = "localhost";
	const char user[] = "root";
	const char psw[] = "root";
	const char table[] = "signin";
	const int port = 3306;
	//返回false则连接失败，返回true则连接成功  
	if (!(mysql_real_connect(&mysql, host, user, psw, table, port, NULL, 0)))
		//中间分别是主机，用户名，密码，数据库名，端口号（可以写默认0或者3306等），可以先写成参数再传进去  
	{
		printf("Error connecting to database:%s\n", mysql_error(&mysql));
		return false;
	}
	else
	{
		printf("DataBase is Connected...\n");
		return true;
	}
}

//释放资源  
void mySQL::FreeConnect()
{
	mysql_close(&mysql);	 //关闭一个服务器连接。
	cout << "已断开连接" << endl;
}

//查找单项记录
//sql sql查询语句
//data 单项记录的各列
bool mySQL::selectOneRow(char* sql, vector<string> &data)
{
	data.clear();
	//设置字符集
	mysql_query(&mysql, "set names utf8");
	//返回0 查询成功，返回1查询失败  
	if (mysql_query(&mysql,sql))        //执行SQL语句  
	{
		printf("Query failed (%s)\n", mysql_error(&mysql));
		return false;
	}
	else
	{
		printf("query success\n");
	}
	
	//获得sql语句结束后返回的结果集 
	MYSQL_RES* res = mysql_store_result(&mysql);
	//获得数据  
	int col = mysql_num_fields(res);  // 获取列数  

	MYSQL_ROW column;//一个行数据的类型安全的表示，column[i]表示数据行的列
	while (column = mysql_fetch_row(res))
	{
		for (int i = 0; i<col; i++)
			data.push_back(column[i]);
	}
	//释放一个结果集合使用的内存。
	mysql_free_result(res);  
	return true;
}
//查询多行
//sql sql查询语句
//data 多项记录的各列
bool mySQL::selectRows(char* sql,vector<vector<string>> & data)
{
	data.clear();
	mysql_query(&mysql, "set names utf8");
	//返回0 查询成功，返回1查询失败  
	if (mysql_query(&mysql, sql))        //执行SQL语句  
	{
		printf("Query failed (%s)\n", mysql_error(&mysql));
		return false;
	}
	else
	{
	    //获得sql语句结束后返回的结果集 
		MYSQL_RES* res = mysql_store_result(&mysql);
		//获得数据  
		int col = mysql_num_fields(res);  // 获取列数  

		MYSQL_ROW column;//一个行数据的类型安全的表示，column[i]表示数据行的第i列
		while (column = mysql_fetch_row(res))
		{
			vector<string> temp;
			for (int i = 0; i < col; i++){
				temp.push_back(column[i]);
			}
			data.push_back(temp);
		}
		//释放一个结果集合使用的内存。
		mysql_free_result(res);
		printf("query success\n");
		return true;
	 }
}
//添加，更新，删除信息
//sql sql语句
bool mySQL::updata(char* sql){

	mysql_query(&mysql, "set names utf8");

	if (mysql_query(&mysql, sql))        //执行SQL语句  
	{
		printf("Query failed (%s)\n", mysql_error(&mysql));
		return false;
	}
	else
	{
		printf("updata success\n");
		return true;
	}
}

//插入一个student元组
//pictureName:图片名，格式为studentID_studentName_1.jpg
bool mySQL::insertStudent(string pictureName,int label){
	string name; //name:student's name
	string id;//id:student's id

	int posFirst = pictureName.find_first_of('_');
	id = pictureName.substr(0, posFirst);
	int posSecond = pictureName.find_last_of('_');
	name = pictureName.substr(posFirst + 1, posSecond - posFirst - 1);

	string command = "insert into Staff (id,label,name) values ('" + id + "'," +to_string(label)+",'"+ name + "')";//存在utf-8编码与中文的问题
	char* sql = new char[100];
	strcpy(sql, command.c_str());
	if (updata(sql)){
		if (insertAttend(id))
			return true;
	}
	return false;
}

//插入一个signInTable元组
//staffNo:员工编号
bool mySQL::insertSignIn(string staffNo,string nowtime){
	//调用updata
	string command = "insert into signintable(time,staffNo) values ('" + nowtime + "'," + staffNo + ")";//存在utf-8编码与中文的问题
	char* sql = new char[100];
	strcpy(sql, command.c_str());
	if (updata(sql))
		return true;
	else return false;
}
bool mySQL::insertAttend(string staffNo){
	int num = 0;
	//调用updata
	string command = "insert into attend(staffNo,num) values ('" + staffNo + "'," + to_string(num) + ")";//存在utf-8编码与中文的问题
	char* sql = new char[100];
	strcpy(sql, command.c_str());
	if (updata(sql))
		return true;
	else return false;
	return true;
}


//查询一个student元组
bool mySQL::selectStudent(string id, vector<string>& data){
	string command = "select * from Staff where id=" + id + ";";
	char* sql = new char[100];
	strcpy(sql, command.c_str());
	if (selectOneRow(sql,data))
		return true;
	else return false;
}

//查询多条signInTable
bool mySQL::selectSignInTable(string id, vector<vector<string>>& data){
	string command = "select * from signintable where staffNo=" + id + ";";
	char* sql = new char[100];
	strcpy(sql, command.c_str());
	if (selectRows(sql, data))
		return true;
	else return false;
}
//清空attend表
bool mySQL::deleteAttend(){
	string command = "delete from attend";
	char* sql = new char[100];
	strcpy(sql, command.c_str());
	if (updata(sql)){
		cout << "成功清空attend表" << endl;
		return true;
	}
	else{
		cout << "清空attend表失败" << endl;
		return false;
	}
}
//清空student表
bool mySQL::deleteStudentTable(){
	string command = "delete from Staff";
	char* sql = new char[100];
	strcpy(sql, command.c_str());
	if (deleteAttend()){
		if (updata(sql)){
			cout << "成功清空student表" << endl;
			return true;
		}
	}
	cout << "清空student表失败" << endl;
	return false;
}
string mySQL::selectNameByLabel(int label){
	string command = "select name from Staff where label=" + to_string(label) ;
	char* sql = new char[60];
	strcpy(sql, command.c_str());
	vector<string> data;
	string name;
	if(selectOneRow(sql, data))
		name = data.back();
	else name = "unknow";
	return name;
}
string mySQL::selectIdByLabel(int label){
	string command = "select id from Staff where label=" + to_string(label);
	char* sql = new char[60];
	strcpy(sql, command.c_str());
	vector<string> data;
	string id;
	if (selectOneRow(sql, data))
		id = data.back();
	else id = "unknow";
	return id;
}
//获取出勤次数
string mySQL::selectAttendedNum(string id){
	string command = "select num from attend where staffNo='" + id+"'";
	char* sql = new char[60];
	strcpy(sql, command.c_str());
	vector<string> data;
	string num = "-1";
	if (selectOneRow(sql, data))
		num = data.back();
	return num;
}
//出勤次数加1
bool mySQL::attended(string staffNo){
	string command = "update attend set num =num+1 where staffNo=" + staffNo ;
	char* sql = new char[100];
	strcpy(sql, command.c_str());
	if (updata(sql))
		return true;
	else return false;
}
//签退时，检查当天是否有签到
bool mySQL::hasAttendOnMorning(string id, struct tm newtime){
	int year = 1900 + newtime.tm_year;
	int month = 1 + newtime.tm_mon;
	int day = newtime.tm_mday;
	int hour = newtime.tm_hour;
	string command = "select time from signintable where staffNo=" + id + " and year(time)=" + to_string(year) + " and month(time)=" + to_string(month) + " and day(time)=" + to_string(day) + " and hour(time)<9 and hour(time)>=6";
	char* sql = new char[100];
	strcpy(sql, command.c_str());
	vector<string> data;
	if (selectOneRow(sql, data)){
		for (int i = 0; i < data.size(); i++){
			cout << data[i] << endl;
		}
		if (!data.empty()){
			return true;
		}
	}
	return false;
}
//获取所有的考勤记录
bool mySQL::selectAllAttend(vector<vector<string>>& data){
	string command = "select * from attend";
	char* sql = new char[100];
	strcpy(sql, command.c_str());
	if (selectRows(sql, data))
		return true;
	else return false;
}