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
//�������ݿ�  
bool mySQL::ConnectDatabase()
{
	//��ʼ��mysql  
	mysql_init(&mysql);  //����mysql�����ݿ�  
	const char host[] = "localhost";
	const char user[] = "root";
	const char psw[] = "root";
	const char table[] = "signin";
	const int port = 3306;
	//����false������ʧ�ܣ�����true�����ӳɹ�  
	if (!(mysql_real_connect(&mysql, host, user, psw, table, port, NULL, 0)))
		//�м�ֱ����������û��������룬���ݿ������˿ںţ�����дĬ��0����3306�ȣ���������д�ɲ����ٴ���ȥ  
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

//�ͷ���Դ  
void mySQL::FreeConnect()
{
	mysql_close(&mysql);	 //�ر�һ�����������ӡ�
	cout << "�ѶϿ�����" << endl;
}

//���ҵ����¼
//sql sql��ѯ���
//data �����¼�ĸ���
bool mySQL::selectOneRow(char* sql, vector<string> &data)
{
	data.clear();
	//�����ַ���
	mysql_query(&mysql, "set names utf8");
	//����0 ��ѯ�ɹ�������1��ѯʧ��  
	if (mysql_query(&mysql,sql))        //ִ��SQL���  
	{
		printf("Query failed (%s)\n", mysql_error(&mysql));
		return false;
	}
	else
	{
		printf("query success\n");
	}
	
	//���sql�������󷵻صĽ���� 
	MYSQL_RES* res = mysql_store_result(&mysql);
	//�������  
	int col = mysql_num_fields(res);  // ��ȡ����  

	MYSQL_ROW column;//һ�������ݵ����Ͱ�ȫ�ı�ʾ��column[i]��ʾ�����е���
	while (column = mysql_fetch_row(res))
	{
		for (int i = 0; i<col; i++)
			data.push_back(column[i]);
	}
	//�ͷ�һ���������ʹ�õ��ڴ档
	mysql_free_result(res);  
	return true;
}
//��ѯ����
//sql sql��ѯ���
//data �����¼�ĸ���
bool mySQL::selectRows(char* sql,vector<vector<string>> & data)
{
	data.clear();
	mysql_query(&mysql, "set names utf8");
	//����0 ��ѯ�ɹ�������1��ѯʧ��  
	if (mysql_query(&mysql, sql))        //ִ��SQL���  
	{
		printf("Query failed (%s)\n", mysql_error(&mysql));
		return false;
	}
	else
	{
	    //���sql�������󷵻صĽ���� 
		MYSQL_RES* res = mysql_store_result(&mysql);
		//�������  
		int col = mysql_num_fields(res);  // ��ȡ����  

		MYSQL_ROW column;//һ�������ݵ����Ͱ�ȫ�ı�ʾ��column[i]��ʾ�����еĵ�i��
		while (column = mysql_fetch_row(res))
		{
			vector<string> temp;
			for (int i = 0; i < col; i++){
				temp.push_back(column[i]);
			}
			data.push_back(temp);
		}
		//�ͷ�һ���������ʹ�õ��ڴ档
		mysql_free_result(res);
		printf("query success\n");
		return true;
	 }
}
//��ӣ����£�ɾ����Ϣ
//sql sql���
bool mySQL::updata(char* sql){

	mysql_query(&mysql, "set names utf8");

	if (mysql_query(&mysql, sql))        //ִ��SQL���  
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

//����һ��studentԪ��
//pictureName:ͼƬ������ʽΪstudentID_studentName_1.jpg
bool mySQL::insertStudent(string pictureName,int label){
	string name; //name:student's name
	string id;//id:student's id

	int posFirst = pictureName.find_first_of('_');
	id = pictureName.substr(0, posFirst);
	int posSecond = pictureName.find_last_of('_');
	name = pictureName.substr(posFirst + 1, posSecond - posFirst - 1);

	string command = "insert into Staff (id,label,name) values ('" + id + "'," +to_string(label)+",'"+ name + "')";//����utf-8���������ĵ�����
	char* sql = new char[100];
	strcpy(sql, command.c_str());
	if (updata(sql)){
		if (insertAttend(id))
			return true;
	}
	return false;
}

//����һ��signInTableԪ��
//staffNo:Ա�����
bool mySQL::insertSignIn(string staffNo,string nowtime){
	//����updata
	string command = "insert into signintable(time,staffNo) values ('" + nowtime + "'," + staffNo + ")";//����utf-8���������ĵ�����
	char* sql = new char[100];
	strcpy(sql, command.c_str());
	if (updata(sql))
		return true;
	else return false;
}
bool mySQL::insertAttend(string staffNo){
	int num = 0;
	//����updata
	string command = "insert into attend(staffNo,num) values ('" + staffNo + "'," + to_string(num) + ")";//����utf-8���������ĵ�����
	char* sql = new char[100];
	strcpy(sql, command.c_str());
	if (updata(sql))
		return true;
	else return false;
	return true;
}


//��ѯһ��studentԪ��
bool mySQL::selectStudent(string id, vector<string>& data){
	string command = "select * from Staff where id=" + id + ";";
	char* sql = new char[100];
	strcpy(sql, command.c_str());
	if (selectOneRow(sql,data))
		return true;
	else return false;
}

//��ѯ����signInTable
bool mySQL::selectSignInTable(string id, vector<vector<string>>& data){
	string command = "select * from signintable where staffNo=" + id + ";";
	char* sql = new char[100];
	strcpy(sql, command.c_str());
	if (selectRows(sql, data))
		return true;
	else return false;
}
//���attend��
bool mySQL::deleteAttend(){
	string command = "delete from attend";
	char* sql = new char[100];
	strcpy(sql, command.c_str());
	if (updata(sql)){
		cout << "�ɹ����attend��" << endl;
		return true;
	}
	else{
		cout << "���attend��ʧ��" << endl;
		return false;
	}
}
//���student��
bool mySQL::deleteStudentTable(){
	string command = "delete from Staff";
	char* sql = new char[100];
	strcpy(sql, command.c_str());
	if (deleteAttend()){
		if (updata(sql)){
			cout << "�ɹ����student��" << endl;
			return true;
		}
	}
	cout << "���student��ʧ��" << endl;
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
//��ȡ���ڴ���
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
//���ڴ�����1
bool mySQL::attended(string staffNo){
	string command = "update attend set num =num+1 where staffNo=" + staffNo ;
	char* sql = new char[100];
	strcpy(sql, command.c_str());
	if (updata(sql))
		return true;
	else return false;
}
//ǩ��ʱ����鵱���Ƿ���ǩ��
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
//��ȡ���еĿ��ڼ�¼
bool mySQL::selectAllAttend(vector<vector<string>>& data){
	string command = "select * from attend";
	char* sql = new char[100];
	strcpy(sql, command.c_str());
	if (selectRows(sql, data))
		return true;
	else return false;
}