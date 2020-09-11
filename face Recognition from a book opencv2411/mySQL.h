#ifndef MYSQL_H
#define MYSQL_H

#include <mysql.h>
#include <string>
#include <vector>
#include <sstream>

using std::vector;
using std::string;

class mySQL
{
private:
	MYSQL mysql;
	MYSQL_RES* res;
public:
	
	mySQL();
	~mySQL();

	bool ConnectDatabase();
	void FreeConnect();
	bool selectOneRow(char* sql,vector<string> &data);
	bool selectRows(char* sql, vector<vector<string>>& data);
	bool updata(char* sql);
	bool insertStudent(string pictureName,int label);
	bool insertSignIn(string staffNo,string nowtime);
	bool insertAttend(string staffNo);
	bool attended(string staffNo);
	bool selectStudent(string id, vector<string>& data);
	bool selectSignInTable(string id,vector<vector<string>>& data);
	bool deleteStudentTable();
	bool deleteAttend();
	bool hasAttendOnMorning(string id,struct tm newtime);
	string selectNameByLabel(int label);
	string selectIdByLabel(int label);
	string selectAttendedNum(string id);
	bool selectAllAttend(vector<vector<string>>& data);
};





#endif