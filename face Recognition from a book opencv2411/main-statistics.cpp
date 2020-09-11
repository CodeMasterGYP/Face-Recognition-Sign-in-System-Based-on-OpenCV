#include <iostream>
#include "mySQL.h"
#include <string>
#include <vector>
#include <cstdio>
#include <sstream>
#include <time.h>
using namespace std;

void getAllAttendence();
int fromString(string str);
string toString(int temp);

mySQL mysql;

int main(){
	getAllAttendence(); 


	return 0;
}
int fromString(string str){
	int out;
	stringstream in(str);
	in >> out;
	return out;
}
string toString(int temp){
	string out;
	stringstream in;
	in << temp;
	in >> out;
	return out;
}
//统计 所有人的考勤次数
void getAllAttendence(){
	vector<vector<string>> data;
	if (mysql.selectAllAttend(data)){
		cout << "staffNo     " << '|' << "number of attendence" << endl;
		for (int i = 0; i < data.size(); i++){
			vector<string> temp = data[i];
			for (int j = 0; j < temp.size(); j++){
				cout << temp[j];
				if (j == 0){
					cout << "|";
				}
			}
			cout << endl;
		}
	}
	else cout << "empty";
	//查询某员工签到记录
	//查看所有人的签到记录
	cout << endl << "end" << endl;
	getchar();
}