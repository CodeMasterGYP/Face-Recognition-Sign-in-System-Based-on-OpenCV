#include <iostream>
#include "mySQL.h"
#include <string>
#include <vector>
#include <cstdio>
#include <sstream>
#include <time.h>
using namespace std;

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
int main(){
	
	mySQL mysql;
	//ͳ��
	vector<vector<string>> data;
	if (mysql.selectAllAttend(data)){
		cout << "staffNo     " << '|' << "number of attendence" << endl;
		for (int i = 0; i < data.size(); i++){
			vector<string> temp= data[i];
			for (int j = 0; j < temp.size(); j++){
				cout<<temp[j] << '|';
			}
			cout << endl;
		}
	}
	else cout << "empty";
	//��ѯĳԱ��ǩ����¼
	//�鿴�����˵�ǩ����¼

	cout <<endl<< "end" << endl;
    getchar();

}