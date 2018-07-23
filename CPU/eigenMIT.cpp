// cMIT declaration
// Comment/Uncomment to choose read parameters (leadA, leadB, AAMI_class, MIT_class, RR, paciente, DS1DS2, SVE_class)
// Declarations for class MIT, where a we have

#include <iostream>
#include <fstream>
#include <Eigen/Dense>
using namespace Eigen;
using namespace std;

class cMIT
{
public:
   cMIT();				// Set up empty directory of entries
   ~cMIT();				// Deallocate the entry list.
   void Readfile();
   int nbeats;
   int npoints;
   int nRR;


Eigen::MatrixXd lead;


   double AAMI_class[ndb];
   char MIT_class[ndb];
   double RR[ndb][nRRdb];


   double paciente[ndb];
   double DS1DS2[ndb];			// Test/train tag 1 (test), (0) train
   double SVE_class[ndb];		
//   MatrixXd SVE_class;


};


//cMIT CLASS METHODS /////////////////////////////////////////////////////////////////////////////////////
// 0 initialisation constructor										//
cMIT::cMIT(){												//
													//
nbeats=ndb; 												//
npoints=pdb;												//
nRR=nRRdb;												//
lead = MatrixXd::Zero(pdb,ndb);										//
//SVE_class = MatrixXd::Zero(ndb,1);									//
cout << "MIT class object created" << endl;								//
}													//
													//
//destructor												//
cMIT::~cMIT(void)											//
{													//
cout << "MIT class object is being deleted" << endl;							//
}													//
													//
													//
//read from files method										//
void cMIT::Readfile(){											//
if (readA){												//
//Read leadA												//    
   cout << "Reading leadA data,";									//
   ifstream leadAfile;											//
   leadAfile.open(mitdblocation"leadA.txt");								//
   if (leadAfile.is_open()) {
   cout << " txt file opened ";										//
   MatrixXd leadAaux = lead.transpose();
													//
   for (int i = 0; i < ndb; i++) {									//
	for (int j=0; j<pdb; j++){									//
        //leadAfile >> leadA[i][j];									//
        leadAfile >> leadAaux(i,j);									//
	}												//    
   }
   lead = leadAaux.transpose();
   cout << "OK (" << lead.rows() << "," << lead.cols() << ")" << endl;                                  //
   }													//
   leadAfile.close();											//
}
else if (readB){											//
//Read leadB												//
   cout << "Reading leadB data,";									//
   ifstream leadBfile;											//
   leadBfile.open(mitdblocation"leadB.txt");								//
   if (leadBfile.is_open()) {
   cout << " txt file opened ";										//
   MatrixXd leadBaux = lead.transpose();
													//
   for (int i = 0; i < ndb; i++) {									//
	for (int j=0; j<pdb; j++){									//
        //leadBfile >> leadB[i][j];									//
        leadBfile >> leadBaux(i,j);									//
	}												//    
   }
   lead = leadBaux.transpose();
   cout << "OK (" << lead.rows() << "," << lead.cols() << ")" << endl;                                  //
   }													//
   leadBfile.close();
}
/*													//
//Read AAMI_class											//
   cout << "Reading AAMI_class data,";									//
   ifstream AAMI_classfile;										//
   AAMI_classfile.open(mitdblocation"AAMI_class.txt");							//
   if (AAMI_classfile.is_open()) cout << " txt file opened ";						//
   cout << AAMI_classfile.good();									//
   for (int i = 0; i < ndb; i++) {									//
        AAMI_classfile >> AAMI_class[i];								//
   }													//    
   cout << AAMI_classfile.eof();									//
   cout << AAMI_classfile.good() << endl;								//
   AAMI_classfile.close();										//
													//
													//
//Read MIT_class											//
   cout << "Reading MIT_class data,";									//
   ifstream MIT_classfile;										//
   //MIT_classfile.open(mitdblocation"MIT_class.txt");      //problems with char MATLAB format	        //
   MIT_classfile.open("MITclass.txt");									//
   if (MIT_classfile.is_open()) cout << " txt file opened ";						//
   cout << MIT_classfile.good();									//
   for (int i = 0; i < ndb; i++) {									//
        MIT_classfile >> MIT_class[i];									//
   }													//    
   cout << MIT_classfile.eof();										//
   cout << MIT_classfile.good() << endl;								//
   MIT_classfile.close();										//
*/													//
//Read RR												//
   cout << "Reading RR data,";										//
   ifstream RRfile;											//
   RRfile.open(mitdblocation"RR_0_1_mean.txt");								//
   if (RRfile.is_open()){
   for (int i = 0; i < ndb; i++) {									//
	for (int j=0; j<nRRdb; j++){									//
        RRfile >> RR[i][j];										//
	}
   }                                        								//    
   cout << "OK (ndb , nRRdb)" << endl;
   }													//
   RRfile.close();											//
    													//
//Read paciente												//
   cout << "Reading paciente data,";									//
   ifstream pacientefile;										//
   pacientefile.open(mitdblocation"paciente.txt");							//
   if (pacientefile.is_open()) cout << " txt file opened ";						//
   cout << pacientefile.good();										//
   for (int i = 0; i < ndb; i++) {									//
        pacientefile >> paciente[i];									//
    }													//
   cout << pacientefile.eof();										//
   cout << pacientefile.good() << endl;									//
   pacientefile.close();										//
													//
//Read DS1DS2												//
   cout << "Reading DS1DS2 data,";									//
   ifstream DS1DS2file;											//
   DS1DS2file.open(mitdblocation"DS1DS2.txt");								//
   if (DS1DS2file.is_open()) cout << " txt file opened ";						//
   cout << DS1DS2file.good();										//
   for (int i = 0; i < ndb; i++) {									//
        DS1DS2file >> DS1DS2[i];									//
    }													//
   cout << DS1DS2file.eof();										//
   cout << DS1DS2file.good() << endl;									//
   DS1DS2file.close();											//
													//
//Read SVE_class											//
   cout << "Reading SVE_class data,";									//
   ifstream SVE_classfile;										//
   SVE_classfile.open(mitdblocation"SVE_class.txt");							//
   if (SVE_classfile.is_open()) cout << " txt file opened ";						//
   cout << SVE_classfile.good();									//
   for (int i = 0; i < ndb; i++) {									//
        SVE_classfile >> SVE_class[i];									//
    }													//
   cout << SVE_classfile.eof();										//
   cout << SVE_classfile.good() << endl;								//
   SVE_classfile.close();										//
													//
}													//
////////////////////////////////////////////////////////////////////////////////////////cMIT CLASS METHODS 
