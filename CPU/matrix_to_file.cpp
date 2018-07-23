//Function to import data from txt to eigen_matrix
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <new>
#include <Eigen/Dense>
using namespace Eigen;
using namespace std;

bool matrix_to_file(Eigen::MatrixXd& mat, int dim1, int dim2, string fname){
ofstream outfile;		
outfile.open(fname.c_str());		
if (outfile.is_open()){
cout << "Saving "<< fname << " data..." << endl;	
	for (int i=0; i < dim1; i++) {		
	for (int j=0; j<dim2; j++){
       if (outfile << mat(i,j) << " ");
	   else{ 
       cerr << "storage interrupted!"<< endl;
	   (outfile.close());
	   return 0;
	   }
    }
    outfile << endl;							
    }
(outfile.close());					
return 1;
}

else{
cerr << "Error opening file" << endl; 
return 0;
}
}
