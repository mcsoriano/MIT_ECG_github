#include <iostream>
#include <Eigen/Dense>
using namespace Eigen;
using namespace std;


void mapminmax(double V[], int dim){
double max=V[0];
double min=V[0];
double x0=0;
double s=0;

//locate max and min
for (int i=1; i<dim; i++){
if (V[i]>max) max=V[i];
if (V[i]<min) min=V[i];
}
 


if (max!=min){
  x0=1+2.*min/(max-min);
  s=2./(max-min);
  for (int i=0; i<dim; i++){
  V[i]=V[i]*s-x0;
  }
  }
else{
  cerr << "Nonexistent range" << endl;
}

return;
}
