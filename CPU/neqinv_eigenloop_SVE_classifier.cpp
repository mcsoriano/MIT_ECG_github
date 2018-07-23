//---------------------------------------------------------------------------------------------//
//                                                                                             //
//            EIGEN (ARRHYTHMIA) SVE CLASSIFIER (MIT ECG database)                             //
//                                                                                             //
//   lead (feat[60+3,100542])                                                                  //
//         MIT -> Train/Test                                       db class in eigenMIT.cpp    //
//---------------------------------------------------------------------------------------------//
#include <iomanip> 
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <new>
#include <Eigen/Dense>
#include "./config.h"
#include "neqinv_eigen_SVE_classifier.h"

using namespace Eigen;
using namespace std;

int main(int argc, char *argv[]) {
//mask number
string snum="";
string etas;
string gammas;
if (argc!=3) {
cerr << "ERROR: Usage prog eta gamma" << endl;
exit (EXIT_FAILURE);
}

//Time stamp
timestamp(0);


    cout << "//--------------------------------------------------------------//" << endl;
    cout << "//                                                              //" << endl;
    cout << "//   EIGEN  (ARRHYTHMIA) SVE CLASSIFIER (MIT ECG database)      //" << endl;
    cout << "//                                    db class in eigenMIT.cpp  //" << endl;
    cout << "//                                                              //" << endl;
    cout << "//   MIT -> Train/Test                                          //" << endl;
    cout << "//--------------------------------------------------------------//" << endl;

        
    cMIT* MIT = new cMIT();                   //MIT class object (on the heap), use -> address for members
    MIT->Readfile();                          //Read from files

    //Look for train-test tags
    long int Ntest=0;
    long int Ntrain=0;
    for (long int i=0; i<MIT->nbeats; i++){
        if (MIT->DS1DS2[i]==1) Ntest=Ntest+1;
    }
    Ntrain=MIT->nbeats-Ntest;
    VectorXi indextest(Ntest);
    VectorXi indextrain(Ntrain);
    
    long int i_test=0;
    long int i_train=0;
    for (long int i=0; i<MIT->nbeats; i++){
        if (MIT->DS1DS2[i]==1){
            indextest(i_test)=i;
            i_test++;
        }
        else{
            indextrain(i_train)=i;
            i_train++;
        }
    }
    cout << "Train/Test tags:   Ntrain=" << Ntrain << " Ntest=" << Ntest << endl;

        VectorXd target=VectorXd::Zero(Ntrain);
        //Get target    
        for (long int i=0; i<Ntrain; i++){
        target(i)=MIT->SVE_class[indextrain(i)];
        }


   //////////////////////
   // RESERVOIR        //
   //////////////////////


         //Feature Selection
         MatrixXd features;
         {
            //Use smaller data window
             MatrixXd Data;
             {
             int window_first=18;
             int window_last=77;
             int window_size=window_last-window_first+1;
             Data=MIT->lead.middleRows(window_first,window_size);
             }
             cout << "lead Data excerpt ("  << Data.rows() << "," << Data.cols() << ")" << endl;

    //Normalise Data (manually)
    double V[pdb];
    int size=Data.rows();
    for (int i=0; i<ndb; i++){
        for (int j=0; j<size; j++){
            V[j]=Data(j,i);
        }
        mapminmax(V, size);
        for (int j=0; j<size; j++){
            Data(j,i)=V[j];
        }
    }

    //Get RR
    cout << "RRs taken" << endl;
    MatrixXd RR(nRRdb,ndb);
    for (int i=0; i<ndb; i++){
        for (int j=0; j<nRRdb; j++){
            RR(j,i)=(MIT->RR[i][j]);
        }
    }

             cout << " RR     ("  << RR.rows() << "," << RR.cols() << ")" << endl;
    

    //Select relevant features to perform training
    cout << "Select data and RR as features" << endl;
    features=MatrixXd::Zero(Data.rows()+RR.rows(),ndb);
    features << Data, RR;
    }						//features selected
    cout << "features("  << features.rows() << "," << features.cols() << ")" << endl;
    delete MIT;


 
          //Echo State Network ESN 
          int nrofneurons=neur_def;                   //N=?
          long int nrofinputsteps=features.cols();    //ninputs=100542
          int diminputs=features.rows();              //nfeatures to input (MIT 58+3=61)
          long int nroftimesteps=nrofinputsteps;      //tsteps=inputs when 1 per timestep
          int sincro=1;                               //parameter to tune sinchronisation between neurons/input steps
          double eta=atof(argv[1]);                   //interconnection
          double gamma=atof(argv[2]);        	      //no scale with number of inputs

          cout << "ESN train parameters set" << endl;
          cout << "sincro=" << sincro << endl;
          cout << "gamma=" << gamma << endl;
          cout << "eta=" << eta << endl;
          cout << "diminputs=" << diminputs << endl;
          cout << "input_s=" << nrofinputsteps << endl;

          //Sigmoid Nonlinear Function
          double bias=bias_val;
          double exponent=exponent_val;
          cout << "Sigmoid Nonlinear Function set" << endl;
          cout << "bias=" << bias << endl;
          cout << "exponent=" << exponent << endl;  

//OUTER LOOP
for (int l=1; l<=k_max_iter; l++){
stringstream stream;
snum="";
stream << l;
snum = stream.str();

stringstream etasstream;
stringstream gammasstream;
etas="";
etasstream << setprecision(2) << fixed << eta;
etas = etasstream.str();
gammas="";
gammasstream << setprecision(2) << fixed << gamma;
gammas = gammasstream.str();

//File names
string wfname="weights" + snum + ".txt";
string outfname="class_test_" + snum + ".txt";
string maskfname=masklocation + snum + ".txt";
string maskbiasfname=masklocation + snum + "_bias.txt";

cout << "Mask and file number " << snum << endl;

   MatrixXd mask;
   MatrixXd mask_bias;
   MatrixXd trainreservoir;
   MatrixXd testreservoir;
   MatrixXd reservoir;
   {
          mask=MatrixXd::Zero(nrofneurons,diminputs);
          mask_bias=VectorXd::Zero(nrofneurons);

   //Load Mask (for reproducibility)
   {
   int filerows=mask.cols(); int filecols=mask.rows(); //filerows=feat filecols=N
   ifstream maskfile;
   maskfile.open(maskfname.c_str());						
   if (maskfile.is_open()) {
     for (int i = 0; i < filerows; i++) {							
        for (int j=0; j<filecols; j++){							
        maskfile >> mask(j,i);							
	}										
     }
     cout << "mask(" << mask.rows() << "," << mask.cols() << ")" << endl;                 
   }		
   else cerr << "Error i/o" << endl;     
   maskfile.close();	       

   int biasrows=mask_bias.rows();
   ifstream biasfile;
   biasfile.open(maskbiasfname.c_str());						
   if (biasfile.is_open()) {
     for (int i = 0; i<biasrows; i++) {							
        biasfile >> mask_bias(i);							
	}	
     cout << "mask_bias(" << mask_bias.rows() << "," << mask_bias.cols() << ")" << endl;                 									
     }
   else cerr << "Error i/o" << endl;     
   biasfile.close();	       
   }

  
           //Perform input iterations, obtain reservoir
           cout << "Performing input iterations" << endl;
           cout << "system=(1./(1+exp(-exponent*(gamma*input+eta*systemold)))-bias)" << endl;
           reservoir=MatrixXd::Zero(nrofneurons,nrofinputsteps);
              {
               VectorXd input=VectorXd::Zero(nrofneurons);
               VectorXd system=VectorXd::Zero(nrofneurons);    // col vector
               VectorXd systemold=VectorXd::Zero(nrofneurons); // col vector 
               VectorXd fcol=VectorXd::Zero(diminputs);
                   for (long int iter=0; iter<nroftimesteps; iter++){
                   fcol=features.col(iter);
                   input=mask*fcol+mask_bias;

                        for (int i=0; i<nrofneurons; i++){
                        system(i)=(1./(1+exp(-exponent*(gamma*input(i)+eta*systemold(i))))-bias);
                        }
                   reservoir.col(iter)=system;

                       for (int i=0; i<nrofneurons; i++){
                       int j=(i+nrofneurons-sincro)%nrofneurons;            
                       systemold(i)=system(j);
                       }
                   }                                    //end input iterations
              }                                         //end of input, systemold and system scope
       
          cout << "R.rows, R.cols = " << reservoir.rows() << " " << reservoir.cols() << endl;  

                //Train/Test
                //column, row
                trainreservoir=MatrixXd::Zero(Ntrain,nrofneurons);
                testreservoir=MatrixXd::Zero(Ntest,nrofneurons);

                  for (long int i=0; i<nrofneurons; i++){
                for (long int j=0; j<Ntrain; j++){
                  trainreservoir(j,i)=reservoir(i,indextrain(j));
                  }
                }
                cout << "TrainR: " << trainreservoir.rows() << "," << trainreservoir.cols() << endl;
                  for (int i=0; i<nrofneurons; i++){
                for (int j=0; j<Ntest; j++){
                  testreservoir(j,i)=reservoir(i,indextest(j));
                  }
                }
                cout << "TestR: " << testreservoir.rows() << "," << testreservoir.cols() << endl;

        }
        //Reservoirs were set


        //WEIGHT COMPUTATION///////////////////////////////////////////////////////////
        cout << "Find out weights (linear regression, neqinv)    " << endl;

        //Declare weight, output and target objects
        MatrixXd outweights=VectorXd::Zero(trainreservoir.rows()+1);
        MatrixXd output=VectorXd::Zero(Ntest);

        {   
        //EIGEN NEQINV LINEAR REGRESSION
        cout << "Computing..." << endl;
           {
            MatrixXd aux(Ntrain, 1+trainreservoir.cols());
            aux << MatrixXd::Ones(Ntrain,1), trainreservoir;
            cout << "aux: " << aux.rows() << "," << aux.cols() << endl;
            MatrixXd auxt(1+trainreservoir.cols(),Ntrain);
            auxt=aux.transpose(); 
            MatrixXd auxinv(1+trainreservoir.cols(), 1+trainreservoir.cols());
            auxinv=(auxt*aux).inverse();
            outweights=auxinv*auxt*target;
            }
            cout << "Process completed!" << endl;

           //Classification (test)
            {
            MatrixXd auxp(testreservoir.rows(), testreservoir.cols()+1);
            auxp << MatrixXd::Ones(Ntest,1), testreservoir; 
            output=auxp*outweights;
            }


      }//END of weight calculation and classification

            //SAVE WEIGHTS
            /*cout << "Saving weights" << endl;
                if (matrix_to_file(outweights, neur_def+1, 1, wfname)){
                }
                else{
                cerr << "...problems saving" << endl;
                }
	   */
            //SAVE OUTPUTS
            cout << "Saving outputs" << endl;
                if (matrix_to_file(output, Ntest, 1, outfname)){
                }
                else{
                cerr << "...problems saving" << endl;
                }
           
}  
    //Time stamp
    timestamp(1);

    return 0;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

//Time stamps    //run_info.txt
void timestamp(int icall)
{
    if (icall == 0) {
        int sys0 = system("hostname > run_info.txt");
        int sys1 = system("pwd >> run_info.txt");
        int sys2 = system("date >> run_info.txt && date +%s > timestamp1");
    }
    else if (icall == 1) {
        int sys3 = system("date >> run_info.txt && date +%s > timestamp2");
        int sys4 = system("paste ./timestamp* | awk '{print $2 -$1}'>>run_info.txt");
        int sys5 = system("rm ./timestamp*");
    }
    return;
}




