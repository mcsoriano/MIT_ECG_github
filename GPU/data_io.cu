//Include header (helper functions and data prototypes)
#include "data_io.h"


//Alloc_Set Data
void alloc_set_data(double **F,	        mdim &Fdim, 
	            double **M,	        mdim &Mdim, 
                    double **MB,	mdim &MBdim, 
                    double **R,	        mdim &Rdim, 
                    double **Tag,	mdim &Tagdim, 
                    double **TrainY,	mdim &TrainYdim, 
                    double **TestY,	mdim &TestYdim, 
                    double **TrainR,	mdim &TrainRdim, 
                    double **TestR, 	mdim &TestRdim){

    std::string location;	//Datafile path and name


    //Data treated internally
    //======================================================
    double *A = NULL, *RR = NULL, *Y = NULL, *Patient = NULL, *W = NULL;
    mdim Adim, RRdim, Ydim, Patient_dim, Wdim;
    //lead A
    Adim.R = nbeats;
    Adim.C = nsamples;
    Adim.S = Adim.R * Adim.C;
    Adim.mem_size = sizeof(double) * Adim.S;
    A = (double *)malloc(Adim.mem_size);

    //RR
    RRdim.R = nbeats;
    RRdim.C = nRR;
    RRdim.S = RRdim.R * RRdim.C;
    RRdim.mem_size = sizeof(double) * RRdim.S;
    RR = (double *)malloc(RRdim.mem_size);

    //Target Y
    Ydim.R = nbeats;
    Ydim.C = 1;
    Ydim.S = Ydim.R * Ydim.C;    
    Ydim.mem_size = sizeof(double) * Ydim.S;
    Y = (double *)malloc(Ydim.mem_size);

    //Patient
    Patient_dim.R = nbeats;
    Patient_dim.C = 1;
    Patient_dim.S = Patient_dim.R * Patient_dim.C;
    Patient_dim.mem_size = sizeof(double) * Patient_dim.S;
    Patient = (double *)malloc(Patient_dim.mem_size);

    //Window
    int wind0 = wind0_val;
    int wind1 = wind1_val;
    int wsize = wsize_val;
    if (wind1-wind0==wsize) printf ("Window size as set (%d) \n", wsize);
    Wdim.R = nbeats;
    Wdim.C = wsize_val;
    Wdim.S = Wdim.R * Wdim.C;
    Wdim.mem_size = sizeof(double) * Wdim.S;
    W = (double *)malloc(Wdim.mem_size);

    //lead A
    location=leadApath;
    printf ("Call read routine for leadA(%lu, %lu -> %lu) \n", Adim.R, Adim.C, Adim.S);
    read_array(A, Adim.S, location, Adefval);

    //RR
    location=RRpath;
    printf ("Call read routine for RR(%lu, %lu -> %lu) \n", RRdim.R, RRdim.C, RRdim.S);
    read_array(RR, RRdim.S, location, RRdefval);

    //Target Y
    location=Ypath;
    printf ("Call read routine for SVE_class Target(%lu, %lu -> %lu) \n", Ydim.R, Ydim.C, Ydim.S);
    read_array(Y, Ydim.S, location);

    //Patient
    location=Patientpath;
    printf ("Call read routine for Patient(%lu, %lu -> %lu) \n", Patient_dim.R, Patient_dim.C, Tagdim.S);
    read_array(Patient, RRdim.R, location);

    //W Give values to windowed data
    printf ("Setting W window submatrix values\n");
    for (unsigned long int i=0; i<Wdim.S; i++){
      unsigned long int j=(i%Wdim.C)+(i/Wdim.C)*Adim.C+wind0;
      W[i]=A[j];
    }

    printf ("W mapminmax normalisation\n");
    // Normalisation
    for (unsigned long int i=0; i<Wdim.R; i++){
    mapminmax(&W[i*Wdim.C], Wdim.C);
    }

    //Patient RRavg (stored in 3rd column)
    printf ("RR avg\n");
    rr_patient_average(RR, RRdim, Patient, Patient_dim);
    printf ("Done\n");
 


    //Data passed to main function
    //======================================================
    //F features
    Fdim.R = nbeats;
    Fdim.C = Wdim.C + RRdim.C;
    Fdim.S = Fdim.R * Fdim.C;
    Fdim.mem_size = sizeof(double) * Fdim.S;
    *F = (double *)malloc(Fdim.mem_size);

    //Mask
    Mdim.R = nfeatures;
    Mdim.C = nneurons;
    Mdim.S = Mdim.R * Mdim.C;    
    Mdim.mem_size = sizeof(double) * Mdim.S;
    *M = (double *)malloc(Mdim.mem_size);

    //Mask bias
    MBdim.R = nneurons;
    MBdim.C = 1;
    MBdim.S = MBdim.R * MBdim.C;    
    MBdim.mem_size = sizeof(double) * MBdim.S;
    *MB = (double *)malloc(MBdim.mem_size);

    //Reservoir
    Rdim.R = nbeats;
    Rdim.C = nneurons;
    Rdim.S = Rdim.R * Rdim.C;
    Rdim.mem_size = sizeof(double) * Rdim.S;
    *R = (double *)malloc(Rdim.mem_size);

    //DS1DS2 Tag
    Tagdim.R = nbeats;
    Tagdim.C = 1;
    Tagdim.S = Tagdim.R * Tagdim.C;    
    Tagdim.mem_size = sizeof(double) * Tagdim.S;
    *Tag = (double *)malloc(Tagdim.mem_size);

    //Train target
    TrainYdim.R = nTrain;
    TrainYdim.C = 1;
    TrainYdim.S = TrainYdim.R * TrainYdim.C;    
    TrainYdim.mem_size = sizeof(double) * TrainYdim.S;
    *TrainY = (double *)malloc(TrainYdim.mem_size);

    //Test target
    TestYdim.R = nTest;
    TestYdim.C = 1;
    TestYdim.S = TestYdim.R * TestYdim.C;    
    TestYdim.mem_size = sizeof(double) * TestYdim.S;
    *TestY = (double *)malloc(TestYdim.mem_size);

    //Add extra column to reservoir(neuron)
    //Train Reservoir rows
    TrainRdim.R = TrainYdim.R;
    TrainRdim.C = Rdim.C+1;
    TrainRdim.S = TrainRdim.R * TrainRdim.C;
    TrainRdim.mem_size = sizeof(double) * TrainRdim.S;
    *TrainR = (double *)malloc(TrainRdim.mem_size);

    //Test Reservoir rows
    TestRdim.R = TestYdim.R;
    TestRdim.C = Rdim.C+1;
    TestRdim.S = TestRdim.R * TestRdim.C;
    TestRdim.mem_size = sizeof(double) * TestRdim.S;
    *TestR = (double *)malloc(TestRdim.mem_size);

/*  //Read Single mask  / Iteratively
    //Mask
    location=Mpath;
    printf ("Call read routine for mask(%lu, %lu -> %lu) \n", Mdim.R, Mdim.C, Mdim.S);
    read_array(M, Mdim.S, location);

    //Mask bias
    location=MBpath;
    printf ("Call read routine for maskbias(%lu, %lu -> %lu) \n", MBdim.R, MBdim.C, MBdim.S);
    read_array(MB, MBdim.S, location);
*/

    //DS1DS2 Tag
    location=Tagpath;
    printf ("Call read routine for DS1DS2(%lu, %lu -> %lu) \n", Tagdim.R, Tagdim.C, Tagdim.S);
    read_array(*Tag, Tagdim.S, location);

    //F Feature selection
    printf ("Feature selection\n");
    unsigned long int k = 0;
    for (unsigned long int i=0; i<Fdim.R; i++){
    for (unsigned long int j=0; j<Fdim.C; j++){
    k = i * Fdim.C + j;
       if (j < Wdim.C) (*F)[k]=W[i*Wdim.C + j];
       else  ((*F)[k])=(log(RR[i*RRdim.C + j-Wdim.C]));
    }
    }

    //Obtain TrainY and TestY
    {
	  long unsigned int i_0=0, i_1=0;
	  for (unsigned long int i=0; i<Tagdim.S; i++){
	      if ((*Tag)[i]==0) {
                 (*TrainY)[i_0]=Y[i];
                 i_0++;
              }
              else {
                 (*TestY)[i_1]=Y[i];
                 i_1++;
              }
	  }
          if ((i_0!=TrainYdim.R)||(i_1!=TestYdim.R))   printf("ERROR: Dim mismatch (i_0,i_1) = (%lu,%lu)\n",i_0, i_1);
    }

    //Free variables which are not passed
    free(A);
    free(RR);
    free(Y);
    free(Patient);
    free(W);
return;
}

// Mapminmax normalization
void mapminmax(double V[], unsigned long int dim) {
  double max = V[0];
  double min = V[0];
  double x0 = 0;
  double s = 0;

  // locate max and min
  for (int i = 1; i < dim; i++) {
    if (V[i] > max)
      max = V[i];
    if (V[i] < min)
      min = V[i];
  }

  // rescale to [-1, 1]
  if (max != min) {
    x0 = 1 + 2. * min / (max - min);
    s = 2. / (max - min);
    for (int i = 0; i < dim; i++) {
      V[i] = V[i] * s - x0;
    }
  } else {
    fprintf(stderr, "Nonexistent range\n");
  }

  return;
}

// rr_avg: Average up to nRRavg previous RR values (1st col) and store (3rd col)
void rr_patient_average(double RR[], mdim RRdim, double Patient[], mdim Patient_dim){
    unsigned long int ipat=0;
    int count=0;
    int nRRvals=nRRavg;
    double Ravg=0;
    double Rcum=0;
    double RRavg[nRRavg];
    int k, kavg;

    //For every patient
    for (unsigned long int i=0; i<Patient_dim.R; i++){
	if (Patient[i]!=ipat){
           for (int j=0; j<nRRvals; j++) RRavg[j]=0;
 	   ipat=Patient[i];
           count=0;
           Rcum=0;
	}
        //Wrap around a window of nRRvals
        k = count%nRRvals;
        if (count >= nRRvals) {
	    Rcum-=RRavg[k];
	    kavg=nRRvals;
        }
        else kavg = k+1;
        
        RRavg[k]=RR[i*RRdim.C];
        Rcum+=RR[i*RRdim.C];

        count++;
        Ravg=Rcum/(kavg);
        RR[(i+1)*RRdim.C-1]=Ravg;

    }
    return;
}




// Read Array: Read array from file
void read_array(double V[], int arr_length, std::string location) {
  printf("Reading data \n");
  double aux = 0;
  std::ifstream infile;
  infile.open(location.c_str());
  if (infile.is_open()) {
    printf(" txt file opened \n");

    unsigned long int i = 0;
    while ((infile >> aux) && (i < arr_length)) {
      V[i] = aux;
      i++;
    }
    printf("OK ( %lu ) \n", i);
  } else {
    printf("ERROR");
  }
  infile.close();
}

// Read Array Control: Read array from file applying nan and inf check
void read_array(double V[], int arr_length, std::string location, double defval) {
  printf("Reading data (controlled) \n");
  double aux = 0;
  std::ifstream infile;
  infile.open(location.c_str());
  if (infile.is_open()) {
    printf(" txt file opened \n");

    unsigned long int i = 0;
    while ((infile >> aux) && (i < arr_length)) {
      if ((isinf(aux)||isnan(aux))) V[i]=defval;
      else V[i] = aux;
      i++;
    }
    printf("OK ( %lu ) \n", i);
  } else {
    printf("ERROR");
  }
  infile.close();
}


// Write: Write array into file
void Write(std::string fname, double *vect, int rows, int cols) {
  std::cout << "Writing data,";
  std::ofstream outfile;
  outfile.open(fname.c_str());
  if (outfile.is_open()) {
    std::cout << fname << " txt file opened " << std::endl;

    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        long int k = j + i * cols;
        double val = vect[k];
        outfile << val << " ";
      }
      outfile << std::endl;
    }
    std::cout << "OK -> saved file(" << rows << "," << cols << ")" << std::endl;
  }
  outfile.close();
  return;
}

// Write 1D: Write array into file as 1-dimensional
void Write1D(std::string fname, double *vect, int elem) {
  std::cout << "Writing data,";
  std::ofstream outfile;
  outfile.open(fname.c_str());
  if (outfile.is_open()) {
    std::cout << fname << " txt file opened " << std::endl;

    for (int i = 0; i < elem; i++) {
      outfile << vect[i] << std::endl;
    }
    std::cout << "OK -> saved file(" << elem << ",1)" << std::endl;
  }
  else{             
	fprintf(stderr, "Failed to save file\n");
        exit(EXIT_FAILURE);
  }
  outfile.close();
  return;
}

// Write 1D Threshold: Write array into file 1-d applying a threshold
void Write1D(std::string fname, double *vect, int elem, double threshold) {
  std::cout << "Writing data,";
  std::ofstream outfile;
  outfile.open(fname.c_str());
  if (outfile.is_open()) {
    std::cout << fname << " txt file opened " << std::endl;

    for (int i = 0; i < elem; i++) {
      outfile << ((vect[i] > threshold) ? 1 : 0) << std::endl;
    }
    std::cout << "OK -> saved file(" << elem << ",1)" << std::endl;
  }
  outfile.close();
  return;
}
