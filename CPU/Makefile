# Flags
GCCFLAGS := -O2
INCLUDES  := -I/home/malfaras/lib/eigen
LIBRARIES := 
#LIBRARIES := -lXXX


# Target rules

all: xSVEloop

#function.o: function.cpp function.h config.h
#	gcc $(INCLUDES) $(GCCFLAGS) -o $@ -c $<

xSVEloop.o: neqinv_eigenloop_SVE_classifier.cpp mapminmax.cpp matrix_to_file.cpp eigenMIT.cpp neqinv_eigen_SVE_classifier.h config.h 
	g++ $(INCLUDES) $(GCCFLAGS) -o $@ -c $<

xSVEloop: xSVEloop.o
	g++ $(GCCFLAGS) -o $@ $+ $(LIBRARIES)

run: all
	./xSVEloop 0.2 0.1

clean:
	rm -f xSVEloop *.o 

cleanobj:
	rm -f *.o 

rebuild: clean all

fresh: clean run

