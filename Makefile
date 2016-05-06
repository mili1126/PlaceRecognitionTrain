OBJS =
EXES = main
CC = g++ -I/usr/local/include/opencv -I/usr/local/include/opencv2 -L/usr/local/lib/ -g
OPENCVLIBS = -lopencv_core -lopencv_imgproc -lopencv_photo -lopencv_imgcodecs -lopencv_videoio -lopencv_highgui -lopencv_xphoto -lopencv_ml -lopencv_objdetect -lopencv_xobjdetect -lopencv_flann -lopencv_features2d -lopencv_calib3d -lopencv_ximgproc -lopencv_video -lopencv_shape -lopencv_xfeatures2d -lopencv_face -lopencv_text -lopencv_datasets -lopencv_tracking -lopencv_surface_matching -lopencv_rgbd -lopencv_structured_light -lopencv_stereo -lopencv_saliency -lopencv_reg -lopencv_plot -lopencv_optflow -lopencv_line_descriptor -lopencv_fuzzy -lopencv_dpm -lopencv_dnn -lopencv_ccalib -lopencv_bioinspired -lopencv_bgsegm -lopencv_aruco -lopencv_videostab -lopencv_superres -lopencv_stitching

clean:
	rm $(OBJS) $(EXES)
	rm -r main.dSYM

sift: main.cpp
	$(CC) -o main main.cpp $(OPENCVLIBS) -DMODE=0

surf: main.cpp
	$(CC) -o main main.cpp $(OPENCVLIBS) -DMODE=1

orb: main.cpp
	$(CC) -o main main.cpp $(OPENCVLIBS) -DMODE=2
