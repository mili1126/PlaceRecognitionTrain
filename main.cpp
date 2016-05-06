//
//  main.cpp
//  Localization
//
//  Created by Ming Li on 4/27/16.
//  Copyright © 2016 Ming Li. All rights reserved.
//

//opencv
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/calib3d.hpp"
//C
#include <stdio.h>
//C++
#include <iostream>
#include <fstream>
#include <sstream>


using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

//constants
const char * VIDEO_FOLDER = "videos/";
const char * OUTPUT_FOLDER = "outputs/";
const double SCALE = 0.3;

//global variables
int keyboard; //input from keyboard
char * videoId;
string featureMode;

Mat frame; //current frame
vector<KeyPoint> keypoints;
Mat descriptor;

Mat previousFrame;
vector<KeyPoint> previousKeypoints;
Mat previousDescriptor;

Mat matchedDescriptor;

vector< Mat > vImg;
cv::Mat result;

//function declaration
void help();
void processVideo(char* videoFilename);

void help()
{
    cout
    << "--------------------------------------------------------------------------" << endl
    << "This program shows how to use the feature points detectors, descriptors"
    << " and matching framework found inside OpenCV."                               << endl
    << endl
    << "Usage:"                                                                     << endl
    << "./main <video filename>"                                                    << endl
    << "for example: ./main 1.MOV"                                                  << endl
    << "--------------------------------------------------------------------------" << endl
    << endl;
}

int main(int argc, char * argv[]) {
    //print help information
    help();

    //check for the input parameter correctness
    if(argc != 2) {
        cerr <<"Incorret input list" << endl;
        cerr <<"exiting..." << endl;
        return EXIT_FAILURE;
    }

    #if MODE == 0
    featureMode = "sift";
    #endif
    #if MODE == 1
    featureMode = "surf";
    #endif
    #if MODE == 2
    featureMode = "orb";
    #endif

    cout << "Feature Mode = " << featureMode << endl;
    //create GUI windows
    namedWindow("PlaceRecognization");

    //input data coming from a video
    char * videoName = (char *) malloc(1 + strlen(VIDEO_FOLDER) + strlen(argv[1]));
    strcpy(videoName, VIDEO_FOLDER);
    strcat(videoName, argv[1]);
    videoId = strtok(argv[1],".");
    cout << "Processing Video " << videoId << endl;
    processVideo(videoName);

    //destroy GUI windows
    destroyAllWindows();
    return EXIT_SUCCESS;
}

void processVideo(char* videoFilename) {
    //create the capture object
    VideoCapture capture(videoFilename);
    if(!capture.isOpened()){
        //error in opening the video input
        cerr << "Unable to open video file: " << videoFilename << endl;
        exit(EXIT_FAILURE);
    }

    //read input data. ESC or 'q' for quitting
    while( (char)keyboard != 'q' && (char)keyboard != 27 ){
        //read the current frame
        if(!capture.read(frame)) {
            cerr << "Unable to read next frame." << endl;
            cerr << "Exiting..." << endl;
            exit(EXIT_FAILURE);
        }

        //get the frame number and write it on the current frame
        stringstream ss;
        rectangle(frame, cv::Point(10, 2), cv::Point(100,20),
        cv::Scalar(255,255,255), -1);
        ss << capture.get(CAP_PROP_POS_FRAMES);
        string frameNumberString = ss.str();
        // putText(frame, frameNumberString.c_str(), cv::Point(15, 15),
        // FONT_HERSHEY_SIMPLEX, 0.5 , cv::Scalar(0,0,0));


        #if MODE == 0
        //-- Step 1: Detect the keypoints using SIFT Detector
        int 	nfeatures = 0;
        int 	nOctaveLayers = 3;
        double 	contrastThreshold = 0.04;
        double 	edgeThreshold = 10;
        double 	sigma = 1.6;
        Ptr<SIFT> detector = SIFT::create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
        #endif

        #if MODE == 1
        //-- Step 1: Detect the keypoints using SURF Detector
        int minHessian = 400;
        Ptr<SURF> detector = SURF::create( minHessian );
        #endif

        #if MODE == 2
        //-- Step 1: Detect the keypoints using ORB Detector
        int 	nfeatures = 500;
        float 	scaleFactor = 1.2f;
        int 	nlevels = 8;
        int 	edgeThreshold = 31;
        int 	firstLevel = 0;
        int 	WTA_K = 2;
        int 	scoreType = ORB::HARRIS_SCORE;
        int 	patchSize = 31;
        int 	fastThreshold = 20;
        Ptr<ORB> detector = ORB::create (nfeatures, scaleFactor,nlevels,edgeThreshold,firstLevel,WTA_K, scoreType,patchSize,fastThreshold);
        #endif



        // //-- Draw keypoints
        // Mat img_keypoints;
        // drawKeypoints( frame, keypoints, img_keypoints, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
        // //-- Show detected (drawn) keypoints
        // imshow("Keypoints ", img_keypoints);


        if (stoi(frameNumberString) > 1) {
            #if MODE == 0
            //-- Step 2: Matching descriptor vectors using BF matcher
            BFMatcher matcher( NORM_L2, false );
            #endif
            #if MODE == 1
            //-- Step 2: Matching descriptor vectors using FLANN matcher
            FlannBasedMatcher matcher;
            #endif
            #if MODE == 2
            //-- Step 2: Matching descriptor vectors using BF matcher
            BFMatcher matcher( NORM_HAMMING, false );
            #endif

            // NOTE: featureDetector is a pointer hence the '->'.
            detector->detectAndCompute(frame, Mat(), keypoints, descriptor);
            detector->detectAndCompute(previousFrame, Mat(), previousKeypoints, previousDescriptor);


            std::vector< DMatch > matches;

            matcher.match( descriptor, previousDescriptor, matches );
            double max_dist = 0; double min_dist = 100;
            //-- Quick calculation of max and min distances between keypoints
            for( int i = 0; i < descriptor.rows; i++ )
            { double dist = matches[i].distance;
                // cout << dist << endl;
                if( dist < min_dist ) min_dist = dist;
                if( dist > max_dist ) max_dist = dist;
            }
            // printf("-- Max dist : %f \n", max_dist );
            // printf("-- Min dist : %f \n", min_dist );
            //-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
            //-- or a small arbitary value in the event that min_dist is very
            //-- small)
            //-- PS.- radiusMatch can also be used here.
            double arbitaryValue;
            #if MODE == 0
            arbitaryValue = 0.05;
            #endif
            #if MODE == 1
            arbitaryValue = 0.02;
            #endif
            #if MODE == 2
            arbitaryValue = 3.0;
            #endif

            std::vector< DMatch > good_matches;
            for( int i = 0; i < descriptor.rows; i++ )
            { if( matches[i].distance <= max(2*min_dist, arbitaryValue) )
                { good_matches.push_back( matches[i]); }
            }
            //-- Draw only "good" matches
            Mat result;
            drawMatches( frame, keypoints, previousFrame, previousKeypoints,
                good_matches, result, Scalar::all(-1), Scalar::all(-1),
                vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
                // //-- Resize and show detected matches
                // Size targetSize(result.size().width*SCALE, result.size().height*SCALE);
                // resize(result, result, targetSize);
                // imshow( "Good Matches", result );
                // for( int i = 0; i < (int)good_matches.size(); i++ )
                // { printf( "-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n", i, good_matches[i].queryIdx, good_matches[i].trainIdx ); }


                //-- Store matched keypoints
                vector<KeyPoint> matchedKeypoints;
                vector<KeyPoint> previousMatchedKeypoints;
                for( int i = 0; i < (int)good_matches.size(); i++ ) {
                    for (int j = 0; j < keypoints.size(); j++) {
                        if (good_matches[i].queryIdx == j) {
                            matchedKeypoints.push_back(keypoints[j]);
                        }
                    }
                    for (int j = 0; j < previousKeypoints.size(); j++) {
                        if (good_matches[i].trainIdx == j) {
                            previousMatchedKeypoints.push_back(previousKeypoints[j]);
                        }
                    }
                }
                // cout << frameNumberString << ": MatchedKeypointsNum = " << matchedKeypoints.size() << endl;
                // detector->detectAndCompute( frame, Mat(), matchedKeypoints, matchedDescriptor);
                // cout << "descriptor size = " << matchedDescriptor.rows << " " << matchedDescriptor.cols << endl;
                // fout << matchedDescriptor << endl;


                // //TODO Output only one descriptor
                // string outputFile = string(OUTPUT_FOLDER) + featureMode + "/" + string(videoId) + ".yml";
                // cout << "outputFileName = " << outputFile << endl;
                // cv::FileStorage fs(outputFile.c_str(), cv::FileStorage::WRITE);
                // fs << "Mat" + string(videoId) << matchedDescriptor;
                // fs.release();
                // keyboard = waitKey( 1000 );
                // capture.release();
                // return;

                std::vector< Point2f > obj;
                std::vector< Point2f > scene;
                cv::KeyPoint::convert	(	matchedKeypoints, obj, std::vector< int >()                 )	;
                cv::KeyPoint::convert	(	previousMatchedKeypoints, scene, std::vector< int >()                 )	;

                // Find the Homography Matrix
                Mat H = findHomography( obj, scene, CV_RANSAC );
                // Use the Homography Matrix to warp the images

                warpPerspective(frame,result,H,cv::Size(frame.cols+previousFrame.cols,frame.rows));
                cv::Mat half(result,cv::Rect(0,0,previousFrame.cols,previousFrame.rows));
                previousFrame.copyTo(half);
                cv::Mat show;
                Size targetSize(result.size().width*SCALE, result.size().height*SCALE);
                // Size targetSize(600, 800);
                resize(result, show, targetSize);
                imshow( "Result", result );

            }
            previousFrame = frame;

            //get the input from the keyboard
            keyboard = waitKey( 0 );
        }
        //delete capture object
        capture.release();
    }
