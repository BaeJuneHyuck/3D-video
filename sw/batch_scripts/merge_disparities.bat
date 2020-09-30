ffmpeg -framerate 15 -f image2 -i disparity_%%d.png -c:v ffv1 -level 3 -g 1 disparities.avi
