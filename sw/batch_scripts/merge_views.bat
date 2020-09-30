ffmpeg -framerate 15 -f image2 -i view_%%d.jpg -c:v mjpeg -q:v 1 views.avi
