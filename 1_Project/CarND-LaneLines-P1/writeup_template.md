# **Finding Lane Lines on the Road**

#### Author:      Liang Xu
Email:       liangxuav at gmail.com
Date:        July 19, 2017

---
**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Find Lane Line pipeline and draw_lines()

#### My pipeline consisted of 6 steps.
1. The images have been converted to grayscale image.
2. Smooth the image use Gaussian kernel. Because the canny edge detection method is sensitive to the noise.
3. Find edges using the canny edge detector
4. Define a polygon to mask. Because, we only care about his region
5. Find straight lines using the hough_lines() function from OpenCV.
6. Draw lines on the graph.

#### draw_lines()
There are several tricks in the draw_lines() function. After the hough_lines detector, several broken lines have been get. There are rarely a prefect line to cover the whole lane line. Here are the list of techniques I have applied to draw the lane lines.
1. Only two lines need to be draw.
2. The slop of the Lane lines can be estimated [0.45,0.75] [-0.9 -0.6]. Filter out the line with different slop.
3. Calculate the lane line function according to the averaged lines from last step.
4. Extent the lane line to the edge of the polygon mask, get the 4 cross points.
5. Reject outliers data for the cross points. Some of the data are far away from the group, remove it.
```
a = [1,2,1,2,1,2,5,4,5,6,7,4,5,6,4,5,6]
b = reject_outliers(np.asarray(a),0.8)
print(b)
[5 4 5 4 5 4 5]
```
6. Apply a data structure called, averageQueue(). This data structure is used to compare with finite length of history data to either accept the new point or reject the new point.
```
class averageQueue():
    def __init__(self,m = 1.5):
        self.queue = deque()
        self.m     = m
    def getQueue(self):
        return self.queue
    def getValue(self,data):
        if len(self.queue) > 10:
            if(abs(data - np.mean(self.queue)) <= self.m * np.std(self.queue)):
                self.queue.append(data)
                self.queue.popleft()
                return data
            else:
                self.queue.popleft()
                self.queue.appendleft(data)
                return self.queue[-1]
        else:
            self.queue.append(data)
            return data
```
### 2. Identify potential shortcomings with your current pipeline
There are several shortcomings of my project. Sensitive to noise, the result rely on the edge detection. If the canny edge detection and hough_lines detection fails because of the noise, we can not get any result.
The algorithm can only apply to the straight line lane.
I know there are some place can blow up my program, for example, some place will be dived zero. For now, I have not seen, so be it right now.
### 3. Suggest possible improvements to your pipeline
1. Add some error handling code.
2. Make the pipeline not sensitive to noise.
3. make the calculation vectorization, means faster.
