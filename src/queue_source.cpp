/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include <iostream>
#include <vector>
#include "opencv2/core.hpp"
#include "opencv2/videostab.hpp"
#include "opencv2/videostab/frame_source.hpp"

#include "opencv2/videostab/ring_buffer.hpp"

#include "opencv2/opencv_modules.hpp"
#ifdef HAVE_OPENCV_VIDEOIO
#  include "opencv2/videoio.hpp"
#endif

#define IFrameSource cv::videostab::IFrameSource

namespace cv
{
    namespace videostab
    {
        class QueueSource;
    } // namespace videostab
} // namespace cv


class CV_EXPORTS cv::videostab::QueueSource : public IFrameSource
{
public:
    QueueSource() { reset();}

    cv::Mat nextFrame()
    {
        cv::Mat frame;
        frame = queue.front();
        if (queue.size()>1)      queue.pop();
        count_++;
        return volatileFrame_ ? frame : frame.clone();
    }

    void addFrame(cv::Mat frame)
    {
        queue.push(frame);
        size_ = frame.size();
    }

    int width() {return size_.width;}
    int height() {return size_.height;}
    int count() {return count_;}
    double fps() {return 15;}
    void reset()
    {
        count_ = 0;
        while(!queue.empty()) queue.pop();
    }

private:
    std::string path_;
    bool volatileFrame_;
    std::queue<cv::Mat> queue;
    int count_;
    cv::Size size_;
};