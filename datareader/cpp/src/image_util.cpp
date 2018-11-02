/* 
 * function:
 *      image operators implemented by opencv
 */

#include <stdio.h>
#include "image_util.h"
#ifdef USETURBOJPEG 
#include "turbojpeg.h"
#endif
namespace vis {

int read_image(const std::string &fname, std::vector<char> &buf) {
    buf.clear();

    FILE *fp = fopen(fname.c_str(), "rb");
    if (!fp) {
        return -1;
    }

    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);

    buf.resize(size);

    fseek(fp, 0, SEEK_SET);
    int readed = fread(&buf[0], 1, size, fp);
    fclose(fp);
    fp = NULL;

    if (readed != size) {
        buf.clear();
        return -2;
    }

    return 0;
}

int save_image(const cv::Mat &img, const std::string &fname) {
    if (img.empty()) {
        return -1;
    }

    std::vector<uchar> buf;
    cv::imencode(".jpg", img, buf);

    FILE *fp = fopen(fname.c_str(), "wb+");

    size_t written = fwrite(&buf[0], 1, buf.size(), fp);
    fclose(fp);

    if (written != buf.size()) {
        return -2;
    }
    return 0;
}
    
#ifdef USETURBOJPEG 
static cv::Mat jpgdecode(const char * buffer, int bufferlen, int iscolor) {
    
    tjhandle handle = tjInitDecompress();

    int width = 0;
    int height = 0;
    int subsample = 0;
    tjDecompressHeader2(handle, (uint8_t*)buffer, bufferlen,
        &width, &height, &subsample);
    cv::Mat img;
    if (iscolor){
        img = cv::Mat(height, width, CV_8UC3);
    }
    else{
        img = cv::Mat(height, width, CV_8UC1);
    }
    int decompressflags = 0;
    //int decompressflags = TJFLAG_FASTUPSAMPLE | TJFLAG_FASTDCT;
    tjDecompress2(handle, (const unsigned char*)buffer, bufferlen,
        img.data, img.cols, 0, img.rows,
        img.elemSize() == 1 ? TJPF_GRAY : TJPF_BGR, decompressflags);
    tjDestroy(handle);
    return img;
}

static bool checkformat(const char * buffer, int bufferlen,
    const char *format, int comparelen){
    if (bufferlen < comparelen){
        return false;
    }
    return strncmp(buffer, format, comparelen) == 0;
}
    
static bool is_jpeg_format(const char * buffer, int bufferlen) {
    char format[2] = {0xFF, 0xD8};
    return checkformat(buffer, bufferlen, format, 2);
}
#endif    

IMPROC_ERR_CODE_TYPE decode_image(const std::vector<char> &buf,
    cv::Mat &result, int mode) {
    IMPROC_ERR_CODE_TYPE ret = IMPROC_OK;
    cv::Mat dec;
#ifdef USETURBOJPEG
    bool isjpeg = is_jpeg_format(buf.data(), buf.size());
    if (isjpeg){  
        dec = jpgdecode(buf.data(), buf.size(), mode);
    }
    else{
        dec = cv::imdecode(buf, mode);
    }
#else
    dec = cv::imdecode(buf, mode);
#endif //USETURBOJPEG 
    
    if (dec.channels() == 3) {
        cv::cvtColor(dec, result, cv::COLOR_BGR2RGB);
    } else {
        result = dec;
    }

    if (result.empty()) {
        ret = IMPROC_INVALID_PARAM;
    }
    return ret;
}

IMPROC_ERR_CODE_TYPE resize_image(const cv::Mat &img, const cv::Size &size,
       cv::Mat &result, int interpolation, double fx, double fy) {
    IMPROC_ERR_CODE_TYPE ret = IMPROC_OK;
    cv::resize(img, result, size, fx, fy, interpolation);
    if (result.empty()) {
        ret = IMPROC_INVALID_PARAM;
    }
    return ret;
}

IMPROC_ERR_CODE_TYPE crop_image(const cv::Mat &img,
        const cv::Rect &rect, cv::Mat &result) {
    IMPROC_ERR_CODE_TYPE ret = IMPROC_OK;
    img(rect).copyTo(result);
    if (result.empty()) {
        ret = IMPROC_INVALID_PARAM;
    }
    return ret;
}

IMPROC_ERR_CODE_TYPE rotate_image(const cv::Mat &img,
        float angle, cv::Mat &result, int resample) {
    IMPROC_ERR_CODE_TYPE ret = IMPROC_OK;

    cv::Point2f ptCp(img.cols * 0.5, img.rows * 0.5);
    cv::Mat trans_mat = cv::getRotationMatrix2D(ptCp, angle, 1.0);
    cv::warpAffine(img, result, trans_mat, img.size(), resample); //Nearest is too rough,
    if (result.empty()) {
        ret = IMPROC_INVALID_PARAM;
    }

    return ret;
}

IMPROC_ERR_CODE_TYPE flip_image(
        const cv::Mat &img, int flip_code, cv::Mat &result) {
    IMPROC_ERR_CODE_TYPE ret = IMPROC_OK;

    cv::flip(img, result, flip_code);
    if (result.empty()) {
        ret = IMPROC_INVALID_PARAM;
    }
    return ret;
}

};//end of namespace 'vis'

/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
