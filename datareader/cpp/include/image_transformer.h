/*
 * function:
 *  implements a producer-consumer pattern:
 *  the raw image data comes from python space, 
 *  after a pipeline of image transformation using multi-threading,
 *  the data results will be put back to python space
 */

#ifndef DATAREADER_CPP_INCLUDE_IMAGE_TRANSFORMER_H
#define DATAREADER_CPP_INCLUDE_IMAGE_TRANSFORMER_H

#include <atomic>
#include <cstdint>
#include <vector>
#include "transformer.h"
#include "concurrent.h"

namespace vis {

class ImageTransformer : public Transformer {
public:
    ImageTransformer ();
    virtual ~ImageTransformer ();

    /*
     * init this transformer
     */
    virtual int init(const kv_conf_t &conf);

    /*
     * add transformation operations to this transformer
     */
    virtual int add_op(const std::string &op_name, const kv_conf_t &conf);

    /*
     * launch this transformer to work
     */
    virtual int start();

    /*
     * stop this transformer,
     * then no more data will be feeded in, but left data can be fetched out
     */
    virtual int stop();

    /*
     * test whether this transformer has already stopped
     */
    virtual bool is_stopped();

    /*
     * put a new image processing task to this transformer
     */
    virtual int put(const transformer_input_data_t &input);

    /*
     * get a transformed image from this transformer
     */
    virtual int get(transformer_output_data_t &output);

    /*
     * real image processing function
     */
    void process(const transformer_input_data_t &input);

    /*
     * return the number of unconsumed data in this transformer,
     * call this function in 'stopped' state please in case of race-condition
     */
    int _unconsumed_num() {
        return (_in_num - _out_num);
    }

private:
    int _id;//unique id for this transformer
    std::atomic<std::uint64_t> _in_num; //number of putted data
    std::atomic<std::uint64_t> _out_num; //number of getted data
    std::vector<kv_conf_t> _ops;// operations to be applied in this transformer
    std::string _state;
    ThreadPool _workers;
    BlockingQueue<transformer_output_data_t *> _output_queue;
    int _swapaxis;
};

};// end of namespace 'vis'

#endif  //__IMAGE_TRANSFORMER_H_

/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
