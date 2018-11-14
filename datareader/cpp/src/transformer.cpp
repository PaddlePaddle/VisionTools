#include "transformer.h"
#include "image_transformer.h"
#include "logger.h"

namespace vistool {

Transformer::Transformer() {
}

Transformer::~Transformer() {
}

Transformer * Transformer::create(const std::string &type) {
    LOG(INFO) << "Transformer::create(" << type << ")";

    if (type == "ImageTransformer") {
        ImageTransformer *t = new ImageTransformer();
        return t;
    } else {
        LOG(WARNING) << "failed to create transformer with type[" 
            << type << "]";
        return NULL;
    }
}

void Transformer::destroy(Transformer *t) {
    LOG(INFO) << "Transformer::destroy";
    if (t) {
        delete t;
    }
}
};// end of namespace 'vistool'

/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
