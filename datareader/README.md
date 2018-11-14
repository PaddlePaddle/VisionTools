
### DataReader
---
A python module used to facilitate the pipeline of image data preprocessing in Machine Learning.
Usually this pipeline includes data operations like `load/parse/decode/resize/crop/xxx`.

---

### Features
  * easy to compose flexible pipelines of preprocessing
  * fast image processing implemented in c++
  * stream data processing for large datasets which maybe not possible to store in local disk

---
### How to install

  * build wheel
    `mkdir output && cd output && cmake .. && make`

  * install wheel
    `python install output/dist/datareader-0.0.1-cp27-cp27mu-linux_x86_64.whl`
