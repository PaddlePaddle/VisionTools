add_test(NAME test_dejpeg
    COMMAND ${PROJ_ROOT}/.set_python_path.sh -d ${CMAKE_CURRENT_BINARY_DIR}/..
              python ${PROJ_ROOT}/transformer/tests/test_dejpeg.py
              WORKING_DIRECTORY ${PROJ_ROOT})
