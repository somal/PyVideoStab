#define PY_ARRAY_UNIQUE_SYMBOL pbcvt_ARRAY_API

#include <iostream>
#include <boost/python.hpp>
#include "opencv2/videostab.hpp"
#include "pyvideostab.cpp"
#include <pyboostcvconverter/pyboostcvconverter.hpp>

namespace pyvideostab {

    using namespace boost::python;
    pyvideostab::PyVideoStab* vs = new pyvideostab::PyVideoStab();

//This example uses Mat directly, but we won't need to worry about the conversion
/**
 * Example function. Basic inner matrix product using implicit matrix conversion.
 * @param leftMat left-hand matrix operand
 * @param rightMat right-hand matrix operand
 * @return an NdArray representing the dot-product of the left and right operands
 */
    cv::Mat stabilize(cv::Mat frame) {
        vs->addFrame(frame);
        return vs->nextFrame();
    }

#if (PY_VERSION_HEX >= 0x03000000)

    static void *init_ar() {
#else
        static void init_ar(){
#endif
        Py_Initialize();

        import_array();
        return NUMPY_IMPORT_ARRAY_RETVAL;
    }

    BOOST_PYTHON_MODULE (pyvideostab) {
        //using namespace XM;
        init_ar();

        //initialize converters
        to_python_converter<cv::Mat,
                pbcvt::matToNDArrayBoostConverter>();
        pbcvt::matFromNDArrayBoostConverter();

        //expose module-level functions
        def("stabilize", stabilize);
    }

} //end namespace pyvideostab
