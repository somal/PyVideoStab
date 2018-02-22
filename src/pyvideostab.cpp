#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include "opencv2/core.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/video.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videostab.hpp"
#include "opencv2/videostab/stabilizer.hpp"
#include "opencv2/opencv_modules.hpp"
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include "numpy/ndarrayobject.h"
#include "opencv2/core/core.hpp"
#include "queue_source.cpp"

#define arg(name) cmd.get<string>(name)
#define argb(name) cmd.get<bool>(name)
#define argi(name) cmd.get<int>(name)
#define argf(name) cmd.get<float>(name)
#define argd(name) cmd.get<double>(name)

namespace py = boost::python;
namespace np = boost::python::numpy;
using namespace std;
using namespace cv;
using namespace cv::videostab;

namespace pyvideostab {
    using namespace py;

    class CV_EXPORTS PyVideoStab
    {
    private:
        Ptr<IFrameSource> stabilizedFrames;
        cv::videostab::StabilizerBase* stabilizer;
        int nframes;
        Ptr<QueueSource> source;
        cv::Mat m;
        string outputPath;
        int outputFps;
    public:
        PyVideoStab()
        {
            source = makePtr<QueueSource>();
            const char *argv[]={};
            stabilizer = get_stabilizer(0, argv);
            // cast stabilizer to simple frame source interface to read stabilized frames
            stabilizedFrames.reset(dynamic_cast<IFrameSource*>(stabilizer));
            nframes=0;
        }

        cv::Mat nextFrame()
        {
            return stabilizedFrames->nextFrame();
        }

        void addFrame(cv::Mat img){
            source->addFrame(img);
        }

        ~PyVideoStab(){
           stabilizedFrames.release();
           delete stabilizedFrames;
           delete stabilizer;
        }

        static MotionModel getMotionModel(const string &str)
        {
            if (str == "transl")
                return MM_TRANSLATION;
            if (str == "transl_and_scale")
                return MM_TRANSLATION_AND_SCALE;
            if (str == "rigid")
                return MM_RIGID;
            if (str == "similarity")
                return MM_SIMILARITY;
            if (str == "affine")
                return MM_AFFINE;
            if (str == "homography")
                return MM_HOMOGRAPHY;
            throw runtime_error("unknown motion model: " + str);
        }

       // motion estimator builders are for concise creation of motion estimators
        class IMotionEstimatorBuilder
        {
        public:
            virtual ~IMotionEstimatorBuilder() {}
            virtual Ptr<ImageMotionEstimatorBase> build() = 0;
        protected:
            IMotionEstimatorBuilder(CommandLineParser &command) : cmd(command) {}
            CommandLineParser cmd;
        };


        class MotionEstimatorRansacL2Builder : public IMotionEstimatorBuilder
        {
        public:
            MotionEstimatorRansacL2Builder(CommandLineParser &command, bool use_gpu, const string &_prefix = "")
                : IMotionEstimatorBuilder(command), gpu(use_gpu), prefix(_prefix) {}

            virtual Ptr<ImageMotionEstimatorBase> build()
            {
                Ptr<MotionEstimatorRansacL2> est = makePtr<MotionEstimatorRansacL2>(getMotionModel(arg(prefix + "model")));

                RansacParams ransac = est->ransacParams();
                if (arg(prefix + "subset") != "auto")
                    ransac.size = argi(prefix + "subset");
                if (arg(prefix + "thresh") != "auto")
                    ransac.thresh = argf(prefix + "thresh");
                ransac.eps = argf(prefix + "outlier-ratio");
                est->setRansacParams(ransac);

                est->setMinInlierRatio(argf(prefix + "min-inlier-ratio"));

                Ptr<IOutlierRejector> outlierRejector = makePtr<NullOutlierRejector>();
                if (arg(prefix + "local-outlier-rejection") == "yes")
                {
                    Ptr<TranslationBasedLocalOutlierRejector> tblor = makePtr<TranslationBasedLocalOutlierRejector>();
                    RansacParams ransacParams = tblor->ransacParams();
                    if (arg(prefix + "thresh") != "auto")
                        ransacParams.thresh = argf(prefix + "thresh");
                    tblor->setRansacParams(ransacParams);
                    outlierRejector = tblor;
                }

        #if defined(HAVE_OPENCV_CUDAIMGPROC) && defined(HAVE_OPENCV_CUDAOPTFLOW)
                if (gpu)
                {
                    Ptr<KeypointBasedMotionEstimatorGpu> kbest = makePtr<KeypointBasedMotionEstimatorGpu>(est);
                    kbest->setOutlierRejector(outlierRejector);
                    return kbest;
                }
        #else
                CV_Assert(gpu == false && "CUDA modules are not available");
        #endif

                Ptr<KeypointBasedMotionEstimator> kbest = makePtr<KeypointBasedMotionEstimator>(est);
                kbest->setDetector(GFTTDetector::create(argi(prefix + "nkps")));
                kbest->setOutlierRejector(outlierRejector);
                return kbest;
            }
        private:
            bool gpu;
            string prefix;
        };


        class MotionEstimatorL1Builder : public IMotionEstimatorBuilder
        {
        public:
            MotionEstimatorL1Builder(CommandLineParser &command, bool use_gpu, const string &_prefix = "")
                : IMotionEstimatorBuilder(command), gpu(use_gpu), prefix(_prefix) {}

            virtual Ptr<ImageMotionEstimatorBase> build()
            {
                Ptr<MotionEstimatorL1> est = makePtr<MotionEstimatorL1>(getMotionModel(arg(prefix + "model")));

                Ptr<IOutlierRejector> outlierRejector = makePtr<NullOutlierRejector>();
                if (arg(prefix + "local-outlier-rejection") == "yes")
                {
                    Ptr<TranslationBasedLocalOutlierRejector> tblor = makePtr<TranslationBasedLocalOutlierRejector>();
                    RansacParams ransacParams = tblor->ransacParams();
                    if (arg(prefix + "thresh") != "auto")
                        ransacParams.thresh = argf(prefix + "thresh");
                    tblor->setRansacParams(ransacParams);
                    outlierRejector = tblor;
                }

        #if defined(HAVE_OPENCV_CUDAIMGPROC) && defined(HAVE_OPENCV_CUDAOPTFLOW)
                if (gpu)
                {
                    Ptr<KeypointBasedMotionEstimatorGpu> kbest = makePtr<KeypointBasedMotionEstimatorGpu>(est);
                    kbest->setOutlierRejector(outlierRejector);
                    return kbest;
                }
        #else
                CV_Assert(gpu == false && "CUDA modules are not available");
        #endif

                Ptr<KeypointBasedMotionEstimator> kbest = makePtr<KeypointBasedMotionEstimator>(est);
                kbest->setDetector(GFTTDetector::create(argi(prefix + "nkps")));
                kbest->setOutlierRejector(outlierRejector);
                return kbest;
            }
        private:
            bool gpu;
            string prefix;
        };

        cv::videostab::StabilizerBase* get_stabilizer(int argc, const char **argv) {
        const char *keys =
                    "{ @1                       |           | }"
                    "{ m  model                 | affine    | }"
                    "{ lp lin-prog-motion-est   | no        | }"
                    "{  subset                  | auto      | }"
                    "{  thresh                  | auto | }"
                    "{  outlier-ratio           | 0.5 | }"
                    "{  min-inlier-ratio        | 0.1 | }"
                    "{  nkps                    | 1000 | }"
                    "{  extra-kps               | 0 | }"
                    "{  local-outlier-rejection | no | }"
                    "{ sm  save-motions         | no | }"
                    "{ lm  load-motions         | no | }"
                    "{ r  radius                | 15 | }"
                    "{  stdev                   | auto | }"
                    "{ lps  lin-prog-stab       | no | }"
                    "{  lps-trim-ratio          | auto | }"
                    "{  lps-w1                  | 1 | }"
                    "{  lps-w2                  | 10 | }"
                    "{  lps-w3                  | 100 | }"
                    "{  lps-w4                  | 100 | }"
                    "{  deblur                  | no | }"
                    "{  deblur-sens             | 0.1 | }"
                    "{ et  est-trim             | no | }"
                    "{ t  trim-ratio            | 0.1 | }"
                    "{ ic  incl-constr          | no | }"
                    "{ bm  border-mode          | replicate | }"
                    "{  mosaic                  | no | }"
                    "{ ms  mosaic-stdev         | 10.0 | }"
                    "{ mi  motion-inpaint       | no | }"
                    "{  mi-dist-thresh          | 5.0 | }"
                    "{ ci color-inpaint         | no | }"
                    "{  ci-radius               | 2 | }"
                    "{ ws  wobble-suppress      | no | }"
                    "{  ws-period               | 30 | }"
                    "{  ws-model                | homography | }"
                    "{  ws-subset               | auto | }"
                    "{  ws-thresh               | auto | }"
                    "{  ws-outlier-ratio        | 0.5 | }"
                    "{  ws-min-inlier-ratio     | 0.1 | }"
                    "{  ws-nkps                 | 1000 | }"
                    "{  ws-extra-kps            | 0 | }"
                    "{  ws-local-outlier-rejection | no | }"
                    "{  ws-lp                   | no | }"
                    "{ sm2 save-motions2        | no | }"
                    "{ lm2 load-motions2        | no | }"
                    "{ gpu                      | no | }"
                    "{ o  output                | stabilized.avi | }"
                    "{ fps                      | auto | }"
                    "{ q quiet                  |  | }"
                    "{ h help                   |  | }";
            CommandLineParser cmd(argc, argv, keys);


            if (arg("gpu") == "yes")
            {
                cout << "initializing GPU..."; cout.flush();
                Mat hostTmp = Mat::zeros(1, 1, CV_32F);
                cuda::GpuMat deviceTmp;
                deviceTmp.upload(hostTmp);
                cout << endl;
            }

            cv::videostab::StabilizerBase *stabilizer = 0;


            string inputPath = arg(0);
            // get source video parameters

    //        Its segmenation fault
            cout << "frame count (rough): " << source->count() << endl;
            //
            if (arg("fps") == "auto")
                outputFps = source->fps();
            else
                outputFps = argd("fps");

            // prepare motion estimation builders
            Ptr<IMotionEstimatorBuilder> motionEstBuilder;
            if (arg("lin-prog-motion-est") == "yes")
                motionEstBuilder.reset(new MotionEstimatorL1Builder(cmd, arg("gpu") == "yes"));
            else
                motionEstBuilder.reset(new MotionEstimatorRansacL2Builder(cmd, arg("gpu") == "yes"));

            Ptr<IMotionEstimatorBuilder> wsMotionEstBuilder;
            if (arg("ws-lp") == "yes")
                wsMotionEstBuilder.reset(new MotionEstimatorL1Builder(cmd, arg("gpu") == "yes", "ws-"));
            else
                wsMotionEstBuilder.reset(new MotionEstimatorRansacL2Builder(cmd, arg("gpu") == "yes", "ws-"));

            // determine whether we must use one pass or two pass stabilizer
            bool isTwoPass =
                    arg("est-trim") == "yes" || arg("wobble-suppress") == "yes" || arg("lin-prog-stab") == "yes";

            if (isTwoPass)
            {
                // we must use two pass stabilizer

                TwoPassStabilizer *twoPassStabilizer = new TwoPassStabilizer();
                stabilizer = twoPassStabilizer;
                twoPassStabilizer->setEstimateTrimRatio(arg("est-trim") == "yes");

                // determine stabilization technique

                if (arg("lin-prog-stab") == "yes")
                {
                    Ptr<LpMotionStabilizer> stab = makePtr<LpMotionStabilizer>();
                    stab->setFrameSize(Size(source->width(), source->height()));
                    stab->setTrimRatio(arg("lps-trim-ratio") == "auto" ? argf("trim-ratio") : argf("lps-trim-ratio"));
                    stab->setWeight1(argf("lps-w1"));
                    stab->setWeight2(argf("lps-w2"));
                    stab->setWeight3(argf("lps-w3"));
                    stab->setWeight4(argf("lps-w4"));
                    twoPassStabilizer->setMotionStabilizer(stab);
                }
                else if (arg("stdev") == "auto") {
                    cv::videostab::GaussianMotionFilter* gmf=new cv::videostab::GaussianMotionFilter(argi("radius"));
                    twoPassStabilizer->setMotionStabilizer(gmf);
                }
                // init wobble suppressor if necessary
                if (arg("wobble-suppress") == "yes")
                {
                    Ptr<MoreAccurateMotionWobbleSuppressorBase> ws = makePtr<MoreAccurateMotionWobbleSuppressor>();
                    if (arg("gpu") == "yes")
                        #ifdef HAVE_OPENCV_CUDAWARPING
                                            ws = makePtr<MoreAccurateMotionWobbleSuppressorGpu>();
                        #else
                                            throw runtime_error("OpenCV is built without CUDA support");
                        #endif

                    ws->setMotionEstimator(wsMotionEstBuilder->build());
                    ws->setPeriod(argi("ws-period"));
                    twoPassStabilizer->setWobbleSuppressor(ws);

                    MotionModel model = ws->motionEstimator()->motionModel();
                    if (arg("load-motions2") != "no")
                    {
                        ws->setMotionEstimator(makePtr<FromFileMotionReader>(arg("load-motions2")));
                        ws->motionEstimator()->setMotionModel(model);
                    }
                    if (arg("save-motions2") != "no")
                    {
                        ws->setMotionEstimator(makePtr<ToFileMotionWriter>(arg("save-motions2"), ws->motionEstimator()));
                        ws->motionEstimator()->setMotionModel(model);
                    }
                }
            }
            else
            {
                // we must use one pass stabilizer

                OnePassStabilizer *onePassStabilizer = new OnePassStabilizer();
                stabilizer = onePassStabilizer;
                if (arg("stdev") == "auto") {
                    cv::videostab::GaussianMotionFilter* gmf=new cv::videostab::GaussianMotionFilter(argi("radius"));
                    onePassStabilizer->setMotionFilter(gmf);
                }
            }

            stabilizer->setFrameSource(source);
            stabilizer->setMotionEstimator(motionEstBuilder->build());

            MotionModel model = stabilizer->motionEstimator()->motionModel();
            if (arg("load-motions") != "no")
            {
                stabilizer->setMotionEstimator(makePtr<FromFileMotionReader>(arg("load-motions")));
                stabilizer->motionEstimator()->setMotionModel(model);
            }
            if (arg("save-motions") != "no")
            {
                stabilizer->setMotionEstimator(makePtr<ToFileMotionWriter>(arg("save-motions"), stabilizer->motionEstimator()));
                stabilizer->motionEstimator()->setMotionModel(model);
            }

            stabilizer->setRadius(argi("radius"));

            // init deblurer
            if (arg("deblur") == "yes")
            {
                Ptr<WeightingDeblurer> deblurer = makePtr<WeightingDeblurer>();
                deblurer->setRadius(argi("radius"));
                deblurer->setSensitivity(argf("deblur-sens"));
                stabilizer->setDeblurer(deblurer);
            }

            // set up trimming paramters
            stabilizer->setTrimRatio(argf("trim-ratio"));
            stabilizer->setCorrectionForInclusion(arg("incl-constr") == "yes");

            if (arg("border-mode") == "reflect")
                stabilizer->setBorderMode(BORDER_REFLECT);
            else if (arg("border-mode") == "replicate")
                stabilizer->setBorderMode(BORDER_REPLICATE);
            else if (arg("border-mode") == "const")
                stabilizer->setBorderMode(BORDER_CONSTANT);
            else
                throw runtime_error("unknown border extrapolation mode: "
                                     + cmd.get<string>("border-mode"));

            // init inpainter
            InpaintingPipeline *inpainters = new InpaintingPipeline();
            Ptr<InpainterBase> inpainters_(inpainters);
            if (arg("mosaic") == "yes")
            {
                Ptr<ConsistentMosaicInpainter> inp = makePtr<ConsistentMosaicInpainter>();
                inp->setStdevThresh(argf("mosaic-stdev"));
                inpainters->pushBack(inp);
            }
            if (arg("motion-inpaint") == "yes")
            {
                Ptr<MotionInpainter> inp = makePtr<MotionInpainter>();
                inp->setDistThreshold(argf("mi-dist-thresh"));
                inpainters->pushBack(inp);
            }
            if (!inpainters->empty())
            {
                inpainters->setRadius(argi("radius"));
                stabilizer->setInpainter(inpainters_);
            }

            if (arg("output") != "no")
                outputPath = arg("output");
            return stabilizer;
    }

    };

} //end namespace pyvideostab