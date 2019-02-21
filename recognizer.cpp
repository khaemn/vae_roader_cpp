#include "recognizer.h"
#include <QQuickItemGrabResult>
#include <opencv2/opencv.hpp>
#include <map>
#include <string>
#include <chrono>
using namespace cv;

const std::map<int,std::string> Recognizer::OBJ_NAMES =
{
    {0,"person            "       },
    {1,"bicycle           "       },
    {2,"car               "       },
    {3,"motorbike         "       },
    {4,"aeroplane         "       },
    {5,"bus               "       },
    {6,"train             "       },
    {7,"truck             "       },
    {8,"boat              "       },
    {9,"traffic light     "     },
    {10,"fire hydrant      "     },
    {11,"stop sign         "     },
    {12,"parking meter     "     },
    {13,"bench             "     },
    {14,"bird              "     },
    {15,"cat               "     },
    {16,"dog               "     },
    {17,"horse             "     },
    {18,"sheep             "     },
    {19,"cow               "     },
    {20,"elephant          "     },
    {21,"bear              "     },
    {22,"zebra             "     },
    {23,"giraffe           "     },
    {24,"backpack          "     },
    {25,"umbrella          "    },
    {26,"handbag           "    },
    {27,"tie               "    },
    {28,"suitcase          "    },
    {29,"frisbee           "    },
    {30,"skis              "    },
    {31,"snowboard         "    },
    {32,"sports ball       "    },
    {33,"kite              "    },
    {34,"baseball bat      "    },
    {35,"baseball glove    "    },
    {36,"skateboard        "    },
    {37,"surfboard         "    },
    {38,"tennis racket     "    },
    {39,"bottle            "    },
    {40,"wine glass        "    },
    {41,"cup               "    },
    {42,"fork              "    },
    {43,"knife             "    },
    {44,"spoon             "    },
    {45,"bowl              "    },
    {46,"banana            "   },
    {47,"apple             "   },
    {48,"sandwich          "   },
    {49,"orange            "   },
    {50,"broccoli          "   },
    {51,"carrot            "   },
    {52,"hot dog           "   },
    {53,"pizza             "   },
    {54,"donut             "   },
    {55,"cake              "   },
    {56,"chair             "   },
    {57,"sofa              "    },
    {58,"pottedplant       "    },
    {59,"bed               "    },
    {60,"diningtable       "    },
    {61,"toilet            "    },
    {62,"tvmonitor         "    },
    {63,"laptop            "    },
    {64,"mouse             "    },
    {65,"remote            "    },
    {66,"keyboard          "    },
    {67,"cell phone        "    },
    {68,"microwave         "    },
    {69,"oven              "    },
    {70,"toaster           "    },
    {71,"sink              "    },
    {72,"refrigerator      "    },
    {73,"book              "    },
    {74,"clock             "    },
    {75,"vase              "    },
    {76,"scissors          "    },
    {77,"teddy bear        "    },
    {78,"hair drier        "    },
    {79,"toothbrush        "    }
};

Recognizer::Recognizer()
{
    init_default();
}

void Recognizer::init_default()
{
    m_detector.reset(new Detector("../darknet/cfg/yolov3-tiny.cfg",
                                  "../darknet/yolov3-tiny.weights"));

}

void Recognizer::recognize()
{
    m_result = m_camera->grabToImage();
    connect(m_result.data(), &QQuickItemGrabResult::ready,
            this, &Recognizer::scan);
}

void Recognizer::scan()
{

    // 1) move to thread
    // 2) messurment of performance - done
    // 3) bounding boxes


    QImage  img = m_result->image().convertToFormat(QImage::Format_ARGB32);
    cv::Mat image(img.height(), img.width(),
                  CV_8UC4,
                  const_cast<uchar*>(img.bits()),
                  static_cast<size_t>(img.bytesPerLine())
                  );
    auto start = std::chrono::system_clock::now();
    // do some work
    std::vector<bbox_t> res = m_detector->detect(image);
    // record end time
    auto end = std::chrono::system_clock::now();

    for(auto elem : res)
    {
        qDebug() << OBJ_NAMES.at(elem.obj_id).c_str() ;
        if(elem.obj_id == 41)
        {
            emit recognized(elem.x, elem.y , elem.w , elem.h);
        }
    }
    std::chrono::duration<double> diff = end-start;
    //qDebug() << "recognizing duration: " << diff.count() ;


}

void Recognizer::setCamera(QQuickItem* ptr)
{
    if (ptr)
    {
        m_camera = ptr;
    }
}

std::vector<bbox_t> Recognizer::recognize(const Mat &input, float thres)
{
    return m_detector->detect(input, 0.7);
}
