import cv2 as cv
import sys

def get_selective_search():
    return cv.ximgproc.segmentation.createSelectiveSearchSegmentation()

#selective-search算法总共有四种策略，不同组合可以达到不同的效果
def config(gs, img, strategy='q'):
    gs.setBaseImage(img)
    if (strategy == 's'):
        gs.switchToSingleStrategy()                 #使用单一策略
    elif (strategy == 'f'):
        gs.switchToSelectiveSearchFast()
    elif (strategy == 'q'):
        gs.switchToSelectiveSearchQuality()
    else:
        print(__doc__)
        sys.exit(1)

#选择行搜索算法产生的边界框为[xmin,ymin,width,height], 将产生的边界框转化为[xmin,ymin,xmax,ymax]
def get_rects(gs):
    rects = gs.process()
    rects[:,2] += rects[:,0]
    rects[:,3] += rects[:,1]

    return rects

if __name__ == '__main__':
    gs = get_selective_search()
    img = cv.imread('../imgs/1.jpg',cv.IMREAD_COLOR)
    config(gs,img,'q')
    rects = get_rects(gs)
    print(rects)