import cv2

class Saving:

def __init__(self, name, age):
  self.name = name
  self.age = age


        #save_txt
    if save_txt and len(outputs) != 0:
        for j, output in enumerate(outputs):
            bbox_left = output[0]
            bbox_top = output[1]
            bbox_right = output[2]
            bbox_bottom = output[3]
            identity = output[-1]
            try:
                conf='%.2f'%confs[identity][0]
            except IndexError:
                conf=0

            # with open(txt_path, 'w') as f:
            #     f.write(('%g ' * 6 + '\n') % (frame_idx, identity, bbox_left, bbox_top, bbox_right, bbox_bottom))  # label format
            ret_txt+='{} {} {} {} {} {}\n'.format(identity, conf, bbox_left, bbox_top, bbox_right, bbox_bottom)


    #make map form txt
    with open(str(Path(out))+'\\'+'%07d'%frame_idx+'.txt', 'w') as f:
        f.write(ret_txt)

    print('%sDone. (%.3fs)' % (ret_str, t2 - t1))

    # Stream results---------------------------------------------------------------------
    if view_img:
        cv2.imshow(path, im0s)
        if cv2.waitKey(1) == ord('q'):  # q to quit
            raise StopIteration

    if dataset.mode == 'image':
        print('saving img!')
    else:
        # Save results (image with detections)
        if vid_path != save_path:  # new video
            vid_path = save_path
            if isinstance(vid_writer, cv2.VideoWriter):
                vid_writer.release()  # release previous video writer

            fps = vid_cap.get(cv2.CAP_PROP_FPS)
            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            vid_writer = cv2.VideoWriter(
                save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
        vid_writer.write(im0s)