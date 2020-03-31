import face_swap_tpl
import swap_face_part1
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Face Swap')

    parser.add_argument('--mode', default='trian', type=str,
                        help='path to input image')
    parser.add_argument('--img_path', default='Data/Rambo.jpg', type=str,
                        help='path to reference image(texture ref)')
    parser.add_argument('--video_path', default='Data/data1.mp4', type=str,
                        help='path to video')

    if parser.parse_args().mode == 'trian':
        if parser.parse_args().img_path == 'swap':
            swap_face_part1.main(True, parser.parse_args().img_path, parser.parse_args().video_path)
        else:
            swap_face_part1.main(False, parser.parse_args().img_path, parser.parse_args().video_path)
    elif parser.parse_args().mode == 'tps':
        if parser.parse_args().img_path == 'swap':
            face_swap_tpl.start(True, parser.parse_args().img_path, parser.parse_args().video_path)
        else:
            face_swap_tpl.start(False, parser.parse_args().img_path, parser.parse_args().video_path)
