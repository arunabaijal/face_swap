import face_swap_tpl
import swap_face_part1
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Face Swap')

    parser.add_argument('-m', '--mode', default='trian', type=str,
                        help='path to input image')
    parser.add_argument('-i', '--ref_path', default='Data/Rambo.jpg', type=str,
                        help='path to reference image(texture ref)')
    parser.add_argument('-v', '--video_path', default='Data/data1.mp4', type=str,
                        help='path to video')

    if parser.parse_args().m == 'trian':
        if parser.parse_args().i == 'swap':
            swap_face_part1.main(True, parser.parse_args().i, parser.parse_args().v)
        else:
            swap_face_part1.main(False, parser.parse_args().i, parser.parse_args().v)
    elif parser.parse_args().m == 'tps':
        if parser.parse_args().i == 'swap':
            face_swap_tpl.start(True, parser.parse_args().i, parser.parse_args().v)
        else:
            face_swap_tpl.start(False, parser.parse_args().i, parser.parse_args().v)
