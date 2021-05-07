from test import test2
import argparse

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--name',type=str,default='abc')
    parser.add_argument('--age',type=int,default=123)

    args=parser.parse_args()
    t1=test2(args)
    t1.func()