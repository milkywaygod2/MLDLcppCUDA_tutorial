#pragma once
/*
CŬ����
O��ü
�ƹ��͵� �Ⱥ��̸� �Լ� //F�Լ�
m_�ɹ�����
i,f...�ڷ���
V����
D��ť
S��Ʈ��
M��Ʈ����(Mat)
_����

12.3v errorC4996 next-deprecated
pruneInfo_t
cusparseSolvePolicy_t
cusparseColorInfo_t
csru2csrInfo_t
csrilu02Info_t
bsrsv2Info_t
bsrilu02Info_t
bsric02Info_t
����������Ʈ�Ӽ�-C/C++��޿��� 4996�����ѻ���
*/

//-- CUDA api
#include <cuda_runtime.h>				//CUDA ��Ÿ��api
#include <cublas_v2.h>					//Nvidia �������lib
#include <cusparse_v2.h>				//suSparse api
#include <device_launch_parameters.h>	//Ŀ�� ���Ľ���� ����


//-- CUDA custom kernel
#include "cudaKernel.h"
#include "cudaIntellisense.hpp"

//--- C
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <conio.h>  //_getch()
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

//--- C++
#include <iostream>
#include <string>
#include <cmath>
#include <array>

#include <vector>		//STL-container
#include <list>			//STL-container
#include <deque>		//STL-container
#include <set>			//STL-container
#include <map>			//STL-container
#include <unordered_set>//STL-container
#include <unordered_map>//STL-container
#include <algorithm>	//STL-algoritm : sort,reverse,find,copy...
#include <iterator>		//STL-iterator : forward,bidirectional,random access
#include <filesystem>	//C++17���� ���� : .exists()�������ü
#include <fstream>		//: fileExists() .is_open()�������� .good()����°�������
#include <random>		//��������, ���ڿ���Ʈ��ó��
#include <sstream>		//��Ʈ����Ʈ��

using namespace std;

//--- My
#include "main.h"
#include "perceptron.h"
#include "cuMat.h"
#include "cuMatSparse.h"

//--- define KEY_
#define KEY_NUL 0
#define KEY_SOH 1 // _____Ctrl+A, ���� ���� ����(Start of Heading)
#define KEY_STX 2 // _____Ctrl+B, ���� �ؽ�Ʈ(Start of Text)
#define KEY_ETX 3 // _____Ctrl+C, ���� �ؽ�Ʈ(End of Text)
#define KEY_EOT 4 // _____Ctrl+D, ���� ����(End of Transmission)
#define KEY_ENQ 5 // _____Ctrl+E, ���� ����(Enquiry)
#define KEY_ACK 6 // _____Ctrl+F, ���� ����(Acknowledge)
#define KEY_BEL 7 // _____Ctrl+G, ��� ����(Bell)
#define KEY_BS 8 // Backspace, �齺���̽�
#define KEY_HT 9 // Tab, ���� ��(Horizontal Tab)
#define KEY_LF 10 // Enter (in a terminal), �� �ٲ�(Line Feed)
#define KEY_VT 11 // _____Ctrl+K, ���� ��(Vertical Tab)
#define KEY_FF 12 // _____Ctrl+L, �� �ǵ�(Form Feed)
#define KEY_CR 13 // Enter, ĳ���� ����(Carriage Return)
#define KEY_SO 14 // _____Ctrl+N, ����Ʈ �ƿ�(Shift Out)
#define KEY_SI 15 // _____Ctrl+O, ����Ʈ ��(Shift In)
#define KEY_DLE 16 // _____Ctrl+P, ������ ��ũ �̽�������(Data Link Escape)
#define KEY_DC1 17 // _____Ctrl+Q, ��ġ ���� 1(Device Control 1)
#define KEY_DC2 18 // _____Ctrl+R, ��ġ ���� 2(Device Control 2)
#define KEY_DC3 19 // _____Ctrl+S, ��ġ ���� 3(Device Control 3)
#define KEY_DC4 20 // _____Ctrl+T, ��ġ ���� 4(Device Control 4)
#define KEY_NAK 21 // _____Ctrl+U, ���� ����(Negative Acknowledge)
#define KEY_SYN 22 // _____Ctrl+V, ����ȭ(Synchronous Idle)
#define KEY_ETB 23 // _____Ctrl+W, ���� ���(End of Transmission Block)
#define KEY_CAN 24 // _____Ctrl+X, ���(Cancel)
#define KEY_EM 25 // _____Ctrl+Y, ���� �̵��(End of Medium)
#define KEY_SUB 26 // _____Ctrl+Z, ��ü(Substitute)
#define KEY_ESC 27 // ESC, �̽�������(Escape)
#define KEY_FS 28 // _____Ctrl+\, ���� ������(File Separator)
#define KEY_GS 29 // _____Ctrl+], �׷� ������(Group Separator)
#define KEY_RS 30 // _____Ctrl+^, ���ڵ� ������(Record Separator)
#define KEY_US 31 // _____Ctrl+_, ���� ������(Unit Separator)
#define KEY_SPACE 32 // Space, ����
#define KEY_EXCLAMATION 33 // !, ����ǥ
#define KEY_QUOTATION_MARK 34 // ", �ֵ���ǥ
#define KEY_NUMBER_SIGN 35 // #, ��
#define KEY_DOLLAR_SIGN 36 // $, �޷� ��ȣ
#define KEY_PERCENT_SIGN 37 // %, �ۼ�Ʈ ��ȣ
#define KEY_AMPERSAND 38 // &, ���ۻ���
#define KEY_APOSTROPHE 39 // ', ��������ǥ
#define KEY_LEFT_PARENTHESIS 40 // (, ���� ��ȣ
#define KEY_RIGHT_PARENTHESIS 41 // ), ������ ��ȣ
#define KEY_ASTERISK 42 // *, ��ǥ
#define KEY_PLUS_SIGN 43 // +, ���ϱ� ��ȣ
#define KEY_COMMA 44 // ,, ��ǥ
#define KEY_MINUS_SIGN 45 // -, ���� ��ȣ
#define KEY_PERIOD 46 // ., ��ħǥ
#define KEY_SLASH 47 // /, ������
#define KEY_0 48 // 0, ���� 0
#define KEY_1 49 // 1, ���� 1
#define KEY_2 50 // 2, ���� 2
#define KEY_3 51 // 3, ���� 3
#define KEY_4 52 // 4, ���� 4
#define KEY_5 53 // 5, ���� 5
#define KEY_6 54 // 6, ���� 6
#define KEY_7 55 // 7, ���� 7
#define KEY_8 56 // 8, ���� 8
#define KEY_9 57 // 9, ���� 9
#define KEY_COLON 58 // :, �ݷ�
#define KEY_SEMICOLON 59 // ;, �����ݷ�
#define KEY_LESS_THAN_SIGN 60 // <, ���� �ε�ȣ
#define KEY_EQUAL_SIGN 61 // =, ��ȣ
#define KEY_GREATER_THAN_SIGN 62 // >, ū �ε�ȣ
#define KEY_QUESTION_MARK 63 // ?, ����ǥ
#define KEY_AT_SIGN 64 // @, �����
#define KEY_A 65 // A, �빮�� A
#define KEY_B 66 // B, �빮�� B
#define KEY_C 67 // C, �빮�� C
#define KEY_D 68 // D, �빮�� D
#define KEY_E 69 // E, �빮�� E
#define KEY_F 70 // F, �빮�� F
#define KEY_G 71 // G, �빮�� G
#define KEY_H 72 // H, �빮�� H
#define KEY_I 73 // I, �빮�� I
#define KEY_J 74 // J, �빮�� J
#define KEY_K 75 // K, �빮�� K
#define KEY_L 76 // L, �빮�� L
#define KEY_M 77 // M, �빮�� M
#define KEY_N 78 // N, �빮�� N
#define KEY_O 79 // O, �빮�� O
#define KEY_P 80 // P, �빮�� P
#define KEY_Q 81 // Q, �빮�� Q
#define KEY_R 82 // R, �빮�� R
#define KEY_S 83 // S, �빮�� S
#define KEY_T 84 // T, �빮�� T
#define KEY_U 85 // U, �빮�� U
#define KEY_V 86 // V, �빮�� V
#define KEY_W 87 // W, �빮�� W
#define KEY_X 88 // X, �빮�� X
#define KEY_Y 89 // Y, �빮�� Y
#define KEY_Z 90 // Z, �빮�� Z
#define KEY_LEFT_SQUARE_BRACKET 91 // [, ���� ���ȣ
#define KEY_BACKSLASH 92 // \, ��������
#define KEY_RIGHT_SQUARE_BRACKET 93 // ], ������ ���ȣ
#define KEY_CIRCUMFLEX_ACCENT 94 // ^, ĳ��
#define KEY_UNDERSCORE 95 // _, �����
#define KEY_GRAVE_ACCENT 96 // `, �׷��̺� �׼�Ʈ
#define KEY_a 97 // a, �ҹ��� a
#define KEY_b 98 // b, �ҹ��� b
#define KEY_c 99 // c, �ҹ��� c
#define KEY_d 100 // d, �ҹ��� d
#define KEY_e 101 // e, �ҹ��� e
#define KEY_f 102 // f, �ҹ��� f
#define KEY_g 103 // g, �ҹ��� g
#define KEY_h 104 // h, �ҹ��� h
#define KEY_i 105 // i, �ҹ��� i
#define KEY_j 106 // j, �ҹ��� j
#define KEY_k 107 // k, �ҹ��� k
#define KEY_l 108 // l, �ҹ��� l
#define KEY_m 109 // m, �ҹ��� m
#define KEY_n 110 // n, �ҹ��� n
#define KEY_o 111 // o, �ҹ��� o
#define KEY_p 112 // p, �ҹ��� p
#define KEY_q 113 // q, �ҹ��� q
#define KEY_r 114 // r, �ҹ��� r
#define KEY_s 115 // s, �ҹ��� s
#define KEY_t 116 // t, �ҹ��� t
#define KEY_u 117 // u, �ҹ��� u
#define KEY_v 118 // v, �ҹ��� v
#define KEY_w 119 // w, �ҹ��� w
#define KEY_x 120 // x, �ҹ��� x
#define KEY_y 121 // y, �ҹ��� y
#define KEY_z 122 // z, �ҹ��� z
#define KEY_LEFT_CURLY_BRACKET 123 // {, ���� �߰�ȣ
#define KEY_VERTICAL_BAR 124 // |, ������
#define KEY_RIGHT_CURLY_BRACKET 125 // }, ������ �߰�ȣ
#define KEY_TILDE 126 // ~, ����ǥ
#define KEY_DEL 127 // DEL, ����