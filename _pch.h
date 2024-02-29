#pragma once
/*
C클래스
O객체
아무것도 안붙이면 함수 //F함수
m_맴버변수
i,f...자료형
V벡터
D디큐
S스트링
M매트릭스(Mat)
_인자

12.3v errorC4996 next-deprecated
pruneInfo_t
cusparseSolvePolicy_t
cusparseColorInfo_t
csru2csrInfo_t
csrilu02Info_t
bsrsv2Info_t
bsrilu02Info_t
bsric02Info_t
현재프로젝트속성-C/C++고급에서 4996제외한상태
*/

//-- CUDA api
#include <cuda_runtime.h>				//CUDA 런타임api
#include <cublas_v2.h>					//Nvidia 선형대수lib
#include <cusparse_v2.h>				//suSparse api
#include <device_launch_parameters.h>	//커널 병렬실행모델 지원


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
#include <filesystem>	//C++17부터 지원 : .exists()존재그자체
#include <fstream>		//: fileExists() .is_open()열리는지 .good()입출력가능한지
#include <random>		//난수생성, 문자열스트림처리
#include <sstream>		//스트링스트림

using namespace std;

//--- My
#include "main.h"
#include "perceptron.h"
#include "cuMat.h"
#include "cuMatSparse.h"

//--- define KEY_
#define KEY_NUL 0
#define KEY_SOH 1 // _____Ctrl+A, 시작 제어 문자(Start of Heading)
#define KEY_STX 2 // _____Ctrl+B, 시작 텍스트(Start of Text)
#define KEY_ETX 3 // _____Ctrl+C, 종료 텍스트(End of Text)
#define KEY_EOT 4 // _____Ctrl+D, 종료 전송(End of Transmission)
#define KEY_ENQ 5 // _____Ctrl+E, 문의 문자(Enquiry)
#define KEY_ACK 6 // _____Ctrl+F, 인정 문자(Acknowledge)
#define KEY_BEL 7 // _____Ctrl+G, 경고 문자(Bell)
#define KEY_BS 8 // Backspace, 백스페이스
#define KEY_HT 9 // Tab, 수평 탭(Horizontal Tab)
#define KEY_LF 10 // Enter (in a terminal), 줄 바꿈(Line Feed)
#define KEY_VT 11 // _____Ctrl+K, 수직 탭(Vertical Tab)
#define KEY_FF 12 // _____Ctrl+L, 폼 피드(Form Feed)
#define KEY_CR 13 // Enter, 캐리지 리턴(Carriage Return)
#define KEY_SO 14 // _____Ctrl+N, 시프트 아웃(Shift Out)
#define KEY_SI 15 // _____Ctrl+O, 시프트 인(Shift In)
#define KEY_DLE 16 // _____Ctrl+P, 데이터 링크 이스케이프(Data Link Escape)
#define KEY_DC1 17 // _____Ctrl+Q, 장치 제어 1(Device Control 1)
#define KEY_DC2 18 // _____Ctrl+R, 장치 제어 2(Device Control 2)
#define KEY_DC3 19 // _____Ctrl+S, 장치 제어 3(Device Control 3)
#define KEY_DC4 20 // _____Ctrl+T, 장치 제어 4(Device Control 4)
#define KEY_NAK 21 // _____Ctrl+U, 부정 응답(Negative Acknowledge)
#define KEY_SYN 22 // _____Ctrl+V, 동기화(Synchronous Idle)
#define KEY_ETB 23 // _____Ctrl+W, 종료 블록(End of Transmission Block)
#define KEY_CAN 24 // _____Ctrl+X, 취소(Cancel)
#define KEY_EM 25 // _____Ctrl+Y, 종료 미디어(End of Medium)
#define KEY_SUB 26 // _____Ctrl+Z, 대체(Substitute)
#define KEY_ESC 27 // ESC, 이스케이프(Escape)
#define KEY_FS 28 // _____Ctrl+\, 파일 구분자(File Separator)
#define KEY_GS 29 // _____Ctrl+], 그룹 구분자(Group Separator)
#define KEY_RS 30 // _____Ctrl+^, 레코드 구분자(Record Separator)
#define KEY_US 31 // _____Ctrl+_, 단위 구분자(Unit Separator)
#define KEY_SPACE 32 // Space, 공백
#define KEY_EXCLAMATION 33 // !, 느낌표
#define KEY_QUOTATION_MARK 34 // ", 쌍따옴표
#define KEY_NUMBER_SIGN 35 // #, 샵
#define KEY_DOLLAR_SIGN 36 // $, 달러 기호
#define KEY_PERCENT_SIGN 37 // %, 퍼센트 기호
#define KEY_AMPERSAND 38 // &, 앰퍼샌드
#define KEY_APOSTROPHE 39 // ', 작은따옴표
#define KEY_LEFT_PARENTHESIS 40 // (, 왼쪽 괄호
#define KEY_RIGHT_PARENTHESIS 41 // ), 오른쪽 괄호
#define KEY_ASTERISK 42 // *, 별표
#define KEY_PLUS_SIGN 43 // +, 더하기 기호
#define KEY_COMMA 44 // ,, 쉼표
#define KEY_MINUS_SIGN 45 // -, 빼기 기호
#define KEY_PERIOD 46 // ., 마침표
#define KEY_SLASH 47 // /, 슬래시
#define KEY_0 48 // 0, 숫자 0
#define KEY_1 49 // 1, 숫자 1
#define KEY_2 50 // 2, 숫자 2
#define KEY_3 51 // 3, 숫자 3
#define KEY_4 52 // 4, 숫자 4
#define KEY_5 53 // 5, 숫자 5
#define KEY_6 54 // 6, 숫자 6
#define KEY_7 55 // 7, 숫자 7
#define KEY_8 56 // 8, 숫자 8
#define KEY_9 57 // 9, 숫자 9
#define KEY_COLON 58 // :, 콜론
#define KEY_SEMICOLON 59 // ;, 세미콜론
#define KEY_LESS_THAN_SIGN 60 // <, 작은 부등호
#define KEY_EQUAL_SIGN 61 // =, 등호
#define KEY_GREATER_THAN_SIGN 62 // >, 큰 부등호
#define KEY_QUESTION_MARK 63 // ?, 물음표
#define KEY_AT_SIGN 64 // @, 골뱅이
#define KEY_A 65 // A, 대문자 A
#define KEY_B 66 // B, 대문자 B
#define KEY_C 67 // C, 대문자 C
#define KEY_D 68 // D, 대문자 D
#define KEY_E 69 // E, 대문자 E
#define KEY_F 70 // F, 대문자 F
#define KEY_G 71 // G, 대문자 G
#define KEY_H 72 // H, 대문자 H
#define KEY_I 73 // I, 대문자 I
#define KEY_J 74 // J, 대문자 J
#define KEY_K 75 // K, 대문자 K
#define KEY_L 76 // L, 대문자 L
#define KEY_M 77 // M, 대문자 M
#define KEY_N 78 // N, 대문자 N
#define KEY_O 79 // O, 대문자 O
#define KEY_P 80 // P, 대문자 P
#define KEY_Q 81 // Q, 대문자 Q
#define KEY_R 82 // R, 대문자 R
#define KEY_S 83 // S, 대문자 S
#define KEY_T 84 // T, 대문자 T
#define KEY_U 85 // U, 대문자 U
#define KEY_V 86 // V, 대문자 V
#define KEY_W 87 // W, 대문자 W
#define KEY_X 88 // X, 대문자 X
#define KEY_Y 89 // Y, 대문자 Y
#define KEY_Z 90 // Z, 대문자 Z
#define KEY_LEFT_SQUARE_BRACKET 91 // [, 왼쪽 대괄호
#define KEY_BACKSLASH 92 // \, 역슬래시
#define KEY_RIGHT_SQUARE_BRACKET 93 // ], 오른쪽 대괄호
#define KEY_CIRCUMFLEX_ACCENT 94 // ^, 캐럿
#define KEY_UNDERSCORE 95 // _, 언더바
#define KEY_GRAVE_ACCENT 96 // `, 그레이브 액센트
#define KEY_a 97 // a, 소문자 a
#define KEY_b 98 // b, 소문자 b
#define KEY_c 99 // c, 소문자 c
#define KEY_d 100 // d, 소문자 d
#define KEY_e 101 // e, 소문자 e
#define KEY_f 102 // f, 소문자 f
#define KEY_g 103 // g, 소문자 g
#define KEY_h 104 // h, 소문자 h
#define KEY_i 105 // i, 소문자 i
#define KEY_j 106 // j, 소문자 j
#define KEY_k 107 // k, 소문자 k
#define KEY_l 108 // l, 소문자 l
#define KEY_m 109 // m, 소문자 m
#define KEY_n 110 // n, 소문자 n
#define KEY_o 111 // o, 소문자 o
#define KEY_p 112 // p, 소문자 p
#define KEY_q 113 // q, 소문자 q
#define KEY_r 114 // r, 소문자 r
#define KEY_s 115 // s, 소문자 s
#define KEY_t 116 // t, 소문자 t
#define KEY_u 117 // u, 소문자 u
#define KEY_v 118 // v, 소문자 v
#define KEY_w 119 // w, 소문자 w
#define KEY_x 120 // x, 소문자 x
#define KEY_y 121 // y, 소문자 y
#define KEY_z 122 // z, 소문자 z
#define KEY_LEFT_CURLY_BRACKET 123 // {, 왼쪽 중괄호
#define KEY_VERTICAL_BAR 124 // |, 수직선
#define KEY_RIGHT_CURLY_BRACKET 125 // }, 오른쪽 중괄호
#define KEY_TILDE 126 // ~, 물결표
#define KEY_DEL 127 // DEL, 삭제