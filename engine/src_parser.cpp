/* A Bison parser, made by GNU Bison 2.3.  */

/* Skeleton implementation for Bison's Yacc-like parsers in C

   Copyright (C) 1984, 1989, 1990, 2000, 2001, 2002, 2003, 2004, 2005, 2006
   Free Software Foundation, Inc.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2, or (at your option)
   any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 51 Franklin Street, Fifth Floor,
   Boston, MA 02110-1301, USA.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.

   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

/* C LALR(1) parser skeleton written by Richard Stallman, by
   simplifying the original so-called "semantic" parser.  */

/* All symbols defined below should begin with yy or YY, to avoid
   infringing on user name space.  This should be done even for local
   variables, as they might otherwise be expanded by user macros.
   There are some unavoidable exceptions within include files to
   define necessary library symbols; they are noted "INFRINGES ON
   USER NAME SPACE" below.  */

/* Identify Bison output.  */
#define YYBISON 1

/* Bison version.  */
#define YYBISON_VERSION "2.3"

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 1

/* Using locations.  */
#define YYLSP_NEEDED 0

/* Substitute the variable and function names.  */
#define yyparse flc_src_parse
#define yylex   flc_src_lex
#define yyerror flc_src_error
#define yylval  flc_src_lval
#define yychar  flc_src_char
#define yydebug flc_src_debug
#define yynerrs flc_src_nerrs


/* Tokens.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
   /* Put the tokens into the symbol table, so that GDB and other debuggers
      know about them.  */
   enum yytokentype {
     EOL = 258,
     INTNUM = 259,
     DBLNUM = 260,
     SYMBOL = 261,
     STRING = 262,
     NIL = 263,
     END = 264,
     DEF = 265,
     WHILE = 266,
     BREAK = 267,
     CONTINUE = 268,
     DROPPING = 269,
     IF = 270,
     ELSE = 271,
     ELIF = 272,
     FOR = 273,
     FORFIRST = 274,
     FORLAST = 275,
     FORALL = 276,
     SWITCH = 277,
     CASE = 278,
     DEFAULT = 279,
     SELECT = 280,
     SENDER = 281,
     SELF = 282,
     GIVE = 283,
     TRY = 284,
     CATCH = 285,
     RAISE = 286,
     CLASS = 287,
     FROM = 288,
     OBJECT = 289,
     RETURN = 290,
     GLOBAL = 291,
     LAMBDA = 292,
     INIT = 293,
     LOAD = 294,
     LAUNCH = 295,
     CONST_KW = 296,
     ATTRIBUTES = 297,
     PASS = 298,
     EXPORT = 299,
     COLON = 300,
     FUNCDECL = 301,
     STATIC = 302,
     FORDOT = 303,
     LOOP = 304,
     OUTER_STRING = 305,
     CLOSEPAR = 306,
     OPENPAR = 307,
     CLOSESQUARE = 308,
     OPENSQUARE = 309,
     DOT = 310,
     ASSIGN_POW = 311,
     ASSIGN_SHL = 312,
     ASSIGN_SHR = 313,
     ASSIGN_BXOR = 314,
     ASSIGN_BOR = 315,
     ASSIGN_BAND = 316,
     ASSIGN_MOD = 317,
     ASSIGN_DIV = 318,
     ASSIGN_MUL = 319,
     ASSIGN_SUB = 320,
     ASSIGN_ADD = 321,
     ARROW = 322,
     FOR_STEP = 323,
     OP_TO = 324,
     COMMA = 325,
     QUESTION = 326,
     OR = 327,
     AND = 328,
     NOT = 329,
     LET = 330,
     LE = 331,
     GE = 332,
     LT = 333,
     GT = 334,
     NEQ = 335,
     EEQ = 336,
     OP_EQ = 337,
     OP_ASSIGN = 338,
     PROVIDES = 339,
     OP_NOTIN = 340,
     OP_IN = 341,
     HASNT = 342,
     HAS = 343,
     DIESIS = 344,
     ATSIGN = 345,
     CAP = 346,
     VBAR = 347,
     AMPER = 348,
     MINUS = 349,
     PLUS = 350,
     PERCENT = 351,
     SLASH = 352,
     STAR = 353,
     POW = 354,
     SHR = 355,
     SHL = 356,
     BANG = 357,
     NEG = 358,
     DECREMENT = 359,
     INCREMENT = 360,
     DOLLAR = 361
   };
#endif
/* Tokens.  */
#define EOL 258
#define INTNUM 259
#define DBLNUM 260
#define SYMBOL 261
#define STRING 262
#define NIL 263
#define END 264
#define DEF 265
#define WHILE 266
#define BREAK 267
#define CONTINUE 268
#define DROPPING 269
#define IF 270
#define ELSE 271
#define ELIF 272
#define FOR 273
#define FORFIRST 274
#define FORLAST 275
#define FORALL 276
#define SWITCH 277
#define CASE 278
#define DEFAULT 279
#define SELECT 280
#define SENDER 281
#define SELF 282
#define GIVE 283
#define TRY 284
#define CATCH 285
#define RAISE 286
#define CLASS 287
#define FROM 288
#define OBJECT 289
#define RETURN 290
#define GLOBAL 291
#define LAMBDA 292
#define INIT 293
#define LOAD 294
#define LAUNCH 295
#define CONST_KW 296
#define ATTRIBUTES 297
#define PASS 298
#define EXPORT 299
#define COLON 300
#define FUNCDECL 301
#define STATIC 302
#define FORDOT 303
#define LOOP 304
#define OUTER_STRING 305
#define CLOSEPAR 306
#define OPENPAR 307
#define CLOSESQUARE 308
#define OPENSQUARE 309
#define DOT 310
#define ASSIGN_POW 311
#define ASSIGN_SHL 312
#define ASSIGN_SHR 313
#define ASSIGN_BXOR 314
#define ASSIGN_BOR 315
#define ASSIGN_BAND 316
#define ASSIGN_MOD 317
#define ASSIGN_DIV 318
#define ASSIGN_MUL 319
#define ASSIGN_SUB 320
#define ASSIGN_ADD 321
#define ARROW 322
#define FOR_STEP 323
#define OP_TO 324
#define COMMA 325
#define QUESTION 326
#define OR 327
#define AND 328
#define NOT 329
#define LET 330
#define LE 331
#define GE 332
#define LT 333
#define GT 334
#define NEQ 335
#define EEQ 336
#define OP_EQ 337
#define OP_ASSIGN 338
#define PROVIDES 339
#define OP_NOTIN 340
#define OP_IN 341
#define HASNT 342
#define HAS 343
#define DIESIS 344
#define ATSIGN 345
#define CAP 346
#define VBAR 347
#define AMPER 348
#define MINUS 349
#define PLUS 350
#define PERCENT 351
#define SLASH 352
#define STAR 353
#define POW 354
#define SHR 355
#define SHL 356
#define BANG 357
#define NEG 358
#define DECREMENT 359
#define INCREMENT 360
#define DOLLAR 361




/* Copy the first part of user declarations.  */
#line 23 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"


#include <math.h>
#include <stdio.h>
#include <iostream>

#include <falcon/setup.h>
#include <falcon/compiler.h>
#include <falcon/src_lexer.h>
#include <falcon/syntree.h>
#include <falcon/error.h>
#include <stdlib.h>

#include <falcon/fassert.h>

#include <falcon/memory.h>

#define YYMALLOC	Falcon::memAlloc
#define YYFREE Falcon::memFree

#define  COMPILER  ( reinterpret_cast< Falcon::Compiler *>(yyparam) )
#define  LINE      ( COMPILER->lexer()->previousLine() )
#define  CURRENT_LINE      ( COMPILER->lexer()->line() )


#define YYPARSE_PARAM yyparam
#define YYLEX_PARAM yyparam

int flc_src_parse( void *param );
void flc_src_error (const char *s);

inline int flc_src_lex (void *lvalp, void *yyparam)
{
   return COMPILER->lexer()->doLex( lvalp );
}

/* Cures a bug in bison 1.8  */
#undef __GNUC_MINOR__



/* Enabling traces.  */
#ifndef YYDEBUG
# define YYDEBUG 0
#endif

/* Enabling verbose error messages.  */
#ifdef YYERROR_VERBOSE
# undef YYERROR_VERBOSE
# define YYERROR_VERBOSE 1
#else
# define YYERROR_VERBOSE 0
#endif

/* Enabling the token table.  */
#ifndef YYTOKEN_TABLE
# define YYTOKEN_TABLE 0
#endif

#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef union
#line 66 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
lex_value_t {
   Falcon::int64 integer;
   Falcon::numeric numeric;
   char * charp;
   Falcon::String *stringp;
   Falcon::Symbol *symbol;
   Falcon::Value *fal_val;
   Falcon::Expression *fal_exp;
   Falcon::Statement *fal_stat;
   Falcon::ArrayDecl *fal_adecl;
   Falcon::DictDecl *fal_ddecl;
   Falcon::SymbolList *fal_symlist;
}
/* Line 187 of yacc.c.  */
#line 371 "/home/gian/Progetti/falcon/core/engine/src_parser.cpp"
	YYSTYPE;
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif



/* Copy the second part of user declarations.  */


/* Line 216 of yacc.c.  */
#line 384 "/home/gian/Progetti/falcon/core/engine/src_parser.cpp"

#ifdef short
# undef short
#endif

#ifdef YYTYPE_UINT8
typedef YYTYPE_UINT8 yytype_uint8;
#else
typedef unsigned char yytype_uint8;
#endif

#ifdef YYTYPE_INT8
typedef YYTYPE_INT8 yytype_int8;
#elif (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
typedef signed char yytype_int8;
#else
typedef short int yytype_int8;
#endif

#ifdef YYTYPE_UINT16
typedef YYTYPE_UINT16 yytype_uint16;
#else
typedef unsigned short int yytype_uint16;
#endif

#ifdef YYTYPE_INT16
typedef YYTYPE_INT16 yytype_int16;
#else
typedef short int yytype_int16;
#endif

#ifndef YYSIZE_T
# ifdef __SIZE_TYPE__
#  define YYSIZE_T __SIZE_TYPE__
# elif defined size_t
#  define YYSIZE_T size_t
# elif ! defined YYSIZE_T && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
#  include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  define YYSIZE_T size_t
# else
#  define YYSIZE_T unsigned int
# endif
#endif

#define YYSIZE_MAXIMUM ((YYSIZE_T) -1)

#ifndef YY_
# if YYENABLE_NLS
#  if ENABLE_NLS
#   include <libintl.h> /* INFRINGES ON USER NAME SPACE */
#   define YY_(msgid) dgettext ("bison-runtime", msgid)
#  endif
# endif
# ifndef YY_
#  define YY_(msgid) msgid
# endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if ! defined lint || defined __GNUC__
# define YYUSE(e) ((void) (e))
#else
# define YYUSE(e) /* empty */
#endif

/* Identity function, used to suppress warnings about constant conditions.  */
#ifndef lint
# define YYID(n) (n)
#else
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static int
YYID (int i)
#else
static int
YYID (i)
    int i;
#endif
{
  return i;
}
#endif

#if ! defined yyoverflow || YYERROR_VERBOSE

/* The parser invokes alloca or malloc; define the necessary symbols.  */

# ifdef YYSTACK_USE_ALLOCA
#  if YYSTACK_USE_ALLOCA
#   ifdef __GNUC__
#    define YYSTACK_ALLOC __builtin_alloca
#   elif defined __BUILTIN_VA_ARG_INCR
#    include <alloca.h> /* INFRINGES ON USER NAME SPACE */
#   elif defined _AIX
#    define YYSTACK_ALLOC __alloca
#   elif defined _MSC_VER
#    include <malloc.h> /* INFRINGES ON USER NAME SPACE */
#    define alloca _alloca
#   else
#    define YYSTACK_ALLOC alloca
#    if ! defined _ALLOCA_H && ! defined _STDLIB_H && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
#     include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#     ifndef _STDLIB_H
#      define _STDLIB_H 1
#     endif
#    endif
#   endif
#  endif
# endif

# ifdef YYSTACK_ALLOC
   /* Pacify GCC's `empty if-body' warning.  */
#  define YYSTACK_FREE(Ptr) do { /* empty */; } while (YYID (0))
#  ifndef YYSTACK_ALLOC_MAXIMUM
    /* The OS might guarantee only one guard page at the bottom of the stack,
       and a page size can be as small as 4096 bytes.  So we cannot safely
       invoke alloca (N) if N exceeds 4096.  Use a slightly smaller number
       to allow for a few compiler-allocated temporary stack slots.  */
#   define YYSTACK_ALLOC_MAXIMUM 4032 /* reasonable circa 2006 */
#  endif
# else
#  define YYSTACK_ALLOC YYMALLOC
#  define YYSTACK_FREE YYFREE
#  ifndef YYSTACK_ALLOC_MAXIMUM
#   define YYSTACK_ALLOC_MAXIMUM YYSIZE_MAXIMUM
#  endif
#  if (defined __cplusplus && ! defined _STDLIB_H \
       && ! ((defined YYMALLOC || defined malloc) \
	     && (defined YYFREE || defined free)))
#   include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#   ifndef _STDLIB_H
#    define _STDLIB_H 1
#   endif
#  endif
#  ifndef YYMALLOC
#   define YYMALLOC malloc
#   if ! defined malloc && ! defined _STDLIB_H && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
void *malloc (YYSIZE_T); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
#  ifndef YYFREE
#   define YYFREE free
#   if ! defined free && ! defined _STDLIB_H && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
void free (void *); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
# endif
#endif /* ! defined yyoverflow || YYERROR_VERBOSE */


#if (! defined yyoverflow \
     && (! defined __cplusplus \
	 || (defined YYSTYPE_IS_TRIVIAL && YYSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc
{
  yytype_int16 yyss;
  YYSTYPE yyvs;
  };

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (sizeof (union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (sizeof (yytype_int16) + sizeof (YYSTYPE)) \
      + YYSTACK_GAP_MAXIMUM)

/* Copy COUNT objects from FROM to TO.  The source and destination do
   not overlap.  */
# ifndef YYCOPY
#  if defined __GNUC__ && 1 < __GNUC__
#   define YYCOPY(To, From, Count) \
      __builtin_memcpy (To, From, (Count) * sizeof (*(From)))
#  else
#   define YYCOPY(To, From, Count)		\
      do					\
	{					\
	  YYSIZE_T yyi;				\
	  for (yyi = 0; yyi < (Count); yyi++)	\
	    (To)[yyi] = (From)[yyi];		\
	}					\
      while (YYID (0))
#  endif
# endif

/* Relocate STACK from its old location to the new one.  The
   local variables YYSIZE and YYSTACKSIZE give the old and new number of
   elements in the stack, and YYPTR gives the new location of the
   stack.  Advance YYPTR to a properly aligned location for the next
   stack.  */
# define YYSTACK_RELOCATE(Stack)					\
    do									\
      {									\
	YYSIZE_T yynewbytes;						\
	YYCOPY (&yyptr->Stack, Stack, yysize);				\
	Stack = &yyptr->Stack;						\
	yynewbytes = yystacksize * sizeof (*Stack) + YYSTACK_GAP_MAXIMUM; \
	yyptr += yynewbytes / sizeof (*yyptr);				\
      }									\
    while (YYID (0))

#endif

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL  3
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   6199

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  107
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  172
/* YYNRULES -- Number of rules.  */
#define YYNRULES  437
/* YYNRULES -- Number of states.  */
#define YYNSTATES  834

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   361

#define YYTRANSLATE(YYX)						\
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
static const yytype_uint8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     1,     2,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
      55,    56,    57,    58,    59,    60,    61,    62,    63,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
      75,    76,    77,    78,    79,    80,    81,    82,    83,    84,
      85,    86,    87,    88,    89,    90,    91,    92,    93,    94,
      95,    96,    97,    98,    99,   100,   101,   102,   103,   104,
     105,   106
};

#if YYDEBUG
/* YYPRHS[YYN] -- Index of the first RHS symbol of rule number YYN in
   YYRHS.  */
static const yytype_uint16 yyprhs[] =
{
       0,     0,     3,     5,     6,     9,    11,    14,    18,    20,
      22,    24,    26,    28,    30,    32,    34,    38,    42,    46,
      48,    50,    54,    58,    62,    65,    69,    75,    78,    80,
      82,    84,    86,    88,    90,    92,    94,    96,    98,   100,
     102,   104,   106,   108,   110,   112,   114,   116,   118,   120,
     124,   128,   133,   139,   144,   149,   156,   163,   165,   167,
     169,   171,   173,   175,   177,   179,   181,   183,   185,   190,
     195,   200,   205,   210,   215,   220,   225,   230,   235,   240,
     241,   247,   248,   255,   258,   262,   265,   269,   273,   276,
     280,   281,   288,   291,   295,   299,   303,   307,   308,   310,
     311,   315,   318,   322,   323,   328,   332,   336,   337,   340,
     343,   347,   350,   354,   358,   359,   365,   368,   376,   386,
     390,   398,   408,   412,   413,   423,   430,   436,   437,   440,
     442,   444,   446,   448,   452,   456,   460,   463,   467,   470,
     474,   476,   477,   484,   488,   492,   493,   500,   504,   508,
     509,   516,   520,   524,   525,   532,   536,   540,   541,   544,
     548,   550,   551,   557,   558,   564,   565,   571,   572,   578,
     579,   580,   584,   585,   587,   590,   593,   596,   598,   602,
     604,   606,   608,   612,   614,   615,   622,   626,   630,   631,
     634,   638,   640,   641,   647,   648,   654,   655,   661,   662,
     668,   670,   674,   675,   677,   679,   685,   690,   694,   698,
     699,   706,   709,   713,   714,   716,   718,   721,   724,   727,
     732,   736,   742,   746,   748,   752,   754,   756,   760,   764,
     770,   773,   781,   782,   792,   796,   804,   805,   814,   817,
     818,   820,   825,   827,   828,   829,   835,   836,   840,   843,
     847,   850,   854,   858,   862,   866,   872,   878,   882,   888,
     894,   898,   901,   905,   909,   911,   915,   920,   924,   927,
     931,   934,   938,   939,   941,   945,   948,   952,   955,   956,
     965,   969,   972,   973,   979,   980,   988,   989,   992,   994,
     998,  1001,  1002,  1008,  1010,  1014,  1016,  1018,  1020,  1021,
    1024,  1026,  1028,  1030,  1032,  1033,  1041,  1047,  1052,  1053,
    1057,  1061,  1063,  1066,  1070,  1075,  1076,  1085,  1088,  1091,
    1092,  1095,  1097,  1099,  1101,  1103,  1104,  1109,  1111,  1115,
    1119,  1121,  1124,  1128,  1132,  1134,  1136,  1138,  1140,  1142,
    1144,  1146,  1148,  1150,  1153,  1158,  1164,  1168,  1170,  1172,
    1176,  1179,  1183,  1187,  1191,  1195,  1199,  1203,  1207,  1211,
    1215,  1219,  1222,  1227,  1232,  1236,  1239,  1242,  1245,  1248,
    1252,  1256,  1260,  1264,  1268,  1272,  1276,  1280,  1284,  1287,
    1291,  1295,  1299,  1303,  1307,  1310,  1313,  1316,  1318,  1320,
    1324,  1329,  1335,  1338,  1340,  1342,  1344,  1346,  1352,  1356,
    1361,  1366,  1372,  1379,  1384,  1385,  1394,  1395,  1397,  1399,
    1402,  1403,  1410,  1417,  1418,  1427,  1430,  1431,  1437,  1439,
    1442,  1448,  1452,  1455,  1460,  1461,  1468,  1472,  1477,  1478,
    1485,  1487,  1491,  1493,  1498,  1500,  1504,  1508
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int16 yyrhs[] =
{
     108,     0,    -1,   109,    -1,    -1,   109,   110,    -1,   111,
      -1,     9,     3,    -1,    23,     1,     3,    -1,   112,    -1,
     202,    -1,   225,    -1,   243,    -1,   113,    -1,   217,    -1,
     218,    -1,   220,    -1,    39,     6,     3,    -1,    39,     7,
       3,    -1,    39,     1,     3,    -1,   115,    -1,     3,    -1,
      46,     1,     3,    -1,    34,     1,     3,    -1,    32,     1,
       3,    -1,     1,     3,    -1,   254,    83,   257,    -1,   114,
      70,   254,    83,   257,    -1,   257,     3,    -1,   116,    -1,
     117,    -1,   118,    -1,   130,    -1,   148,    -1,   152,    -1,
     165,    -1,   180,    -1,   135,    -1,   146,    -1,   147,    -1,
     191,    -1,   192,    -1,   201,    -1,   252,    -1,   248,    -1,
     215,    -1,   216,    -1,   156,    -1,   157,    -1,   158,    -1,
      10,   114,     3,    -1,    10,     1,     3,    -1,   256,    83,
     257,     3,    -1,   256,    83,   106,   106,     3,    -1,   256,
      83,   275,     3,    -1,   256,    83,   263,     3,    -1,   256,
      70,   277,    83,   257,     3,    -1,   256,    70,   277,    83,
     275,     3,    -1,   119,    -1,   120,    -1,   121,    -1,   122,
      -1,   123,    -1,   124,    -1,   125,    -1,   126,    -1,   127,
      -1,   128,    -1,   129,    -1,   257,    66,   257,     3,    -1,
     256,    65,   257,     3,    -1,   256,    64,   257,     3,    -1,
     256,    63,   257,     3,    -1,   256,    62,   257,     3,    -1,
     256,    56,   257,     3,    -1,   256,    61,   257,     3,    -1,
     256,    60,   257,     3,    -1,   256,    59,   257,     3,    -1,
     256,    57,   257,     3,    -1,   256,    58,   257,     3,    -1,
      -1,   133,   131,   145,     9,     3,    -1,    -1,    49,   113,
     132,   145,     9,     3,    -1,   134,   113,    -1,    11,   257,
       3,    -1,    49,     3,    -1,    11,     1,     3,    -1,    11,
     257,    45,    -1,    49,    45,    -1,    11,     1,    45,    -1,
      -1,   137,   136,   145,   139,     9,     3,    -1,   138,   113,
      -1,    15,   257,     3,    -1,    15,     1,     3,    -1,    15,
     257,    45,    -1,    15,     1,    45,    -1,    -1,   142,    -1,
      -1,   141,   140,   145,    -1,    16,     3,    -1,    16,     1,
       3,    -1,    -1,   144,   143,   145,   139,    -1,    17,   257,
       3,    -1,    17,     1,     3,    -1,    -1,   145,   113,    -1,
      12,     3,    -1,    12,     1,     3,    -1,    13,     3,    -1,
      13,    14,     3,    -1,    13,     1,     3,    -1,    -1,   150,
     149,   145,     9,     3,    -1,   151,   113,    -1,    18,   256,
      83,   257,    69,   257,     3,    -1,    18,   256,    83,   257,
      69,   257,    68,   257,     3,    -1,    18,     1,     3,    -1,
      18,   256,    83,   257,    69,   257,    45,    -1,    18,   256,
      83,   257,    69,   257,    68,   257,    45,    -1,    18,     1,
      45,    -1,    -1,    18,   277,    86,   257,     3,   153,   154,
       9,     3,    -1,    18,   277,    86,   257,    45,   113,    -1,
      18,   277,    86,     1,     3,    -1,    -1,   155,   154,    -1,
     113,    -1,   159,    -1,   161,    -1,   163,    -1,    48,   257,
       3,    -1,    48,     1,     3,    -1,    78,   275,     3,    -1,
      78,     3,    -1,    79,   275,     3,    -1,    79,     3,    -1,
      78,     1,     3,    -1,    50,    -1,    -1,    19,     3,   160,
     145,     9,     3,    -1,    19,    45,   113,    -1,    19,     1,
       3,    -1,    -1,    20,     3,   162,   145,     9,     3,    -1,
      20,    45,   113,    -1,    20,     1,     3,    -1,    -1,    21,
       3,   164,   145,     9,     3,    -1,    21,    45,   113,    -1,
      21,     1,     3,    -1,    -1,   167,   166,   168,   174,     9,
       3,    -1,    22,   257,     3,    -1,    22,     1,     3,    -1,
      -1,   168,   169,    -1,   168,     1,     3,    -1,     3,    -1,
      -1,    23,   178,     3,   170,   145,    -1,    -1,    23,   178,
      45,   171,   113,    -1,    -1,    23,     1,     3,   172,   145,
      -1,    -1,    23,     1,    45,   173,   113,    -1,    -1,    -1,
     176,   175,   177,    -1,    -1,    24,    -1,    24,     1,    -1,
       3,   145,    -1,    45,   113,    -1,   179,    -1,   178,    70,
     179,    -1,     8,    -1,     4,    -1,     7,    -1,     4,    69,
       4,    -1,     6,    -1,    -1,   182,   181,   183,   174,     9,
       3,    -1,    25,   257,     3,    -1,    25,     1,     3,    -1,
      -1,   183,   184,    -1,   183,     1,     3,    -1,     3,    -1,
      -1,    23,   189,     3,   185,   145,    -1,    -1,    23,   189,
      45,   186,   113,    -1,    -1,    23,     1,     3,   187,   145,
      -1,    -1,    23,     1,    45,   188,   113,    -1,   190,    -1,
     189,    70,   190,    -1,    -1,     4,    -1,     6,    -1,    28,
     275,    69,   257,     3,    -1,    28,   275,     1,     3,    -1,
      28,     1,     3,    -1,    29,    45,   113,    -1,    -1,   194,
     193,   145,   195,     9,     3,    -1,    29,     3,    -1,    29,
       1,     3,    -1,    -1,   196,    -1,   197,    -1,   196,   197,
      -1,   198,   145,    -1,    30,     3,    -1,    30,    86,   254,
       3,    -1,    30,   199,     3,    -1,    30,   199,    86,   254,
       3,    -1,    30,     1,     3,    -1,   200,    -1,   199,    70,
     200,    -1,     4,    -1,     6,    -1,    31,   257,     3,    -1,
      31,     1,     3,    -1,   203,   210,   145,     9,     3,    -1,
     205,   113,    -1,   207,    52,   261,   208,   261,    51,     3,
      -1,    -1,   207,    52,   261,   208,     1,   204,   261,    51,
       3,    -1,   207,     1,     3,    -1,   207,    52,   261,   208,
     261,    51,    45,    -1,    -1,   207,    52,   261,     1,   206,
     261,    51,    45,    -1,    46,     6,    -1,    -1,   209,    -1,
     208,    70,   261,   209,    -1,     6,    -1,    -1,    -1,   213,
     211,   145,     9,     3,    -1,    -1,   214,   212,   113,    -1,
      47,     3,    -1,    47,     1,     3,    -1,    47,    45,    -1,
      47,     1,    45,    -1,    40,   259,     3,    -1,    40,     1,
       3,    -1,    43,   257,     3,    -1,    43,   257,    86,   257,
       3,    -1,    43,   257,    86,     1,     3,    -1,    43,     1,
       3,    -1,    41,     6,    83,   253,     3,    -1,    41,     6,
      83,     1,     3,    -1,    41,     1,     3,    -1,    44,     3,
      -1,    44,   219,     3,    -1,    44,     1,     3,    -1,     6,
      -1,   219,    70,     6,    -1,   221,   224,     9,     3,    -1,
     222,   223,     3,    -1,    42,     3,    -1,    42,     1,     3,
      -1,    42,    45,    -1,    42,     1,    45,    -1,    -1,     6,
      -1,   223,    70,     6,    -1,   223,     3,    -1,   224,   223,
       3,    -1,     1,     3,    -1,    -1,    32,     6,   226,   227,
     236,   241,     9,     3,    -1,   228,   230,     3,    -1,     1,
       3,    -1,    -1,    52,   261,   208,   261,    51,    -1,    -1,
      52,   261,   208,     1,   229,   261,    51,    -1,    -1,    33,
     231,    -1,   232,    -1,   231,    70,   232,    -1,     6,   233,
      -1,    -1,    52,   261,   234,   261,    51,    -1,   235,    -1,
     234,    70,   235,    -1,   253,    -1,     6,    -1,    27,    -1,
      -1,   236,   237,    -1,     3,    -1,   202,    -1,   240,    -1,
     238,    -1,    -1,    38,     3,   239,   210,   145,     9,     3,
      -1,    47,     6,    83,   257,     3,    -1,     6,    83,   257,
       3,    -1,    -1,    88,   242,     3,    -1,    88,     1,     3,
      -1,     6,    -1,    74,     6,    -1,   242,    70,     6,    -1,
     242,    70,    74,     6,    -1,    -1,    34,     6,   244,   245,
     246,   241,     9,     3,    -1,   230,     3,    -1,     1,     3,
      -1,    -1,   246,   247,    -1,     3,    -1,   202,    -1,   240,
      -1,   238,    -1,    -1,    36,   249,   250,     3,    -1,   251,
      -1,   250,    70,   251,    -1,   250,    70,     1,    -1,     6,
      -1,    35,     3,    -1,    35,   257,     3,    -1,    35,     1,
       3,    -1,     8,    -1,     4,    -1,     5,    -1,     7,    -1,
       6,    -1,   254,    -1,    27,    -1,    26,    -1,   255,    -1,
     256,   258,    -1,   256,    54,   257,    53,    -1,   256,    54,
      98,   257,    53,    -1,   256,    55,     6,    -1,   253,    -1,
     256,    -1,   257,    95,   257,    -1,    94,   257,    -1,   257,
      94,   257,    -1,   257,    98,   257,    -1,   257,    97,   257,
      -1,   257,    96,   257,    -1,   257,    99,   257,    -1,   257,
      93,   257,    -1,   257,    92,   257,    -1,   257,    91,   257,
      -1,   257,   101,   257,    -1,   257,   100,   257,    -1,   102,
     257,    -1,    75,   256,    82,   257,    -1,    75,   256,    83,
     257,    -1,   257,    80,   257,    -1,   257,   105,    -1,   105,
     257,    -1,   257,   104,    -1,   104,   257,    -1,   257,    81,
     257,    -1,   257,    82,   257,    -1,   257,    83,   257,    -1,
     257,    79,   257,    -1,   257,    78,   257,    -1,   257,    77,
     257,    -1,   257,    76,   257,    -1,   257,    73,   257,    -1,
     257,    72,   257,    -1,    74,   257,    -1,   257,    88,   257,
      -1,   257,    87,   257,    -1,   257,    86,   257,    -1,   257,
      85,   257,    -1,   257,    84,     6,    -1,   106,   257,    -1,
      90,   257,    -1,    89,   257,    -1,   267,    -1,   259,    -1,
     259,    55,     6,    -1,   259,    54,   257,    53,    -1,   259,
      54,    98,   257,    53,    -1,   259,   258,    -1,   270,    -1,
     271,    -1,   273,    -1,   258,    -1,    52,   261,   257,   261,
      51,    -1,    54,    45,    53,    -1,    54,   257,    45,    53,
      -1,    54,    45,   257,    53,    -1,    54,   257,    45,   257,
      53,    -1,   257,    52,   261,   276,   261,    51,    -1,   257,
      52,   261,    51,    -1,    -1,   257,    52,   261,   276,     1,
     260,   261,    51,    -1,    -1,   262,    -1,     3,    -1,   262,
       3,    -1,    -1,    46,   264,   265,   210,   145,     9,    -1,
      52,   261,   208,   261,    51,     3,    -1,    -1,    52,   261,
     208,     1,   266,   261,    51,     3,    -1,     1,     3,    -1,
      -1,    37,   268,   269,    67,   257,    -1,   208,    -1,     1,
       3,    -1,   257,    71,   257,    45,   257,    -1,   257,    71,
       1,    -1,    54,    53,    -1,    54,   276,   261,    53,    -1,
      -1,    54,   276,     1,   272,   261,    53,    -1,    54,    67,
      53,    -1,    54,   278,   261,    53,    -1,    -1,    54,   278,
       1,   274,   261,    53,    -1,   257,    -1,   275,    70,   257,
      -1,   257,    -1,   276,    70,   261,   257,    -1,   254,    -1,
     277,    70,   254,    -1,   257,    67,   257,    -1,   278,    70,
     261,   257,    67,   257,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   192,   192,   195,   197,   201,   202,   203,   208,   213,
     218,   223,   228,   233,   234,   235,   239,   245,   251,   259,
     260,   263,   264,   265,   266,   271,   276,   283,   284,   285,
     286,   287,   288,   289,   290,   291,   292,   293,   294,   295,
     296,   297,   298,   299,   300,   301,   302,   303,   304,   308,
     310,   316,   320,   324,   328,   332,   337,   347,   348,   349,
     350,   351,   352,   353,   354,   355,   356,   357,   361,   368,
     375,   382,   389,   396,   403,   410,   417,   423,   429,   437,
     437,   451,   451,   467,   475,   476,   477,   481,   482,   483,
     487,   487,   502,   512,   513,   517,   518,   522,   524,   525,
     525,   534,   535,   540,   540,   552,   553,   556,   558,   564,
     573,   581,   591,   600,   608,   608,   622,   638,   642,   646,
     654,   658,   662,   672,   671,   695,   709,   713,   715,   719,
     726,   727,   728,   732,   745,   753,   757,   763,   768,   775,
     784,   794,   794,   808,   816,   820,   820,   833,   841,   845,
     845,   859,   867,   871,   871,   888,   889,   896,   898,   899,
     903,   905,   904,   915,   915,   927,   927,   939,   939,   955,
     958,   957,   970,   971,   972,   975,   976,   982,   983,   987,
     996,  1008,  1019,  1030,  1051,  1051,  1068,  1069,  1076,  1078,
    1079,  1083,  1085,  1084,  1095,  1095,  1108,  1108,  1120,  1120,
    1138,  1139,  1142,  1143,  1155,  1176,  1180,  1185,  1193,  1200,
    1199,  1218,  1219,  1222,  1224,  1228,  1229,  1233,  1238,  1256,
    1276,  1286,  1297,  1305,  1306,  1310,  1322,  1345,  1346,  1353,
    1363,  1372,  1373,  1373,  1377,  1381,  1382,  1382,  1389,  1443,
    1445,  1446,  1450,  1465,  1468,  1467,  1479,  1478,  1493,  1494,
    1498,  1499,  1508,  1512,  1520,  1527,  1537,  1543,  1555,  1565,
    1570,  1582,  1591,  1598,  1606,  1611,  1623,  1630,  1640,  1641,
    1644,  1645,  1648,  1650,  1654,  1661,  1662,  1663,  1675,  1674,
    1733,  1736,  1742,  1744,  1745,  1745,  1751,  1753,  1757,  1758,
    1762,  1796,  1798,  1807,  1808,  1812,  1813,  1822,  1825,  1827,
    1831,  1832,  1835,  1854,  1858,  1858,  1892,  1914,  1941,  1943,
    1944,  1951,  1959,  1965,  1971,  1985,  1984,  2048,  2049,  2055,
    2057,  2061,  2062,  2065,  2084,  2093,  2092,  2110,  2111,  2112,
    2119,  2135,  2136,  2137,  2147,  2148,  2149,  2150,  2154,  2172,
    2173,  2174,  2185,  2186,  2191,  2196,  2202,  2211,  2212,  2213,
    2214,  2215,  2216,  2217,  2218,  2219,  2220,  2221,  2222,  2223,
    2224,  2225,  2226,  2228,  2230,  2231,  2232,  2233,  2234,  2235,
    2236,  2237,  2238,  2239,  2240,  2241,  2242,  2243,  2244,  2245,
    2246,  2247,  2248,  2249,  2250,  2251,  2252,  2253,  2254,  2255,
    2259,  2263,  2267,  2271,  2272,  2273,  2274,  2275,  2280,  2283,
    2286,  2289,  2295,  2301,  2306,  2306,  2314,  2316,  2320,  2321,
    2326,  2325,  2368,  2369,  2369,  2373,  2382,  2381,  2424,  2425,
    2434,  2435,  2445,  2446,  2450,  2450,  2458,  2459,  2460,  2460,
    2468,  2469,  2473,  2474,  2478,  2485,  2492,  2493
};
#endif

#if YYDEBUG || YYERROR_VERBOSE || YYTOKEN_TABLE
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "$end", "error", "$undefined", "EOL", "INTNUM", "DBLNUM", "SYMBOL",
  "STRING", "NIL", "END", "DEF", "WHILE", "BREAK", "CONTINUE", "DROPPING",
  "IF", "ELSE", "ELIF", "FOR", "FORFIRST", "FORLAST", "FORALL", "SWITCH",
  "CASE", "DEFAULT", "SELECT", "SENDER", "SELF", "GIVE", "TRY", "CATCH",
  "RAISE", "CLASS", "FROM", "OBJECT", "RETURN", "GLOBAL", "LAMBDA", "INIT",
  "LOAD", "LAUNCH", "CONST_KW", "ATTRIBUTES", "PASS", "EXPORT", "COLON",
  "FUNCDECL", "STATIC", "FORDOT", "LOOP", "OUTER_STRING", "CLOSEPAR",
  "OPENPAR", "CLOSESQUARE", "OPENSQUARE", "DOT", "ASSIGN_POW",
  "ASSIGN_SHL", "ASSIGN_SHR", "ASSIGN_BXOR", "ASSIGN_BOR", "ASSIGN_BAND",
  "ASSIGN_MOD", "ASSIGN_DIV", "ASSIGN_MUL", "ASSIGN_SUB", "ASSIGN_ADD",
  "ARROW", "FOR_STEP", "OP_TO", "COMMA", "QUESTION", "OR", "AND", "NOT",
  "LET", "LE", "GE", "LT", "GT", "NEQ", "EEQ", "OP_EQ", "OP_ASSIGN",
  "PROVIDES", "OP_NOTIN", "OP_IN", "HASNT", "HAS", "DIESIS", "ATSIGN",
  "CAP", "VBAR", "AMPER", "MINUS", "PLUS", "PERCENT", "SLASH", "STAR",
  "POW", "SHR", "SHL", "BANG", "NEG", "DECREMENT", "INCREMENT", "DOLLAR",
  "$accept", "input", "body", "line", "toplevel_statement",
  "load_statement", "statement", "assignment_def_list", "base_statement",
  "def_statement", "assignment", "op_assignment", "autoadd", "autosub",
  "automul", "autodiv", "automod", "autopow", "autoband", "autobor",
  "autobxor", "autoshl", "autoshr", "while_statement", "@1", "@2",
  "while_decl", "while_short_decl", "if_statement", "@3", "if_decl",
  "if_short_decl", "elif_or_else", "@4", "else_decl", "elif_statement",
  "@5", "elif_decl", "statement_list", "break_statement",
  "continue_statement", "for_statement", "@6", "for_decl",
  "for_decl_short", "forin_statement", "@7", "forin_statement_list",
  "forin_statement_elem", "fordot_statement", "self_print_statement",
  "outer_print_statement", "first_loop_block", "@8", "last_loop_block",
  "@9", "all_loop_block", "@10", "switch_statement", "@11", "switch_decl",
  "case_list", "case_statement", "@12", "@13", "@14", "@15",
  "default_statement", "@16", "default_decl", "default_body",
  "case_expression_list", "case_element", "select_statement", "@17",
  "select_decl", "selcase_list", "selcase_statement", "@18", "@19", "@20",
  "@21", "selcase_expression_list", "selcase_element", "give_statement",
  "try_statement", "@22", "try_decl", "catch_statements", "catch_list",
  "catch_statement", "catch_decl", "catchcase_element_list",
  "catchcase_element", "raise_statement", "func_statement", "func_decl",
  "@23", "func_decl_short", "@24", "func_begin", "param_list",
  "param_symbol", "static_block", "@25", "@26", "static_decl",
  "static_short_decl", "launch_statement", "pass_statement",
  "const_statement", "export_statement", "export_symbol_list",
  "attributes_statement", "attributes_decl", "attributes_short_decl",
  "attribute_list", "attribute_vert_list", "class_decl", "@27",
  "class_def_inner", "class_param_list", "@28", "from_clause",
  "inherit_list", "inherit_token", "inherit_call", "inherit_param_list",
  "inherit_param_token", "class_statement_list", "class_statement",
  "init_decl", "@29", "property_decl", "has_list", "has_clause_list",
  "object_decl", "@30", "object_decl_inner", "object_statement_list",
  "object_statement", "global_statement", "@31", "global_symbol_list",
  "globalized_symbol", "return_statement", "const_atom", "atomic_symbol",
  "var_atom", "variable", "expression", "range_decl", "func_call", "@32",
  "opt_eol", "eol_seq", "nameless_func", "@33", "nameless_func_decl_inner",
  "@34", "lambda_expr", "@35", "lambda_expr_inner", "iif_expr",
  "array_decl", "@36", "dict_decl", "@37", "expression_list",
  "par_expression_list", "symbol_list", "expression_pair_list", 0
};
#endif

# ifdef YYPRINT
/* YYTOKNUM[YYLEX-NUM] -- Internal token number corresponding to
   token YYLEX-NUM.  */
static const yytype_uint16 yytoknum[] =
{
       0,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   279,   280,   281,   282,   283,   284,
     285,   286,   287,   288,   289,   290,   291,   292,   293,   294,
     295,   296,   297,   298,   299,   300,   301,   302,   303,   304,
     305,   306,   307,   308,   309,   310,   311,   312,   313,   314,
     315,   316,   317,   318,   319,   320,   321,   322,   323,   324,
     325,   326,   327,   328,   329,   330,   331,   332,   333,   334,
     335,   336,   337,   338,   339,   340,   341,   342,   343,   344,
     345,   346,   347,   348,   349,   350,   351,   352,   353,   354,
     355,   356,   357,   358,   359,   360,   361
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint16 yyr1[] =
{
       0,   107,   108,   109,   109,   110,   110,   110,   111,   111,
     111,   111,   111,   111,   111,   111,   112,   112,   112,   113,
     113,   113,   113,   113,   113,   114,   114,   115,   115,   115,
     115,   115,   115,   115,   115,   115,   115,   115,   115,   115,
     115,   115,   115,   115,   115,   115,   115,   115,   115,   116,
     116,   117,   117,   117,   117,   117,   117,   118,   118,   118,
     118,   118,   118,   118,   118,   118,   118,   118,   119,   120,
     121,   122,   123,   124,   125,   126,   127,   128,   129,   131,
     130,   132,   130,   130,   133,   133,   133,   134,   134,   134,
     136,   135,   135,   137,   137,   138,   138,   139,   139,   140,
     139,   141,   141,   143,   142,   144,   144,   145,   145,   146,
     146,   147,   147,   147,   149,   148,   148,   150,   150,   150,
     151,   151,   151,   153,   152,   152,   152,   154,   154,   155,
     155,   155,   155,   156,   156,   157,   157,   157,   157,   157,
     158,   160,   159,   159,   159,   162,   161,   161,   161,   164,
     163,   163,   163,   166,   165,   167,   167,   168,   168,   168,
     169,   170,   169,   171,   169,   172,   169,   173,   169,   174,
     175,   174,   176,   176,   176,   177,   177,   178,   178,   179,
     179,   179,   179,   179,   181,   180,   182,   182,   183,   183,
     183,   184,   185,   184,   186,   184,   187,   184,   188,   184,
     189,   189,   190,   190,   190,   191,   191,   191,   192,   193,
     192,   194,   194,   195,   195,   196,   196,   197,   198,   198,
     198,   198,   198,   199,   199,   200,   200,   201,   201,   202,
     202,   203,   204,   203,   203,   205,   206,   205,   207,   208,
     208,   208,   209,   210,   211,   210,   212,   210,   213,   213,
     214,   214,   215,   215,   216,   216,   216,   216,   217,   217,
     217,   218,   218,   218,   219,   219,   220,   220,   221,   221,
     222,   222,   223,   223,   223,   224,   224,   224,   226,   225,
     227,   227,   228,   228,   229,   228,   230,   230,   231,   231,
     232,   233,   233,   234,   234,   235,   235,   235,   236,   236,
     237,   237,   237,   237,   239,   238,   240,   240,   241,   241,
     241,   242,   242,   242,   242,   244,   243,   245,   245,   246,
     246,   247,   247,   247,   247,   249,   248,   250,   250,   250,
     251,   252,   252,   252,   253,   253,   253,   253,   254,   255,
     255,   255,   256,   256,   256,   256,   256,   257,   257,   257,
     257,   257,   257,   257,   257,   257,   257,   257,   257,   257,
     257,   257,   257,   257,   257,   257,   257,   257,   257,   257,
     257,   257,   257,   257,   257,   257,   257,   257,   257,   257,
     257,   257,   257,   257,   257,   257,   257,   257,   257,   257,
     257,   257,   257,   257,   257,   257,   257,   257,   258,   258,
     258,   258,   259,   259,   260,   259,   261,   261,   262,   262,
     264,   263,   265,   266,   265,   265,   268,   267,   269,   269,
     270,   270,   271,   271,   272,   271,   273,   273,   274,   273,
     275,   275,   276,   276,   277,   277,   278,   278
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     1,     0,     2,     1,     2,     3,     1,     1,
       1,     1,     1,     1,     1,     1,     3,     3,     3,     1,
       1,     3,     3,     3,     2,     3,     5,     2,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     3,
       3,     4,     5,     4,     4,     6,     6,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     4,     4,
       4,     4,     4,     4,     4,     4,     4,     4,     4,     0,
       5,     0,     6,     2,     3,     2,     3,     3,     2,     3,
       0,     6,     2,     3,     3,     3,     3,     0,     1,     0,
       3,     2,     3,     0,     4,     3,     3,     0,     2,     2,
       3,     2,     3,     3,     0,     5,     2,     7,     9,     3,
       7,     9,     3,     0,     9,     6,     5,     0,     2,     1,
       1,     1,     1,     3,     3,     3,     2,     3,     2,     3,
       1,     0,     6,     3,     3,     0,     6,     3,     3,     0,
       6,     3,     3,     0,     6,     3,     3,     0,     2,     3,
       1,     0,     5,     0,     5,     0,     5,     0,     5,     0,
       0,     3,     0,     1,     2,     2,     2,     1,     3,     1,
       1,     1,     3,     1,     0,     6,     3,     3,     0,     2,
       3,     1,     0,     5,     0,     5,     0,     5,     0,     5,
       1,     3,     0,     1,     1,     5,     4,     3,     3,     0,
       6,     2,     3,     0,     1,     1,     2,     2,     2,     4,
       3,     5,     3,     1,     3,     1,     1,     3,     3,     5,
       2,     7,     0,     9,     3,     7,     0,     8,     2,     0,
       1,     4,     1,     0,     0,     5,     0,     3,     2,     3,
       2,     3,     3,     3,     3,     5,     5,     3,     5,     5,
       3,     2,     3,     3,     1,     3,     4,     3,     2,     3,
       2,     3,     0,     1,     3,     2,     3,     2,     0,     8,
       3,     2,     0,     5,     0,     7,     0,     2,     1,     3,
       2,     0,     5,     1,     3,     1,     1,     1,     0,     2,
       1,     1,     1,     1,     0,     7,     5,     4,     0,     3,
       3,     1,     2,     3,     4,     0,     8,     2,     2,     0,
       2,     1,     1,     1,     1,     0,     4,     1,     3,     3,
       1,     2,     3,     3,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     2,     4,     5,     3,     1,     1,     3,
       2,     3,     3,     3,     3,     3,     3,     3,     3,     3,
       3,     2,     4,     4,     3,     2,     2,     2,     2,     3,
       3,     3,     3,     3,     3,     3,     3,     3,     2,     3,
       3,     3,     3,     3,     2,     2,     2,     1,     1,     3,
       4,     5,     2,     1,     1,     1,     1,     5,     3,     4,
       4,     5,     6,     4,     0,     8,     0,     1,     1,     2,
       0,     6,     6,     0,     8,     2,     0,     5,     1,     2,
       5,     3,     2,     4,     0,     6,     3,     4,     0,     6,
       1,     3,     1,     4,     1,     3,     3,     6
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint16 yydefact[] =
{
       3,     0,     0,     1,     0,    20,   335,   336,   338,   337,
     334,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   341,   340,     0,     0,     0,     0,     0,     0,   325,
     416,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     140,   406,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     4,     5,     8,    12,    19,    28,
      29,    30,    57,    58,    59,    60,    61,    62,    63,    64,
      65,    66,    67,    31,    79,     0,    36,    90,     0,    37,
      38,    32,   114,     0,    33,    46,    47,    48,    34,   153,
      35,   184,    39,    40,   209,    41,     9,   243,     0,     0,
      44,    45,    13,    14,    15,     0,   272,    10,    11,    43,
      42,   347,   339,   342,   348,     0,   396,   388,   387,   393,
     394,   395,    24,     6,     0,     0,     0,     0,   348,     0,
       0,   109,     0,   111,     0,     0,     0,     0,   339,     0,
       0,     0,     0,     0,     0,     0,     0,   430,     0,     0,
     211,     0,     0,     0,     0,   278,     0,   315,     0,   331,
       0,     0,     0,     0,     0,     0,     0,     0,   388,     0,
       0,     0,   268,   270,     0,     0,     0,   261,   264,     0,
       0,   238,     0,     0,    20,     0,     0,    88,     0,    81,
     408,     0,   407,     0,   422,     0,   432,     0,     0,   378,
       0,     0,   136,     0,   138,     0,   386,   385,   350,   361,
     368,   366,   384,   107,    83,   107,    92,   107,   116,   157,
     188,   107,     0,   107,   244,   246,   230,     0,   406,     0,
     273,     0,   272,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   343,    27,
     406,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   367,
     365,     0,     0,   392,    50,    49,     0,     0,    86,    89,
      84,    87,   110,   113,   112,    94,    96,    93,    95,   119,
     122,     0,     0,     0,   156,   155,     7,   187,   186,   207,
       0,     0,     0,   212,   208,   228,   227,    23,     0,    22,
       0,   333,   332,   330,     0,   327,     0,   242,   418,   240,
       0,    18,    16,    17,   253,   252,   260,     0,   269,   271,
     257,   254,     0,   263,   262,     0,    21,   134,   133,   107,
     406,   409,   398,     0,   426,     0,     0,   424,   406,     0,
     428,   406,     0,     0,     0,   139,   135,   137,     0,     0,
       0,     0,     0,     0,     0,   248,   250,     0,   107,     0,
     234,     0,   277,   275,     0,     0,     0,   267,     0,     0,
     346,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   434,     0,   410,     0,   430,     0,     0,     0,     0,
     421,     0,   377,   376,   375,   374,   373,   372,   364,   369,
     370,   371,   383,   382,   381,   380,   379,   358,   357,   356,
     351,   349,   354,   353,   352,   355,   360,   359,     0,     0,
     389,     0,    25,     0,   435,     0,     0,   206,     0,   431,
       0,   406,   298,   286,     0,     0,     0,   319,   326,     0,
     419,   406,     0,     0,     0,     0,   381,   265,     0,     0,
     400,   399,     0,   436,   406,     0,   423,   406,     0,   427,
     362,   363,     0,   108,     0,     0,     0,    99,    98,   103,
       0,     0,   160,     0,     0,   158,     0,   170,     0,   191,
       0,     0,   189,     0,     0,   214,   215,   107,   249,   251,
       0,     0,   247,   236,     0,   274,   266,   276,     0,   344,
      73,    77,    78,    76,    75,    74,    72,    71,    70,    69,
       0,     0,     0,    51,    54,    53,   403,   432,     0,    68,
       0,     0,   390,     0,     0,   126,   123,     0,   205,   281,
     239,   308,     0,   318,   291,   287,   288,   317,   308,   329,
     328,     0,   417,   259,   258,   256,   255,     0,   397,   401,
       0,   433,     0,     0,    80,     0,   101,     0,     0,     0,
     107,   107,   115,   159,     0,   180,   183,   181,   179,     0,
     177,   174,     0,     0,   190,     0,   203,   204,     0,   200,
       0,     0,   218,   225,   226,     0,     0,   223,     0,   216,
       0,   229,     0,   406,   232,     0,   345,   430,     0,     0,
     406,   243,    52,   404,     0,   420,   391,    26,     0,     0,
     125,     0,   300,     0,     0,     0,     0,     0,   301,   299,
     303,   302,     0,   280,   406,   290,     0,   321,   322,   324,
     323,     0,   320,   241,    82,   425,   429,     0,   102,   106,
     105,    91,     0,     0,   165,   167,     0,   161,   163,     0,
     154,   107,     0,   171,   196,   198,   192,   194,   202,   185,
     222,     0,   220,     0,     0,   210,   245,     0,   406,     0,
      55,    56,   415,   239,   107,   406,   402,   117,   120,     0,
       0,     0,     0,   129,     0,     0,   130,   131,   132,   284,
       0,     0,   304,     0,     0,   311,     0,     0,     0,     0,
     289,     0,   437,   104,   107,     0,   182,   107,     0,   178,
       0,   176,   107,     0,   107,     0,   201,   219,   224,     0,
       0,     0,   231,   235,     0,     0,     0,     0,     0,   141,
       0,     0,   145,     0,     0,   149,     0,     0,   128,   406,
     283,     0,   243,     0,   310,   312,   309,     0,   279,   296,
     297,   406,   293,   295,   316,     0,   168,     0,   164,     0,
     199,     0,   195,   221,   237,     0,   413,     0,   411,   405,
     118,   121,   144,   107,   143,   148,   107,   147,   152,   107,
     151,   124,     0,   307,   107,     0,   313,     0,     0,     0,
     233,   406,     0,     0,     0,     0,   285,     0,   306,   314,
     294,   292,     0,   412,     0,     0,     0,     0,     0,   142,
     146,   150,   305,   414
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     1,     2,    54,    55,    56,   483,   125,    58,    59,
      60,    61,    62,    63,    64,    65,    66,    67,    68,    69,
      70,    71,    72,    73,   213,   349,    74,    75,    76,   215,
      77,    78,   486,   580,   487,   488,   581,   489,   368,    79,
      80,    81,   217,    82,    83,    84,   629,   704,   705,    85,
      86,    87,   706,   793,   707,   796,   708,   799,    88,   219,
      89,   371,   495,   727,   728,   724,   725,   496,   593,   497,
     673,   589,   590,    90,   220,    91,   372,   502,   734,   735,
     732,   733,   598,   599,    92,    93,   221,    94,   504,   505,
     506,   507,   606,   607,    95,    96,    97,   688,    98,   613,
      99,   328,   329,   223,   378,   379,   224,   225,   100,   101,
     102,   103,   179,   104,   105,   106,   231,   232,   107,   318,
     452,   453,   759,   456,   555,   556,   645,   771,   772,   551,
     639,   640,   762,   641,   642,   717,   108,   320,   457,   558,
     652,   109,   161,   324,   325,   110,   111,   112,   113,   128,
     115,   116,   117,   695,   191,   192,   406,   531,   621,   811,
     118,   162,   330,   119,   120,   474,   121,   477,   148,   197,
     140,   198
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -617
static const yytype_int16 yypact[] =
{
    -617,    25,   821,  -617,    41,  -617,  -617,  -617,  -617,  -617,
    -617,    44,    55,   421,   379,    50,   546,   475,   693,    84,
    3208,  -617,  -617,  3263,   232,  3318,   280,   297,    64,  -617,
    -617,   219,  3373,   378,   238,  3428,   511,   394,  3483,  1775,
    -617,   119,  5189,  5470,   479,   156,   195,  5470,  5470,  5470,
    5470,  5470,  5470,  5470,  -617,  -617,  -617,  -617,  -617,  -617,
    -617,  -617,  -617,  -617,  -617,  -617,  -617,  -617,  -617,  -617,
    -617,  -617,  -617,  -617,  -617,  3153,  -617,  -617,  3153,  -617,
    -617,  -617,  -617,  3153,  -617,  -617,  -617,  -617,  -617,  -617,
    -617,  -617,  -617,  -617,  -617,  -617,  -617,   131,  3153,    27,
    -617,  -617,  -617,  -617,  -617,    49,   222,  -617,  -617,  -617,
    -617,  -617,  -617,  -617,  1140,  3815,  -617,   417,  -617,  -617,
    -617,  -617,  -617,  -617,   188,    23,   128,    21,   439,  3872,
     249,  -617,   275,  -617,   299,    38,  3929,   209,   201,   414,
     202,   303,  4093,   342,   358,  4130,   362,  5983,    39,   370,
    -617,  3153,   385,  4180,   389,  -617,   412,  -617,   418,  -617,
    4217,   309,   105,   428,   432,   441,   449,  5983,   213,   450,
     382,   214,  -617,  -617,   484,  4267,   486,  -617,  -617,    56,
     495,  -617,   496,  4304,  -617,   502,   503,  -617,   506,  -617,
    -617,  5470,   505,  5320,  -617,   460,  5540,   120,   139,  6094,
     409,   513,  -617,    86,  -617,   110,   587,   587,   175,   175,
     175,   175,   175,  -617,  -617,  -617,  -617,  -617,  -617,  -617,
    -617,  -617,   369,  -617,  -617,  -617,  -617,   515,   119,   516,
    -617,   134,   308,   148,  5213,   518,  5470,  5470,  5470,  5470,
    5470,  5470,  5470,  5470,  5470,  5470,   523,  5354,  -617,  -617,
     119,  5470,  3538,  5470,  5470,  5470,  5470,  5470,  5470,  5470,
    5470,  5470,  5470,   531,  5470,  5470,  5470,  5470,  5470,  5470,
    5470,  5470,  5470,  5470,  5470,  5470,  5470,  5470,  5470,  -617,
    -617,  5247,   542,  -617,  -617,  -617,   523,  5470,  -617,  -617,
    -617,  -617,  -617,  -617,  -617,  -617,  -617,  -617,  -617,  -617,
    -617,  5470,   523,  3593,  -617,  -617,  -617,  -617,  -617,  -617,
     519,  5470,  5470,  -617,  -617,  -617,  -617,  -617,   329,  -617,
     373,  -617,  -617,  -617,   173,  -617,   552,  -617,   451,  -617,
     482,  -617,  -617,  -617,  -617,  -617,  -617,   527,  -617,  -617,
    -617,  -617,  3648,  -617,  -617,   550,  -617,  -617,  -617,  -617,
    4354,  -617,  -617,  5748,  -617,  5378,  5470,  -617,   119,   509,
    -617,   119,   517,  5470,  5470,  -617,  -617,  -617,  1881,  1457,
    1987,   395,   433,  1563,   220,  -617,  -617,  2093,  -617,  3153,
    -617,    33,  -617,  -617,   562,   555,   174,  -617,  5470,  5597,
    -617,  4391,  4441,  4478,  4528,  4565,  4615,  4652,  4702,  4739,
    4789,  -617,     5,  -617,  5504,  4826,   566,   183,  5412,  4876,
    -617,  5711,  6057,  6094,   899,   899,   899,   899,   899,   899,
     899,   899,  -617,   587,   587,   587,   587,   914,   914,   995,
     312,   312,   350,   350,   350,   190,   175,   175,  5470,  5654,
    -617,   488,  5983,  5785,  -617,   571,  3986,  -617,  4913,  5983,
     572,   119,  -617,   543,   575,   573,   577,  -617,  -617,   461,
    -617,   119,  5470,   578,   579,   581,   785,  -617,  2199,   534,
    -617,  -617,  5822,  5983,   119,  5470,  -617,   119,  5470,  -617,
     899,   899,   583,  -617,   402,  3703,   580,  -617,  -617,  -617,
     584,   585,  -617,   535,   388,  -617,   582,  -617,   589,  -617,
      42,   588,  -617,    29,   592,   563,  -617,  -617,  -617,  -617,
     591,  2305,  -617,  -617,   144,  -617,  -617,  -617,  5859,  -617,
    -617,  -617,  -617,  -617,  -617,  -617,  -617,  -617,  -617,  -617,
    5470,   205,  3709,  -617,  -617,  -617,  -617,  5983,   164,  -617,
    5470,  5896,  -617,  5470,  5470,  -617,  -617,  3153,  -617,  -617,
     590,    11,   600,  -617,   554,   537,  -617,  -617,    16,  -617,
    -617,   590,  5983,  -617,  -617,  -617,  -617,   601,  -617,  -617,
     556,  5983,   558,  5946,  -617,   610,  -617,   612,  4963,   613,
    -617,  -617,  -617,  -617,   221,   548,  -617,  -617,  -617,   149,
    -617,  -617,   615,   237,  -617,   248,  -617,  -617,   168,  -617,
     616,   620,  -617,  -617,  -617,   523,     8,  -617,   621,  -617,
    1669,  -617,   626,   119,  -617,   586,  -617,  5000,   185,   627,
     119,   131,  -617,  -617,   593,  6020,  -617,  5983,  3765,   927,
    -617,   178,  -617,   549,   628,   632,   636,    12,  -617,  -617,
    -617,  -617,   624,  -617,   119,  -617,   573,  -617,  -617,  -617,
    -617,   634,  -617,  -617,  -617,  -617,  -617,  5470,  -617,  -617,
    -617,  -617,  2411,  1457,  -617,  -617,   641,  -617,  -617,   557,
    -617,  -617,  3153,  -617,  -617,  -617,  -617,  -617,   455,  -617,
    -617,   643,  -617,   476,   523,  -617,  -617,   596,   119,   260,
    -617,  -617,  -617,   590,  -617,   119,  -617,  -617,  -617,  5470,
     398,   429,   438,  -617,   640,   927,  -617,  -617,  -617,  -617,
     602,  5470,  -617,   574,   651,  -617,   649,   194,   653,   304,
    -617,   655,  5983,  -617,  -617,  3153,  -617,  -617,  3153,  -617,
    2517,  -617,  -617,  3153,  -617,  3153,  -617,  -617,  -617,   656,
     617,   614,  -617,  -617,   186,  2623,   618,  4043,   660,  -617,
    3153,   661,  -617,  3153,   663,  -617,  3153,   664,  -617,   119,
    -617,  5050,   131,  5470,  -617,  -617,  -617,     6,  -617,  -617,
    -617,   204,  -617,  -617,  -617,  1033,  -617,  1139,  -617,  1245,
    -617,  1351,  -617,  -617,  -617,   665,  -617,   622,  -617,  -617,
    -617,  -617,  -617,  -617,  -617,  -617,  -617,  -617,  -617,  -617,
    -617,  -617,   625,  -617,  -617,  5087,  -617,   666,   304,   638,
    -617,   119,   672,  2729,  2835,  2941,  -617,  3047,  -617,  -617,
    -617,  -617,   642,  -617,   674,   692,   699,   703,   704,  -617,
    -617,  -617,  -617,  -617
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -617,  -617,  -617,  -617,  -617,  -617,    -1,  -617,  -617,  -617,
    -617,  -617,  -617,  -617,  -617,  -617,  -617,  -617,  -617,  -617,
    -617,  -617,  -617,  -617,  -617,  -617,  -617,  -617,  -617,  -617,
    -617,  -617,    45,  -617,  -617,  -617,  -617,  -617,  -194,  -617,
    -617,  -617,  -617,  -617,  -617,  -617,  -617,    -9,  -617,  -617,
    -617,  -617,  -617,  -617,  -617,  -617,  -617,  -617,  -617,  -617,
    -617,  -617,  -617,  -617,  -617,  -617,  -617,   337,  -617,  -617,
    -617,  -617,    43,  -617,  -617,  -617,  -617,  -617,  -617,  -617,
    -617,  -617,  -617,    32,  -617,  -617,  -617,  -617,  -617,  -617,
     206,  -617,  -617,    30,  -617,  -410,  -617,  -617,  -617,  -617,
    -617,  -365,   153,  -616,  -617,  -617,  -617,  -617,  -617,  -617,
    -617,  -617,  -617,  -617,  -617,  -617,   -96,  -617,  -617,  -617,
    -617,  -617,  -617,   262,  -617,    70,  -617,  -617,   -90,  -617,
    -617,   163,  -617,   167,   171,  -617,  -617,  -617,  -617,  -617,
    -617,  -617,  -617,  -617,   263,  -617,  -329,   -10,  -617,    -2,
      82,  -108,   705,  -617,   -54,  -617,  -617,  -617,  -617,  -617,
    -617,  -617,  -617,  -617,  -617,  -617,  -617,  -617,   -42,   327,
     490,  -617
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -435
static const yytype_int16 yytable[] =
{
     114,    57,   126,   203,   205,   694,   248,   138,   464,   283,
     233,   682,   806,   714,   632,   139,   514,   633,   715,   647,
     248,   369,   633,   370,   288,     3,   285,   373,   227,   377,
     601,   248,   602,   603,   513,   604,  -239,   114,   189,   327,
     310,   295,   200,   595,   122,  -202,   596,   123,   597,   634,
     229,   132,  -272,   133,   634,   230,   124,   635,   636,   344,
     283,     8,   635,   636,   134,   158,   289,   159,     6,     7,
       8,     9,    10,   114,   214,   302,   114,   216,   683,   228,
     807,   114,   218,   296,  -239,   143,   716,  -202,   530,   366,
      21,    22,   248,   286,   684,   129,   114,   226,   136,   637,
     142,    30,   145,  -239,   637,   147,   326,   153,   311,   312,
     160,   327,  -202,   367,   167,   605,    41,   175,    42,  -272,
     183,   357,   190,   190,   196,   199,   345,   147,   147,   206,
     207,   208,   209,   210,   211,   212,   386,   383,    43,    44,
     360,   638,   190,   359,   362,   614,   804,   190,   648,   114,
     314,   387,   667,    47,    48,   468,   312,   201,    49,   202,
       6,     7,     8,     9,    10,   623,    50,   190,    51,    52,
      53,   676,  -239,  -406,   381,  -239,   458,   517,   222,   709,
     312,   190,    21,    22,   511,   631,   535,   786,   691,   190,
     358,   284,  -406,    30,   668,  -406,   408,   766,   204,     6,
       7,     8,     9,    10,   384,   407,   619,   190,    41,   361,
      42,   287,   299,   677,   461,  -406,   335,   338,   384,   669,
     163,    21,    22,   508,   664,   164,   165,   250,   230,  -406,
      43,    44,    30,   149,   358,   150,   401,  -406,   678,   171,
     671,   172,   250,   459,   384,    47,    48,    41,   461,    42,
      49,   674,   292,   312,   300,   312,   461,   620,    50,   339,
      51,    52,    53,   742,   767,   509,   665,   281,   282,    43,
      44,  -434,   302,   350,   808,   353,   441,   151,   293,   279,
     280,   154,   672,   173,    47,    48,   155,  -434,   303,    49,
     277,   278,   444,   675,   279,   280,   469,    50,   156,    51,
      52,    53,   294,   157,   475,   743,   304,   478,     6,     7,
     769,     9,    10,   610,   230,   323,   389,   385,   391,   392,
     393,   394,   395,   396,   397,   398,   399,   400,   744,   405,
     450,   770,  -282,   409,   411,   412,   413,   414,   415,   416,
     417,   418,   419,   420,   421,   306,   423,   424,   425,   426,
     427,   428,   429,   430,   431,   432,   433,   434,   435,   436,
     437,   307,  -282,   439,   250,   309,   114,   114,   114,   442,
     374,   114,   375,   313,   454,   114,  -286,   114,   512,   169,
     130,   451,   131,   443,   170,   446,   662,   663,   315,   591,
     773,  -173,   317,   448,   449,   180,   491,   550,   492,   748,
     181,   749,   250,   575,  -169,   576,   455,   561,   273,   274,
     275,   276,   277,   278,   376,   319,   279,   280,   493,   494,
     570,   321,   127,   572,   466,     6,     7,     8,     9,    10,
     751,   331,   752,  -173,   498,   332,   499,   472,   473,   754,
    -172,   755,  -169,   750,   333,   480,   481,    21,    22,   276,
     277,   278,   334,   336,   279,   280,   500,   494,    30,   596,
     615,   597,   559,   234,   235,   337,   114,   323,   234,   235,
     518,   281,   282,    41,   753,    42,   137,   730,  -172,   773,
     603,     8,   604,   756,   624,     8,   212,   340,   618,   343,
     537,   363,   364,   234,   235,    43,    44,   301,   346,   347,
     745,    21,    22,   154,   156,    21,    22,   180,   351,   114,
      47,    48,   176,   354,   177,    49,   365,   178,   380,   382,
     541,   461,   447,    50,   390,    51,    52,    53,   463,     8,
     775,     6,     7,   777,     9,    10,   584,   422,   779,   585,
     781,   586,   587,   588,   562,   114,   630,   135,   440,   462,
       6,     7,     8,     9,    10,   460,   467,   571,   516,   687,
     573,   585,   476,   586,   587,   588,   693,   578,   515,   534,
     479,   543,    21,    22,   545,   549,   455,   710,   553,   554,
     557,   563,   564,    30,   565,   568,   574,   582,   583,   579,
     719,   592,   594,   503,   611,   681,   327,   600,    41,   813,
      42,   608,   814,   643,   654,   815,   644,   646,   114,   655,
     817,   656,   617,   658,   212,   659,   661,   666,   670,   679,
      43,    44,   625,   680,   685,   627,   628,   114,   703,   686,
     692,   712,   711,   718,   741,    47,    48,   689,   181,   250,
      49,   746,   713,   721,   696,   726,   737,   740,    50,   757,
      51,    52,    53,   760,   764,   765,   768,   763,   774,   783,
     114,   114,   784,   792,   795,   785,   798,   801,   810,   789,
     114,   731,   819,   812,   739,   823,   816,   829,   268,   269,
     270,   271,   272,   273,   274,   275,   276,   277,   278,   821,
     787,   279,   280,   828,   141,   830,   758,     6,     7,     8,
       9,    10,   831,   114,   703,   802,   832,   833,   723,   501,
     736,   609,   729,   738,   653,   552,   720,   809,   820,    21,
      22,   649,   560,   114,   776,   650,   114,   778,   114,   651,
      30,   114,   780,   114,   782,   538,   402,   168,     0,   722,
       0,     0,     0,   114,     0,    41,     0,    42,   114,   794,
       0,   114,   797,     0,   114,   800,     0,   822,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    43,    44,     0,
       0,     0,     0,   114,     0,   114,     0,   114,     0,   114,
       0,   747,    47,    48,     0,     0,     0,    49,   566,     0,
       0,     0,     0,   761,     0,    50,     0,    51,    52,    53,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   114,   114,   114,     0,   114,     0,     0,     0,     0,
       0,    -2,     4,     0,     5,     6,     7,     8,     9,    10,
      11,    12,    13,    14,    15,     0,    16,   250,     0,    17,
       0,     0,     0,    18,    19,   805,    20,    21,    22,    23,
      24,     0,    25,    26,     0,    27,    28,    29,    30,     0,
      31,    32,    33,    34,    35,    36,     0,    37,     0,    38,
      39,    40,     0,    41,     0,    42,   268,   269,   270,   271,
     272,   273,   274,   275,   276,   277,   278,     0,     0,   279,
     280,     0,     0,     0,     0,    43,    44,     0,     0,    45,
      46,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      47,    48,     0,     0,     0,    49,     0,     0,     0,     0,
       0,     0,     0,    50,     0,    51,    52,    53,     4,     0,
       5,     6,     7,     8,     9,    10,  -127,    12,    13,    14,
      15,     0,    16,     0,     0,    17,   700,   701,   702,    18,
       0,   250,    20,    21,    22,    23,    24,     0,    25,   185,
       0,   186,    28,    29,    30,     0,   250,    32,     0,     0,
      35,     0,     0,   188,     0,    38,    39,    40,     0,    41,
       0,    42,     0,   263,   264,   265,   266,   267,     0,     0,
     268,   269,   270,   271,   272,   273,   274,   275,   276,   277,
     278,    43,    44,   279,   280,    45,    46,   270,   271,   272,
     273,   274,   275,   276,   277,   278,    47,    48,   279,   280,
       0,    49,     0,     0,     0,     0,     0,     0,     0,    50,
       0,    51,    52,    53,     4,     0,     5,     6,     7,     8,
       9,    10,  -166,    12,    13,    14,    15,   250,    16,     0,
       0,    17,     0,     0,     0,    18,  -166,  -166,    20,    21,
      22,    23,    24,     0,    25,   185,     0,   186,    28,    29,
      30,     0,     0,    32,     0,     0,    35,     0,  -166,   188,
       0,    38,    39,    40,     0,    41,     0,    42,     0,   271,
     272,   273,   274,   275,   276,   277,   278,     0,     0,   279,
     280,     0,     0,     0,     0,     0,     0,    43,    44,     0,
       0,    45,    46,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    47,    48,     0,     0,     0,    49,     0,     0,
       0,     0,     0,     0,     0,    50,     0,    51,    52,    53,
       4,     0,     5,     6,     7,     8,     9,    10,  -162,    12,
      13,    14,    15,     0,    16,     0,     0,    17,     0,     0,
       0,    18,  -162,  -162,    20,    21,    22,    23,    24,     0,
      25,   185,     0,   186,    28,    29,    30,     0,     0,    32,
       0,     0,    35,     0,  -162,   188,     0,    38,    39,    40,
       0,    41,     0,    42,   234,   235,   236,   237,   238,   239,
     240,   241,   242,   243,   244,   245,     0,     0,     0,     0,
     246,     0,     0,    43,    44,     0,     0,    45,    46,     0,
       0,     0,     0,   247,     0,     0,     0,     0,    47,    48,
       0,     0,     0,    49,     0,     0,     0,     0,     0,     0,
       0,    50,     0,    51,    52,    53,     4,     0,     5,     6,
       7,     8,     9,    10,  -197,    12,    13,    14,    15,     0,
      16,     0,     0,    17,     0,     0,     0,    18,  -197,  -197,
      20,    21,    22,    23,    24,     0,    25,   185,     0,   186,
      28,    29,    30,     0,     0,    32,     0,     0,    35,     0,
    -197,   188,     0,    38,    39,    40,     0,    41,     0,    42,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    43,
      44,     0,     0,    45,    46,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    47,    48,     0,     0,     0,    49,
       0,     0,     0,     0,     0,     0,     0,    50,     0,    51,
      52,    53,     4,     0,     5,     6,     7,     8,     9,    10,
    -193,    12,    13,    14,    15,     0,    16,     0,     0,    17,
       0,     0,     0,    18,  -193,  -193,    20,    21,    22,    23,
      24,     0,    25,   185,     0,   186,    28,    29,    30,     0,
       0,    32,     0,     0,    35,     0,  -193,   188,     0,    38,
      39,    40,     0,    41,     0,    42,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    43,    44,     0,     0,    45,
      46,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      47,    48,     0,     0,     0,    49,     0,     0,     0,     0,
       0,     0,     0,    50,     0,    51,    52,    53,     4,     0,
       5,     6,     7,     8,     9,    10,   -97,    12,    13,    14,
      15,     0,    16,   484,   485,    17,     0,     0,     0,    18,
       0,     0,    20,    21,    22,    23,    24,     0,    25,   185,
       0,   186,    28,    29,    30,     0,     0,    32,     0,     0,
      35,     0,     0,   188,     0,    38,    39,    40,     0,    41,
       0,    42,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    43,    44,     0,     0,    45,    46,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    47,    48,     0,     0,
       0,    49,     0,     0,     0,     0,     0,     0,     0,    50,
       0,    51,    52,    53,     4,     0,     5,     6,     7,     8,
       9,    10,  -213,    12,    13,    14,    15,     0,    16,     0,
       0,    17,     0,     0,     0,    18,     0,     0,    20,    21,
      22,    23,    24,   503,    25,   185,     0,   186,    28,    29,
      30,     0,     0,    32,     0,     0,    35,     0,     0,   188,
       0,    38,    39,    40,     0,    41,     0,    42,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    43,    44,     0,
       0,    45,    46,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    47,    48,     0,     0,     0,    49,     0,     0,
       0,     0,     0,     0,     0,    50,     0,    51,    52,    53,
       4,     0,     5,     6,     7,     8,     9,    10,  -217,    12,
      13,    14,    15,     0,    16,     0,     0,    17,     0,     0,
       0,    18,     0,     0,    20,    21,    22,    23,    24,  -217,
      25,   185,     0,   186,    28,    29,    30,     0,     0,    32,
       0,     0,    35,     0,     0,   188,     0,    38,    39,    40,
       0,    41,     0,    42,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    43,    44,     0,     0,    45,    46,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    47,    48,
       0,     0,     0,    49,     0,     0,     0,     0,     0,     0,
       0,    50,     0,    51,    52,    53,     4,     0,   184,     6,
       7,     8,     9,    10,     0,    12,    13,    14,    15,     0,
      16,     0,     0,    17,     0,     0,     0,    18,     0,     0,
      20,    21,    22,    23,    24,     0,    25,   185,     0,   186,
      28,    29,    30,     0,     0,    32,     0,     0,    35,     0,
     187,   188,     0,    38,    39,    40,     0,    41,     0,    42,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    43,
      44,     0,     0,    45,    46,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    47,    48,     0,     0,     0,    49,
       0,     0,     0,     0,     0,     0,     0,    50,     0,    51,
      52,    53,     4,     0,     5,     6,     7,     8,     9,    10,
     482,    12,    13,    14,    15,     0,    16,     0,     0,    17,
       0,     0,     0,    18,     0,     0,    20,    21,    22,    23,
      24,     0,    25,   185,     0,   186,    28,    29,    30,     0,
       0,    32,     0,     0,    35,     0,     0,   188,     0,    38,
      39,    40,     0,    41,     0,    42,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    43,    44,     0,     0,    45,
      46,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      47,    48,     0,     0,     0,    49,     0,     0,     0,     0,
       0,     0,     0,    50,     0,    51,    52,    53,     4,     0,
       5,     6,     7,     8,     9,    10,   490,    12,    13,    14,
      15,     0,    16,     0,     0,    17,     0,     0,     0,    18,
       0,     0,    20,    21,    22,    23,    24,     0,    25,   185,
       0,   186,    28,    29,    30,     0,     0,    32,     0,     0,
      35,     0,     0,   188,     0,    38,    39,    40,     0,    41,
       0,    42,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    43,    44,     0,     0,    45,    46,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    47,    48,     0,     0,
       0,    49,     0,     0,     0,     0,     0,     0,     0,    50,
       0,    51,    52,    53,     4,     0,     5,     6,     7,     8,
       9,    10,   510,    12,    13,    14,    15,     0,    16,     0,
       0,    17,     0,     0,     0,    18,     0,     0,    20,    21,
      22,    23,    24,     0,    25,   185,     0,   186,    28,    29,
      30,     0,     0,    32,     0,     0,    35,     0,     0,   188,
       0,    38,    39,    40,     0,    41,     0,    42,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    43,    44,     0,
       0,    45,    46,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    47,    48,     0,     0,     0,    49,     0,     0,
       0,     0,     0,     0,     0,    50,     0,    51,    52,    53,
       4,     0,     5,     6,     7,     8,     9,    10,   567,    12,
      13,    14,    15,     0,    16,     0,     0,    17,     0,     0,
       0,    18,     0,     0,    20,    21,    22,    23,    24,     0,
      25,   185,     0,   186,    28,    29,    30,     0,     0,    32,
       0,     0,    35,     0,     0,   188,     0,    38,    39,    40,
       0,    41,     0,    42,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    43,    44,     0,     0,    45,    46,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    47,    48,
       0,     0,     0,    49,     0,     0,     0,     0,     0,     0,
       0,    50,     0,    51,    52,    53,     4,     0,     5,     6,
       7,     8,     9,    10,   612,    12,    13,    14,    15,     0,
      16,     0,     0,    17,     0,     0,     0,    18,     0,     0,
      20,    21,    22,    23,    24,     0,    25,   185,     0,   186,
      28,    29,    30,     0,     0,    32,     0,     0,    35,     0,
       0,   188,     0,    38,    39,    40,     0,    41,     0,    42,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    43,
      44,     0,     0,    45,    46,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    47,    48,     0,     0,     0,    49,
       0,     0,     0,     0,     0,     0,     0,    50,     0,    51,
      52,    53,     4,     0,     5,     6,     7,     8,     9,    10,
    -100,    12,    13,    14,    15,     0,    16,     0,     0,    17,
       0,     0,     0,    18,     0,     0,    20,    21,    22,    23,
      24,     0,    25,   185,     0,   186,    28,    29,    30,     0,
       0,    32,     0,     0,    35,     0,     0,   188,     0,    38,
      39,    40,     0,    41,     0,    42,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    43,    44,     0,     0,    45,
      46,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      47,    48,     0,     0,     0,    49,     0,     0,     0,     0,
       0,     0,     0,    50,     0,    51,    52,    53,     4,     0,
       5,     6,     7,     8,     9,    10,  -175,    12,    13,    14,
      15,     0,    16,     0,     0,    17,     0,     0,     0,    18,
       0,     0,    20,    21,    22,    23,    24,     0,    25,   185,
       0,   186,    28,    29,    30,     0,     0,    32,     0,     0,
      35,     0,     0,   188,     0,    38,    39,    40,     0,    41,
       0,    42,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    43,    44,     0,     0,    45,    46,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    47,    48,     0,     0,
       0,    49,     0,     0,     0,     0,     0,     0,     0,    50,
       0,    51,    52,    53,     4,     0,     5,     6,     7,     8,
       9,    10,   788,    12,    13,    14,    15,     0,    16,     0,
       0,    17,     0,     0,     0,    18,     0,     0,    20,    21,
      22,    23,    24,     0,    25,   185,     0,   186,    28,    29,
      30,     0,     0,    32,     0,     0,    35,     0,     0,   188,
       0,    38,    39,    40,     0,    41,     0,    42,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    43,    44,     0,
       0,    45,    46,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    47,    48,     0,     0,     0,    49,     0,     0,
       0,     0,     0,     0,     0,    50,     0,    51,    52,    53,
       4,     0,     5,     6,     7,     8,     9,    10,   824,    12,
      13,    14,    15,     0,    16,     0,     0,    17,     0,     0,
       0,    18,     0,     0,    20,    21,    22,    23,    24,     0,
      25,   185,     0,   186,    28,    29,    30,     0,     0,    32,
       0,     0,    35,     0,     0,   188,     0,    38,    39,    40,
       0,    41,     0,    42,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    43,    44,     0,     0,    45,    46,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    47,    48,
       0,     0,     0,    49,     0,     0,     0,     0,     0,     0,
       0,    50,     0,    51,    52,    53,     4,     0,     5,     6,
       7,     8,     9,    10,   825,    12,    13,    14,    15,     0,
      16,     0,     0,    17,     0,     0,     0,    18,     0,     0,
      20,    21,    22,    23,    24,     0,    25,   185,     0,   186,
      28,    29,    30,     0,     0,    32,     0,     0,    35,     0,
       0,   188,     0,    38,    39,    40,     0,    41,     0,    42,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    43,
      44,     0,     0,    45,    46,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    47,    48,     0,     0,     0,    49,
       0,     0,     0,     0,     0,     0,     0,    50,     0,    51,
      52,    53,     4,     0,     5,     6,     7,     8,     9,    10,
     826,    12,    13,    14,    15,     0,    16,     0,     0,    17,
       0,     0,     0,    18,     0,     0,    20,    21,    22,    23,
      24,     0,    25,   185,     0,   186,    28,    29,    30,     0,
       0,    32,     0,     0,    35,     0,     0,   188,     0,    38,
      39,    40,     0,    41,     0,    42,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    43,    44,     0,     0,    45,
      46,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      47,    48,     0,     0,     0,    49,     0,     0,     0,     0,
       0,     0,     0,    50,     0,    51,    52,    53,     4,     0,
       5,     6,     7,     8,     9,    10,   827,    12,    13,    14,
      15,     0,    16,     0,     0,    17,     0,     0,     0,    18,
       0,     0,    20,    21,    22,    23,    24,     0,    25,   185,
       0,   186,    28,    29,    30,     0,     0,    32,     0,     0,
      35,     0,     0,   188,     0,    38,    39,    40,     0,    41,
       0,    42,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    43,    44,     0,     0,    45,    46,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    47,    48,     0,     0,
       0,    49,     0,     0,     0,     0,     0,     0,     0,    50,
       0,    51,    52,    53,     4,     0,     5,     6,     7,     8,
       9,    10,     0,    12,    13,    14,    15,     0,    16,     0,
       0,    17,     0,     0,     0,    18,     0,     0,    20,    21,
      22,    23,    24,     0,    25,   185,     0,   186,    28,    29,
      30,     0,     0,    32,     0,     0,    35,     0,     0,   188,
       0,    38,    39,    40,     0,    41,     0,    42,     0,   144,
       0,     0,     6,     7,     8,     9,    10,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    43,    44,     0,
       0,    45,    46,     0,    21,    22,     0,     0,     0,     0,
       0,     0,    47,    48,     0,    30,     0,    49,     0,     0,
       0,     0,     0,     0,     0,    50,     0,    51,    52,    53,
      41,     0,    42,     0,   146,     0,     0,     6,     7,     8,
       9,    10,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    43,    44,     0,     0,     0,     0,     0,    21,
      22,     0,     0,     0,     0,     0,     0,    47,    48,     0,
      30,     0,    49,     0,     0,     0,     0,     0,     0,     0,
      50,     0,    51,    52,    53,    41,     0,    42,     0,   152,
       0,     0,     6,     7,     8,     9,    10,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    43,    44,     0,
       0,     0,     0,     0,    21,    22,     0,     0,     0,     0,
       0,     0,    47,    48,     0,    30,     0,    49,     0,     0,
       0,     0,     0,     0,     0,    50,     0,    51,    52,    53,
      41,     0,    42,     0,   166,     0,     0,     6,     7,     8,
       9,    10,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    43,    44,     0,     0,     0,     0,     0,    21,
      22,     0,     0,     0,     0,     0,     0,    47,    48,     0,
      30,     0,    49,     0,     0,     0,     0,     0,     0,     0,
      50,     0,    51,    52,    53,    41,     0,    42,     0,   174,
       0,     0,     6,     7,     8,     9,    10,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    43,    44,     0,
       0,     0,     0,     0,    21,    22,     0,     0,     0,     0,
       0,     0,    47,    48,     0,    30,     0,    49,     0,     0,
       0,     0,     0,     0,     0,    50,     0,    51,    52,    53,
      41,     0,    42,     0,   182,     0,     0,     6,     7,     8,
       9,    10,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    43,    44,     0,     0,     0,     0,     0,    21,
      22,     0,     0,     0,     0,     0,     0,    47,    48,     0,
      30,     0,    49,     0,     0,     0,     0,     0,     0,     0,
      50,     0,    51,    52,    53,    41,     0,    42,     0,   410,
       0,     0,     6,     7,     8,     9,    10,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    43,    44,     0,
       0,     0,     0,     0,    21,    22,     0,     0,     0,     0,
       0,     0,    47,    48,     0,    30,     0,    49,     0,     0,
       0,     0,     0,     0,     0,    50,     0,    51,    52,    53,
      41,     0,    42,     0,   445,     0,     0,     6,     7,     8,
       9,    10,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    43,    44,     0,     0,     0,     0,     0,    21,
      22,     0,     0,     0,     0,     0,     0,    47,    48,     0,
      30,     0,    49,     0,     0,     0,     0,     0,     0,     0,
      50,     0,    51,    52,    53,    41,     0,    42,     0,   465,
       0,     0,     6,     7,     8,     9,    10,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    43,    44,     0,
       0,     0,     0,     0,    21,    22,     0,     0,     0,     0,
       0,     0,    47,    48,     0,    30,     0,    49,     0,     0,
       0,     0,     0,     0,     0,    50,     0,    51,    52,    53,
      41,     0,    42,     0,   577,     0,     0,     6,     7,     8,
       9,    10,   622,     6,     7,     8,     9,    10,     0,     0,
       0,     0,    43,    44,     0,     0,     0,     0,     0,    21,
      22,     0,     0,     0,     0,    21,    22,    47,    48,     0,
      30,     0,    49,     0,     0,     0,    30,     0,     0,     0,
      50,     0,    51,    52,    53,    41,     0,    42,     0,     0,
       0,    41,     0,    42,     0,     0,     0,     0,   697,     0,
       0,     0,     0,     0,     0,     0,     0,    43,    44,     0,
       0,     0,     0,    43,    44,     0,     0,     0,     0,     0,
       0,     0,    47,    48,     0,     0,     0,    49,    47,    48,
       0,     0,     0,    49,     0,    50,     0,    51,    52,    53,
     698,    50,     0,    51,    52,    53,     0,   250,   249,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   699,     0,     0,   252,   253,   254,     0,
       0,   255,   256,   257,   258,   259,   260,   261,   262,   263,
     264,   265,   266,   267,     0,     0,   268,   269,   270,   271,
     272,   273,   274,   275,   276,   277,   278,   250,     0,   279,
     280,     0,     0,     0,     0,   290,     0,     0,     0,     0,
       0,   251,     0,     0,     0,     0,   252,   253,   254,     0,
       0,   255,   256,   257,   258,   259,   260,   261,   262,   263,
     264,   265,   266,   267,     0,     0,   268,   269,   270,   271,
     272,   273,   274,   275,   276,   277,   278,   291,     0,   279,
     280,     0,     0,     0,   250,     0,     0,     0,     0,     0,
       0,     0,   297,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   252,   253,   254,     0,     0,   255,   256,
     257,   258,   259,   260,   261,   262,   263,   264,   265,   266,
     267,     0,     0,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   298,     0,   279,   280,     0,     0,
       0,   250,     0,     0,     0,     0,     0,     0,     0,   546,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     252,   253,   254,     0,     0,   255,   256,   257,   258,   259,
     260,   261,   262,   263,   264,   265,   266,   267,     0,     0,
     268,   269,   270,   271,   272,   273,   274,   275,   276,   277,
     278,   547,     0,   279,   280,     0,     0,     0,   250,     0,
       0,     0,     0,     0,     0,     0,   790,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,   252,   253,   254,
       0,     0,   255,   256,   257,   258,   259,   260,   261,   262,
     263,   264,   265,   266,   267,     0,     0,   268,   269,   270,
     271,   272,   273,   274,   275,   276,   277,   278,   791,     0,
     279,   280,     0,     0,     0,   250,   305,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   252,   253,   254,     0,     0,   255,
     256,   257,   258,   259,   260,   261,   262,   263,   264,   265,
     266,   267,     0,   308,   268,   269,   270,   271,   272,   273,
     274,   275,   276,   277,   278,   250,     0,   279,   280,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   252,   253,   254,     0,     0,   255,
     256,   257,   258,   259,   260,   261,   262,   263,   264,   265,
     266,   267,   250,   316,   268,   269,   270,   271,   272,   273,
     274,   275,   276,   277,   278,     0,     0,   279,   280,     0,
       0,   252,   253,   254,     0,     0,   255,   256,   257,   258,
     259,   260,   261,   262,   263,   264,   265,   266,   267,     0,
     322,   268,   269,   270,   271,   272,   273,   274,   275,   276,
     277,   278,   250,     0,   279,   280,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   252,   253,   254,     0,     0,   255,   256,   257,   258,
     259,   260,   261,   262,   263,   264,   265,   266,   267,   250,
     341,   268,   269,   270,   271,   272,   273,   274,   275,   276,
     277,   278,     0,     0,   279,   280,     0,     0,   252,   253,
     254,     0,     0,   255,   256,   257,   258,   259,   260,   261,
     262,   263,   264,   265,   266,   267,     0,   348,   268,   269,
     270,   271,   272,   273,   274,   275,   276,   277,   278,   250,
       0,   279,   280,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   252,   253,
     254,     0,     0,   255,   256,   257,   258,   259,   260,   261,
     262,   263,   264,   342,   266,   267,   250,   190,   268,   269,
     270,   271,   272,   273,   274,   275,   276,   277,   278,     0,
       0,   279,   280,     0,     0,   252,   253,   254,     0,     0,
     255,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,     0,   520,   268,   269,   270,   271,   272,
     273,   274,   275,   276,   277,   278,   250,     0,   279,   280,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   252,   253,   254,     0,     0,
     255,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,   250,   521,   268,   269,   270,   271,   272,
     273,   274,   275,   276,   277,   278,     0,     0,   279,   280,
       0,     0,   252,   253,   254,     0,     0,   255,   256,   257,
     258,   259,   260,   261,   262,   263,   264,   265,   266,   267,
       0,   522,   268,   269,   270,   271,   272,   273,   274,   275,
     276,   277,   278,   250,     0,   279,   280,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   252,   253,   254,     0,     0,   255,   256,   257,
     258,   259,   260,   261,   262,   263,   264,   265,   266,   267,
     250,   523,   268,   269,   270,   271,   272,   273,   274,   275,
     276,   277,   278,     0,     0,   279,   280,     0,     0,   252,
     253,   254,     0,     0,   255,   256,   257,   258,   259,   260,
     261,   262,   263,   264,   265,   266,   267,     0,   524,   268,
     269,   270,   271,   272,   273,   274,   275,   276,   277,   278,
     250,     0,   279,   280,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   252,
     253,   254,     0,     0,   255,   256,   257,   258,   259,   260,
     261,   262,   263,   264,   265,   266,   267,   250,   525,   268,
     269,   270,   271,   272,   273,   274,   275,   276,   277,   278,
       0,     0,   279,   280,     0,     0,   252,   253,   254,     0,
       0,   255,   256,   257,   258,   259,   260,   261,   262,   263,
     264,   265,   266,   267,     0,   526,   268,   269,   270,   271,
     272,   273,   274,   275,   276,   277,   278,   250,     0,   279,
     280,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   252,   253,   254,     0,
       0,   255,   256,   257,   258,   259,   260,   261,   262,   263,
     264,   265,   266,   267,   250,   527,   268,   269,   270,   271,
     272,   273,   274,   275,   276,   277,   278,     0,     0,   279,
     280,     0,     0,   252,   253,   254,     0,     0,   255,   256,
     257,   258,   259,   260,   261,   262,   263,   264,   265,   266,
     267,     0,   528,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   250,     0,   279,   280,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   252,   253,   254,     0,     0,   255,   256,
     257,   258,   259,   260,   261,   262,   263,   264,   265,   266,
     267,   250,   529,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,     0,     0,   279,   280,     0,     0,
     252,   253,   254,     0,     0,   255,   256,   257,   258,   259,
     260,   261,   262,   263,   264,   265,   266,   267,     0,   533,
     268,   269,   270,   271,   272,   273,   274,   275,   276,   277,
     278,   250,     0,   279,   280,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     252,   253,   254,     0,     0,   255,   256,   257,   258,   259,
     260,   261,   262,   263,   264,   265,   266,   267,   250,   539,
     268,   269,   270,   271,   272,   273,   274,   275,   276,   277,
     278,     0,     0,   279,   280,     0,     0,   252,   253,   254,
       0,     0,   255,   256,   257,   258,   259,   260,   261,   262,
     263,   264,   265,   266,   267,     0,   548,   268,   269,   270,
     271,   272,   273,   274,   275,   276,   277,   278,   250,     0,
     279,   280,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,   252,   253,   254,
       0,     0,   255,   256,   257,   258,   259,   260,   261,   262,
     263,   264,   265,   266,   267,   250,   660,   268,   269,   270,
     271,   272,   273,   274,   275,   276,   277,   278,     0,     0,
     279,   280,     0,     0,   252,   253,   254,     0,     0,   255,
     256,   257,   258,   259,   260,   261,   262,   263,   264,   265,
     266,   267,     0,   690,   268,   269,   270,   271,   272,   273,
     274,   275,   276,   277,   278,   250,     0,   279,   280,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   252,   253,   254,     0,     0,   255,
     256,   257,   258,   259,   260,   261,   262,   263,   264,   265,
     266,   267,   250,   803,   268,   269,   270,   271,   272,   273,
     274,   275,   276,   277,   278,     0,     0,   279,   280,     0,
       0,   252,   253,   254,     0,     0,   255,   256,   257,   258,
     259,   260,   261,   262,   263,   264,   265,   266,   267,     0,
     818,   268,   269,   270,   271,   272,   273,   274,   275,   276,
     277,   278,   250,     0,   279,   280,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   252,   253,   254,     0,     0,   255,   256,   257,   258,
     259,   260,   261,   262,   263,   264,   265,   266,   267,   250,
       0,   268,   269,   270,   271,   272,   273,   274,   275,   276,
     277,   278,     0,     0,   279,   280,     0,     0,   252,   253,
     254,     0,     0,   255,   256,   257,   258,   259,   260,   261,
     262,   263,   264,   265,   266,   267,     0,     0,   268,   269,
     270,   271,   272,   273,   274,   275,   276,   277,   278,     0,
       0,   279,   280,     6,     7,     8,     9,    10,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    21,    22,     6,     7,     8,
       9,    10,     0,     0,     0,     0,    30,     0,     0,     0,
       0,     0,     0,     0,   193,     0,     0,     0,     0,    21,
      22,    41,   194,    42,     0,     0,     0,     0,     0,     0,
      30,     6,     7,     8,     9,    10,   195,     0,   193,     0,
       0,     0,     0,    43,    44,    41,     0,    42,     0,     0,
       0,     0,     0,    21,    22,     0,     0,     0,    47,    48,
       0,     0,     0,    49,    30,     0,     0,    43,    44,     0,
       0,    50,   193,    51,    52,    53,     0,     0,     0,    41,
       0,    42,    47,    48,     0,     0,     0,    49,     0,     0,
       0,   388,     0,     0,     0,    50,     0,    51,    52,    53,
       0,    43,    44,     0,     6,     7,     8,     9,    10,     0,
       0,     0,     0,     0,     0,     0,    47,    48,     0,     0,
       0,    49,     0,     0,     0,   438,    21,    22,     0,    50,
       0,    51,    52,    53,     0,     0,     0,    30,     6,     7,
       8,     9,    10,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    41,   352,    42,     0,     0,     0,     0,     0,
      21,    22,     6,     7,     8,     9,    10,     0,     0,     0,
       0,    30,     0,     0,    43,    44,     0,     0,     0,     0,
     403,     0,     0,     0,    21,    22,    41,     0,    42,    47,
      48,     0,     0,     0,    49,    30,     6,     7,     8,     9,
      10,     0,    50,     0,    51,    52,    53,     0,    43,    44,
      41,   471,    42,     0,     0,     0,     0,     0,    21,    22,
       0,     0,     0,    47,    48,     0,     0,     0,    49,    30,
       0,     0,    43,    44,     0,     0,    50,     0,    51,    52,
     404,     0,     0,   536,    41,     0,    42,    47,    48,     0,
       0,     0,    49,     0,     6,     7,     8,     9,    10,     0,
      50,     0,    51,    52,    53,     0,    43,    44,     0,     0,
       0,     0,     0,     0,     0,     0,    21,    22,     0,     0,
       0,    47,    48,     0,     0,     0,    49,    30,     6,     7,
       8,     9,    10,     0,    50,     0,    51,    52,    53,     0,
       0,     0,    41,     0,    42,     0,     0,     0,     0,     0,
      21,    22,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    30,     0,     0,    43,    44,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    41,     0,    42,    47,
      48,     0,     0,     0,    49,     0,     0,     0,     0,     0,
       0,     0,    50,     0,    51,    52,    53,     0,    43,    44,
       0,     0,     0,     0,     0,   355,     0,     0,     0,     0,
       0,     0,   250,    47,    48,     0,     0,     0,    49,     0,
       0,     0,     0,     0,     0,     0,    50,   356,    51,    52,
     532,   252,   253,   254,     0,     0,   255,   256,   257,   258,
     259,   260,   261,   262,   263,   264,   265,   266,   267,     0,
       0,   268,   269,   270,   271,   272,   273,   274,   275,   276,
     277,   278,   355,     0,   279,   280,     0,     0,     0,   250,
     519,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   252,   253,
     254,     0,     0,   255,   256,   257,   258,   259,   260,   261,
     262,   263,   264,   265,   266,   267,     0,     0,   268,   269,
     270,   271,   272,   273,   274,   275,   276,   277,   278,   355,
       0,   279,   280,     0,     0,     0,   250,   542,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   252,   253,   254,     0,     0,
     255,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,     0,     0,   268,   269,   270,   271,   272,
     273,   274,   275,   276,   277,   278,   540,     0,   279,   280,
       0,     0,     0,   250,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   252,   253,   254,     0,     0,   255,   256,   257,
     258,   259,   260,   261,   262,   263,   264,   265,   266,   267,
     250,   470,   268,   269,   270,   271,   272,   273,   274,   275,
     276,   277,   278,     0,     0,   279,   280,     0,     0,   252,
     253,   254,     0,     0,   255,   256,   257,   258,   259,   260,
     261,   262,   263,   264,   265,   266,   267,   250,     0,   268,
     269,   270,   271,   272,   273,   274,   275,   276,   277,   278,
       0,     0,   279,   280,   544,     0,   252,   253,   254,     0,
       0,   255,   256,   257,   258,   259,   260,   261,   262,   263,
     264,   265,   266,   267,   250,   569,   268,   269,   270,   271,
     272,   273,   274,   275,   276,   277,   278,     0,     0,   279,
     280,     0,     0,   252,   253,   254,     0,     0,   255,   256,
     257,   258,   259,   260,   261,   262,   263,   264,   265,   266,
     267,   250,   616,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,     0,     0,   279,   280,     0,     0,
     252,   253,   254,     0,     0,   255,   256,   257,   258,   259,
     260,   261,   262,   263,   264,   265,   266,   267,   250,   626,
     268,   269,   270,   271,   272,   273,   274,   275,   276,   277,
     278,     0,     0,   279,   280,     0,     0,   252,   253,   254,
       0,     0,   255,   256,   257,   258,   259,   260,   261,   262,
     263,   264,   265,   266,   267,     0,     0,   268,   269,   270,
     271,   272,   273,   274,   275,   276,   277,   278,   250,     0,
     279,   280,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   657,     0,     0,     0,   252,   253,   254,
       0,     0,   255,   256,   257,   258,   259,   260,   261,   262,
     263,   264,   265,   266,   267,   250,     0,   268,   269,   270,
     271,   272,   273,   274,   275,   276,   277,   278,     0,     0,
     279,   280,     0,     0,   252,   253,   254,     0,     0,   255,
     256,   257,   258,   259,   260,   261,   262,   263,   264,   265,
     266,   267,   250,     0,   268,   269,   270,   271,   272,   273,
     274,   275,   276,   277,   278,     0,     0,   279,   280,     0,
       0,     0,   253,   254,     0,     0,   255,   256,   257,   258,
     259,   260,   261,   262,   263,   264,   265,   266,   267,   250,
       0,   268,   269,   270,   271,   272,   273,   274,   275,   276,
     277,   278,     0,     0,   279,   280,     0,     0,     0,     0,
     254,     0,     0,   255,   256,   257,   258,   259,   260,   261,
     262,   263,   264,   265,   266,   267,   250,     0,   268,   269,
     270,   271,   272,   273,   274,   275,   276,   277,   278,     0,
       0,   279,   280,     0,     0,     0,     0,     0,     0,     0,
     255,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,     0,     0,   268,   269,   270,   271,   272,
     273,   274,   275,   276,   277,   278,     0,     0,   279,   280
};

static const yytype_int16 yycheck[] =
{
       2,     2,    12,    45,    46,   621,   114,    17,   337,   117,
     106,     3,     6,     1,     3,    17,   381,     6,     6,     3,
     128,   215,     6,   217,     3,     0,     3,   221,     1,   223,
       1,   139,     3,     4,     1,     6,     3,    39,    39,     6,
       1,     3,    44,     1,     3,     3,     4,     3,     6,    38,
       1,     1,     3,     3,    38,     6,     1,    46,    47,     3,
     168,     6,    46,    47,    14,     1,    45,     3,     4,     5,
       6,     7,     8,    75,    75,    70,    78,    78,    70,    52,
      74,    83,    83,    45,    51,     1,    74,    45,    83,     3,
      26,    27,   200,    70,    86,    13,    98,    98,    16,    88,
      18,    37,    20,    70,    88,    23,     1,    25,    69,    70,
      28,     6,    70,     3,    32,    86,    52,    35,    54,    70,
      38,     1,     3,     3,    42,    43,    70,    45,    46,    47,
      48,    49,    50,    51,    52,    53,   232,     3,    74,    75,
       1,   551,     3,   197,   198,     1,   762,     3,   558,   151,
     151,     3,     3,    89,    90,   349,    70,     1,    94,     3,
       4,     5,     6,     7,     8,     1,   102,     3,   104,   105,
     106,     3,    67,    53,   228,    70,     3,     3,    47,     1,
      70,     3,    26,    27,   378,   550,     3,     1,     3,     3,
      70,     3,    53,    37,    45,    51,   250,     3,     3,     4,
       5,     6,     7,     8,    70,   247,     1,     3,    52,    70,
      54,    83,     3,    45,    70,    51,     3,     3,    70,    70,
       1,    26,    27,     3,     3,     6,     7,    52,     6,    51,
      74,    75,    37,     1,    70,     3,   246,    51,    70,     1,
       3,     3,    52,    70,    70,    89,    90,    52,    70,    54,
      94,     3,     3,    70,    45,    70,    70,    52,   102,    45,
     104,   105,   106,     3,    70,    45,    45,    54,    55,    74,
      75,    70,    70,   191,    70,   193,   286,    45,     3,   104,
     105,     1,    45,    45,    89,    90,     6,    86,    86,    94,
     100,   101,   302,    45,   104,   105,   350,   102,     1,   104,
     105,   106,     3,     6,   358,    45,     3,   361,     4,     5,
       6,     7,     8,   507,     6,     6,   234,     9,   236,   237,
     238,   239,   240,   241,   242,   243,   244,   245,   693,   247,
       1,    27,     3,   251,   252,   253,   254,   255,   256,   257,
     258,   259,   260,   261,   262,     3,   264,   265,   266,   267,
     268,   269,   270,   271,   272,   273,   274,   275,   276,   277,
     278,     3,    33,   281,    52,     3,   368,   369,   370,   287,
       1,   373,     3,     3,     1,   377,     3,   379,   379,     1,
       1,    52,     3,   301,     6,   303,   580,   581,     3,     1,
     719,     3,     3,   311,   312,     1,     1,   451,     3,     1,
       6,     3,    52,     1,     9,     3,    33,   461,    96,    97,
      98,    99,   100,   101,    45,     3,   104,   105,    23,    24,
     474,     3,     1,   477,   342,     4,     5,     6,     7,     8,
       1,     3,     3,    45,     1,     3,     3,   355,   356,     1,
      45,     3,     9,    45,     3,   363,   364,    26,    27,    99,
     100,   101,     3,     3,   104,   105,    23,    24,    37,     4,
     514,     6,     1,    54,    55,    83,   468,     6,    54,    55,
     388,    54,    55,    52,    45,    54,     1,   671,    45,   808,
       4,     6,     6,    45,   538,     6,   404,     3,   530,     3,
     408,    82,    83,    54,    55,    74,    75,    83,     3,     3,
     694,    26,    27,     1,     1,    26,    27,     1,     3,   511,
      89,    90,     1,    53,     3,    94,     3,     6,     3,     3,
     438,    70,     3,   102,     6,   104,   105,   106,     1,     6,
     724,     4,     5,   727,     7,     8,     1,     6,   732,     4,
     734,     6,     7,     8,   462,   547,   547,     1,     6,    67,
       4,     5,     6,     7,     8,     3,     6,   475,     3,   613,
     478,     4,    53,     6,     7,     8,   620,   485,     6,     3,
      53,    83,    26,    27,     3,     3,    33,   631,     3,     6,
       3,     3,     3,    37,     3,    51,     3,     3,     3,     9,
     644,     9,     3,    30,     3,   605,     6,     9,    52,   793,
      54,     9,   796,     3,     3,   799,    52,    70,   610,    53,
     804,    53,   530,     3,   532,     3,     3,    69,     3,     3,
      74,    75,   540,     3,     3,   543,   544,   629,   629,     3,
       3,     3,    83,     9,   688,    89,    90,    51,     6,    52,
      94,   695,     6,     9,    51,     4,     3,    51,   102,     9,
     104,   105,   106,    51,     3,     6,     3,    83,     3,     3,
     662,   663,    45,     3,     3,    51,     3,     3,     3,    51,
     672,   672,     6,    51,   684,     3,    51,     3,    91,    92,
      93,    94,    95,    96,    97,    98,    99,   100,   101,    51,
     744,   104,   105,    51,     1,     3,   705,     4,     5,     6,
       7,     8,     3,   705,   705,   759,     3,     3,   663,   372,
     678,   505,   669,   683,   561,   453,   646,   771,   808,    26,
      27,   558,   459,   725,   725,   558,   728,   728,   730,   558,
      37,   733,   733,   735,   735,   408,   246,    32,    -1,   657,
      -1,    -1,    -1,   745,    -1,    52,    -1,    54,   750,   750,
      -1,   753,   753,    -1,   756,   756,    -1,   811,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    74,    75,    -1,
      -1,    -1,    -1,   775,    -1,   777,    -1,   779,    -1,   781,
      -1,   699,    89,    90,    -1,    -1,    -1,    94,     3,    -1,
      -1,    -1,    -1,   711,    -1,   102,    -1,   104,   105,   106,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,   813,   814,   815,    -1,   817,    -1,    -1,    -1,    -1,
      -1,     0,     1,    -1,     3,     4,     5,     6,     7,     8,
       9,    10,    11,    12,    13,    -1,    15,    52,    -1,    18,
      -1,    -1,    -1,    22,    23,   763,    25,    26,    27,    28,
      29,    -1,    31,    32,    -1,    34,    35,    36,    37,    -1,
      39,    40,    41,    42,    43,    44,    -1,    46,    -1,    48,
      49,    50,    -1,    52,    -1,    54,    91,    92,    93,    94,
      95,    96,    97,    98,    99,   100,   101,    -1,    -1,   104,
     105,    -1,    -1,    -1,    -1,    74,    75,    -1,    -1,    78,
      79,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      89,    90,    -1,    -1,    -1,    94,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,   102,    -1,   104,   105,   106,     1,    -1,
       3,     4,     5,     6,     7,     8,     9,    10,    11,    12,
      13,    -1,    15,    -1,    -1,    18,    19,    20,    21,    22,
      -1,    52,    25,    26,    27,    28,    29,    -1,    31,    32,
      -1,    34,    35,    36,    37,    -1,    52,    40,    -1,    -1,
      43,    -1,    -1,    46,    -1,    48,    49,    50,    -1,    52,
      -1,    54,    -1,    84,    85,    86,    87,    88,    -1,    -1,
      91,    92,    93,    94,    95,    96,    97,    98,    99,   100,
     101,    74,    75,   104,   105,    78,    79,    93,    94,    95,
      96,    97,    98,    99,   100,   101,    89,    90,   104,   105,
      -1,    94,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   102,
      -1,   104,   105,   106,     1,    -1,     3,     4,     5,     6,
       7,     8,     9,    10,    11,    12,    13,    52,    15,    -1,
      -1,    18,    -1,    -1,    -1,    22,    23,    24,    25,    26,
      27,    28,    29,    -1,    31,    32,    -1,    34,    35,    36,
      37,    -1,    -1,    40,    -1,    -1,    43,    -1,    45,    46,
      -1,    48,    49,    50,    -1,    52,    -1,    54,    -1,    94,
      95,    96,    97,    98,    99,   100,   101,    -1,    -1,   104,
     105,    -1,    -1,    -1,    -1,    -1,    -1,    74,    75,    -1,
      -1,    78,    79,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    89,    90,    -1,    -1,    -1,    94,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,   102,    -1,   104,   105,   106,
       1,    -1,     3,     4,     5,     6,     7,     8,     9,    10,
      11,    12,    13,    -1,    15,    -1,    -1,    18,    -1,    -1,
      -1,    22,    23,    24,    25,    26,    27,    28,    29,    -1,
      31,    32,    -1,    34,    35,    36,    37,    -1,    -1,    40,
      -1,    -1,    43,    -1,    45,    46,    -1,    48,    49,    50,
      -1,    52,    -1,    54,    54,    55,    56,    57,    58,    59,
      60,    61,    62,    63,    64,    65,    -1,    -1,    -1,    -1,
      70,    -1,    -1,    74,    75,    -1,    -1,    78,    79,    -1,
      -1,    -1,    -1,    83,    -1,    -1,    -1,    -1,    89,    90,
      -1,    -1,    -1,    94,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,   102,    -1,   104,   105,   106,     1,    -1,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    -1,
      15,    -1,    -1,    18,    -1,    -1,    -1,    22,    23,    24,
      25,    26,    27,    28,    29,    -1,    31,    32,    -1,    34,
      35,    36,    37,    -1,    -1,    40,    -1,    -1,    43,    -1,
      45,    46,    -1,    48,    49,    50,    -1,    52,    -1,    54,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    74,
      75,    -1,    -1,    78,    79,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    89,    90,    -1,    -1,    -1,    94,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,   102,    -1,   104,
     105,   106,     1,    -1,     3,     4,     5,     6,     7,     8,
       9,    10,    11,    12,    13,    -1,    15,    -1,    -1,    18,
      -1,    -1,    -1,    22,    23,    24,    25,    26,    27,    28,
      29,    -1,    31,    32,    -1,    34,    35,    36,    37,    -1,
      -1,    40,    -1,    -1,    43,    -1,    45,    46,    -1,    48,
      49,    50,    -1,    52,    -1,    54,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    74,    75,    -1,    -1,    78,
      79,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      89,    90,    -1,    -1,    -1,    94,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,   102,    -1,   104,   105,   106,     1,    -1,
       3,     4,     5,     6,     7,     8,     9,    10,    11,    12,
      13,    -1,    15,    16,    17,    18,    -1,    -1,    -1,    22,
      -1,    -1,    25,    26,    27,    28,    29,    -1,    31,    32,
      -1,    34,    35,    36,    37,    -1,    -1,    40,    -1,    -1,
      43,    -1,    -1,    46,    -1,    48,    49,    50,    -1,    52,
      -1,    54,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    74,    75,    -1,    -1,    78,    79,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    89,    90,    -1,    -1,
      -1,    94,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   102,
      -1,   104,   105,   106,     1,    -1,     3,     4,     5,     6,
       7,     8,     9,    10,    11,    12,    13,    -1,    15,    -1,
      -1,    18,    -1,    -1,    -1,    22,    -1,    -1,    25,    26,
      27,    28,    29,    30,    31,    32,    -1,    34,    35,    36,
      37,    -1,    -1,    40,    -1,    -1,    43,    -1,    -1,    46,
      -1,    48,    49,    50,    -1,    52,    -1,    54,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    74,    75,    -1,
      -1,    78,    79,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    89,    90,    -1,    -1,    -1,    94,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,   102,    -1,   104,   105,   106,
       1,    -1,     3,     4,     5,     6,     7,     8,     9,    10,
      11,    12,    13,    -1,    15,    -1,    -1,    18,    -1,    -1,
      -1,    22,    -1,    -1,    25,    26,    27,    28,    29,    30,
      31,    32,    -1,    34,    35,    36,    37,    -1,    -1,    40,
      -1,    -1,    43,    -1,    -1,    46,    -1,    48,    49,    50,
      -1,    52,    -1,    54,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    74,    75,    -1,    -1,    78,    79,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    89,    90,
      -1,    -1,    -1,    94,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,   102,    -1,   104,   105,   106,     1,    -1,     3,     4,
       5,     6,     7,     8,    -1,    10,    11,    12,    13,    -1,
      15,    -1,    -1,    18,    -1,    -1,    -1,    22,    -1,    -1,
      25,    26,    27,    28,    29,    -1,    31,    32,    -1,    34,
      35,    36,    37,    -1,    -1,    40,    -1,    -1,    43,    -1,
      45,    46,    -1,    48,    49,    50,    -1,    52,    -1,    54,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    74,
      75,    -1,    -1,    78,    79,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    89,    90,    -1,    -1,    -1,    94,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,   102,    -1,   104,
     105,   106,     1,    -1,     3,     4,     5,     6,     7,     8,
       9,    10,    11,    12,    13,    -1,    15,    -1,    -1,    18,
      -1,    -1,    -1,    22,    -1,    -1,    25,    26,    27,    28,
      29,    -1,    31,    32,    -1,    34,    35,    36,    37,    -1,
      -1,    40,    -1,    -1,    43,    -1,    -1,    46,    -1,    48,
      49,    50,    -1,    52,    -1,    54,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    74,    75,    -1,    -1,    78,
      79,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      89,    90,    -1,    -1,    -1,    94,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,   102,    -1,   104,   105,   106,     1,    -1,
       3,     4,     5,     6,     7,     8,     9,    10,    11,    12,
      13,    -1,    15,    -1,    -1,    18,    -1,    -1,    -1,    22,
      -1,    -1,    25,    26,    27,    28,    29,    -1,    31,    32,
      -1,    34,    35,    36,    37,    -1,    -1,    40,    -1,    -1,
      43,    -1,    -1,    46,    -1,    48,    49,    50,    -1,    52,
      -1,    54,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    74,    75,    -1,    -1,    78,    79,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    89,    90,    -1,    -1,
      -1,    94,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   102,
      -1,   104,   105,   106,     1,    -1,     3,     4,     5,     6,
       7,     8,     9,    10,    11,    12,    13,    -1,    15,    -1,
      -1,    18,    -1,    -1,    -1,    22,    -1,    -1,    25,    26,
      27,    28,    29,    -1,    31,    32,    -1,    34,    35,    36,
      37,    -1,    -1,    40,    -1,    -1,    43,    -1,    -1,    46,
      -1,    48,    49,    50,    -1,    52,    -1,    54,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    74,    75,    -1,
      -1,    78,    79,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    89,    90,    -1,    -1,    -1,    94,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,   102,    -1,   104,   105,   106,
       1,    -1,     3,     4,     5,     6,     7,     8,     9,    10,
      11,    12,    13,    -1,    15,    -1,    -1,    18,    -1,    -1,
      -1,    22,    -1,    -1,    25,    26,    27,    28,    29,    -1,
      31,    32,    -1,    34,    35,    36,    37,    -1,    -1,    40,
      -1,    -1,    43,    -1,    -1,    46,    -1,    48,    49,    50,
      -1,    52,    -1,    54,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    74,    75,    -1,    -1,    78,    79,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    89,    90,
      -1,    -1,    -1,    94,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,   102,    -1,   104,   105,   106,     1,    -1,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    -1,
      15,    -1,    -1,    18,    -1,    -1,    -1,    22,    -1,    -1,
      25,    26,    27,    28,    29,    -1,    31,    32,    -1,    34,
      35,    36,    37,    -1,    -1,    40,    -1,    -1,    43,    -1,
      -1,    46,    -1,    48,    49,    50,    -1,    52,    -1,    54,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    74,
      75,    -1,    -1,    78,    79,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    89,    90,    -1,    -1,    -1,    94,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,   102,    -1,   104,
     105,   106,     1,    -1,     3,     4,     5,     6,     7,     8,
       9,    10,    11,    12,    13,    -1,    15,    -1,    -1,    18,
      -1,    -1,    -1,    22,    -1,    -1,    25,    26,    27,    28,
      29,    -1,    31,    32,    -1,    34,    35,    36,    37,    -1,
      -1,    40,    -1,    -1,    43,    -1,    -1,    46,    -1,    48,
      49,    50,    -1,    52,    -1,    54,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    74,    75,    -1,    -1,    78,
      79,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      89,    90,    -1,    -1,    -1,    94,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,   102,    -1,   104,   105,   106,     1,    -1,
       3,     4,     5,     6,     7,     8,     9,    10,    11,    12,
      13,    -1,    15,    -1,    -1,    18,    -1,    -1,    -1,    22,
      -1,    -1,    25,    26,    27,    28,    29,    -1,    31,    32,
      -1,    34,    35,    36,    37,    -1,    -1,    40,    -1,    -1,
      43,    -1,    -1,    46,    -1,    48,    49,    50,    -1,    52,
      -1,    54,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    74,    75,    -1,    -1,    78,    79,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    89,    90,    -1,    -1,
      -1,    94,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   102,
      -1,   104,   105,   106,     1,    -1,     3,     4,     5,     6,
       7,     8,     9,    10,    11,    12,    13,    -1,    15,    -1,
      -1,    18,    -1,    -1,    -1,    22,    -1,    -1,    25,    26,
      27,    28,    29,    -1,    31,    32,    -1,    34,    35,    36,
      37,    -1,    -1,    40,    -1,    -1,    43,    -1,    -1,    46,
      -1,    48,    49,    50,    -1,    52,    -1,    54,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    74,    75,    -1,
      -1,    78,    79,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    89,    90,    -1,    -1,    -1,    94,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,   102,    -1,   104,   105,   106,
       1,    -1,     3,     4,     5,     6,     7,     8,     9,    10,
      11,    12,    13,    -1,    15,    -1,    -1,    18,    -1,    -1,
      -1,    22,    -1,    -1,    25,    26,    27,    28,    29,    -1,
      31,    32,    -1,    34,    35,    36,    37,    -1,    -1,    40,
      -1,    -1,    43,    -1,    -1,    46,    -1,    48,    49,    50,
      -1,    52,    -1,    54,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    74,    75,    -1,    -1,    78,    79,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    89,    90,
      -1,    -1,    -1,    94,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,   102,    -1,   104,   105,   106,     1,    -1,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    -1,
      15,    -1,    -1,    18,    -1,    -1,    -1,    22,    -1,    -1,
      25,    26,    27,    28,    29,    -1,    31,    32,    -1,    34,
      35,    36,    37,    -1,    -1,    40,    -1,    -1,    43,    -1,
      -1,    46,    -1,    48,    49,    50,    -1,    52,    -1,    54,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    74,
      75,    -1,    -1,    78,    79,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    89,    90,    -1,    -1,    -1,    94,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,   102,    -1,   104,
     105,   106,     1,    -1,     3,     4,     5,     6,     7,     8,
       9,    10,    11,    12,    13,    -1,    15,    -1,    -1,    18,
      -1,    -1,    -1,    22,    -1,    -1,    25,    26,    27,    28,
      29,    -1,    31,    32,    -1,    34,    35,    36,    37,    -1,
      -1,    40,    -1,    -1,    43,    -1,    -1,    46,    -1,    48,
      49,    50,    -1,    52,    -1,    54,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    74,    75,    -1,    -1,    78,
      79,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      89,    90,    -1,    -1,    -1,    94,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,   102,    -1,   104,   105,   106,     1,    -1,
       3,     4,     5,     6,     7,     8,     9,    10,    11,    12,
      13,    -1,    15,    -1,    -1,    18,    -1,    -1,    -1,    22,
      -1,    -1,    25,    26,    27,    28,    29,    -1,    31,    32,
      -1,    34,    35,    36,    37,    -1,    -1,    40,    -1,    -1,
      43,    -1,    -1,    46,    -1,    48,    49,    50,    -1,    52,
      -1,    54,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    74,    75,    -1,    -1,    78,    79,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    89,    90,    -1,    -1,
      -1,    94,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   102,
      -1,   104,   105,   106,     1,    -1,     3,     4,     5,     6,
       7,     8,    -1,    10,    11,    12,    13,    -1,    15,    -1,
      -1,    18,    -1,    -1,    -1,    22,    -1,    -1,    25,    26,
      27,    28,    29,    -1,    31,    32,    -1,    34,    35,    36,
      37,    -1,    -1,    40,    -1,    -1,    43,    -1,    -1,    46,
      -1,    48,    49,    50,    -1,    52,    -1,    54,    -1,     1,
      -1,    -1,     4,     5,     6,     7,     8,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    74,    75,    -1,
      -1,    78,    79,    -1,    26,    27,    -1,    -1,    -1,    -1,
      -1,    -1,    89,    90,    -1,    37,    -1,    94,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,   102,    -1,   104,   105,   106,
      52,    -1,    54,    -1,     1,    -1,    -1,     4,     5,     6,
       7,     8,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    74,    75,    -1,    -1,    -1,    -1,    -1,    26,
      27,    -1,    -1,    -1,    -1,    -1,    -1,    89,    90,    -1,
      37,    -1,    94,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
     102,    -1,   104,   105,   106,    52,    -1,    54,    -1,     1,
      -1,    -1,     4,     5,     6,     7,     8,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    74,    75,    -1,
      -1,    -1,    -1,    -1,    26,    27,    -1,    -1,    -1,    -1,
      -1,    -1,    89,    90,    -1,    37,    -1,    94,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,   102,    -1,   104,   105,   106,
      52,    -1,    54,    -1,     1,    -1,    -1,     4,     5,     6,
       7,     8,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    74,    75,    -1,    -1,    -1,    -1,    -1,    26,
      27,    -1,    -1,    -1,    -1,    -1,    -1,    89,    90,    -1,
      37,    -1,    94,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
     102,    -1,   104,   105,   106,    52,    -1,    54,    -1,     1,
      -1,    -1,     4,     5,     6,     7,     8,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    74,    75,    -1,
      -1,    -1,    -1,    -1,    26,    27,    -1,    -1,    -1,    -1,
      -1,    -1,    89,    90,    -1,    37,    -1,    94,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,   102,    -1,   104,   105,   106,
      52,    -1,    54,    -1,     1,    -1,    -1,     4,     5,     6,
       7,     8,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    74,    75,    -1,    -1,    -1,    -1,    -1,    26,
      27,    -1,    -1,    -1,    -1,    -1,    -1,    89,    90,    -1,
      37,    -1,    94,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
     102,    -1,   104,   105,   106,    52,    -1,    54,    -1,     1,
      -1,    -1,     4,     5,     6,     7,     8,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    74,    75,    -1,
      -1,    -1,    -1,    -1,    26,    27,    -1,    -1,    -1,    -1,
      -1,    -1,    89,    90,    -1,    37,    -1,    94,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,   102,    -1,   104,   105,   106,
      52,    -1,    54,    -1,     1,    -1,    -1,     4,     5,     6,
       7,     8,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    74,    75,    -1,    -1,    -1,    -1,    -1,    26,
      27,    -1,    -1,    -1,    -1,    -1,    -1,    89,    90,    -1,
      37,    -1,    94,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
     102,    -1,   104,   105,   106,    52,    -1,    54,    -1,     1,
      -1,    -1,     4,     5,     6,     7,     8,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    74,    75,    -1,
      -1,    -1,    -1,    -1,    26,    27,    -1,    -1,    -1,    -1,
      -1,    -1,    89,    90,    -1,    37,    -1,    94,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,   102,    -1,   104,   105,   106,
      52,    -1,    54,    -1,     1,    -1,    -1,     4,     5,     6,
       7,     8,     3,     4,     5,     6,     7,     8,    -1,    -1,
      -1,    -1,    74,    75,    -1,    -1,    -1,    -1,    -1,    26,
      27,    -1,    -1,    -1,    -1,    26,    27,    89,    90,    -1,
      37,    -1,    94,    -1,    -1,    -1,    37,    -1,    -1,    -1,
     102,    -1,   104,   105,   106,    52,    -1,    54,    -1,    -1,
      -1,    52,    -1,    54,    -1,    -1,    -1,    -1,     3,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    74,    75,    -1,
      -1,    -1,    -1,    74,    75,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    89,    90,    -1,    -1,    -1,    94,    89,    90,
      -1,    -1,    -1,    94,    -1,   102,    -1,   104,   105,   106,
      45,   102,    -1,   104,   105,   106,    -1,    52,     3,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    68,    -1,    -1,    71,    72,    73,    -1,
      -1,    76,    77,    78,    79,    80,    81,    82,    83,    84,
      85,    86,    87,    88,    -1,    -1,    91,    92,    93,    94,
      95,    96,    97,    98,    99,   100,   101,    52,    -1,   104,
     105,    -1,    -1,    -1,    -1,     3,    -1,    -1,    -1,    -1,
      -1,    66,    -1,    -1,    -1,    -1,    71,    72,    73,    -1,
      -1,    76,    77,    78,    79,    80,    81,    82,    83,    84,
      85,    86,    87,    88,    -1,    -1,    91,    92,    93,    94,
      95,    96,    97,    98,    99,   100,   101,    45,    -1,   104,
     105,    -1,    -1,    -1,    52,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,     3,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    71,    72,    73,    -1,    -1,    76,    77,
      78,    79,    80,    81,    82,    83,    84,    85,    86,    87,
      88,    -1,    -1,    91,    92,    93,    94,    95,    96,    97,
      98,    99,   100,   101,    45,    -1,   104,   105,    -1,    -1,
      -1,    52,    -1,    -1,    -1,    -1,    -1,    -1,    -1,     3,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      71,    72,    73,    -1,    -1,    76,    77,    78,    79,    80,
      81,    82,    83,    84,    85,    86,    87,    88,    -1,    -1,
      91,    92,    93,    94,    95,    96,    97,    98,    99,   100,
     101,    45,    -1,   104,   105,    -1,    -1,    -1,    52,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,     3,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    71,    72,    73,
      -1,    -1,    76,    77,    78,    79,    80,    81,    82,    83,
      84,    85,    86,    87,    88,    -1,    -1,    91,    92,    93,
      94,    95,    96,    97,    98,    99,   100,   101,    45,    -1,
     104,   105,    -1,    -1,    -1,    52,     3,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    71,    72,    73,    -1,    -1,    76,
      77,    78,    79,    80,    81,    82,    83,    84,    85,    86,
      87,    88,    -1,     3,    91,    92,    93,    94,    95,    96,
      97,    98,    99,   100,   101,    52,    -1,   104,   105,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    71,    72,    73,    -1,    -1,    76,
      77,    78,    79,    80,    81,    82,    83,    84,    85,    86,
      87,    88,    52,     3,    91,    92,    93,    94,    95,    96,
      97,    98,    99,   100,   101,    -1,    -1,   104,   105,    -1,
      -1,    71,    72,    73,    -1,    -1,    76,    77,    78,    79,
      80,    81,    82,    83,    84,    85,    86,    87,    88,    -1,
       3,    91,    92,    93,    94,    95,    96,    97,    98,    99,
     100,   101,    52,    -1,   104,   105,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    71,    72,    73,    -1,    -1,    76,    77,    78,    79,
      80,    81,    82,    83,    84,    85,    86,    87,    88,    52,
       3,    91,    92,    93,    94,    95,    96,    97,    98,    99,
     100,   101,    -1,    -1,   104,   105,    -1,    -1,    71,    72,
      73,    -1,    -1,    76,    77,    78,    79,    80,    81,    82,
      83,    84,    85,    86,    87,    88,    -1,     3,    91,    92,
      93,    94,    95,    96,    97,    98,    99,   100,   101,    52,
      -1,   104,   105,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    71,    72,
      73,    -1,    -1,    76,    77,    78,    79,    80,    81,    82,
      83,    84,    85,    86,    87,    88,    52,     3,    91,    92,
      93,    94,    95,    96,    97,    98,    99,   100,   101,    -1,
      -1,   104,   105,    -1,    -1,    71,    72,    73,    -1,    -1,
      76,    77,    78,    79,    80,    81,    82,    83,    84,    85,
      86,    87,    88,    -1,     3,    91,    92,    93,    94,    95,
      96,    97,    98,    99,   100,   101,    52,    -1,   104,   105,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    71,    72,    73,    -1,    -1,
      76,    77,    78,    79,    80,    81,    82,    83,    84,    85,
      86,    87,    88,    52,     3,    91,    92,    93,    94,    95,
      96,    97,    98,    99,   100,   101,    -1,    -1,   104,   105,
      -1,    -1,    71,    72,    73,    -1,    -1,    76,    77,    78,
      79,    80,    81,    82,    83,    84,    85,    86,    87,    88,
      -1,     3,    91,    92,    93,    94,    95,    96,    97,    98,
      99,   100,   101,    52,    -1,   104,   105,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    71,    72,    73,    -1,    -1,    76,    77,    78,
      79,    80,    81,    82,    83,    84,    85,    86,    87,    88,
      52,     3,    91,    92,    93,    94,    95,    96,    97,    98,
      99,   100,   101,    -1,    -1,   104,   105,    -1,    -1,    71,
      72,    73,    -1,    -1,    76,    77,    78,    79,    80,    81,
      82,    83,    84,    85,    86,    87,    88,    -1,     3,    91,
      92,    93,    94,    95,    96,    97,    98,    99,   100,   101,
      52,    -1,   104,   105,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    71,
      72,    73,    -1,    -1,    76,    77,    78,    79,    80,    81,
      82,    83,    84,    85,    86,    87,    88,    52,     3,    91,
      92,    93,    94,    95,    96,    97,    98,    99,   100,   101,
      -1,    -1,   104,   105,    -1,    -1,    71,    72,    73,    -1,
      -1,    76,    77,    78,    79,    80,    81,    82,    83,    84,
      85,    86,    87,    88,    -1,     3,    91,    92,    93,    94,
      95,    96,    97,    98,    99,   100,   101,    52,    -1,   104,
     105,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    71,    72,    73,    -1,
      -1,    76,    77,    78,    79,    80,    81,    82,    83,    84,
      85,    86,    87,    88,    52,     3,    91,    92,    93,    94,
      95,    96,    97,    98,    99,   100,   101,    -1,    -1,   104,
     105,    -1,    -1,    71,    72,    73,    -1,    -1,    76,    77,
      78,    79,    80,    81,    82,    83,    84,    85,    86,    87,
      88,    -1,     3,    91,    92,    93,    94,    95,    96,    97,
      98,    99,   100,   101,    52,    -1,   104,   105,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    71,    72,    73,    -1,    -1,    76,    77,
      78,    79,    80,    81,    82,    83,    84,    85,    86,    87,
      88,    52,     3,    91,    92,    93,    94,    95,    96,    97,
      98,    99,   100,   101,    -1,    -1,   104,   105,    -1,    -1,
      71,    72,    73,    -1,    -1,    76,    77,    78,    79,    80,
      81,    82,    83,    84,    85,    86,    87,    88,    -1,     3,
      91,    92,    93,    94,    95,    96,    97,    98,    99,   100,
     101,    52,    -1,   104,   105,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      71,    72,    73,    -1,    -1,    76,    77,    78,    79,    80,
      81,    82,    83,    84,    85,    86,    87,    88,    52,     3,
      91,    92,    93,    94,    95,    96,    97,    98,    99,   100,
     101,    -1,    -1,   104,   105,    -1,    -1,    71,    72,    73,
      -1,    -1,    76,    77,    78,    79,    80,    81,    82,    83,
      84,    85,    86,    87,    88,    -1,     3,    91,    92,    93,
      94,    95,    96,    97,    98,    99,   100,   101,    52,    -1,
     104,   105,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    71,    72,    73,
      -1,    -1,    76,    77,    78,    79,    80,    81,    82,    83,
      84,    85,    86,    87,    88,    52,     3,    91,    92,    93,
      94,    95,    96,    97,    98,    99,   100,   101,    -1,    -1,
     104,   105,    -1,    -1,    71,    72,    73,    -1,    -1,    76,
      77,    78,    79,    80,    81,    82,    83,    84,    85,    86,
      87,    88,    -1,     3,    91,    92,    93,    94,    95,    96,
      97,    98,    99,   100,   101,    52,    -1,   104,   105,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    71,    72,    73,    -1,    -1,    76,
      77,    78,    79,    80,    81,    82,    83,    84,    85,    86,
      87,    88,    52,     3,    91,    92,    93,    94,    95,    96,
      97,    98,    99,   100,   101,    -1,    -1,   104,   105,    -1,
      -1,    71,    72,    73,    -1,    -1,    76,    77,    78,    79,
      80,    81,    82,    83,    84,    85,    86,    87,    88,    -1,
       3,    91,    92,    93,    94,    95,    96,    97,    98,    99,
     100,   101,    52,    -1,   104,   105,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    71,    72,    73,    -1,    -1,    76,    77,    78,    79,
      80,    81,    82,    83,    84,    85,    86,    87,    88,    52,
      -1,    91,    92,    93,    94,    95,    96,    97,    98,    99,
     100,   101,    -1,    -1,   104,   105,    -1,    -1,    71,    72,
      73,    -1,    -1,    76,    77,    78,    79,    80,    81,    82,
      83,    84,    85,    86,    87,    88,    -1,    -1,    91,    92,
      93,    94,    95,    96,    97,    98,    99,   100,   101,    -1,
      -1,   104,   105,     4,     5,     6,     7,     8,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    26,    27,     4,     5,     6,
       7,     8,    -1,    -1,    -1,    -1,    37,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    45,    -1,    -1,    -1,    -1,    26,
      27,    52,    53,    54,    -1,    -1,    -1,    -1,    -1,    -1,
      37,     4,     5,     6,     7,     8,    67,    -1,    45,    -1,
      -1,    -1,    -1,    74,    75,    52,    -1,    54,    -1,    -1,
      -1,    -1,    -1,    26,    27,    -1,    -1,    -1,    89,    90,
      -1,    -1,    -1,    94,    37,    -1,    -1,    74,    75,    -1,
      -1,   102,    45,   104,   105,   106,    -1,    -1,    -1,    52,
      -1,    54,    89,    90,    -1,    -1,    -1,    94,    -1,    -1,
      -1,    98,    -1,    -1,    -1,   102,    -1,   104,   105,   106,
      -1,    74,    75,    -1,     4,     5,     6,     7,     8,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    89,    90,    -1,    -1,
      -1,    94,    -1,    -1,    -1,    98,    26,    27,    -1,   102,
      -1,   104,   105,   106,    -1,    -1,    -1,    37,     4,     5,
       6,     7,     8,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    52,    53,    54,    -1,    -1,    -1,    -1,    -1,
      26,    27,     4,     5,     6,     7,     8,    -1,    -1,    -1,
      -1,    37,    -1,    -1,    74,    75,    -1,    -1,    -1,    -1,
      46,    -1,    -1,    -1,    26,    27,    52,    -1,    54,    89,
      90,    -1,    -1,    -1,    94,    37,     4,     5,     6,     7,
       8,    -1,   102,    -1,   104,   105,   106,    -1,    74,    75,
      52,    53,    54,    -1,    -1,    -1,    -1,    -1,    26,    27,
      -1,    -1,    -1,    89,    90,    -1,    -1,    -1,    94,    37,
      -1,    -1,    74,    75,    -1,    -1,   102,    -1,   104,   105,
     106,    -1,    -1,    51,    52,    -1,    54,    89,    90,    -1,
      -1,    -1,    94,    -1,     4,     5,     6,     7,     8,    -1,
     102,    -1,   104,   105,   106,    -1,    74,    75,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    26,    27,    -1,    -1,
      -1,    89,    90,    -1,    -1,    -1,    94,    37,     4,     5,
       6,     7,     8,    -1,   102,    -1,   104,   105,   106,    -1,
      -1,    -1,    52,    -1,    54,    -1,    -1,    -1,    -1,    -1,
      26,    27,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    37,    -1,    -1,    74,    75,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    52,    -1,    54,    89,
      90,    -1,    -1,    -1,    94,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,   102,    -1,   104,   105,   106,    -1,    74,    75,
      -1,    -1,    -1,    -1,    -1,    45,    -1,    -1,    -1,    -1,
      -1,    -1,    52,    89,    90,    -1,    -1,    -1,    94,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,   102,    67,   104,   105,
     106,    71,    72,    73,    -1,    -1,    76,    77,    78,    79,
      80,    81,    82,    83,    84,    85,    86,    87,    88,    -1,
      -1,    91,    92,    93,    94,    95,    96,    97,    98,    99,
     100,   101,    45,    -1,   104,   105,    -1,    -1,    -1,    52,
      53,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    71,    72,
      73,    -1,    -1,    76,    77,    78,    79,    80,    81,    82,
      83,    84,    85,    86,    87,    88,    -1,    -1,    91,    92,
      93,    94,    95,    96,    97,    98,    99,   100,   101,    45,
      -1,   104,   105,    -1,    -1,    -1,    52,    53,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    71,    72,    73,    -1,    -1,
      76,    77,    78,    79,    80,    81,    82,    83,    84,    85,
      86,    87,    88,    -1,    -1,    91,    92,    93,    94,    95,
      96,    97,    98,    99,   100,   101,    45,    -1,   104,   105,
      -1,    -1,    -1,    52,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    71,    72,    73,    -1,    -1,    76,    77,    78,
      79,    80,    81,    82,    83,    84,    85,    86,    87,    88,
      52,    53,    91,    92,    93,    94,    95,    96,    97,    98,
      99,   100,   101,    -1,    -1,   104,   105,    -1,    -1,    71,
      72,    73,    -1,    -1,    76,    77,    78,    79,    80,    81,
      82,    83,    84,    85,    86,    87,    88,    52,    -1,    91,
      92,    93,    94,    95,    96,    97,    98,    99,   100,   101,
      -1,    -1,   104,   105,    69,    -1,    71,    72,    73,    -1,
      -1,    76,    77,    78,    79,    80,    81,    82,    83,    84,
      85,    86,    87,    88,    52,    53,    91,    92,    93,    94,
      95,    96,    97,    98,    99,   100,   101,    -1,    -1,   104,
     105,    -1,    -1,    71,    72,    73,    -1,    -1,    76,    77,
      78,    79,    80,    81,    82,    83,    84,    85,    86,    87,
      88,    52,    53,    91,    92,    93,    94,    95,    96,    97,
      98,    99,   100,   101,    -1,    -1,   104,   105,    -1,    -1,
      71,    72,    73,    -1,    -1,    76,    77,    78,    79,    80,
      81,    82,    83,    84,    85,    86,    87,    88,    52,    53,
      91,    92,    93,    94,    95,    96,    97,    98,    99,   100,
     101,    -1,    -1,   104,   105,    -1,    -1,    71,    72,    73,
      -1,    -1,    76,    77,    78,    79,    80,    81,    82,    83,
      84,    85,    86,    87,    88,    -1,    -1,    91,    92,    93,
      94,    95,    96,    97,    98,    99,   100,   101,    52,    -1,
     104,   105,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    67,    -1,    -1,    -1,    71,    72,    73,
      -1,    -1,    76,    77,    78,    79,    80,    81,    82,    83,
      84,    85,    86,    87,    88,    52,    -1,    91,    92,    93,
      94,    95,    96,    97,    98,    99,   100,   101,    -1,    -1,
     104,   105,    -1,    -1,    71,    72,    73,    -1,    -1,    76,
      77,    78,    79,    80,    81,    82,    83,    84,    85,    86,
      87,    88,    52,    -1,    91,    92,    93,    94,    95,    96,
      97,    98,    99,   100,   101,    -1,    -1,   104,   105,    -1,
      -1,    -1,    72,    73,    -1,    -1,    76,    77,    78,    79,
      80,    81,    82,    83,    84,    85,    86,    87,    88,    52,
      -1,    91,    92,    93,    94,    95,    96,    97,    98,    99,
     100,   101,    -1,    -1,   104,   105,    -1,    -1,    -1,    -1,
      73,    -1,    -1,    76,    77,    78,    79,    80,    81,    82,
      83,    84,    85,    86,    87,    88,    52,    -1,    91,    92,
      93,    94,    95,    96,    97,    98,    99,   100,   101,    -1,
      -1,   104,   105,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      76,    77,    78,    79,    80,    81,    82,    83,    84,    85,
      86,    87,    88,    -1,    -1,    91,    92,    93,    94,    95,
      96,    97,    98,    99,   100,   101,    -1,    -1,   104,   105
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint16 yystos[] =
{
       0,   108,   109,     0,     1,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    13,    15,    18,    22,    23,
      25,    26,    27,    28,    29,    31,    32,    34,    35,    36,
      37,    39,    40,    41,    42,    43,    44,    46,    48,    49,
      50,    52,    54,    74,    75,    78,    79,    89,    90,    94,
     102,   104,   105,   106,   110,   111,   112,   113,   115,   116,
     117,   118,   119,   120,   121,   122,   123,   124,   125,   126,
     127,   128,   129,   130,   133,   134,   135,   137,   138,   146,
     147,   148,   150,   151,   152,   156,   157,   158,   165,   167,
     180,   182,   191,   192,   194,   201,   202,   203,   205,   207,
     215,   216,   217,   218,   220,   221,   222,   225,   243,   248,
     252,   253,   254,   255,   256,   257,   258,   259,   267,   270,
     271,   273,     3,     3,     1,   114,   254,     1,   256,   257,
       1,     3,     1,     3,    14,     1,   257,     1,   254,   256,
     277,     1,   257,     1,     1,   257,     1,   257,   275,     1,
       3,    45,     1,   257,     1,     6,     1,     6,     1,     3,
     257,   249,   268,     1,     6,     7,     1,   257,   259,     1,
       6,     1,     3,    45,     1,   257,     1,     3,     6,   219,
       1,     6,     1,   257,     3,    32,    34,    45,    46,   113,
       3,   261,   262,    45,    53,    67,   257,   276,   278,   257,
     256,     1,     3,   275,     3,   275,   257,   257,   257,   257,
     257,   257,   257,   131,   113,   136,   113,   149,   113,   166,
     181,   193,    47,   210,   213,   214,   113,     1,    52,     1,
       6,   223,   224,   223,    54,    55,    56,    57,    58,    59,
      60,    61,    62,    63,    64,    65,    70,    83,   258,     3,
      52,    66,    71,    72,    73,    76,    77,    78,    79,    80,
      81,    82,    83,    84,    85,    86,    87,    88,    91,    92,
      93,    94,    95,    96,    97,    98,    99,   100,   101,   104,
     105,    54,    55,   258,     3,     3,    70,    83,     3,    45,
       3,    45,     3,     3,     3,     3,    45,     3,    45,     3,
      45,    83,    70,    86,     3,     3,     3,     3,     3,     3,
       1,    69,    70,     3,   113,     3,     3,     3,   226,     3,
     244,     3,     3,     6,   250,   251,     1,     6,   208,   209,
     269,     3,     3,     3,     3,     3,     3,    83,     3,    45,
       3,     3,    86,     3,     3,    70,     3,     3,     3,   132,
     257,     3,    53,   257,    53,    45,    67,     1,    70,   261,
       1,    70,   261,    82,    83,     3,     3,     3,   145,   145,
     145,   168,   183,   145,     1,     3,    45,   145,   211,   212,
       3,   261,     3,     3,    70,     9,   223,     3,    98,   257,
       6,   257,   257,   257,   257,   257,   257,   257,   257,   257,
     257,   254,   277,    46,   106,   257,   263,   275,   261,   257,
       1,   257,   257,   257,   257,   257,   257,   257,   257,   257,
     257,   257,     6,   257,   257,   257,   257,   257,   257,   257,
     257,   257,   257,   257,   257,   257,   257,   257,    98,   257,
       6,   254,   257,   257,   254,     1,   257,     3,   257,   257,
       1,    52,   227,   228,     1,    33,   230,   245,     3,    70,
       3,    70,    67,     1,   253,     1,   257,     6,   145,   261,
      53,    53,   257,   257,   272,   261,    53,   274,   261,    53,
     257,   257,     9,   113,    16,    17,   139,   141,   142,   144,
       9,     1,     3,    23,    24,   169,   174,   176,     1,     3,
      23,   174,   184,    30,   195,   196,   197,   198,     3,    45,
       9,   145,   113,     1,   208,     6,     3,     3,   257,    53,
       3,     3,     3,     3,     3,     3,     3,     3,     3,     3,
      83,   264,   106,     3,     3,     3,    51,   257,   276,     3,
      45,   257,    53,    83,    69,     3,     3,    45,     3,     3,
     261,   236,   230,     3,     6,   231,   232,     3,   246,     1,
     251,   261,   257,     3,     3,     3,     3,     9,    51,    53,
     261,   257,   261,   257,     3,     1,     3,     1,   257,     9,
     140,   143,     3,     3,     1,     4,     6,     7,     8,   178,
     179,     1,     9,   175,     3,     1,     4,     6,   189,   190,
       9,     1,     3,     4,     6,    86,   199,   200,     9,   197,
     145,     3,     9,   206,     1,   261,    53,   257,   275,     1,
      52,   265,     3,     1,   261,   257,    53,   257,   257,   153,
     113,   208,     3,     6,    38,    46,    47,    88,   202,   237,
     238,   240,   241,     3,    52,   233,    70,     3,   202,   238,
     240,   241,   247,   209,     3,    53,    53,    67,     3,     3,
       3,     3,   145,   145,     3,    45,    69,     3,    45,    70,
       3,     3,    45,   177,     3,    45,     3,    45,    70,     3,
       3,   254,     3,    70,    86,     3,     3,   261,   204,    51,
       3,     3,     3,   261,   210,   260,    51,     3,    45,    68,
      19,    20,    21,   113,   154,   155,   159,   161,   163,     1,
     261,    83,     3,     6,     1,     6,    74,   242,     9,   261,
     232,     9,   257,   139,   172,   173,     4,   170,   171,   179,
     145,   113,   187,   188,   185,   186,   190,     3,   200,   254,
      51,   261,     3,    45,   208,   145,   261,   257,     1,     3,
      45,     1,     3,    45,     1,     3,    45,     9,   154,   229,
      51,   257,   239,    83,     3,     6,     3,    70,     3,     6,
      27,   234,   235,   253,     3,   145,   113,   145,   113,   145,
     113,   145,   113,     3,    45,    51,     1,   261,     9,    51,
       3,    45,     3,   160,   113,     3,   162,   113,     3,   164,
     113,     3,   261,     3,   210,   257,     6,    74,    70,   261,
       3,   266,    51,   145,   145,   145,    51,   145,     3,     6,
     235,    51,   261,     3,     9,     9,     9,     9,    51,     3,
       3,     3,     3,     3
};

#define yyerrok		(yyerrstatus = 0)
#define yyclearin	(yychar = YYEMPTY)
#define YYEMPTY		(-2)
#define YYEOF		0

#define YYACCEPT	goto yyacceptlab
#define YYABORT		goto yyabortlab
#define YYERROR		goto yyerrorlab


/* Like YYERROR except do call yyerror.  This remains here temporarily
   to ease the transition to the new meaning of YYERROR, for GCC.
   Once GCC version 2 has supplanted version 1, this can go.  */

#define YYFAIL		goto yyerrlab

#define YYRECOVERING()  (!!yyerrstatus)

#define YYBACKUP(Token, Value)					\
do								\
  if (yychar == YYEMPTY && yylen == 1)				\
    {								\
      yychar = (Token);						\
      yylval = (Value);						\
      yytoken = YYTRANSLATE (yychar);				\
      YYPOPSTACK (1);						\
      goto yybackup;						\
    }								\
  else								\
    {								\
      yyerror (YY_("syntax error: cannot back up")); \
      YYERROR;							\
    }								\
while (YYID (0))


#define YYTERROR	1
#define YYERRCODE	256


/* YYLLOC_DEFAULT -- Set CURRENT to span from RHS[1] to RHS[N].
   If N is 0, then set CURRENT to the empty location which ends
   the previous symbol: RHS[0] (always defined).  */

#define YYRHSLOC(Rhs, K) ((Rhs)[K])
#ifndef YYLLOC_DEFAULT
# define YYLLOC_DEFAULT(Current, Rhs, N)				\
    do									\
      if (YYID (N))                                                    \
	{								\
	  (Current).first_line   = YYRHSLOC (Rhs, 1).first_line;	\
	  (Current).first_column = YYRHSLOC (Rhs, 1).first_column;	\
	  (Current).last_line    = YYRHSLOC (Rhs, N).last_line;		\
	  (Current).last_column  = YYRHSLOC (Rhs, N).last_column;	\
	}								\
      else								\
	{								\
	  (Current).first_line   = (Current).last_line   =		\
	    YYRHSLOC (Rhs, 0).last_line;				\
	  (Current).first_column = (Current).last_column =		\
	    YYRHSLOC (Rhs, 0).last_column;				\
	}								\
    while (YYID (0))
#endif


/* YY_LOCATION_PRINT -- Print the location on the stream.
   This macro was not mandated originally: define only if we know
   we won't break user code: when these are the locations we know.  */

#ifndef YY_LOCATION_PRINT
# if YYLTYPE_IS_TRIVIAL
#  define YY_LOCATION_PRINT(File, Loc)			\
     fprintf (File, "%d.%d-%d.%d",			\
	      (Loc).first_line, (Loc).first_column,	\
	      (Loc).last_line,  (Loc).last_column)
# else
#  define YY_LOCATION_PRINT(File, Loc) ((void) 0)
# endif
#endif


/* YYLEX -- calling `yylex' with the right arguments.  */

#ifdef YYLEX_PARAM
# define YYLEX yylex (&yylval, YYLEX_PARAM)
#else
# define YYLEX yylex (&yylval)
#endif

/* Enable debugging if requested.  */
#if YYDEBUG

# ifndef YYFPRINTF
#  include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#  define YYFPRINTF fprintf
# endif

# define YYDPRINTF(Args)			\
do {						\
  if (yydebug)					\
    YYFPRINTF Args;				\
} while (YYID (0))

# define YY_SYMBOL_PRINT(Title, Type, Value, Location)			  \
do {									  \
  if (yydebug)								  \
    {									  \
      YYFPRINTF (stderr, "%s ", Title);					  \
      yy_symbol_print (stderr,						  \
		  Type, Value); \
      YYFPRINTF (stderr, "\n");						  \
    }									  \
} while (YYID (0))


/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

/*ARGSUSED*/
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_symbol_value_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep)
#else
static void
yy_symbol_value_print (yyoutput, yytype, yyvaluep)
    FILE *yyoutput;
    int yytype;
    YYSTYPE const * const yyvaluep;
#endif
{
  if (!yyvaluep)
    return;
# ifdef YYPRINT
  if (yytype < YYNTOKENS)
    YYPRINT (yyoutput, yytoknum[yytype], *yyvaluep);
# else
  YYUSE (yyoutput);
# endif
  switch (yytype)
    {
      default:
	break;
    }
}


/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_symbol_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep)
#else
static void
yy_symbol_print (yyoutput, yytype, yyvaluep)
    FILE *yyoutput;
    int yytype;
    YYSTYPE const * const yyvaluep;
#endif
{
  if (yytype < YYNTOKENS)
    YYFPRINTF (yyoutput, "token %s (", yytname[yytype]);
  else
    YYFPRINTF (yyoutput, "nterm %s (", yytname[yytype]);

  yy_symbol_value_print (yyoutput, yytype, yyvaluep);
  YYFPRINTF (yyoutput, ")");
}

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_stack_print (yytype_int16 *bottom, yytype_int16 *top)
#else
static void
yy_stack_print (bottom, top)
    yytype_int16 *bottom;
    yytype_int16 *top;
#endif
{
  YYFPRINTF (stderr, "Stack now");
  for (; bottom <= top; ++bottom)
    YYFPRINTF (stderr, " %d", *bottom);
  YYFPRINTF (stderr, "\n");
}

# define YY_STACK_PRINT(Bottom, Top)				\
do {								\
  if (yydebug)							\
    yy_stack_print ((Bottom), (Top));				\
} while (YYID (0))


/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_reduce_print (YYSTYPE *yyvsp, int yyrule)
#else
static void
yy_reduce_print (yyvsp, yyrule)
    YYSTYPE *yyvsp;
    int yyrule;
#endif
{
  int yynrhs = yyr2[yyrule];
  int yyi;
  unsigned long int yylno = yyrline[yyrule];
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %lu):\n",
	     yyrule - 1, yylno);
  /* The symbols being reduced.  */
  for (yyi = 0; yyi < yynrhs; yyi++)
    {
      fprintf (stderr, "   $%d = ", yyi + 1);
      yy_symbol_print (stderr, yyrhs[yyprhs[yyrule] + yyi],
		       &(yyvsp[(yyi + 1) - (yynrhs)])
		       		       );
      fprintf (stderr, "\n");
    }
}

# define YY_REDUCE_PRINT(Rule)		\
do {					\
  if (yydebug)				\
    yy_reduce_print (yyvsp, Rule); \
} while (YYID (0))

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int yydebug;
#else /* !YYDEBUG */
# define YYDPRINTF(Args)
# define YY_SYMBOL_PRINT(Title, Type, Value, Location)
# define YY_STACK_PRINT(Bottom, Top)
# define YY_REDUCE_PRINT(Rule)
#endif /* !YYDEBUG */


/* YYINITDEPTH -- initial size of the parser's stacks.  */
#ifndef	YYINITDEPTH
# define YYINITDEPTH 200
#endif

/* YYMAXDEPTH -- maximum size the stacks can grow to (effective only
   if the built-in stack extension method is used).

   Do not make this value too large; the results are undefined if
   YYSTACK_ALLOC_MAXIMUM < YYSTACK_BYTES (YYMAXDEPTH)
   evaluated with infinite-precision integer arithmetic.  */

#ifndef YYMAXDEPTH
# define YYMAXDEPTH 10000
#endif



#if YYERROR_VERBOSE

# ifndef yystrlen
#  if defined __GLIBC__ && defined _STRING_H
#   define yystrlen strlen
#  else
/* Return the length of YYSTR.  */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static YYSIZE_T
yystrlen (const char *yystr)
#else
static YYSIZE_T
yystrlen (yystr)
    const char *yystr;
#endif
{
  YYSIZE_T yylen;
  for (yylen = 0; yystr[yylen]; yylen++)
    continue;
  return yylen;
}
#  endif
# endif

# ifndef yystpcpy
#  if defined __GLIBC__ && defined _STRING_H && defined _GNU_SOURCE
#   define yystpcpy stpcpy
#  else
/* Copy YYSRC to YYDEST, returning the address of the terminating '\0' in
   YYDEST.  */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static char *
yystpcpy (char *yydest, const char *yysrc)
#else
static char *
yystpcpy (yydest, yysrc)
    char *yydest;
    const char *yysrc;
#endif
{
  char *yyd = yydest;
  const char *yys = yysrc;

  while ((*yyd++ = *yys++) != '\0')
    continue;

  return yyd - 1;
}
#  endif
# endif

# ifndef yytnamerr
/* Copy to YYRES the contents of YYSTR after stripping away unnecessary
   quotes and backslashes, so that it's suitable for yyerror.  The
   heuristic is that double-quoting is unnecessary unless the string
   contains an apostrophe, a comma, or backslash (other than
   backslash-backslash).  YYSTR is taken from yytname.  If YYRES is
   null, do not copy; instead, return the length of what the result
   would have been.  */
static YYSIZE_T
yytnamerr (char *yyres, const char *yystr)
{
  if (*yystr == '"')
    {
      YYSIZE_T yyn = 0;
      char const *yyp = yystr;

      for (;;)
	switch (*++yyp)
	  {
	  case '\'':
	  case ',':
	    goto do_not_strip_quotes;

	  case '\\':
	    if (*++yyp != '\\')
	      goto do_not_strip_quotes;
	    /* Fall through.  */
	  default:
	    if (yyres)
	      yyres[yyn] = *yyp;
	    yyn++;
	    break;

	  case '"':
	    if (yyres)
	      yyres[yyn] = '\0';
	    return yyn;
	  }
    do_not_strip_quotes: ;
    }

  if (! yyres)
    return yystrlen (yystr);

  return yystpcpy (yyres, yystr) - yyres;
}
# endif

/* Copy into YYRESULT an error message about the unexpected token
   YYCHAR while in state YYSTATE.  Return the number of bytes copied,
   including the terminating null byte.  If YYRESULT is null, do not
   copy anything; just return the number of bytes that would be
   copied.  As a special case, return 0 if an ordinary "syntax error"
   message will do.  Return YYSIZE_MAXIMUM if overflow occurs during
   size calculation.  */
static YYSIZE_T
yysyntax_error (char *yyresult, int yystate, int yychar)
{
  int yyn = yypact[yystate];

  if (! (YYPACT_NINF < yyn && yyn <= YYLAST))
    return 0;
  else
    {
      int yytype = YYTRANSLATE (yychar);
      YYSIZE_T yysize0 = yytnamerr (0, yytname[yytype]);
      YYSIZE_T yysize = yysize0;
      YYSIZE_T yysize1;
      int yysize_overflow = 0;
      enum { YYERROR_VERBOSE_ARGS_MAXIMUM = 5 };
      char const *yyarg[YYERROR_VERBOSE_ARGS_MAXIMUM];
      int yyx;

# if 0
      /* This is so xgettext sees the translatable formats that are
	 constructed on the fly.  */
      YY_("syntax error, unexpected %s");
      YY_("syntax error, unexpected %s, expecting %s");
      YY_("syntax error, unexpected %s, expecting %s or %s");
      YY_("syntax error, unexpected %s, expecting %s or %s or %s");
      YY_("syntax error, unexpected %s, expecting %s or %s or %s or %s");
# endif
      char *yyfmt;
      char const *yyf;
      static char const yyunexpected[] = "syntax error, unexpected %s";
      static char const yyexpecting[] = ", expecting %s";
      static char const yyor[] = " or %s";
      char yyformat[sizeof yyunexpected
		    + sizeof yyexpecting - 1
		    + ((YYERROR_VERBOSE_ARGS_MAXIMUM - 2)
		       * (sizeof yyor - 1))];
      char const *yyprefix = yyexpecting;

      /* Start YYX at -YYN if negative to avoid negative indexes in
	 YYCHECK.  */
      int yyxbegin = yyn < 0 ? -yyn : 0;

      /* Stay within bounds of both yycheck and yytname.  */
      int yychecklim = YYLAST - yyn + 1;
      int yyxend = yychecklim < YYNTOKENS ? yychecklim : YYNTOKENS;
      int yycount = 1;

      yyarg[0] = yytname[yytype];
      yyfmt = yystpcpy (yyformat, yyunexpected);

      for (yyx = yyxbegin; yyx < yyxend; ++yyx)
	if (yycheck[yyx + yyn] == yyx && yyx != YYTERROR)
	  {
	    if (yycount == YYERROR_VERBOSE_ARGS_MAXIMUM)
	      {
		yycount = 1;
		yysize = yysize0;
		yyformat[sizeof yyunexpected - 1] = '\0';
		break;
	      }
	    yyarg[yycount++] = yytname[yyx];
	    yysize1 = yysize + yytnamerr (0, yytname[yyx]);
	    yysize_overflow |= (yysize1 < yysize);
	    yysize = yysize1;
	    yyfmt = yystpcpy (yyfmt, yyprefix);
	    yyprefix = yyor;
	  }

      yyf = YY_(yyformat);
      yysize1 = yysize + yystrlen (yyf);
      yysize_overflow |= (yysize1 < yysize);
      yysize = yysize1;

      if (yysize_overflow)
	return YYSIZE_MAXIMUM;

      if (yyresult)
	{
	  /* Avoid sprintf, as that infringes on the user's name space.
	     Don't have undefined behavior even if the translation
	     produced a string with the wrong number of "%s"s.  */
	  char *yyp = yyresult;
	  int yyi = 0;
	  while ((*yyp = *yyf) != '\0')
	    {
	      if (*yyp == '%' && yyf[1] == 's' && yyi < yycount)
		{
		  yyp += yytnamerr (yyp, yyarg[yyi++]);
		  yyf += 2;
		}
	      else
		{
		  yyp++;
		  yyf++;
		}
	    }
	}
      return yysize;
    }
}
#endif /* YYERROR_VERBOSE */


/*-----------------------------------------------.
| Release the memory associated to this symbol.  |
`-----------------------------------------------*/

/*ARGSUSED*/
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yydestruct (const char *yymsg, int yytype, YYSTYPE *yyvaluep)
#else
static void
yydestruct (yymsg, yytype, yyvaluep)
    const char *yymsg;
    int yytype;
    YYSTYPE *yyvaluep;
#endif
{
  YYUSE (yyvaluep);

  if (!yymsg)
    yymsg = "Deleting";
  YY_SYMBOL_PRINT (yymsg, yytype, yyvaluep, yylocationp);

  switch (yytype)
    {

      default:
	break;
    }
}


/* Prevent warnings from -Wmissing-prototypes.  */

#ifdef YYPARSE_PARAM
#if defined __STDC__ || defined __cplusplus
int yyparse (void *YYPARSE_PARAM);
#else
int yyparse ();
#endif
#else /* ! YYPARSE_PARAM */
#if defined __STDC__ || defined __cplusplus
int yyparse (void);
#else
int yyparse ();
#endif
#endif /* ! YYPARSE_PARAM */






/*----------.
| yyparse.  |
`----------*/

#ifdef YYPARSE_PARAM
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
int
yyparse (void *YYPARSE_PARAM)
#else
int
yyparse (YYPARSE_PARAM)
    void *YYPARSE_PARAM;
#endif
#else /* ! YYPARSE_PARAM */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
int
yyparse (void)
#else
int
yyparse ()

#endif
#endif
{
  /* The look-ahead symbol.  */
int yychar;

/* The semantic value of the look-ahead symbol.  */
YYSTYPE yylval;

/* Number of syntax errors so far.  */
int yynerrs;

  int yystate;
  int yyn;
  int yyresult;
  /* Number of tokens to shift before error messages enabled.  */
  int yyerrstatus;
  /* Look-ahead token as an internal (translated) token number.  */
  int yytoken = 0;
#if YYERROR_VERBOSE
  /* Buffer for error messages, and its allocated size.  */
  char yymsgbuf[128];
  char *yymsg = yymsgbuf;
  YYSIZE_T yymsg_alloc = sizeof yymsgbuf;
#endif

  /* Three stacks and their tools:
     `yyss': related to states,
     `yyvs': related to semantic values,
     `yyls': related to locations.

     Refer to the stacks thru separate pointers, to allow yyoverflow
     to reallocate them elsewhere.  */

  /* The state stack.  */
  yytype_int16 yyssa[YYINITDEPTH];
  yytype_int16 *yyss = yyssa;
  yytype_int16 *yyssp;

  /* The semantic value stack.  */
  YYSTYPE yyvsa[YYINITDEPTH];
  YYSTYPE *yyvs = yyvsa;
  YYSTYPE *yyvsp;



#define YYPOPSTACK(N)   (yyvsp -= (N), yyssp -= (N))

  YYSIZE_T yystacksize = YYINITDEPTH;

  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE yyval;


  /* The number of symbols on the RHS of the reduced rule.
     Keep to zero when no symbol should be popped.  */
  int yylen = 0;

  YYDPRINTF ((stderr, "Starting parse\n"));

  yystate = 0;
  yyerrstatus = 0;
  yynerrs = 0;
  yychar = YYEMPTY;		/* Cause a token to be read.  */

  /* Initialize stack pointers.
     Waste one element of value and location stack
     so that they stay on the same level as the state stack.
     The wasted elements are never initialized.  */

  yyssp = yyss;
  yyvsp = yyvs;

  goto yysetstate;

/*------------------------------------------------------------.
| yynewstate -- Push a new state, which is found in yystate.  |
`------------------------------------------------------------*/
 yynewstate:
  /* In all cases, when you get here, the value and location stacks
     have just been pushed.  So pushing a state here evens the stacks.  */
  yyssp++;

 yysetstate:
  *yyssp = yystate;

  if (yyss + yystacksize - 1 <= yyssp)
    {
      /* Get the current used size of the three stacks, in elements.  */
      YYSIZE_T yysize = yyssp - yyss + 1;

#ifdef yyoverflow
      {
	/* Give user a chance to reallocate the stack.  Use copies of
	   these so that the &'s don't force the real ones into
	   memory.  */
	YYSTYPE *yyvs1 = yyvs;
	yytype_int16 *yyss1 = yyss;


	/* Each stack pointer address is followed by the size of the
	   data in use in that stack, in bytes.  This used to be a
	   conditional around just the two extra args, but that might
	   be undefined if yyoverflow is a macro.  */
	yyoverflow (YY_("memory exhausted"),
		    &yyss1, yysize * sizeof (*yyssp),
		    &yyvs1, yysize * sizeof (*yyvsp),

		    &yystacksize);

	yyss = yyss1;
	yyvs = yyvs1;
      }
#else /* no yyoverflow */
# ifndef YYSTACK_RELOCATE
      goto yyexhaustedlab;
# else
      /* Extend the stack our own way.  */
      if (YYMAXDEPTH <= yystacksize)
	goto yyexhaustedlab;
      yystacksize *= 2;
      if (YYMAXDEPTH < yystacksize)
	yystacksize = YYMAXDEPTH;

      {
	yytype_int16 *yyss1 = yyss;
	union yyalloc *yyptr =
	  (union yyalloc *) YYSTACK_ALLOC (YYSTACK_BYTES (yystacksize));
	if (! yyptr)
	  goto yyexhaustedlab;
	YYSTACK_RELOCATE (yyss);
	YYSTACK_RELOCATE (yyvs);

#  undef YYSTACK_RELOCATE
	if (yyss1 != yyssa)
	  YYSTACK_FREE (yyss1);
      }
# endif
#endif /* no yyoverflow */

      yyssp = yyss + yysize - 1;
      yyvsp = yyvs + yysize - 1;


      YYDPRINTF ((stderr, "Stack size increased to %lu\n",
		  (unsigned long int) yystacksize));

      if (yyss + yystacksize - 1 <= yyssp)
	YYABORT;
    }

  YYDPRINTF ((stderr, "Entering state %d\n", yystate));

  goto yybackup;

/*-----------.
| yybackup.  |
`-----------*/
yybackup:

  /* Do appropriate processing given the current state.  Read a
     look-ahead token if we need one and don't already have one.  */

  /* First try to decide what to do without reference to look-ahead token.  */
  yyn = yypact[yystate];
  if (yyn == YYPACT_NINF)
    goto yydefault;

  /* Not known => get a look-ahead token if don't already have one.  */

  /* YYCHAR is either YYEMPTY or YYEOF or a valid look-ahead symbol.  */
  if (yychar == YYEMPTY)
    {
      YYDPRINTF ((stderr, "Reading a token: "));
      yychar = YYLEX;
    }

  if (yychar <= YYEOF)
    {
      yychar = yytoken = YYEOF;
      YYDPRINTF ((stderr, "Now at end of input.\n"));
    }
  else
    {
      yytoken = YYTRANSLATE (yychar);
      YY_SYMBOL_PRINT ("Next token is", yytoken, &yylval, &yylloc);
    }

  /* If the proper action on seeing token YYTOKEN is to reduce or to
     detect an error, take that action.  */
  yyn += yytoken;
  if (yyn < 0 || YYLAST < yyn || yycheck[yyn] != yytoken)
    goto yydefault;
  yyn = yytable[yyn];
  if (yyn <= 0)
    {
      if (yyn == 0 || yyn == YYTABLE_NINF)
	goto yyerrlab;
      yyn = -yyn;
      goto yyreduce;
    }

  if (yyn == YYFINAL)
    YYACCEPT;

  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (yyerrstatus)
    yyerrstatus--;

  /* Shift the look-ahead token.  */
  YY_SYMBOL_PRINT ("Shifting", yytoken, &yylval, &yylloc);

  /* Discard the shifted token unless it is eof.  */
  if (yychar != YYEOF)
    yychar = YYEMPTY;

  yystate = yyn;
  *++yyvsp = yylval;

  goto yynewstate;


/*-----------------------------------------------------------.
| yydefault -- do the default action for the current state.  |
`-----------------------------------------------------------*/
yydefault:
  yyn = yydefact[yystate];
  if (yyn == 0)
    goto yyerrlab;
  goto yyreduce;


/*-----------------------------.
| yyreduce -- Do a reduction.  |
`-----------------------------*/
yyreduce:
  /* yyn is the number of a rule to reduce with.  */
  yylen = yyr2[yyn];

  /* If YYLEN is nonzero, implement the default value of the action:
     `$$ = $1'.

     Otherwise, the following line sets YYVAL to garbage.
     This behavior is undocumented and Bison
     users should not rely upon it.  Assigning to YYVAL
     unconditionally makes the parser a bit smaller, and it avoids a
     GCC warning that YYVAL may be used uninitialized.  */
  yyval = yyvsp[1-yylen];


  YY_REDUCE_PRINT (yyn);
  switch (yyn)
    {
        case 6:
#line 202 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_lone_end ); ;}
    break;

  case 7:
#line 203 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_case_outside ); ;}
    break;

  case 8:
#line 209 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(1) - (1)].stringp) != 0 )
            COMPILER->addLoad( *(yyvsp[(1) - (1)].stringp) );
      ;}
    break;

  case 9:
#line 214 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if( (yyvsp[(1) - (1)].fal_stat) != 0 )
            COMPILER->addFunction( (yyvsp[(1) - (1)].fal_stat) );
      ;}
    break;

  case 10:
#line 219 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(1) - (1)].fal_stat) != 0 )
            COMPILER->addClass( (yyvsp[(1) - (1)].fal_stat) );
      ;}
    break;

  case 11:
#line 224 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(1) - (1)].fal_stat) != 0 )
            COMPILER->addClass( (yyvsp[(1) - (1)].fal_stat) );
      ;}
    break;

  case 12:
#line 229 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if( (yyvsp[(1) - (1)].fal_stat) != 0 )
            COMPILER->addStatement( (yyvsp[(1) - (1)].fal_stat) );
      ;}
    break;

  case 16:
#line 240 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getContext() != 0 )
            COMPILER->raiseError(Falcon::e_toplevel_load );
         (yyval.stringp) = (yyvsp[(2) - (3)].stringp);
      ;}
    break;

  case 17:
#line 246 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getContext() != 0 )
            COMPILER->raiseError(Falcon::e_toplevel_load );
         (yyval.stringp) = (yyvsp[(2) - (3)].stringp);
      ;}
    break;

  case 18:
#line 252 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_load );
         (yyval.stringp) = 0;
      ;}
    break;

  case 19:
#line 259 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->checkLocalUndefined(); (yyval.fal_stat) = (yyvsp[(1) - (1)].fal_stat); ;}
    break;

  case 20:
#line 260 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = 0; ;}
    break;

  case 21:
#line 263 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_toplevel_func ); (yyval.fal_stat) = 0; ;}
    break;

  case 22:
#line 264 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_toplevel_obj ); (yyval.fal_stat) = 0; ;}
    break;

  case 23:
#line 265 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_toplevel_class ); (yyval.fal_stat) = 0; ;}
    break;

  case 24:
#line 266 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syntax ); (yyval.fal_stat) = 0;;}
    break;

  case 25:
#line 271 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defContext( true ); COMPILER->defRequired();
      COMPILER->defineVal( (yyvsp[(1) - (3)].fal_val) );
      COMPILER->addStatement( new Falcon::StmtAssignment( CURRENT_LINE, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) );
   ;}
    break;

  case 26:
#line 276 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(3) - (5)].fal_val) );
      COMPILER->addStatement( new Falcon::StmtAssignment( CURRENT_LINE, (yyvsp[(3) - (5)].fal_val), (yyvsp[(5) - (5)].fal_val) ) );
   ;}
    break;

  case 27:
#line 283 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = new Falcon::StmtAutoexpr( LINE, (yyvsp[(1) - (2)].fal_val) ); ;}
    break;

  case 49:
#line 309 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->defContext( false );  (yyval.fal_stat)=0; ;}
    break;

  case 50:
#line 311 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError( Falcon::e_syn_def ); ;}
    break;

  case 51:
#line 316 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAssignment( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 52:
#line 320 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (5)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtUnref( LINE, (yyvsp[(1) - (5)].fal_val) );
   ;}
    break;

  case 53:
#line 324 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
         (yyval.fal_stat) = new Falcon::StmtAssignment( LINE, (yyvsp[(1) - (4)].fal_val), new Falcon::Value( (yyvsp[(3) - (4)].fal_adecl) ) );
      ;}
    break;

  case 54:
#line 328 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
         (yyval.fal_stat) = new Falcon::StmtAssignment( LINE, (yyvsp[(1) - (4)].fal_val), new Falcon::Value( (yyvsp[(3) - (4)].fal_val) ) );
      ;}
    break;

  case 55:
#line 332 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(1) - (6)].fal_val) );
         (yyvsp[(3) - (6)].fal_adecl)->pushFront( (yyvsp[(1) - (6)].fal_val) );
         (yyval.fal_stat) = new Falcon::StmtAssignment( LINE, new Falcon::Value((yyvsp[(3) - (6)].fal_adecl)), (yyvsp[(5) - (6)].fal_val) );
      ;}
    break;

  case 56:
#line 337 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(1) - (6)].fal_val) );
         (yyvsp[(3) - (6)].fal_adecl)->pushFront( (yyvsp[(1) - (6)].fal_val) );
         (yyval.fal_stat) = new Falcon::StmtAssignment( LINE, new Falcon::Value((yyvsp[(3) - (6)].fal_adecl)), new Falcon::Value( (yyvsp[(5) - (6)].fal_adecl) ) );
      ;}
    break;

  case 68:
#line 361 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAutoAdd( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 69:
#line 368 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAutoSub( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 70:
#line 375 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAutoMul( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 71:
#line 382 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAutoDiv( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 72:
#line 389 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAutoMod( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 73:
#line 396 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAutoPow( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 74:
#line 403 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAutoBAND( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 75:
#line 410 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAutoBOR( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 76:
#line 417 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAutoBXOR( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 77:
#line 423 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAutoSHL( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 78:
#line 429 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAutoSHR( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 79:
#line 437 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtWhile *w = new Falcon::StmtWhile( LINE, (yyvsp[(1) - (1)].fal_val) );
         COMPILER->pushLoop( w );
         COMPILER->pushContext( w );
         COMPILER->pushContextSet( &w->children() );
      ;}
    break;

  case 80:
#line 444 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtWhile *w = static_cast<Falcon::StmtWhile *>(COMPILER->getContext());
         COMPILER->popLoop();
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = w;
      ;}
    break;

  case 81:
#line 451 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::StmtWhile *w = new Falcon::StmtWhile( LINE, 0 );
         COMPILER->pushLoop( w );
         COMPILER->pushContext( w );
         COMPILER->pushContextSet( &w->children() );
         COMPILER->addStatement( (yyvsp[(2) - (2)].fal_stat) );
      ;}
    break;

  case 82:
#line 459 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtWhile *w = static_cast<Falcon::StmtWhile *>(COMPILER->getContext());
         COMPILER->popLoop();
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = w;
      ;}
    break;

  case 83:
#line 467 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtWhile *w = new Falcon::StmtWhile( LINE, (yyvsp[(1) - (2)].fal_val) );
         if ( (yyvsp[(2) - (2)].fal_stat) != 0 )
            w->children().push_back( (yyvsp[(2) - (2)].fal_stat) );
         (yyval.fal_stat) = w;
      ;}
    break;

  case 84:
#line 475 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); ;}
    break;

  case 85:
#line 476 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = 0; ;}
    break;

  case 86:
#line 477 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_while ); (yyval.fal_val) = 0; ;}
    break;

  case 87:
#line 481 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); ;}
    break;

  case 88:
#line 482 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = 0; ;}
    break;

  case 89:
#line 483 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_while, "", CURRENT_LINE ); (yyval.fal_val) = 0; ;}
    break;

  case 90:
#line 487 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtIf *stmt = new Falcon::StmtIf( LINE, (yyvsp[(1) - (1)].fal_val) );
         COMPILER->pushContext( stmt );
         COMPILER->pushContextSet( &stmt->children() );
      ;}
    break;

  case 91:
#line 495 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtIf *stmt = static_cast<Falcon::StmtIf *>(COMPILER->getContext());
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = stmt;
      ;}
    break;

  case 92:
#line 502 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // use LINE as statement includes EOL
         Falcon::StmtIf *stmt = new Falcon::StmtIf( LINE, (yyvsp[(1) - (2)].fal_val) );
         if( (yyvsp[(2) - (2)].fal_stat) != 0 )
            stmt->children().push_back( (yyvsp[(2) - (2)].fal_stat) );
         (yyval.fal_stat) = stmt;
      ;}
    break;

  case 93:
#line 512 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); ;}
    break;

  case 94:
#line 513 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {  COMPILER->raiseError(Falcon::e_syn_if ); (yyval.fal_val) = 0; ;}
    break;

  case 95:
#line 517 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); ;}
    break;

  case 96:
#line 518 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {  COMPILER->raiseError(Falcon::e_syn_if, "", CURRENT_LINE ); (yyval.fal_val) = 0; ;}
    break;

  case 99:
#line 525 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtIf *stmt = static_cast<Falcon::StmtIf *>(COMPILER->getContext());
         COMPILER->popContextSet();
         COMPILER->pushContextSet( &stmt->elseChildren() );
      ;}
    break;

  case 102:
#line 535 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_else ); ;}
    break;

  case 103:
#line 540 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtIf *stmt = static_cast<Falcon::StmtIf *>(COMPILER->getContext());
         COMPILER->popContextSet();
         Falcon::StmtElif *elif = new Falcon::StmtElif( LINE, (yyvsp[(1) - (1)].fal_val) );
         stmt->elifChildren().push_back( elif );
         COMPILER->pushContextSet( &elif->children() );
      ;}
    break;

  case 105:
#line 552 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); ;}
    break;

  case 106:
#line 553 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_elif ); (yyval.fal_val) = 0; ;}
    break;

  case 108:
#line 558 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->addStatement( (yyvsp[(2) - (2)].fal_stat) );
   ;}
    break;

  case 109:
#line 565 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getLoop() == 0 ) {
            COMPILER->raiseError(Falcon::e_break_out );
            (yyval.fal_stat) = 0;
         }
         else
            (yyval.fal_stat) = new Falcon::StmtBreak( LINE );
      ;}
    break;

  case 110:
#line 574 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_break );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 111:
#line 582 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getLoop() == 0 ) {
            COMPILER->raiseError(Falcon::e_continue_out );
            (yyval.fal_stat) = 0;
         }
         else
            (yyval.fal_stat) = new Falcon::StmtContinue( LINE );
      ;}
    break;

  case 112:
#line 592 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getLoop() == 0 ) {
            COMPILER->raiseError(Falcon::e_continue_out );
            (yyval.fal_stat) = 0;
         }
         else
            (yyval.fal_stat) = new Falcon::StmtContinue( LINE, true );
      ;}
    break;

  case 113:
#line 601 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_continue );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 114:
#line 608 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtFor *f = static_cast<Falcon::StmtFor *>( (yyvsp[(1) - (1)].fal_stat) );
         COMPILER->pushLoop( f );
         COMPILER->pushContext( f );
         COMPILER->pushContextSet( &f->children() );
      ;}
    break;

  case 115:
#line 615 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtFor *f = static_cast<Falcon::StmtFor *>(COMPILER->getContext());
         COMPILER->popLoop();
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = f;
      ;}
    break;

  case 116:
#line 623 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(1) - (2)].fal_stat) != 0 )
         {
            Falcon::StmtFor *f = static_cast<Falcon::StmtFor *>((yyvsp[(1) - (2)].fal_stat));
            if ( (yyvsp[(2) - (2)].fal_stat) != 0 )
                f->children().push_back( (yyvsp[(2) - (2)].fal_stat) );
            (yyval.fal_stat) = f;
         }
         else
            delete (yyvsp[(2) - (2)].fal_stat);
      ;}
    break;

  case 117:
#line 638 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(2) - (7)].fal_val) );
         (yyval.fal_stat) = new Falcon::StmtFor( LINE, (yyvsp[(2) - (7)].fal_val), (yyvsp[(4) - (7)].fal_val), (yyvsp[(6) - (7)].fal_val) );
      ;}
    break;

  case 118:
#line 642 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(2) - (9)].fal_val) );
         (yyval.fal_stat) = new Falcon::StmtFor( LINE, (yyvsp[(2) - (9)].fal_val), (yyvsp[(4) - (9)].fal_val), (yyvsp[(6) - (9)].fal_val), (yyvsp[(8) - (9)].fal_val) );
      ;}
    break;

  case 119:
#line 647 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_for );
         (yyval.fal_stat) = new Falcon::StmtFor( LINE, 0, 0, 0 );
      ;}
    break;

  case 120:
#line 654 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(2) - (7)].fal_val) );
         (yyval.fal_stat) = new Falcon::StmtFor( CURRENT_LINE, (yyvsp[(2) - (7)].fal_val), (yyvsp[(4) - (7)].fal_val), (yyvsp[(6) - (7)].fal_val) );
      ;}
    break;

  case 121:
#line 658 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(2) - (9)].fal_val) );
         (yyval.fal_stat) = new Falcon::StmtFor( CURRENT_LINE, (yyvsp[(2) - (9)].fal_val), (yyvsp[(4) - (9)].fal_val), (yyvsp[(6) - (9)].fal_val), (yyvsp[(8) - (9)].fal_val) );
      ;}
    break;

  case 122:
#line 663 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_for, "", CURRENT_LINE );
         (yyval.fal_stat) = new Falcon::StmtFor( CURRENT_LINE, 0, 0, 0 );
      ;}
    break;

  case 123:
#line 672 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f;
         Falcon::ArrayDecl *decl = (yyvsp[(2) - (5)].fal_adecl);
         if ( decl->front() == decl->back() ) {
            f = new Falcon::StmtForin( LINE, (Falcon::Value *) decl->front(), (yyvsp[(4) - (5)].fal_val) );
            decl->deletor(0);
            delete decl;
         }
         else
            f = new Falcon::StmtForin( LINE, new Falcon::Value(decl), (yyvsp[(4) - (5)].fal_val) );
         COMPILER->pushLoop( f );
         COMPILER->pushContext( f );
         COMPILER->pushContextSet( &f->children() );
      ;}
    break;

  case 124:
#line 688 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         COMPILER->popLoop();
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = f;
      ;}
    break;

  case 125:
#line 696 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f;
         Falcon::ArrayDecl *decl = (yyvsp[(2) - (6)].fal_adecl);
         if ( decl->front() == decl->back() ) {
            f = new Falcon::StmtForin( CURRENT_LINE, (Falcon::Value *) decl->front(), (yyvsp[(4) - (6)].fal_val) );
            decl->deletor(0);
            delete decl;
         }
         else
            f = new Falcon::StmtForin( CURRENT_LINE, new Falcon::Value(decl), (yyvsp[(4) - (6)].fal_val) );
         if ( (yyvsp[(6) - (6)].fal_stat) != 0 )
             f->children().push_back( (yyvsp[(6) - (6)].fal_stat) );
      ;}
    break;

  case 126:
#line 710 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError( Falcon::e_syn_forin ); ;}
    break;

  case 129:
#line 719 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(1) - (1)].fal_stat) != 0 )
         {
            Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
            f->children().push_back( (yyvsp[(1) - (1)].fal_stat) );
         }
      ;}
    break;

  case 133:
#line 733 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getLoop());
         if ( f == 0 || f->type() != Falcon::Statement::t_forin )
         {
            COMPILER->raiseError( Falcon::e_syn_fordot );
            delete (yyvsp[(2) - (3)].fal_val);
            (yyval.fal_stat) = 0;
         }
         else {
            (yyval.fal_stat) = new Falcon::StmtFordot( LINE, (yyvsp[(2) - (3)].fal_val) );
         }
      ;}
    break;

  case 134:
#line 746 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_fordot );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 135:
#line 754 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = new Falcon::StmtSelfPrint( LINE, (yyvsp[(2) - (3)].fal_adecl) );
      ;}
    break;

  case 136:
#line 758 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_self_print );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 137:
#line 764 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyvsp[(2) - (3)].fal_adecl)->pushBack( new Falcon::Value( COMPILER->addString( "\n" ) ) );
         (yyval.fal_stat) = new Falcon::StmtSelfPrint( LINE, (yyvsp[(2) - (3)].fal_adecl) );
      ;}
    break;

  case 138:
#line 769 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::ArrayDecl *adecl = new Falcon::ArrayDecl();
         adecl->pushBack( new Falcon::Value( COMPILER->addString( "\n" ) ) );
         (yyval.fal_stat) = new Falcon::StmtSelfPrint( LINE, adecl );
      ;}
    break;

  case 139:
#line 776 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_self_print );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 140:
#line 785 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::ArrayDecl *adecl = new Falcon::ArrayDecl();
      adecl->pushBack( new Falcon::Value( (yyvsp[(1) - (1)].stringp) ) );
      (yyval.fal_stat) = new Falcon::StmtSelfPrint( LINE, adecl );
   ;}
    break;

  case 141:
#line 794 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         if( ! f->firstBlock().empty() )
         {
            COMPILER->raiseError( Falcon::e_already_forfirst );
         }
         COMPILER->pushContextSet( &f->firstBlock() );
		 // Push anyhow an empty item, that is needed for to check again for thio blosk
		 f->firstBlock().push_back( new Falcon::StmtNone( LINE ) );
      ;}
    break;

  case 142:
#line 806 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->popContextSet(); ;}
    break;

  case 143:
#line 808 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         if( ! f->firstBlock().empty() )
         {
            COMPILER->raiseError( Falcon::e_already_forfirst );
         }
         f->firstBlock().push_back( (yyvsp[(3) - (3)].fal_stat) );
      ;}
    break;

  case 144:
#line 816 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_forfirst ); ;}
    break;

  case 145:
#line 820 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         if( ! f->lastBlock().empty() )
         {
            COMPILER->raiseError( Falcon::e_already_forlast );
         }
		 // Push anyhow an empty item, that is needed for empty last blocks
		 f->lastBlock().push_back( new Falcon::StmtNone( LINE ) );
         COMPILER->pushContextSet( &f->lastBlock() );
      ;}
    break;

  case 146:
#line 832 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->popContextSet(); ;}
    break;

  case 147:
#line 833 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         if( ! f->lastBlock().empty() )
         {
            COMPILER->raiseError( Falcon::e_already_forlast );
         }
         f->lastBlock().push_back( (yyvsp[(3) - (3)].fal_stat) );
      ;}
    break;

  case 148:
#line 841 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_forlast ); ;}
    break;

  case 149:
#line 845 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         if( ! f->allBlock().empty() )
         {
            COMPILER->raiseError( Falcon::e_already_forall );
         }
		 // Push anyhow an empty item, that is needed for empty last blocks
		 f->allBlock().push_back( new Falcon::StmtNone( LINE ) );
         COMPILER->pushContextSet( &f->allBlock() );
      ;}
    break;

  case 150:
#line 857 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->popContextSet(); ;}
    break;

  case 151:
#line 859 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         if( ! f->allBlock().empty() )
         {
            COMPILER->raiseError( Falcon::e_already_forall );
         }
         f->allBlock().push_back( (yyvsp[(3) - (3)].fal_stat) );
      ;}
    break;

  case 152:
#line 867 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_forall ); ;}
    break;

  case 153:
#line 871 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::StmtSwitch *stmt = new Falcon::StmtSwitch( LINE, (yyvsp[(1) - (1)].fal_val) );
         COMPILER->pushContext( stmt );
         COMPILER->pushContextSet( &stmt->blocks() );
      ;}
    break;

  case 154:
#line 879 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = stmt;
      ;}
    break;

  case 155:
#line 888 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); ;}
    break;

  case 156:
#line 890 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_switch_decl );
         (yyval.fal_val) = 0;
      ;}
    break;

  case 159:
#line 899 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_switch_body ); ;}
    break;

  case 161:
#line 905 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      ;}
    break;

  case 163:
#line 915 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( CURRENT_LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      ;}
    break;

  case 164:
#line 923 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addStatement( (yyvsp[(5) - (5)].fal_stat) );
      ;}
    break;

  case 165:
#line 927 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_case_decl );

         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      ;}
    break;

  case 167:
#line 939 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_case_decl );

         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( CURRENT_LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      ;}
    break;

  case 168:
#line 949 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addStatement( (yyvsp[(5) - (5)].fal_stat) );
      ;}
    break;

  case 170:
#line 958 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();
         if ( ! stmt->defaultBlock().empty() )
         {
            COMPILER->raiseError(Falcon::e_switch_default, "", CURRENT_LINE );
         }
         COMPILER->pushContextSet( &stmt->defaultBlock() );
      ;}
    break;

  case 174:
#line 972 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_default_decl ); ;}
    break;

  case 176:
#line 976 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addStatement( (yyvsp[(2) - (2)].fal_stat) );
      ;}
    break;

  case 179:
#line 988 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         // todo: correct error
         if ( stmt->nilBlock() != -1 )
            COMPILER->raiseError(Falcon::e_switch_clash, "nil entry", CURRENT_LINE );
         stmt->nilBlock( stmt->currentBlock() );
      ;}
    break;

  case 180:
#line 997 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         // todo: correct error
         Falcon::Value *val = new Falcon::Value( (yyvsp[(1) - (1)].integer) );
         if ( ! stmt->addIntCase( val ) )
         {
            COMPILER->raiseError(Falcon::e_switch_clash, "", CURRENT_LINE );
            delete val;
         }
      ;}
    break;

  case 181:
#line 1009 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         Falcon::Value *val = new Falcon::Value( (yyvsp[(1) - (1)].stringp) );
         if ( ! stmt->addStringCase( val ) )
         {
            COMPILER->raiseError(Falcon::e_switch_clash, "", CURRENT_LINE );
            delete val;
         }
      ;}
    break;

  case 182:
#line 1020 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         Falcon::Value *val = new Falcon::Value( new Falcon::RangeDecl( new Falcon::Value( (yyvsp[(1) - (3)].integer) ), new Falcon::Value( (yyvsp[(3) - (3)].integer) ) ) );
         if ( ! stmt->addRangeCase( val ) )
         {
            COMPILER->raiseError(Falcon::e_switch_clash, "", CURRENT_LINE );
            delete val;
         }
      ;}
    break;

  case 183:
#line 1031 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         Falcon::Symbol *sym = COMPILER->searchLocalSymbol( (yyvsp[(1) - (1)].stringp) );
         if( sym == 0 )
            sym = COMPILER->addGlobalSymbol( (yyvsp[(1) - (1)].stringp) );
         Falcon::Value *val = new Falcon::Value( sym );

         if ( ! stmt->addSymbolCase( val ) )
         {
            COMPILER->raiseError(Falcon::e_switch_clash, "", CURRENT_LINE );
            delete val;
         }
      ;}
    break;

  case 184:
#line 1051 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::StmtSelect *stmt = new Falcon::StmtSelect( LINE, (yyvsp[(1) - (1)].fal_val) );
         COMPILER->pushContext( stmt );
         COMPILER->pushContextSet( &stmt->blocks() );
      ;}
    break;

  case 185:
#line 1059 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = stmt;
      ;}
    break;

  case 186:
#line 1068 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); ;}
    break;

  case 187:
#line 1070 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_select_decl );
         (yyval.fal_val) = 0;
      ;}
    break;

  case 190:
#line 1079 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_select_body ); ;}
    break;

  case 192:
#line 1085 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      ;}
    break;

  case 194:
#line 1095 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( CURRENT_LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      ;}
    break;

  case 195:
#line 1104 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addStatement( (yyvsp[(5) - (5)].fal_stat) );
      ;}
    break;

  case 196:
#line 1108 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_case_decl );

         Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      ;}
    break;

  case 198:
#line 1120 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_case_decl );

        Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( CURRENT_LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      ;}
    break;

  case 199:
#line 1130 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addStatement( (yyvsp[(5) - (5)].fal_stat) );
      ;}
    break;

  case 203:
#line 1144 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         // todo: correct error
         Falcon::Value *val = new Falcon::Value( (yyvsp[(1) - (1)].integer) );
         if ( ! stmt->addIntCase( val ) )
         {
            COMPILER->raiseError(Falcon::e_switch_clash, "", CURRENT_LINE );
            delete val;
         }
      ;}
    break;

  case 204:
#line 1156 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         Falcon::Symbol *sym = COMPILER->searchLocalSymbol( (yyvsp[(1) - (1)].stringp) );
         if( sym == 0 )
            sym = COMPILER->addGlobalSymbol( (yyvsp[(1) - (1)].stringp) );
         Falcon::Value *val = new Falcon::Value( sym );

         if ( ! stmt->addSymbolCase( val ) )
         {
            COMPILER->raiseError(Falcon::e_switch_clash, "", CURRENT_LINE );
            delete val;
         }
      ;}
    break;

  case 205:
#line 1177 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = new Falcon::StmtGive( LINE, (yyvsp[(4) - (5)].fal_val), (yyvsp[(2) - (5)].fal_adecl) );
      ;}
    break;

  case 206:
#line 1181 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = new Falcon::StmtGive( LINE, 0, (yyvsp[(2) - (4)].fal_adecl) );
         COMPILER->raiseError(Falcon::e_syn_give );
      ;}
    break;

  case 207:
#line 1185 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_give ); (yyval.fal_stat) = 0; ;}
    break;

  case 208:
#line 1193 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::StmtTry *t = new Falcon::StmtTry( CURRENT_LINE );
      if ( (yyvsp[(3) - (3)].fal_stat) != 0 )
          t->children().push_back( (yyvsp[(3) - (3)].fal_stat) );
      (yyval.fal_stat) = t;
   ;}
    break;

  case 209:
#line 1200 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtTry *t = new Falcon::StmtTry( LINE );
         COMPILER->pushContext( t );
         COMPILER->pushContextSet( &t->children() );
      ;}
    break;

  case 210:
#line 1210 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = COMPILER->getContext();
         COMPILER->popContext();
         COMPILER->popContextSet();
      ;}
    break;

  case 212:
#line 1219 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_try ); ;}
    break;

  case 218:
#line 1239 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->popContextSet(); // popping previous catch

         Falcon::StmtTry *t = static_cast<Falcon::StmtTry *>( COMPILER->getContext() );

         // if we have already a default, raise an error
         if( t->defaultHandler() != 0 )
         {
            COMPILER->raiseError(Falcon::e_catch_adef );
         }
         // but continue by pushing this new context
         Falcon::StmtCatchBlock *lst = new Falcon::StmtCatchBlock( LINE, 0 );
         t->defaultHandler( lst ); // will delete the previous one

         COMPILER->pushContextSet( &lst->children() );
      ;}
    break;

  case 219:
#line 1257 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->popContextSet(); // popping previous catch

         Falcon::StmtTry *t = static_cast<Falcon::StmtTry *>( COMPILER->getContext() );

         // if we have already a default, raise an error
         if( t->defaultHandler() != 0 )
         {
            COMPILER->raiseError(Falcon::e_catch_adef );
         }

         // but continue by pushing this new context
         COMPILER->defineVal( (yyvsp[(3) - (4)].fal_val) );
         Falcon::StmtCatchBlock *lst = new Falcon::StmtCatchBlock( LINE, (yyvsp[(3) - (4)].fal_val) );
         t->defaultHandler( lst ); // will delete the previous one

         COMPILER->pushContextSet( &lst->children() );
      ;}
    break;

  case 220:
#line 1277 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->popContextSet(); // popping previous catch

         Falcon::StmtTry *t = static_cast<Falcon::StmtTry *>( COMPILER->getContext() );
         Falcon::StmtCatchBlock *lst = new Falcon::StmtCatchBlock( LINE, 0 );
         COMPILER->pushContextSet( &lst->children() );
         t->addHandler( lst );
      ;}
    break;

  case 221:
#line 1287 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->popContextSet(); // popping previous catch

         Falcon::StmtTry *t = static_cast<Falcon::StmtTry *>( COMPILER->getContext() );
         COMPILER->defineVal( (yyvsp[(4) - (5)].fal_val) );
         Falcon::StmtCatchBlock *lst = new Falcon::StmtCatchBlock( LINE, (yyvsp[(4) - (5)].fal_val) );
         COMPILER->pushContextSet( &lst->children() );
         t->addHandler( lst );
      ;}
    break;

  case 222:
#line 1298 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->raiseError( Falcon::e_syn_catch );
   ;}
    break;

  case 225:
#line 1311 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtTry *stmt = static_cast<Falcon::StmtTry *>(COMPILER->getContext());
         Falcon::Value *val = new Falcon::Value( (yyvsp[(1) - (1)].integer) );

         if ( ! stmt->addIntCase( val ) )
         {
            COMPILER->raiseError(Falcon::e_catch_clash, "", CURRENT_LINE );
            delete val;
         }
      ;}
    break;

  case 226:
#line 1323 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtTry *stmt = static_cast<Falcon::StmtTry *>(COMPILER->getContext());
         Falcon::Symbol *sym = COMPILER->searchLocalSymbol( (yyvsp[(1) - (1)].stringp) );
         if( sym == 0 ) {
            sym = COMPILER->addGlobalSymbol( (yyvsp[(1) - (1)].stringp) );
         }
         Falcon::Value *val = new Falcon::Value( sym );

         if ( ! stmt->addSymbolCase( val ) )
         {
            COMPILER->raiseError(Falcon::e_catch_clash, "", CURRENT_LINE );
            delete val;
         }
      ;}
    break;

  case 227:
#line 1345 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = new Falcon::StmtRaise( LINE, (yyvsp[(2) - (3)].fal_val) ); ;}
    break;

  case 228:
#line 1346 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_raise ); (yyval.fal_stat) = 0; ;}
    break;

  case 229:
#line 1358 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = COMPILER->getContext();
         COMPILER->closeFunction();
      ;}
    break;

  case 230:
#line 1364 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addStatement( (yyvsp[(2) - (2)].fal_stat) );
         (yyval.fal_stat) = COMPILER->getContext();
         COMPILER->closeFunction();
      ;}
    break;

  case 232:
#line 1373 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); ;}
    break;

  case 233:
#line 1374 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_funcdecl, "", COMPILER->tempLine() );
      ;}
    break;

  case 234:
#line 1377 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_funcdecl ); ;}
    break;

  case 236:
#line 1382 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); ;}
    break;

  case 237:
#line 1383 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_funcdecl, "", COMPILER->tempLine() );
      ;}
    break;

  case 238:
#line 1390 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::FuncDef *def = new Falcon::FuncDef( 0 );
         // the SYMBOL which names the function goes in the old symbol table, while the parameters
         // will go in the new symbol table.

         // if we are in a class, I have to create the symbol classname.functionname
         Falcon::Statement *parent = COMPILER->getContext();
         Falcon::String *func_name;
         if ( parent != 0 && parent->type() == Falcon::Statement::t_class ) {
            Falcon::StmtClass *stmt_cls = static_cast< Falcon::StmtClass *>( parent );
            Falcon::String complete_name = stmt_cls->symbol()->name() + "." + *(yyvsp[(2) - (2)].stringp);
            func_name = COMPILER->addString( complete_name );
         }
         else
            func_name = (yyvsp[(2) - (2)].stringp);

         // find the global symbol for this.
         Falcon::Symbol *sym = COMPILER->searchGlobalSymbol( func_name );

         // Not defined?
         if( sym == 0 ) {
            sym = COMPILER->addGlobalSymbol( func_name );
         }
         else if ( sym->isFunction() || sym->isClass() ) {
            COMPILER->raiseError(Falcon::e_already_def, sym->name() );
         }

         // anyhow, also in case of error, destroys the previous information to allow a correct parsing
         // of the rest.
         sym->setFunction( def );

         // and eventually add it as a class property
         if ( parent != 0 && parent->type() == Falcon::Statement::t_class ) {
            Falcon::StmtClass *stmt_cls = static_cast< Falcon::StmtClass *>( parent );
            Falcon::ClassDef *cd = stmt_cls->symbol()->getClassDef();
            if ( cd->hasProperty( *(yyvsp[(2) - (2)].stringp) ) ) {
               COMPILER->raiseError(Falcon::e_prop_adef, *(yyvsp[(2) - (2)].stringp) );
            }
            else {
                cd->addProperty( (yyvsp[(2) - (2)].stringp), new Falcon::VarDef( sym ) );
            }
         }

         Falcon::StmtFunction *func = new Falcon::StmtFunction( COMPILER->lexer()->line(), sym );
         // prepare the statement allocation context
         COMPILER->pushContext( func );
         COMPILER->pushFunctionContext( func );
         COMPILER->pushContextSet( &func->statements() );
         COMPILER->pushFunction( def );
      ;}
    break;

  case 242:
#line 1451 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Symbol *sym = COMPILER->searchLocalSymbol( (yyvsp[(1) - (1)].stringp) );
         if ( sym != 0 ) {
            COMPILER->raiseError(Falcon::e_already_def, sym->name() );
         }
         else {
            Falcon::FuncDef *func = COMPILER->getFunction();
            Falcon::Symbol *sym = new Falcon::Symbol( COMPILER->module(), (yyvsp[(1) - (1)].stringp) );
            COMPILER->module()->addSymbol( sym );
            func->addParameter( sym );
         }
      ;}
    break;

  case 244:
#line 1468 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());
         COMPILER->pushContextSet( &func->staticBlock() );
         COMPILER->staticPrefix( &func->symbol()->name() );
      ;}
    break;

  case 245:
#line 1474 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->popContextSet();
         COMPILER->staticPrefix(0);
      ;}
    break;

  case 246:
#line 1479 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());
         COMPILER->pushContextSet( &func->staticBlock() );
         COMPILER->staticPrefix( &func->symbol()->name() );
      ;}
    break;

  case 247:
#line 1485 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addStatement( (yyvsp[(3) - (3)].fal_stat) );
         COMPILER->popContextSet();
         COMPILER->staticPrefix(0);
      ;}
    break;

  case 249:
#line 1494 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_static ); ;}
    break;

  case 251:
#line 1499 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_static, "", CURRENT_LINE ); ;}
    break;

  case 252:
#line 1509 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = new Falcon::StmtLaunch( LINE, (yyvsp[(2) - (3)].fal_val) );
      ;}
    break;

  case 253:
#line 1512 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_launch ); (yyval.fal_stat) = 0; ;}
    break;

  case 254:
#line 1521 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getFunction() == 0 )
            COMPILER->raiseError(Falcon::e_pass_outside );
         else
            (yyval.fal_stat) = new Falcon::StmtPass( LINE, (yyvsp[(2) - (3)].fal_val) );
      ;}
    break;

  case 255:
#line 1528 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // define the expression anyhow so we don't have fake errors below
         COMPILER->defineVal( (yyvsp[(4) - (5)].fal_val) );

         if ( COMPILER->getFunction() == 0 )
            COMPILER->raiseError(Falcon::e_pass_outside );
         else
            (yyval.fal_stat) = new Falcon::StmtPass( LINE, (yyvsp[(2) - (5)].fal_val), (yyvsp[(4) - (5)].fal_val) );
      ;}
    break;

  case 256:
#line 1538 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         delete (yyvsp[(2) - (5)].fal_val);
         COMPILER->raiseError(Falcon::e_syn_pass_in );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 257:
#line 1544 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_pass );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 258:
#line 1556 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // TODO: evalute const expressions on the fly.
         Falcon::Value *val = (yyvsp[(4) - (5)].fal_val); //COMPILER->exprSimplify( $4 );
         // will raise an error in case the expression is not atomic.
         COMPILER->addConstant( *(yyvsp[(2) - (5)].stringp), val, LINE );
         // we don't need the expression anymore
         // no other action:
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 259:
#line 1566 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_inv_const_val );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 260:
#line 1571 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_const );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 261:
#line 1583 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->sourceTree()->isExportAll() )
            COMPILER->raiseError(Falcon::e_export_all );
         else
            COMPILER->sourceTree()->setExportAll();
         // no effect
         (yyval.fal_stat)=0;
      ;}
    break;

  case 262:
#line 1592 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->sourceTree()->isExportAll() )
            COMPILER->raiseError(Falcon::e_export_all );
         // no effect
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 263:
#line 1599 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_export );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 264:
#line 1607 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Symbol *sym = COMPILER->addGlobalSymbol( (yyvsp[(1) - (1)].stringp) );
         sym->exported(true);
      ;}
    break;

  case 265:
#line 1612 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Symbol *sym = COMPILER->addGlobalSymbol( (yyvsp[(3) - (3)].stringp) );
         sym->exported(true);
      ;}
    break;

  case 266:
#line 1626 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // no other action:
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 267:
#line 1633 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // no other action:
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 269:
#line 1641 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_attributes ); ;}
    break;

  case 271:
#line 1645 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_attributes, "", CURRENT_LINE ); ;}
    break;

  case 273:
#line 1651 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addAttribute( (yyvsp[(1) - (1)].stringp) );
         ;}
    break;

  case 274:
#line 1655 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addAttribute( (yyvsp[(3) - (3)].stringp) );
         ;}
    break;

  case 277:
#line 1664 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->raiseError(Falcon::e_inv_attrib );
   ;}
    break;

  case 278:
#line 1675 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::ClassDef *def = new Falcon::ClassDef( 0, 0 );
         // the SYMBOL which names the function goes in the old symbol table, while the parameters
         // will go in the new symbol table.

         // find the global symbol for this.
         Falcon::Symbol *sym = COMPILER->searchGlobalSymbol( (yyvsp[(2) - (2)].stringp) );

         // Not defined?
         if( sym == 0 ) {
            sym = COMPILER->addGlobalSymbol( (yyvsp[(2) - (2)].stringp) );
         }
         else if ( sym->isFunction() || sym->isClass() ) {
            COMPILER->raiseError(Falcon::e_already_def,  sym->name() );
         }

         // anyhow, also in case of error, destroys the previous information to allow a correct parsing
         // of the rest.
         sym->setClass( def );

         Falcon::StmtClass *cls = new Falcon::StmtClass( COMPILER->lexer()->line(), sym );
         // prepare the statement allocation context
         COMPILER->pushContext( cls );

         // We don't have a context set here
         COMPILER->pushFunction( def );
      ;}
    break;

  case 279:
#line 1709 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = COMPILER->getContext();
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>((yyval.fal_stat));

         // if the class has no constructor, create one in case of inheritance.
         if( cls->ctorFunction() == 0  )
         {
            Falcon::ClassDef *cd = cls->symbol()->getClassDef();
            if ( cd->inheritance().size() != 0 )
            {
               Falcon::StmtFunction *func = func = COMPILER->buildCtorFor( cls );
               // COMPILER->addStatement( func ); should be done in buildCtorFor
               // cls->ctorFunction( func ); idem
            }
         }

         COMPILER->popContext();
         //We didn't pushed a context set
         COMPILER->popFunction();
      ;}
    break;

  case 281:
#line 1737 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_class );
      ;}
    break;

  case 284:
#line 1745 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); ;}
    break;

  case 285:
#line 1746 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_class, "", COMPILER->tempLine() );
      ;}
    break;

  case 290:
#line 1763 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
         // creates or find the symbol.
         Falcon::Symbol *sym = COMPILER->addGlobalSymbol((yyvsp[(1) - (2)].stringp));
         Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();
         Falcon::InheritDef *idef = new Falcon::InheritDef(sym);

         if ( clsdef->addInheritance( idef ) )
         {
            if ( (yyvsp[(2) - (2)].fal_adecl) != 0 )
            {
               // save the carried
               Falcon::ListElement *iter = (yyvsp[(2) - (2)].fal_adecl)->begin();
               while( iter != 0 )
               {
                  Falcon::Value *val = (Falcon::Value *) iter->data();
                  idef->addParameter( val->genVarDefSym() );
                  iter = iter->next();
               }

               // dispose of the carrier
               delete (yyvsp[(2) - (2)].fal_adecl);
            }
         }
         else {
            COMPILER->raiseError(Falcon::e_prop_adef );
            delete idef;
         }
      ;}
    break;

  case 291:
#line 1796 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_adecl) = 0; ;}
    break;

  case 292:
#line 1801 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      (yyval.fal_adecl) = (yyvsp[(3) - (5)].fal_adecl);
   ;}
    break;

  case 293:
#line 1807 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_adecl) = new Falcon::ArrayDecl(); (yyval.fal_adecl)->pushBack( (yyvsp[(1) - (1)].fal_val) ); ;}
    break;

  case 294:
#line 1808 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyvsp[(1) - (3)].fal_adecl)->pushBack( (yyvsp[(3) - (3)].fal_val) ); (yyval.fal_adecl) = (yyvsp[(1) - (3)].fal_adecl); ;}
    break;

  case 296:
#line 1814 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // the symbol must be a parameter, or we raise an error
         Falcon::Symbol *sym = COMPILER->searchLocalSymbol( (yyvsp[(1) - (1)].stringp) );
         if ( sym == 0 || sym->type() != Falcon::Symbol::tparam ) {
            sym = COMPILER->addGlobalSymbol( (yyvsp[(1) - (1)].stringp) );
         }
         (yyval.fal_val) = new Falcon::Value( sym );
      ;}
    break;

  case 297:
#line 1822 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setSelf(); ;}
    break;

  case 301:
#line 1832 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->addFunction( (yyvsp[(1) - (1)].fal_stat) );
   ;}
    break;

  case 302:
#line 1835 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
      if ( cls->initGiven() ) {
         COMPILER->raiseError(Falcon::e_prop_pinit );
      }
      COMPILER->checkLocalUndefined();
      // have we got a complex property statement?
      if ( (yyvsp[(1) - (1)].fal_stat) != 0 )
      {
         // as we didn't push the class context set, we have to do it by ourselves
         // see if the class has already a constructor.
         Falcon::StmtFunction *ctor_stmt = cls->ctorFunction();
         if ( ctor_stmt == 0 ) {
            ctor_stmt = COMPILER->buildCtorFor( cls );
         }

         ctor_stmt->statements().push_back( (yyvsp[(1) - (1)].fal_stat) );  // this goes directly in the auto constructor.
      }
   ;}
    break;

  case 304:
#line 1858 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
         if( cls->initGiven() ) {
            COMPILER->raiseError(Falcon::e_init_given );
         }
         else
         {
            cls->initGiven( true );
            Falcon::StmtFunction *func = cls->ctorFunction();
            if ( func == 0 ) {
               func = COMPILER->buildCtorFor( cls );
            }

            // prepare the statement allocation context
            COMPILER->pushContext( func );
            COMPILER->pushContextSet( &func->statements() );
            COMPILER->pushFunction( func->symbol()->getFuncDef() );
         }
      ;}
    break;

  case 305:
#line 1882 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());

         COMPILER->popContext();
         COMPILER->popContextSet();
         COMPILER->popFunction();
      ;}
    break;

  case 306:
#line 1893 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->checkLocalUndefined();
      Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
      Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();
      Falcon::VarDef *def = (yyvsp[(4) - (5)].fal_val)->genVarDef();

      if ( def != 0 ) {
         Falcon::String prop_name = cls->symbol()->name() + "." + *(yyvsp[(2) - (5)].stringp);
         Falcon::Symbol *sym = COMPILER->addGlobalVar( COMPILER->addString(prop_name), def );
         if( clsdef->hasProperty( *(yyvsp[(2) - (5)].stringp) ) )
            COMPILER->raiseError(Falcon::e_prop_adef, *(yyvsp[(2) - (5)].stringp) );
         else
            clsdef->addProperty( (yyvsp[(2) - (5)].stringp), new Falcon::VarDef( Falcon::VarDef::t_reference, sym) );
      }
      else {
         COMPILER->raiseError(Falcon::e_static_const );
      }
      delete (yyvsp[(4) - (5)].fal_val); // the expression is not needed anymore
      (yyval.fal_stat) = 0; // we don't add any statement
   ;}
    break;

  case 307:
#line 1915 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->checkLocalUndefined();
      Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
      Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();
      Falcon::VarDef *def = (yyvsp[(3) - (4)].fal_val)->genVarDef();

      if ( def != 0 ) {
         if( clsdef->hasProperty( *(yyvsp[(1) - (4)].stringp) ) )
            COMPILER->raiseError(Falcon::e_prop_adef, *(yyvsp[(1) - (4)].stringp) );
         else
            clsdef->addProperty( (yyvsp[(1) - (4)].stringp), def );
         delete (yyvsp[(3) - (4)].fal_val); // the expression is not needed anymore
         (yyval.fal_stat) = 0; // we don't add any statement
      }
      else {
         // create anyhow a nil property
          if( clsdef->hasProperty( *(yyvsp[(1) - (4)].stringp) ) )
            COMPILER->raiseError(Falcon::e_prop_adef, *(yyvsp[(1) - (4)].stringp) );
         else
            clsdef->addProperty( (yyvsp[(1) - (4)].stringp), new Falcon::VarDef() );
         // but also prepare a statement to be executed by the auto-constructor.
         (yyval.fal_stat) = new Falcon::StmtVarDef( LINE, (yyvsp[(1) - (4)].stringp), (yyvsp[(3) - (4)].fal_val) );
      }
   ;}
    break;

  case 310:
#line 1945 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->raiseError(Falcon::e_syn_hasdef );
   ;}
    break;

  case 311:
#line 1952 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
         Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();

         // The symbolmay be undefined or defined; it's not our task to define it here.
         clsdef->has().pushBack( COMPILER->addGlobalSymbol( (yyvsp[(1) - (1)].stringp) ) );
      ;}
    break;

  case 312:
#line 1960 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
         Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();
         clsdef->hasnt().pushBack( COMPILER->addGlobalSymbol( (yyvsp[(2) - (2)].stringp) ) );
      ;}
    break;

  case 313:
#line 1966 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
         Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();
         clsdef->has().pushBack( COMPILER->addGlobalSymbol( (yyvsp[(3) - (3)].stringp) ) );
      ;}
    break;

  case 314:
#line 1972 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
         Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();
         clsdef->hasnt().pushBack( COMPILER->addGlobalSymbol( (yyvsp[(4) - (4)].stringp) ) );
      ;}
    break;

  case 315:
#line 1985 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::ClassDef *def = new Falcon::ClassDef( 0, 0 );
         // the SYMBOL which names the function goes in the old symbol table, while the parameters
         // will go in the new symbol table.

         // we create a special symbol for the class.
         Falcon::String cl_name = "%";
         cl_name += *(yyvsp[(2) - (2)].stringp);
         Falcon::Symbol *clsym = COMPILER->addGlobalSymbol( COMPILER->addString( cl_name ) );
         clsym->setClass( def );

         // find the global symbol for this.
         Falcon::Symbol *sym = COMPILER->searchGlobalSymbol( (yyvsp[(2) - (2)].stringp) );

         // Not defined?
         if( sym == 0 ) {
            sym = COMPILER->addGlobalSymbol( (yyvsp[(2) - (2)].stringp) );
         }
         else if ( sym->isFunction() || sym->isClass() ) {
            COMPILER->raiseError(Falcon::e_already_def,  sym->name() );
         }

         // anyhow, also in case of error, destroys the previous information to allow a correct parsing
         // of the rest.
         sym->setInstance( clsym );

         Falcon::StmtClass *cls = new Falcon::StmtClass( COMPILER->lexer()->line(), clsym );
         // prepare the statement allocation context
         COMPILER->pushContext( cls );

         //Statements here goes in the auto constructor.
         //COMPILER->pushContextSet( &cls->autoCtor() );
         COMPILER->pushFunction( def );
      ;}
    break;

  case 316:
#line 2025 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = COMPILER->getContext();
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>((yyval.fal_stat));

         // if the class has no constructor, create one in case of inheritance.
         if( cls->ctorFunction() == 0  )
         {
            Falcon::ClassDef *cd = cls->symbol()->getClassDef();
            if ( cd->inheritance().size() != 0 )
            {
               Falcon::StmtFunction *func = func = COMPILER->buildCtorFor( cls );
               // COMPILER->addStatement( func ); should be done in buildCtorFor
               // cls->ctorFunction( func ); idem
            }
         }

         COMPILER->popContext();
         //COMPILER->popContextSet();
         COMPILER->popFunction();
      ;}
    break;

  case 318:
#line 2050 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_object );
      ;}
    break;

  case 322:
#line 2062 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->addFunction( (yyvsp[(1) - (1)].fal_stat) );
   ;}
    break;

  case 323:
#line 2065 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
      if ( cls->initGiven() ) {
         COMPILER->raiseError(Falcon::e_prop_pinit );
      }
      COMPILER->checkLocalUndefined();
      // have we got a complex property statement?
      if ( (yyvsp[(1) - (1)].fal_stat) != 0 )
      {
         // as we didn't push the class context set, we have to do it by ourselves
         // see if the class has already a constructor.
         Falcon::StmtFunction *ctor_stmt = cls->ctorFunction();
         if ( ctor_stmt == 0 ) {
            ctor_stmt = COMPILER->buildCtorFor( cls );
         }

         ctor_stmt->statements().push_back( (yyvsp[(1) - (1)].fal_stat) );  // this goes directly in the auto constructor.
      }
   ;}
    break;

  case 325:
#line 2093 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtGlobal *glob = new Falcon::StmtGlobal( CURRENT_LINE );
         COMPILER->pushContext( glob );
      ;}
    break;

  case 326:
#line 2098 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // raise an error if we are not in a local context
         if ( ! COMPILER->isLocalContext() )
         {
            COMPILER->raiseError(Falcon::e_global_notin_func, "", LINE );
         }
         (yyval.fal_stat) = COMPILER->getContext();
         COMPILER->popContext();
      ;}
    break;

  case 329:
#line 2113 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_global );
      ;}
    break;

  case 330:
#line 2120 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // we create (or retrieve) a globalized symbol
         Falcon::Symbol *sym = COMPILER->globalize( (yyvsp[(1) - (1)].stringp) );

         // then we add the symbol to the global statement (it's just for symbolic asm generation).
         Falcon::StmtGlobal *glob = static_cast<Falcon::StmtGlobal *>( COMPILER->getContext() );
         glob->addSymbol( sym );
      ;}
    break;

  case 331:
#line 2135 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = new Falcon::StmtReturn(LINE, 0); ;}
    break;

  case 332:
#line 2136 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = new Falcon::StmtReturn( LINE, (yyvsp[(2) - (3)].fal_val) ); ;}
    break;

  case 333:
#line 2137 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_return ); (yyval.fal_stat) = 0; ;}
    break;

  case 334:
#line 2147 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); ;}
    break;

  case 335:
#line 2148 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].integer) ); ;}
    break;

  case 336:
#line 2149 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].numeric) ); ;}
    break;

  case 337:
#line 2150 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].stringp) ); ;}
    break;

  case 338:
#line 2155 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Value *val;
         Falcon::Symbol *sym = COMPILER->searchLocalSymbol( (yyvsp[(1) - (1)].stringp) );
         if( sym == 0 ) {
            val = new Falcon::Value();
            val->setSymdef( (yyvsp[(1) - (1)].stringp) );
            // warning: the symbol is still undefined.
            COMPILER->addSymdef( val );
         }
         else {
            val = new Falcon::Value( sym );
         }
         (yyval.fal_val) = val;
     ;}
    break;

  case 340:
#line 2173 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setSelf(); ;}
    break;

  case 341:
#line 2174 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setSender(); ;}
    break;

  case 343:
#line 2186 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_access, (yyvsp[(1) - (2)].fal_val), (yyvsp[(2) - (2)].fal_val) );
         (yyval.fal_val) = new Falcon::Value( exp );
      ;}
    break;

  case 344:
#line 2191 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_access, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
         (yyval.fal_val) = new Falcon::Value( exp );
      ;}
    break;

  case 345:
#line 2196 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_byte_access, (yyvsp[(1) - (5)].fal_val), (yyvsp[(4) - (5)].fal_val) );
         (yyval.fal_val) = new Falcon::Value( exp );
      ;}
    break;

  case 346:
#line 2202 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_obj_access, (yyvsp[(1) - (3)].fal_val), new Falcon::Value( (yyvsp[(3) - (3)].stringp) ) );
         (yyval.fal_val) = new Falcon::Value( exp );
      ;}
    break;

  case 349:
#line 2213 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_plus, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 350:
#line 2214 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_neg, (yyvsp[(2) - (2)].fal_val) ) ); ;}
    break;

  case 351:
#line 2215 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_minus, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 352:
#line 2216 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_times, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 353:
#line 2217 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_divide, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 354:
#line 2218 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_modulo, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 355:
#line 2219 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_power, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 356:
#line 2220 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_and, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 357:
#line 2221 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_or, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 358:
#line 2222 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_xor, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 359:
#line 2223 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_shift_left, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 360:
#line 2224 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_shift_right, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 361:
#line 2225 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_not, (yyvsp[(2) - (2)].fal_val) ) ); ;}
    break;

  case 362:
#line 2226 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->defineVal( (yyvsp[(2) - (4)].fal_val) ); (yyval.fal_val) =
        new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_let, (yyvsp[(2) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) ); ;}
    break;

  case 363:
#line 2228 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->defineVal( (yyvsp[(2) - (4)].fal_val) ); (yyval.fal_val) =
        new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_let, (yyvsp[(2) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) ); ;}
    break;

  case 364:
#line 2230 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_neq, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 365:
#line 2231 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_post_inc, (yyvsp[(1) - (2)].fal_val) ) ); ;}
    break;

  case 366:
#line 2232 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_pre_inc, (yyvsp[(2) - (2)].fal_val) ) ); ;}
    break;

  case 367:
#line 2233 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_post_dec, (yyvsp[(1) - (2)].fal_val) ) ); ;}
    break;

  case 368:
#line 2234 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_pre_dec, (yyvsp[(2) - (2)].fal_val) ) ); ;}
    break;

  case 369:
#line 2235 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_eq, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 370:
#line 2236 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_eq, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 371:
#line 2237 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_eq, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 372:
#line 2238 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_gt, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 373:
#line 2239 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_lt, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 374:
#line 2240 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_ge, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 375:
#line 2241 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_le, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 376:
#line 2242 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_and, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 377:
#line 2243 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_or, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 378:
#line 2244 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_not, (yyvsp[(2) - (2)].fal_val) ) ); ;}
    break;

  case 379:
#line 2245 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_has, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 380:
#line 2246 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_hasnt, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 381:
#line 2247 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_in, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 382:
#line 2248 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_notin, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 383:
#line 2249 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_provides, (yyvsp[(1) - (3)].fal_val), new Falcon::Value( (yyvsp[(3) - (3)].stringp) ) ) ); ;}
    break;

  case 384:
#line 2250 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (2)].fal_val) ); ;}
    break;

  case 385:
#line 2251 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_strexpand, (yyvsp[(2) - (2)].fal_val) ) ); ;}
    break;

  case 386:
#line 2252 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_indirect, (yyvsp[(2) - (2)].fal_val) ) ); ;}
    break;

  case 389:
#line 2255 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_obj_access, (yyvsp[(1) - (3)].fal_val), new Falcon::Value( (yyvsp[(3) - (3)].stringp) ) );
         (yyval.fal_val) = new Falcon::Value( exp );
      ;}
    break;

  case 390:
#line 2259 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_access, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
         (yyval.fal_val) = new Falcon::Value( exp );
      ;}
    break;

  case 391:
#line 2263 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_byte_access, (yyvsp[(1) - (5)].fal_val), (yyvsp[(4) - (5)].fal_val) );
         (yyval.fal_val) = new Falcon::Value( exp );
      ;}
    break;

  case 392:
#line 2267 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_access, (yyvsp[(1) - (2)].fal_val), (yyvsp[(2) - (2)].fal_val) );
         (yyval.fal_val) = new Falcon::Value( exp );
      ;}
    break;

  case 397:
#line 2275 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(3) - (5)].fal_val); ;}
    break;

  case 398:
#line 2280 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( new Falcon::Value( (Falcon::int64) 0 ) ) );
      ;}
    break;

  case 399:
#line 2283 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( (yyvsp[(2) - (4)].fal_val) ) );
      ;}
    break;

  case 400:
#line 2286 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( new Falcon::Value( (Falcon::int64) 0 ), (yyvsp[(3) - (4)].fal_val) ) );
      ;}
    break;

  case 401:
#line 2289 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( (yyvsp[(2) - (5)].fal_val), (yyvsp[(4) - (5)].fal_val) ) );
      ;}
    break;

  case 402:
#line 2296 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_funcall,
                                      (yyvsp[(1) - (6)].fal_val), new Falcon::Value( (yyvsp[(4) - (6)].fal_adecl) ) ) );
      ;}
    break;

  case 403:
#line 2302 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_funcall, (yyvsp[(1) - (4)].fal_val), 0 ) );
      ;}
    break;

  case 404:
#line 2306 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); ;}
    break;

  case 405:
#line 2307 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         delete (yyvsp[(4) - (8)].fal_adecl);
         COMPILER->raiseError(Falcon::e_syn_funcall, "", COMPILER->tempLine() );
         (yyval.fal_val) = new Falcon::Value;
      ;}
    break;

  case 410:
#line 2326 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::FuncDef *def = new Falcon::FuncDef( 0 );
         // set the def as a lambda.
         COMPILER->incLambdaCount();
         int id = COMPILER->lambdaCount();
         // find the global symbol for this.
         char buf[48];
         sprintf( buf, "_lambda#_id_%d", id );
         Falcon::String *name = COMPILER->addString( buf );
         Falcon::Symbol *sym = COMPILER->searchGlobalSymbol( name );

         // Not defined?
         fassert( sym == 0 );
         sym = COMPILER->addGlobalSymbol( name );

         // anyhow, also in case of error, destroys the previous information to allow a correct parsing
         // of the rest.
         sym->setFunction( def );

         Falcon::StmtFunction *func = new Falcon::StmtFunction( COMPILER->lexer()->line(), sym );
         COMPILER->addFunction( func );
         func->setLambda( id );
         // prepare the statement allocation context
         COMPILER->pushContext( func );
         COMPILER->pushFunctionContext( func );
         COMPILER->pushContextSet( &func->statements() );
         COMPILER->pushFunction( def );
      ;}
    break;

  case 411:
#line 2359 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());
            (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_lambda ,
               new Falcon::Value( func->symbol() ) ) );
            COMPILER->closeFunction();
         ;}
    break;

  case 413:
#line 2369 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); ;}
    break;

  case 414:
#line 2370 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_funcdecl );
      ;}
    break;

  case 415:
#line 2374 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_funcdecl );
      ;}
    break;

  case 416:
#line 2382 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::FuncDef *def = new Falcon::FuncDef( 0 );
         // set the def as a lambda.
         COMPILER->incLambdaCount();
         int id = COMPILER->lambdaCount();
         // find the global symbol for this.
         char buf[48];
         sprintf( buf, "_lambda#_id_%d", id );
         Falcon::String *name = COMPILER->addString( buf );
         Falcon::Symbol *sym = COMPILER->searchGlobalSymbol( name );

         // Not defined?
         fassert( sym == 0 );
         sym = COMPILER->addGlobalSymbol( name );

         // anyhow, also in case of error, destroys the previous information to allow a correct parsing
         // of the rest.
         sym->setFunction( def );

         Falcon::StmtFunction *func = new Falcon::StmtFunction( COMPILER->lexer()->line(), sym );
         COMPILER->addFunction( func );
         func->setLambda( id );
         // prepare the statement allocation context
         COMPILER->pushContext( func );
         COMPILER->pushFunctionContext( func );
         COMPILER->pushContextSet( &func->statements() );
         COMPILER->pushFunction( def );
      ;}
    break;

  case 417:
#line 2413 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());
            COMPILER->addStatement( new Falcon::StmtReturn( LINE, (yyvsp[(5) - (5)].fal_val) ) );
            (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_lambda ,
               new Falcon::Value( func->symbol() ) ) );
            COMPILER->closeFunction();
         ;}
    break;

  case 419:
#line 2426 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_lambda );
      ;}
    break;

  case 420:
#line 2434 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_iif, (yyvsp[(1) - (5)].fal_val), (yyvsp[(3) - (5)].fal_val), (yyvsp[(5) - (5)].fal_val) ) ); ;}
    break;

  case 421:
#line 2436 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         delete (yyvsp[(1) - (3)].fal_val);
         COMPILER->raiseError(Falcon::e_syn_iif );
         (yyval.fal_val) = new Falcon::Value;
      ;}
    break;

  case 422:
#line 2445 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {  (yyval.fal_val) = new Falcon::Value( new Falcon::ArrayDecl() ); ;}
    break;

  case 423:
#line 2447 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (4)].fal_adecl) );
      ;}
    break;

  case 424:
#line 2450 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); ;}
    break;

  case 425:
#line 2451 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_arraydecl, "", COMPILER->tempLine() );
         (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (6)].fal_adecl) );
      ;}
    break;

  case 426:
#line 2458 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {  (yyval.fal_val) = new Falcon::Value( new Falcon::DictDecl() ); ;}
    break;

  case 427:
#line 2459 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (4)].fal_ddecl) ); ;}
    break;

  case 428:
#line 2460 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); ;}
    break;

  case 429:
#line 2461 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_dictdecl, "", COMPILER->tempLine() );
         (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (6)].fal_ddecl) );
      ;}
    break;

  case 430:
#line 2468 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_adecl) = new Falcon::ArrayDecl(); (yyval.fal_adecl)->pushBack( (yyvsp[(1) - (1)].fal_val) ); ;}
    break;

  case 431:
#line 2469 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyvsp[(1) - (3)].fal_adecl)->pushBack( (yyvsp[(3) - (3)].fal_val) ); (yyval.fal_adecl) = (yyvsp[(1) - (3)].fal_adecl); ;}
    break;

  case 432:
#line 2473 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_adecl) = new Falcon::ArrayDecl(); (yyval.fal_adecl)->pushBack( (yyvsp[(1) - (1)].fal_val) ); ;}
    break;

  case 433:
#line 2474 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyvsp[(1) - (4)].fal_adecl)->pushBack( (yyvsp[(4) - (4)].fal_val) ); (yyval.fal_adecl) = (yyvsp[(1) - (4)].fal_adecl); ;}
    break;

  case 434:
#line 2478 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(1) - (1)].fal_val) );
         Falcon::ArrayDecl *ad = new Falcon::ArrayDecl();
         ad->pushBack( (yyvsp[(1) - (1)].fal_val) );
         COMPILER->defineVal( (yyvsp[(1) - (1)].fal_val) );
         (yyval.fal_adecl) = ad;
      ;}
    break;

  case 435:
#line 2485 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(3) - (3)].fal_val) );
         (yyvsp[(1) - (3)].fal_adecl)->pushBack( (yyvsp[(3) - (3)].fal_val) );
      ;}
    break;

  case 436:
#line 2492 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_ddecl) = new Falcon::DictDecl(); (yyval.fal_ddecl)->pushBack( (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ); ;}
    break;

  case 437:
#line 2493 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyvsp[(1) - (6)].fal_ddecl)->pushBack( (yyvsp[(4) - (6)].fal_val), (yyvsp[(6) - (6)].fal_val) ); (yyval.fal_ddecl) = (yyvsp[(1) - (6)].fal_ddecl); ;}
    break;


/* Line 1267 of yacc.c.  */
#line 6193 "/home/gian/Progetti/falcon/core/engine/src_parser.cpp"
      default: break;
    }
  YY_SYMBOL_PRINT ("-> $$ =", yyr1[yyn], &yyval, &yyloc);

  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);

  *++yyvsp = yyval;


  /* Now `shift' the result of the reduction.  Determine what state
     that goes to, based on the state we popped back to and the rule
     number reduced by.  */

  yyn = yyr1[yyn];

  yystate = yypgoto[yyn - YYNTOKENS] + *yyssp;
  if (0 <= yystate && yystate <= YYLAST && yycheck[yystate] == *yyssp)
    yystate = yytable[yystate];
  else
    yystate = yydefgoto[yyn - YYNTOKENS];

  goto yynewstate;


/*------------------------------------.
| yyerrlab -- here on detecting error |
`------------------------------------*/
yyerrlab:
  /* If not already recovering from an error, report this error.  */
  if (!yyerrstatus)
    {
      ++yynerrs;
#if ! YYERROR_VERBOSE
      yyerror (YY_("syntax error"));
#else
      {
	YYSIZE_T yysize = yysyntax_error (0, yystate, yychar);
	if (yymsg_alloc < yysize && yymsg_alloc < YYSTACK_ALLOC_MAXIMUM)
	  {
	    YYSIZE_T yyalloc = 2 * yysize;
	    if (! (yysize <= yyalloc && yyalloc <= YYSTACK_ALLOC_MAXIMUM))
	      yyalloc = YYSTACK_ALLOC_MAXIMUM;
	    if (yymsg != yymsgbuf)
	      YYSTACK_FREE (yymsg);
	    yymsg = (char *) YYSTACK_ALLOC (yyalloc);
	    if (yymsg)
	      yymsg_alloc = yyalloc;
	    else
	      {
		yymsg = yymsgbuf;
		yymsg_alloc = sizeof yymsgbuf;
	      }
	  }

	if (0 < yysize && yysize <= yymsg_alloc)
	  {
	    (void) yysyntax_error (yymsg, yystate, yychar);
	    yyerror (yymsg);
	  }
	else
	  {
	    yyerror (YY_("syntax error"));
	    if (yysize != 0)
	      goto yyexhaustedlab;
	  }
      }
#endif
    }



  if (yyerrstatus == 3)
    {
      /* If just tried and failed to reuse look-ahead token after an
	 error, discard it.  */

      if (yychar <= YYEOF)
	{
	  /* Return failure if at end of input.  */
	  if (yychar == YYEOF)
	    YYABORT;
	}
      else
	{
	  yydestruct ("Error: discarding",
		      yytoken, &yylval);
	  yychar = YYEMPTY;
	}
    }

  /* Else will try to reuse look-ahead token after shifting the error
     token.  */
  goto yyerrlab1;


/*---------------------------------------------------.
| yyerrorlab -- error raised explicitly by YYERROR.  |
`---------------------------------------------------*/
yyerrorlab:

  /* Pacify compilers like GCC when the user code never invokes
     YYERROR and the label yyerrorlab therefore never appears in user
     code.  */
  if (/*CONSTCOND*/ 0)
     goto yyerrorlab;

  /* Do not reclaim the symbols of the rule which action triggered
     this YYERROR.  */
  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);
  yystate = *yyssp;
  goto yyerrlab1;


/*-------------------------------------------------------------.
| yyerrlab1 -- common code for both syntax error and YYERROR.  |
`-------------------------------------------------------------*/
yyerrlab1:
  yyerrstatus = 3;	/* Each real token shifted decrements this.  */

  for (;;)
    {
      yyn = yypact[yystate];
      if (yyn != YYPACT_NINF)
	{
	  yyn += YYTERROR;
	  if (0 <= yyn && yyn <= YYLAST && yycheck[yyn] == YYTERROR)
	    {
	      yyn = yytable[yyn];
	      if (0 < yyn)
		break;
	    }
	}

      /* Pop the current state because it cannot handle the error token.  */
      if (yyssp == yyss)
	YYABORT;


      yydestruct ("Error: popping",
		  yystos[yystate], yyvsp);
      YYPOPSTACK (1);
      yystate = *yyssp;
      YY_STACK_PRINT (yyss, yyssp);
    }

  if (yyn == YYFINAL)
    YYACCEPT;

  *++yyvsp = yylval;


  /* Shift the error token.  */
  YY_SYMBOL_PRINT ("Shifting", yystos[yyn], yyvsp, yylsp);

  yystate = yyn;
  goto yynewstate;


/*-------------------------------------.
| yyacceptlab -- YYACCEPT comes here.  |
`-------------------------------------*/
yyacceptlab:
  yyresult = 0;
  goto yyreturn;

/*-----------------------------------.
| yyabortlab -- YYABORT comes here.  |
`-----------------------------------*/
yyabortlab:
  yyresult = 1;
  goto yyreturn;

#ifndef yyoverflow
/*-------------------------------------------------.
| yyexhaustedlab -- memory exhaustion comes here.  |
`-------------------------------------------------*/
yyexhaustedlab:
  yyerror (YY_("memory exhausted"));
  yyresult = 2;
  /* Fall through.  */
#endif

yyreturn:
  if (yychar != YYEOF && yychar != YYEMPTY)
     yydestruct ("Cleanup: discarding lookahead",
		 yytoken, &yylval);
  /* Do not reclaim the symbols of the rule which action triggered
     this YYABORT or YYACCEPT.  */
  YYPOPSTACK (yylen);
  YY_STACK_PRINT (yyss, yyssp);
  while (yyssp != yyss)
    {
      yydestruct ("Cleanup: popping",
		  yystos[*yyssp], yyvsp);
      YYPOPSTACK (1);
    }
#ifndef yyoverflow
  if (yyss != yyssa)
    YYSTACK_FREE (yyss);
#endif
#if YYERROR_VERBOSE
  if (yymsg != yymsgbuf)
    YYSTACK_FREE (yymsg);
#endif
  /* Make sure YYID is used.  */
  return YYID (yyresult);
}


#line 2497 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
 /* c code */


void flc_src_error (const char *s)  /* Called by yyparse on error */
{
   /* do nothing: manage it in the action */
}

/* end of src_parser.yy */


