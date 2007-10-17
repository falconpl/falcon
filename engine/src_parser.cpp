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
#line 23 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"


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
#line 66 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
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
#line 371 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.cpp"
	YYSTYPE;
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif



/* Copy the second part of user declarations.  */


/* Line 216 of yacc.c.  */
#line 384 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.cpp"

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
#define YYLAST   6025

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  107
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  168
/* YYNRULES -- Number of rules.  */
#define YYNRULES  430
/* YYNRULES -- Number of states.  */
#define YYNSTATES  819

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
     124,   128,   133,   139,   144,   151,   158,   160,   162,   164,
     166,   168,   170,   172,   174,   176,   178,   180,   185,   190,
     195,   200,   205,   210,   215,   220,   225,   230,   235,   236,
     242,   245,   249,   252,   256,   260,   263,   267,   268,   275,
     278,   282,   286,   290,   294,   295,   297,   298,   302,   305,
     309,   310,   315,   319,   323,   324,   327,   330,   334,   337,
     341,   345,   346,   352,   355,   363,   373,   377,   385,   395,
     399,   400,   410,   417,   423,   424,   427,   429,   431,   433,
     435,   439,   443,   447,   450,   454,   457,   461,   463,   464,
     471,   475,   479,   480,   487,   491,   495,   496,   503,   507,
     511,   512,   519,   523,   527,   528,   531,   535,   537,   538,
     544,   545,   551,   552,   558,   559,   565,   566,   567,   571,
     572,   574,   577,   580,   583,   585,   589,   591,   593,   595,
     599,   601,   602,   609,   613,   617,   618,   621,   625,   627,
     628,   634,   635,   641,   642,   648,   649,   655,   657,   661,
     662,   664,   666,   672,   677,   681,   685,   686,   693,   696,
     700,   701,   703,   705,   708,   711,   714,   719,   723,   729,
     733,   735,   739,   741,   743,   747,   751,   757,   760,   768,
     769,   779,   783,   791,   792,   801,   804,   805,   807,   812,
     814,   815,   816,   822,   823,   827,   830,   834,   837,   841,
     845,   849,   853,   859,   865,   869,   875,   881,   885,   888,
     892,   896,   898,   902,   907,   911,   914,   918,   921,   925,
     926,   928,   932,   935,   939,   942,   943,   952,   956,   959,
     960,   966,   967,   975,   976,   979,   981,   985,   988,   989,
     995,   997,  1001,  1003,  1005,  1007,  1008,  1011,  1013,  1015,
    1017,  1019,  1020,  1028,  1034,  1039,  1040,  1044,  1048,  1050,
    1053,  1057,  1062,  1063,  1072,  1075,  1078,  1079,  1082,  1084,
    1086,  1088,  1090,  1091,  1096,  1098,  1102,  1106,  1108,  1111,
    1115,  1119,  1121,  1123,  1125,  1127,  1129,  1131,  1133,  1135,
    1137,  1140,  1145,  1151,  1155,  1157,  1159,  1163,  1166,  1170,
    1174,  1178,  1182,  1186,  1190,  1194,  1198,  1202,  1206,  1209,
    1214,  1219,  1223,  1226,  1229,  1232,  1235,  1239,  1243,  1247,
    1251,  1255,  1259,  1263,  1267,  1271,  1274,  1278,  1282,  1286,
    1290,  1294,  1297,  1300,  1303,  1305,  1307,  1311,  1316,  1322,
    1325,  1327,  1329,  1331,  1333,  1339,  1343,  1348,  1353,  1359,
    1366,  1371,  1372,  1381,  1382,  1384,  1386,  1389,  1390,  1397,
    1404,  1405,  1414,  1417,  1423,  1427,  1430,  1435,  1436,  1443,
    1447,  1452,  1453,  1460,  1462,  1466,  1468,  1473,  1475,  1479,
    1483
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int16 yyrhs[] =
{
     108,     0,    -1,   109,    -1,    -1,   109,   110,    -1,   111,
      -1,     9,     3,    -1,    23,     1,     3,    -1,   112,    -1,
     201,    -1,   224,    -1,   242,    -1,   113,    -1,   216,    -1,
     217,    -1,   219,    -1,    39,     6,     3,    -1,    39,     7,
       3,    -1,    39,     1,     3,    -1,   115,    -1,     3,    -1,
      46,     1,     3,    -1,    34,     1,     3,    -1,    32,     1,
       3,    -1,     1,     3,    -1,   253,    83,   256,    -1,   114,
      70,   253,    83,   256,    -1,   256,     3,    -1,   116,    -1,
     117,    -1,   118,    -1,   130,    -1,   147,    -1,   151,    -1,
     164,    -1,   179,    -1,   134,    -1,   145,    -1,   146,    -1,
     190,    -1,   191,    -1,   200,    -1,   251,    -1,   247,    -1,
     214,    -1,   215,    -1,   155,    -1,   156,    -1,   157,    -1,
      10,   114,     3,    -1,    10,     1,     3,    -1,   255,    83,
     256,     3,    -1,   255,    83,   106,   106,     3,    -1,   255,
      83,   271,     3,    -1,   255,    70,   273,    83,   256,     3,
      -1,   255,    70,   273,    83,   271,     3,    -1,   119,    -1,
     120,    -1,   121,    -1,   122,    -1,   123,    -1,   124,    -1,
     125,    -1,   126,    -1,   127,    -1,   128,    -1,   129,    -1,
     256,    66,   256,     3,    -1,   255,    65,   256,     3,    -1,
     255,    64,   256,     3,    -1,   255,    63,   256,     3,    -1,
     255,    62,   256,     3,    -1,   255,    56,   256,     3,    -1,
     255,    61,   256,     3,    -1,   255,    60,   256,     3,    -1,
     255,    59,   256,     3,    -1,   255,    57,   256,     3,    -1,
     255,    58,   256,     3,    -1,    -1,   132,   131,   144,     9,
       3,    -1,   133,   113,    -1,    11,   256,     3,    -1,    49,
       3,    -1,    11,     1,     3,    -1,    11,   256,    45,    -1,
      49,    45,    -1,    11,     1,    45,    -1,    -1,   136,   135,
     144,   138,     9,     3,    -1,   137,   113,    -1,    15,   256,
       3,    -1,    15,     1,     3,    -1,    15,   256,    45,    -1,
      15,     1,    45,    -1,    -1,   141,    -1,    -1,   140,   139,
     144,    -1,    16,     3,    -1,    16,     1,     3,    -1,    -1,
     143,   142,   144,   138,    -1,    17,   256,     3,    -1,    17,
       1,     3,    -1,    -1,   144,   113,    -1,    12,     3,    -1,
      12,     1,     3,    -1,    13,     3,    -1,    13,    14,     3,
      -1,    13,     1,     3,    -1,    -1,   149,   148,   144,     9,
       3,    -1,   150,   113,    -1,    18,   255,    83,   256,    69,
     256,     3,    -1,    18,   255,    83,   256,    69,   256,    68,
     256,     3,    -1,    18,     1,     3,    -1,    18,   255,    83,
     256,    69,   256,    45,    -1,    18,   255,    83,   256,    69,
     256,    68,   256,    45,    -1,    18,     1,    45,    -1,    -1,
      18,   273,    86,   256,     3,   152,   153,     9,     3,    -1,
      18,   273,    86,   256,    45,   113,    -1,    18,   273,    86,
       1,     3,    -1,    -1,   154,   153,    -1,   113,    -1,   158,
      -1,   160,    -1,   162,    -1,    48,   256,     3,    -1,    48,
       1,     3,    -1,    78,   271,     3,    -1,    78,     3,    -1,
      79,   271,     3,    -1,    79,     3,    -1,    78,     1,     3,
      -1,    50,    -1,    -1,    19,     3,   159,   144,     9,     3,
      -1,    19,    45,   113,    -1,    19,     1,     3,    -1,    -1,
      20,     3,   161,   144,     9,     3,    -1,    20,    45,   113,
      -1,    20,     1,     3,    -1,    -1,    21,     3,   163,   144,
       9,     3,    -1,    21,    45,   113,    -1,    21,     1,     3,
      -1,    -1,   166,   165,   167,   173,     9,     3,    -1,    22,
     256,     3,    -1,    22,     1,     3,    -1,    -1,   167,   168,
      -1,   167,     1,     3,    -1,     3,    -1,    -1,    23,   177,
       3,   169,   144,    -1,    -1,    23,   177,    45,   170,   113,
      -1,    -1,    23,     1,     3,   171,   144,    -1,    -1,    23,
       1,    45,   172,   113,    -1,    -1,    -1,   175,   174,   176,
      -1,    -1,    24,    -1,    24,     1,    -1,     3,   144,    -1,
      45,   113,    -1,   178,    -1,   177,    70,   178,    -1,     8,
      -1,     4,    -1,     7,    -1,     4,    69,     4,    -1,     6,
      -1,    -1,   181,   180,   182,   173,     9,     3,    -1,    25,
     256,     3,    -1,    25,     1,     3,    -1,    -1,   182,   183,
      -1,   182,     1,     3,    -1,     3,    -1,    -1,    23,   188,
       3,   184,   144,    -1,    -1,    23,   188,    45,   185,   113,
      -1,    -1,    23,     1,     3,   186,   144,    -1,    -1,    23,
       1,    45,   187,   113,    -1,   189,    -1,   188,    70,   189,
      -1,    -1,     4,    -1,     6,    -1,    28,   271,    69,   256,
       3,    -1,    28,   271,     1,     3,    -1,    28,     1,     3,
      -1,    29,    45,   113,    -1,    -1,   193,   192,   144,   194,
       9,     3,    -1,    29,     3,    -1,    29,     1,     3,    -1,
      -1,   195,    -1,   196,    -1,   195,   196,    -1,   197,   144,
      -1,    30,     3,    -1,    30,    86,   253,     3,    -1,    30,
     198,     3,    -1,    30,   198,    86,   253,     3,    -1,    30,
       1,     3,    -1,   199,    -1,   198,    70,   199,    -1,     4,
      -1,     6,    -1,    31,   256,     3,    -1,    31,     1,     3,
      -1,   202,   209,   144,     9,     3,    -1,   204,   113,    -1,
     206,    52,   260,   207,   260,    51,     3,    -1,    -1,   206,
      52,   260,   207,     1,   203,   260,    51,     3,    -1,   206,
       1,     3,    -1,   206,    52,   260,   207,   260,    51,    45,
      -1,    -1,   206,    52,   260,     1,   205,   260,    51,    45,
      -1,    46,     6,    -1,    -1,   208,    -1,   207,    70,   260,
     208,    -1,     6,    -1,    -1,    -1,   212,   210,   144,     9,
       3,    -1,    -1,   213,   211,   113,    -1,    47,     3,    -1,
      47,     1,     3,    -1,    47,    45,    -1,    47,     1,    45,
      -1,    40,   258,     3,    -1,    40,     1,     3,    -1,    43,
     256,     3,    -1,    43,   256,    86,   256,     3,    -1,    43,
     256,    86,     1,     3,    -1,    43,     1,     3,    -1,    41,
       6,    83,   252,     3,    -1,    41,     6,    83,     1,     3,
      -1,    41,     1,     3,    -1,    44,     3,    -1,    44,   218,
       3,    -1,    44,     1,     3,    -1,     6,    -1,   218,    70,
       6,    -1,   220,   223,     9,     3,    -1,   221,   222,     3,
      -1,    42,     3,    -1,    42,     1,     3,    -1,    42,    45,
      -1,    42,     1,    45,    -1,    -1,     6,    -1,   222,    70,
       6,    -1,   222,     3,    -1,   223,   222,     3,    -1,     1,
       3,    -1,    -1,    32,     6,   225,   226,   235,   240,     9,
       3,    -1,   227,   229,     3,    -1,     1,     3,    -1,    -1,
      52,   260,   207,   260,    51,    -1,    -1,    52,   260,   207,
       1,   228,   260,    51,    -1,    -1,    33,   230,    -1,   231,
      -1,   230,    70,   231,    -1,     6,   232,    -1,    -1,    52,
     260,   233,   260,    51,    -1,   234,    -1,   233,    70,   234,
      -1,   252,    -1,     6,    -1,    27,    -1,    -1,   235,   236,
      -1,     3,    -1,   201,    -1,   239,    -1,   237,    -1,    -1,
      38,     3,   238,   209,   144,     9,     3,    -1,    47,     6,
      83,   256,     3,    -1,     6,    83,   256,     3,    -1,    -1,
      88,   241,     3,    -1,    88,     1,     3,    -1,     6,    -1,
      74,     6,    -1,   241,    70,     6,    -1,   241,    70,    74,
       6,    -1,    -1,    34,     6,   243,   244,   245,   240,     9,
       3,    -1,   229,     3,    -1,     1,     3,    -1,    -1,   245,
     246,    -1,     3,    -1,   201,    -1,   239,    -1,   237,    -1,
      -1,    36,   248,   249,     3,    -1,   250,    -1,   249,    70,
     250,    -1,   249,    70,     1,    -1,     6,    -1,    35,     3,
      -1,    35,   256,     3,    -1,    35,     1,     3,    -1,     8,
      -1,     4,    -1,     5,    -1,     7,    -1,     6,    -1,   253,
      -1,    27,    -1,    26,    -1,   254,    -1,   255,   257,    -1,
     255,    54,   256,    53,    -1,   255,    54,    98,   256,    53,
      -1,   255,    55,     6,    -1,   252,    -1,   255,    -1,   256,
      95,   256,    -1,    94,   256,    -1,   256,    94,   256,    -1,
     256,    98,   256,    -1,   256,    97,   256,    -1,   256,    96,
     256,    -1,   256,    99,   256,    -1,   256,    93,   256,    -1,
     256,    92,   256,    -1,   256,    91,   256,    -1,   256,   101,
     256,    -1,   256,   100,   256,    -1,   102,   256,    -1,    75,
     255,    82,   256,    -1,    75,   255,    83,   256,    -1,   256,
      80,   256,    -1,   256,   105,    -1,   105,   256,    -1,   256,
     104,    -1,   104,   256,    -1,   256,    81,   256,    -1,   256,
      82,   256,    -1,   256,    83,   256,    -1,   256,    79,   256,
      -1,   256,    78,   256,    -1,   256,    77,   256,    -1,   256,
      76,   256,    -1,   256,    73,   256,    -1,   256,    72,   256,
      -1,    74,   256,    -1,   256,    88,   256,    -1,   256,    87,
     256,    -1,   256,    86,   256,    -1,   256,    85,   256,    -1,
     256,    84,     6,    -1,   106,   256,    -1,    90,   256,    -1,
      89,   256,    -1,   262,    -1,   258,    -1,   258,    55,     6,
      -1,   258,    54,   256,    53,    -1,   258,    54,    98,   256,
      53,    -1,   258,   257,    -1,   266,    -1,   267,    -1,   269,
      -1,   257,    -1,    52,   260,   256,   260,    51,    -1,    54,
      45,    53,    -1,    54,   256,    45,    53,    -1,    54,    45,
     256,    53,    -1,    54,   256,    45,   256,    53,    -1,   256,
      52,   260,   272,   260,    51,    -1,   256,    52,   260,    51,
      -1,    -1,   256,    52,   260,   272,     1,   259,   260,    51,
      -1,    -1,   261,    -1,     3,    -1,   261,     3,    -1,    -1,
      37,   263,   264,   209,   144,     9,    -1,    52,   260,   207,
     260,    51,     3,    -1,    -1,    52,   260,   207,     1,   265,
     260,    51,     3,    -1,     1,     3,    -1,   256,    71,   256,
      45,   256,    -1,   256,    71,     1,    -1,    54,    53,    -1,
      54,   272,   260,    53,    -1,    -1,    54,   272,     1,   268,
     260,    53,    -1,    54,    67,    53,    -1,    54,   274,   260,
      53,    -1,    -1,    54,   274,     1,   270,   260,    53,    -1,
     256,    -1,   271,    70,   256,    -1,   256,    -1,   272,    70,
     260,   256,    -1,   253,    -1,   273,    70,   253,    -1,   256,
      67,   256,    -1,   274,    70,   260,   256,    67,   256,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   192,   192,   195,   197,   201,   202,   203,   208,   213,
     218,   223,   228,   233,   234,   235,   239,   245,   251,   259,
     260,   263,   264,   265,   266,   271,   276,   283,   284,   285,
     286,   287,   288,   289,   290,   291,   292,   293,   294,   295,
     296,   297,   298,   299,   300,   301,   302,   303,   304,   308,
     310,   316,   320,   324,   328,   333,   343,   344,   345,   346,
     347,   348,   349,   350,   351,   352,   353,   357,   364,   371,
     378,   385,   392,   399,   406,   413,   419,   425,   433,   433,
     448,   456,   457,   458,   462,   463,   464,   468,   468,   483,
     493,   494,   498,   499,   503,   505,   506,   506,   515,   516,
     521,   521,   533,   534,   537,   539,   545,   554,   562,   572,
     581,   589,   589,   603,   619,   623,   627,   635,   639,   643,
     653,   652,   676,   690,   694,   696,   700,   707,   708,   709,
     713,   726,   734,   738,   744,   749,   756,   765,   775,   775,
     789,   797,   801,   801,   814,   822,   826,   826,   840,   848,
     852,   852,   869,   870,   877,   879,   880,   884,   886,   885,
     896,   896,   908,   908,   920,   920,   936,   939,   938,   951,
     952,   953,   956,   957,   963,   964,   968,   977,   989,  1000,
    1011,  1032,  1032,  1049,  1050,  1057,  1059,  1060,  1064,  1066,
    1065,  1076,  1076,  1089,  1089,  1101,  1101,  1119,  1120,  1123,
    1124,  1136,  1157,  1161,  1166,  1174,  1181,  1180,  1199,  1200,
    1203,  1205,  1209,  1210,  1214,  1219,  1237,  1257,  1267,  1278,
    1286,  1287,  1291,  1303,  1326,  1327,  1334,  1344,  1353,  1354,
    1354,  1358,  1362,  1363,  1363,  1370,  1424,  1426,  1427,  1431,
    1446,  1449,  1448,  1460,  1459,  1474,  1475,  1479,  1480,  1489,
    1493,  1501,  1508,  1518,  1524,  1536,  1546,  1551,  1563,  1572,
    1579,  1587,  1592,  1604,  1611,  1621,  1622,  1625,  1626,  1629,
    1631,  1635,  1642,  1643,  1644,  1656,  1655,  1714,  1717,  1723,
    1725,  1726,  1726,  1732,  1734,  1738,  1739,  1743,  1777,  1779,
    1788,  1789,  1793,  1794,  1803,  1806,  1808,  1812,  1813,  1816,
    1835,  1839,  1839,  1873,  1895,  1922,  1924,  1925,  1932,  1940,
    1946,  1952,  1966,  1965,  2029,  2030,  2036,  2038,  2042,  2043,
    2046,  2065,  2074,  2073,  2091,  2092,  2093,  2100,  2116,  2117,
    2118,  2128,  2129,  2130,  2131,  2135,  2153,  2154,  2155,  2166,
    2167,  2172,  2177,  2183,  2192,  2193,  2194,  2195,  2196,  2197,
    2198,  2199,  2200,  2201,  2202,  2203,  2204,  2205,  2206,  2207,
    2209,  2211,  2212,  2213,  2214,  2215,  2216,  2217,  2218,  2219,
    2220,  2221,  2222,  2223,  2224,  2225,  2226,  2227,  2228,  2229,
    2230,  2231,  2232,  2233,  2234,  2235,  2236,  2240,  2244,  2248,
    2252,  2253,  2254,  2255,  2256,  2261,  2264,  2267,  2270,  2276,
    2282,  2287,  2287,  2295,  2297,  2301,  2302,  2307,  2306,  2349,
    2350,  2350,  2354,  2361,  2362,  2372,  2373,  2377,  2377,  2385,
    2386,  2387,  2387,  2395,  2396,  2400,  2401,  2405,  2412,  2419,
    2420
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
  "autobxor", "autoshl", "autoshr", "while_statement", "@1", "while_decl",
  "while_short_decl", "if_statement", "@2", "if_decl", "if_short_decl",
  "elif_or_else", "@3", "else_decl", "elif_statement", "@4", "elif_decl",
  "statement_list", "break_statement", "continue_statement",
  "for_statement", "@5", "for_decl", "for_decl_short", "forin_statement",
  "@6", "forin_statement_list", "forin_statement_elem", "fordot_statement",
  "self_print_statement", "outer_print_statement", "first_loop_block",
  "@7", "last_loop_block", "@8", "all_loop_block", "@9",
  "switch_statement", "@10", "switch_decl", "case_list", "case_statement",
  "@11", "@12", "@13", "@14", "default_statement", "@15", "default_decl",
  "default_body", "case_expression_list", "case_element",
  "select_statement", "@16", "select_decl", "selcase_list",
  "selcase_statement", "@17", "@18", "@19", "@20",
  "selcase_expression_list", "selcase_element", "give_statement",
  "try_statement", "@21", "try_decl", "catch_statements", "catch_list",
  "catch_statement", "catch_decl", "catchcase_element_list",
  "catchcase_element", "raise_statement", "func_statement", "func_decl",
  "@22", "func_decl_short", "@23", "func_begin", "param_list",
  "param_symbol", "static_block", "@24", "@25", "static_decl",
  "static_short_decl", "launch_statement", "pass_statement",
  "const_statement", "export_statement", "export_symbol_list",
  "attributes_statement", "attributes_decl", "attributes_short_decl",
  "attribute_list", "attribute_vert_list", "class_decl", "@26",
  "class_def_inner", "class_param_list", "@27", "from_clause",
  "inherit_list", "inherit_token", "inherit_call", "inherit_param_list",
  "inherit_param_token", "class_statement_list", "class_statement",
  "init_decl", "@28", "property_decl", "has_list", "has_clause_list",
  "object_decl", "@29", "object_decl_inner", "object_statement_list",
  "object_statement", "global_statement", "@30", "global_symbol_list",
  "globalized_symbol", "return_statement", "const_atom", "atomic_symbol",
  "var_atom", "variable", "expression", "range_decl", "func_call", "@31",
  "opt_eol", "eol_seq", "lambda_expr", "@32", "lambda_decl_inner", "@33",
  "iif_expr", "array_decl", "@34", "dict_decl", "@35", "expression_list",
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
     116,   117,   117,   117,   117,   117,   118,   118,   118,   118,
     118,   118,   118,   118,   118,   118,   118,   119,   120,   121,
     122,   123,   124,   125,   126,   127,   128,   129,   131,   130,
     130,   132,   132,   132,   133,   133,   133,   135,   134,   134,
     136,   136,   137,   137,   138,   138,   139,   138,   140,   140,
     142,   141,   143,   143,   144,   144,   145,   145,   146,   146,
     146,   148,   147,   147,   149,   149,   149,   150,   150,   150,
     152,   151,   151,   151,   153,   153,   154,   154,   154,   154,
     155,   155,   156,   156,   156,   156,   156,   157,   159,   158,
     158,   158,   161,   160,   160,   160,   163,   162,   162,   162,
     165,   164,   166,   166,   167,   167,   167,   168,   169,   168,
     170,   168,   171,   168,   172,   168,   173,   174,   173,   175,
     175,   175,   176,   176,   177,   177,   178,   178,   178,   178,
     178,   180,   179,   181,   181,   182,   182,   182,   183,   184,
     183,   185,   183,   186,   183,   187,   183,   188,   188,   189,
     189,   189,   190,   190,   190,   191,   192,   191,   193,   193,
     194,   194,   195,   195,   196,   197,   197,   197,   197,   197,
     198,   198,   199,   199,   200,   200,   201,   201,   202,   203,
     202,   202,   204,   205,   204,   206,   207,   207,   207,   208,
     209,   210,   209,   211,   209,   212,   212,   213,   213,   214,
     214,   215,   215,   215,   215,   216,   216,   216,   217,   217,
     217,   218,   218,   219,   219,   220,   220,   221,   221,   222,
     222,   222,   223,   223,   223,   225,   224,   226,   226,   227,
     227,   228,   227,   229,   229,   230,   230,   231,   232,   232,
     233,   233,   234,   234,   234,   235,   235,   236,   236,   236,
     236,   238,   237,   239,   239,   240,   240,   240,   241,   241,
     241,   241,   243,   242,   244,   244,   245,   245,   246,   246,
     246,   246,   248,   247,   249,   249,   249,   250,   251,   251,
     251,   252,   252,   252,   252,   253,   254,   254,   254,   255,
     255,   255,   255,   255,   256,   256,   256,   256,   256,   256,
     256,   256,   256,   256,   256,   256,   256,   256,   256,   256,
     256,   256,   256,   256,   256,   256,   256,   256,   256,   256,
     256,   256,   256,   256,   256,   256,   256,   256,   256,   256,
     256,   256,   256,   256,   256,   256,   256,   256,   256,   256,
     256,   256,   256,   256,   256,   257,   257,   257,   257,   258,
     258,   259,   258,   260,   260,   261,   261,   263,   262,   264,
     265,   264,   264,   266,   266,   267,   267,   268,   267,   269,
     269,   270,   269,   271,   271,   272,   272,   273,   273,   274,
     274
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     1,     0,     2,     1,     2,     3,     1,     1,
       1,     1,     1,     1,     1,     1,     3,     3,     3,     1,
       1,     3,     3,     3,     2,     3,     5,     2,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     3,
       3,     4,     5,     4,     6,     6,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     4,     4,     4,
       4,     4,     4,     4,     4,     4,     4,     4,     0,     5,
       2,     3,     2,     3,     3,     2,     3,     0,     6,     2,
       3,     3,     3,     3,     0,     1,     0,     3,     2,     3,
       0,     4,     3,     3,     0,     2,     2,     3,     2,     3,
       3,     0,     5,     2,     7,     9,     3,     7,     9,     3,
       0,     9,     6,     5,     0,     2,     1,     1,     1,     1,
       3,     3,     3,     2,     3,     2,     3,     1,     0,     6,
       3,     3,     0,     6,     3,     3,     0,     6,     3,     3,
       0,     6,     3,     3,     0,     2,     3,     1,     0,     5,
       0,     5,     0,     5,     0,     5,     0,     0,     3,     0,
       1,     2,     2,     2,     1,     3,     1,     1,     1,     3,
       1,     0,     6,     3,     3,     0,     2,     3,     1,     0,
       5,     0,     5,     0,     5,     0,     5,     1,     3,     0,
       1,     1,     5,     4,     3,     3,     0,     6,     2,     3,
       0,     1,     1,     2,     2,     2,     4,     3,     5,     3,
       1,     3,     1,     1,     3,     3,     5,     2,     7,     0,
       9,     3,     7,     0,     8,     2,     0,     1,     4,     1,
       0,     0,     5,     0,     3,     2,     3,     2,     3,     3,
       3,     3,     5,     5,     3,     5,     5,     3,     2,     3,
       3,     1,     3,     4,     3,     2,     3,     2,     3,     0,
       1,     3,     2,     3,     2,     0,     8,     3,     2,     0,
       5,     0,     7,     0,     2,     1,     3,     2,     0,     5,
       1,     3,     1,     1,     1,     0,     2,     1,     1,     1,
       1,     0,     7,     5,     4,     0,     3,     3,     1,     2,
       3,     4,     0,     8,     2,     2,     0,     2,     1,     1,
       1,     1,     0,     4,     1,     3,     3,     1,     2,     3,
       3,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       2,     4,     5,     3,     1,     1,     3,     2,     3,     3,
       3,     3,     3,     3,     3,     3,     3,     3,     2,     4,
       4,     3,     2,     2,     2,     2,     3,     3,     3,     3,
       3,     3,     3,     3,     3,     2,     3,     3,     3,     3,
       3,     2,     2,     2,     1,     1,     3,     4,     5,     2,
       1,     1,     1,     1,     5,     3,     4,     4,     5,     6,
       4,     0,     8,     0,     1,     1,     2,     0,     6,     6,
       0,     8,     2,     5,     3,     2,     4,     0,     6,     3,
       4,     0,     6,     1,     3,     1,     4,     1,     3,     3,
       6
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint16 yydefact[] =
{
       3,     0,     0,     1,     0,    20,   332,   333,   335,   334,
     331,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   338,   337,     0,     0,     0,     0,     0,     0,   322,
     407,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     137,   403,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     4,     5,     8,    12,    19,    28,
      29,    30,    56,    57,    58,    59,    60,    61,    62,    63,
      64,    65,    66,    31,    78,     0,    36,    87,     0,    37,
      38,    32,   111,     0,    33,    46,    47,    48,    34,   150,
      35,   181,    39,    40,   206,    41,     9,   240,     0,     0,
      44,    45,    13,    14,    15,     0,   269,    10,    11,    43,
      42,   344,   336,   339,   345,     0,   393,   385,   384,   390,
     391,   392,    24,     6,     0,     0,     0,     0,   345,     0,
       0,   106,     0,   108,     0,     0,     0,     0,   336,     0,
       0,     0,     0,     0,     0,     0,     0,   423,     0,     0,
     208,     0,     0,     0,     0,   275,     0,   312,     0,   328,
       0,     0,     0,     0,     0,     0,     0,     0,   385,     0,
       0,     0,   265,   267,     0,     0,     0,   258,   261,     0,
       0,   235,     0,     0,    82,    85,   405,     0,   404,     0,
     415,     0,   425,     0,     0,   375,     0,     0,   133,     0,
     135,     0,   383,   382,   347,   358,   365,   363,   381,   104,
       0,     0,     0,    80,   104,    89,   104,   113,   154,   185,
     104,     0,   104,   241,   243,   227,     0,   403,     0,   270,
       0,   269,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,   340,    27,   403,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   364,   362,
       0,     0,   389,    50,    49,     0,     0,    83,    86,    81,
      84,   107,   110,   109,    91,    93,    90,    92,   116,   119,
       0,     0,     0,   153,   152,     7,   184,   183,   204,     0,
       0,     0,   209,   205,   225,   224,    23,     0,    22,     0,
     330,   329,   327,     0,   324,     0,   403,   240,    18,    16,
      17,   250,   249,   257,     0,   266,   268,   254,   251,     0,
     260,   259,     0,    21,   131,   130,   403,   406,   395,     0,
     419,     0,     0,   417,   403,     0,   421,   403,     0,     0,
       0,   136,   132,   134,     0,     0,     0,     0,     0,     0,
       0,   245,   247,     0,   104,     0,   231,     0,   274,   272,
       0,     0,     0,   264,     0,     0,   343,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,   427,     0,     0,
     423,     0,     0,     0,   414,     0,   374,   373,   372,   371,
     370,   369,   361,   366,   367,   368,   380,   379,   378,   377,
     376,   355,   354,   353,   348,   346,   351,   350,   349,   352,
     357,   356,     0,     0,   386,     0,    25,     0,   428,     0,
       0,   203,     0,   424,     0,   403,   295,   283,     0,     0,
       0,   316,   323,     0,   412,   236,   104,     0,     0,     0,
     378,   262,     0,   397,   396,     0,   429,   403,     0,   416,
     403,     0,   420,   359,   360,     0,   105,     0,     0,     0,
      96,    95,   100,     0,     0,   157,     0,     0,   155,     0,
     167,     0,   188,     0,     0,   186,     0,     0,   211,   212,
     104,   246,   248,     0,     0,   244,   233,   239,     0,   237,
     271,   263,   273,     0,   341,    72,    76,    77,    75,    74,
      73,    71,    70,    69,    68,     0,     0,    51,    53,   400,
     425,     0,    67,     0,     0,   387,     0,     0,   123,   120,
       0,   202,   278,   236,   305,     0,   315,   288,   284,   285,
     314,   305,   326,   325,     0,     0,   256,   255,   253,   252,
     394,   398,     0,   426,     0,     0,    79,     0,    98,     0,
       0,     0,   104,   104,   112,   156,     0,   177,   180,   178,
     176,     0,   174,   171,     0,     0,   187,     0,   200,   201,
       0,   197,     0,     0,   215,   222,   223,     0,     0,   220,
       0,   213,     0,   226,     0,   403,   229,   403,     0,   342,
     423,     0,    52,   401,     0,   413,   388,    26,     0,     0,
     122,     0,   297,     0,     0,     0,     0,     0,   298,   296,
     300,   299,     0,   277,   403,   287,     0,   318,   319,   321,
     320,     0,   317,   410,     0,   408,   418,   422,     0,    99,
     103,   102,    88,     0,     0,   162,   164,     0,   158,   160,
       0,   151,   104,     0,   168,   193,   195,   189,   191,   199,
     182,   219,     0,   217,     0,     0,   207,   242,     0,   403,
       0,     0,    54,    55,   403,   399,   114,   117,     0,     0,
       0,     0,   126,     0,     0,   127,   128,   129,   281,     0,
       0,   301,     0,     0,   308,     0,     0,     0,     0,   286,
       0,   403,     0,   430,   101,   104,     0,   179,   104,     0,
     175,     0,   173,   104,     0,   104,     0,   198,   216,   221,
       0,     0,     0,   238,   228,   232,     0,     0,     0,   138,
       0,     0,   142,     0,     0,   146,     0,     0,   125,   403,
     280,     0,   240,     0,   307,   309,   306,     0,   276,   293,
     294,   403,   290,   292,   313,     0,   409,     0,   165,     0,
     161,     0,   196,     0,   192,   218,   234,     0,   402,   115,
     118,   141,   104,   140,   145,   104,   144,   149,   104,   148,
     121,     0,   304,   104,     0,   310,     0,     0,     0,     0,
     230,     0,     0,     0,   282,     0,   303,   311,   291,   289,
     411,     0,     0,     0,     0,   139,   143,   147,   302
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     1,     2,    54,    55,    56,   476,   125,    58,    59,
      60,    61,    62,    63,    64,    65,    66,    67,    68,    69,
      70,    71,    72,    73,   209,    74,    75,    76,   214,    77,
      78,   479,   572,   480,   481,   573,   482,   364,    79,    80,
      81,   216,    82,    83,    84,   619,   693,   694,    85,    86,
      87,   695,   782,   696,   785,   697,   788,    88,   218,    89,
     367,   488,   718,   719,   715,   716,   489,   585,   490,   664,
     581,   582,    90,   219,    91,   368,   495,   725,   726,   723,
     724,   590,   591,    92,    93,   220,    94,   497,   498,   499,
     500,   598,   599,    95,    96,    97,   679,    98,   605,    99,
     508,   509,   222,   374,   375,   223,   224,   100,   101,   102,
     103,   179,   104,   105,   106,   230,   231,   107,   317,   446,
     447,   749,   450,   548,   549,   635,   761,   762,   544,   629,
     630,   752,   631,   632,   706,   108,   319,   451,   551,   642,
     109,   161,   323,   324,   110,   111,   112,   113,   128,   115,
     116,   117,   684,   187,   188,   118,   162,   327,   711,   119,
     120,   467,   121,   470,   148,   193,   140,   194
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -536
static const yytype_int16 yypact[] =
{
    -536,    12,   804,  -536,    17,  -536,  -536,  -536,  -536,  -536,
    -536,    32,   321,   674,    63,   179,  2873,   315,  2928,    44,
    2983,  -536,  -536,  3038,    45,  3093,   331,   414,   379,  -536,
    -536,   489,  3148,   511,   191,  3203,   507,   515,  3258,   185,
    -536,    50,  4994,  5269,   383,   521,  3484,  5269,  5269,  5269,
    5269,  5269,  5269,  5269,  -536,  -536,  -536,  -536,  -536,  -536,
    -536,  -536,  -536,  -536,  -536,  -536,  -536,  -536,  -536,  -536,
    -536,  -536,  -536,  -536,  -536,  2818,  -536,  -536,  2818,  -536,
    -536,  -536,  -536,  2818,  -536,  -536,  -536,  -536,  -536,  -536,
    -536,  -536,  -536,  -536,  -536,  -536,  -536,    47,  2818,    16,
    -536,  -536,  -536,  -536,  -536,    30,    97,  -536,  -536,  -536,
    -536,  -536,  -536,  -536,   911,  3615,  -536,   271,  -536,  -536,
    -536,  -536,  -536,  -536,   104,    35,    51,   301,   346,  3677,
     170,  -536,   203,  -536,   207,   307,  3734,   310,   132,    -5,
     247,   231,  3898,   254,   312,  3935,   342,  5809,    29,   421,
    -536,  2818,   422,  3985,   433,  -536,   435,  -536,   460,  -536,
    4022,   232,    20,   461,   483,   486,   494,  5809,   304,   496,
     391,   327,  -536,  -536,   503,  4072,   506,  -536,  -536,   101,
     516,  -536,   527,  4109,  -536,  -536,  -536,  5269,   528,  5125,
    -536,   265,  5366,   105,   167,  5920,   344,   529,  -536,   106,
    -536,   129,   485,   485,   219,   219,   219,   219,   219,  -536,
     487,   532,   539,  -536,  -536,  -536,  -536,  -536,  -536,  -536,
    -536,   194,  -536,  -536,  -536,  -536,   538,    50,   540,  -536,
     141,   175,   184,  5018,   543,  5269,  5269,  5269,  5269,  5269,
    5269,  5269,  5269,  5269,  5269,   544,  5275,  -536,  -536,    50,
    5269,  3313,  5269,  5269,  5269,  5269,  5269,  5269,  5269,  5269,
    5269,  5269,   545,  5269,  5269,  5269,  5269,  5269,  5269,  5269,
    5269,  5269,  5269,  5269,  5269,  5269,  5269,  5269,  -536,  -536,
    5052,   551,  -536,  -536,  -536,   544,  5269,  -536,  -536,  -536,
    -536,  -536,  -536,  -536,  -536,  -536,  -536,  -536,  -536,  -536,
    5269,   544,  3368,  -536,  -536,  -536,  -536,  -536,  -536,   541,
    5269,  5269,  -536,  -536,  -536,  -536,  -536,   176,  -536,   401,
    -536,  -536,  -536,   188,  -536,   556,    50,    47,  -536,  -536,
    -536,  -536,  -536,  -536,   471,  -536,  -536,  -536,  -536,  3423,
    -536,  -536,   554,  -536,  -536,  -536,  4159,  -536,  -536,  5574,
    -536,  5159,  5269,  -536,    50,   508,  -536,    50,   509,  5269,
    5269,  -536,  -536,  -536,  1652,   111,  1758,   305,   491,  1440,
     362,  -536,  -536,  1864,  -536,  2818,  -536,   163,  -536,  -536,
     557,   563,   221,  -536,  5269,  5423,  -536,  4196,  4246,  4283,
    4333,  4370,  4420,  4457,  4507,  4544,  4594,  -536,   -44,  5330,
    4631,   222,  5166,  4681,  -536,  5537,  5883,  5920,   776,   776,
     776,   776,   776,   776,   776,   776,  -536,   485,   485,   485,
     485,   572,   572,   656,   361,   361,   235,   235,   235,   291,
     219,   219,  5269,  5480,  -536,   488,  5809,  5611,  -536,   565,
    3791,  -536,  4718,  5809,   566,    50,  -536,   555,   584,   585,
     589,  -536,  -536,   517,  -536,   587,  -536,   591,   595,   596,
     351,  -536,   550,  -536,  -536,  5648,  5809,    50,  5269,  -536,
      50,  5269,  -536,   776,   776,   599,  -536,   218,  3478,   594,
    -536,  -536,  -536,   601,   602,  -536,   497,   299,  -536,   597,
    -536,   604,  -536,    85,   600,  -536,     7,   603,   583,  -536,
    -536,  -536,  -536,   611,  1970,  -536,  -536,  -536,   171,  -536,
    -536,  -536,  -536,  5685,  -536,  -536,  -536,  -536,  -536,  -536,
    -536,  -536,  -536,  -536,  -536,  5269,  3539,  -536,  -536,  -536,
    5809,   287,  -536,  5269,  5722,  -536,  5269,  5269,  -536,  -536,
    2818,  -536,  -536,   587,    37,   613,  -536,   567,   552,  -536,
    -536,    64,  -536,  -536,   296,  2076,  -536,  -536,  -536,  -536,
    -536,  -536,   568,  5809,   575,  5772,  -536,   617,  -536,   626,
    4768,   627,  -536,  -536,  -536,  -536,   373,   564,  -536,  -536,
    -536,   138,  -536,  -536,   629,   394,  -536,   425,  -536,  -536,
     159,  -536,   633,   635,  -536,  -536,  -536,   544,    15,  -536,
     636,  -536,  1546,  -536,   637,    50,  -536,    50,   590,  -536,
    4805,   223,  -536,  -536,   592,  5846,  -536,  5809,  3578,   910,
    -536,   300,  -536,   561,   642,   640,   643,    13,  -536,  -536,
    -536,  -536,   639,  -536,    50,  -536,   585,  -536,  -536,  -536,
    -536,   641,  -536,  -536,   605,  -536,  -536,  -536,  5269,  -536,
    -536,  -536,  -536,  2182,   111,  -536,  -536,   649,  -536,  -536,
     548,  -536,  -536,  2818,  -536,  -536,  -536,  -536,  -536,   408,
    -536,  -536,   651,  -536,   431,   544,  -536,  -536,   607,    50,
     587,   426,  -536,  -536,    50,  -536,  -536,  -536,  5269,   374,
     378,   387,  -536,   646,   910,  -536,  -536,  -536,  -536,   608,
    5269,  -536,   577,   660,  -536,   658,   224,   671,   683,  -536,
     680,    50,   681,  5809,  -536,  -536,  2818,  -536,  -536,  2818,
    -536,  2288,  -536,  -536,  2818,  -536,  2818,  -536,  -536,  -536,
     682,   650,   645,  -536,  -536,  -536,   647,  3848,   691,  -536,
    2818,   696,  -536,  2818,   699,  -536,  2818,   700,  -536,    50,
    -536,  4855,    47,  5269,  -536,  -536,  -536,    18,  -536,  -536,
    -536,   228,  -536,  -536,  -536,   653,  -536,  1016,  -536,  1122,
    -536,  1228,  -536,  1334,  -536,  -536,  -536,   702,  -536,  -536,
    -536,  -536,  -536,  -536,  -536,  -536,  -536,  -536,  -536,  -536,
    -536,   655,  -536,  -536,  4892,  -536,   701,   683,   661,   710,
    -536,  2394,  2500,  2606,  -536,  2712,  -536,  -536,  -536,  -536,
    -536,   713,   717,   718,   724,  -536,  -536,  -536,  -536
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -536,  -536,  -536,  -536,  -536,  -536,    -1,  -536,  -536,  -536,
    -536,  -536,  -536,  -536,  -536,  -536,  -536,  -536,  -536,  -536,
    -536,  -536,  -536,  -536,  -536,  -536,  -536,  -536,  -536,  -536,
    -536,    75,  -536,  -536,  -536,  -536,  -536,  -151,  -536,  -536,
    -536,  -536,  -536,  -536,  -536,  -536,    36,  -536,  -536,  -536,
    -536,  -536,  -536,  -536,  -536,  -536,  -536,  -536,  -536,  -536,
    -536,  -536,  -536,  -536,  -536,  -536,   363,  -536,  -536,  -536,
    -536,    72,  -536,  -536,  -536,  -536,  -536,  -536,  -536,  -536,
    -536,  -536,    65,  -536,  -536,  -536,  -536,  -536,  -536,   237,
    -536,  -536,    59,  -536,  -535,  -536,  -536,  -536,  -536,  -536,
    -448,    56,  -322,  -536,  -536,  -536,  -536,  -536,  -536,  -536,
    -536,  -536,  -536,  -536,  -536,  -100,  -536,  -536,  -536,  -536,
    -536,  -536,   290,  -536,   107,  -536,  -536,   -57,  -536,  -536,
     195,  -536,   196,   208,  -536,  -536,  -536,  -536,  -536,  -536,
    -536,  -536,  -536,   313,  -536,  -330,    11,  -536,    -2,     9,
      39,   726,  -536,  -114,  -536,  -536,  -536,  -536,  -536,  -536,
    -536,  -536,  -536,  -536,   -43,   368,   530,  -536
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -428
static const yytype_int16 yytable[] =
{
     114,    57,   199,   201,   458,   456,   232,   554,   593,   628,
     594,   595,     3,   596,   703,   139,   638,   226,   673,   704,
     122,   325,   129,   126,   795,   136,   301,   142,   138,   145,
     309,   228,   147,  -269,   153,   123,   229,   160,   284,   525,
     622,   167,   196,   623,   175,   143,   149,   183,   150,   233,
     234,   192,   195,   186,   147,   147,   202,   203,   204,   205,
     206,   207,   208,   365,   130,   366,   131,   637,   227,   369,
     623,   373,   326,   114,   213,   624,   114,   215,   300,   355,
     358,   114,   217,   625,   626,   674,   587,   705,  -199,   588,
     151,   589,   796,   597,   221,   621,   114,   225,   310,   311,
    -269,   675,   624,   229,   341,   285,   353,   283,   186,   362,
     625,   626,     4,   377,     5,     6,     7,     8,     9,    10,
     -94,    12,    13,    14,    15,   627,    16,   477,   478,    17,
    -199,   382,   363,    18,   286,   402,    20,    21,    22,    23,
      24,   658,    25,   210,   379,   211,    28,    29,    30,   114,
     313,    32,   627,   247,    35,  -199,   282,   212,  -403,    38,
      39,    40,   667,    41,   506,    42,  -236,   247,   356,   507,
     186,   342,   606,   291,   186,   354,   311,   444,   247,  -279,
     132,   229,   133,   659,   381,    43,    44,   383,   184,    45,
      46,   452,   171,   134,   172,   370,   346,   371,   349,   311,
      47,    48,  -427,   401,   668,    49,   292,   282,   660,  -279,
     293,   380,   455,    50,  -236,    51,    52,    53,  -427,   567,
    -403,   568,  -403,   504,   512,   528,   683,   756,   445,   669,
     185,   186,   462,  -236,   303,   247,   173,   357,   322,   372,
     468,   607,   385,   471,   387,   388,   389,   390,   391,   392,
     393,   394,   395,   396,   380,   400,   397,   305,   453,   403,
     405,   406,   407,   408,   409,   410,   411,   412,   413,   414,
     415,   249,   417,   418,   419,   420,   421,   422,   423,   424,
     425,   426,   427,   428,   429,   430,   431,   249,   613,   433,
     186,   380,   311,   311,   757,   436,   435,   643,   797,   186,
     583,   698,  -170,   186,   287,   555,   484,   332,   485,   437,
     294,   440,   438,   298,  -166,   306,   137,   301,   350,   442,
     443,     8,   124,   278,   279,   280,   281,     8,   486,   487,
     335,   543,   154,   302,   275,   276,   277,   155,  -403,   278,
     279,    21,    22,   249,  -170,   308,   288,  -403,   460,   602,
    -169,  -403,   295,   562,   559,   299,   564,   354,   280,   281,
     465,   466,   114,   114,   114,   501,   607,   114,   473,   474,
     607,   114,   336,   114,   505,   738,   655,   739,   763,   741,
     158,   742,   159,     6,     7,     8,     9,    10,   744,     8,
     745,   276,   277,   513,   608,   278,   279,   662,   233,   234,
     233,   234,   448,   249,  -283,    21,    22,   502,   208,    21,
      22,   530,   588,   249,   589,   156,    30,   614,   656,   740,
     157,   653,   654,   743,   312,   314,   359,   360,   665,   734,
     793,    41,   746,    42,   449,   595,   316,   596,   318,   663,
     644,   534,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,    43,    44,   278,   279,   272,   273,   274,
     275,   276,   277,   320,   328,   278,   279,   763,    47,    48,
     666,   735,   457,    49,   334,     6,     7,   563,     9,    10,
     565,    50,   611,    51,    52,    53,   329,   570,   154,   330,
     163,   678,   491,   680,   492,   164,   165,   331,   576,   333,
    -166,   577,   114,   578,   579,   580,   337,   699,   176,   340,
     177,   721,   169,   178,   493,   487,   180,   170,   552,   343,
     708,   181,   197,   322,   198,     6,     7,     8,     9,    10,
     344,   347,   361,   156,   610,   208,  -169,   249,   114,   620,
     180,   376,   615,   378,   441,   617,   618,    21,    22,   386,
       8,   416,   577,   114,   578,   579,   580,   434,    30,   454,
     461,   469,   472,   510,   767,   732,   511,   769,   538,   542,
     736,   536,   771,    41,   773,    42,   267,   268,   269,   270,
     271,   272,   273,   274,   275,   276,   277,   546,   449,   278,
     279,   547,   550,   507,   556,    43,    44,   765,   557,   558,
     114,   560,   566,   571,   574,   575,   584,   586,   672,   592,
      47,    48,   600,   496,   603,    49,   633,   114,   692,   634,
     649,   646,   636,    50,   249,    51,    52,    53,   647,   650,
     652,   801,   661,   657,   802,   791,   670,   803,   671,   676,
     677,   681,   805,   685,   700,   701,   181,   798,   707,   702,
     710,   114,   114,   717,   728,   747,   712,   713,   731,   750,
     753,   114,   722,   754,   755,   269,   270,   271,   272,   273,
     274,   275,   276,   277,   758,   127,   278,   279,     6,     7,
       8,     9,    10,   764,   766,   775,   730,     6,     7,   759,
       9,    10,   114,   692,   781,   776,   777,   737,   778,   784,
      21,    22,   787,   790,   799,   800,   804,   807,   249,   751,
     760,    30,   809,   810,   114,   768,   815,   114,   770,   114,
     816,   817,   114,   772,   114,   774,    41,   818,    42,   714,
     748,   494,   720,   729,   727,   601,   733,   545,   114,   783,
     808,   114,   786,   709,   114,   789,   639,   640,    43,    44,
     270,   271,   272,   273,   274,   275,   276,   277,   168,   641,
     278,   279,   794,    47,    48,   114,   553,   114,    49,   114,
     531,   114,     0,     0,     0,   398,    50,     0,    51,    52,
      53,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   114,
     114,   114,     0,   114,    -2,     4,     0,     5,     6,     7,
       8,     9,    10,    11,    12,    13,    14,    15,     0,    16,
       0,     0,    17,     0,     0,     0,    18,    19,   249,    20,
      21,    22,    23,    24,     0,    25,    26,     0,    27,    28,
      29,    30,     0,    31,    32,    33,    34,    35,    36,     0,
      37,     0,    38,    39,    40,     0,    41,     0,    42,     0,
     262,   263,   264,   265,   266,     0,     0,   267,   268,   269,
     270,   271,   272,   273,   274,   275,   276,   277,    43,    44,
     278,   279,    45,    46,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    47,    48,     0,     0,     0,    49,     0,
       0,     0,     0,     0,     0,     0,    50,     0,    51,    52,
      53,     4,     0,     5,     6,     7,     8,     9,    10,  -124,
      12,    13,    14,    15,     0,    16,     0,     0,    17,   689,
     690,   691,    18,     0,     0,    20,    21,    22,    23,    24,
       0,    25,   210,     0,   211,    28,    29,    30,     0,     0,
      32,     0,     0,    35,     0,     0,   212,     0,    38,    39,
      40,     0,    41,     0,    42,   233,   234,   235,   236,   237,
     238,   239,   240,   241,   242,   243,   244,     0,     0,     0,
       0,   245,     0,     0,    43,    44,     0,     0,    45,    46,
       0,     0,     0,     0,   246,     0,     0,     0,     0,    47,
      48,     0,     0,     0,    49,     0,     0,     0,     0,     0,
       0,     0,    50,     0,    51,    52,    53,     4,     0,     5,
       6,     7,     8,     9,    10,  -163,    12,    13,    14,    15,
       0,    16,     0,     0,    17,     0,     0,     0,    18,  -163,
    -163,    20,    21,    22,    23,    24,     0,    25,   210,     0,
     211,    28,    29,    30,     0,     0,    32,     0,     0,    35,
       0,  -163,   212,     0,    38,    39,    40,     0,    41,     0,
      42,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      43,    44,     0,     0,    45,    46,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    47,    48,     0,     0,     0,
      49,     0,     0,     0,     0,     0,     0,     0,    50,     0,
      51,    52,    53,     4,     0,     5,     6,     7,     8,     9,
      10,  -159,    12,    13,    14,    15,     0,    16,     0,     0,
      17,     0,     0,     0,    18,  -159,  -159,    20,    21,    22,
      23,    24,     0,    25,   210,     0,   211,    28,    29,    30,
       0,     0,    32,     0,     0,    35,     0,  -159,   212,     0,
      38,    39,    40,     0,    41,     0,    42,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    43,    44,     0,     0,
      45,    46,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    47,    48,     0,     0,     0,    49,     0,     0,     0,
       0,     0,     0,     0,    50,     0,    51,    52,    53,     4,
       0,     5,     6,     7,     8,     9,    10,  -194,    12,    13,
      14,    15,     0,    16,     0,     0,    17,     0,     0,     0,
      18,  -194,  -194,    20,    21,    22,    23,    24,     0,    25,
     210,     0,   211,    28,    29,    30,     0,     0,    32,     0,
       0,    35,     0,  -194,   212,     0,    38,    39,    40,     0,
      41,     0,    42,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    43,    44,     0,     0,    45,    46,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    47,    48,     0,
       0,     0,    49,     0,     0,     0,     0,     0,     0,     0,
      50,     0,    51,    52,    53,     4,     0,     5,     6,     7,
       8,     9,    10,  -190,    12,    13,    14,    15,     0,    16,
       0,     0,    17,     0,     0,     0,    18,  -190,  -190,    20,
      21,    22,    23,    24,     0,    25,   210,     0,   211,    28,
      29,    30,     0,     0,    32,     0,     0,    35,     0,  -190,
     212,     0,    38,    39,    40,     0,    41,     0,    42,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    43,    44,
       0,     0,    45,    46,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    47,    48,     0,     0,     0,    49,     0,
       0,     0,     0,     0,     0,     0,    50,     0,    51,    52,
      53,     4,     0,     5,     6,     7,     8,     9,    10,  -210,
      12,    13,    14,    15,     0,    16,     0,     0,    17,     0,
       0,     0,    18,     0,     0,    20,    21,    22,    23,    24,
     496,    25,   210,     0,   211,    28,    29,    30,     0,     0,
      32,     0,     0,    35,     0,     0,   212,     0,    38,    39,
      40,     0,    41,     0,    42,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    43,    44,     0,     0,    45,    46,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    47,
      48,     0,     0,     0,    49,     0,     0,     0,     0,     0,
       0,     0,    50,     0,    51,    52,    53,     4,     0,     5,
       6,     7,     8,     9,    10,  -214,    12,    13,    14,    15,
       0,    16,     0,     0,    17,     0,     0,     0,    18,     0,
       0,    20,    21,    22,    23,    24,  -214,    25,   210,     0,
     211,    28,    29,    30,     0,     0,    32,     0,     0,    35,
       0,     0,   212,     0,    38,    39,    40,     0,    41,     0,
      42,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      43,    44,     0,     0,    45,    46,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    47,    48,     0,     0,     0,
      49,     0,     0,     0,     0,     0,     0,     0,    50,     0,
      51,    52,    53,     4,     0,     5,     6,     7,     8,     9,
      10,   475,    12,    13,    14,    15,     0,    16,     0,     0,
      17,     0,     0,     0,    18,     0,     0,    20,    21,    22,
      23,    24,     0,    25,   210,     0,   211,    28,    29,    30,
       0,     0,    32,     0,     0,    35,     0,     0,   212,     0,
      38,    39,    40,     0,    41,     0,    42,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    43,    44,     0,     0,
      45,    46,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    47,    48,     0,     0,     0,    49,     0,     0,     0,
       0,     0,     0,     0,    50,     0,    51,    52,    53,     4,
       0,     5,     6,     7,     8,     9,    10,   483,    12,    13,
      14,    15,     0,    16,     0,     0,    17,     0,     0,     0,
      18,     0,     0,    20,    21,    22,    23,    24,     0,    25,
     210,     0,   211,    28,    29,    30,     0,     0,    32,     0,
       0,    35,     0,     0,   212,     0,    38,    39,    40,     0,
      41,     0,    42,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    43,    44,     0,     0,    45,    46,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    47,    48,     0,
       0,     0,    49,     0,     0,     0,     0,     0,     0,     0,
      50,     0,    51,    52,    53,     4,     0,     5,     6,     7,
       8,     9,    10,   503,    12,    13,    14,    15,     0,    16,
       0,     0,    17,     0,     0,     0,    18,     0,     0,    20,
      21,    22,    23,    24,     0,    25,   210,     0,   211,    28,
      29,    30,     0,     0,    32,     0,     0,    35,     0,     0,
     212,     0,    38,    39,    40,     0,    41,     0,    42,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    43,    44,
       0,     0,    45,    46,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    47,    48,     0,     0,     0,    49,     0,
       0,     0,     0,     0,     0,     0,    50,     0,    51,    52,
      53,     4,     0,     5,     6,     7,     8,     9,    10,   604,
      12,    13,    14,    15,     0,    16,     0,     0,    17,     0,
       0,     0,    18,     0,     0,    20,    21,    22,    23,    24,
       0,    25,   210,     0,   211,    28,    29,    30,     0,     0,
      32,     0,     0,    35,     0,     0,   212,     0,    38,    39,
      40,     0,    41,     0,    42,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    43,    44,     0,     0,    45,    46,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    47,
      48,     0,     0,     0,    49,     0,     0,     0,     0,     0,
       0,     0,    50,     0,    51,    52,    53,     4,     0,     5,
       6,     7,     8,     9,    10,   645,    12,    13,    14,    15,
       0,    16,     0,     0,    17,     0,     0,     0,    18,     0,
       0,    20,    21,    22,    23,    24,     0,    25,   210,     0,
     211,    28,    29,    30,     0,     0,    32,     0,     0,    35,
       0,     0,   212,     0,    38,    39,    40,     0,    41,     0,
      42,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      43,    44,     0,     0,    45,    46,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    47,    48,     0,     0,     0,
      49,     0,     0,     0,     0,     0,     0,     0,    50,     0,
      51,    52,    53,     4,     0,     5,     6,     7,     8,     9,
      10,   -97,    12,    13,    14,    15,     0,    16,     0,     0,
      17,     0,     0,     0,    18,     0,     0,    20,    21,    22,
      23,    24,     0,    25,   210,     0,   211,    28,    29,    30,
       0,     0,    32,     0,     0,    35,     0,     0,   212,     0,
      38,    39,    40,     0,    41,     0,    42,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    43,    44,     0,     0,
      45,    46,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    47,    48,     0,     0,     0,    49,     0,     0,     0,
       0,     0,     0,     0,    50,     0,    51,    52,    53,     4,
       0,     5,     6,     7,     8,     9,    10,  -172,    12,    13,
      14,    15,     0,    16,     0,     0,    17,     0,     0,     0,
      18,     0,     0,    20,    21,    22,    23,    24,     0,    25,
     210,     0,   211,    28,    29,    30,     0,     0,    32,     0,
       0,    35,     0,     0,   212,     0,    38,    39,    40,     0,
      41,     0,    42,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    43,    44,     0,     0,    45,    46,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    47,    48,     0,
       0,     0,    49,     0,     0,     0,     0,     0,     0,     0,
      50,     0,    51,    52,    53,     4,     0,     5,     6,     7,
       8,     9,    10,   811,    12,    13,    14,    15,     0,    16,
       0,     0,    17,     0,     0,     0,    18,     0,     0,    20,
      21,    22,    23,    24,     0,    25,   210,     0,   211,    28,
      29,    30,     0,     0,    32,     0,     0,    35,     0,     0,
     212,     0,    38,    39,    40,     0,    41,     0,    42,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    43,    44,
       0,     0,    45,    46,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    47,    48,     0,     0,     0,    49,     0,
       0,     0,     0,     0,     0,     0,    50,     0,    51,    52,
      53,     4,     0,     5,     6,     7,     8,     9,    10,   812,
      12,    13,    14,    15,     0,    16,     0,     0,    17,     0,
       0,     0,    18,     0,     0,    20,    21,    22,    23,    24,
       0,    25,   210,     0,   211,    28,    29,    30,     0,     0,
      32,     0,     0,    35,     0,     0,   212,     0,    38,    39,
      40,     0,    41,     0,    42,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    43,    44,     0,     0,    45,    46,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    47,
      48,     0,     0,     0,    49,     0,     0,     0,     0,     0,
       0,     0,    50,     0,    51,    52,    53,     4,     0,     5,
       6,     7,     8,     9,    10,   813,    12,    13,    14,    15,
       0,    16,     0,     0,    17,     0,     0,     0,    18,     0,
       0,    20,    21,    22,    23,    24,     0,    25,   210,     0,
     211,    28,    29,    30,     0,     0,    32,     0,     0,    35,
       0,     0,   212,     0,    38,    39,    40,     0,    41,     0,
      42,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      43,    44,     0,     0,    45,    46,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    47,    48,     0,     0,     0,
      49,     0,     0,     0,     0,     0,     0,     0,    50,     0,
      51,    52,    53,     4,     0,     5,     6,     7,     8,     9,
      10,   814,    12,    13,    14,    15,     0,    16,     0,     0,
      17,     0,     0,     0,    18,     0,     0,    20,    21,    22,
      23,    24,     0,    25,   210,     0,   211,    28,    29,    30,
       0,     0,    32,     0,     0,    35,     0,     0,   212,     0,
      38,    39,    40,     0,    41,     0,    42,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    43,    44,     0,     0,
      45,    46,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    47,    48,     0,     0,     0,    49,     0,     0,     0,
       0,     0,     0,     0,    50,     0,    51,    52,    53,     4,
       0,     5,     6,     7,     8,     9,    10,     0,    12,    13,
      14,    15,     0,    16,     0,     0,    17,     0,     0,     0,
      18,     0,     0,    20,    21,    22,    23,    24,     0,    25,
     210,     0,   211,    28,    29,    30,     0,     0,    32,     0,
       0,    35,     0,     0,   212,     0,    38,    39,    40,     0,
      41,     0,    42,     0,   135,     0,     0,     6,     7,     8,
       9,    10,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    43,    44,     0,     0,    45,    46,     0,    21,
      22,     0,     0,     0,     0,     0,     0,    47,    48,     0,
      30,     0,    49,     0,     0,     0,     0,     0,     0,     0,
      50,     0,    51,    52,    53,    41,     0,    42,     0,   141,
       0,     0,     6,     7,     8,     9,    10,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    43,    44,     0,
       0,     0,     0,     0,    21,    22,     0,     0,     0,     0,
       0,     0,    47,    48,     0,    30,     0,    49,     0,     0,
       0,     0,     0,     0,     0,    50,     0,    51,    52,    53,
      41,     0,    42,     0,   144,     0,     0,     6,     7,     8,
       9,    10,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    43,    44,     0,     0,     0,     0,     0,    21,
      22,     0,     0,     0,     0,     0,     0,    47,    48,     0,
      30,     0,    49,     0,     0,     0,     0,     0,     0,     0,
      50,     0,    51,    52,    53,    41,     0,    42,     0,   146,
       0,     0,     6,     7,     8,     9,    10,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    43,    44,     0,
       0,     0,     0,     0,    21,    22,     0,     0,     0,     0,
       0,     0,    47,    48,     0,    30,     0,    49,     0,     0,
       0,     0,     0,     0,     0,    50,     0,    51,    52,    53,
      41,     0,    42,     0,   152,     0,     0,     6,     7,     8,
       9,    10,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    43,    44,     0,     0,     0,     0,     0,    21,
      22,     0,     0,     0,     0,     0,     0,    47,    48,     0,
      30,     0,    49,     0,     0,     0,     0,     0,     0,     0,
      50,     0,    51,    52,    53,    41,     0,    42,     0,   166,
       0,     0,     6,     7,     8,     9,    10,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    43,    44,     0,
       0,     0,     0,     0,    21,    22,     0,     0,     0,     0,
       0,     0,    47,    48,     0,    30,     0,    49,     0,     0,
       0,     0,     0,     0,     0,    50,     0,    51,    52,    53,
      41,     0,    42,     0,   174,     0,     0,     6,     7,     8,
       9,    10,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    43,    44,     0,     0,     0,     0,     0,    21,
      22,     0,     0,     0,     0,     0,     0,    47,    48,     0,
      30,     0,    49,     0,     0,     0,     0,     0,     0,     0,
      50,     0,    51,    52,    53,    41,     0,    42,     0,   182,
       0,     0,     6,     7,     8,     9,    10,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    43,    44,     0,
       0,     0,     0,     0,    21,    22,     0,     0,     0,     0,
       0,     0,    47,    48,     0,    30,     0,    49,     0,     0,
       0,     0,     0,     0,     0,    50,     0,    51,    52,    53,
      41,     0,    42,     0,   404,     0,     0,     6,     7,     8,
       9,    10,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    43,    44,     0,     0,     0,     0,     0,    21,
      22,     0,     0,     0,     0,     0,     0,    47,    48,     0,
      30,     0,    49,     0,     0,     0,     0,     0,     0,     0,
      50,     0,    51,    52,    53,    41,     0,    42,     0,   439,
       0,     0,     6,     7,     8,     9,    10,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    43,    44,     0,
       0,     0,     0,     0,    21,    22,     0,     0,     0,     0,
       0,     0,    47,    48,     0,    30,     0,    49,     0,     0,
       0,     0,     0,     0,     0,    50,     0,    51,    52,    53,
      41,     0,    42,     0,   459,     0,     0,     6,     7,     8,
       9,    10,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    43,    44,     0,     0,     0,     0,     0,    21,
      22,     0,     0,     0,     0,     0,     0,    47,    48,     0,
      30,     0,    49,     0,     0,     0,     0,     0,     0,     0,
      50,     0,    51,    52,    53,    41,     0,    42,     0,   569,
       0,     0,     6,     7,     8,     9,    10,   200,     6,     7,
       8,     9,    10,     0,     0,     0,     0,    43,    44,     0,
       0,     0,     0,     0,    21,    22,     0,     0,     0,     0,
      21,    22,    47,    48,     0,    30,     0,    49,     0,     0,
       0,    30,     0,     0,     0,    50,     0,    51,    52,    53,
      41,     0,    42,     0,     0,     0,    41,     0,    42,     0,
       0,     0,   612,     6,     7,     8,     9,    10,     0,     0,
       0,     0,    43,    44,     0,     0,     0,     0,    43,    44,
       0,     0,     0,     0,     0,    21,    22,    47,    48,     0,
       0,     0,    49,    47,    48,     0,    30,     0,    49,     0,
      50,   686,    51,    52,    53,     0,    50,     0,    51,    52,
      53,    41,     0,    42,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    43,    44,     0,     0,     0,   248,     0,
       0,     0,     0,   687,     0,     0,     0,     0,    47,    48,
     249,     0,     0,    49,     0,     0,     0,     0,     0,     0,
       0,    50,     0,    51,    52,    53,   688,     0,     0,   251,
     252,   253,     0,     0,   254,   255,   256,   257,   258,   259,
     260,   261,   262,   263,   264,   265,   266,   249,     0,   267,
     268,   269,   270,   271,   272,   273,   274,   275,   276,   277,
     289,   250,   278,   279,     0,     0,   251,   252,   253,     0,
       0,   254,   255,   256,   257,   258,   259,   260,   261,   262,
     263,   264,   265,   266,     0,     0,   267,   268,   269,   270,
     271,   272,   273,   274,   275,   276,   277,     0,     0,   278,
     279,     0,   290,     0,     0,     0,     0,     0,     0,   249,
       0,     0,     0,     0,     0,     0,     0,   296,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   251,   252,
     253,     0,     0,   254,   255,   256,   257,   258,   259,   260,
     261,   262,   263,   264,   265,   266,     0,     0,   267,   268,
     269,   270,   271,   272,   273,   274,   275,   276,   277,   297,
       0,   278,   279,     0,     0,     0,   249,     0,     0,     0,
       0,     0,     0,     0,   539,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   251,   252,   253,     0,     0,
     254,   255,   256,   257,   258,   259,   260,   261,   262,   263,
     264,   265,   266,     0,     0,   267,   268,   269,   270,   271,
     272,   273,   274,   275,   276,   277,   540,     0,   278,   279,
       0,     0,     0,   249,     0,     0,     0,     0,     0,     0,
       0,   779,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   251,   252,   253,     0,     0,   254,   255,   256,
     257,   258,   259,   260,   261,   262,   263,   264,   265,   266,
       0,     0,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   780,     0,   278,   279,     0,     0,     0,
     249,   304,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   251,
     252,   253,     0,     0,   254,   255,   256,   257,   258,   259,
     260,   261,   262,   263,   264,   265,   266,     0,   307,   267,
     268,   269,   270,   271,   272,   273,   274,   275,   276,   277,
     249,     0,   278,   279,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   251,
     252,   253,     0,     0,   254,   255,   256,   257,   258,   259,
     260,   261,   262,   263,   264,   265,   266,   249,   315,   267,
     268,   269,   270,   271,   272,   273,   274,   275,   276,   277,
       0,     0,   278,   279,     0,     0,   251,   252,   253,     0,
       0,   254,   255,   256,   257,   258,   259,   260,   261,   262,
     263,   264,   265,   266,     0,   321,   267,   268,   269,   270,
     271,   272,   273,   274,   275,   276,   277,   249,     0,   278,
     279,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   251,   252,   253,     0,
       0,   254,   255,   256,   257,   258,   259,   260,   261,   262,
     263,   264,   265,   266,   249,   338,   267,   268,   269,   270,
     271,   272,   273,   274,   275,   276,   277,     0,     0,   278,
     279,     0,     0,   251,   252,   253,     0,     0,   254,   255,
     256,   257,   258,   259,   260,   261,   262,   263,   264,   265,
     266,     0,   345,   267,   268,   269,   270,   271,   272,   273,
     274,   275,   276,   277,   249,     0,   278,   279,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   251,   252,   253,     0,     0,   254,   255,
     256,   257,   258,   259,   260,   261,   262,   263,   339,   265,
     266,   249,   186,   267,   268,   269,   270,   271,   272,   273,
     274,   275,   276,   277,     0,     0,   278,   279,     0,     0,
     251,   252,   253,     0,     0,   254,   255,   256,   257,   258,
     259,   260,   261,   262,   263,   264,   265,   266,     0,   515,
     267,   268,   269,   270,   271,   272,   273,   274,   275,   276,
     277,   249,     0,   278,   279,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     251,   252,   253,     0,     0,   254,   255,   256,   257,   258,
     259,   260,   261,   262,   263,   264,   265,   266,   249,   516,
     267,   268,   269,   270,   271,   272,   273,   274,   275,   276,
     277,     0,     0,   278,   279,     0,     0,   251,   252,   253,
       0,     0,   254,   255,   256,   257,   258,   259,   260,   261,
     262,   263,   264,   265,   266,     0,   517,   267,   268,   269,
     270,   271,   272,   273,   274,   275,   276,   277,   249,     0,
     278,   279,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,   251,   252,   253,
       0,     0,   254,   255,   256,   257,   258,   259,   260,   261,
     262,   263,   264,   265,   266,   249,   518,   267,   268,   269,
     270,   271,   272,   273,   274,   275,   276,   277,     0,     0,
     278,   279,     0,     0,   251,   252,   253,     0,     0,   254,
     255,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,     0,   519,   267,   268,   269,   270,   271,   272,
     273,   274,   275,   276,   277,   249,     0,   278,   279,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   251,   252,   253,     0,     0,   254,
     255,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   249,   520,   267,   268,   269,   270,   271,   272,
     273,   274,   275,   276,   277,     0,     0,   278,   279,     0,
       0,   251,   252,   253,     0,     0,   254,   255,   256,   257,
     258,   259,   260,   261,   262,   263,   264,   265,   266,     0,
     521,   267,   268,   269,   270,   271,   272,   273,   274,   275,
     276,   277,   249,     0,   278,   279,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   251,   252,   253,     0,     0,   254,   255,   256,   257,
     258,   259,   260,   261,   262,   263,   264,   265,   266,   249,
     522,   267,   268,   269,   270,   271,   272,   273,   274,   275,
     276,   277,     0,     0,   278,   279,     0,     0,   251,   252,
     253,     0,     0,   254,   255,   256,   257,   258,   259,   260,
     261,   262,   263,   264,   265,   266,     0,   523,   267,   268,
     269,   270,   271,   272,   273,   274,   275,   276,   277,   249,
       0,   278,   279,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   251,   252,
     253,     0,     0,   254,   255,   256,   257,   258,   259,   260,
     261,   262,   263,   264,   265,   266,   249,   524,   267,   268,
     269,   270,   271,   272,   273,   274,   275,   276,   277,     0,
       0,   278,   279,     0,     0,   251,   252,   253,     0,     0,
     254,   255,   256,   257,   258,   259,   260,   261,   262,   263,
     264,   265,   266,     0,   527,   267,   268,   269,   270,   271,
     272,   273,   274,   275,   276,   277,   249,     0,   278,   279,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   251,   252,   253,     0,     0,
     254,   255,   256,   257,   258,   259,   260,   261,   262,   263,
     264,   265,   266,   249,   532,   267,   268,   269,   270,   271,
     272,   273,   274,   275,   276,   277,     0,     0,   278,   279,
       0,     0,   251,   252,   253,     0,     0,   254,   255,   256,
     257,   258,   259,   260,   261,   262,   263,   264,   265,   266,
       0,   541,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   249,     0,   278,   279,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   251,   252,   253,     0,     0,   254,   255,   256,
     257,   258,   259,   260,   261,   262,   263,   264,   265,   266,
     249,   651,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,     0,     0,   278,   279,     0,     0,   251,
     252,   253,     0,     0,   254,   255,   256,   257,   258,   259,
     260,   261,   262,   263,   264,   265,   266,     0,   682,   267,
     268,   269,   270,   271,   272,   273,   274,   275,   276,   277,
     249,     0,   278,   279,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   251,
     252,   253,     0,     0,   254,   255,   256,   257,   258,   259,
     260,   261,   262,   263,   264,   265,   266,   249,   792,   267,
     268,   269,   270,   271,   272,   273,   274,   275,   276,   277,
       0,     0,   278,   279,     0,     0,   251,   252,   253,     0,
       0,   254,   255,   256,   257,   258,   259,   260,   261,   262,
     263,   264,   265,   266,     0,   806,   267,   268,   269,   270,
     271,   272,   273,   274,   275,   276,   277,   249,     0,   278,
     279,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   251,   252,   253,     0,
       0,   254,   255,   256,   257,   258,   259,   260,   261,   262,
     263,   264,   265,   266,   249,     0,   267,   268,   269,   270,
     271,   272,   273,   274,   275,   276,   277,     0,     0,   278,
     279,     0,     0,   251,   252,   253,     0,     0,   254,   255,
     256,   257,   258,   259,   260,   261,   262,   263,   264,   265,
     266,     0,     0,   267,   268,   269,   270,   271,   272,   273,
     274,   275,   276,   277,     0,     0,   278,   279,     6,     7,
       8,     9,    10,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      21,    22,     6,     7,     8,     9,    10,     0,     0,     0,
       0,    30,     0,     0,     0,     0,     0,     0,     0,   189,
       0,     0,     0,     0,    21,    22,    41,   190,    42,     0,
       0,     0,     0,     0,     0,    30,     6,     7,     8,     9,
      10,   191,     0,   189,     0,     0,     0,     0,    43,    44,
      41,     0,    42,     0,     0,     0,     0,     0,    21,    22,
       0,     0,     0,    47,    48,     0,     0,     0,    49,    30,
       0,     0,    43,    44,     0,     0,    50,   189,    51,    52,
      53,     0,     0,     0,    41,     0,    42,    47,    48,     0,
       0,     0,    49,     0,     0,     0,   384,     0,     0,     0,
      50,     0,    51,    52,    53,     0,    43,    44,     0,     6,
       7,     8,     9,    10,     0,     0,     0,     0,     0,     0,
       0,    47,    48,     0,     0,     0,    49,     0,     0,     0,
     432,    21,    22,     0,    50,     0,    51,    52,    53,     0,
       0,     0,    30,     6,     7,     8,     9,    10,     0,     0,
       6,     7,     8,     9,    10,     0,     0,    41,   348,    42,
       0,     0,     0,     0,     0,    21,    22,     0,     0,     0,
       0,     0,    21,    22,     0,     0,    30,     0,     0,    43,
      44,     0,     0,    30,     0,     0,     0,     0,     0,     0,
       0,    41,   464,    42,    47,    48,     0,   529,    41,    49,
      42,     0,     0,     0,     0,     0,     0,    50,     0,    51,
      52,    53,     0,    43,    44,     0,     0,     0,     0,     0,
      43,    44,     0,     0,     0,     0,     0,     0,    47,    48,
       0,     0,     0,    49,     0,    47,    48,     0,     0,     0,
      49,    50,     0,    51,    52,    53,     0,     0,    50,     0,
      51,    52,    53,     6,     7,     8,     9,    10,     0,     6,
       7,     8,     9,    10,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    21,    22,     0,     0,     0,
       0,    21,    22,     0,     0,     0,    30,     0,     0,     0,
       0,     0,    30,     0,     0,     0,     0,     0,     0,     0,
       0,    41,     0,    42,     0,     0,     0,    41,     0,    42,
       0,     0,     0,     0,     6,     7,     8,     9,    10,     0,
       0,     0,     0,    43,    44,     0,     0,     0,     0,    43,
      44,     0,     0,     0,     0,     0,    21,    22,    47,    48,
       0,     0,     0,    49,    47,    48,     0,    30,     0,    49,
       0,    50,     0,    51,    52,    53,     0,    50,     0,    51,
      52,   399,    41,     0,    42,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    43,    44,     0,     0,     0,     0,
       0,   351,     0,     0,     0,     0,     0,     0,   249,    47,
      48,     0,     0,     0,    49,     0,     0,     0,     0,     0,
       0,     0,    50,   352,    51,    52,   526,   251,   252,   253,
       0,     0,   254,   255,   256,   257,   258,   259,   260,   261,
     262,   263,   264,   265,   266,     0,     0,   267,   268,   269,
     270,   271,   272,   273,   274,   275,   276,   277,   351,     0,
     278,   279,     0,     0,     0,   249,   514,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   251,   252,   253,     0,     0,   254,
     255,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,     0,     0,   267,   268,   269,   270,   271,   272,
     273,   274,   275,   276,   277,   351,     0,   278,   279,     0,
       0,     0,   249,   535,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   251,   252,   253,     0,     0,   254,   255,   256,   257,
     258,   259,   260,   261,   262,   263,   264,   265,   266,     0,
       0,   267,   268,   269,   270,   271,   272,   273,   274,   275,
     276,   277,   533,     0,   278,   279,     0,     0,     0,   249,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   251,   252,
     253,     0,     0,   254,   255,   256,   257,   258,   259,   260,
     261,   262,   263,   264,   265,   266,   249,   463,   267,   268,
     269,   270,   271,   272,   273,   274,   275,   276,   277,     0,
       0,   278,   279,     0,     0,   251,   252,   253,     0,     0,
     254,   255,   256,   257,   258,   259,   260,   261,   262,   263,
     264,   265,   266,   249,     0,   267,   268,   269,   270,   271,
     272,   273,   274,   275,   276,   277,     0,     0,   278,   279,
     537,     0,   251,   252,   253,     0,     0,   254,   255,   256,
     257,   258,   259,   260,   261,   262,   263,   264,   265,   266,
     249,   561,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,     0,     0,   278,   279,     0,     0,   251,
     252,   253,     0,     0,   254,   255,   256,   257,   258,   259,
     260,   261,   262,   263,   264,   265,   266,   249,   609,   267,
     268,   269,   270,   271,   272,   273,   274,   275,   276,   277,
       0,     0,   278,   279,     0,     0,   251,   252,   253,     0,
       0,   254,   255,   256,   257,   258,   259,   260,   261,   262,
     263,   264,   265,   266,   249,   616,   267,   268,   269,   270,
     271,   272,   273,   274,   275,   276,   277,     0,     0,   278,
     279,     0,     0,   251,   252,   253,     0,     0,   254,   255,
     256,   257,   258,   259,   260,   261,   262,   263,   264,   265,
     266,     0,     0,   267,   268,   269,   270,   271,   272,   273,
     274,   275,   276,   277,   249,     0,   278,   279,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   648,
       0,     0,     0,   251,   252,   253,     0,     0,   254,   255,
     256,   257,   258,   259,   260,   261,   262,   263,   264,   265,
     266,   249,     0,   267,   268,   269,   270,   271,   272,   273,
     274,   275,   276,   277,     0,     0,   278,   279,     0,     0,
     251,   252,   253,     0,     0,   254,   255,   256,   257,   258,
     259,   260,   261,   262,   263,   264,   265,   266,   249,     0,
     267,   268,   269,   270,   271,   272,   273,   274,   275,   276,
     277,     0,     0,   278,   279,     0,     0,     0,   252,   253,
       0,     0,   254,   255,   256,   257,   258,   259,   260,   261,
     262,   263,   264,   265,   266,   249,     0,   267,   268,   269,
     270,   271,   272,   273,   274,   275,   276,   277,     0,     0,
     278,   279,     0,     0,     0,     0,   253,     0,     0,   254,
     255,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   249,     0,   267,   268,   269,   270,   271,   272,
     273,   274,   275,   276,   277,     0,     0,   278,   279,     0,
       0,     0,     0,     0,     0,     0,   254,   255,   256,   257,
     258,   259,   260,   261,   262,   263,   264,   265,   266,     0,
       0,   267,   268,   269,   270,   271,   272,   273,   274,   275,
     276,   277,     0,     0,   278,   279
};

static const yytype_int16 yycheck[] =
{
       2,     2,    45,    46,   334,   327,   106,   455,     1,   544,
       3,     4,     0,     6,     1,    17,   551,     1,     3,     6,
       3,     1,    13,    12,     6,    16,    70,    18,    17,    20,
       1,     1,    23,     3,    25,     3,     6,    28,     3,    83,
       3,    32,    44,     6,    35,     1,     1,    38,     3,    54,
      55,    42,    43,     3,    45,    46,    47,    48,    49,    50,
      51,    52,    53,   214,     1,   216,     3,     3,    52,   220,
       6,   222,    52,    75,    75,    38,    78,    78,    83,   193,
     194,    83,    83,    46,    47,    70,     1,    74,     3,     4,
      45,     6,    74,    86,    47,   543,    98,    98,    69,    70,
      70,    86,    38,     6,     3,    70,     1,     3,     3,     3,
      46,    47,     1,   227,     3,     4,     5,     6,     7,     8,
       9,    10,    11,    12,    13,    88,    15,    16,    17,    18,
      45,   231,     3,    22,    83,   249,    25,    26,    27,    28,
      29,     3,    31,    32,     3,    34,    35,    36,    37,   151,
     151,    40,    88,   114,    43,    70,   117,    46,    53,    48,
      49,    50,     3,    52,     1,    54,     3,   128,     1,     6,
       3,    70,     1,     3,     3,    70,    70,     1,   139,     3,
       1,     6,     3,    45,     9,    74,    75,     3,     3,    78,
      79,     3,     1,    14,     3,     1,   187,     3,   189,    70,
      89,    90,    70,   246,    45,    94,     3,   168,    70,    33,
       3,    70,   326,   102,    51,   104,   105,   106,    86,     1,
      53,     3,    51,   374,     3,     3,     3,     3,    52,    70,
      45,     3,   346,    70,     3,   196,    45,    70,     6,    45,
     354,    70,   233,   357,   235,   236,   237,   238,   239,   240,
     241,   242,   243,   244,    70,   246,   245,     3,    70,   250,
     251,   252,   253,   254,   255,   256,   257,   258,   259,   260,
     261,    52,   263,   264,   265,   266,   267,   268,   269,   270,
     271,   272,   273,   274,   275,   276,   277,    52,     1,   280,
       3,    70,    70,    70,    70,   286,   285,     1,    70,     3,
       1,     1,     3,     3,     3,   456,     1,     3,     3,   300,
       3,   302,   301,     3,     9,     3,     1,    70,    53,   310,
     311,     6,     1,   104,   105,    54,    55,     6,    23,    24,
       3,   445,     1,    86,    99,   100,   101,     6,    51,   104,
     105,    26,    27,    52,    45,     3,    45,    51,   339,   500,
      45,    51,    45,   467,     3,    45,   470,    70,    54,    55,
     351,   352,   364,   365,   366,     3,    70,   369,   359,   360,
      70,   373,    45,   375,   375,     1,     3,     3,   708,     1,
       1,     3,     3,     4,     5,     6,     7,     8,     1,     6,
       3,   100,   101,   384,   508,   104,   105,     3,    54,    55,
      54,    55,     1,    52,     3,    26,    27,    45,   399,    26,
      27,   402,     4,    52,     6,     1,    37,   531,    45,    45,
       6,   572,   573,    45,     3,     3,    82,    83,     3,     3,
     752,    52,    45,    54,    33,     4,     3,     6,     3,    45,
     554,   432,    91,    92,    93,    94,    95,    96,    97,    98,
      99,   100,   101,    74,    75,   104,   105,    96,    97,    98,
      99,   100,   101,     3,     3,   104,   105,   797,    89,    90,
      45,    45,     1,    94,    83,     4,     5,   468,     7,     8,
     471,   102,   525,   104,   105,   106,     3,   478,     1,     3,
       1,   605,     1,   607,     3,     6,     7,     3,     1,     3,
       9,     4,   504,     6,     7,     8,     3,   621,     1,     3,
       3,   662,     1,     6,    23,    24,     1,     6,     1,     3,
     634,     6,     1,     6,     3,     4,     5,     6,     7,     8,
       3,     3,     3,     1,   525,   526,    45,    52,   540,   540,
       1,     3,   533,     3,     3,   536,   537,    26,    27,     6,
       6,     6,     4,   555,     6,     7,     8,     6,    37,     3,
       6,    53,    53,     6,   715,   679,     3,   718,     3,     3,
     684,    83,   723,    52,   725,    54,    91,    92,    93,    94,
      95,    96,    97,    98,    99,   100,   101,     3,    33,   104,
     105,     6,     3,     6,     3,    74,    75,   711,     3,     3,
     602,    51,     3,     9,     3,     3,     9,     3,   597,     9,
      89,    90,     9,    30,     3,    94,     3,   619,   619,    52,
       3,    53,    70,   102,    52,   104,   105,   106,    53,     3,
       3,   782,     3,    69,   785,   749,     3,   788,     3,     3,
       3,    51,   793,    51,    83,     3,     6,   761,     9,     6,
       9,   653,   654,     4,     3,     9,    51,   648,    51,    51,
      83,   663,   663,     3,     6,    93,    94,    95,    96,    97,
      98,    99,   100,   101,     3,     1,   104,   105,     4,     5,
       6,     7,     8,     3,     3,     3,   675,     4,     5,     6,
       7,     8,   694,   694,     3,    45,    51,   688,    51,     3,
      26,    27,     3,     3,    51,     3,    51,     6,    52,   700,
      27,    37,    51,     3,   716,   716,     3,   719,   719,   721,
       3,     3,   724,   724,   726,   726,    52,     3,    54,   654,
     694,   368,   660,   674,   669,   498,   680,   447,   740,   740,
     797,   743,   743,   636,   746,   746,   551,   551,    74,    75,
      94,    95,    96,    97,    98,    99,   100,   101,    32,   551,
     104,   105,   753,    89,    90,   767,   453,   769,    94,   771,
     402,   773,    -1,    -1,    -1,   245,   102,    -1,   104,   105,
     106,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   801,
     802,   803,    -1,   805,     0,     1,    -1,     3,     4,     5,
       6,     7,     8,     9,    10,    11,    12,    13,    -1,    15,
      -1,    -1,    18,    -1,    -1,    -1,    22,    23,    52,    25,
      26,    27,    28,    29,    -1,    31,    32,    -1,    34,    35,
      36,    37,    -1,    39,    40,    41,    42,    43,    44,    -1,
      46,    -1,    48,    49,    50,    -1,    52,    -1,    54,    -1,
      84,    85,    86,    87,    88,    -1,    -1,    91,    92,    93,
      94,    95,    96,    97,    98,    99,   100,   101,    74,    75,
     104,   105,    78,    79,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    89,    90,    -1,    -1,    -1,    94,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,   102,    -1,   104,   105,
     106,     1,    -1,     3,     4,     5,     6,     7,     8,     9,
      10,    11,    12,    13,    -1,    15,    -1,    -1,    18,    19,
      20,    21,    22,    -1,    -1,    25,    26,    27,    28,    29,
      -1,    31,    32,    -1,    34,    35,    36,    37,    -1,    -1,
      40,    -1,    -1,    43,    -1,    -1,    46,    -1,    48,    49,
      50,    -1,    52,    -1,    54,    54,    55,    56,    57,    58,
      59,    60,    61,    62,    63,    64,    65,    -1,    -1,    -1,
      -1,    70,    -1,    -1,    74,    75,    -1,    -1,    78,    79,
      -1,    -1,    -1,    -1,    83,    -1,    -1,    -1,    -1,    89,
      90,    -1,    -1,    -1,    94,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,   102,    -1,   104,   105,   106,     1,    -1,     3,
       4,     5,     6,     7,     8,     9,    10,    11,    12,    13,
      -1,    15,    -1,    -1,    18,    -1,    -1,    -1,    22,    23,
      24,    25,    26,    27,    28,    29,    -1,    31,    32,    -1,
      34,    35,    36,    37,    -1,    -1,    40,    -1,    -1,    43,
      -1,    45,    46,    -1,    48,    49,    50,    -1,    52,    -1,
      54,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      74,    75,    -1,    -1,    78,    79,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    89,    90,    -1,    -1,    -1,
      94,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   102,    -1,
     104,   105,   106,     1,    -1,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    13,    -1,    15,    -1,    -1,
      18,    -1,    -1,    -1,    22,    23,    24,    25,    26,    27,
      28,    29,    -1,    31,    32,    -1,    34,    35,    36,    37,
      -1,    -1,    40,    -1,    -1,    43,    -1,    45,    46,    -1,
      48,    49,    50,    -1,    52,    -1,    54,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    74,    75,    -1,    -1,
      78,    79,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    89,    90,    -1,    -1,    -1,    94,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,   102,    -1,   104,   105,   106,     1,
      -1,     3,     4,     5,     6,     7,     8,     9,    10,    11,
      12,    13,    -1,    15,    -1,    -1,    18,    -1,    -1,    -1,
      22,    23,    24,    25,    26,    27,    28,    29,    -1,    31,
      32,    -1,    34,    35,    36,    37,    -1,    -1,    40,    -1,
      -1,    43,    -1,    45,    46,    -1,    48,    49,    50,    -1,
      52,    -1,    54,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    74,    75,    -1,    -1,    78,    79,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    89,    90,    -1,
      -1,    -1,    94,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
     102,    -1,   104,   105,   106,     1,    -1,     3,     4,     5,
       6,     7,     8,     9,    10,    11,    12,    13,    -1,    15,
      -1,    -1,    18,    -1,    -1,    -1,    22,    23,    24,    25,
      26,    27,    28,    29,    -1,    31,    32,    -1,    34,    35,
      36,    37,    -1,    -1,    40,    -1,    -1,    43,    -1,    45,
      46,    -1,    48,    49,    50,    -1,    52,    -1,    54,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    74,    75,
      -1,    -1,    78,    79,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    89,    90,    -1,    -1,    -1,    94,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,   102,    -1,   104,   105,
     106,     1,    -1,     3,     4,     5,     6,     7,     8,     9,
      10,    11,    12,    13,    -1,    15,    -1,    -1,    18,    -1,
      -1,    -1,    22,    -1,    -1,    25,    26,    27,    28,    29,
      30,    31,    32,    -1,    34,    35,    36,    37,    -1,    -1,
      40,    -1,    -1,    43,    -1,    -1,    46,    -1,    48,    49,
      50,    -1,    52,    -1,    54,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    74,    75,    -1,    -1,    78,    79,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    89,
      90,    -1,    -1,    -1,    94,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,   102,    -1,   104,   105,   106,     1,    -1,     3,
       4,     5,     6,     7,     8,     9,    10,    11,    12,    13,
      -1,    15,    -1,    -1,    18,    -1,    -1,    -1,    22,    -1,
      -1,    25,    26,    27,    28,    29,    30,    31,    32,    -1,
      34,    35,    36,    37,    -1,    -1,    40,    -1,    -1,    43,
      -1,    -1,    46,    -1,    48,    49,    50,    -1,    52,    -1,
      54,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      74,    75,    -1,    -1,    78,    79,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    89,    90,    -1,    -1,    -1,
      94,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   102,    -1,
     104,   105,   106,     1,    -1,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    13,    -1,    15,    -1,    -1,
      18,    -1,    -1,    -1,    22,    -1,    -1,    25,    26,    27,
      28,    29,    -1,    31,    32,    -1,    34,    35,    36,    37,
      -1,    -1,    40,    -1,    -1,    43,    -1,    -1,    46,    -1,
      48,    49,    50,    -1,    52,    -1,    54,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    74,    75,    -1,    -1,
      78,    79,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    89,    90,    -1,    -1,    -1,    94,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,   102,    -1,   104,   105,   106,     1,
      -1,     3,     4,     5,     6,     7,     8,     9,    10,    11,
      12,    13,    -1,    15,    -1,    -1,    18,    -1,    -1,    -1,
      22,    -1,    -1,    25,    26,    27,    28,    29,    -1,    31,
      32,    -1,    34,    35,    36,    37,    -1,    -1,    40,    -1,
      -1,    43,    -1,    -1,    46,    -1,    48,    49,    50,    -1,
      52,    -1,    54,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    74,    75,    -1,    -1,    78,    79,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    89,    90,    -1,
      -1,    -1,    94,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
     102,    -1,   104,   105,   106,     1,    -1,     3,     4,     5,
       6,     7,     8,     9,    10,    11,    12,    13,    -1,    15,
      -1,    -1,    18,    -1,    -1,    -1,    22,    -1,    -1,    25,
      26,    27,    28,    29,    -1,    31,    32,    -1,    34,    35,
      36,    37,    -1,    -1,    40,    -1,    -1,    43,    -1,    -1,
      46,    -1,    48,    49,    50,    -1,    52,    -1,    54,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    74,    75,
      -1,    -1,    78,    79,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    89,    90,    -1,    -1,    -1,    94,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,   102,    -1,   104,   105,
     106,     1,    -1,     3,     4,     5,     6,     7,     8,     9,
      10,    11,    12,    13,    -1,    15,    -1,    -1,    18,    -1,
      -1,    -1,    22,    -1,    -1,    25,    26,    27,    28,    29,
      -1,    31,    32,    -1,    34,    35,    36,    37,    -1,    -1,
      40,    -1,    -1,    43,    -1,    -1,    46,    -1,    48,    49,
      50,    -1,    52,    -1,    54,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    74,    75,    -1,    -1,    78,    79,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    89,
      90,    -1,    -1,    -1,    94,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,   102,    -1,   104,   105,   106,     1,    -1,     3,
       4,     5,     6,     7,     8,     9,    10,    11,    12,    13,
      -1,    15,    -1,    -1,    18,    -1,    -1,    -1,    22,    -1,
      -1,    25,    26,    27,    28,    29,    -1,    31,    32,    -1,
      34,    35,    36,    37,    -1,    -1,    40,    -1,    -1,    43,
      -1,    -1,    46,    -1,    48,    49,    50,    -1,    52,    -1,
      54,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      74,    75,    -1,    -1,    78,    79,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    89,    90,    -1,    -1,    -1,
      94,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   102,    -1,
     104,   105,   106,     1,    -1,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    13,    -1,    15,    -1,    -1,
      18,    -1,    -1,    -1,    22,    -1,    -1,    25,    26,    27,
      28,    29,    -1,    31,    32,    -1,    34,    35,    36,    37,
      -1,    -1,    40,    -1,    -1,    43,    -1,    -1,    46,    -1,
      48,    49,    50,    -1,    52,    -1,    54,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    74,    75,    -1,    -1,
      78,    79,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    89,    90,    -1,    -1,    -1,    94,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,   102,    -1,   104,   105,   106,     1,
      -1,     3,     4,     5,     6,     7,     8,     9,    10,    11,
      12,    13,    -1,    15,    -1,    -1,    18,    -1,    -1,    -1,
      22,    -1,    -1,    25,    26,    27,    28,    29,    -1,    31,
      32,    -1,    34,    35,    36,    37,    -1,    -1,    40,    -1,
      -1,    43,    -1,    -1,    46,    -1,    48,    49,    50,    -1,
      52,    -1,    54,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    74,    75,    -1,    -1,    78,    79,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    89,    90,    -1,
      -1,    -1,    94,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
     102,    -1,   104,   105,   106,     1,    -1,     3,     4,     5,
       6,     7,     8,     9,    10,    11,    12,    13,    -1,    15,
      -1,    -1,    18,    -1,    -1,    -1,    22,    -1,    -1,    25,
      26,    27,    28,    29,    -1,    31,    32,    -1,    34,    35,
      36,    37,    -1,    -1,    40,    -1,    -1,    43,    -1,    -1,
      46,    -1,    48,    49,    50,    -1,    52,    -1,    54,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    74,    75,
      -1,    -1,    78,    79,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    89,    90,    -1,    -1,    -1,    94,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,   102,    -1,   104,   105,
     106,     1,    -1,     3,     4,     5,     6,     7,     8,     9,
      10,    11,    12,    13,    -1,    15,    -1,    -1,    18,    -1,
      -1,    -1,    22,    -1,    -1,    25,    26,    27,    28,    29,
      -1,    31,    32,    -1,    34,    35,    36,    37,    -1,    -1,
      40,    -1,    -1,    43,    -1,    -1,    46,    -1,    48,    49,
      50,    -1,    52,    -1,    54,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    74,    75,    -1,    -1,    78,    79,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    89,
      90,    -1,    -1,    -1,    94,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,   102,    -1,   104,   105,   106,     1,    -1,     3,
       4,     5,     6,     7,     8,     9,    10,    11,    12,    13,
      -1,    15,    -1,    -1,    18,    -1,    -1,    -1,    22,    -1,
      -1,    25,    26,    27,    28,    29,    -1,    31,    32,    -1,
      34,    35,    36,    37,    -1,    -1,    40,    -1,    -1,    43,
      -1,    -1,    46,    -1,    48,    49,    50,    -1,    52,    -1,
      54,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      74,    75,    -1,    -1,    78,    79,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    89,    90,    -1,    -1,    -1,
      94,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   102,    -1,
     104,   105,   106,     1,    -1,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    13,    -1,    15,    -1,    -1,
      18,    -1,    -1,    -1,    22,    -1,    -1,    25,    26,    27,
      28,    29,    -1,    31,    32,    -1,    34,    35,    36,    37,
      -1,    -1,    40,    -1,    -1,    43,    -1,    -1,    46,    -1,
      48,    49,    50,    -1,    52,    -1,    54,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    74,    75,    -1,    -1,
      78,    79,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    89,    90,    -1,    -1,    -1,    94,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,   102,    -1,   104,   105,   106,     1,
      -1,     3,     4,     5,     6,     7,     8,    -1,    10,    11,
      12,    13,    -1,    15,    -1,    -1,    18,    -1,    -1,    -1,
      22,    -1,    -1,    25,    26,    27,    28,    29,    -1,    31,
      32,    -1,    34,    35,    36,    37,    -1,    -1,    40,    -1,
      -1,    43,    -1,    -1,    46,    -1,    48,    49,    50,    -1,
      52,    -1,    54,    -1,     1,    -1,    -1,     4,     5,     6,
       7,     8,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    74,    75,    -1,    -1,    78,    79,    -1,    26,
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
      -1,    -1,     4,     5,     6,     7,     8,     3,     4,     5,
       6,     7,     8,    -1,    -1,    -1,    -1,    74,    75,    -1,
      -1,    -1,    -1,    -1,    26,    27,    -1,    -1,    -1,    -1,
      26,    27,    89,    90,    -1,    37,    -1,    94,    -1,    -1,
      -1,    37,    -1,    -1,    -1,   102,    -1,   104,   105,   106,
      52,    -1,    54,    -1,    -1,    -1,    52,    -1,    54,    -1,
      -1,    -1,     3,     4,     5,     6,     7,     8,    -1,    -1,
      -1,    -1,    74,    75,    -1,    -1,    -1,    -1,    74,    75,
      -1,    -1,    -1,    -1,    -1,    26,    27,    89,    90,    -1,
      -1,    -1,    94,    89,    90,    -1,    37,    -1,    94,    -1,
     102,     3,   104,   105,   106,    -1,   102,    -1,   104,   105,
     106,    52,    -1,    54,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    74,    75,    -1,    -1,    -1,     3,    -1,
      -1,    -1,    -1,    45,    -1,    -1,    -1,    -1,    89,    90,
      52,    -1,    -1,    94,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,   102,    -1,   104,   105,   106,    68,    -1,    -1,    71,
      72,    73,    -1,    -1,    76,    77,    78,    79,    80,    81,
      82,    83,    84,    85,    86,    87,    88,    52,    -1,    91,
      92,    93,    94,    95,    96,    97,    98,    99,   100,   101,
       3,    66,   104,   105,    -1,    -1,    71,    72,    73,    -1,
      -1,    76,    77,    78,    79,    80,    81,    82,    83,    84,
      85,    86,    87,    88,    -1,    -1,    91,    92,    93,    94,
      95,    96,    97,    98,    99,   100,   101,    -1,    -1,   104,
     105,    -1,    45,    -1,    -1,    -1,    -1,    -1,    -1,    52,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,     3,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    71,    72,
      73,    -1,    -1,    76,    77,    78,    79,    80,    81,    82,
      83,    84,    85,    86,    87,    88,    -1,    -1,    91,    92,
      93,    94,    95,    96,    97,    98,    99,   100,   101,    45,
      -1,   104,   105,    -1,    -1,    -1,    52,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,     3,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    71,    72,    73,    -1,    -1,
      76,    77,    78,    79,    80,    81,    82,    83,    84,    85,
      86,    87,    88,    -1,    -1,    91,    92,    93,    94,    95,
      96,    97,    98,    99,   100,   101,    45,    -1,   104,   105,
      -1,    -1,    -1,    52,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,     3,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    71,    72,    73,    -1,    -1,    76,    77,    78,
      79,    80,    81,    82,    83,    84,    85,    86,    87,    88,
      -1,    -1,    91,    92,    93,    94,    95,    96,    97,    98,
      99,   100,   101,    45,    -1,   104,   105,    -1,    -1,    -1,
      52,     3,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    71,
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
      85,    86,    87,    88,    52,    -1,    91,    92,    93,    94,
      95,    96,    97,    98,    99,   100,   101,    -1,    -1,   104,
     105,    -1,    -1,    71,    72,    73,    -1,    -1,    76,    77,
      78,    79,    80,    81,    82,    83,    84,    85,    86,    87,
      88,    -1,    -1,    91,    92,    93,    94,    95,    96,    97,
      98,    99,   100,   101,    -1,    -1,   104,   105,     4,     5,
       6,     7,     8,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      26,    27,     4,     5,     6,     7,     8,    -1,    -1,    -1,
      -1,    37,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    45,
      -1,    -1,    -1,    -1,    26,    27,    52,    53,    54,    -1,
      -1,    -1,    -1,    -1,    -1,    37,     4,     5,     6,     7,
       8,    67,    -1,    45,    -1,    -1,    -1,    -1,    74,    75,
      52,    -1,    54,    -1,    -1,    -1,    -1,    -1,    26,    27,
      -1,    -1,    -1,    89,    90,    -1,    -1,    -1,    94,    37,
      -1,    -1,    74,    75,    -1,    -1,   102,    45,   104,   105,
     106,    -1,    -1,    -1,    52,    -1,    54,    89,    90,    -1,
      -1,    -1,    94,    -1,    -1,    -1,    98,    -1,    -1,    -1,
     102,    -1,   104,   105,   106,    -1,    74,    75,    -1,     4,
       5,     6,     7,     8,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    89,    90,    -1,    -1,    -1,    94,    -1,    -1,    -1,
      98,    26,    27,    -1,   102,    -1,   104,   105,   106,    -1,
      -1,    -1,    37,     4,     5,     6,     7,     8,    -1,    -1,
       4,     5,     6,     7,     8,    -1,    -1,    52,    53,    54,
      -1,    -1,    -1,    -1,    -1,    26,    27,    -1,    -1,    -1,
      -1,    -1,    26,    27,    -1,    -1,    37,    -1,    -1,    74,
      75,    -1,    -1,    37,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    52,    53,    54,    89,    90,    -1,    51,    52,    94,
      54,    -1,    -1,    -1,    -1,    -1,    -1,   102,    -1,   104,
     105,   106,    -1,    74,    75,    -1,    -1,    -1,    -1,    -1,
      74,    75,    -1,    -1,    -1,    -1,    -1,    -1,    89,    90,
      -1,    -1,    -1,    94,    -1,    89,    90,    -1,    -1,    -1,
      94,   102,    -1,   104,   105,   106,    -1,    -1,   102,    -1,
     104,   105,   106,     4,     5,     6,     7,     8,    -1,     4,
       5,     6,     7,     8,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    26,    27,    -1,    -1,    -1,
      -1,    26,    27,    -1,    -1,    -1,    37,    -1,    -1,    -1,
      -1,    -1,    37,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    52,    -1,    54,    -1,    -1,    -1,    52,    -1,    54,
      -1,    -1,    -1,    -1,     4,     5,     6,     7,     8,    -1,
      -1,    -1,    -1,    74,    75,    -1,    -1,    -1,    -1,    74,
      75,    -1,    -1,    -1,    -1,    -1,    26,    27,    89,    90,
      -1,    -1,    -1,    94,    89,    90,    -1,    37,    -1,    94,
      -1,   102,    -1,   104,   105,   106,    -1,   102,    -1,   104,
     105,   106,    52,    -1,    54,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    74,    75,    -1,    -1,    -1,    -1,
      -1,    45,    -1,    -1,    -1,    -1,    -1,    -1,    52,    89,
      90,    -1,    -1,    -1,    94,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,   102,    67,   104,   105,   106,    71,    72,    73,
      -1,    -1,    76,    77,    78,    79,    80,    81,    82,    83,
      84,    85,    86,    87,    88,    -1,    -1,    91,    92,    93,
      94,    95,    96,    97,    98,    99,   100,   101,    45,    -1,
     104,   105,    -1,    -1,    -1,    52,    53,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    71,    72,    73,    -1,    -1,    76,
      77,    78,    79,    80,    81,    82,    83,    84,    85,    86,
      87,    88,    -1,    -1,    91,    92,    93,    94,    95,    96,
      97,    98,    99,   100,   101,    45,    -1,   104,   105,    -1,
      -1,    -1,    52,    53,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    71,    72,    73,    -1,    -1,    76,    77,    78,    79,
      80,    81,    82,    83,    84,    85,    86,    87,    88,    -1,
      -1,    91,    92,    93,    94,    95,    96,    97,    98,    99,
     100,   101,    45,    -1,   104,   105,    -1,    -1,    -1,    52,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    71,    72,
      73,    -1,    -1,    76,    77,    78,    79,    80,    81,    82,
      83,    84,    85,    86,    87,    88,    52,    53,    91,    92,
      93,    94,    95,    96,    97,    98,    99,   100,   101,    -1,
      -1,   104,   105,    -1,    -1,    71,    72,    73,    -1,    -1,
      76,    77,    78,    79,    80,    81,    82,    83,    84,    85,
      86,    87,    88,    52,    -1,    91,    92,    93,    94,    95,
      96,    97,    98,    99,   100,   101,    -1,    -1,   104,   105,
      69,    -1,    71,    72,    73,    -1,    -1,    76,    77,    78,
      79,    80,    81,    82,    83,    84,    85,    86,    87,    88,
      52,    53,    91,    92,    93,    94,    95,    96,    97,    98,
      99,   100,   101,    -1,    -1,   104,   105,    -1,    -1,    71,
      72,    73,    -1,    -1,    76,    77,    78,    79,    80,    81,
      82,    83,    84,    85,    86,    87,    88,    52,    53,    91,
      92,    93,    94,    95,    96,    97,    98,    99,   100,   101,
      -1,    -1,   104,   105,    -1,    -1,    71,    72,    73,    -1,
      -1,    76,    77,    78,    79,    80,    81,    82,    83,    84,
      85,    86,    87,    88,    52,    53,    91,    92,    93,    94,
      95,    96,    97,    98,    99,   100,   101,    -1,    -1,   104,
     105,    -1,    -1,    71,    72,    73,    -1,    -1,    76,    77,
      78,    79,    80,    81,    82,    83,    84,    85,    86,    87,
      88,    -1,    -1,    91,    92,    93,    94,    95,    96,    97,
      98,    99,   100,   101,    52,    -1,   104,   105,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    67,
      -1,    -1,    -1,    71,    72,    73,    -1,    -1,    76,    77,
      78,    79,    80,    81,    82,    83,    84,    85,    86,    87,
      88,    52,    -1,    91,    92,    93,    94,    95,    96,    97,
      98,    99,   100,   101,    -1,    -1,   104,   105,    -1,    -1,
      71,    72,    73,    -1,    -1,    76,    77,    78,    79,    80,
      81,    82,    83,    84,    85,    86,    87,    88,    52,    -1,
      91,    92,    93,    94,    95,    96,    97,    98,    99,   100,
     101,    -1,    -1,   104,   105,    -1,    -1,    -1,    72,    73,
      -1,    -1,    76,    77,    78,    79,    80,    81,    82,    83,
      84,    85,    86,    87,    88,    52,    -1,    91,    92,    93,
      94,    95,    96,    97,    98,    99,   100,   101,    -1,    -1,
     104,   105,    -1,    -1,    -1,    -1,    73,    -1,    -1,    76,
      77,    78,    79,    80,    81,    82,    83,    84,    85,    86,
      87,    88,    52,    -1,    91,    92,    93,    94,    95,    96,
      97,    98,    99,   100,   101,    -1,    -1,   104,   105,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    76,    77,    78,    79,
      80,    81,    82,    83,    84,    85,    86,    87,    88,    -1,
      -1,    91,    92,    93,    94,    95,    96,    97,    98,    99,
     100,   101,    -1,    -1,   104,   105
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
     127,   128,   129,   130,   132,   133,   134,   136,   137,   145,
     146,   147,   149,   150,   151,   155,   156,   157,   164,   166,
     179,   181,   190,   191,   193,   200,   201,   202,   204,   206,
     214,   215,   216,   217,   219,   220,   221,   224,   242,   247,
     251,   252,   253,   254,   255,   256,   257,   258,   262,   266,
     267,   269,     3,     3,     1,   114,   253,     1,   255,   256,
       1,     3,     1,     3,    14,     1,   256,     1,   253,   255,
     273,     1,   256,     1,     1,   256,     1,   256,   271,     1,
       3,    45,     1,   256,     1,     6,     1,     6,     1,     3,
     256,   248,   263,     1,     6,     7,     1,   256,   258,     1,
       6,     1,     3,    45,     1,   256,     1,     3,     6,   218,
       1,     6,     1,   256,     3,    45,     3,   260,   261,    45,
      53,    67,   256,   272,   274,   256,   255,     1,     3,   271,
       3,   271,   256,   256,   256,   256,   256,   256,   256,   131,
      32,    34,    46,   113,   135,   113,   148,   113,   165,   180,
     192,    47,   209,   212,   213,   113,     1,    52,     1,     6,
     222,   223,   222,    54,    55,    56,    57,    58,    59,    60,
      61,    62,    63,    64,    65,    70,    83,   257,     3,    52,
      66,    71,    72,    73,    76,    77,    78,    79,    80,    81,
      82,    83,    84,    85,    86,    87,    88,    91,    92,    93,
      94,    95,    96,    97,    98,    99,   100,   101,   104,   105,
      54,    55,   257,     3,     3,    70,    83,     3,    45,     3,
      45,     3,     3,     3,     3,    45,     3,    45,     3,    45,
      83,    70,    86,     3,     3,     3,     3,     3,     3,     1,
      69,    70,     3,   113,     3,     3,     3,   225,     3,   243,
       3,     3,     6,   249,   250,     1,    52,   264,     3,     3,
       3,     3,     3,     3,    83,     3,    45,     3,     3,    86,
       3,     3,    70,     3,     3,     3,   256,     3,    53,   256,
      53,    45,    67,     1,    70,   260,     1,    70,   260,    82,
      83,     3,     3,     3,   144,   144,   144,   167,   182,   144,
       1,     3,    45,   144,   210,   211,     3,   260,     3,     3,
      70,     9,   222,     3,    98,   256,     6,   256,   256,   256,
     256,   256,   256,   256,   256,   256,   256,   253,   273,   106,
     256,   271,   260,   256,     1,   256,   256,   256,   256,   256,
     256,   256,   256,   256,   256,   256,     6,   256,   256,   256,
     256,   256,   256,   256,   256,   256,   256,   256,   256,   256,
     256,   256,    98,   256,     6,   253,   256,   256,   253,     1,
     256,     3,   256,   256,     1,    52,   226,   227,     1,    33,
     229,   244,     3,    70,     3,   260,   209,     1,   252,     1,
     256,     6,   260,    53,    53,   256,   256,   268,   260,    53,
     270,   260,    53,   256,   256,     9,   113,    16,    17,   138,
     140,   141,   143,     9,     1,     3,    23,    24,   168,   173,
     175,     1,     3,    23,   173,   183,    30,   194,   195,   196,
     197,     3,    45,     9,   144,   113,     1,     6,   207,   208,
       6,     3,     3,   256,    53,     3,     3,     3,     3,     3,
       3,     3,     3,     3,     3,    83,   106,     3,     3,    51,
     256,   272,     3,    45,   256,    53,    83,    69,     3,     3,
      45,     3,     3,   260,   235,   229,     3,     6,   230,   231,
       3,   245,     1,   250,   207,   144,     3,     3,     3,     3,
      51,    53,   260,   256,   260,   256,     3,     1,     3,     1,
     256,     9,   139,   142,     3,     3,     1,     4,     6,     7,
       8,   177,   178,     1,     9,   174,     3,     1,     4,     6,
     188,   189,     9,     1,     3,     4,     6,    86,   198,   199,
       9,   196,   144,     3,     9,   205,     1,    70,   260,    53,
     256,   271,     3,     1,   260,   256,    53,   256,   256,   152,
     113,   207,     3,     6,    38,    46,    47,    88,   201,   236,
     237,   239,   240,     3,    52,   232,    70,     3,   201,   237,
     239,   240,   246,     1,   260,     9,    53,    53,    67,     3,
       3,     3,     3,   144,   144,     3,    45,    69,     3,    45,
      70,     3,     3,    45,   176,     3,    45,     3,    45,    70,
       3,     3,   253,     3,    70,    86,     3,     3,   260,   203,
     260,    51,     3,     3,   259,    51,     3,    45,    68,    19,
      20,    21,   113,   153,   154,   158,   160,   162,     1,   260,
      83,     3,     6,     1,     6,    74,   241,     9,   260,   231,
       9,   265,    51,   256,   138,   171,   172,     4,   169,   170,
     178,   144,   113,   186,   187,   184,   185,   189,     3,   199,
     253,    51,   260,   208,     3,    45,   260,   256,     1,     3,
      45,     1,     3,    45,     1,     3,    45,     9,   153,   228,
      51,   256,   238,    83,     3,     6,     3,    70,     3,     6,
      27,   233,   234,   252,     3,   260,     3,   144,   113,   144,
     113,   144,   113,   144,   113,     3,    45,    51,    51,     3,
      45,     3,   159,   113,     3,   161,   113,     3,   163,   113,
       3,   260,     3,   209,   256,     6,    74,    70,   260,    51,
       3,   144,   144,   144,    51,   144,     3,     6,   234,    51,
       3,     9,     9,     9,     9,     3,     3,     3,     3
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
#line 202 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_lone_end ); ;}
    break;

  case 7:
#line 203 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_case_outside ); ;}
    break;

  case 8:
#line 209 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         if ( (yyvsp[(1) - (1)].stringp) != 0 )
            COMPILER->addLoad( *(yyvsp[(1) - (1)].stringp) );
      ;}
    break;

  case 9:
#line 214 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         if( (yyvsp[(1) - (1)].fal_stat) != 0 )
            COMPILER->addFunction( (yyvsp[(1) - (1)].fal_stat) );
      ;}
    break;

  case 10:
#line 219 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         if ( (yyvsp[(1) - (1)].fal_stat) != 0 )
            COMPILER->addClass( (yyvsp[(1) - (1)].fal_stat) );
      ;}
    break;

  case 11:
#line 224 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         if ( (yyvsp[(1) - (1)].fal_stat) != 0 )
            COMPILER->addClass( (yyvsp[(1) - (1)].fal_stat) );
      ;}
    break;

  case 12:
#line 229 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         if( (yyvsp[(1) - (1)].fal_stat) != 0 )
            COMPILER->addStatement( (yyvsp[(1) - (1)].fal_stat) );
      ;}
    break;

  case 16:
#line 240 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         if ( COMPILER->getContext() != 0 )
            COMPILER->raiseError(Falcon::e_toplevel_load );
         (yyval.stringp) = (yyvsp[(2) - (3)].stringp);
      ;}
    break;

  case 17:
#line 246 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         if ( COMPILER->getContext() != 0 )
            COMPILER->raiseError(Falcon::e_toplevel_load );
         (yyval.stringp) = (yyvsp[(2) - (3)].stringp);
      ;}
    break;

  case 18:
#line 252 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_load );
         (yyval.stringp) = 0;
      ;}
    break;

  case 19:
#line 259 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { COMPILER->checkLocalUndefined(); (yyval.fal_stat) = (yyvsp[(1) - (1)].fal_stat); ;}
    break;

  case 20:
#line 260 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { (yyval.fal_stat) = 0; ;}
    break;

  case 21:
#line 263 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_toplevel_func ); (yyval.fal_stat) = 0; ;}
    break;

  case 22:
#line 264 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_toplevel_obj ); (yyval.fal_stat) = 0; ;}
    break;

  case 23:
#line 265 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_toplevel_class ); (yyval.fal_stat) = 0; ;}
    break;

  case 24:
#line 266 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syntax ); (yyval.fal_stat) = 0;;}
    break;

  case 25:
#line 271 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
      COMPILER->defContext( true ); COMPILER->defRequired();
      COMPILER->defineVal( (yyvsp[(1) - (3)].fal_val) );
      COMPILER->addStatement( new Falcon::StmtAssignment( CURRENT_LINE, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) );
   ;}
    break;

  case 26:
#line 276 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(3) - (5)].fal_val) );
      COMPILER->addStatement( new Falcon::StmtAssignment( CURRENT_LINE, (yyvsp[(3) - (5)].fal_val), (yyvsp[(5) - (5)].fal_val) ) );
   ;}
    break;

  case 27:
#line 283 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { (yyval.fal_stat) = new Falcon::StmtAutoexpr( LINE, (yyvsp[(1) - (2)].fal_val) ); ;}
    break;

  case 49:
#line 309 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { COMPILER->defContext( false );  (yyval.fal_stat)=0; ;}
    break;

  case 50:
#line 311 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { COMPILER->raiseError( Falcon::e_syn_def ); ;}
    break;

  case 51:
#line 316 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAssignment( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 52:
#line 320 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (5)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtUnref( LINE, (yyvsp[(1) - (5)].fal_val) );
   ;}
    break;

  case 53:
#line 324 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
         (yyval.fal_stat) = new Falcon::StmtAssignment( LINE, (yyvsp[(1) - (4)].fal_val), new Falcon::Value( (yyvsp[(3) - (4)].fal_adecl) ) );
      ;}
    break;

  case 54:
#line 328 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(1) - (6)].fal_val) );
         (yyvsp[(3) - (6)].fal_adecl)->pushFront( (yyvsp[(1) - (6)].fal_val) );
         (yyval.fal_stat) = new Falcon::StmtAssignment( LINE, new Falcon::Value((yyvsp[(3) - (6)].fal_adecl)), (yyvsp[(5) - (6)].fal_val) );
      ;}
    break;

  case 55:
#line 333 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(1) - (6)].fal_val) );
         (yyvsp[(3) - (6)].fal_adecl)->pushFront( (yyvsp[(1) - (6)].fal_val) );
         (yyval.fal_stat) = new Falcon::StmtAssignment( LINE, new Falcon::Value((yyvsp[(3) - (6)].fal_adecl)), new Falcon::Value( (yyvsp[(5) - (6)].fal_adecl) ) );
      ;}
    break;

  case 67:
#line 357 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAutoAdd( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 68:
#line 364 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAutoSub( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 69:
#line 371 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAutoMul( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 70:
#line 378 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAutoDiv( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 71:
#line 385 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAutoMod( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 72:
#line 392 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAutoPow( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 73:
#line 399 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAutoBAND( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 74:
#line 406 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAutoBOR( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 75:
#line 413 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAutoBXOR( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 76:
#line 419 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAutoSHL( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 77:
#line 425 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAutoSHR( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 78:
#line 433 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         Falcon::StmtWhile *w = new Falcon::StmtWhile( LINE, (yyvsp[(1) - (1)].fal_val) );
         COMPILER->pushLoop( w );
         COMPILER->pushContext( w );
         COMPILER->pushContextSet( &w->children() );
      ;}
    break;

  case 79:
#line 440 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         Falcon::StmtWhile *w = static_cast<Falcon::StmtWhile *>(COMPILER->getContext());
         COMPILER->popLoop();
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = w;
      ;}
    break;

  case 80:
#line 448 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         Falcon::StmtWhile *w = new Falcon::StmtWhile( LINE, (yyvsp[(1) - (2)].fal_val) );
         if ( (yyvsp[(2) - (2)].fal_stat) != 0 )
            w->children().push_back( (yyvsp[(2) - (2)].fal_stat) );
         (yyval.fal_stat) = w;
      ;}
    break;

  case 81:
#line 456 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); ;}
    break;

  case 82:
#line 457 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { (yyval.fal_val) = 0; ;}
    break;

  case 83:
#line 458 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_while ); (yyval.fal_val) = 0; ;}
    break;

  case 84:
#line 462 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); ;}
    break;

  case 85:
#line 463 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { (yyval.fal_val) = 0; ;}
    break;

  case 86:
#line 464 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_while, "", CURRENT_LINE ); (yyval.fal_val) = 0; ;}
    break;

  case 87:
#line 468 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         Falcon::StmtIf *stmt = new Falcon::StmtIf( LINE, (yyvsp[(1) - (1)].fal_val) );
         COMPILER->pushContext( stmt );
         COMPILER->pushContextSet( &stmt->children() );
      ;}
    break;

  case 88:
#line 476 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         Falcon::StmtIf *stmt = static_cast<Falcon::StmtIf *>(COMPILER->getContext());
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = stmt;
      ;}
    break;

  case 89:
#line 483 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         // use LINE as statement includes EOL
         Falcon::StmtIf *stmt = new Falcon::StmtIf( LINE, (yyvsp[(1) - (2)].fal_val) );
         if( (yyvsp[(2) - (2)].fal_stat) != 0 )
            stmt->children().push_back( (yyvsp[(2) - (2)].fal_stat) );
         (yyval.fal_stat) = stmt;
      ;}
    break;

  case 90:
#line 493 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); ;}
    break;

  case 91:
#line 494 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {  COMPILER->raiseError(Falcon::e_syn_if ); (yyval.fal_val) = 0; ;}
    break;

  case 92:
#line 498 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); ;}
    break;

  case 93:
#line 499 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {  COMPILER->raiseError(Falcon::e_syn_if, "", CURRENT_LINE ); (yyval.fal_val) = 0; ;}
    break;

  case 96:
#line 506 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         Falcon::StmtIf *stmt = static_cast<Falcon::StmtIf *>(COMPILER->getContext());
         COMPILER->popContextSet();
         COMPILER->pushContextSet( &stmt->elseChildren() );
      ;}
    break;

  case 99:
#line 516 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_else ); ;}
    break;

  case 100:
#line 521 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         Falcon::StmtIf *stmt = static_cast<Falcon::StmtIf *>(COMPILER->getContext());
         COMPILER->popContextSet();
         Falcon::StmtElif *elif = new Falcon::StmtElif( LINE, (yyvsp[(1) - (1)].fal_val) );
         stmt->elifChildren().push_back( elif );
         COMPILER->pushContextSet( &elif->children() );
      ;}
    break;

  case 102:
#line 533 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); ;}
    break;

  case 103:
#line 534 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_elif ); (yyval.fal_val) = 0; ;}
    break;

  case 105:
#line 539 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
      COMPILER->addStatement( (yyvsp[(2) - (2)].fal_stat) );
   ;}
    break;

  case 106:
#line 546 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         if ( COMPILER->getLoop() == 0 ) {
            COMPILER->raiseError(Falcon::e_break_out );
            (yyval.fal_stat) = 0;
         }
         else
            (yyval.fal_stat) = new Falcon::StmtBreak( LINE );
      ;}
    break;

  case 107:
#line 555 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_break );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 108:
#line 563 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         if ( COMPILER->getLoop() == 0 ) {
            COMPILER->raiseError(Falcon::e_continue_out );
            (yyval.fal_stat) = 0;
         }
         else
            (yyval.fal_stat) = new Falcon::StmtContinue( LINE );
      ;}
    break;

  case 109:
#line 573 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         if ( COMPILER->getLoop() == 0 ) {
            COMPILER->raiseError(Falcon::e_continue_out );
            (yyval.fal_stat) = 0;
         }
         else
            (yyval.fal_stat) = new Falcon::StmtContinue( LINE, true );
      ;}
    break;

  case 110:
#line 582 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_continue );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 111:
#line 589 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         Falcon::StmtFor *f = static_cast<Falcon::StmtFor *>( (yyvsp[(1) - (1)].fal_stat) );
         COMPILER->pushLoop( f );
         COMPILER->pushContext( f );
         COMPILER->pushContextSet( &f->children() );
      ;}
    break;

  case 112:
#line 596 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         Falcon::StmtFor *f = static_cast<Falcon::StmtFor *>(COMPILER->getContext());
         COMPILER->popLoop();
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = f;
      ;}
    break;

  case 113:
#line 604 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
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

  case 114:
#line 619 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(2) - (7)].fal_val) );
         (yyval.fal_stat) = new Falcon::StmtFor( LINE, (yyvsp[(2) - (7)].fal_val), (yyvsp[(4) - (7)].fal_val), (yyvsp[(6) - (7)].fal_val) );
      ;}
    break;

  case 115:
#line 623 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(2) - (9)].fal_val) );
         (yyval.fal_stat) = new Falcon::StmtFor( LINE, (yyvsp[(2) - (9)].fal_val), (yyvsp[(4) - (9)].fal_val), (yyvsp[(6) - (9)].fal_val), (yyvsp[(8) - (9)].fal_val) );
      ;}
    break;

  case 116:
#line 628 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_for );
         (yyval.fal_stat) = new Falcon::StmtFor( LINE, 0, 0, 0 );
      ;}
    break;

  case 117:
#line 635 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(2) - (7)].fal_val) );
         (yyval.fal_stat) = new Falcon::StmtFor( CURRENT_LINE, (yyvsp[(2) - (7)].fal_val), (yyvsp[(4) - (7)].fal_val), (yyvsp[(6) - (7)].fal_val) );
      ;}
    break;

  case 118:
#line 639 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(2) - (9)].fal_val) );
         (yyval.fal_stat) = new Falcon::StmtFor( CURRENT_LINE, (yyvsp[(2) - (9)].fal_val), (yyvsp[(4) - (9)].fal_val), (yyvsp[(6) - (9)].fal_val), (yyvsp[(8) - (9)].fal_val) );
      ;}
    break;

  case 119:
#line 644 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_for, "", CURRENT_LINE );
         (yyval.fal_stat) = new Falcon::StmtFor( CURRENT_LINE, 0, 0, 0 );
      ;}
    break;

  case 120:
#line 653 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
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

  case 121:
#line 669 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         COMPILER->popLoop();
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = f;
      ;}
    break;

  case 122:
#line 677 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
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

  case 123:
#line 691 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { COMPILER->raiseError( Falcon::e_syn_forin ); ;}
    break;

  case 126:
#line 700 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         if ( (yyvsp[(1) - (1)].fal_stat) != 0 )
         {
            Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
            f->children().push_back( (yyvsp[(1) - (1)].fal_stat) );
         }
      ;}
    break;

  case 130:
#line 714 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
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

  case 131:
#line 727 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_fordot );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 132:
#line 735 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         (yyval.fal_stat) = new Falcon::StmtSelfPrint( LINE, (yyvsp[(2) - (3)].fal_adecl) );
      ;}
    break;

  case 133:
#line 739 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_self_print );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 134:
#line 745 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         (yyvsp[(2) - (3)].fal_adecl)->pushBack( new Falcon::Value( COMPILER->addString( "\n" ) ) );
         (yyval.fal_stat) = new Falcon::StmtSelfPrint( LINE, (yyvsp[(2) - (3)].fal_adecl) );
      ;}
    break;

  case 135:
#line 750 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         Falcon::ArrayDecl *adecl = new Falcon::ArrayDecl();
         adecl->pushBack( new Falcon::Value( COMPILER->addString( "\n" ) ) );
         (yyval.fal_stat) = new Falcon::StmtSelfPrint( LINE, adecl );
      ;}
    break;

  case 136:
#line 757 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_self_print );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 137:
#line 766 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
      Falcon::ArrayDecl *adecl = new Falcon::ArrayDecl();
      adecl->pushBack( new Falcon::Value( (yyvsp[(1) - (1)].stringp) ) );
      (yyval.fal_stat) = new Falcon::StmtSelfPrint( LINE, adecl );
   ;}
    break;

  case 138:
#line 775 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
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

  case 139:
#line 787 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { COMPILER->popContextSet(); ;}
    break;

  case 140:
#line 789 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         if( ! f->firstBlock().empty() )
         {
            COMPILER->raiseError( Falcon::e_already_forfirst );
         }
         f->firstBlock().push_back( (yyvsp[(3) - (3)].fal_stat) );
      ;}
    break;

  case 141:
#line 797 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_forfirst ); ;}
    break;

  case 142:
#line 801 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
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

  case 143:
#line 813 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { COMPILER->popContextSet(); ;}
    break;

  case 144:
#line 814 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         if( ! f->lastBlock().empty() )
         {
            COMPILER->raiseError( Falcon::e_already_forlast );
         }
         f->lastBlock().push_back( (yyvsp[(3) - (3)].fal_stat) );
      ;}
    break;

  case 145:
#line 822 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_forlast ); ;}
    break;

  case 146:
#line 826 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
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

  case 147:
#line 838 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { COMPILER->popContextSet(); ;}
    break;

  case 148:
#line 840 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         if( ! f->allBlock().empty() )
         {
            COMPILER->raiseError( Falcon::e_already_forall );
         }
         f->allBlock().push_back( (yyvsp[(3) - (3)].fal_stat) );
      ;}
    break;

  case 149:
#line 848 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_forall ); ;}
    break;

  case 150:
#line 852 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
      Falcon::StmtSwitch *stmt = new Falcon::StmtSwitch( LINE, (yyvsp[(1) - (1)].fal_val) );
         COMPILER->pushContext( stmt );
         COMPILER->pushContextSet( &stmt->blocks() );
      ;}
    break;

  case 151:
#line 860 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = stmt;
      ;}
    break;

  case 152:
#line 869 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); ;}
    break;

  case 153:
#line 871 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_switch_decl );
         (yyval.fal_val) = 0;
      ;}
    break;

  case 156:
#line 880 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_switch_body ); ;}
    break;

  case 158:
#line 886 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      ;}
    break;

  case 160:
#line 896 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( CURRENT_LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      ;}
    break;

  case 161:
#line 904 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
            COMPILER->addStatement( (yyvsp[(5) - (5)].fal_stat) );
      ;}
    break;

  case 162:
#line 908 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_case_decl );

         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      ;}
    break;

  case 164:
#line 920 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_case_decl );

         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( CURRENT_LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      ;}
    break;

  case 165:
#line 930 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
            COMPILER->addStatement( (yyvsp[(5) - (5)].fal_stat) );
      ;}
    break;

  case 167:
#line 939 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
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

  case 171:
#line 953 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_default_decl ); ;}
    break;

  case 173:
#line 957 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         COMPILER->addStatement( (yyvsp[(2) - (2)].fal_stat) );
      ;}
    break;

  case 176:
#line 969 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         // todo: correct error
         if ( stmt->nilBlock() != -1 )
            COMPILER->raiseError(Falcon::e_switch_clash, "nil entry", CURRENT_LINE );
         stmt->nilBlock( stmt->currentBlock() );
      ;}
    break;

  case 177:
#line 978 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
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

  case 178:
#line 990 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
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

  case 179:
#line 1001 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
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

  case 180:
#line 1012 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
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

  case 181:
#line 1032 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
      Falcon::StmtSelect *stmt = new Falcon::StmtSelect( LINE, (yyvsp[(1) - (1)].fal_val) );
         COMPILER->pushContext( stmt );
         COMPILER->pushContextSet( &stmt->blocks() );
      ;}
    break;

  case 182:
#line 1040 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = stmt;
      ;}
    break;

  case 183:
#line 1049 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); ;}
    break;

  case 184:
#line 1051 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_select_decl );
         (yyval.fal_val) = 0;
      ;}
    break;

  case 187:
#line 1060 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_select_body ); ;}
    break;

  case 189:
#line 1066 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      ;}
    break;

  case 191:
#line 1076 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( CURRENT_LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      ;}
    break;

  case 192:
#line 1085 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
            COMPILER->addStatement( (yyvsp[(5) - (5)].fal_stat) );
      ;}
    break;

  case 193:
#line 1089 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_case_decl );

         Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      ;}
    break;

  case 195:
#line 1101 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_case_decl );

        Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( CURRENT_LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      ;}
    break;

  case 196:
#line 1111 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
            COMPILER->addStatement( (yyvsp[(5) - (5)].fal_stat) );
      ;}
    break;

  case 200:
#line 1125 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
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

  case 201:
#line 1137 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
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

  case 202:
#line 1158 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         (yyval.fal_stat) = new Falcon::StmtGive( LINE, (yyvsp[(4) - (5)].fal_val), (yyvsp[(2) - (5)].fal_adecl) );
      ;}
    break;

  case 203:
#line 1162 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         (yyval.fal_stat) = new Falcon::StmtGive( LINE, 0, (yyvsp[(2) - (4)].fal_adecl) );
         COMPILER->raiseError(Falcon::e_syn_give );
      ;}
    break;

  case 204:
#line 1166 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_give ); (yyval.fal_stat) = 0; ;}
    break;

  case 205:
#line 1174 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
      Falcon::StmtTry *t = new Falcon::StmtTry( CURRENT_LINE );
      if ( (yyvsp[(3) - (3)].fal_stat) != 0 )
          t->children().push_back( (yyvsp[(3) - (3)].fal_stat) );
      (yyval.fal_stat) = t;
   ;}
    break;

  case 206:
#line 1181 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         Falcon::StmtTry *t = new Falcon::StmtTry( LINE );
         COMPILER->pushContext( t );
         COMPILER->pushContextSet( &t->children() );
      ;}
    break;

  case 207:
#line 1191 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         (yyval.fal_stat) = COMPILER->getContext();
         COMPILER->popContext();
         COMPILER->popContextSet();
      ;}
    break;

  case 209:
#line 1200 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_try ); ;}
    break;

  case 215:
#line 1220 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
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

  case 216:
#line 1238 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
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

  case 217:
#line 1258 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         COMPILER->popContextSet(); // popping previous catch

         Falcon::StmtTry *t = static_cast<Falcon::StmtTry *>( COMPILER->getContext() );
         Falcon::StmtCatchBlock *lst = new Falcon::StmtCatchBlock( LINE, 0 );
         COMPILER->pushContextSet( &lst->children() );
         t->addHandler( lst );
      ;}
    break;

  case 218:
#line 1268 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         COMPILER->popContextSet(); // popping previous catch

         Falcon::StmtTry *t = static_cast<Falcon::StmtTry *>( COMPILER->getContext() );
         COMPILER->defineVal( (yyvsp[(4) - (5)].fal_val) );
         Falcon::StmtCatchBlock *lst = new Falcon::StmtCatchBlock( LINE, (yyvsp[(4) - (5)].fal_val) );
         COMPILER->pushContextSet( &lst->children() );
         t->addHandler( lst );
      ;}
    break;

  case 219:
#line 1279 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
      COMPILER->raiseError( Falcon::e_syn_catch );
   ;}
    break;

  case 222:
#line 1292 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
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

  case 223:
#line 1304 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
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

  case 224:
#line 1326 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { (yyval.fal_stat) = new Falcon::StmtRaise( LINE, (yyvsp[(2) - (3)].fal_val) ); ;}
    break;

  case 225:
#line 1327 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_raise ); (yyval.fal_stat) = 0; ;}
    break;

  case 226:
#line 1339 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         (yyval.fal_stat) = COMPILER->getContext();
         COMPILER->closeFunction();
      ;}
    break;

  case 227:
#line 1345 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         COMPILER->addStatement( (yyvsp[(2) - (2)].fal_stat) );
         (yyval.fal_stat) = COMPILER->getContext();
         COMPILER->closeFunction();
      ;}
    break;

  case 229:
#line 1354 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); ;}
    break;

  case 230:
#line 1355 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_funcdecl, "", COMPILER->tempLine() );
      ;}
    break;

  case 231:
#line 1358 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_funcdecl ); ;}
    break;

  case 233:
#line 1363 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); ;}
    break;

  case 234:
#line 1364 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_funcdecl, "", COMPILER->tempLine() );
      ;}
    break;

  case 235:
#line 1371 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
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

  case 239:
#line 1432 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
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

  case 241:
#line 1449 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());
         COMPILER->pushContextSet( &func->staticBlock() );
         COMPILER->staticPrefix( &func->symbol()->name() );
      ;}
    break;

  case 242:
#line 1455 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         COMPILER->popContextSet();
         COMPILER->staticPrefix(0);
      ;}
    break;

  case 243:
#line 1460 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());
         COMPILER->pushContextSet( &func->staticBlock() );
         COMPILER->staticPrefix( &func->symbol()->name() );
      ;}
    break;

  case 244:
#line 1466 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         COMPILER->addStatement( (yyvsp[(3) - (3)].fal_stat) );
         COMPILER->popContextSet();
         COMPILER->staticPrefix(0);
      ;}
    break;

  case 246:
#line 1475 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_static ); ;}
    break;

  case 248:
#line 1480 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_static, "", CURRENT_LINE ); ;}
    break;

  case 249:
#line 1490 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         (yyval.fal_stat) = new Falcon::StmtLaunch( LINE, (yyvsp[(2) - (3)].fal_val) );
      ;}
    break;

  case 250:
#line 1493 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_launch ); (yyval.fal_stat) = 0; ;}
    break;

  case 251:
#line 1502 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         if ( COMPILER->getFunction() == 0 )
            COMPILER->raiseError(Falcon::e_pass_outside );
         else
            (yyval.fal_stat) = new Falcon::StmtPass( LINE, (yyvsp[(2) - (3)].fal_val) );
      ;}
    break;

  case 252:
#line 1509 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         // define the expression anyhow so we don't have fake errors below
         COMPILER->defineVal( (yyvsp[(4) - (5)].fal_val) );

         if ( COMPILER->getFunction() == 0 )
            COMPILER->raiseError(Falcon::e_pass_outside );
         else
            (yyval.fal_stat) = new Falcon::StmtPass( LINE, (yyvsp[(2) - (5)].fal_val), (yyvsp[(4) - (5)].fal_val) );
      ;}
    break;

  case 253:
#line 1519 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         delete (yyvsp[(2) - (5)].fal_val);
         COMPILER->raiseError(Falcon::e_syn_pass_in );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 254:
#line 1525 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_pass );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 255:
#line 1537 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
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

  case 256:
#line 1547 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_inv_const_val );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 257:
#line 1552 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_const );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 258:
#line 1564 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         if ( COMPILER->sourceTree()->isExportAll() )
            COMPILER->raiseError(Falcon::e_export_all );
         else
            COMPILER->sourceTree()->setExportAll();
         // no effect
         (yyval.fal_stat)=0;
      ;}
    break;

  case 259:
#line 1573 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         if ( COMPILER->sourceTree()->isExportAll() )
            COMPILER->raiseError(Falcon::e_export_all );
         // no effect
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 260:
#line 1580 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_export );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 261:
#line 1588 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         Falcon::Symbol *sym = COMPILER->addGlobalSymbol( (yyvsp[(1) - (1)].stringp) );
         sym->exported(true);
      ;}
    break;

  case 262:
#line 1593 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         Falcon::Symbol *sym = COMPILER->addGlobalSymbol( (yyvsp[(3) - (3)].stringp) );
         sym->exported(true);
      ;}
    break;

  case 263:
#line 1607 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         // no other action:
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 264:
#line 1614 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         // no other action:
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 266:
#line 1622 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_attributes ); ;}
    break;

  case 268:
#line 1626 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_attributes, "", CURRENT_LINE ); ;}
    break;

  case 270:
#line 1632 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
            COMPILER->addAttribute( (yyvsp[(1) - (1)].stringp) );
         ;}
    break;

  case 271:
#line 1636 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
            COMPILER->addAttribute( (yyvsp[(3) - (3)].stringp) );
         ;}
    break;

  case 274:
#line 1645 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
      COMPILER->raiseError(Falcon::e_inv_attrib );
   ;}
    break;

  case 275:
#line 1656 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
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

  case 276:
#line 1690 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
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

  case 278:
#line 1718 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_class );
      ;}
    break;

  case 281:
#line 1726 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); ;}
    break;

  case 282:
#line 1727 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_class, "", COMPILER->tempLine() );
      ;}
    break;

  case 287:
#line 1744 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
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

  case 288:
#line 1777 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { (yyval.fal_adecl) = 0; ;}
    break;

  case 289:
#line 1782 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
      (yyval.fal_adecl) = (yyvsp[(3) - (5)].fal_adecl);
   ;}
    break;

  case 290:
#line 1788 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { (yyval.fal_adecl) = new Falcon::ArrayDecl(); (yyval.fal_adecl)->pushBack( (yyvsp[(1) - (1)].fal_val) ); ;}
    break;

  case 291:
#line 1789 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { (yyvsp[(1) - (3)].fal_adecl)->pushBack( (yyvsp[(3) - (3)].fal_val) ); (yyval.fal_adecl) = (yyvsp[(1) - (3)].fal_adecl); ;}
    break;

  case 293:
#line 1795 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         // the symbol must be a parameter, or we raise an error
         Falcon::Symbol *sym = COMPILER->searchLocalSymbol( (yyvsp[(1) - (1)].stringp) );
         if ( sym == 0 || sym->type() != Falcon::Symbol::tparam ) {
            sym = COMPILER->addGlobalSymbol( (yyvsp[(1) - (1)].stringp) );
         }
         (yyval.fal_val) = new Falcon::Value( sym );
      ;}
    break;

  case 294:
#line 1803 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setSelf(); ;}
    break;

  case 298:
#line 1813 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
      COMPILER->addFunction( (yyvsp[(1) - (1)].fal_stat) );
   ;}
    break;

  case 299:
#line 1816 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
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

  case 301:
#line 1839 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
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

  case 302:
#line 1863 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());

         COMPILER->popContext();
         COMPILER->popContextSet();
         COMPILER->popFunction();
      ;}
    break;

  case 303:
#line 1874 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
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

  case 304:
#line 1896 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
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

  case 307:
#line 1926 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
      COMPILER->raiseError(Falcon::e_syn_hasdef );
   ;}
    break;

  case 308:
#line 1933 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
         Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();

         // The symbolmay be undefined or defined; it's not our task to define it here.
         clsdef->has().pushBack( COMPILER->addGlobalSymbol( (yyvsp[(1) - (1)].stringp) ) );
      ;}
    break;

  case 309:
#line 1941 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
         Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();
         clsdef->hasnt().pushBack( COMPILER->addGlobalSymbol( (yyvsp[(2) - (2)].stringp) ) );
      ;}
    break;

  case 310:
#line 1947 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
         Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();
         clsdef->has().pushBack( COMPILER->addGlobalSymbol( (yyvsp[(3) - (3)].stringp) ) );
      ;}
    break;

  case 311:
#line 1953 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
         Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();
         clsdef->hasnt().pushBack( COMPILER->addGlobalSymbol( (yyvsp[(4) - (4)].stringp) ) );
      ;}
    break;

  case 312:
#line 1966 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
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

  case 313:
#line 2006 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
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

  case 315:
#line 2031 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_object );
      ;}
    break;

  case 319:
#line 2043 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
      COMPILER->addFunction( (yyvsp[(1) - (1)].fal_stat) );
   ;}
    break;

  case 320:
#line 2046 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
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

  case 322:
#line 2074 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         Falcon::StmtGlobal *glob = new Falcon::StmtGlobal( CURRENT_LINE );
         COMPILER->pushContext( glob );
      ;}
    break;

  case 323:
#line 2079 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
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

  case 326:
#line 2094 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_global );
      ;}
    break;

  case 327:
#line 2101 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         // we create (or retrieve) a globalized symbol
         Falcon::Symbol *sym = COMPILER->globalize( (yyvsp[(1) - (1)].stringp) );

         // then we add the symbol to the global statement (it's just for symbolic asm generation).
         Falcon::StmtGlobal *glob = static_cast<Falcon::StmtGlobal *>( COMPILER->getContext() );
         glob->addSymbol( sym );
      ;}
    break;

  case 328:
#line 2116 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { (yyval.fal_stat) = new Falcon::StmtReturn(LINE, 0); ;}
    break;

  case 329:
#line 2117 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { (yyval.fal_stat) = new Falcon::StmtReturn( LINE, (yyvsp[(2) - (3)].fal_val) ); ;}
    break;

  case 330:
#line 2118 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_return ); (yyval.fal_stat) = 0; ;}
    break;

  case 331:
#line 2128 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); ;}
    break;

  case 332:
#line 2129 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].integer) ); ;}
    break;

  case 333:
#line 2130 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].numeric) ); ;}
    break;

  case 334:
#line 2131 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].stringp) ); ;}
    break;

  case 335:
#line 2136 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
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

  case 337:
#line 2154 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setSelf(); ;}
    break;

  case 338:
#line 2155 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setSender(); ;}
    break;

  case 340:
#line 2167 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_access, (yyvsp[(1) - (2)].fal_val), (yyvsp[(2) - (2)].fal_val) );
         (yyval.fal_val) = new Falcon::Value( exp );
      ;}
    break;

  case 341:
#line 2172 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_access, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
         (yyval.fal_val) = new Falcon::Value( exp );
      ;}
    break;

  case 342:
#line 2177 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_byte_access, (yyvsp[(1) - (5)].fal_val), (yyvsp[(4) - (5)].fal_val) );
         (yyval.fal_val) = new Falcon::Value( exp );
      ;}
    break;

  case 343:
#line 2183 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_obj_access, (yyvsp[(1) - (3)].fal_val), new Falcon::Value( (yyvsp[(3) - (3)].stringp) ) );
         (yyval.fal_val) = new Falcon::Value( exp );
      ;}
    break;

  case 346:
#line 2194 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_plus, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 347:
#line 2195 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_neg, (yyvsp[(2) - (2)].fal_val) ) ); ;}
    break;

  case 348:
#line 2196 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_minus, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 349:
#line 2197 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_times, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 350:
#line 2198 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_divide, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 351:
#line 2199 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_modulo, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 352:
#line 2200 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_power, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 353:
#line 2201 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_and, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 354:
#line 2202 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_or, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 355:
#line 2203 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_xor, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 356:
#line 2204 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_shift_left, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 357:
#line 2205 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_shift_right, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 358:
#line 2206 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_not, (yyvsp[(2) - (2)].fal_val) ) ); ;}
    break;

  case 359:
#line 2207 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { COMPILER->defineVal( (yyvsp[(2) - (4)].fal_val) ); (yyval.fal_val) =
        new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_let, (yyvsp[(2) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) ); ;}
    break;

  case 360:
#line 2209 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { COMPILER->defineVal( (yyvsp[(2) - (4)].fal_val) ); (yyval.fal_val) =
        new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_let, (yyvsp[(2) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) ); ;}
    break;

  case 361:
#line 2211 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_neq, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 362:
#line 2212 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_post_inc, (yyvsp[(1) - (2)].fal_val) ) ); ;}
    break;

  case 363:
#line 2213 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_pre_inc, (yyvsp[(2) - (2)].fal_val) ) ); ;}
    break;

  case 364:
#line 2214 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_post_dec, (yyvsp[(1) - (2)].fal_val) ) ); ;}
    break;

  case 365:
#line 2215 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_pre_dec, (yyvsp[(2) - (2)].fal_val) ) ); ;}
    break;

  case 366:
#line 2216 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_eq, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 367:
#line 2217 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_eq, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 368:
#line 2218 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_eq, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 369:
#line 2219 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_gt, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 370:
#line 2220 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_lt, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 371:
#line 2221 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_ge, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 372:
#line 2222 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_le, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 373:
#line 2223 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_and, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 374:
#line 2224 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_or, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 375:
#line 2225 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_not, (yyvsp[(2) - (2)].fal_val) ) ); ;}
    break;

  case 376:
#line 2226 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_has, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 377:
#line 2227 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_hasnt, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 378:
#line 2228 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_in, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 379:
#line 2229 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_notin, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 380:
#line 2230 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_provides, (yyvsp[(1) - (3)].fal_val), new Falcon::Value( (yyvsp[(3) - (3)].stringp) ) ) ); ;}
    break;

  case 381:
#line 2231 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (2)].fal_val) ); ;}
    break;

  case 382:
#line 2232 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_strexpand, (yyvsp[(2) - (2)].fal_val) ) ); ;}
    break;

  case 383:
#line 2233 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_indirect, (yyvsp[(2) - (2)].fal_val) ) ); ;}
    break;

  case 386:
#line 2236 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_obj_access, (yyvsp[(1) - (3)].fal_val), new Falcon::Value( (yyvsp[(3) - (3)].stringp) ) );
         (yyval.fal_val) = new Falcon::Value( exp );
      ;}
    break;

  case 387:
#line 2240 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_access, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
         (yyval.fal_val) = new Falcon::Value( exp );
      ;}
    break;

  case 388:
#line 2244 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_byte_access, (yyvsp[(1) - (5)].fal_val), (yyvsp[(4) - (5)].fal_val) );
         (yyval.fal_val) = new Falcon::Value( exp );
      ;}
    break;

  case 389:
#line 2248 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_access, (yyvsp[(1) - (2)].fal_val), (yyvsp[(2) - (2)].fal_val) );
         (yyval.fal_val) = new Falcon::Value( exp );
      ;}
    break;

  case 394:
#line 2256 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(3) - (5)].fal_val); ;}
    break;

  case 395:
#line 2261 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( new Falcon::Value( (Falcon::int64) 0 ) ) );
      ;}
    break;

  case 396:
#line 2264 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( (yyvsp[(2) - (4)].fal_val) ) );
      ;}
    break;

  case 397:
#line 2267 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( new Falcon::Value( (Falcon::int64) 0 ), (yyvsp[(3) - (4)].fal_val) ) );
      ;}
    break;

  case 398:
#line 2270 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( (yyvsp[(2) - (5)].fal_val), (yyvsp[(4) - (5)].fal_val) ) );
      ;}
    break;

  case 399:
#line 2277 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_funcall,
                                      (yyvsp[(1) - (6)].fal_val), new Falcon::Value( (yyvsp[(4) - (6)].fal_adecl) ) ) );
      ;}
    break;

  case 400:
#line 2283 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_funcall, (yyvsp[(1) - (4)].fal_val), 0 ) );
      ;}
    break;

  case 401:
#line 2287 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); ;}
    break;

  case 402:
#line 2288 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         delete (yyvsp[(4) - (8)].fal_adecl);
         COMPILER->raiseError(Falcon::e_syn_funcall, "", COMPILER->tempLine() );
         (yyval.fal_val) = new Falcon::Value;
      ;}
    break;

  case 407:
#line 2307 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
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

  case 408:
#line 2340 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
            Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());
            (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_lambda ,
               new Falcon::Value( func->symbol() ) ) );
            COMPILER->closeFunction();
         ;}
    break;

  case 410:
#line 2350 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); ;}
    break;

  case 411:
#line 2351 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_lambda );
      ;}
    break;

  case 412:
#line 2355 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_lambda );
      ;}
    break;

  case 413:
#line 2361 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_iif, (yyvsp[(1) - (5)].fal_val), (yyvsp[(3) - (5)].fal_val), (yyvsp[(5) - (5)].fal_val) ) ); ;}
    break;

  case 414:
#line 2363 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         delete (yyvsp[(1) - (3)].fal_val);
         COMPILER->raiseError(Falcon::e_syn_iif );
         (yyval.fal_val) = new Falcon::Value;
      ;}
    break;

  case 415:
#line 2372 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {  (yyval.fal_val) = new Falcon::Value( new Falcon::ArrayDecl() ); ;}
    break;

  case 416:
#line 2374 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (4)].fal_adecl) );
      ;}
    break;

  case 417:
#line 2377 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); ;}
    break;

  case 418:
#line 2378 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_arraydecl, "", COMPILER->tempLine() );
         (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (6)].fal_adecl) );
      ;}
    break;

  case 419:
#line 2385 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {  (yyval.fal_val) = new Falcon::Value( new Falcon::DictDecl() ); ;}
    break;

  case 420:
#line 2386 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (4)].fal_ddecl) ); ;}
    break;

  case 421:
#line 2387 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); ;}
    break;

  case 422:
#line 2388 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_dictdecl, "", COMPILER->tempLine() );
         (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (6)].fal_ddecl) );
      ;}
    break;

  case 423:
#line 2395 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { (yyval.fal_adecl) = new Falcon::ArrayDecl(); (yyval.fal_adecl)->pushBack( (yyvsp[(1) - (1)].fal_val) ); ;}
    break;

  case 424:
#line 2396 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { (yyvsp[(1) - (3)].fal_adecl)->pushBack( (yyvsp[(3) - (3)].fal_val) ); (yyval.fal_adecl) = (yyvsp[(1) - (3)].fal_adecl); ;}
    break;

  case 425:
#line 2400 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { (yyval.fal_adecl) = new Falcon::ArrayDecl(); (yyval.fal_adecl)->pushBack( (yyvsp[(1) - (1)].fal_val) ); ;}
    break;

  case 426:
#line 2401 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { (yyvsp[(1) - (4)].fal_adecl)->pushBack( (yyvsp[(4) - (4)].fal_val) ); (yyval.fal_adecl) = (yyvsp[(1) - (4)].fal_adecl); ;}
    break;

  case 427:
#line 2405 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(1) - (1)].fal_val) );
         Falcon::ArrayDecl *ad = new Falcon::ArrayDecl();
         ad->pushBack( (yyvsp[(1) - (1)].fal_val) );
         COMPILER->defineVal( (yyvsp[(1) - (1)].fal_val) );
         (yyval.fal_adecl) = ad;
      ;}
    break;

  case 428:
#line 2412 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(3) - (3)].fal_val) );
         (yyvsp[(1) - (3)].fal_adecl)->pushBack( (yyvsp[(3) - (3)].fal_val) );
      ;}
    break;

  case 429:
#line 2419 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { (yyval.fal_ddecl) = new Falcon::DictDecl(); (yyval.fal_ddecl)->pushBack( (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ); ;}
    break;

  case 430:
#line 2420 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
    { (yyvsp[(1) - (6)].fal_ddecl)->pushBack( (yyvsp[(4) - (6)].fal_val), (yyvsp[(6) - (6)].fal_val) ); (yyval.fal_ddecl) = (yyvsp[(1) - (6)].fal_ddecl); ;}
    break;


/* Line 1267 of yacc.c.  */
#line 6067 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.cpp"
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


#line 2424 "/home/gian/Progetti/falcon/core_ftd/engine/src_parser.yy"
 /* c code */


void flc_src_error (const char *s)  /* Called by yyparse on error */
{
   /* do nothing: manage it in the action */
}

/* end of src_parser.yy */


