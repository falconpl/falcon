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
     CLOSEPAR = 305,
     OPENPAR = 306,
     CLOSESQUARE = 307,
     OPENSQUARE = 308,
     DOT = 309,
     ASSIGN_POW = 310,
     ASSIGN_SHL = 311,
     ASSIGN_SHR = 312,
     ASSIGN_BXOR = 313,
     ASSIGN_BOR = 314,
     ASSIGN_BAND = 315,
     ASSIGN_MOD = 316,
     ASSIGN_DIV = 317,
     ASSIGN_MUL = 318,
     ASSIGN_SUB = 319,
     ASSIGN_ADD = 320,
     ARROW = 321,
     FOR_STEP = 322,
     OP_TO = 323,
     COMMA = 324,
     QUESTION = 325,
     OR = 326,
     AND = 327,
     NOT = 328,
     LET = 329,
     LE = 330,
     GE = 331,
     LT = 332,
     GT = 333,
     NEQ = 334,
     EEQ = 335,
     OP_EQ = 336,
     OP_ASSIGN = 337,
     PROVIDES = 338,
     OP_NOTIN = 339,
     OP_IN = 340,
     HASNT = 341,
     HAS = 342,
     DIESIS = 343,
     ATSIGN = 344,
     CAP = 345,
     VBAR = 346,
     AMPER = 347,
     MINUS = 348,
     PLUS = 349,
     PERCENT = 350,
     SLASH = 351,
     STAR = 352,
     POW = 353,
     SHR = 354,
     SHL = 355,
     BANG = 356,
     NEG = 357,
     DECREMENT = 358,
     INCREMENT = 359,
     DOLLAR = 360
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
#define CLOSEPAR 305
#define OPENPAR 306
#define CLOSESQUARE 307
#define OPENSQUARE 308
#define DOT 309
#define ASSIGN_POW 310
#define ASSIGN_SHL 311
#define ASSIGN_SHR 312
#define ASSIGN_BXOR 313
#define ASSIGN_BOR 314
#define ASSIGN_BAND 315
#define ASSIGN_MOD 316
#define ASSIGN_DIV 317
#define ASSIGN_MUL 318
#define ASSIGN_SUB 319
#define ASSIGN_ADD 320
#define ARROW 321
#define FOR_STEP 322
#define OP_TO 323
#define COMMA 324
#define QUESTION 325
#define OR 326
#define AND 327
#define NOT 328
#define LET 329
#define LE 330
#define GE 331
#define LT 332
#define GT 333
#define NEQ 334
#define EEQ 335
#define OP_EQ 336
#define OP_ASSIGN 337
#define PROVIDES 338
#define OP_NOTIN 339
#define OP_IN 340
#define HASNT 341
#define HAS 342
#define DIESIS 343
#define ATSIGN 344
#define CAP 345
#define VBAR 346
#define AMPER 347
#define MINUS 348
#define PLUS 349
#define PERCENT 350
#define SLASH 351
#define STAR 352
#define POW 353
#define SHR 354
#define SHL 355
#define BANG 356
#define NEG 357
#define DECREMENT 358
#define INCREMENT 359
#define DOLLAR 360




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
#line 369 "/home/gian/Progetti/falcon/core/engine/src_parser.cpp"
	YYSTYPE;
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif



/* Copy the second part of user declarations.  */


/* Line 216 of yacc.c.  */
#line 382 "/home/gian/Progetti/falcon/core/engine/src_parser.cpp"

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
#define YYLAST   6407

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  106
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  167
/* YYNRULES -- Number of rules.  */
#define YYNRULES  428
/* YYNRULES -- Number of states.  */
#define YYNSTATES  817

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   360

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
     105
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
     102,   104,   106,   108,   110,   112,   114,   116,   118,   122,
     126,   131,   137,   142,   149,   156,   158,   160,   162,   164,
     166,   168,   170,   172,   174,   176,   178,   183,   188,   193,
     198,   203,   208,   213,   218,   223,   228,   233,   234,   240,
     243,   247,   250,   254,   258,   261,   265,   266,   273,   276,
     280,   284,   288,   292,   293,   295,   296,   300,   303,   307,
     308,   313,   317,   321,   322,   325,   328,   332,   335,   339,
     343,   344,   350,   353,   361,   371,   375,   383,   393,   397,
     398,   408,   415,   421,   422,   425,   427,   429,   431,   433,
     437,   441,   445,   448,   452,   455,   459,   460,   467,   471,
     475,   476,   483,   487,   491,   492,   499,   503,   507,   508,
     515,   519,   523,   524,   527,   531,   533,   534,   540,   541,
     547,   548,   554,   555,   561,   562,   563,   567,   568,   570,
     573,   576,   579,   581,   585,   587,   589,   591,   595,   597,
     598,   605,   609,   613,   614,   617,   621,   623,   624,   630,
     631,   637,   638,   644,   645,   651,   653,   657,   658,   660,
     662,   668,   673,   677,   681,   682,   689,   692,   696,   697,
     699,   701,   704,   707,   710,   715,   719,   725,   729,   731,
     735,   737,   739,   743,   747,   753,   756,   764,   765,   775,
     779,   787,   788,   797,   800,   801,   803,   808,   810,   811,
     812,   818,   819,   823,   826,   830,   833,   837,   841,   845,
     849,   855,   861,   865,   871,   877,   881,   884,   888,   892,
     894,   898,   903,   907,   910,   914,   917,   921,   922,   924,
     928,   931,   935,   938,   939,   948,   952,   955,   956,   962,
     963,   971,   972,   975,   977,   981,   984,   985,   991,   993,
     997,   999,  1001,  1003,  1004,  1007,  1009,  1011,  1013,  1015,
    1016,  1024,  1030,  1035,  1036,  1040,  1044,  1046,  1049,  1053,
    1058,  1059,  1068,  1071,  1074,  1075,  1078,  1080,  1082,  1084,
    1086,  1087,  1092,  1094,  1098,  1102,  1104,  1107,  1111,  1115,
    1117,  1119,  1121,  1123,  1125,  1127,  1129,  1131,  1133,  1136,
    1141,  1147,  1151,  1153,  1155,  1159,  1162,  1166,  1170,  1174,
    1178,  1182,  1186,  1190,  1194,  1198,  1202,  1205,  1210,  1215,
    1219,  1222,  1225,  1228,  1231,  1235,  1239,  1243,  1247,  1251,
    1255,  1259,  1263,  1267,  1270,  1274,  1278,  1282,  1286,  1290,
    1293,  1296,  1299,  1301,  1303,  1307,  1312,  1318,  1321,  1323,
    1325,  1327,  1329,  1335,  1339,  1344,  1349,  1355,  1362,  1367,
    1368,  1377,  1378,  1380,  1382,  1385,  1386,  1393,  1400,  1401,
    1410,  1413,  1419,  1423,  1426,  1431,  1432,  1439,  1443,  1448,
    1449,  1456,  1458,  1462,  1464,  1469,  1471,  1475,  1479
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int16 yyrhs[] =
{
     107,     0,    -1,   108,    -1,    -1,   108,   109,    -1,   110,
      -1,     9,     3,    -1,    23,     1,     3,    -1,   111,    -1,
     199,    -1,   222,    -1,   240,    -1,   112,    -1,   214,    -1,
     215,    -1,   217,    -1,    39,     6,     3,    -1,    39,     7,
       3,    -1,    39,     1,     3,    -1,   114,    -1,     3,    -1,
      46,     1,     3,    -1,    34,     1,     3,    -1,    32,     1,
       3,    -1,     1,     3,    -1,   251,    82,   254,    -1,   113,
      69,   251,    82,   254,    -1,   254,     3,    -1,   115,    -1,
     116,    -1,   117,    -1,   129,    -1,   146,    -1,   150,    -1,
     162,    -1,   177,    -1,   133,    -1,   144,    -1,   145,    -1,
     188,    -1,   189,    -1,   198,    -1,   249,    -1,   245,    -1,
     212,    -1,   213,    -1,   154,    -1,   155,    -1,    10,   113,
       3,    -1,    10,     1,     3,    -1,   253,    82,   254,     3,
      -1,   253,    82,   105,   105,     3,    -1,   253,    82,   269,
       3,    -1,   253,    69,   271,    82,   254,     3,    -1,   253,
      69,   271,    82,   269,     3,    -1,   118,    -1,   119,    -1,
     120,    -1,   121,    -1,   122,    -1,   123,    -1,   124,    -1,
     125,    -1,   126,    -1,   127,    -1,   128,    -1,   254,    65,
     254,     3,    -1,   253,    64,   254,     3,    -1,   253,    63,
     254,     3,    -1,   253,    62,   254,     3,    -1,   253,    61,
     254,     3,    -1,   253,    55,   254,     3,    -1,   253,    60,
     254,     3,    -1,   253,    59,   254,     3,    -1,   253,    58,
     254,     3,    -1,   253,    56,   254,     3,    -1,   253,    57,
     254,     3,    -1,    -1,   131,   130,   143,     9,     3,    -1,
     132,   112,    -1,    11,   254,     3,    -1,    49,     3,    -1,
      11,     1,     3,    -1,    11,   254,    45,    -1,    49,    45,
      -1,    11,     1,    45,    -1,    -1,   135,   134,   143,   137,
       9,     3,    -1,   136,   112,    -1,    15,   254,     3,    -1,
      15,     1,     3,    -1,    15,   254,    45,    -1,    15,     1,
      45,    -1,    -1,   140,    -1,    -1,   139,   138,   143,    -1,
      16,     3,    -1,    16,     1,     3,    -1,    -1,   142,   141,
     143,   137,    -1,    17,   254,     3,    -1,    17,     1,     3,
      -1,    -1,   143,   112,    -1,    12,     3,    -1,    12,     1,
       3,    -1,    13,     3,    -1,    13,    14,     3,    -1,    13,
       1,     3,    -1,    -1,   148,   147,   143,     9,     3,    -1,
     149,   112,    -1,    18,   253,    82,   254,    68,   254,     3,
      -1,    18,   253,    82,   254,    68,   254,    67,   254,     3,
      -1,    18,     1,     3,    -1,    18,   253,    82,   254,    68,
     254,    45,    -1,    18,   253,    82,   254,    68,   254,    67,
     254,    45,    -1,    18,     1,    45,    -1,    -1,    18,   271,
      85,   254,     3,   151,   152,     9,     3,    -1,    18,   271,
      85,   254,    45,   112,    -1,    18,   271,    85,     1,     3,
      -1,    -1,   153,   152,    -1,   112,    -1,   156,    -1,   158,
      -1,   160,    -1,    48,   254,     3,    -1,    48,     1,     3,
      -1,    77,   269,     3,    -1,    77,     3,    -1,    78,   269,
       3,    -1,    78,     3,    -1,    77,     1,     3,    -1,    -1,
      19,     3,   157,   143,     9,     3,    -1,    19,    45,   112,
      -1,    19,     1,     3,    -1,    -1,    20,     3,   159,   143,
       9,     3,    -1,    20,    45,   112,    -1,    20,     1,     3,
      -1,    -1,    21,     3,   161,   143,     9,     3,    -1,    21,
      45,   112,    -1,    21,     1,     3,    -1,    -1,   164,   163,
     165,   171,     9,     3,    -1,    22,   254,     3,    -1,    22,
       1,     3,    -1,    -1,   165,   166,    -1,   165,     1,     3,
      -1,     3,    -1,    -1,    23,   175,     3,   167,   143,    -1,
      -1,    23,   175,    45,   168,   112,    -1,    -1,    23,     1,
       3,   169,   143,    -1,    -1,    23,     1,    45,   170,   112,
      -1,    -1,    -1,   173,   172,   174,    -1,    -1,    24,    -1,
      24,     1,    -1,     3,   143,    -1,    45,   112,    -1,   176,
      -1,   175,    69,   176,    -1,     8,    -1,     4,    -1,     7,
      -1,     4,    68,     4,    -1,     6,    -1,    -1,   179,   178,
     180,   171,     9,     3,    -1,    25,   254,     3,    -1,    25,
       1,     3,    -1,    -1,   180,   181,    -1,   180,     1,     3,
      -1,     3,    -1,    -1,    23,   186,     3,   182,   143,    -1,
      -1,    23,   186,    45,   183,   112,    -1,    -1,    23,     1,
       3,   184,   143,    -1,    -1,    23,     1,    45,   185,   112,
      -1,   187,    -1,   186,    69,   187,    -1,    -1,     4,    -1,
       6,    -1,    28,   269,    68,   254,     3,    -1,    28,   269,
       1,     3,    -1,    28,     1,     3,    -1,    29,    45,   112,
      -1,    -1,   191,   190,   143,   192,     9,     3,    -1,    29,
       3,    -1,    29,     1,     3,    -1,    -1,   193,    -1,   194,
      -1,   193,   194,    -1,   195,   143,    -1,    30,     3,    -1,
      30,    85,   251,     3,    -1,    30,   196,     3,    -1,    30,
     196,    85,   251,     3,    -1,    30,     1,     3,    -1,   197,
      -1,   196,    69,   197,    -1,     4,    -1,     6,    -1,    31,
     254,     3,    -1,    31,     1,     3,    -1,   200,   207,   143,
       9,     3,    -1,   202,   112,    -1,   204,    51,   258,   205,
     258,    50,     3,    -1,    -1,   204,    51,   258,   205,     1,
     201,   258,    50,     3,    -1,   204,     1,     3,    -1,   204,
      51,   258,   205,   258,    50,    45,    -1,    -1,   204,    51,
     258,     1,   203,   258,    50,    45,    -1,    46,     6,    -1,
      -1,   206,    -1,   205,    69,   258,   206,    -1,     6,    -1,
      -1,    -1,   210,   208,   143,     9,     3,    -1,    -1,   211,
     209,   112,    -1,    47,     3,    -1,    47,     1,     3,    -1,
      47,    45,    -1,    47,     1,    45,    -1,    40,   256,     3,
      -1,    40,     1,     3,    -1,    43,   254,     3,    -1,    43,
     254,    85,   254,     3,    -1,    43,   254,    85,     1,     3,
      -1,    43,     1,     3,    -1,    41,     6,    82,   250,     3,
      -1,    41,     6,    82,     1,     3,    -1,    41,     1,     3,
      -1,    44,     3,    -1,    44,   216,     3,    -1,    44,     1,
       3,    -1,     6,    -1,   216,    69,     6,    -1,   218,   221,
       9,     3,    -1,   219,   220,     3,    -1,    42,     3,    -1,
      42,     1,     3,    -1,    42,    45,    -1,    42,     1,    45,
      -1,    -1,     6,    -1,   220,    69,     6,    -1,   220,     3,
      -1,   221,   220,     3,    -1,     1,     3,    -1,    -1,    32,
       6,   223,   224,   233,   238,     9,     3,    -1,   225,   227,
       3,    -1,     1,     3,    -1,    -1,    51,   258,   205,   258,
      50,    -1,    -1,    51,   258,   205,     1,   226,   258,    50,
      -1,    -1,    33,   228,    -1,   229,    -1,   228,    69,   229,
      -1,     6,   230,    -1,    -1,    51,   258,   231,   258,    50,
      -1,   232,    -1,   231,    69,   232,    -1,   250,    -1,     6,
      -1,    27,    -1,    -1,   233,   234,    -1,     3,    -1,   199,
      -1,   237,    -1,   235,    -1,    -1,    38,     3,   236,   207,
     143,     9,     3,    -1,    47,     6,    82,   254,     3,    -1,
       6,    82,   254,     3,    -1,    -1,    87,   239,     3,    -1,
      87,     1,     3,    -1,     6,    -1,    73,     6,    -1,   239,
      69,     6,    -1,   239,    69,    73,     6,    -1,    -1,    34,
       6,   241,   242,   243,   238,     9,     3,    -1,   227,     3,
      -1,     1,     3,    -1,    -1,   243,   244,    -1,     3,    -1,
     199,    -1,   237,    -1,   235,    -1,    -1,    36,   246,   247,
       3,    -1,   248,    -1,   247,    69,   248,    -1,   247,    69,
       1,    -1,     6,    -1,    35,     3,    -1,    35,   254,     3,
      -1,    35,     1,     3,    -1,     8,    -1,     4,    -1,     5,
      -1,     7,    -1,     6,    -1,   251,    -1,    27,    -1,    26,
      -1,   252,    -1,   253,   255,    -1,   253,    53,   254,    52,
      -1,   253,    53,    97,   254,    52,    -1,   253,    54,     6,
      -1,   250,    -1,   253,    -1,   254,    94,   254,    -1,    93,
     254,    -1,   254,    93,   254,    -1,   254,    97,   254,    -1,
     254,    96,   254,    -1,   254,    95,   254,    -1,   254,    98,
     254,    -1,   254,    92,   254,    -1,   254,    91,   254,    -1,
     254,    90,   254,    -1,   254,   100,   254,    -1,   254,    99,
     254,    -1,   101,   254,    -1,    74,   253,    81,   254,    -1,
      74,   253,    82,   254,    -1,   254,    79,   254,    -1,   254,
     104,    -1,   104,   254,    -1,   254,   103,    -1,   103,   254,
      -1,   254,    80,   254,    -1,   254,    81,   254,    -1,   254,
      82,   254,    -1,   254,    78,   254,    -1,   254,    77,   254,
      -1,   254,    76,   254,    -1,   254,    75,   254,    -1,   254,
      72,   254,    -1,   254,    71,   254,    -1,    73,   254,    -1,
     254,    87,   254,    -1,   254,    86,   254,    -1,   254,    85,
     254,    -1,   254,    84,   254,    -1,   254,    83,     6,    -1,
     105,   254,    -1,    89,   254,    -1,    88,   254,    -1,   260,
      -1,   256,    -1,   256,    54,     6,    -1,   256,    53,   254,
      52,    -1,   256,    53,    97,   254,    52,    -1,   256,   255,
      -1,   264,    -1,   265,    -1,   267,    -1,   255,    -1,    51,
     258,   254,   258,    50,    -1,    53,    45,    52,    -1,    53,
     254,    45,    52,    -1,    53,    45,   254,    52,    -1,    53,
     254,    45,   254,    52,    -1,   254,    51,   258,   270,   258,
      50,    -1,   254,    51,   258,    50,    -1,    -1,   254,    51,
     258,   270,     1,   257,   258,    50,    -1,    -1,   259,    -1,
       3,    -1,   259,     3,    -1,    -1,    37,   261,   262,   207,
     143,     9,    -1,    51,   258,   205,   258,    50,     3,    -1,
      -1,    51,   258,   205,     1,   263,   258,    50,     3,    -1,
       1,     3,    -1,   254,    70,   254,    45,   254,    -1,   254,
      70,     1,    -1,    53,    52,    -1,    53,   270,   258,    52,
      -1,    -1,    53,   270,     1,   266,   258,    52,    -1,    53,
      66,    52,    -1,    53,   272,   258,    52,    -1,    -1,    53,
     272,     1,   268,   258,    52,    -1,   254,    -1,   269,    69,
     254,    -1,   254,    -1,   270,    69,   258,   254,    -1,   251,
      -1,   271,    69,   251,    -1,   254,    66,   254,    -1,   272,
      69,   258,   254,    66,   254,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   188,   188,   191,   193,   197,   198,   199,   204,   209,
     214,   219,   224,   229,   230,   231,   235,   241,   247,   255,
     256,   259,   260,   261,   262,   267,   272,   279,   280,   281,
     282,   283,   284,   285,   286,   287,   288,   289,   290,   291,
     292,   293,   294,   295,   296,   297,   298,   299,   303,   305,
     311,   315,   319,   323,   328,   338,   339,   340,   341,   342,
     343,   344,   345,   346,   347,   348,   352,   359,   366,   373,
     380,   387,   394,   401,   408,   414,   420,   428,   428,   443,
     451,   452,   453,   457,   458,   459,   463,   463,   478,   488,
     489,   493,   494,   498,   500,   501,   501,   510,   511,   516,
     516,   528,   529,   532,   534,   540,   549,   557,   567,   576,
     584,   584,   598,   614,   618,   622,   630,   634,   638,   648,
     647,   671,   685,   689,   691,   695,   702,   703,   704,   708,
     721,   729,   733,   739,   744,   751,   759,   759,   773,   781,
     785,   785,   798,   806,   810,   810,   824,   832,   836,   836,
     853,   854,   861,   863,   864,   868,   870,   869,   880,   880,
     892,   892,   904,   904,   920,   923,   922,   935,   936,   937,
     940,   941,   947,   948,   952,   961,   973,   984,   995,  1016,
    1016,  1033,  1034,  1041,  1043,  1044,  1048,  1050,  1049,  1060,
    1060,  1073,  1073,  1085,  1085,  1103,  1104,  1107,  1108,  1120,
    1141,  1145,  1150,  1158,  1165,  1164,  1183,  1184,  1187,  1189,
    1193,  1194,  1198,  1203,  1221,  1241,  1251,  1262,  1270,  1271,
    1275,  1287,  1310,  1311,  1318,  1328,  1337,  1338,  1338,  1342,
    1346,  1347,  1347,  1354,  1408,  1410,  1411,  1415,  1430,  1433,
    1432,  1444,  1443,  1458,  1459,  1463,  1464,  1473,  1477,  1485,
    1492,  1502,  1508,  1520,  1530,  1535,  1547,  1556,  1563,  1571,
    1576,  1588,  1595,  1605,  1606,  1609,  1610,  1613,  1615,  1619,
    1626,  1627,  1628,  1640,  1639,  1698,  1701,  1707,  1709,  1710,
    1710,  1716,  1718,  1722,  1723,  1727,  1761,  1763,  1772,  1773,
    1777,  1778,  1787,  1790,  1792,  1796,  1797,  1800,  1819,  1823,
    1823,  1857,  1879,  1906,  1908,  1909,  1916,  1924,  1930,  1936,
    1950,  1949,  2013,  2014,  2020,  2022,  2026,  2027,  2030,  2049,
    2058,  2057,  2075,  2076,  2077,  2084,  2100,  2101,  2102,  2112,
    2113,  2114,  2115,  2119,  2137,  2138,  2139,  2150,  2151,  2156,
    2161,  2167,  2176,  2177,  2178,  2179,  2180,  2181,  2182,  2183,
    2184,  2185,  2186,  2187,  2188,  2189,  2190,  2191,  2193,  2195,
    2196,  2197,  2198,  2199,  2200,  2201,  2202,  2203,  2204,  2205,
    2206,  2207,  2208,  2209,  2210,  2211,  2212,  2213,  2214,  2215,
    2216,  2217,  2218,  2219,  2220,  2224,  2228,  2232,  2236,  2237,
    2238,  2239,  2240,  2245,  2248,  2251,  2254,  2260,  2266,  2271,
    2271,  2279,  2281,  2285,  2286,  2291,  2290,  2333,  2334,  2334,
    2338,  2345,  2346,  2356,  2357,  2361,  2361,  2369,  2370,  2371,
    2371,  2379,  2380,  2384,  2385,  2389,  2396,  2403,  2404
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
  "FUNCDECL", "STATIC", "FORDOT", "LOOP", "CLOSEPAR", "OPENPAR",
  "CLOSESQUARE", "OPENSQUARE", "DOT", "ASSIGN_POW", "ASSIGN_SHL",
  "ASSIGN_SHR", "ASSIGN_BXOR", "ASSIGN_BOR", "ASSIGN_BAND", "ASSIGN_MOD",
  "ASSIGN_DIV", "ASSIGN_MUL", "ASSIGN_SUB", "ASSIGN_ADD", "ARROW",
  "FOR_STEP", "OP_TO", "COMMA", "QUESTION", "OR", "AND", "NOT", "LET",
  "LE", "GE", "LT", "GT", "NEQ", "EEQ", "OP_EQ", "OP_ASSIGN", "PROVIDES",
  "OP_NOTIN", "OP_IN", "HASNT", "HAS", "DIESIS", "ATSIGN", "CAP", "VBAR",
  "AMPER", "MINUS", "PLUS", "PERCENT", "SLASH", "STAR", "POW", "SHR",
  "SHL", "BANG", "NEG", "DECREMENT", "INCREMENT", "DOLLAR", "$accept",
  "input", "body", "line", "toplevel_statement", "load_statement",
  "statement", "assignment_def_list", "base_statement", "def_statement",
  "assignment", "op_assignment", "autoadd", "autosub", "automul",
  "autodiv", "automod", "autopow", "autoband", "autobor", "autobxor",
  "autoshl", "autoshr", "while_statement", "@1", "while_decl",
  "while_short_decl", "if_statement", "@2", "if_decl", "if_short_decl",
  "elif_or_else", "@3", "else_decl", "elif_statement", "@4", "elif_decl",
  "statement_list", "break_statement", "continue_statement",
  "for_statement", "@5", "for_decl", "for_decl_short", "forin_statement",
  "@6", "forin_statement_list", "forin_statement_elem", "fordot_statement",
  "self_print_statement", "first_loop_block", "@7", "last_loop_block",
  "@8", "all_loop_block", "@9", "switch_statement", "@10", "switch_decl",
  "case_list", "case_statement", "@11", "@12", "@13", "@14",
  "default_statement", "@15", "default_decl", "default_body",
  "case_expression_list", "case_element", "select_statement", "@16",
  "select_decl", "selcase_list", "selcase_statement", "@17", "@18", "@19",
  "@20", "selcase_expression_list", "selcase_element", "give_statement",
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
     355,   356,   357,   358,   359,   360
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint16 yyr1[] =
{
       0,   106,   107,   108,   108,   109,   109,   109,   110,   110,
     110,   110,   110,   110,   110,   110,   111,   111,   111,   112,
     112,   112,   112,   112,   112,   113,   113,   114,   114,   114,
     114,   114,   114,   114,   114,   114,   114,   114,   114,   114,
     114,   114,   114,   114,   114,   114,   114,   114,   115,   115,
     116,   116,   116,   116,   116,   117,   117,   117,   117,   117,
     117,   117,   117,   117,   117,   117,   118,   119,   120,   121,
     122,   123,   124,   125,   126,   127,   128,   130,   129,   129,
     131,   131,   131,   132,   132,   132,   134,   133,   133,   135,
     135,   136,   136,   137,   137,   138,   137,   139,   139,   141,
     140,   142,   142,   143,   143,   144,   144,   145,   145,   145,
     147,   146,   146,   148,   148,   148,   149,   149,   149,   151,
     150,   150,   150,   152,   152,   153,   153,   153,   153,   154,
     154,   155,   155,   155,   155,   155,   157,   156,   156,   156,
     159,   158,   158,   158,   161,   160,   160,   160,   163,   162,
     164,   164,   165,   165,   165,   166,   167,   166,   168,   166,
     169,   166,   170,   166,   171,   172,   171,   173,   173,   173,
     174,   174,   175,   175,   176,   176,   176,   176,   176,   178,
     177,   179,   179,   180,   180,   180,   181,   182,   181,   183,
     181,   184,   181,   185,   181,   186,   186,   187,   187,   187,
     188,   188,   188,   189,   190,   189,   191,   191,   192,   192,
     193,   193,   194,   195,   195,   195,   195,   195,   196,   196,
     197,   197,   198,   198,   199,   199,   200,   201,   200,   200,
     202,   203,   202,   204,   205,   205,   205,   206,   207,   208,
     207,   209,   207,   210,   210,   211,   211,   212,   212,   213,
     213,   213,   213,   214,   214,   214,   215,   215,   215,   216,
     216,   217,   217,   218,   218,   219,   219,   220,   220,   220,
     221,   221,   221,   223,   222,   224,   224,   225,   225,   226,
     225,   227,   227,   228,   228,   229,   230,   230,   231,   231,
     232,   232,   232,   233,   233,   234,   234,   234,   234,   236,
     235,   237,   237,   238,   238,   238,   239,   239,   239,   239,
     241,   240,   242,   242,   243,   243,   244,   244,   244,   244,
     246,   245,   247,   247,   247,   248,   249,   249,   249,   250,
     250,   250,   250,   251,   252,   252,   252,   253,   253,   253,
     253,   253,   254,   254,   254,   254,   254,   254,   254,   254,
     254,   254,   254,   254,   254,   254,   254,   254,   254,   254,
     254,   254,   254,   254,   254,   254,   254,   254,   254,   254,
     254,   254,   254,   254,   254,   254,   254,   254,   254,   254,
     254,   254,   254,   254,   254,   254,   254,   254,   254,   254,
     254,   254,   254,   255,   255,   255,   255,   256,   256,   257,
     256,   258,   258,   259,   259,   261,   260,   262,   263,   262,
     262,   264,   264,   265,   265,   266,   265,   267,   267,   268,
     267,   269,   269,   270,   270,   271,   271,   272,   272
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     1,     0,     2,     1,     2,     3,     1,     1,
       1,     1,     1,     1,     1,     1,     3,     3,     3,     1,
       1,     3,     3,     3,     2,     3,     5,     2,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     3,     3,
       4,     5,     4,     6,     6,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     4,     4,     4,     4,
       4,     4,     4,     4,     4,     4,     4,     0,     5,     2,
       3,     2,     3,     3,     2,     3,     0,     6,     2,     3,
       3,     3,     3,     0,     1,     0,     3,     2,     3,     0,
       4,     3,     3,     0,     2,     2,     3,     2,     3,     3,
       0,     5,     2,     7,     9,     3,     7,     9,     3,     0,
       9,     6,     5,     0,     2,     1,     1,     1,     1,     3,
       3,     3,     2,     3,     2,     3,     0,     6,     3,     3,
       0,     6,     3,     3,     0,     6,     3,     3,     0,     6,
       3,     3,     0,     2,     3,     1,     0,     5,     0,     5,
       0,     5,     0,     5,     0,     0,     3,     0,     1,     2,
       2,     2,     1,     3,     1,     1,     1,     3,     1,     0,
       6,     3,     3,     0,     2,     3,     1,     0,     5,     0,
       5,     0,     5,     0,     5,     1,     3,     0,     1,     1,
       5,     4,     3,     3,     0,     6,     2,     3,     0,     1,
       1,     2,     2,     2,     4,     3,     5,     3,     1,     3,
       1,     1,     3,     3,     5,     2,     7,     0,     9,     3,
       7,     0,     8,     2,     0,     1,     4,     1,     0,     0,
       5,     0,     3,     2,     3,     2,     3,     3,     3,     3,
       5,     5,     3,     5,     5,     3,     2,     3,     3,     1,
       3,     4,     3,     2,     3,     2,     3,     0,     1,     3,
       2,     3,     2,     0,     8,     3,     2,     0,     5,     0,
       7,     0,     2,     1,     3,     2,     0,     5,     1,     3,
       1,     1,     1,     0,     2,     1,     1,     1,     1,     0,
       7,     5,     4,     0,     3,     3,     1,     2,     3,     4,
       0,     8,     2,     2,     0,     2,     1,     1,     1,     1,
       0,     4,     1,     3,     3,     1,     2,     3,     3,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     2,     4,
       5,     3,     1,     1,     3,     2,     3,     3,     3,     3,
       3,     3,     3,     3,     3,     3,     2,     4,     4,     3,
       2,     2,     2,     2,     3,     3,     3,     3,     3,     3,
       3,     3,     3,     2,     3,     3,     3,     3,     3,     2,
       2,     2,     1,     1,     3,     4,     5,     2,     1,     1,
       1,     1,     5,     3,     4,     4,     5,     6,     4,     0,
       8,     0,     1,     1,     2,     0,     6,     6,     0,     8,
       2,     5,     3,     2,     4,     0,     6,     3,     4,     0,
       6,     1,     3,     1,     4,     1,     3,     3,     6
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint16 yydefact[] =
{
       3,     0,     0,     1,     0,    20,   330,   331,   333,   332,
     329,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   336,   335,     0,     0,     0,     0,     0,     0,   320,
     405,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     401,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     4,     5,     8,    12,    19,    28,    29,
      30,    55,    56,    57,    58,    59,    60,    61,    62,    63,
      64,    65,    31,    77,     0,    36,    86,     0,    37,    38,
      32,   110,     0,    33,    46,    47,    34,   148,    35,   179,
      39,    40,   204,    41,     9,   238,     0,     0,    44,    45,
      13,    14,    15,     0,   267,    10,    11,    43,    42,   342,
     334,   337,   343,     0,   391,   383,   382,   388,   389,   390,
      24,     6,     0,     0,     0,     0,   343,     0,     0,   105,
       0,   107,     0,     0,     0,     0,   334,     0,     0,     0,
       0,     0,     0,     0,     0,   421,     0,     0,   206,     0,
       0,     0,     0,   273,     0,   310,     0,   326,     0,     0,
       0,     0,     0,     0,     0,     0,   383,     0,     0,     0,
     263,   265,     0,     0,     0,   256,   259,     0,     0,   233,
       0,     0,    81,    84,   403,     0,   402,     0,   413,     0,
     423,     0,     0,   373,     0,     0,   132,     0,   134,     0,
     381,   380,   345,   356,   363,   361,   379,   103,     0,     0,
       0,    79,   103,    88,   103,   112,   152,   183,   103,     0,
     103,   239,   241,   225,     0,   401,     0,   268,     0,   267,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   338,    27,   401,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   362,   360,     0,     0,
     387,    49,    48,     0,     0,    82,    85,    80,    83,   106,
     109,   108,    90,    92,    89,    91,   115,   118,     0,     0,
       0,   151,   150,     7,   182,   181,   202,     0,     0,     0,
     207,   203,   223,   222,    23,     0,    22,     0,   328,   327,
     325,     0,   322,     0,   401,   238,    18,    16,    17,   248,
     247,   255,     0,   264,   266,   252,   249,     0,   258,   257,
       0,    21,   130,   129,   401,   404,   393,     0,   417,     0,
       0,   415,   401,     0,   419,   401,     0,     0,     0,   135,
     131,   133,     0,     0,     0,     0,     0,     0,     0,   243,
     245,     0,   103,     0,   229,     0,   272,   270,     0,     0,
       0,   262,     0,     0,   341,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   425,     0,     0,   421,     0,
       0,     0,   412,     0,   372,   371,   370,   369,   368,   367,
     359,   364,   365,   366,   378,   377,   376,   375,   374,   353,
     352,   351,   346,   344,   349,   348,   347,   350,   355,   354,
       0,     0,   384,     0,    25,     0,   426,     0,     0,   201,
       0,   422,     0,   401,   293,   281,     0,     0,     0,   314,
     321,     0,   410,   234,   103,     0,     0,     0,   376,   260,
       0,   395,   394,     0,   427,   401,     0,   414,   401,     0,
     418,   357,   358,     0,   104,     0,     0,     0,    95,    94,
      99,     0,     0,   155,     0,     0,   153,     0,   165,     0,
     186,     0,     0,   184,     0,     0,   209,   210,   103,   244,
     246,     0,     0,   242,   231,   237,     0,   235,   269,   261,
     271,     0,   339,    71,    75,    76,    74,    73,    72,    70,
      69,    68,    67,     0,     0,    50,    52,   398,   423,     0,
      66,     0,     0,   385,     0,     0,   122,   119,     0,   200,
     276,   234,   303,     0,   313,   286,   282,   283,   312,   303,
     324,   323,     0,     0,   254,   253,   251,   250,   392,   396,
       0,   424,     0,     0,    78,     0,    97,     0,     0,     0,
     103,   103,   111,   154,     0,   175,   178,   176,   174,     0,
     172,   169,     0,     0,   185,     0,   198,   199,     0,   195,
       0,     0,   213,   220,   221,     0,     0,   218,     0,   211,
       0,   224,     0,   401,   227,   401,     0,   340,   421,     0,
      51,   399,     0,   411,   386,    26,     0,     0,   121,     0,
     295,     0,     0,     0,     0,     0,   296,   294,   298,   297,
       0,   275,   401,   285,     0,   316,   317,   319,   318,     0,
     315,   408,     0,   406,   416,   420,     0,    98,   102,   101,
      87,     0,     0,   160,   162,     0,   156,   158,     0,   149,
     103,     0,   166,   191,   193,   187,   189,   197,   180,   217,
       0,   215,     0,     0,   205,   240,     0,   401,     0,     0,
      53,    54,   401,   397,   113,   116,     0,     0,     0,     0,
     125,     0,     0,   126,   127,   128,   279,     0,     0,   299,
       0,     0,   306,     0,     0,     0,     0,   284,     0,   401,
       0,   428,   100,   103,     0,   177,   103,     0,   173,     0,
     171,   103,     0,   103,     0,   196,   214,   219,     0,     0,
       0,   236,   226,   230,     0,     0,     0,   136,     0,     0,
     140,     0,     0,   144,     0,     0,   124,   401,   278,     0,
     238,     0,   305,   307,   304,     0,   274,   291,   292,   401,
     288,   290,   311,     0,   407,     0,   163,     0,   159,     0,
     194,     0,   190,   216,   232,     0,   400,   114,   117,   139,
     103,   138,   143,   103,   142,   147,   103,   146,   120,     0,
     302,   103,     0,   308,     0,     0,     0,     0,   228,     0,
       0,     0,   280,     0,   301,   309,   289,   287,   409,     0,
       0,     0,     0,   137,   141,   145,   300
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     1,     2,    53,    54,    55,   474,   123,    57,    58,
      59,    60,    61,    62,    63,    64,    65,    66,    67,    68,
      69,    70,    71,    72,   207,    73,    74,    75,   212,    76,
      77,   477,   570,   478,   479,   571,   480,   362,    78,    79,
      80,   214,    81,    82,    83,   617,   691,   692,    84,    85,
     693,   780,   694,   783,   695,   786,    86,   216,    87,   365,
     486,   716,   717,   713,   714,   487,   583,   488,   662,   579,
     580,    88,   217,    89,   366,   493,   723,   724,   721,   722,
     588,   589,    90,    91,   218,    92,   495,   496,   497,   498,
     596,   597,    93,    94,    95,   677,    96,   603,    97,   506,
     507,   220,   372,   373,   221,   222,    98,    99,   100,   101,
     177,   102,   103,   104,   228,   229,   105,   315,   444,   445,
     747,   448,   546,   547,   633,   759,   760,   542,   627,   628,
     750,   629,   630,   704,   106,   317,   449,   549,   640,   107,
     159,   321,   322,   108,   109,   110,   111,   126,   113,   114,
     115,   682,   185,   186,   116,   160,   325,   709,   117,   118,
     465,   119,   468,   146,   191,   138,   192
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -445
static const yytype_int16 yypact[] =
{
    -445,    36,   862,  -445,    45,  -445,  -445,  -445,  -445,  -445,
    -445,    48,    32,   751,   149,   507,  2912,   497,  2967,    64,
    3022,  -445,  -445,  3077,   275,  3132,   155,   287,    16,  -445,
    -445,   428,  3187,   305,   300,  3242,   519,   358,  3297,   120,
      71,  5281,  5608,   479,   551,  3572,  5608,  5608,  5608,  5608,
    5608,  5608,  5608,  -445,  -445,  -445,  -445,  -445,  -445,  -445,
    -445,  -445,  -445,  -445,  -445,  -445,  -445,  -445,  -445,  -445,
    -445,  -445,  -445,  -445,  2857,  -445,  -445,  2857,  -445,  -445,
    -445,  -445,  2857,  -445,  -445,  -445,  -445,  -445,  -445,  -445,
    -445,  -445,  -445,  -445,  -445,    35,  2857,   112,  -445,  -445,
    -445,  -445,  -445,    53,    90,  -445,  -445,  -445,  -445,  -445,
    -445,  -445,   863,  3703,  -445,   302,  -445,  -445,  -445,  -445,
    -445,  -445,   141,    34,    47,   181,   373,  3764,   167,  -445,
     180,  -445,   198,   239,  3820,   251,    86,   227,   355,   205,
    3982,   208,   220,  4036,   271,  6192,    39,   301,  -445,  2857,
     304,  4086,   312,  -445,   322,  -445,   330,  -445,  4140,   367,
     126,   401,   438,   452,   472,  6192,   121,   493,   294,   289,
    -445,  -445,   496,  4190,   501,  -445,  -445,    61,   504,  -445,
     508,  4244,  -445,  -445,  -445,  5608,   510,  5398,  -445,   462,
    5752,   196,   199,  6303,   434,   515,  -445,    74,  -445,    82,
     719,   719,    50,    50,    50,    50,    50,  -445,   539,   542,
     549,  -445,  -445,  -445,  -445,  -445,  -445,  -445,  -445,   325,
    -445,  -445,  -445,  -445,   550,    71,   558,  -445,    89,   257,
     109,  5290,   557,  5608,  5608,  5608,  5608,  5608,  5608,  5608,
    5608,  5608,  5608,   559,  5614,  -445,  -445,    71,  5608,  3352,
    5608,  5608,  5608,  5608,  5608,  5608,  5608,  5608,  5608,  5608,
     560,  5608,  5608,  5608,  5608,  5608,  5608,  5608,  5608,  5608,
    5608,  5608,  5608,  5608,  5608,  5608,  -445,  -445,  5392,   563,
    -445,  -445,  -445,   559,  5608,  -445,  -445,  -445,  -445,  -445,
    -445,  -445,  -445,  -445,  -445,  -445,  -445,  -445,  5608,   559,
    3407,  -445,  -445,  -445,  -445,  -445,  -445,   567,  5608,  5608,
    -445,  -445,  -445,  -445,  -445,    65,  -445,   494,  -445,  -445,
    -445,   156,  -445,   568,    71,    35,  -445,  -445,  -445,  -445,
    -445,  -445,   585,  -445,  -445,  -445,  -445,  3462,  -445,  -445,
     566,  -445,  -445,  -445,  4294,  -445,  -445,  5957,  -445,  5500,
    5608,  -445,    71,   528,  -445,    71,   531,  5608,  5608,  -445,
    -445,  -445,  1702,  1387,  1807,   276,   572,  1492,   387,  -445,
    -445,  1912,  -445,  2857,  -445,    49,  -445,  -445,   578,   588,
     163,  -445,  5608,  5808,  -445,  4348,  4398,  4452,  4502,  4556,
    4606,  4660,  4710,  4764,  4814,  -445,   -57,  5716,  4868,   170,
    5506,  4918,  -445,  5920,  6266,  6303,   374,   374,   374,   374,
     374,   374,   374,   374,  -445,   719,   719,   719,   719,   244,
     244,   435,   254,   254,   219,   219,   219,   380,    50,    50,
    5608,  5864,  -445,   521,  6192,  5994,  -445,   603,  3876,  -445,
    4972,  6192,   604,    71,  -445,   575,   606,   605,   609,  -445,
    -445,   362,  -445,   608,  -445,   623,   624,   626,    42,  -445,
     580,  -445,  -445,  6031,  6192,    71,  5608,  -445,    71,  5608,
    -445,   374,   374,   628,  -445,   436,  3517,   625,  -445,  -445,
    -445,   629,   630,  -445,   593,   329,  -445,   627,  -445,   632,
    -445,    57,   633,  -445,    25,   634,   607,  -445,  -445,  -445,
    -445,   635,  2017,  -445,  -445,  -445,   203,  -445,  -445,  -445,
    -445,  6068,  -445,  -445,  -445,  -445,  -445,  -445,  -445,  -445,
    -445,  -445,  -445,  5608,  3627,  -445,  -445,  -445,  6192,   206,
    -445,  5608,  6105,  -445,  5608,  5608,  -445,  -445,  2857,  -445,
    -445,   608,    24,   638,  -445,   594,   577,  -445,  -445,    41,
    -445,  -445,   258,  2122,  -445,  -445,  -445,  -445,  -445,  -445,
     595,  6192,   596,  6155,  -445,   650,  -445,   654,  5022,   655,
    -445,  -445,  -445,  -445,   417,   597,  -445,  -445,  -445,    46,
    -445,  -445,   658,   418,  -445,   446,  -445,  -445,    80,  -445,
     659,   661,  -445,  -445,  -445,   559,    15,  -445,   663,  -445,
    1597,  -445,   666,    71,  -445,    71,   620,  -445,  5076,   178,
    -445,  -445,   621,  6229,  -445,  6192,  3666,   209,  -445,   266,
    -445,   536,   669,   667,   668,    33,  -445,  -445,  -445,  -445,
     671,  -445,    71,  -445,   605,  -445,  -445,  -445,  -445,   672,
    -445,  -445,   636,  -445,  -445,  -445,  5608,  -445,  -445,  -445,
    -445,  2227,  1387,  -445,  -445,   673,  -445,  -445,   615,  -445,
    -445,  2857,  -445,  -445,  -445,  -445,  -445,   432,  -445,  -445,
     679,  -445,   556,   559,  -445,  -445,   637,    71,   608,   447,
    -445,  -445,    71,  -445,  -445,  -445,  5608,   388,   444,   445,
    -445,   674,   209,  -445,  -445,  -445,  -445,   639,  5608,  -445,
     602,   682,  -445,   670,   195,   685,   541,  -445,   689,    71,
     690,  6192,  -445,  -445,  2857,  -445,  -445,  2857,  -445,  2332,
    -445,  -445,  2857,  -445,  2857,  -445,  -445,  -445,   691,   651,
     645,  -445,  -445,  -445,   647,  3932,   695,  -445,  2857,   696,
    -445,  2857,   697,  -445,  2857,   698,  -445,    71,  -445,  5126,
      35,  5608,  -445,  -445,  -445,     5,  -445,  -445,  -445,   202,
    -445,  -445,  -445,   653,  -445,   967,  -445,  1072,  -445,  1177,
    -445,  1282,  -445,  -445,  -445,   702,  -445,  -445,  -445,  -445,
    -445,  -445,  -445,  -445,  -445,  -445,  -445,  -445,  -445,   656,
    -445,  -445,  5180,  -445,   701,   541,   660,   705,  -445,  2437,
    2542,  2647,  -445,  2752,  -445,  -445,  -445,  -445,  -445,   706,
     708,   711,   716,  -445,  -445,  -445,  -445
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -445,  -445,  -445,  -445,  -445,  -445,    -1,  -445,  -445,  -445,
    -445,  -445,  -445,  -445,  -445,  -445,  -445,  -445,  -445,  -445,
    -445,  -445,  -445,  -445,  -445,  -445,  -445,  -445,  -445,  -445,
    -445,    72,  -445,  -445,  -445,  -445,  -445,  -204,  -445,  -445,
    -445,  -445,  -445,  -445,  -445,  -445,    37,  -445,  -445,  -445,
    -445,  -445,  -445,  -445,  -445,  -445,  -445,  -445,  -445,  -445,
    -445,  -445,  -445,  -445,  -445,   359,  -445,  -445,  -445,  -445,
      68,  -445,  -445,  -445,  -445,  -445,  -445,  -445,  -445,  -445,
    -445,    60,  -445,  -445,  -445,  -445,  -445,  -445,   232,  -445,
    -445,    58,  -445,  -362,  -445,  -445,  -445,  -445,  -445,  -444,
      54,  -320,  -445,  -445,  -445,  -445,  -445,  -445,  -445,  -445,
    -445,  -445,  -445,  -445,   -98,  -445,  -445,  -445,  -445,  -445,
    -445,   288,  -445,   100,  -445,  -445,   -60,  -445,  -445,   189,
    -445,   192,   197,  -445,  -445,  -445,  -445,  -445,  -445,  -445,
    -445,  -445,   293,  -445,  -319,   -10,  -445,    -2,   144,   -80,
     713,  -445,    99,  -445,  -445,  -445,  -445,  -445,  -445,  -445,
    -445,  -445,  -445,   -41,   347,   505,  -445
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -426
static const yytype_int16 yytable[] =
{
     112,    56,   124,   197,   199,   454,   230,   136,   363,   552,
     364,   793,   299,   456,   367,   137,   371,   156,   671,   157,
       6,     7,     8,     9,    10,   523,   591,   620,   592,   593,
     621,   594,   245,   122,   701,   280,     3,   282,     8,   702,
     307,   194,    21,    22,   635,   557,   245,   621,   120,   656,
     504,   121,  -234,    30,   226,   505,  -267,   245,   585,   227,
    -197,   586,   622,   587,   339,   141,   442,    40,  -277,    41,
     623,   624,   112,   211,   184,   112,   213,   360,   794,   622,
     112,   215,   219,   665,   672,   361,   280,   623,   624,    42,
      43,   657,   377,   247,   112,   223,   227,   619,  -277,  -234,
     673,   247,  -197,   283,    46,    47,   703,   308,   309,    48,
     595,   625,   381,   224,   245,   658,   443,    49,  -234,    50,
      51,    52,  -267,   182,   330,   666,  -197,   323,   625,   284,
     340,   380,   265,   266,   267,   268,   269,   270,   271,   272,
     273,   274,   275,   309,   281,   276,   277,   112,   311,   667,
     128,   309,   129,   276,   277,  -425,   152,   127,   378,   450,
     134,   153,   140,   225,   143,   183,   510,   145,   502,   151,
     289,  -425,   158,   526,   278,   279,   165,   324,   378,   173,
     626,   681,   181,   290,   285,   190,   193,   636,   145,   145,
     200,   201,   202,   203,   204,   205,   206,   351,   754,   184,
     354,   291,   184,   399,   604,   184,   184,   611,   301,   184,
       4,   303,     5,     6,     7,     8,     9,    10,  -123,    12,
      13,    14,    15,   304,    16,   451,   286,    17,   687,   688,
     689,    18,   378,   395,    20,    21,    22,    23,    24,   309,
      25,   208,   292,   209,    28,    29,    30,   309,  -401,    32,
     553,  -401,    35,  -401,   296,   210,  -401,    38,    39,   641,
      40,   184,    41,   227,   755,   352,   379,   696,   355,   184,
     247,   795,   605,   433,   306,   352,   147,   482,   148,   483,
     231,   232,    42,    43,   293,  -164,    44,    45,   154,   436,
     353,   356,   333,   155,   600,   247,   297,    46,    47,   484,
     485,   169,    48,   170,   310,   247,   167,   312,  -401,   298,
      49,   168,    50,    51,    52,   314,  -401,   273,   274,   275,
     149,  -167,   276,   277,   375,   316,   368,   605,   369,   344,
     581,   347,  -168,   318,   334,   605,   267,   268,   269,   270,
     271,   272,   273,   274,   275,   171,   400,   276,   277,   270,
     271,   272,   273,   274,   275,   278,   279,   276,   277,   178,
     112,   112,   112,   550,   179,   112,   651,   652,   320,   112,
     370,   112,   503,   320,  -168,   383,   332,   385,   386,   387,
     388,   389,   390,   391,   392,   393,   394,   761,   398,   736,
     499,   737,   401,   403,   404,   405,   406,   407,   408,   409,
     410,   411,   412,   413,   326,   415,   416,   417,   418,   419,
     420,   421,   422,   423,   424,   425,   426,   427,   428,   429,
     653,   660,   431,   453,   299,   247,   231,   232,   434,   161,
     791,   247,   500,   738,   162,   163,   586,   565,   587,   566,
     300,   327,   435,   460,   438,   739,   742,   740,   743,   663,
     732,   466,   440,   441,   469,   328,   719,   260,   261,   262,
     263,   264,   654,   661,   265,   266,   267,   268,   269,   270,
     271,   272,   273,   274,   275,   329,   761,   276,   277,   274,
     275,   458,   609,   276,   277,     8,   247,   231,   232,   741,
     744,   664,   733,   463,   464,   446,   331,  -281,   135,   335,
     112,   471,   472,     8,   338,    21,    22,   341,   130,   765,
     131,   342,   767,   345,   348,   357,   358,   769,   359,   771,
     174,   132,   175,    21,    22,   176,   511,   447,   268,   269,
     270,   271,   272,   273,   274,   275,   112,   618,   276,   277,
     152,   206,   541,   154,   528,     6,     7,   757,     9,    10,
     178,   112,   195,   374,   196,     6,     7,     8,     9,    10,
     593,   376,   594,   384,   560,     8,   414,   562,   758,   432,
     439,   452,   459,   489,   532,   490,   799,    21,    22,   800,
     467,  -164,   801,   470,   508,   670,   455,   803,    30,     6,
       7,   509,     9,    10,   574,   491,   485,   575,   112,   576,
     577,   578,    40,   534,    41,   606,   536,   540,   447,   544,
     561,   545,   548,   563,   505,   112,   690,  -167,   698,   575,
     568,   576,   577,   578,    42,    43,   554,   555,   612,   556,
     558,   564,   572,   573,   569,   584,   582,   494,   601,    46,
      47,   631,   590,   598,    48,   632,   634,   644,   645,   112,
     112,   642,    49,   647,    50,    51,    52,   648,   650,   112,
     720,   659,   668,   728,   669,   655,   674,   608,   206,   675,
     679,   683,   699,   179,   700,   613,   753,   715,   615,   616,
     705,   708,   726,   745,   751,   752,   710,   729,   756,   748,
     112,   690,   762,   764,   773,   775,   774,   776,   779,   782,
     785,   788,   676,   797,   678,   798,   802,   805,   808,   813,
     807,   814,   112,   766,   815,   112,   768,   112,   697,   816,
     112,   770,   112,   772,   712,   492,   718,   725,   599,   746,
     727,   706,   731,   543,   707,   806,   112,   781,   637,   112,
     784,   638,   112,   787,   551,   166,   639,   529,   396,     0,
       0,     0,   125,     0,     0,     6,     7,     8,     9,    10,
       0,     0,     0,   112,     0,   112,     0,   112,     0,   112,
     247,     0,     0,     0,     0,     0,   730,    21,    22,     0,
       0,   734,     0,     0,     0,     0,     0,     0,    30,     0,
     711,     0,     0,     0,     0,     0,     0,   112,   112,   112,
       0,   112,    40,     0,    41,     0,     0,     0,   763,   265,
     266,   267,   268,   269,   270,   271,   272,   273,   274,   275,
       0,     0,   276,   277,    42,    43,     0,     0,     0,     0,
     735,     0,     0,     0,     0,     0,     0,     0,     0,    46,
      47,     0,   749,     0,    48,     0,   789,     0,     0,     0,
       0,     0,    49,     0,    50,    51,    52,     0,   796,     0,
       0,     0,    -2,     4,     0,     5,     6,     7,     8,     9,
      10,    11,    12,    13,    14,    15,     0,    16,     0,     0,
      17,     0,     0,     0,    18,    19,     0,    20,    21,    22,
      23,    24,     0,    25,    26,   792,    27,    28,    29,    30,
       0,    31,    32,    33,    34,    35,    36,     0,    37,     0,
      38,    39,     0,    40,     0,    41,   231,   232,   233,   234,
     235,   236,   237,   238,   239,   240,   241,   242,     0,     0,
       0,     0,   243,     0,     0,    42,    43,     0,     0,    44,
      45,     0,     0,     0,     0,   244,     0,     0,     0,     0,
      46,    47,     0,     0,     0,    48,     0,     0,     0,     0,
       0,     0,     0,    49,     0,    50,    51,    52,     4,     0,
       5,     6,     7,     8,     9,    10,  -161,    12,    13,    14,
      15,     0,    16,     0,     0,    17,     0,     0,     0,    18,
    -161,  -161,    20,    21,    22,    23,    24,     0,    25,   208,
       0,   209,    28,    29,    30,     0,     0,    32,     0,     0,
      35,     0,  -161,   210,     0,    38,    39,     0,    40,     0,
      41,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      42,    43,     0,     0,    44,    45,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    46,    47,     0,     0,     0,
      48,     0,     0,     0,     0,     0,     0,     0,    49,     0,
      50,    51,    52,     4,     0,     5,     6,     7,     8,     9,
      10,  -157,    12,    13,    14,    15,     0,    16,     0,     0,
      17,     0,     0,     0,    18,  -157,  -157,    20,    21,    22,
      23,    24,     0,    25,   208,     0,   209,    28,    29,    30,
       0,     0,    32,     0,     0,    35,     0,  -157,   210,     0,
      38,    39,     0,    40,     0,    41,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    42,    43,     0,     0,    44,
      45,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      46,    47,     0,     0,     0,    48,     0,     0,     0,     0,
       0,     0,     0,    49,     0,    50,    51,    52,     4,     0,
       5,     6,     7,     8,     9,    10,  -192,    12,    13,    14,
      15,     0,    16,     0,     0,    17,     0,     0,     0,    18,
    -192,  -192,    20,    21,    22,    23,    24,     0,    25,   208,
       0,   209,    28,    29,    30,     0,     0,    32,     0,     0,
      35,     0,  -192,   210,     0,    38,    39,     0,    40,     0,
      41,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      42,    43,     0,     0,    44,    45,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    46,    47,     0,     0,     0,
      48,     0,     0,     0,     0,     0,     0,     0,    49,     0,
      50,    51,    52,     4,     0,     5,     6,     7,     8,     9,
      10,  -188,    12,    13,    14,    15,     0,    16,     0,     0,
      17,     0,     0,     0,    18,  -188,  -188,    20,    21,    22,
      23,    24,     0,    25,   208,     0,   209,    28,    29,    30,
       0,     0,    32,     0,     0,    35,     0,  -188,   210,     0,
      38,    39,     0,    40,     0,    41,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    42,    43,     0,     0,    44,
      45,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      46,    47,     0,     0,     0,    48,     0,     0,     0,     0,
       0,     0,     0,    49,     0,    50,    51,    52,     4,     0,
       5,     6,     7,     8,     9,    10,   -93,    12,    13,    14,
      15,     0,    16,   475,   476,    17,     0,     0,     0,    18,
       0,     0,    20,    21,    22,    23,    24,     0,    25,   208,
       0,   209,    28,    29,    30,     0,     0,    32,     0,     0,
      35,     0,     0,   210,     0,    38,    39,     0,    40,     0,
      41,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      42,    43,     0,     0,    44,    45,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    46,    47,     0,     0,     0,
      48,     0,     0,     0,     0,     0,     0,     0,    49,     0,
      50,    51,    52,     4,     0,     5,     6,     7,     8,     9,
      10,  -208,    12,    13,    14,    15,     0,    16,     0,     0,
      17,     0,     0,     0,    18,     0,     0,    20,    21,    22,
      23,    24,   494,    25,   208,     0,   209,    28,    29,    30,
       0,     0,    32,     0,     0,    35,     0,     0,   210,     0,
      38,    39,     0,    40,     0,    41,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    42,    43,     0,     0,    44,
      45,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      46,    47,     0,     0,     0,    48,     0,     0,     0,     0,
       0,     0,     0,    49,     0,    50,    51,    52,     4,     0,
       5,     6,     7,     8,     9,    10,  -212,    12,    13,    14,
      15,     0,    16,     0,     0,    17,     0,     0,     0,    18,
       0,     0,    20,    21,    22,    23,    24,  -212,    25,   208,
       0,   209,    28,    29,    30,     0,     0,    32,     0,     0,
      35,     0,     0,   210,     0,    38,    39,     0,    40,     0,
      41,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      42,    43,     0,     0,    44,    45,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    46,    47,     0,     0,     0,
      48,     0,     0,     0,     0,     0,     0,     0,    49,     0,
      50,    51,    52,     4,     0,     5,     6,     7,     8,     9,
      10,   473,    12,    13,    14,    15,     0,    16,     0,     0,
      17,     0,     0,     0,    18,     0,     0,    20,    21,    22,
      23,    24,     0,    25,   208,     0,   209,    28,    29,    30,
       0,     0,    32,     0,     0,    35,     0,     0,   210,     0,
      38,    39,     0,    40,     0,    41,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    42,    43,     0,     0,    44,
      45,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      46,    47,     0,     0,     0,    48,     0,     0,     0,     0,
       0,     0,     0,    49,     0,    50,    51,    52,     4,     0,
       5,     6,     7,     8,     9,    10,   481,    12,    13,    14,
      15,     0,    16,     0,     0,    17,     0,     0,     0,    18,
       0,     0,    20,    21,    22,    23,    24,     0,    25,   208,
       0,   209,    28,    29,    30,     0,     0,    32,     0,     0,
      35,     0,     0,   210,     0,    38,    39,     0,    40,     0,
      41,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      42,    43,     0,     0,    44,    45,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    46,    47,     0,     0,     0,
      48,     0,     0,     0,     0,     0,     0,     0,    49,     0,
      50,    51,    52,     4,     0,     5,     6,     7,     8,     9,
      10,   501,    12,    13,    14,    15,     0,    16,     0,     0,
      17,     0,     0,     0,    18,     0,     0,    20,    21,    22,
      23,    24,     0,    25,   208,     0,   209,    28,    29,    30,
       0,     0,    32,     0,     0,    35,     0,     0,   210,     0,
      38,    39,     0,    40,     0,    41,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    42,    43,     0,     0,    44,
      45,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      46,    47,     0,     0,     0,    48,     0,     0,     0,     0,
       0,     0,     0,    49,     0,    50,    51,    52,     4,     0,
       5,     6,     7,     8,     9,    10,   602,    12,    13,    14,
      15,     0,    16,     0,     0,    17,     0,     0,     0,    18,
       0,     0,    20,    21,    22,    23,    24,     0,    25,   208,
       0,   209,    28,    29,    30,     0,     0,    32,     0,     0,
      35,     0,     0,   210,     0,    38,    39,     0,    40,     0,
      41,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      42,    43,     0,     0,    44,    45,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    46,    47,     0,     0,     0,
      48,     0,     0,     0,     0,     0,     0,     0,    49,     0,
      50,    51,    52,     4,     0,     5,     6,     7,     8,     9,
      10,   643,    12,    13,    14,    15,     0,    16,     0,     0,
      17,     0,     0,     0,    18,     0,     0,    20,    21,    22,
      23,    24,     0,    25,   208,     0,   209,    28,    29,    30,
       0,     0,    32,     0,     0,    35,     0,     0,   210,     0,
      38,    39,     0,    40,     0,    41,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    42,    43,     0,     0,    44,
      45,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      46,    47,     0,     0,     0,    48,     0,     0,     0,     0,
       0,     0,     0,    49,     0,    50,    51,    52,     4,     0,
       5,     6,     7,     8,     9,    10,   -96,    12,    13,    14,
      15,     0,    16,     0,     0,    17,     0,     0,     0,    18,
       0,     0,    20,    21,    22,    23,    24,     0,    25,   208,
       0,   209,    28,    29,    30,     0,     0,    32,     0,     0,
      35,     0,     0,   210,     0,    38,    39,     0,    40,     0,
      41,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      42,    43,     0,     0,    44,    45,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    46,    47,     0,     0,     0,
      48,     0,     0,     0,     0,     0,     0,     0,    49,     0,
      50,    51,    52,     4,     0,     5,     6,     7,     8,     9,
      10,  -170,    12,    13,    14,    15,     0,    16,     0,     0,
      17,     0,     0,     0,    18,     0,     0,    20,    21,    22,
      23,    24,     0,    25,   208,     0,   209,    28,    29,    30,
       0,     0,    32,     0,     0,    35,     0,     0,   210,     0,
      38,    39,     0,    40,     0,    41,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    42,    43,     0,     0,    44,
      45,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      46,    47,     0,     0,     0,    48,     0,     0,     0,     0,
       0,     0,     0,    49,     0,    50,    51,    52,     4,     0,
       5,     6,     7,     8,     9,    10,   809,    12,    13,    14,
      15,     0,    16,     0,     0,    17,     0,     0,     0,    18,
       0,     0,    20,    21,    22,    23,    24,     0,    25,   208,
       0,   209,    28,    29,    30,     0,     0,    32,     0,     0,
      35,     0,     0,   210,     0,    38,    39,     0,    40,     0,
      41,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      42,    43,     0,     0,    44,    45,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    46,    47,     0,     0,     0,
      48,     0,     0,     0,     0,     0,     0,     0,    49,     0,
      50,    51,    52,     4,     0,     5,     6,     7,     8,     9,
      10,   810,    12,    13,    14,    15,     0,    16,     0,     0,
      17,     0,     0,     0,    18,     0,     0,    20,    21,    22,
      23,    24,     0,    25,   208,     0,   209,    28,    29,    30,
       0,     0,    32,     0,     0,    35,     0,     0,   210,     0,
      38,    39,     0,    40,     0,    41,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    42,    43,     0,     0,    44,
      45,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      46,    47,     0,     0,     0,    48,     0,     0,     0,     0,
       0,     0,     0,    49,     0,    50,    51,    52,     4,     0,
       5,     6,     7,     8,     9,    10,   811,    12,    13,    14,
      15,     0,    16,     0,     0,    17,     0,     0,     0,    18,
       0,     0,    20,    21,    22,    23,    24,     0,    25,   208,
       0,   209,    28,    29,    30,     0,     0,    32,     0,     0,
      35,     0,     0,   210,     0,    38,    39,     0,    40,     0,
      41,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      42,    43,     0,     0,    44,    45,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    46,    47,     0,     0,     0,
      48,     0,     0,     0,     0,     0,     0,     0,    49,     0,
      50,    51,    52,     4,     0,     5,     6,     7,     8,     9,
      10,   812,    12,    13,    14,    15,     0,    16,     0,     0,
      17,     0,     0,     0,    18,     0,     0,    20,    21,    22,
      23,    24,     0,    25,   208,     0,   209,    28,    29,    30,
       0,     0,    32,     0,     0,    35,     0,     0,   210,     0,
      38,    39,     0,    40,     0,    41,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    42,    43,     0,     0,    44,
      45,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      46,    47,     0,     0,     0,    48,     0,     0,     0,     0,
       0,     0,     0,    49,     0,    50,    51,    52,     4,     0,
       5,     6,     7,     8,     9,    10,     0,    12,    13,    14,
      15,     0,    16,     0,     0,    17,     0,     0,     0,    18,
       0,     0,    20,    21,    22,    23,    24,     0,    25,   208,
       0,   209,    28,    29,    30,     0,     0,    32,     0,     0,
      35,     0,     0,   210,     0,    38,    39,     0,    40,     0,
      41,     0,     0,   133,     0,     0,     6,     7,     8,     9,
      10,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      42,    43,     0,     0,    44,    45,     0,     0,    21,    22,
       0,     0,     0,     0,     0,    46,    47,     0,     0,    30,
      48,     0,     0,     0,     0,     0,     0,     0,    49,     0,
      50,    51,    52,    40,     0,    41,     0,     0,   139,     0,
       0,     6,     7,     8,     9,    10,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    42,    43,     0,     0,     0,
       0,     0,     0,    21,    22,     0,     0,     0,     0,     0,
      46,    47,     0,     0,    30,    48,     0,     0,     0,     0,
       0,     0,     0,    49,     0,    50,    51,    52,    40,     0,
      41,     0,     0,   142,     0,     0,     6,     7,     8,     9,
      10,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      42,    43,     0,     0,     0,     0,     0,     0,    21,    22,
       0,     0,     0,     0,     0,    46,    47,     0,     0,    30,
      48,     0,     0,     0,     0,     0,     0,     0,    49,     0,
      50,    51,    52,    40,     0,    41,     0,     0,   144,     0,
       0,     6,     7,     8,     9,    10,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    42,    43,     0,     0,     0,
       0,     0,     0,    21,    22,     0,     0,     0,     0,     0,
      46,    47,     0,     0,    30,    48,     0,     0,     0,     0,
       0,     0,     0,    49,     0,    50,    51,    52,    40,     0,
      41,     0,     0,   150,     0,     0,     6,     7,     8,     9,
      10,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      42,    43,     0,     0,     0,     0,     0,     0,    21,    22,
       0,     0,     0,     0,     0,    46,    47,     0,     0,    30,
      48,     0,     0,     0,     0,     0,     0,     0,    49,     0,
      50,    51,    52,    40,     0,    41,     0,     0,   164,     0,
       0,     6,     7,     8,     9,    10,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    42,    43,     0,     0,     0,
       0,     0,     0,    21,    22,     0,     0,     0,     0,     0,
      46,    47,     0,     0,    30,    48,     0,     0,     0,     0,
       0,     0,     0,    49,     0,    50,    51,    52,    40,     0,
      41,     0,     0,   172,     0,     0,     6,     7,     8,     9,
      10,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      42,    43,     0,     0,     0,     0,     0,     0,    21,    22,
       0,     0,     0,     0,     0,    46,    47,     0,     0,    30,
      48,     0,     0,     0,     0,     0,     0,     0,    49,     0,
      50,    51,    52,    40,     0,    41,     0,     0,   180,     0,
       0,     6,     7,     8,     9,    10,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    42,    43,     0,     0,     0,
       0,     0,     0,    21,    22,     0,     0,     0,     0,     0,
      46,    47,     0,     0,    30,    48,     0,     0,     0,     0,
       0,     0,     0,    49,     0,    50,    51,    52,    40,     0,
      41,     0,     0,   402,     0,     0,     6,     7,     8,     9,
      10,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      42,    43,     0,     0,     0,     0,     0,     0,    21,    22,
       0,     0,     0,     0,     0,    46,    47,     0,     0,    30,
      48,     0,     0,     0,     0,     0,     0,     0,    49,     0,
      50,    51,    52,    40,     0,    41,     0,     0,   437,     0,
       0,     6,     7,     8,     9,    10,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    42,    43,     0,     0,     0,
       0,     0,     0,    21,    22,     0,     0,     0,     0,     0,
      46,    47,     0,     0,    30,    48,     0,     0,     0,     0,
       0,     0,     0,    49,     0,    50,    51,    52,    40,     0,
      41,     0,     0,   457,     0,     0,     6,     7,     8,     9,
      10,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      42,    43,     0,     0,     0,     0,     0,     0,    21,    22,
       0,     0,     0,     0,     0,    46,    47,     0,     0,    30,
      48,     0,     0,     0,     0,     0,     0,     0,    49,     0,
      50,    51,    52,    40,     0,    41,     0,     0,   567,     0,
       0,     6,     7,     8,     9,    10,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    42,    43,     0,     0,     0,
       0,     0,     0,    21,    22,     0,     0,     0,     0,     0,
      46,    47,     0,     0,    30,    48,     0,     0,     0,     0,
       0,     0,     0,    49,     0,    50,    51,    52,    40,     0,
      41,     0,     0,     0,     0,   198,     6,     7,     8,     9,
      10,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      42,    43,     0,     0,     0,     0,     0,     0,    21,    22,
       0,     0,     0,     0,     0,    46,    47,     0,     0,    30,
      48,     0,     0,     0,     0,     0,     0,     0,    49,     0,
      50,    51,    52,    40,     0,    41,     0,     0,     0,     0,
     610,     6,     7,     8,     9,    10,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    42,    43,     0,     0,     0,
       0,     0,     0,    21,    22,     0,     0,     0,     0,     0,
      46,    47,     0,     0,    30,    48,     0,     0,     0,   684,
       0,     0,     0,    49,     0,    50,    51,    52,    40,     0,
      41,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      42,    43,     0,     0,     0,     0,   246,     0,     0,     0,
       0,   685,     0,     0,     0,    46,    47,   247,     0,     0,
      48,     0,     0,     0,     0,     0,     0,     0,    49,     0,
      50,    51,    52,   686,     0,     0,   249,   250,   251,     0,
       0,   252,   253,   254,   255,   256,   257,   258,   259,   260,
     261,   262,   263,   264,   247,     0,   265,   266,   267,   268,
     269,   270,   271,   272,   273,   274,   275,   287,   248,   276,
     277,     0,     0,   249,   250,   251,     0,     0,   252,   253,
     254,   255,   256,   257,   258,   259,   260,   261,   262,   263,
     264,     0,     0,   265,   266,   267,   268,   269,   270,   271,
     272,   273,   274,   275,     0,     0,   276,   277,     0,   288,
       0,     0,     0,     0,     0,   247,     0,     0,     0,     0,
       0,     0,     0,   294,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   249,   250,   251,     0,     0,   252,
     253,   254,   255,   256,   257,   258,   259,   260,   261,   262,
     263,   264,     0,     0,   265,   266,   267,   268,   269,   270,
     271,   272,   273,   274,   275,   295,     0,   276,   277,     0,
       0,   247,     0,     0,     0,     0,     0,     0,     0,   537,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     249,   250,   251,     0,     0,   252,   253,   254,   255,   256,
     257,   258,   259,   260,   261,   262,   263,   264,     0,     0,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   538,     0,   276,   277,     0,     0,   247,     0,     0,
       0,     0,     0,     0,     0,   777,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   249,   250,   251,     0,
       0,   252,   253,   254,   255,   256,   257,   258,   259,   260,
     261,   262,   263,   264,     0,     0,   265,   266,   267,   268,
     269,   270,   271,   272,   273,   274,   275,   778,     0,   276,
     277,     0,     0,   247,     0,   302,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   249,   250,   251,     0,     0,   252,   253,   254,
     255,   256,   257,   258,   259,   260,   261,   262,   263,   264,
       0,     0,   265,   266,   267,   268,   269,   270,   271,   272,
     273,   274,   275,   247,     0,   276,   277,     0,     0,   305,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   249,   250,   251,     0,     0,   252,   253,   254,
     255,   256,   257,   258,   259,   260,   261,   262,   263,   264,
       0,     0,   265,   266,   267,   268,   269,   270,   271,   272,
     273,   274,   275,     0,     0,   276,   277,   247,     0,   313,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   249,   250,   251,     0,
       0,   252,   253,   254,   255,   256,   257,   258,   259,   260,
     261,   262,   263,   264,     0,     0,   265,   266,   267,   268,
     269,   270,   271,   272,   273,   274,   275,   247,     0,   276,
     277,     0,     0,   319,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   249,   250,   251,     0,
       0,   252,   253,   254,   255,   256,   257,   258,   259,   260,
     261,   262,   263,   264,     0,     0,   265,   266,   267,   268,
     269,   270,   271,   272,   273,   274,   275,     0,     0,   276,
     277,   247,     0,   336,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     249,   250,   251,     0,     0,   252,   253,   254,   255,   256,
     257,   258,   259,   260,   261,   262,   263,   264,     0,     0,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   247,     0,   276,   277,     0,     0,   343,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     249,   250,   251,     0,     0,   252,   253,   254,   255,   256,
     257,   258,   259,   260,   261,   337,   263,   264,     0,     0,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,     0,     0,   276,   277,   247,     0,   184,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   249,   250,   251,     0,     0,   252,
     253,   254,   255,   256,   257,   258,   259,   260,   261,   262,
     263,   264,     0,     0,   265,   266,   267,   268,   269,   270,
     271,   272,   273,   274,   275,   247,     0,   276,   277,     0,
       0,   513,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   249,   250,   251,     0,     0,   252,
     253,   254,   255,   256,   257,   258,   259,   260,   261,   262,
     263,   264,     0,     0,   265,   266,   267,   268,   269,   270,
     271,   272,   273,   274,   275,     0,     0,   276,   277,   247,
       0,   514,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   249,   250,
     251,     0,     0,   252,   253,   254,   255,   256,   257,   258,
     259,   260,   261,   262,   263,   264,     0,     0,   265,   266,
     267,   268,   269,   270,   271,   272,   273,   274,   275,   247,
       0,   276,   277,     0,     0,   515,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   249,   250,
     251,     0,     0,   252,   253,   254,   255,   256,   257,   258,
     259,   260,   261,   262,   263,   264,     0,     0,   265,   266,
     267,   268,   269,   270,   271,   272,   273,   274,   275,     0,
       0,   276,   277,   247,     0,   516,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   249,   250,   251,     0,     0,   252,   253,   254,
     255,   256,   257,   258,   259,   260,   261,   262,   263,   264,
       0,     0,   265,   266,   267,   268,   269,   270,   271,   272,
     273,   274,   275,   247,     0,   276,   277,     0,     0,   517,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   249,   250,   251,     0,     0,   252,   253,   254,
     255,   256,   257,   258,   259,   260,   261,   262,   263,   264,
       0,     0,   265,   266,   267,   268,   269,   270,   271,   272,
     273,   274,   275,     0,     0,   276,   277,   247,     0,   518,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   249,   250,   251,     0,
       0,   252,   253,   254,   255,   256,   257,   258,   259,   260,
     261,   262,   263,   264,     0,     0,   265,   266,   267,   268,
     269,   270,   271,   272,   273,   274,   275,   247,     0,   276,
     277,     0,     0,   519,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   249,   250,   251,     0,
       0,   252,   253,   254,   255,   256,   257,   258,   259,   260,
     261,   262,   263,   264,     0,     0,   265,   266,   267,   268,
     269,   270,   271,   272,   273,   274,   275,     0,     0,   276,
     277,   247,     0,   520,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     249,   250,   251,     0,     0,   252,   253,   254,   255,   256,
     257,   258,   259,   260,   261,   262,   263,   264,     0,     0,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   247,     0,   276,   277,     0,     0,   521,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     249,   250,   251,     0,     0,   252,   253,   254,   255,   256,
     257,   258,   259,   260,   261,   262,   263,   264,     0,     0,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,     0,     0,   276,   277,   247,     0,   522,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   249,   250,   251,     0,     0,   252,
     253,   254,   255,   256,   257,   258,   259,   260,   261,   262,
     263,   264,     0,     0,   265,   266,   267,   268,   269,   270,
     271,   272,   273,   274,   275,   247,     0,   276,   277,     0,
       0,   525,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   249,   250,   251,     0,     0,   252,
     253,   254,   255,   256,   257,   258,   259,   260,   261,   262,
     263,   264,     0,     0,   265,   266,   267,   268,   269,   270,
     271,   272,   273,   274,   275,     0,     0,   276,   277,   247,
       0,   530,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   249,   250,
     251,     0,     0,   252,   253,   254,   255,   256,   257,   258,
     259,   260,   261,   262,   263,   264,     0,     0,   265,   266,
     267,   268,   269,   270,   271,   272,   273,   274,   275,   247,
       0,   276,   277,     0,     0,   539,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   249,   250,
     251,     0,     0,   252,   253,   254,   255,   256,   257,   258,
     259,   260,   261,   262,   263,   264,     0,     0,   265,   266,
     267,   268,   269,   270,   271,   272,   273,   274,   275,     0,
       0,   276,   277,   247,     0,   649,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   249,   250,   251,     0,     0,   252,   253,   254,
     255,   256,   257,   258,   259,   260,   261,   262,   263,   264,
       0,     0,   265,   266,   267,   268,   269,   270,   271,   272,
     273,   274,   275,   247,     0,   276,   277,     0,     0,   680,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   249,   250,   251,     0,     0,   252,   253,   254,
     255,   256,   257,   258,   259,   260,   261,   262,   263,   264,
       0,     0,   265,   266,   267,   268,   269,   270,   271,   272,
     273,   274,   275,     0,     0,   276,   277,   247,     0,   790,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   249,   250,   251,     0,
       0,   252,   253,   254,   255,   256,   257,   258,   259,   260,
     261,   262,   263,   264,     0,     0,   265,   266,   267,   268,
     269,   270,   271,   272,   273,   274,   275,   247,     0,   276,
     277,     0,     0,   804,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   249,   250,   251,     0,
       0,   252,   253,   254,   255,   256,   257,   258,   259,   260,
     261,   262,   263,   264,     0,     0,   265,   266,   267,   268,
     269,   270,   271,   272,   273,   274,   275,     0,     0,   276,
     277,   247,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     249,   250,   251,     0,     0,   252,   253,   254,   255,   256,
     257,   258,   259,   260,   261,   262,   263,   264,     0,     0,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,     0,     0,   276,   277,     6,     7,     8,     9,    10,
       0,     0,     0,     0,     6,     7,     8,     9,    10,     0,
       0,     0,     0,     0,     0,     0,     0,    21,    22,     0,
       0,     0,     0,     0,     0,     0,    21,    22,    30,     0,
       0,     0,     0,     0,     0,     0,   187,    30,     0,     0,
       0,     0,    40,   188,    41,   187,     0,     0,     0,     0,
       0,    40,     0,    41,     0,     0,     0,   189,     0,     0,
       0,     0,     0,     0,    42,    43,     0,     0,     0,     0,
       0,     0,     0,    42,    43,     0,     0,     0,     0,    46,
      47,     0,     0,     0,    48,     0,     0,     0,    46,    47,
       0,     0,    49,    48,    50,    51,    52,   382,     0,     0,
       0,    49,     0,    50,    51,    52,     6,     7,     8,     9,
      10,     0,     6,     7,     8,     9,    10,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    21,    22,
       0,     0,     0,     0,    21,    22,     0,     0,     0,    30,
       0,     0,     0,     0,     0,    30,     0,   187,     0,     0,
       0,     0,     0,    40,     0,    41,     0,     0,     0,    40,
     346,    41,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    42,    43,     0,     0,     0,
       0,    42,    43,     0,     0,     0,     0,     0,     0,     0,
      46,    47,     0,     0,     0,    48,    46,    47,     0,   430,
       0,    48,     0,    49,     0,    50,    51,    52,     0,    49,
       0,    50,    51,    52,     6,     7,     8,     9,    10,     0,
       6,     7,     8,     9,    10,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    21,    22,     0,     0,
       0,     0,    21,    22,     0,     0,     0,    30,     0,     0,
       0,     0,     0,    30,     0,     0,     0,     0,     0,     0,
       0,    40,   462,    41,     0,     0,   527,    40,     0,    41,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    42,    43,     0,     0,     0,     0,    42,
      43,     0,     0,     0,     0,     0,     0,     0,    46,    47,
       0,     0,     0,    48,    46,    47,     0,     0,     0,    48,
       0,    49,     0,    50,    51,    52,     0,    49,     0,    50,
      51,    52,     6,     7,     8,     9,    10,     0,     6,     7,
       8,     9,    10,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    21,    22,     0,     0,     0,     0,
      21,    22,     0,     0,     0,    30,     0,     0,     0,     0,
       0,    30,     0,     0,     0,     0,     0,     0,     0,    40,
       0,    41,     0,     0,     0,    40,     0,    41,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    42,    43,     0,     0,     0,     0,    42,    43,     0,
       0,     0,     0,     0,     0,     0,    46,    47,     0,     0,
       0,    48,    46,    47,     0,     0,     0,    48,     0,    49,
       0,    50,    51,    52,     0,    49,     0,    50,    51,   397,
       6,     7,     8,     9,    10,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    21,    22,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    30,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    40,     0,    41,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    42,
      43,     0,     0,     0,     0,     0,     0,   349,     0,     0,
       0,     0,     0,   247,    46,    47,     0,     0,     0,    48,
       0,     0,     0,     0,     0,     0,     0,    49,   350,    50,
      51,   524,   249,   250,   251,     0,     0,   252,   253,   254,
     255,   256,   257,   258,   259,   260,   261,   262,   263,   264,
       0,     0,   265,   266,   267,   268,   269,   270,   271,   272,
     273,   274,   275,   349,     0,   276,   277,     0,     0,   247,
     512,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   249,   250,
     251,     0,     0,   252,   253,   254,   255,   256,   257,   258,
     259,   260,   261,   262,   263,   264,     0,     0,   265,   266,
     267,   268,   269,   270,   271,   272,   273,   274,   275,   349,
       0,   276,   277,     0,     0,   247,   533,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   249,   250,   251,     0,     0,   252,
     253,   254,   255,   256,   257,   258,   259,   260,   261,   262,
     263,   264,     0,     0,   265,   266,   267,   268,   269,   270,
     271,   272,   273,   274,   275,   531,     0,   276,   277,     0,
       0,   247,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     249,   250,   251,     0,     0,   252,   253,   254,   255,   256,
     257,   258,   259,   260,   261,   262,   263,   264,   247,   461,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,     0,     0,   276,   277,     0,     0,   249,   250,   251,
       0,     0,   252,   253,   254,   255,   256,   257,   258,   259,
     260,   261,   262,   263,   264,   247,     0,   265,   266,   267,
     268,   269,   270,   271,   272,   273,   274,   275,     0,     0,
     276,   277,   535,     0,   249,   250,   251,     0,     0,   252,
     253,   254,   255,   256,   257,   258,   259,   260,   261,   262,
     263,   264,   247,   559,   265,   266,   267,   268,   269,   270,
     271,   272,   273,   274,   275,     0,     0,   276,   277,     0,
       0,   249,   250,   251,     0,     0,   252,   253,   254,   255,
     256,   257,   258,   259,   260,   261,   262,   263,   264,   247,
     607,   265,   266,   267,   268,   269,   270,   271,   272,   273,
     274,   275,     0,     0,   276,   277,     0,     0,   249,   250,
     251,     0,     0,   252,   253,   254,   255,   256,   257,   258,
     259,   260,   261,   262,   263,   264,   247,   614,   265,   266,
     267,   268,   269,   270,   271,   272,   273,   274,   275,     0,
       0,   276,   277,     0,     0,   249,   250,   251,     0,     0,
     252,   253,   254,   255,   256,   257,   258,   259,   260,   261,
     262,   263,   264,     0,     0,   265,   266,   267,   268,   269,
     270,   271,   272,   273,   274,   275,   247,     0,   276,   277,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   646,     0,     0,     0,   249,   250,   251,     0,     0,
     252,   253,   254,   255,   256,   257,   258,   259,   260,   261,
     262,   263,   264,   247,     0,   265,   266,   267,   268,   269,
     270,   271,   272,   273,   274,   275,     0,     0,   276,   277,
       0,     0,   249,   250,   251,     0,     0,   252,   253,   254,
     255,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     247,     0,   265,   266,   267,   268,   269,   270,   271,   272,
     273,   274,   275,     0,     0,   276,   277,     0,     0,     0,
     250,   251,     0,     0,   252,   253,   254,   255,   256,   257,
     258,   259,   260,   261,   262,   263,   264,   247,     0,   265,
     266,   267,   268,   269,   270,   271,   272,   273,   274,   275,
       0,     0,   276,   277,     0,     0,     0,     0,   251,     0,
       0,   252,   253,   254,   255,   256,   257,   258,   259,   260,
     261,   262,   263,   264,   247,     0,   265,   266,   267,   268,
     269,   270,   271,   272,   273,   274,   275,     0,     0,   276,
     277,     0,     0,     0,     0,     0,     0,     0,   252,   253,
     254,   255,   256,   257,   258,   259,   260,   261,   262,   263,
     264,     0,     0,   265,   266,   267,   268,   269,   270,   271,
     272,   273,   274,   275,     0,     0,   276,   277
};

static const yytype_int16 yycheck[] =
{
       2,     2,    12,    44,    45,   325,   104,    17,   212,   453,
     214,     6,    69,   332,   218,    17,   220,     1,     3,     3,
       4,     5,     6,     7,     8,    82,     1,     3,     3,     4,
       6,     6,   112,     1,     1,   115,     0,     3,     6,     6,
       1,    43,    26,    27,     3,     3,   126,     6,     3,     3,
       1,     3,     3,    37,     1,     6,     3,   137,     1,     6,
       3,     4,    38,     6,     3,     1,     1,    51,     3,    53,
      46,    47,    74,    74,     3,    77,    77,     3,    73,    38,
      82,    82,    47,     3,    69,     3,   166,    46,    47,    73,
      74,    45,     3,    51,    96,    96,     6,   541,    33,    50,
      85,    51,    45,    69,    88,    89,    73,    68,    69,    93,
      85,    87,     3,     1,   194,    69,    51,   101,    69,   103,
     104,   105,    69,     3,     3,    45,    69,     1,    87,    82,
      69,   229,    90,    91,    92,    93,    94,    95,    96,    97,
      98,    99,   100,    69,     3,   103,   104,   149,   149,    69,
       1,    69,     3,   103,   104,    69,     1,    13,    69,     3,
      16,     6,    18,    51,    20,    45,     3,    23,   372,    25,
       3,    85,    28,     3,    53,    54,    32,    51,    69,    35,
     542,     3,    38,     3,     3,    41,    42,   549,    44,    45,
      46,    47,    48,    49,    50,    51,    52,     1,     3,     3,
       1,     3,     3,   244,     1,     3,     3,     1,     3,     3,
       1,     3,     3,     4,     5,     6,     7,     8,     9,    10,
      11,    12,    13,     3,    15,    69,    45,    18,    19,    20,
      21,    22,    69,   243,    25,    26,    27,    28,    29,    69,
      31,    32,     3,    34,    35,    36,    37,    69,    52,    40,
     454,    52,    43,    50,     3,    46,    50,    48,    49,     1,
      51,     3,    53,     6,    69,    69,     9,     1,    69,     3,
      51,    69,    69,   283,     3,    69,     1,     1,     3,     3,
      53,    54,    73,    74,    45,     9,    77,    78,     1,   299,
     191,   192,     3,     6,   498,    51,    45,    88,    89,    23,
      24,     1,    93,     3,     3,    51,     1,     3,    50,    82,
     101,     6,   103,   104,   105,     3,    50,    98,    99,   100,
      45,    45,   103,   104,   225,     3,     1,    69,     3,   185,
       1,   187,     3,     3,    45,    69,    92,    93,    94,    95,
      96,    97,    98,    99,   100,    45,   247,   103,   104,    95,
      96,    97,    98,    99,   100,    53,    54,   103,   104,     1,
     362,   363,   364,     1,     6,   367,   570,   571,     6,   371,
      45,   373,   373,     6,    45,   231,    82,   233,   234,   235,
     236,   237,   238,   239,   240,   241,   242,   706,   244,     1,
       3,     3,   248,   249,   250,   251,   252,   253,   254,   255,
     256,   257,   258,   259,     3,   261,   262,   263,   264,   265,
     266,   267,   268,   269,   270,   271,   272,   273,   274,   275,
       3,     3,   278,   324,    69,    51,    53,    54,   284,     1,
     750,    51,    45,    45,     6,     7,     4,     1,     6,     3,
      85,     3,   298,   344,   300,     1,     1,     3,     3,     3,
       3,   352,   308,   309,   355,     3,   660,    83,    84,    85,
      86,    87,    45,    45,    90,    91,    92,    93,    94,    95,
      96,    97,    98,    99,   100,     3,   795,   103,   104,    99,
     100,   337,   523,   103,   104,     6,    51,    53,    54,    45,
      45,    45,    45,   349,   350,     1,     3,     3,     1,     3,
     502,   357,   358,     6,     3,    26,    27,     3,     1,   713,
       3,     3,   716,     3,    52,    81,    82,   721,     3,   723,
       1,    14,     3,    26,    27,     6,   382,    33,    93,    94,
      95,    96,    97,    98,    99,   100,   538,   538,   103,   104,
       1,   397,   443,     1,   400,     4,     5,     6,     7,     8,
       1,   553,     1,     3,     3,     4,     5,     6,     7,     8,
       4,     3,     6,     6,   465,     6,     6,   468,    27,     6,
       3,     3,     6,     1,   430,     3,   780,    26,    27,   783,
      52,     9,   786,    52,     6,   595,     1,   791,    37,     4,
       5,     3,     7,     8,     1,    23,    24,     4,   600,     6,
       7,     8,    51,    82,    53,   506,     3,     3,    33,     3,
     466,     6,     3,   469,     6,   617,   617,    45,    82,     4,
     476,     6,     7,     8,    73,    74,     3,     3,   529,     3,
      50,     3,     3,     3,     9,     3,     9,    30,     3,    88,
      89,     3,     9,     9,    93,    51,    69,    52,    52,   651,
     652,   552,   101,     3,   103,   104,   105,     3,     3,   661,
     661,     3,     3,   673,     3,    68,     3,   523,   524,     3,
      50,    50,     3,     6,     6,   531,     6,     4,   534,   535,
       9,     9,     3,     9,    82,     3,    50,    50,     3,    50,
     692,   692,     3,     3,     3,    50,    45,    50,     3,     3,
       3,     3,   603,    50,   605,     3,    50,     6,     3,     3,
      50,     3,   714,   714,     3,   717,   717,   719,   619,     3,
     722,   722,   724,   724,   652,   366,   658,   667,   496,   692,
     672,   632,   678,   445,   634,   795,   738,   738,   549,   741,
     741,   549,   744,   744,   451,    32,   549,   400,   243,    -1,
      -1,    -1,     1,    -1,    -1,     4,     5,     6,     7,     8,
      -1,    -1,    -1,   765,    -1,   767,    -1,   769,    -1,   771,
      51,    -1,    -1,    -1,    -1,    -1,   677,    26,    27,    -1,
      -1,   682,    -1,    -1,    -1,    -1,    -1,    -1,    37,    -1,
     646,    -1,    -1,    -1,    -1,    -1,    -1,   799,   800,   801,
      -1,   803,    51,    -1,    53,    -1,    -1,    -1,   709,    90,
      91,    92,    93,    94,    95,    96,    97,    98,    99,   100,
      -1,    -1,   103,   104,    73,    74,    -1,    -1,    -1,    -1,
     686,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    88,
      89,    -1,   698,    -1,    93,    -1,   747,    -1,    -1,    -1,
      -1,    -1,   101,    -1,   103,   104,   105,    -1,   759,    -1,
      -1,    -1,     0,     1,    -1,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    13,    -1,    15,    -1,    -1,
      18,    -1,    -1,    -1,    22,    23,    -1,    25,    26,    27,
      28,    29,    -1,    31,    32,   751,    34,    35,    36,    37,
      -1,    39,    40,    41,    42,    43,    44,    -1,    46,    -1,
      48,    49,    -1,    51,    -1,    53,    53,    54,    55,    56,
      57,    58,    59,    60,    61,    62,    63,    64,    -1,    -1,
      -1,    -1,    69,    -1,    -1,    73,    74,    -1,    -1,    77,
      78,    -1,    -1,    -1,    -1,    82,    -1,    -1,    -1,    -1,
      88,    89,    -1,    -1,    -1,    93,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,   101,    -1,   103,   104,   105,     1,    -1,
       3,     4,     5,     6,     7,     8,     9,    10,    11,    12,
      13,    -1,    15,    -1,    -1,    18,    -1,    -1,    -1,    22,
      23,    24,    25,    26,    27,    28,    29,    -1,    31,    32,
      -1,    34,    35,    36,    37,    -1,    -1,    40,    -1,    -1,
      43,    -1,    45,    46,    -1,    48,    49,    -1,    51,    -1,
      53,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      73,    74,    -1,    -1,    77,    78,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    88,    89,    -1,    -1,    -1,
      93,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   101,    -1,
     103,   104,   105,     1,    -1,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    13,    -1,    15,    -1,    -1,
      18,    -1,    -1,    -1,    22,    23,    24,    25,    26,    27,
      28,    29,    -1,    31,    32,    -1,    34,    35,    36,    37,
      -1,    -1,    40,    -1,    -1,    43,    -1,    45,    46,    -1,
      48,    49,    -1,    51,    -1,    53,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    73,    74,    -1,    -1,    77,
      78,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      88,    89,    -1,    -1,    -1,    93,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,   101,    -1,   103,   104,   105,     1,    -1,
       3,     4,     5,     6,     7,     8,     9,    10,    11,    12,
      13,    -1,    15,    -1,    -1,    18,    -1,    -1,    -1,    22,
      23,    24,    25,    26,    27,    28,    29,    -1,    31,    32,
      -1,    34,    35,    36,    37,    -1,    -1,    40,    -1,    -1,
      43,    -1,    45,    46,    -1,    48,    49,    -1,    51,    -1,
      53,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      73,    74,    -1,    -1,    77,    78,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    88,    89,    -1,    -1,    -1,
      93,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   101,    -1,
     103,   104,   105,     1,    -1,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    13,    -1,    15,    -1,    -1,
      18,    -1,    -1,    -1,    22,    23,    24,    25,    26,    27,
      28,    29,    -1,    31,    32,    -1,    34,    35,    36,    37,
      -1,    -1,    40,    -1,    -1,    43,    -1,    45,    46,    -1,
      48,    49,    -1,    51,    -1,    53,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    73,    74,    -1,    -1,    77,
      78,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      88,    89,    -1,    -1,    -1,    93,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,   101,    -1,   103,   104,   105,     1,    -1,
       3,     4,     5,     6,     7,     8,     9,    10,    11,    12,
      13,    -1,    15,    16,    17,    18,    -1,    -1,    -1,    22,
      -1,    -1,    25,    26,    27,    28,    29,    -1,    31,    32,
      -1,    34,    35,    36,    37,    -1,    -1,    40,    -1,    -1,
      43,    -1,    -1,    46,    -1,    48,    49,    -1,    51,    -1,
      53,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      73,    74,    -1,    -1,    77,    78,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    88,    89,    -1,    -1,    -1,
      93,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   101,    -1,
     103,   104,   105,     1,    -1,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    13,    -1,    15,    -1,    -1,
      18,    -1,    -1,    -1,    22,    -1,    -1,    25,    26,    27,
      28,    29,    30,    31,    32,    -1,    34,    35,    36,    37,
      -1,    -1,    40,    -1,    -1,    43,    -1,    -1,    46,    -1,
      48,    49,    -1,    51,    -1,    53,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    73,    74,    -1,    -1,    77,
      78,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      88,    89,    -1,    -1,    -1,    93,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,   101,    -1,   103,   104,   105,     1,    -1,
       3,     4,     5,     6,     7,     8,     9,    10,    11,    12,
      13,    -1,    15,    -1,    -1,    18,    -1,    -1,    -1,    22,
      -1,    -1,    25,    26,    27,    28,    29,    30,    31,    32,
      -1,    34,    35,    36,    37,    -1,    -1,    40,    -1,    -1,
      43,    -1,    -1,    46,    -1,    48,    49,    -1,    51,    -1,
      53,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      73,    74,    -1,    -1,    77,    78,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    88,    89,    -1,    -1,    -1,
      93,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   101,    -1,
     103,   104,   105,     1,    -1,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    13,    -1,    15,    -1,    -1,
      18,    -1,    -1,    -1,    22,    -1,    -1,    25,    26,    27,
      28,    29,    -1,    31,    32,    -1,    34,    35,    36,    37,
      -1,    -1,    40,    -1,    -1,    43,    -1,    -1,    46,    -1,
      48,    49,    -1,    51,    -1,    53,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    73,    74,    -1,    -1,    77,
      78,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      88,    89,    -1,    -1,    -1,    93,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,   101,    -1,   103,   104,   105,     1,    -1,
       3,     4,     5,     6,     7,     8,     9,    10,    11,    12,
      13,    -1,    15,    -1,    -1,    18,    -1,    -1,    -1,    22,
      -1,    -1,    25,    26,    27,    28,    29,    -1,    31,    32,
      -1,    34,    35,    36,    37,    -1,    -1,    40,    -1,    -1,
      43,    -1,    -1,    46,    -1,    48,    49,    -1,    51,    -1,
      53,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      73,    74,    -1,    -1,    77,    78,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    88,    89,    -1,    -1,    -1,
      93,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   101,    -1,
     103,   104,   105,     1,    -1,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    13,    -1,    15,    -1,    -1,
      18,    -1,    -1,    -1,    22,    -1,    -1,    25,    26,    27,
      28,    29,    -1,    31,    32,    -1,    34,    35,    36,    37,
      -1,    -1,    40,    -1,    -1,    43,    -1,    -1,    46,    -1,
      48,    49,    -1,    51,    -1,    53,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    73,    74,    -1,    -1,    77,
      78,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      88,    89,    -1,    -1,    -1,    93,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,   101,    -1,   103,   104,   105,     1,    -1,
       3,     4,     5,     6,     7,     8,     9,    10,    11,    12,
      13,    -1,    15,    -1,    -1,    18,    -1,    -1,    -1,    22,
      -1,    -1,    25,    26,    27,    28,    29,    -1,    31,    32,
      -1,    34,    35,    36,    37,    -1,    -1,    40,    -1,    -1,
      43,    -1,    -1,    46,    -1,    48,    49,    -1,    51,    -1,
      53,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      73,    74,    -1,    -1,    77,    78,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    88,    89,    -1,    -1,    -1,
      93,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   101,    -1,
     103,   104,   105,     1,    -1,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    13,    -1,    15,    -1,    -1,
      18,    -1,    -1,    -1,    22,    -1,    -1,    25,    26,    27,
      28,    29,    -1,    31,    32,    -1,    34,    35,    36,    37,
      -1,    -1,    40,    -1,    -1,    43,    -1,    -1,    46,    -1,
      48,    49,    -1,    51,    -1,    53,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    73,    74,    -1,    -1,    77,
      78,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      88,    89,    -1,    -1,    -1,    93,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,   101,    -1,   103,   104,   105,     1,    -1,
       3,     4,     5,     6,     7,     8,     9,    10,    11,    12,
      13,    -1,    15,    -1,    -1,    18,    -1,    -1,    -1,    22,
      -1,    -1,    25,    26,    27,    28,    29,    -1,    31,    32,
      -1,    34,    35,    36,    37,    -1,    -1,    40,    -1,    -1,
      43,    -1,    -1,    46,    -1,    48,    49,    -1,    51,    -1,
      53,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      73,    74,    -1,    -1,    77,    78,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    88,    89,    -1,    -1,    -1,
      93,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   101,    -1,
     103,   104,   105,     1,    -1,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    13,    -1,    15,    -1,    -1,
      18,    -1,    -1,    -1,    22,    -1,    -1,    25,    26,    27,
      28,    29,    -1,    31,    32,    -1,    34,    35,    36,    37,
      -1,    -1,    40,    -1,    -1,    43,    -1,    -1,    46,    -1,
      48,    49,    -1,    51,    -1,    53,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    73,    74,    -1,    -1,    77,
      78,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      88,    89,    -1,    -1,    -1,    93,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,   101,    -1,   103,   104,   105,     1,    -1,
       3,     4,     5,     6,     7,     8,     9,    10,    11,    12,
      13,    -1,    15,    -1,    -1,    18,    -1,    -1,    -1,    22,
      -1,    -1,    25,    26,    27,    28,    29,    -1,    31,    32,
      -1,    34,    35,    36,    37,    -1,    -1,    40,    -1,    -1,
      43,    -1,    -1,    46,    -1,    48,    49,    -1,    51,    -1,
      53,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      73,    74,    -1,    -1,    77,    78,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    88,    89,    -1,    -1,    -1,
      93,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   101,    -1,
     103,   104,   105,     1,    -1,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    13,    -1,    15,    -1,    -1,
      18,    -1,    -1,    -1,    22,    -1,    -1,    25,    26,    27,
      28,    29,    -1,    31,    32,    -1,    34,    35,    36,    37,
      -1,    -1,    40,    -1,    -1,    43,    -1,    -1,    46,    -1,
      48,    49,    -1,    51,    -1,    53,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    73,    74,    -1,    -1,    77,
      78,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      88,    89,    -1,    -1,    -1,    93,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,   101,    -1,   103,   104,   105,     1,    -1,
       3,     4,     5,     6,     7,     8,     9,    10,    11,    12,
      13,    -1,    15,    -1,    -1,    18,    -1,    -1,    -1,    22,
      -1,    -1,    25,    26,    27,    28,    29,    -1,    31,    32,
      -1,    34,    35,    36,    37,    -1,    -1,    40,    -1,    -1,
      43,    -1,    -1,    46,    -1,    48,    49,    -1,    51,    -1,
      53,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      73,    74,    -1,    -1,    77,    78,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    88,    89,    -1,    -1,    -1,
      93,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   101,    -1,
     103,   104,   105,     1,    -1,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    13,    -1,    15,    -1,    -1,
      18,    -1,    -1,    -1,    22,    -1,    -1,    25,    26,    27,
      28,    29,    -1,    31,    32,    -1,    34,    35,    36,    37,
      -1,    -1,    40,    -1,    -1,    43,    -1,    -1,    46,    -1,
      48,    49,    -1,    51,    -1,    53,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    73,    74,    -1,    -1,    77,
      78,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      88,    89,    -1,    -1,    -1,    93,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,   101,    -1,   103,   104,   105,     1,    -1,
       3,     4,     5,     6,     7,     8,    -1,    10,    11,    12,
      13,    -1,    15,    -1,    -1,    18,    -1,    -1,    -1,    22,
      -1,    -1,    25,    26,    27,    28,    29,    -1,    31,    32,
      -1,    34,    35,    36,    37,    -1,    -1,    40,    -1,    -1,
      43,    -1,    -1,    46,    -1,    48,    49,    -1,    51,    -1,
      53,    -1,    -1,     1,    -1,    -1,     4,     5,     6,     7,
       8,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      73,    74,    -1,    -1,    77,    78,    -1,    -1,    26,    27,
      -1,    -1,    -1,    -1,    -1,    88,    89,    -1,    -1,    37,
      93,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   101,    -1,
     103,   104,   105,    51,    -1,    53,    -1,    -1,     1,    -1,
      -1,     4,     5,     6,     7,     8,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    73,    74,    -1,    -1,    -1,
      -1,    -1,    -1,    26,    27,    -1,    -1,    -1,    -1,    -1,
      88,    89,    -1,    -1,    37,    93,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,   101,    -1,   103,   104,   105,    51,    -1,
      53,    -1,    -1,     1,    -1,    -1,     4,     5,     6,     7,
       8,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      73,    74,    -1,    -1,    -1,    -1,    -1,    -1,    26,    27,
      -1,    -1,    -1,    -1,    -1,    88,    89,    -1,    -1,    37,
      93,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   101,    -1,
     103,   104,   105,    51,    -1,    53,    -1,    -1,     1,    -1,
      -1,     4,     5,     6,     7,     8,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    73,    74,    -1,    -1,    -1,
      -1,    -1,    -1,    26,    27,    -1,    -1,    -1,    -1,    -1,
      88,    89,    -1,    -1,    37,    93,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,   101,    -1,   103,   104,   105,    51,    -1,
      53,    -1,    -1,     1,    -1,    -1,     4,     5,     6,     7,
       8,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      73,    74,    -1,    -1,    -1,    -1,    -1,    -1,    26,    27,
      -1,    -1,    -1,    -1,    -1,    88,    89,    -1,    -1,    37,
      93,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   101,    -1,
     103,   104,   105,    51,    -1,    53,    -1,    -1,     1,    -1,
      -1,     4,     5,     6,     7,     8,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    73,    74,    -1,    -1,    -1,
      -1,    -1,    -1,    26,    27,    -1,    -1,    -1,    -1,    -1,
      88,    89,    -1,    -1,    37,    93,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,   101,    -1,   103,   104,   105,    51,    -1,
      53,    -1,    -1,     1,    -1,    -1,     4,     5,     6,     7,
       8,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      73,    74,    -1,    -1,    -1,    -1,    -1,    -1,    26,    27,
      -1,    -1,    -1,    -1,    -1,    88,    89,    -1,    -1,    37,
      93,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   101,    -1,
     103,   104,   105,    51,    -1,    53,    -1,    -1,     1,    -1,
      -1,     4,     5,     6,     7,     8,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    73,    74,    -1,    -1,    -1,
      -1,    -1,    -1,    26,    27,    -1,    -1,    -1,    -1,    -1,
      88,    89,    -1,    -1,    37,    93,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,   101,    -1,   103,   104,   105,    51,    -1,
      53,    -1,    -1,     1,    -1,    -1,     4,     5,     6,     7,
       8,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      73,    74,    -1,    -1,    -1,    -1,    -1,    -1,    26,    27,
      -1,    -1,    -1,    -1,    -1,    88,    89,    -1,    -1,    37,
      93,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   101,    -1,
     103,   104,   105,    51,    -1,    53,    -1,    -1,     1,    -1,
      -1,     4,     5,     6,     7,     8,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    73,    74,    -1,    -1,    -1,
      -1,    -1,    -1,    26,    27,    -1,    -1,    -1,    -1,    -1,
      88,    89,    -1,    -1,    37,    93,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,   101,    -1,   103,   104,   105,    51,    -1,
      53,    -1,    -1,     1,    -1,    -1,     4,     5,     6,     7,
       8,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      73,    74,    -1,    -1,    -1,    -1,    -1,    -1,    26,    27,
      -1,    -1,    -1,    -1,    -1,    88,    89,    -1,    -1,    37,
      93,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   101,    -1,
     103,   104,   105,    51,    -1,    53,    -1,    -1,     1,    -1,
      -1,     4,     5,     6,     7,     8,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    73,    74,    -1,    -1,    -1,
      -1,    -1,    -1,    26,    27,    -1,    -1,    -1,    -1,    -1,
      88,    89,    -1,    -1,    37,    93,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,   101,    -1,   103,   104,   105,    51,    -1,
      53,    -1,    -1,    -1,    -1,     3,     4,     5,     6,     7,
       8,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      73,    74,    -1,    -1,    -1,    -1,    -1,    -1,    26,    27,
      -1,    -1,    -1,    -1,    -1,    88,    89,    -1,    -1,    37,
      93,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   101,    -1,
     103,   104,   105,    51,    -1,    53,    -1,    -1,    -1,    -1,
       3,     4,     5,     6,     7,     8,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    73,    74,    -1,    -1,    -1,
      -1,    -1,    -1,    26,    27,    -1,    -1,    -1,    -1,    -1,
      88,    89,    -1,    -1,    37,    93,    -1,    -1,    -1,     3,
      -1,    -1,    -1,   101,    -1,   103,   104,   105,    51,    -1,
      53,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      73,    74,    -1,    -1,    -1,    -1,     3,    -1,    -1,    -1,
      -1,    45,    -1,    -1,    -1,    88,    89,    51,    -1,    -1,
      93,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   101,    -1,
     103,   104,   105,    67,    -1,    -1,    70,    71,    72,    -1,
      -1,    75,    76,    77,    78,    79,    80,    81,    82,    83,
      84,    85,    86,    87,    51,    -1,    90,    91,    92,    93,
      94,    95,    96,    97,    98,    99,   100,     3,    65,   103,
     104,    -1,    -1,    70,    71,    72,    -1,    -1,    75,    76,
      77,    78,    79,    80,    81,    82,    83,    84,    85,    86,
      87,    -1,    -1,    90,    91,    92,    93,    94,    95,    96,
      97,    98,    99,   100,    -1,    -1,   103,   104,    -1,    45,
      -1,    -1,    -1,    -1,    -1,    51,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,     3,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    70,    71,    72,    -1,    -1,    75,
      76,    77,    78,    79,    80,    81,    82,    83,    84,    85,
      86,    87,    -1,    -1,    90,    91,    92,    93,    94,    95,
      96,    97,    98,    99,   100,    45,    -1,   103,   104,    -1,
      -1,    51,    -1,    -1,    -1,    -1,    -1,    -1,    -1,     3,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      70,    71,    72,    -1,    -1,    75,    76,    77,    78,    79,
      80,    81,    82,    83,    84,    85,    86,    87,    -1,    -1,
      90,    91,    92,    93,    94,    95,    96,    97,    98,    99,
     100,    45,    -1,   103,   104,    -1,    -1,    51,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,     3,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    70,    71,    72,    -1,
      -1,    75,    76,    77,    78,    79,    80,    81,    82,    83,
      84,    85,    86,    87,    -1,    -1,    90,    91,    92,    93,
      94,    95,    96,    97,    98,    99,   100,    45,    -1,   103,
     104,    -1,    -1,    51,    -1,     3,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    70,    71,    72,    -1,    -1,    75,    76,    77,
      78,    79,    80,    81,    82,    83,    84,    85,    86,    87,
      -1,    -1,    90,    91,    92,    93,    94,    95,    96,    97,
      98,    99,   100,    51,    -1,   103,   104,    -1,    -1,     3,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    70,    71,    72,    -1,    -1,    75,    76,    77,
      78,    79,    80,    81,    82,    83,    84,    85,    86,    87,
      -1,    -1,    90,    91,    92,    93,    94,    95,    96,    97,
      98,    99,   100,    -1,    -1,   103,   104,    51,    -1,     3,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    70,    71,    72,    -1,
      -1,    75,    76,    77,    78,    79,    80,    81,    82,    83,
      84,    85,    86,    87,    -1,    -1,    90,    91,    92,    93,
      94,    95,    96,    97,    98,    99,   100,    51,    -1,   103,
     104,    -1,    -1,     3,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    70,    71,    72,    -1,
      -1,    75,    76,    77,    78,    79,    80,    81,    82,    83,
      84,    85,    86,    87,    -1,    -1,    90,    91,    92,    93,
      94,    95,    96,    97,    98,    99,   100,    -1,    -1,   103,
     104,    51,    -1,     3,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      70,    71,    72,    -1,    -1,    75,    76,    77,    78,    79,
      80,    81,    82,    83,    84,    85,    86,    87,    -1,    -1,
      90,    91,    92,    93,    94,    95,    96,    97,    98,    99,
     100,    51,    -1,   103,   104,    -1,    -1,     3,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      70,    71,    72,    -1,    -1,    75,    76,    77,    78,    79,
      80,    81,    82,    83,    84,    85,    86,    87,    -1,    -1,
      90,    91,    92,    93,    94,    95,    96,    97,    98,    99,
     100,    -1,    -1,   103,   104,    51,    -1,     3,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    70,    71,    72,    -1,    -1,    75,
      76,    77,    78,    79,    80,    81,    82,    83,    84,    85,
      86,    87,    -1,    -1,    90,    91,    92,    93,    94,    95,
      96,    97,    98,    99,   100,    51,    -1,   103,   104,    -1,
      -1,     3,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    70,    71,    72,    -1,    -1,    75,
      76,    77,    78,    79,    80,    81,    82,    83,    84,    85,
      86,    87,    -1,    -1,    90,    91,    92,    93,    94,    95,
      96,    97,    98,    99,   100,    -1,    -1,   103,   104,    51,
      -1,     3,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    70,    71,
      72,    -1,    -1,    75,    76,    77,    78,    79,    80,    81,
      82,    83,    84,    85,    86,    87,    -1,    -1,    90,    91,
      92,    93,    94,    95,    96,    97,    98,    99,   100,    51,
      -1,   103,   104,    -1,    -1,     3,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    70,    71,
      72,    -1,    -1,    75,    76,    77,    78,    79,    80,    81,
      82,    83,    84,    85,    86,    87,    -1,    -1,    90,    91,
      92,    93,    94,    95,    96,    97,    98,    99,   100,    -1,
      -1,   103,   104,    51,    -1,     3,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    70,    71,    72,    -1,    -1,    75,    76,    77,
      78,    79,    80,    81,    82,    83,    84,    85,    86,    87,
      -1,    -1,    90,    91,    92,    93,    94,    95,    96,    97,
      98,    99,   100,    51,    -1,   103,   104,    -1,    -1,     3,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    70,    71,    72,    -1,    -1,    75,    76,    77,
      78,    79,    80,    81,    82,    83,    84,    85,    86,    87,
      -1,    -1,    90,    91,    92,    93,    94,    95,    96,    97,
      98,    99,   100,    -1,    -1,   103,   104,    51,    -1,     3,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    70,    71,    72,    -1,
      -1,    75,    76,    77,    78,    79,    80,    81,    82,    83,
      84,    85,    86,    87,    -1,    -1,    90,    91,    92,    93,
      94,    95,    96,    97,    98,    99,   100,    51,    -1,   103,
     104,    -1,    -1,     3,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    70,    71,    72,    -1,
      -1,    75,    76,    77,    78,    79,    80,    81,    82,    83,
      84,    85,    86,    87,    -1,    -1,    90,    91,    92,    93,
      94,    95,    96,    97,    98,    99,   100,    -1,    -1,   103,
     104,    51,    -1,     3,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      70,    71,    72,    -1,    -1,    75,    76,    77,    78,    79,
      80,    81,    82,    83,    84,    85,    86,    87,    -1,    -1,
      90,    91,    92,    93,    94,    95,    96,    97,    98,    99,
     100,    51,    -1,   103,   104,    -1,    -1,     3,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      70,    71,    72,    -1,    -1,    75,    76,    77,    78,    79,
      80,    81,    82,    83,    84,    85,    86,    87,    -1,    -1,
      90,    91,    92,    93,    94,    95,    96,    97,    98,    99,
     100,    -1,    -1,   103,   104,    51,    -1,     3,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    70,    71,    72,    -1,    -1,    75,
      76,    77,    78,    79,    80,    81,    82,    83,    84,    85,
      86,    87,    -1,    -1,    90,    91,    92,    93,    94,    95,
      96,    97,    98,    99,   100,    51,    -1,   103,   104,    -1,
      -1,     3,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    70,    71,    72,    -1,    -1,    75,
      76,    77,    78,    79,    80,    81,    82,    83,    84,    85,
      86,    87,    -1,    -1,    90,    91,    92,    93,    94,    95,
      96,    97,    98,    99,   100,    -1,    -1,   103,   104,    51,
      -1,     3,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    70,    71,
      72,    -1,    -1,    75,    76,    77,    78,    79,    80,    81,
      82,    83,    84,    85,    86,    87,    -1,    -1,    90,    91,
      92,    93,    94,    95,    96,    97,    98,    99,   100,    51,
      -1,   103,   104,    -1,    -1,     3,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    70,    71,
      72,    -1,    -1,    75,    76,    77,    78,    79,    80,    81,
      82,    83,    84,    85,    86,    87,    -1,    -1,    90,    91,
      92,    93,    94,    95,    96,    97,    98,    99,   100,    -1,
      -1,   103,   104,    51,    -1,     3,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    70,    71,    72,    -1,    -1,    75,    76,    77,
      78,    79,    80,    81,    82,    83,    84,    85,    86,    87,
      -1,    -1,    90,    91,    92,    93,    94,    95,    96,    97,
      98,    99,   100,    51,    -1,   103,   104,    -1,    -1,     3,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    70,    71,    72,    -1,    -1,    75,    76,    77,
      78,    79,    80,    81,    82,    83,    84,    85,    86,    87,
      -1,    -1,    90,    91,    92,    93,    94,    95,    96,    97,
      98,    99,   100,    -1,    -1,   103,   104,    51,    -1,     3,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    70,    71,    72,    -1,
      -1,    75,    76,    77,    78,    79,    80,    81,    82,    83,
      84,    85,    86,    87,    -1,    -1,    90,    91,    92,    93,
      94,    95,    96,    97,    98,    99,   100,    51,    -1,   103,
     104,    -1,    -1,     3,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    70,    71,    72,    -1,
      -1,    75,    76,    77,    78,    79,    80,    81,    82,    83,
      84,    85,    86,    87,    -1,    -1,    90,    91,    92,    93,
      94,    95,    96,    97,    98,    99,   100,    -1,    -1,   103,
     104,    51,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      70,    71,    72,    -1,    -1,    75,    76,    77,    78,    79,
      80,    81,    82,    83,    84,    85,    86,    87,    -1,    -1,
      90,    91,    92,    93,    94,    95,    96,    97,    98,    99,
     100,    -1,    -1,   103,   104,     4,     5,     6,     7,     8,
      -1,    -1,    -1,    -1,     4,     5,     6,     7,     8,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    26,    27,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    26,    27,    37,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    45,    37,    -1,    -1,
      -1,    -1,    51,    52,    53,    45,    -1,    -1,    -1,    -1,
      -1,    51,    -1,    53,    -1,    -1,    -1,    66,    -1,    -1,
      -1,    -1,    -1,    -1,    73,    74,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    73,    74,    -1,    -1,    -1,    -1,    88,
      89,    -1,    -1,    -1,    93,    -1,    -1,    -1,    88,    89,
      -1,    -1,   101,    93,   103,   104,   105,    97,    -1,    -1,
      -1,   101,    -1,   103,   104,   105,     4,     5,     6,     7,
       8,    -1,     4,     5,     6,     7,     8,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    26,    27,
      -1,    -1,    -1,    -1,    26,    27,    -1,    -1,    -1,    37,
      -1,    -1,    -1,    -1,    -1,    37,    -1,    45,    -1,    -1,
      -1,    -1,    -1,    51,    -1,    53,    -1,    -1,    -1,    51,
      52,    53,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    73,    74,    -1,    -1,    -1,
      -1,    73,    74,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      88,    89,    -1,    -1,    -1,    93,    88,    89,    -1,    97,
      -1,    93,    -1,   101,    -1,   103,   104,   105,    -1,   101,
      -1,   103,   104,   105,     4,     5,     6,     7,     8,    -1,
       4,     5,     6,     7,     8,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    26,    27,    -1,    -1,
      -1,    -1,    26,    27,    -1,    -1,    -1,    37,    -1,    -1,
      -1,    -1,    -1,    37,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    51,    52,    53,    -1,    -1,    50,    51,    -1,    53,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    73,    74,    -1,    -1,    -1,    -1,    73,
      74,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    88,    89,
      -1,    -1,    -1,    93,    88,    89,    -1,    -1,    -1,    93,
      -1,   101,    -1,   103,   104,   105,    -1,   101,    -1,   103,
     104,   105,     4,     5,     6,     7,     8,    -1,     4,     5,
       6,     7,     8,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    26,    27,    -1,    -1,    -1,    -1,
      26,    27,    -1,    -1,    -1,    37,    -1,    -1,    -1,    -1,
      -1,    37,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    51,
      -1,    53,    -1,    -1,    -1,    51,    -1,    53,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    73,    74,    -1,    -1,    -1,    -1,    73,    74,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    88,    89,    -1,    -1,
      -1,    93,    88,    89,    -1,    -1,    -1,    93,    -1,   101,
      -1,   103,   104,   105,    -1,   101,    -1,   103,   104,   105,
       4,     5,     6,     7,     8,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    26,    27,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    37,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    51,    -1,    53,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    73,
      74,    -1,    -1,    -1,    -1,    -1,    -1,    45,    -1,    -1,
      -1,    -1,    -1,    51,    88,    89,    -1,    -1,    -1,    93,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,   101,    66,   103,
     104,   105,    70,    71,    72,    -1,    -1,    75,    76,    77,
      78,    79,    80,    81,    82,    83,    84,    85,    86,    87,
      -1,    -1,    90,    91,    92,    93,    94,    95,    96,    97,
      98,    99,   100,    45,    -1,   103,   104,    -1,    -1,    51,
      52,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    70,    71,
      72,    -1,    -1,    75,    76,    77,    78,    79,    80,    81,
      82,    83,    84,    85,    86,    87,    -1,    -1,    90,    91,
      92,    93,    94,    95,    96,    97,    98,    99,   100,    45,
      -1,   103,   104,    -1,    -1,    51,    52,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    70,    71,    72,    -1,    -1,    75,
      76,    77,    78,    79,    80,    81,    82,    83,    84,    85,
      86,    87,    -1,    -1,    90,    91,    92,    93,    94,    95,
      96,    97,    98,    99,   100,    45,    -1,   103,   104,    -1,
      -1,    51,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      70,    71,    72,    -1,    -1,    75,    76,    77,    78,    79,
      80,    81,    82,    83,    84,    85,    86,    87,    51,    52,
      90,    91,    92,    93,    94,    95,    96,    97,    98,    99,
     100,    -1,    -1,   103,   104,    -1,    -1,    70,    71,    72,
      -1,    -1,    75,    76,    77,    78,    79,    80,    81,    82,
      83,    84,    85,    86,    87,    51,    -1,    90,    91,    92,
      93,    94,    95,    96,    97,    98,    99,   100,    -1,    -1,
     103,   104,    68,    -1,    70,    71,    72,    -1,    -1,    75,
      76,    77,    78,    79,    80,    81,    82,    83,    84,    85,
      86,    87,    51,    52,    90,    91,    92,    93,    94,    95,
      96,    97,    98,    99,   100,    -1,    -1,   103,   104,    -1,
      -1,    70,    71,    72,    -1,    -1,    75,    76,    77,    78,
      79,    80,    81,    82,    83,    84,    85,    86,    87,    51,
      52,    90,    91,    92,    93,    94,    95,    96,    97,    98,
      99,   100,    -1,    -1,   103,   104,    -1,    -1,    70,    71,
      72,    -1,    -1,    75,    76,    77,    78,    79,    80,    81,
      82,    83,    84,    85,    86,    87,    51,    52,    90,    91,
      92,    93,    94,    95,    96,    97,    98,    99,   100,    -1,
      -1,   103,   104,    -1,    -1,    70,    71,    72,    -1,    -1,
      75,    76,    77,    78,    79,    80,    81,    82,    83,    84,
      85,    86,    87,    -1,    -1,    90,    91,    92,    93,    94,
      95,    96,    97,    98,    99,   100,    51,    -1,   103,   104,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    66,    -1,    -1,    -1,    70,    71,    72,    -1,    -1,
      75,    76,    77,    78,    79,    80,    81,    82,    83,    84,
      85,    86,    87,    51,    -1,    90,    91,    92,    93,    94,
      95,    96,    97,    98,    99,   100,    -1,    -1,   103,   104,
      -1,    -1,    70,    71,    72,    -1,    -1,    75,    76,    77,
      78,    79,    80,    81,    82,    83,    84,    85,    86,    87,
      51,    -1,    90,    91,    92,    93,    94,    95,    96,    97,
      98,    99,   100,    -1,    -1,   103,   104,    -1,    -1,    -1,
      71,    72,    -1,    -1,    75,    76,    77,    78,    79,    80,
      81,    82,    83,    84,    85,    86,    87,    51,    -1,    90,
      91,    92,    93,    94,    95,    96,    97,    98,    99,   100,
      -1,    -1,   103,   104,    -1,    -1,    -1,    -1,    72,    -1,
      -1,    75,    76,    77,    78,    79,    80,    81,    82,    83,
      84,    85,    86,    87,    51,    -1,    90,    91,    92,    93,
      94,    95,    96,    97,    98,    99,   100,    -1,    -1,   103,
     104,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    75,    76,
      77,    78,    79,    80,    81,    82,    83,    84,    85,    86,
      87,    -1,    -1,    90,    91,    92,    93,    94,    95,    96,
      97,    98,    99,   100,    -1,    -1,   103,   104
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint16 yystos[] =
{
       0,   107,   108,     0,     1,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    13,    15,    18,    22,    23,
      25,    26,    27,    28,    29,    31,    32,    34,    35,    36,
      37,    39,    40,    41,    42,    43,    44,    46,    48,    49,
      51,    53,    73,    74,    77,    78,    88,    89,    93,   101,
     103,   104,   105,   109,   110,   111,   112,   114,   115,   116,
     117,   118,   119,   120,   121,   122,   123,   124,   125,   126,
     127,   128,   129,   131,   132,   133,   135,   136,   144,   145,
     146,   148,   149,   150,   154,   155,   162,   164,   177,   179,
     188,   189,   191,   198,   199,   200,   202,   204,   212,   213,
     214,   215,   217,   218,   219,   222,   240,   245,   249,   250,
     251,   252,   253,   254,   255,   256,   260,   264,   265,   267,
       3,     3,     1,   113,   251,     1,   253,   254,     1,     3,
       1,     3,    14,     1,   254,     1,   251,   253,   271,     1,
     254,     1,     1,   254,     1,   254,   269,     1,     3,    45,
       1,   254,     1,     6,     1,     6,     1,     3,   254,   246,
     261,     1,     6,     7,     1,   254,   256,     1,     6,     1,
       3,    45,     1,   254,     1,     3,     6,   216,     1,     6,
       1,   254,     3,    45,     3,   258,   259,    45,    52,    66,
     254,   270,   272,   254,   253,     1,     3,   269,     3,   269,
     254,   254,   254,   254,   254,   254,   254,   130,    32,    34,
      46,   112,   134,   112,   147,   112,   163,   178,   190,    47,
     207,   210,   211,   112,     1,    51,     1,     6,   220,   221,
     220,    53,    54,    55,    56,    57,    58,    59,    60,    61,
      62,    63,    64,    69,    82,   255,     3,    51,    65,    70,
      71,    72,    75,    76,    77,    78,    79,    80,    81,    82,
      83,    84,    85,    86,    87,    90,    91,    92,    93,    94,
      95,    96,    97,    98,    99,   100,   103,   104,    53,    54,
     255,     3,     3,    69,    82,     3,    45,     3,    45,     3,
       3,     3,     3,    45,     3,    45,     3,    45,    82,    69,
      85,     3,     3,     3,     3,     3,     3,     1,    68,    69,
       3,   112,     3,     3,     3,   223,     3,   241,     3,     3,
       6,   247,   248,     1,    51,   262,     3,     3,     3,     3,
       3,     3,    82,     3,    45,     3,     3,    85,     3,     3,
      69,     3,     3,     3,   254,     3,    52,   254,    52,    45,
      66,     1,    69,   258,     1,    69,   258,    81,    82,     3,
       3,     3,   143,   143,   143,   165,   180,   143,     1,     3,
      45,   143,   208,   209,     3,   258,     3,     3,    69,     9,
     220,     3,    97,   254,     6,   254,   254,   254,   254,   254,
     254,   254,   254,   254,   254,   251,   271,   105,   254,   269,
     258,   254,     1,   254,   254,   254,   254,   254,   254,   254,
     254,   254,   254,   254,     6,   254,   254,   254,   254,   254,
     254,   254,   254,   254,   254,   254,   254,   254,   254,   254,
      97,   254,     6,   251,   254,   254,   251,     1,   254,     3,
     254,   254,     1,    51,   224,   225,     1,    33,   227,   242,
       3,    69,     3,   258,   207,     1,   250,     1,   254,     6,
     258,    52,    52,   254,   254,   266,   258,    52,   268,   258,
      52,   254,   254,     9,   112,    16,    17,   137,   139,   140,
     142,     9,     1,     3,    23,    24,   166,   171,   173,     1,
       3,    23,   171,   181,    30,   192,   193,   194,   195,     3,
      45,     9,   143,   112,     1,     6,   205,   206,     6,     3,
       3,   254,    52,     3,     3,     3,     3,     3,     3,     3,
       3,     3,     3,    82,   105,     3,     3,    50,   254,   270,
       3,    45,   254,    52,    82,    68,     3,     3,    45,     3,
       3,   258,   233,   227,     3,     6,   228,   229,     3,   243,
       1,   248,   205,   143,     3,     3,     3,     3,    50,    52,
     258,   254,   258,   254,     3,     1,     3,     1,   254,     9,
     138,   141,     3,     3,     1,     4,     6,     7,     8,   175,
     176,     1,     9,   172,     3,     1,     4,     6,   186,   187,
       9,     1,     3,     4,     6,    85,   196,   197,     9,   194,
     143,     3,     9,   203,     1,    69,   258,    52,   254,   269,
       3,     1,   258,   254,    52,   254,   254,   151,   112,   205,
       3,     6,    38,    46,    47,    87,   199,   234,   235,   237,
     238,     3,    51,   230,    69,     3,   199,   235,   237,   238,
     244,     1,   258,     9,    52,    52,    66,     3,     3,     3,
       3,   143,   143,     3,    45,    68,     3,    45,    69,     3,
       3,    45,   174,     3,    45,     3,    45,    69,     3,     3,
     251,     3,    69,    85,     3,     3,   258,   201,   258,    50,
       3,     3,   257,    50,     3,    45,    67,    19,    20,    21,
     112,   152,   153,   156,   158,   160,     1,   258,    82,     3,
       6,     1,     6,    73,   239,     9,   258,   229,     9,   263,
      50,   254,   137,   169,   170,     4,   167,   168,   176,   143,
     112,   184,   185,   182,   183,   187,     3,   197,   251,    50,
     258,   206,     3,    45,   258,   254,     1,     3,    45,     1,
       3,    45,     1,     3,    45,     9,   152,   226,    50,   254,
     236,    82,     3,     6,     3,    69,     3,     6,    27,   231,
     232,   250,     3,   258,     3,   143,   112,   143,   112,   143,
     112,   143,   112,     3,    45,    50,    50,     3,    45,     3,
     157,   112,     3,   159,   112,     3,   161,   112,     3,   258,
       3,   207,   254,     6,    73,    69,   258,    50,     3,   143,
     143,   143,    50,   143,     3,     6,   232,    50,     3,     9,
       9,     9,     9,     3,     3,     3,     3
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
#line 198 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_lone_end ); ;}
    break;

  case 7:
#line 199 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_case_outside ); ;}
    break;

  case 8:
#line 205 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(1) - (1)].stringp) != 0 )
            COMPILER->addLoad( *(yyvsp[(1) - (1)].stringp) );
      ;}
    break;

  case 9:
#line 210 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if( (yyvsp[(1) - (1)].fal_stat) != 0 )
            COMPILER->addFunction( (yyvsp[(1) - (1)].fal_stat) );
      ;}
    break;

  case 10:
#line 215 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(1) - (1)].fal_stat) != 0 )
            COMPILER->addClass( (yyvsp[(1) - (1)].fal_stat) );
      ;}
    break;

  case 11:
#line 220 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(1) - (1)].fal_stat) != 0 )
            COMPILER->addClass( (yyvsp[(1) - (1)].fal_stat) );
      ;}
    break;

  case 12:
#line 225 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if( (yyvsp[(1) - (1)].fal_stat) != 0 )
            COMPILER->addStatement( (yyvsp[(1) - (1)].fal_stat) );
      ;}
    break;

  case 16:
#line 236 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getContext() != 0 )
            COMPILER->raiseError(Falcon::e_toplevel_load );
         (yyval.stringp) = (yyvsp[(2) - (3)].stringp);
      ;}
    break;

  case 17:
#line 242 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getContext() != 0 )
            COMPILER->raiseError(Falcon::e_toplevel_load );
         (yyval.stringp) = (yyvsp[(2) - (3)].stringp);
      ;}
    break;

  case 18:
#line 248 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_load );
         (yyval.stringp) = 0;
      ;}
    break;

  case 19:
#line 255 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->checkLocalUndefined(); (yyval.fal_stat) = (yyvsp[(1) - (1)].fal_stat); ;}
    break;

  case 20:
#line 256 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = 0; ;}
    break;

  case 21:
#line 259 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_toplevel_func ); (yyval.fal_stat) = 0; ;}
    break;

  case 22:
#line 260 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_toplevel_obj ); (yyval.fal_stat) = 0; ;}
    break;

  case 23:
#line 261 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_toplevel_class ); (yyval.fal_stat) = 0; ;}
    break;

  case 24:
#line 262 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syntax ); (yyval.fal_stat) = 0;;}
    break;

  case 25:
#line 267 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defContext( true ); COMPILER->defRequired();
      COMPILER->defineVal( (yyvsp[(1) - (3)].fal_val) );
      COMPILER->addStatement( new Falcon::StmtAssignment( CURRENT_LINE, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) );
   ;}
    break;

  case 26:
#line 272 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(3) - (5)].fal_val) );
      COMPILER->addStatement( new Falcon::StmtAssignment( CURRENT_LINE, (yyvsp[(3) - (5)].fal_val), (yyvsp[(5) - (5)].fal_val) ) );
   ;}
    break;

  case 27:
#line 279 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = new Falcon::StmtAutoexpr( LINE, (yyvsp[(1) - (2)].fal_val) ); ;}
    break;

  case 48:
#line 304 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->defContext( false );  (yyval.fal_stat)=0; ;}
    break;

  case 49:
#line 306 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError( Falcon::e_syn_def ); ;}
    break;

  case 50:
#line 311 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAssignment( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 51:
#line 315 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (5)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtUnref( LINE, (yyvsp[(1) - (5)].fal_val) );
   ;}
    break;

  case 52:
#line 319 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
         (yyval.fal_stat) = new Falcon::StmtAssignment( LINE, (yyvsp[(1) - (4)].fal_val), new Falcon::Value( (yyvsp[(3) - (4)].fal_adecl) ) );
      ;}
    break;

  case 53:
#line 323 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(1) - (6)].fal_val) );
         (yyvsp[(3) - (6)].fal_adecl)->pushFront( (yyvsp[(1) - (6)].fal_val) );
         (yyval.fal_stat) = new Falcon::StmtAssignment( LINE, new Falcon::Value((yyvsp[(3) - (6)].fal_adecl)), (yyvsp[(5) - (6)].fal_val) );
      ;}
    break;

  case 54:
#line 328 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(1) - (6)].fal_val) );
         (yyvsp[(3) - (6)].fal_adecl)->pushFront( (yyvsp[(1) - (6)].fal_val) );
         (yyval.fal_stat) = new Falcon::StmtAssignment( LINE, new Falcon::Value((yyvsp[(3) - (6)].fal_adecl)), new Falcon::Value( (yyvsp[(5) - (6)].fal_adecl) ) );
      ;}
    break;

  case 66:
#line 352 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAutoAdd( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 67:
#line 359 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAutoSub( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 68:
#line 366 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAutoMul( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 69:
#line 373 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAutoDiv( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 70:
#line 380 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAutoMod( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 71:
#line 387 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAutoPow( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 72:
#line 394 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAutoBAND( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 73:
#line 401 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAutoBOR( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 74:
#line 408 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAutoBXOR( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 75:
#line 414 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAutoSHL( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 76:
#line 420 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAutoSHR( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 77:
#line 428 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtWhile *w = new Falcon::StmtWhile( LINE, (yyvsp[(1) - (1)].fal_val) );
         COMPILER->pushLoop( w );
         COMPILER->pushContext( w );
         COMPILER->pushContextSet( &w->children() );
      ;}
    break;

  case 78:
#line 435 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtWhile *w = static_cast<Falcon::StmtWhile *>(COMPILER->getContext());
         COMPILER->popLoop();
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = w;
      ;}
    break;

  case 79:
#line 443 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtWhile *w = new Falcon::StmtWhile( LINE, (yyvsp[(1) - (2)].fal_val) );
         if ( (yyvsp[(2) - (2)].fal_stat) != 0 )
            w->children().push_back( (yyvsp[(2) - (2)].fal_stat) );
         (yyval.fal_stat) = w;
      ;}
    break;

  case 80:
#line 451 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); ;}
    break;

  case 81:
#line 452 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = 0; ;}
    break;

  case 82:
#line 453 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_while ); (yyval.fal_val) = 0; ;}
    break;

  case 83:
#line 457 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); ;}
    break;

  case 84:
#line 458 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = 0; ;}
    break;

  case 85:
#line 459 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_while, "", CURRENT_LINE ); (yyval.fal_val) = 0; ;}
    break;

  case 86:
#line 463 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtIf *stmt = new Falcon::StmtIf( LINE, (yyvsp[(1) - (1)].fal_val) );
         COMPILER->pushContext( stmt );
         COMPILER->pushContextSet( &stmt->children() );
      ;}
    break;

  case 87:
#line 471 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtIf *stmt = static_cast<Falcon::StmtIf *>(COMPILER->getContext());
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = stmt;
      ;}
    break;

  case 88:
#line 478 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // use LINE as statement includes EOL
         Falcon::StmtIf *stmt = new Falcon::StmtIf( LINE, (yyvsp[(1) - (2)].fal_val) );
         if( (yyvsp[(2) - (2)].fal_stat) != 0 )
            stmt->children().push_back( (yyvsp[(2) - (2)].fal_stat) );
         (yyval.fal_stat) = stmt;
      ;}
    break;

  case 89:
#line 488 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); ;}
    break;

  case 90:
#line 489 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {  COMPILER->raiseError(Falcon::e_syn_if ); (yyval.fal_val) = 0; ;}
    break;

  case 91:
#line 493 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); ;}
    break;

  case 92:
#line 494 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {  COMPILER->raiseError(Falcon::e_syn_if, "", CURRENT_LINE ); (yyval.fal_val) = 0; ;}
    break;

  case 95:
#line 501 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtIf *stmt = static_cast<Falcon::StmtIf *>(COMPILER->getContext());
         COMPILER->popContextSet();
         COMPILER->pushContextSet( &stmt->elseChildren() );
      ;}
    break;

  case 98:
#line 511 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_else ); ;}
    break;

  case 99:
#line 516 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtIf *stmt = static_cast<Falcon::StmtIf *>(COMPILER->getContext());
         COMPILER->popContextSet();
         Falcon::StmtElif *elif = new Falcon::StmtElif( LINE, (yyvsp[(1) - (1)].fal_val) );
         stmt->elifChildren().push_back( elif );
         COMPILER->pushContextSet( &elif->children() );
      ;}
    break;

  case 101:
#line 528 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); ;}
    break;

  case 102:
#line 529 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_elif ); (yyval.fal_val) = 0; ;}
    break;

  case 104:
#line 534 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->addStatement( (yyvsp[(2) - (2)].fal_stat) );
   ;}
    break;

  case 105:
#line 541 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getLoop() == 0 ) {
            COMPILER->raiseError(Falcon::e_break_out );
            (yyval.fal_stat) = 0;
         }
         else
            (yyval.fal_stat) = new Falcon::StmtBreak( LINE );
      ;}
    break;

  case 106:
#line 550 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_break );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 107:
#line 558 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getLoop() == 0 ) {
            COMPILER->raiseError(Falcon::e_continue_out );
            (yyval.fal_stat) = 0;
         }
         else
            (yyval.fal_stat) = new Falcon::StmtContinue( LINE );
      ;}
    break;

  case 108:
#line 568 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getLoop() == 0 ) {
            COMPILER->raiseError(Falcon::e_continue_out );
            (yyval.fal_stat) = 0;
         }
         else
            (yyval.fal_stat) = new Falcon::StmtContinue( LINE, true );
      ;}
    break;

  case 109:
#line 577 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_continue );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 110:
#line 584 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtFor *f = static_cast<Falcon::StmtFor *>( (yyvsp[(1) - (1)].fal_stat) );
         COMPILER->pushLoop( f );
         COMPILER->pushContext( f );
         COMPILER->pushContextSet( &f->children() );
      ;}
    break;

  case 111:
#line 591 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtFor *f = static_cast<Falcon::StmtFor *>(COMPILER->getContext());
         COMPILER->popLoop();
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = f;
      ;}
    break;

  case 112:
#line 599 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 113:
#line 614 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(2) - (7)].fal_val) );
         (yyval.fal_stat) = new Falcon::StmtFor( LINE, (yyvsp[(2) - (7)].fal_val), (yyvsp[(4) - (7)].fal_val), (yyvsp[(6) - (7)].fal_val) );
      ;}
    break;

  case 114:
#line 618 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(2) - (9)].fal_val) );
         (yyval.fal_stat) = new Falcon::StmtFor( LINE, (yyvsp[(2) - (9)].fal_val), (yyvsp[(4) - (9)].fal_val), (yyvsp[(6) - (9)].fal_val), (yyvsp[(8) - (9)].fal_val) );
      ;}
    break;

  case 115:
#line 623 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_for );
         (yyval.fal_stat) = new Falcon::StmtFor( LINE, 0, 0, 0 );
      ;}
    break;

  case 116:
#line 630 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(2) - (7)].fal_val) );
         (yyval.fal_stat) = new Falcon::StmtFor( CURRENT_LINE, (yyvsp[(2) - (7)].fal_val), (yyvsp[(4) - (7)].fal_val), (yyvsp[(6) - (7)].fal_val) );
      ;}
    break;

  case 117:
#line 634 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(2) - (9)].fal_val) );
         (yyval.fal_stat) = new Falcon::StmtFor( CURRENT_LINE, (yyvsp[(2) - (9)].fal_val), (yyvsp[(4) - (9)].fal_val), (yyvsp[(6) - (9)].fal_val), (yyvsp[(8) - (9)].fal_val) );
      ;}
    break;

  case 118:
#line 639 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_for, "", CURRENT_LINE );
         (yyval.fal_stat) = new Falcon::StmtFor( CURRENT_LINE, 0, 0, 0 );
      ;}
    break;

  case 119:
#line 648 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 120:
#line 664 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         COMPILER->popLoop();
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = f;
      ;}
    break;

  case 121:
#line 672 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 122:
#line 686 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError( Falcon::e_syn_forin ); ;}
    break;

  case 125:
#line 695 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(1) - (1)].fal_stat) != 0 )
         {
            Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
            f->children().push_back( (yyvsp[(1) - (1)].fal_stat) );
         }
      ;}
    break;

  case 129:
#line 709 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 130:
#line 722 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_fordot );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 131:
#line 730 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = new Falcon::StmtSelfPrint( LINE, (yyvsp[(2) - (3)].fal_adecl) );
      ;}
    break;

  case 132:
#line 734 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_self_print );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 133:
#line 740 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyvsp[(2) - (3)].fal_adecl)->pushBack( new Falcon::Value( COMPILER->addString( "\n" ) ) );
         (yyval.fal_stat) = new Falcon::StmtSelfPrint( LINE, (yyvsp[(2) - (3)].fal_adecl) );
      ;}
    break;

  case 134:
#line 745 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::ArrayDecl *adecl = new Falcon::ArrayDecl();
         adecl->pushBack( new Falcon::Value( COMPILER->addString( "\n" ) ) );
         (yyval.fal_stat) = new Falcon::StmtSelfPrint( LINE, adecl );
      ;}
    break;

  case 135:
#line 752 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_self_print );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 136:
#line 759 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 137:
#line 771 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->popContextSet(); ;}
    break;

  case 138:
#line 773 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         if( ! f->firstBlock().empty() )
         {
            COMPILER->raiseError( Falcon::e_already_forfirst );
         }
         f->firstBlock().push_back( (yyvsp[(3) - (3)].fal_stat) );
      ;}
    break;

  case 139:
#line 781 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_forfirst ); ;}
    break;

  case 140:
#line 785 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 141:
#line 797 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->popContextSet(); ;}
    break;

  case 142:
#line 798 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         if( ! f->lastBlock().empty() )
         {
            COMPILER->raiseError( Falcon::e_already_forlast );
         }
         f->lastBlock().push_back( (yyvsp[(3) - (3)].fal_stat) );
      ;}
    break;

  case 143:
#line 806 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_forlast ); ;}
    break;

  case 144:
#line 810 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 145:
#line 822 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->popContextSet(); ;}
    break;

  case 146:
#line 824 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         if( ! f->allBlock().empty() )
         {
            COMPILER->raiseError( Falcon::e_already_forall );
         }
         f->allBlock().push_back( (yyvsp[(3) - (3)].fal_stat) );
      ;}
    break;

  case 147:
#line 832 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_forall ); ;}
    break;

  case 148:
#line 836 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::StmtSwitch *stmt = new Falcon::StmtSwitch( LINE, (yyvsp[(1) - (1)].fal_val) );
         COMPILER->pushContext( stmt );
         COMPILER->pushContextSet( &stmt->blocks() );
      ;}
    break;

  case 149:
#line 844 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = stmt;
      ;}
    break;

  case 150:
#line 853 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); ;}
    break;

  case 151:
#line 855 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_switch_decl );
         (yyval.fal_val) = 0;
      ;}
    break;

  case 154:
#line 864 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_switch_body ); ;}
    break;

  case 156:
#line 870 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      ;}
    break;

  case 158:
#line 880 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( CURRENT_LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      ;}
    break;

  case 159:
#line 888 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addStatement( (yyvsp[(5) - (5)].fal_stat) );
      ;}
    break;

  case 160:
#line 892 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_case_decl );

         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      ;}
    break;

  case 162:
#line 904 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_case_decl );

         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( CURRENT_LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      ;}
    break;

  case 163:
#line 914 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addStatement( (yyvsp[(5) - (5)].fal_stat) );
      ;}
    break;

  case 165:
#line 923 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 169:
#line 937 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_default_decl ); ;}
    break;

  case 171:
#line 941 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addStatement( (yyvsp[(2) - (2)].fal_stat) );
      ;}
    break;

  case 174:
#line 953 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         // todo: correct error
         if ( stmt->nilBlock() != -1 )
            COMPILER->raiseError(Falcon::e_switch_clash, "nil entry", CURRENT_LINE );
         stmt->nilBlock( stmt->currentBlock() );
      ;}
    break;

  case 175:
#line 962 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 176:
#line 974 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 177:
#line 985 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 178:
#line 996 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 179:
#line 1016 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::StmtSelect *stmt = new Falcon::StmtSelect( LINE, (yyvsp[(1) - (1)].fal_val) );
         COMPILER->pushContext( stmt );
         COMPILER->pushContextSet( &stmt->blocks() );
      ;}
    break;

  case 180:
#line 1024 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = stmt;
      ;}
    break;

  case 181:
#line 1033 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); ;}
    break;

  case 182:
#line 1035 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_select_decl );
         (yyval.fal_val) = 0;
      ;}
    break;

  case 185:
#line 1044 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_select_body ); ;}
    break;

  case 187:
#line 1050 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      ;}
    break;

  case 189:
#line 1060 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( CURRENT_LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      ;}
    break;

  case 190:
#line 1069 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addStatement( (yyvsp[(5) - (5)].fal_stat) );
      ;}
    break;

  case 191:
#line 1073 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_case_decl );

         Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      ;}
    break;

  case 193:
#line 1085 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_case_decl );

        Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( CURRENT_LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      ;}
    break;

  case 194:
#line 1095 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addStatement( (yyvsp[(5) - (5)].fal_stat) );
      ;}
    break;

  case 198:
#line 1109 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 199:
#line 1121 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 200:
#line 1142 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = new Falcon::StmtGive( LINE, (yyvsp[(4) - (5)].fal_val), (yyvsp[(2) - (5)].fal_adecl) );
      ;}
    break;

  case 201:
#line 1146 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = new Falcon::StmtGive( LINE, 0, (yyvsp[(2) - (4)].fal_adecl) );
         COMPILER->raiseError(Falcon::e_syn_give );
      ;}
    break;

  case 202:
#line 1150 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_give ); (yyval.fal_stat) = 0; ;}
    break;

  case 203:
#line 1158 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::StmtTry *t = new Falcon::StmtTry( CURRENT_LINE );
      if ( (yyvsp[(3) - (3)].fal_stat) != 0 )
          t->children().push_back( (yyvsp[(3) - (3)].fal_stat) );
      (yyval.fal_stat) = t;
   ;}
    break;

  case 204:
#line 1165 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtTry *t = new Falcon::StmtTry( LINE );
         COMPILER->pushContext( t );
         COMPILER->pushContextSet( &t->children() );
      ;}
    break;

  case 205:
#line 1175 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = COMPILER->getContext();
         COMPILER->popContext();
         COMPILER->popContextSet();
      ;}
    break;

  case 207:
#line 1184 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_try ); ;}
    break;

  case 213:
#line 1204 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 214:
#line 1222 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 215:
#line 1242 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->popContextSet(); // popping previous catch

         Falcon::StmtTry *t = static_cast<Falcon::StmtTry *>( COMPILER->getContext() );
         Falcon::StmtCatchBlock *lst = new Falcon::StmtCatchBlock( LINE, 0 );
         COMPILER->pushContextSet( &lst->children() );
         t->addHandler( lst );
      ;}
    break;

  case 216:
#line 1252 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->popContextSet(); // popping previous catch

         Falcon::StmtTry *t = static_cast<Falcon::StmtTry *>( COMPILER->getContext() );
         COMPILER->defineVal( (yyvsp[(4) - (5)].fal_val) );
         Falcon::StmtCatchBlock *lst = new Falcon::StmtCatchBlock( LINE, (yyvsp[(4) - (5)].fal_val) );
         COMPILER->pushContextSet( &lst->children() );
         t->addHandler( lst );
      ;}
    break;

  case 217:
#line 1263 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->raiseError( Falcon::e_syn_catch );
   ;}
    break;

  case 220:
#line 1276 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 221:
#line 1288 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 222:
#line 1310 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = new Falcon::StmtRaise( LINE, (yyvsp[(2) - (3)].fal_val) ); ;}
    break;

  case 223:
#line 1311 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_raise ); (yyval.fal_stat) = 0; ;}
    break;

  case 224:
#line 1323 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = COMPILER->getContext();
         COMPILER->closeFunction();
      ;}
    break;

  case 225:
#line 1329 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addStatement( (yyvsp[(2) - (2)].fal_stat) );
         (yyval.fal_stat) = COMPILER->getContext();
         COMPILER->closeFunction();
      ;}
    break;

  case 227:
#line 1338 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); ;}
    break;

  case 228:
#line 1339 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_funcdecl, "", COMPILER->tempLine() );
      ;}
    break;

  case 229:
#line 1342 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_funcdecl ); ;}
    break;

  case 231:
#line 1347 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); ;}
    break;

  case 232:
#line 1348 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_funcdecl, "", COMPILER->tempLine() );
      ;}
    break;

  case 233:
#line 1355 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 237:
#line 1416 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 239:
#line 1433 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());
         COMPILER->pushContextSet( &func->staticBlock() );
         COMPILER->staticPrefix( &func->symbol()->name() );
      ;}
    break;

  case 240:
#line 1439 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->popContextSet();
         COMPILER->staticPrefix(0);
      ;}
    break;

  case 241:
#line 1444 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());
         COMPILER->pushContextSet( &func->staticBlock() );
         COMPILER->staticPrefix( &func->symbol()->name() );
      ;}
    break;

  case 242:
#line 1450 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addStatement( (yyvsp[(3) - (3)].fal_stat) );
         COMPILER->popContextSet();
         COMPILER->staticPrefix(0);
      ;}
    break;

  case 244:
#line 1459 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_static ); ;}
    break;

  case 246:
#line 1464 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_static, "", CURRENT_LINE ); ;}
    break;

  case 247:
#line 1474 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = new Falcon::StmtLaunch( LINE, (yyvsp[(2) - (3)].fal_val) );
      ;}
    break;

  case 248:
#line 1477 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_launch ); (yyval.fal_stat) = 0; ;}
    break;

  case 249:
#line 1486 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getFunction() == 0 )
            COMPILER->raiseError(Falcon::e_pass_outside );
         else
            (yyval.fal_stat) = new Falcon::StmtPass( LINE, (yyvsp[(2) - (3)].fal_val) );
      ;}
    break;

  case 250:
#line 1493 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // define the expression anyhow so we don't have fake errors below
         COMPILER->defineVal( (yyvsp[(4) - (5)].fal_val) );

         if ( COMPILER->getFunction() == 0 )
            COMPILER->raiseError(Falcon::e_pass_outside );
         else
            (yyval.fal_stat) = new Falcon::StmtPass( LINE, (yyvsp[(2) - (5)].fal_val), (yyvsp[(4) - (5)].fal_val) );
      ;}
    break;

  case 251:
#line 1503 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         delete (yyvsp[(2) - (5)].fal_val);
         COMPILER->raiseError(Falcon::e_syn_pass_in );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 252:
#line 1509 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_pass );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 253:
#line 1521 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 254:
#line 1531 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_inv_const_val );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 255:
#line 1536 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_const );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 256:
#line 1548 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->sourceTree()->isExportAll() )
            COMPILER->raiseError(Falcon::e_export_all );
         else
            COMPILER->sourceTree()->setExportAll();
         // no effect
         (yyval.fal_stat)=0;
      ;}
    break;

  case 257:
#line 1557 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->sourceTree()->isExportAll() )
            COMPILER->raiseError(Falcon::e_export_all );
         // no effect
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 258:
#line 1564 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_export );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 259:
#line 1572 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Symbol *sym = COMPILER->addGlobalSymbol( (yyvsp[(1) - (1)].stringp) );
         sym->exported(true);
      ;}
    break;

  case 260:
#line 1577 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Symbol *sym = COMPILER->addGlobalSymbol( (yyvsp[(3) - (3)].stringp) );
         sym->exported(true);
      ;}
    break;

  case 261:
#line 1591 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // no other action:
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 262:
#line 1598 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // no other action:
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 264:
#line 1606 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_attributes ); ;}
    break;

  case 266:
#line 1610 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_attributes, "", CURRENT_LINE ); ;}
    break;

  case 268:
#line 1616 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addAttribute( (yyvsp[(1) - (1)].stringp) );
         ;}
    break;

  case 269:
#line 1620 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addAttribute( (yyvsp[(3) - (3)].stringp) );
         ;}
    break;

  case 272:
#line 1629 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->raiseError(Falcon::e_inv_attrib );
   ;}
    break;

  case 273:
#line 1640 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 274:
#line 1674 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 276:
#line 1702 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_class );
      ;}
    break;

  case 279:
#line 1710 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); ;}
    break;

  case 280:
#line 1711 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_class, "", COMPILER->tempLine() );
      ;}
    break;

  case 285:
#line 1728 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 286:
#line 1761 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_adecl) = 0; ;}
    break;

  case 287:
#line 1766 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      (yyval.fal_adecl) = (yyvsp[(3) - (5)].fal_adecl);
   ;}
    break;

  case 288:
#line 1772 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_adecl) = new Falcon::ArrayDecl(); (yyval.fal_adecl)->pushBack( (yyvsp[(1) - (1)].fal_val) ); ;}
    break;

  case 289:
#line 1773 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyvsp[(1) - (3)].fal_adecl)->pushBack( (yyvsp[(3) - (3)].fal_val) ); (yyval.fal_adecl) = (yyvsp[(1) - (3)].fal_adecl); ;}
    break;

  case 291:
#line 1779 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // the symbol must be a parameter, or we raise an error
         Falcon::Symbol *sym = COMPILER->searchLocalSymbol( (yyvsp[(1) - (1)].stringp) );
         if ( sym == 0 || sym->type() != Falcon::Symbol::tparam ) {
            sym = COMPILER->addGlobalSymbol( (yyvsp[(1) - (1)].stringp) );
         }
         (yyval.fal_val) = new Falcon::Value( sym );
      ;}
    break;

  case 292:
#line 1787 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setSelf(); ;}
    break;

  case 296:
#line 1797 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->addFunction( (yyvsp[(1) - (1)].fal_stat) );
   ;}
    break;

  case 297:
#line 1800 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 299:
#line 1823 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 300:
#line 1847 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());

         COMPILER->popContext();
         COMPILER->popContextSet();
         COMPILER->popFunction();
      ;}
    break;

  case 301:
#line 1858 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 302:
#line 1880 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 305:
#line 1910 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->raiseError(Falcon::e_syn_hasdef );
   ;}
    break;

  case 306:
#line 1917 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
         Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();

         // The symbolmay be undefined or defined; it's not our task to define it here.
         clsdef->has().pushBack( COMPILER->addGlobalSymbol( (yyvsp[(1) - (1)].stringp) ) );
      ;}
    break;

  case 307:
#line 1925 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
         Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();
         clsdef->hasnt().pushBack( COMPILER->addGlobalSymbol( (yyvsp[(2) - (2)].stringp) ) );
      ;}
    break;

  case 308:
#line 1931 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
         Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();
         clsdef->has().pushBack( COMPILER->addGlobalSymbol( (yyvsp[(3) - (3)].stringp) ) );
      ;}
    break;

  case 309:
#line 1937 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
         Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();
         clsdef->hasnt().pushBack( COMPILER->addGlobalSymbol( (yyvsp[(4) - (4)].stringp) ) );
      ;}
    break;

  case 310:
#line 1950 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 311:
#line 1990 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 313:
#line 2015 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_object );
      ;}
    break;

  case 317:
#line 2027 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->addFunction( (yyvsp[(1) - (1)].fal_stat) );
   ;}
    break;

  case 318:
#line 2030 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 320:
#line 2058 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtGlobal *glob = new Falcon::StmtGlobal( CURRENT_LINE );
         COMPILER->pushContext( glob );
      ;}
    break;

  case 321:
#line 2063 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 324:
#line 2078 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_global );
      ;}
    break;

  case 325:
#line 2085 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // we create (or retrieve) a globalized symbol
         Falcon::Symbol *sym = COMPILER->globalize( (yyvsp[(1) - (1)].stringp) );

         // then we add the symbol to the global statement (it's just for symbolic asm generation).
         Falcon::StmtGlobal *glob = static_cast<Falcon::StmtGlobal *>( COMPILER->getContext() );
         glob->addSymbol( sym );
      ;}
    break;

  case 326:
#line 2100 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = new Falcon::StmtReturn(LINE, 0); ;}
    break;

  case 327:
#line 2101 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = new Falcon::StmtReturn( LINE, (yyvsp[(2) - (3)].fal_val) ); ;}
    break;

  case 328:
#line 2102 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_return ); (yyval.fal_stat) = 0; ;}
    break;

  case 329:
#line 2112 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); ;}
    break;

  case 330:
#line 2113 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].integer) ); ;}
    break;

  case 331:
#line 2114 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].numeric) ); ;}
    break;

  case 332:
#line 2115 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].stringp) ); ;}
    break;

  case 333:
#line 2120 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 335:
#line 2138 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setSelf(); ;}
    break;

  case 336:
#line 2139 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setSender(); ;}
    break;

  case 338:
#line 2151 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_access, (yyvsp[(1) - (2)].fal_val), (yyvsp[(2) - (2)].fal_val) );
         (yyval.fal_val) = new Falcon::Value( exp );
      ;}
    break;

  case 339:
#line 2156 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_access, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
         (yyval.fal_val) = new Falcon::Value( exp );
      ;}
    break;

  case 340:
#line 2161 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_byte_access, (yyvsp[(1) - (5)].fal_val), (yyvsp[(4) - (5)].fal_val) );
         (yyval.fal_val) = new Falcon::Value( exp );
      ;}
    break;

  case 341:
#line 2167 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_obj_access, (yyvsp[(1) - (3)].fal_val), new Falcon::Value( (yyvsp[(3) - (3)].stringp) ) );
         (yyval.fal_val) = new Falcon::Value( exp );
      ;}
    break;

  case 344:
#line 2178 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_plus, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 345:
#line 2179 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_neg, (yyvsp[(2) - (2)].fal_val) ) ); ;}
    break;

  case 346:
#line 2180 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_minus, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 347:
#line 2181 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_times, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 348:
#line 2182 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_divide, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 349:
#line 2183 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_modulo, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 350:
#line 2184 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_power, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 351:
#line 2185 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_and, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 352:
#line 2186 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_or, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 353:
#line 2187 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_xor, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 354:
#line 2188 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_shift_left, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 355:
#line 2189 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_shift_right, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 356:
#line 2190 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_not, (yyvsp[(2) - (2)].fal_val) ) ); ;}
    break;

  case 357:
#line 2191 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->defineVal( (yyvsp[(2) - (4)].fal_val) ); (yyval.fal_val) =
        new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_let, (yyvsp[(2) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) ); ;}
    break;

  case 358:
#line 2193 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->defineVal( (yyvsp[(2) - (4)].fal_val) ); (yyval.fal_val) =
        new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_let, (yyvsp[(2) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) ); ;}
    break;

  case 359:
#line 2195 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_neq, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 360:
#line 2196 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_post_inc, (yyvsp[(1) - (2)].fal_val) ) ); ;}
    break;

  case 361:
#line 2197 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_pre_inc, (yyvsp[(2) - (2)].fal_val) ) ); ;}
    break;

  case 362:
#line 2198 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_post_dec, (yyvsp[(1) - (2)].fal_val) ) ); ;}
    break;

  case 363:
#line 2199 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_pre_dec, (yyvsp[(2) - (2)].fal_val) ) ); ;}
    break;

  case 364:
#line 2200 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_eq, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 365:
#line 2201 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_eq, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 366:
#line 2202 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_eq, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 367:
#line 2203 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_gt, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 368:
#line 2204 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_lt, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 369:
#line 2205 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_ge, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 370:
#line 2206 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_le, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 371:
#line 2207 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_and, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 372:
#line 2208 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_or, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 373:
#line 2209 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_not, (yyvsp[(2) - (2)].fal_val) ) ); ;}
    break;

  case 374:
#line 2210 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_has, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 375:
#line 2211 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_hasnt, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 376:
#line 2212 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_in, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 377:
#line 2213 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_notin, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 378:
#line 2214 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_provides, (yyvsp[(1) - (3)].fal_val), new Falcon::Value( (yyvsp[(3) - (3)].stringp) ) ) ); ;}
    break;

  case 379:
#line 2215 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (2)].fal_val) ); ;}
    break;

  case 380:
#line 2216 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_strexpand, (yyvsp[(2) - (2)].fal_val) ) ); ;}
    break;

  case 381:
#line 2217 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_indirect, (yyvsp[(2) - (2)].fal_val) ) ); ;}
    break;

  case 384:
#line 2220 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_obj_access, (yyvsp[(1) - (3)].fal_val), new Falcon::Value( (yyvsp[(3) - (3)].stringp) ) );
         (yyval.fal_val) = new Falcon::Value( exp );
      ;}
    break;

  case 385:
#line 2224 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_access, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
         (yyval.fal_val) = new Falcon::Value( exp );
      ;}
    break;

  case 386:
#line 2228 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_byte_access, (yyvsp[(1) - (5)].fal_val), (yyvsp[(4) - (5)].fal_val) );
         (yyval.fal_val) = new Falcon::Value( exp );
      ;}
    break;

  case 387:
#line 2232 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_access, (yyvsp[(1) - (2)].fal_val), (yyvsp[(2) - (2)].fal_val) );
         (yyval.fal_val) = new Falcon::Value( exp );
      ;}
    break;

  case 392:
#line 2240 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(3) - (5)].fal_val); ;}
    break;

  case 393:
#line 2245 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( new Falcon::Value( (Falcon::int64) 0 ) ) );
      ;}
    break;

  case 394:
#line 2248 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( (yyvsp[(2) - (4)].fal_val) ) );
      ;}
    break;

  case 395:
#line 2251 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( new Falcon::Value( (Falcon::int64) 0 ), (yyvsp[(3) - (4)].fal_val) ) );
      ;}
    break;

  case 396:
#line 2254 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( (yyvsp[(2) - (5)].fal_val), (yyvsp[(4) - (5)].fal_val) ) );
      ;}
    break;

  case 397:
#line 2261 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_funcall,
                                      (yyvsp[(1) - (6)].fal_val), new Falcon::Value( (yyvsp[(4) - (6)].fal_adecl) ) ) );
      ;}
    break;

  case 398:
#line 2267 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_funcall, (yyvsp[(1) - (4)].fal_val), 0 ) );
      ;}
    break;

  case 399:
#line 2271 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); ;}
    break;

  case 400:
#line 2272 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         delete (yyvsp[(4) - (8)].fal_adecl);
         COMPILER->raiseError(Falcon::e_syn_funcall, "", COMPILER->tempLine() );
         (yyval.fal_val) = new Falcon::Value;
      ;}
    break;

  case 405:
#line 2291 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 406:
#line 2324 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());
            (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_lambda ,
               new Falcon::Value( func->symbol() ) ) );
            COMPILER->closeFunction();
         ;}
    break;

  case 408:
#line 2334 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); ;}
    break;

  case 409:
#line 2335 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_lambda );
      ;}
    break;

  case 410:
#line 2339 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_lambda );
      ;}
    break;

  case 411:
#line 2345 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_iif, (yyvsp[(1) - (5)].fal_val), (yyvsp[(3) - (5)].fal_val), (yyvsp[(5) - (5)].fal_val) ) ); ;}
    break;

  case 412:
#line 2347 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         delete (yyvsp[(1) - (3)].fal_val);
         COMPILER->raiseError(Falcon::e_syn_iif );
         (yyval.fal_val) = new Falcon::Value;
      ;}
    break;

  case 413:
#line 2356 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {  (yyval.fal_val) = new Falcon::Value( new Falcon::ArrayDecl() ); ;}
    break;

  case 414:
#line 2358 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (4)].fal_adecl) );
      ;}
    break;

  case 415:
#line 2361 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); ;}
    break;

  case 416:
#line 2362 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_arraydecl, "", COMPILER->tempLine() );
         (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (6)].fal_adecl) );
      ;}
    break;

  case 417:
#line 2369 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {  (yyval.fal_val) = new Falcon::Value( new Falcon::DictDecl() ); ;}
    break;

  case 418:
#line 2370 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (4)].fal_ddecl) ); ;}
    break;

  case 419:
#line 2371 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); ;}
    break;

  case 420:
#line 2372 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_dictdecl, "", COMPILER->tempLine() );
         (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (6)].fal_ddecl) );
      ;}
    break;

  case 421:
#line 2379 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_adecl) = new Falcon::ArrayDecl(); (yyval.fal_adecl)->pushBack( (yyvsp[(1) - (1)].fal_val) ); ;}
    break;

  case 422:
#line 2380 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyvsp[(1) - (3)].fal_adecl)->pushBack( (yyvsp[(3) - (3)].fal_val) ); (yyval.fal_adecl) = (yyvsp[(1) - (3)].fal_adecl); ;}
    break;

  case 423:
#line 2384 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_adecl) = new Falcon::ArrayDecl(); (yyval.fal_adecl)->pushBack( (yyvsp[(1) - (1)].fal_val) ); ;}
    break;

  case 424:
#line 2385 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyvsp[(1) - (4)].fal_adecl)->pushBack( (yyvsp[(4) - (4)].fal_val) ); (yyval.fal_adecl) = (yyvsp[(1) - (4)].fal_adecl); ;}
    break;

  case 425:
#line 2389 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(1) - (1)].fal_val) );
         Falcon::ArrayDecl *ad = new Falcon::ArrayDecl();
         ad->pushBack( (yyvsp[(1) - (1)].fal_val) );
         COMPILER->defineVal( (yyvsp[(1) - (1)].fal_val) );
         (yyval.fal_adecl) = ad;
      ;}
    break;

  case 426:
#line 2396 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(3) - (3)].fal_val) );
         (yyvsp[(1) - (3)].fal_adecl)->pushBack( (yyvsp[(3) - (3)].fal_val) );
      ;}
    break;

  case 427:
#line 2403 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_ddecl) = new Falcon::DictDecl(); (yyval.fal_ddecl)->pushBack( (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ); ;}
    break;

  case 428:
#line 2404 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyvsp[(1) - (6)].fal_ddecl)->pushBack( (yyvsp[(4) - (6)].fal_val), (yyvsp[(6) - (6)].fal_val) ); (yyval.fal_ddecl) = (yyvsp[(1) - (6)].fal_ddecl); ;}
    break;


/* Line 1267 of yacc.c.  */
#line 6127 "/home/gian/Progetti/falcon/core/engine/src_parser.cpp"
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


#line 2408 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
 /* c code */


void flc_src_error (const char *s)  /* Called by yyparse on error */
{
   /* do nothing: manage it in the action */
}

/* end of src_parser.yy */


