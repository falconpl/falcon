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
     FORMIDDLE = 276,
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
     IMPORT = 300,
     DIRECTIVE = 301,
     COLON = 302,
     FUNCDECL = 303,
     STATIC = 304,
     FORDOT = 305,
     LISTPAR = 306,
     LOOP = 307,
     TRUE_TOKEN = 308,
     FALSE_TOKEN = 309,
     OUTER_STRING = 310,
     CLOSEPAR = 311,
     OPENPAR = 312,
     CLOSESQUARE = 313,
     OPENSQUARE = 314,
     DOT = 315,
     ASSIGN_POW = 316,
     ASSIGN_SHL = 317,
     ASSIGN_SHR = 318,
     ASSIGN_BXOR = 319,
     ASSIGN_BOR = 320,
     ASSIGN_BAND = 321,
     ASSIGN_MOD = 322,
     ASSIGN_DIV = 323,
     ASSIGN_MUL = 324,
     ASSIGN_SUB = 325,
     ASSIGN_ADD = 326,
     ARROW = 327,
     FOR_STEP = 328,
     OP_TO = 329,
     COMMA = 330,
     QUESTION = 331,
     OR = 332,
     AND = 333,
     NOT = 334,
     LET = 335,
     LE = 336,
     GE = 337,
     LT = 338,
     GT = 339,
     NEQ = 340,
     EEQ = 341,
     OP_EQ = 342,
     OP_ASSIGN = 343,
     PROVIDES = 344,
     OP_NOTIN = 345,
     OP_IN = 346,
     HASNT = 347,
     HAS = 348,
     DIESIS = 349,
     ATSIGN = 350,
     CAP = 351,
     VBAR = 352,
     AMPER = 353,
     MINUS = 354,
     PLUS = 355,
     PERCENT = 356,
     SLASH = 357,
     STAR = 358,
     POW = 359,
     SHR = 360,
     SHL = 361,
     BANG = 362,
     NEG = 363,
     DECREMENT = 364,
     INCREMENT = 365,
     DOLLAR = 366
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
#define FORMIDDLE 276
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
#define IMPORT 300
#define DIRECTIVE 301
#define COLON 302
#define FUNCDECL 303
#define STATIC 304
#define FORDOT 305
#define LISTPAR 306
#define LOOP 307
#define TRUE_TOKEN 308
#define FALSE_TOKEN 309
#define OUTER_STRING 310
#define CLOSEPAR 311
#define OPENPAR 312
#define CLOSESQUARE 313
#define OPENSQUARE 314
#define DOT 315
#define ASSIGN_POW 316
#define ASSIGN_SHL 317
#define ASSIGN_SHR 318
#define ASSIGN_BXOR 319
#define ASSIGN_BOR 320
#define ASSIGN_BAND 321
#define ASSIGN_MOD 322
#define ASSIGN_DIV 323
#define ASSIGN_MUL 324
#define ASSIGN_SUB 325
#define ASSIGN_ADD 326
#define ARROW 327
#define FOR_STEP 328
#define OP_TO 329
#define COMMA 330
#define QUESTION 331
#define OR 332
#define AND 333
#define NOT 334
#define LET 335
#define LE 336
#define GE 337
#define LT 338
#define GT 339
#define NEQ 340
#define EEQ 341
#define OP_EQ 342
#define OP_ASSIGN 343
#define PROVIDES 344
#define OP_NOTIN 345
#define OP_IN 346
#define HASNT 347
#define HAS 348
#define DIESIS 349
#define ATSIGN 350
#define CAP 351
#define VBAR 352
#define AMPER 353
#define MINUS 354
#define PLUS 355
#define PERCENT 356
#define SLASH 357
#define STAR 358
#define POW 359
#define SHR 360
#define SHL 361
#define BANG 362
#define NEG 363
#define DECREMENT 364
#define INCREMENT 365
#define DOLLAR 366




/* Copy the first part of user declarations.  */
#line 22 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"


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
#define  CTX_LINE  ( COMPILER->lexer()->ctxOpenLine() )
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
#line 382 "/home/gian/Progetti/falcon/core/engine/src_parser.cpp"
	YYSTYPE;
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif



/* Copy the second part of user declarations.  */


/* Line 216 of yacc.c.  */
#line 395 "/home/gian/Progetti/falcon/core/engine/src_parser.cpp"

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
#define YYLAST   6688

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  112
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  174
/* YYNRULES -- Number of rules.  */
#define YYNRULES  453
/* YYNRULES -- Number of states.  */
#define YYNSTATES  834

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   366

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
     105,   106,   107,   108,   109,   110,   111
};

#if YYDEBUG
/* YYPRHS[YYN] -- Index of the first RHS symbol of rule number YYN in
   YYRHS.  */
static const yytype_uint16 yyprhs[] =
{
       0,     0,     3,     5,     6,     9,    11,    14,    18,    20,
      22,    24,    26,    28,    30,    32,    34,    36,    38,    40,
      43,    47,    51,    55,    57,    59,    63,    67,    71,    74,
      78,    84,    87,    89,    91,    93,    95,    97,    99,   101,
     103,   105,   107,   109,   111,   113,   115,   117,   119,   121,
     123,   125,   127,   129,   133,   137,   142,   148,   153,   158,
     165,   172,   174,   176,   178,   180,   182,   184,   186,   188,
     190,   192,   194,   199,   204,   209,   214,   219,   224,   229,
     234,   239,   244,   249,   250,   256,   259,   263,   265,   269,
     273,   276,   280,   281,   288,   291,   295,   299,   303,   307,
     308,   310,   311,   315,   318,   322,   323,   328,   332,   336,
     337,   340,   343,   347,   350,   354,   358,   359,   365,   368,
     376,   386,   390,   398,   408,   412,   413,   423,   424,   432,
     438,   439,   442,   444,   446,   448,   450,   454,   458,   462,
     465,   469,   472,   476,   480,   482,   483,   490,   494,   498,
     499,   506,   510,   514,   515,   522,   526,   530,   531,   538,
     542,   546,   547,   550,   554,   556,   557,   563,   564,   570,
     571,   577,   578,   584,   585,   586,   590,   591,   593,   596,
     599,   602,   604,   608,   610,   612,   614,   618,   620,   621,
     628,   632,   636,   637,   640,   644,   646,   647,   653,   654,
     660,   661,   667,   668,   674,   676,   680,   681,   683,   685,
     691,   696,   700,   704,   705,   712,   715,   719,   720,   722,
     724,   727,   730,   733,   738,   742,   748,   752,   754,   758,
     760,   762,   766,   770,   776,   779,   785,   786,   794,   798,
     804,   805,   812,   815,   816,   818,   822,   824,   825,   826,
     832,   833,   837,   840,   844,   847,   851,   855,   859,   863,
     869,   875,   879,   885,   891,   895,   898,   902,   906,   908,
     912,   916,   920,   922,   926,   930,   934,   936,   940,   944,
     948,   953,   957,   960,   964,   967,   971,   972,   974,   978,
     981,   985,   988,   989,   998,  1002,  1005,  1006,  1010,  1011,
    1017,  1018,  1021,  1023,  1027,  1030,  1031,  1035,  1037,  1041,
    1043,  1045,  1047,  1048,  1051,  1053,  1055,  1057,  1059,  1060,
    1068,  1074,  1079,  1080,  1084,  1088,  1090,  1093,  1097,  1102,
    1103,  1112,  1115,  1118,  1119,  1122,  1124,  1126,  1128,  1130,
    1131,  1136,  1138,  1142,  1146,  1148,  1151,  1155,  1159,  1161,
    1163,  1165,  1167,  1169,  1171,  1173,  1175,  1177,  1179,  1181,
    1184,  1189,  1195,  1199,  1201,  1203,  1207,  1210,  1214,  1218,
    1222,  1226,  1230,  1234,  1238,  1242,  1246,  1250,  1253,  1258,
    1263,  1267,  1270,  1273,  1276,  1279,  1283,  1287,  1291,  1295,
    1299,  1303,  1307,  1311,  1315,  1318,  1322,  1326,  1330,  1334,
    1338,  1341,  1344,  1347,  1349,  1351,  1355,  1360,  1366,  1369,
    1371,  1373,  1375,  1377,  1381,  1385,  1390,  1395,  1401,  1406,
    1410,  1411,  1418,  1419,  1426,  1431,  1435,  1438,  1439,  1445,
    1447,  1450,  1456,  1462,  1467,  1471,  1474,  1478,  1482,  1485,
    1489,  1493,  1497,  1501,  1506,  1508,  1512,  1514,  1517,  1519,
    1523,  1525,  1529,  1533
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int16 yyrhs[] =
{
     113,     0,    -1,   114,    -1,    -1,   114,   115,    -1,   116,
      -1,     9,     3,    -1,    23,     1,     3,    -1,   118,    -1,
     228,    -1,   208,    -1,   236,    -1,   254,    -1,   119,    -1,
     223,    -1,   224,    -1,   226,    -1,   231,    -1,     4,    -1,
      99,     4,    -1,    39,     6,     3,    -1,    39,     7,     3,
      -1,    39,     1,     3,    -1,   121,    -1,     3,    -1,    48,
       1,     3,    -1,    34,     1,     3,    -1,    32,     1,     3,
      -1,     1,     3,    -1,   265,    88,   268,    -1,   120,    75,
     265,    88,   268,    -1,   268,     3,    -1,   122,    -1,   123,
      -1,   124,    -1,   136,    -1,   153,    -1,   157,    -1,   171,
      -1,   186,    -1,   140,    -1,   151,    -1,   152,    -1,   197,
      -1,   198,    -1,   207,    -1,   263,    -1,   259,    -1,   221,
      -1,   222,    -1,   162,    -1,   163,    -1,   164,    -1,    10,
     120,     3,    -1,    10,     1,     3,    -1,   267,    88,   268,
       3,    -1,   267,    88,   111,   111,     3,    -1,   267,    88,
     281,     3,    -1,   267,    88,   272,     3,    -1,   267,    75,
     284,    88,   268,     3,    -1,   267,    75,   284,    88,   281,
       3,    -1,   125,    -1,   126,    -1,   127,    -1,   128,    -1,
     129,    -1,   130,    -1,   131,    -1,   132,    -1,   133,    -1,
     134,    -1,   135,    -1,   268,    71,   268,     3,    -1,   267,
      70,   268,     3,    -1,   267,    69,   268,     3,    -1,   267,
      68,   268,     3,    -1,   267,    67,   268,     3,    -1,   267,
      61,   268,     3,    -1,   267,    66,   268,     3,    -1,   267,
      65,   268,     3,    -1,   267,    64,   268,     3,    -1,   267,
      62,   268,     3,    -1,   267,    63,   268,     3,    -1,    -1,
     138,   137,   150,     9,     3,    -1,   139,   119,    -1,    11,
     268,     3,    -1,    52,    -1,    11,     1,     3,    -1,    11,
     268,    47,    -1,    52,    47,    -1,    11,     1,    47,    -1,
      -1,   142,   141,   150,   144,     9,     3,    -1,   143,   119,
      -1,    15,   268,     3,    -1,    15,     1,     3,    -1,    15,
     268,    47,    -1,    15,     1,    47,    -1,    -1,   147,    -1,
      -1,   146,   145,   150,    -1,    16,     3,    -1,    16,     1,
       3,    -1,    -1,   149,   148,   150,   144,    -1,    17,   268,
       3,    -1,    17,     1,     3,    -1,    -1,   150,   119,    -1,
      12,     3,    -1,    12,     1,     3,    -1,    13,     3,    -1,
      13,    14,     3,    -1,    13,     1,     3,    -1,    -1,   155,
     154,   150,     9,     3,    -1,   156,   119,    -1,    18,   267,
      88,   268,    74,   268,     3,    -1,    18,   267,    88,   268,
      74,   268,    73,   268,     3,    -1,    18,     1,     3,    -1,
      18,   267,    88,   268,    74,   268,    47,    -1,    18,   267,
      88,   268,    74,   268,    73,   268,    47,    -1,    18,     1,
      47,    -1,    -1,    18,   283,    91,   268,     3,   158,   160,
       9,     3,    -1,    -1,    18,   283,    91,   268,    47,   159,
     119,    -1,    18,   283,    91,     1,     3,    -1,    -1,   161,
     160,    -1,   119,    -1,   165,    -1,   167,    -1,   169,    -1,
      50,   268,     3,    -1,    50,     1,     3,    -1,   105,   281,
       3,    -1,   105,     3,    -1,    84,   281,     3,    -1,    84,
       3,    -1,   105,     1,     3,    -1,    84,     1,     3,    -1,
      55,    -1,    -1,    19,     3,   166,   150,     9,     3,    -1,
      19,    47,   119,    -1,    19,     1,     3,    -1,    -1,    20,
       3,   168,   150,     9,     3,    -1,    20,    47,   119,    -1,
      20,     1,     3,    -1,    -1,    21,     3,   170,   150,     9,
       3,    -1,    21,    47,   119,    -1,    21,     1,     3,    -1,
      -1,   173,   172,   174,   180,     9,     3,    -1,    22,   268,
       3,    -1,    22,     1,     3,    -1,    -1,   174,   175,    -1,
     174,     1,     3,    -1,     3,    -1,    -1,    23,   184,     3,
     176,   150,    -1,    -1,    23,   184,    47,   177,   119,    -1,
      -1,    23,     1,     3,   178,   150,    -1,    -1,    23,     1,
      47,   179,   119,    -1,    -1,    -1,   182,   181,   183,    -1,
      -1,    24,    -1,    24,     1,    -1,     3,   150,    -1,    47,
     119,    -1,   185,    -1,   184,    75,   185,    -1,     8,    -1,
     117,    -1,     7,    -1,   117,    74,   117,    -1,     6,    -1,
      -1,   188,   187,   189,   180,     9,     3,    -1,    25,   268,
       3,    -1,    25,     1,     3,    -1,    -1,   189,   190,    -1,
     189,     1,     3,    -1,     3,    -1,    -1,    23,   195,     3,
     191,   150,    -1,    -1,    23,   195,    47,   192,   119,    -1,
      -1,    23,     1,     3,   193,   150,    -1,    -1,    23,     1,
      47,   194,   119,    -1,   196,    -1,   195,    75,   196,    -1,
      -1,     4,    -1,     6,    -1,    28,   281,    74,   268,     3,
      -1,    28,   281,     1,     3,    -1,    28,     1,     3,    -1,
      29,    47,   119,    -1,    -1,   200,   199,   150,   201,     9,
       3,    -1,    29,     3,    -1,    29,     1,     3,    -1,    -1,
     202,    -1,   203,    -1,   202,   203,    -1,   204,   150,    -1,
      30,     3,    -1,    30,    91,   265,     3,    -1,    30,   205,
       3,    -1,    30,   205,    91,   265,     3,    -1,    30,     1,
       3,    -1,   206,    -1,   205,    75,   206,    -1,     4,    -1,
       6,    -1,    31,   268,     3,    -1,    31,     1,     3,    -1,
     209,   216,   150,     9,     3,    -1,   211,   119,    -1,   213,
      57,   214,    56,     3,    -1,    -1,   213,    57,   214,     1,
     210,    56,     3,    -1,   213,     1,     3,    -1,   213,    57,
     214,    56,    47,    -1,    -1,   213,    57,     1,   212,    56,
      47,    -1,    48,     6,    -1,    -1,   215,    -1,   214,    75,
     215,    -1,     6,    -1,    -1,    -1,   219,   217,   150,     9,
       3,    -1,    -1,   220,   218,   119,    -1,    49,     3,    -1,
      49,     1,     3,    -1,    49,    47,    -1,    49,     1,    47,
      -1,    40,   270,     3,    -1,    40,     1,     3,    -1,    43,
     268,     3,    -1,    43,   268,    91,   268,     3,    -1,    43,
     268,    91,     1,     3,    -1,    43,     1,     3,    -1,    41,
       6,    88,   264,     3,    -1,    41,     6,    88,     1,     3,
      -1,    41,     1,     3,    -1,    44,     3,    -1,    44,   225,
       3,    -1,    44,     1,     3,    -1,     6,    -1,   225,    75,
       6,    -1,    45,   227,     3,    -1,    45,     1,     3,    -1,
       6,    -1,   225,    75,     6,    -1,    46,   229,     3,    -1,
      46,     1,     3,    -1,   230,    -1,   229,    75,   230,    -1,
       6,    87,     6,    -1,     6,    87,   117,    -1,   232,   235,
       9,     3,    -1,   233,   234,     3,    -1,    42,     3,    -1,
      42,     1,     3,    -1,    42,    47,    -1,    42,     1,    47,
      -1,    -1,     6,    -1,   234,    75,     6,    -1,   234,     3,
      -1,   235,   234,     3,    -1,     1,     3,    -1,    -1,    32,
       6,   237,   238,   247,   252,     9,     3,    -1,   239,   241,
       3,    -1,     1,     3,    -1,    -1,    57,   214,    56,    -1,
      -1,    57,   214,     1,   240,    56,    -1,    -1,    33,   242,
      -1,   243,    -1,   242,    75,   243,    -1,     6,   244,    -1,
      -1,    57,   245,    56,    -1,   246,    -1,   245,    75,   246,
      -1,   264,    -1,     6,    -1,    27,    -1,    -1,   247,   248,
      -1,     3,    -1,   208,    -1,   251,    -1,   249,    -1,    -1,
      38,     3,   250,   216,   150,     9,     3,    -1,    49,     6,
      88,   268,     3,    -1,     6,    88,   268,     3,    -1,    -1,
      93,   253,     3,    -1,    93,     1,     3,    -1,     6,    -1,
      79,     6,    -1,   253,    75,     6,    -1,   253,    75,    79,
       6,    -1,    -1,    34,     6,   255,   256,   257,   252,     9,
       3,    -1,   241,     3,    -1,     1,     3,    -1,    -1,   257,
     258,    -1,     3,    -1,   208,    -1,   251,    -1,   249,    -1,
      -1,    36,   260,   261,     3,    -1,   262,    -1,   261,    75,
     262,    -1,   261,    75,     1,    -1,     6,    -1,    35,     3,
      -1,    35,   268,     3,    -1,    35,     1,     3,    -1,     8,
      -1,    53,    -1,    54,    -1,     4,    -1,     5,    -1,     7,
      -1,     6,    -1,   265,    -1,    27,    -1,    26,    -1,   266,
      -1,   267,   269,    -1,   267,    59,   268,    58,    -1,   267,
      59,   103,   268,    58,    -1,   267,    60,     6,    -1,   264,
      -1,   267,    -1,   268,   100,   268,    -1,    99,   268,    -1,
     268,    99,   268,    -1,   268,   103,   268,    -1,   268,   102,
     268,    -1,   268,   101,   268,    -1,   268,   104,   268,    -1,
     268,    98,   268,    -1,   268,    97,   268,    -1,   268,    96,
     268,    -1,   268,   106,   268,    -1,   268,   105,   268,    -1,
     107,   268,    -1,    80,   267,    87,   268,    -1,    80,   267,
      88,   268,    -1,   268,    85,   268,    -1,   268,   110,    -1,
     110,   268,    -1,   268,   109,    -1,   109,   268,    -1,   268,
      86,   268,    -1,   268,    87,   268,    -1,   268,    88,   268,
      -1,   268,    84,   268,    -1,   268,    83,   268,    -1,   268,
      82,   268,    -1,   268,    81,   268,    -1,   268,    78,   268,
      -1,   268,    77,   268,    -1,    79,   268,    -1,   268,    93,
     268,    -1,   268,    92,   268,    -1,   268,    91,   268,    -1,
     268,    90,   268,    -1,   268,    89,     6,    -1,   111,   268,
      -1,    95,   268,    -1,    94,   268,    -1,   275,    -1,   270,
      -1,   270,    60,     6,    -1,   270,    59,   268,    58,    -1,
     270,    59,   103,   268,    58,    -1,   270,   269,    -1,   278,
      -1,   279,    -1,   280,    -1,   269,    -1,    57,   268,    56,
      -1,    59,    47,    58,    -1,    59,   268,    47,    58,    -1,
      59,    47,   268,    58,    -1,    59,   268,    47,   268,    58,
      -1,   268,    57,   281,    56,    -1,   268,    57,    56,    -1,
      -1,   268,    57,   281,     1,   271,    56,    -1,    -1,    48,
     273,   274,   216,   150,     9,    -1,    57,   214,    56,     3,
      -1,    57,   214,     1,    -1,     1,     3,    -1,    -1,    37,
     276,   277,    72,   268,    -1,   214,    -1,     1,     3,    -1,
     268,    76,   268,    47,   268,    -1,   268,    76,   268,    47,
       1,    -1,   268,    76,   268,     1,    -1,   268,    76,     1,
      -1,    59,    58,    -1,    59,   281,    58,    -1,    59,   281,
       1,    -1,    51,    58,    -1,    51,   282,    58,    -1,    51,
     282,     1,    -1,    59,    72,    58,    -1,    59,   285,    58,
      -1,    59,   285,     1,    58,    -1,   268,    -1,   281,    75,
     268,    -1,   268,    -1,   282,   268,    -1,   265,    -1,   283,
      75,   265,    -1,   267,    -1,   284,    75,   267,    -1,   268,
      72,   268,    -1,   285,    75,   268,    72,   268,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   199,   199,   202,   204,   208,   209,   210,   215,   220,
     221,   226,   231,   236,   241,   242,   243,   244,   248,   249,
     253,   259,   265,   273,   274,   277,   278,   279,   280,   285,
     290,   297,   298,   299,   300,   301,   302,   303,   304,   305,
     306,   307,   308,   309,   310,   311,   312,   313,   314,   315,
     316,   317,   318,   322,   324,   330,   334,   338,   342,   346,
     351,   361,   362,   363,   364,   365,   366,   367,   368,   369,
     370,   371,   375,   382,   389,   396,   403,   410,   417,   424,
     431,   437,   443,   451,   451,   465,   473,   474,   475,   479,
     480,   481,   485,   485,   500,   510,   511,   515,   516,   520,
     522,   523,   523,   532,   533,   538,   538,   550,   551,   554,
     556,   562,   571,   579,   589,   598,   606,   606,   620,   636,
     640,   644,   652,   656,   660,   670,   669,   694,   693,   719,
     723,   725,   729,   736,   737,   738,   742,   755,   763,   767,
     773,   779,   786,   791,   800,   810,   810,   824,   833,   837,
     837,   850,   859,   863,   863,   879,   888,   892,   892,   909,
     910,   917,   919,   920,   924,   926,   925,   936,   936,   948,
     948,   960,   960,   976,   979,   978,   991,   992,   993,   996,
     997,  1003,  1004,  1008,  1017,  1029,  1040,  1051,  1072,  1072,
    1089,  1090,  1097,  1099,  1100,  1104,  1106,  1105,  1116,  1116,
    1129,  1129,  1141,  1141,  1159,  1160,  1163,  1164,  1176,  1197,
    1201,  1206,  1214,  1221,  1220,  1239,  1240,  1243,  1245,  1249,
    1250,  1254,  1259,  1277,  1297,  1307,  1318,  1326,  1327,  1331,
    1343,  1366,  1367,  1374,  1384,  1393,  1394,  1394,  1398,  1402,
    1403,  1403,  1410,  1464,  1466,  1467,  1471,  1486,  1489,  1488,
    1500,  1499,  1514,  1515,  1519,  1520,  1529,  1533,  1541,  1548,
    1558,  1564,  1576,  1586,  1591,  1603,  1612,  1619,  1627,  1632,
    1640,  1644,  1652,  1657,  1669,  1674,  1682,  1683,  1687,  1691,
    1703,  1710,  1720,  1721,  1724,  1725,  1728,  1730,  1734,  1741,
    1742,  1743,  1755,  1754,  1813,  1816,  1822,  1824,  1825,  1825,
    1831,  1833,  1837,  1838,  1842,  1876,  1878,  1887,  1888,  1892,
    1893,  1902,  1905,  1907,  1911,  1912,  1915,  1933,  1937,  1937,
    1971,  1993,  2020,  2022,  2023,  2030,  2038,  2044,  2050,  2064,
    2063,  2127,  2128,  2134,  2136,  2140,  2141,  2144,  2163,  2172,
    2171,  2189,  2190,  2191,  2198,  2214,  2215,  2216,  2226,  2227,
    2228,  2229,  2230,  2231,  2235,  2253,  2254,  2255,  2266,  2267,
    2272,  2277,  2283,  2296,  2297,  2298,  2299,  2300,  2301,  2302,
    2303,  2304,  2305,  2306,  2307,  2308,  2309,  2310,  2311,  2313,
    2315,  2316,  2317,  2318,  2319,  2320,  2321,  2322,  2323,  2324,
    2325,  2326,  2327,  2328,  2329,  2330,  2331,  2332,  2333,  2334,
    2335,  2336,  2337,  2338,  2339,  2340,  2348,  2352,  2356,  2360,
    2361,  2362,  2363,  2364,  2369,  2372,  2375,  2378,  2384,  2390,
    2395,  2395,  2405,  2404,  2447,  2448,  2452,  2461,  2460,  2504,
    2505,  2514,  2519,  2526,  2533,  2543,  2544,  2548,  2555,  2556,
    2560,  2569,  2570,  2571,  2579,  2580,  2584,  2585,  2589,  2595,
    2602,  2608,  2615,  2616
};
#endif

#if YYDEBUG || YYERROR_VERBOSE || YYTOKEN_TABLE
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "$end", "error", "$undefined", "EOL", "INTNUM", "DBLNUM", "SYMBOL",
  "STRING", "NIL", "END", "DEF", "WHILE", "BREAK", "CONTINUE", "DROPPING",
  "IF", "ELSE", "ELIF", "FOR", "FORFIRST", "FORLAST", "FORMIDDLE",
  "SWITCH", "CASE", "DEFAULT", "SELECT", "SENDER", "SELF", "GIVE", "TRY",
  "CATCH", "RAISE", "CLASS", "FROM", "OBJECT", "RETURN", "GLOBAL",
  "LAMBDA", "INIT", "LOAD", "LAUNCH", "CONST_KW", "ATTRIBUTES", "PASS",
  "EXPORT", "IMPORT", "DIRECTIVE", "COLON", "FUNCDECL", "STATIC", "FORDOT",
  "LISTPAR", "LOOP", "TRUE_TOKEN", "FALSE_TOKEN", "OUTER_STRING",
  "CLOSEPAR", "OPENPAR", "CLOSESQUARE", "OPENSQUARE", "DOT", "ASSIGN_POW",
  "ASSIGN_SHL", "ASSIGN_SHR", "ASSIGN_BXOR", "ASSIGN_BOR", "ASSIGN_BAND",
  "ASSIGN_MOD", "ASSIGN_DIV", "ASSIGN_MUL", "ASSIGN_SUB", "ASSIGN_ADD",
  "ARROW", "FOR_STEP", "OP_TO", "COMMA", "QUESTION", "OR", "AND", "NOT",
  "LET", "LE", "GE", "LT", "GT", "NEQ", "EEQ", "OP_EQ", "OP_ASSIGN",
  "PROVIDES", "OP_NOTIN", "OP_IN", "HASNT", "HAS", "DIESIS", "ATSIGN",
  "CAP", "VBAR", "AMPER", "MINUS", "PLUS", "PERCENT", "SLASH", "STAR",
  "POW", "SHR", "SHL", "BANG", "NEG", "DECREMENT", "INCREMENT", "DOLLAR",
  "$accept", "input", "body", "line", "toplevel_statement",
  "INTNUM_WITH_MINUS", "load_statement", "statement",
  "assignment_def_list", "base_statement", "def_statement", "assignment",
  "op_assignment", "autoadd", "autosub", "automul", "autodiv", "automod",
  "autopow", "autoband", "autobor", "autobxor", "autoshl", "autoshr",
  "while_statement", "@1", "while_decl", "while_short_decl",
  "if_statement", "@2", "if_decl", "if_short_decl", "elif_or_else", "@3",
  "else_decl", "elif_statement", "@4", "elif_decl", "statement_list",
  "break_statement", "continue_statement", "for_statement", "@5",
  "for_decl", "for_decl_short", "forin_statement", "@6", "@7",
  "forin_statement_list", "forin_statement_elem", "fordot_statement",
  "self_print_statement", "outer_print_statement", "first_loop_block",
  "@8", "last_loop_block", "@9", "middle_loop_block", "@10",
  "switch_statement", "@11", "switch_decl", "case_list", "case_statement",
  "@12", "@13", "@14", "@15", "default_statement", "@16", "default_decl",
  "default_body", "case_expression_list", "case_element",
  "select_statement", "@17", "select_decl", "selcase_list",
  "selcase_statement", "@18", "@19", "@20", "@21",
  "selcase_expression_list", "selcase_element", "give_statement",
  "try_statement", "@22", "try_decl", "catch_statements", "catch_list",
  "catch_statement", "catch_decl", "catchcase_element_list",
  "catchcase_element", "raise_statement", "func_statement", "func_decl",
  "@23", "func_decl_short", "@24", "func_begin", "param_list",
  "param_symbol", "static_block", "@25", "@26", "static_decl",
  "static_short_decl", "launch_statement", "pass_statement",
  "const_statement", "export_statement", "export_symbol_list",
  "import_statement", "import_symbol_list", "directive_statement",
  "directive_pair_list", "directive_pair", "attributes_statement",
  "attributes_decl", "attributes_short_decl", "attribute_list",
  "attribute_vert_list", "class_decl", "@27", "class_def_inner",
  "class_param_list", "@28", "from_clause", "inherit_list",
  "inherit_token", "inherit_call", "inherit_param_list",
  "inherit_param_token", "class_statement_list", "class_statement",
  "init_decl", "@29", "property_decl", "has_list", "has_clause_list",
  "object_decl", "@30", "object_decl_inner", "object_statement_list",
  "object_statement", "global_statement", "@31", "global_symbol_list",
  "globalized_symbol", "return_statement", "const_atom", "atomic_symbol",
  "var_atom", "variable", "expression", "range_decl", "func_call", "@32",
  "nameless_func", "@33", "nameless_func_decl_inner", "lambda_expr", "@34",
  "lambda_expr_inner", "iif_expr", "array_decl", "dict_decl",
  "expression_list", "listpar_expression_list", "symbol_list",
  "assignment_list", "expression_pair_list", 0
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
     355,   356,   357,   358,   359,   360,   361,   362,   363,   364,
     365,   366
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint16 yyr1[] =
{
       0,   112,   113,   114,   114,   115,   115,   115,   116,   116,
     116,   116,   116,   116,   116,   116,   116,   116,   117,   117,
     118,   118,   118,   119,   119,   119,   119,   119,   119,   120,
     120,   121,   121,   121,   121,   121,   121,   121,   121,   121,
     121,   121,   121,   121,   121,   121,   121,   121,   121,   121,
     121,   121,   121,   122,   122,   123,   123,   123,   123,   123,
     123,   124,   124,   124,   124,   124,   124,   124,   124,   124,
     124,   124,   125,   126,   127,   128,   129,   130,   131,   132,
     133,   134,   135,   137,   136,   136,   138,   138,   138,   139,
     139,   139,   141,   140,   140,   142,   142,   143,   143,   144,
     144,   145,   144,   146,   146,   148,   147,   149,   149,   150,
     150,   151,   151,   152,   152,   152,   154,   153,   153,   155,
     155,   155,   156,   156,   156,   158,   157,   159,   157,   157,
     160,   160,   161,   161,   161,   161,   162,   162,   163,   163,
     163,   163,   163,   163,   164,   166,   165,   165,   165,   168,
     167,   167,   167,   170,   169,   169,   169,   172,   171,   173,
     173,   174,   174,   174,   175,   176,   175,   177,   175,   178,
     175,   179,   175,   180,   181,   180,   182,   182,   182,   183,
     183,   184,   184,   185,   185,   185,   185,   185,   187,   186,
     188,   188,   189,   189,   189,   190,   191,   190,   192,   190,
     193,   190,   194,   190,   195,   195,   196,   196,   196,   197,
     197,   197,   198,   199,   198,   200,   200,   201,   201,   202,
     202,   203,   204,   204,   204,   204,   204,   205,   205,   206,
     206,   207,   207,   208,   208,   209,   210,   209,   209,   211,
     212,   211,   213,   214,   214,   214,   215,   216,   217,   216,
     218,   216,   219,   219,   220,   220,   221,   221,   222,   222,
     222,   222,   223,   223,   223,   224,   224,   224,   225,   225,
     226,   226,   227,   227,   228,   228,   229,   229,   230,   230,
     231,   231,   232,   232,   233,   233,   234,   234,   234,   235,
     235,   235,   237,   236,   238,   238,   239,   239,   240,   239,
     241,   241,   242,   242,   243,   244,   244,   245,   245,   246,
     246,   246,   247,   247,   248,   248,   248,   248,   250,   249,
     251,   251,   252,   252,   252,   253,   253,   253,   253,   255,
     254,   256,   256,   257,   257,   258,   258,   258,   258,   260,
     259,   261,   261,   261,   262,   263,   263,   263,   264,   264,
     264,   264,   264,   264,   265,   266,   266,   266,   267,   267,
     267,   267,   267,   268,   268,   268,   268,   268,   268,   268,
     268,   268,   268,   268,   268,   268,   268,   268,   268,   268,
     268,   268,   268,   268,   268,   268,   268,   268,   268,   268,
     268,   268,   268,   268,   268,   268,   268,   268,   268,   268,
     268,   268,   268,   268,   268,   268,   268,   268,   268,   268,
     268,   268,   268,   268,   269,   269,   269,   269,   270,   270,
     271,   270,   273,   272,   274,   274,   274,   276,   275,   277,
     277,   278,   278,   278,   278,   279,   279,   279,   279,   279,
     279,   280,   280,   280,   281,   281,   282,   282,   283,   283,
     284,   284,   285,   285
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     1,     0,     2,     1,     2,     3,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     2,
       3,     3,     3,     1,     1,     3,     3,     3,     2,     3,
       5,     2,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     3,     3,     4,     5,     4,     4,     6,
       6,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     4,     4,     4,     4,     4,     4,     4,     4,
       4,     4,     4,     0,     5,     2,     3,     1,     3,     3,
       2,     3,     0,     6,     2,     3,     3,     3,     3,     0,
       1,     0,     3,     2,     3,     0,     4,     3,     3,     0,
       2,     2,     3,     2,     3,     3,     0,     5,     2,     7,
       9,     3,     7,     9,     3,     0,     9,     0,     7,     5,
       0,     2,     1,     1,     1,     1,     3,     3,     3,     2,
       3,     2,     3,     3,     1,     0,     6,     3,     3,     0,
       6,     3,     3,     0,     6,     3,     3,     0,     6,     3,
       3,     0,     2,     3,     1,     0,     5,     0,     5,     0,
       5,     0,     5,     0,     0,     3,     0,     1,     2,     2,
       2,     1,     3,     1,     1,     1,     3,     1,     0,     6,
       3,     3,     0,     2,     3,     1,     0,     5,     0,     5,
       0,     5,     0,     5,     1,     3,     0,     1,     1,     5,
       4,     3,     3,     0,     6,     2,     3,     0,     1,     1,
       2,     2,     2,     4,     3,     5,     3,     1,     3,     1,
       1,     3,     3,     5,     2,     5,     0,     7,     3,     5,
       0,     6,     2,     0,     1,     3,     1,     0,     0,     5,
       0,     3,     2,     3,     2,     3,     3,     3,     3,     5,
       5,     3,     5,     5,     3,     2,     3,     3,     1,     3,
       3,     3,     1,     3,     3,     3,     1,     3,     3,     3,
       4,     3,     2,     3,     2,     3,     0,     1,     3,     2,
       3,     2,     0,     8,     3,     2,     0,     3,     0,     5,
       0,     2,     1,     3,     2,     0,     3,     1,     3,     1,
       1,     1,     0,     2,     1,     1,     1,     1,     0,     7,
       5,     4,     0,     3,     3,     1,     2,     3,     4,     0,
       8,     2,     2,     0,     2,     1,     1,     1,     1,     0,
       4,     1,     3,     3,     1,     2,     3,     3,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     2,
       4,     5,     3,     1,     1,     3,     2,     3,     3,     3,
       3,     3,     3,     3,     3,     3,     3,     2,     4,     4,
       3,     2,     2,     2,     2,     3,     3,     3,     3,     3,
       3,     3,     3,     3,     2,     3,     3,     3,     3,     3,
       2,     2,     2,     1,     1,     3,     4,     5,     2,     1,
       1,     1,     1,     3,     3,     4,     4,     5,     4,     3,
       0,     6,     0,     6,     4,     3,     2,     0,     5,     1,
       2,     5,     5,     4,     3,     2,     3,     3,     2,     3,
       3,     3,     3,     4,     1,     3,     1,     2,     1,     3,
       1,     3,     3,     5
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint16 yydefact[] =
{
       3,     0,     0,     1,     0,    24,   351,   352,   354,   353,
     348,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   357,   356,     0,     0,     0,     0,     0,     0,   339,
     427,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    87,   349,   350,   144,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     4,
       5,     8,    13,    23,    32,    33,    34,    61,    62,    63,
      64,    65,    66,    67,    68,    69,    70,    71,    35,    83,
       0,    40,    92,     0,    41,    42,    36,   116,     0,    37,
      50,    51,    52,    38,   157,    39,   188,    43,    44,   213,
      45,    10,   247,     0,     0,    48,    49,    14,    15,    16,
       9,    17,     0,   286,    11,    12,    47,    46,   363,   355,
     358,   364,     0,   412,   404,   403,   409,   410,   411,    28,
       6,     0,     0,     0,     0,   364,     0,     0,   111,     0,
     113,     0,     0,     0,     0,   355,     0,     0,     0,     0,
       0,     0,     0,     0,   444,     0,     0,   215,     0,     0,
       0,     0,   292,     0,   329,     0,   345,     0,     0,     0,
       0,     0,     0,     0,     0,   404,     0,     0,     0,   282,
     284,     0,     0,     0,   265,   268,     0,     0,   268,     0,
       0,     0,     0,     0,   276,     0,   242,     0,     0,   438,
     446,     0,    90,     0,     0,   435,     0,   444,     0,     0,
     394,     0,     0,   141,     0,   402,   401,   366,     0,   139,
       0,   377,   384,   382,   400,   109,     0,     0,     0,    85,
     109,    94,   109,   118,   161,   192,   109,     0,   109,   248,
     250,   234,     0,     0,     0,   287,     0,   286,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   359,    31,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   383,   381,     0,     0,   408,    54,
      53,     0,     0,    88,    91,    86,    89,   112,   115,   114,
      96,    98,    95,    97,   121,   124,     0,     0,     0,   160,
     159,     7,   191,   190,   211,     0,     0,     0,   216,   212,
     232,   231,    27,     0,    26,     0,   347,   346,   344,     0,
     341,     0,   246,   429,   244,     0,    22,    20,    21,   257,
     256,   264,     0,   283,   285,   261,   258,     0,   267,   266,
       0,   271,     0,   270,   275,     0,   274,     0,    25,   137,
     136,   440,   439,   447,   413,   414,     0,   441,     0,     0,
     437,   436,     0,   442,     0,     0,     0,   143,   140,   142,
     138,     0,     0,     0,     0,     0,     0,     0,   252,   254,
       0,   109,     0,   238,   240,     0,   291,   289,     0,     0,
       0,   281,     0,     0,   362,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   450,     0,   422,     0,   444,
       0,     0,   419,     0,     0,   434,     0,   393,   392,   391,
     390,   389,   388,   380,   385,   386,   387,   399,   398,   397,
     396,   395,   374,   373,   372,   367,   365,   370,   369,   368,
     371,   376,   375,     0,     0,   405,     0,    29,     0,   449,
       0,     0,   210,     0,   445,     0,   243,   312,   300,     0,
       0,     0,   333,   340,     0,   430,     0,     0,     0,     0,
       0,   397,   269,   269,    18,   278,     0,   279,   277,   416,
     415,     0,   452,   443,     0,   378,   379,     0,   110,     0,
       0,     0,   101,   100,   105,     0,     0,   164,     0,     0,
     162,     0,   174,     0,   195,     0,     0,   193,     0,     0,
     218,   219,   109,   253,   255,     0,     0,   251,     0,   236,
       0,   288,   280,   290,     0,   360,    77,    81,    82,    80,
      79,    78,    76,    75,    74,    73,     0,     0,     0,     0,
      55,    58,    57,   420,   418,    72,   433,     0,     0,   406,
       0,     0,   129,   125,   127,   209,   295,     0,   322,     0,
     332,   305,   301,   302,   331,   322,   343,   342,   245,   428,
     263,   262,   260,   259,    19,   417,     0,    84,     0,   103,
       0,     0,     0,   109,   109,   117,   163,     0,   187,   185,
     183,   184,     0,   181,   178,     0,     0,   194,     0,   207,
     208,     0,   204,     0,     0,   222,   229,   230,     0,     0,
     227,     0,   220,     0,   233,     0,     0,     0,   235,   239,
     361,   451,   444,     0,     0,   243,   247,    56,     0,   432,
     431,   407,    30,     0,     0,     0,   298,   297,   314,     0,
       0,     0,     0,     0,   315,   313,   317,   316,     0,   294,
       0,   304,     0,   335,   336,   338,   337,     0,   334,   453,
     104,   108,   107,    93,     0,     0,   169,   171,     0,   165,
     167,     0,   158,   109,     0,   175,   200,   202,   196,   198,
     206,   189,   226,     0,   224,     0,     0,   214,   249,   241,
       0,    59,    60,   426,     0,   109,   421,   119,   122,     0,
       0,     0,     0,   132,     0,     0,   133,   134,   135,   128,
       0,     0,   318,     0,     0,   325,     0,     0,     0,   310,
     311,     0,   307,   309,   303,     0,   106,   109,     0,   186,
     109,     0,   182,     0,   180,   109,     0,   109,     0,   205,
     223,   228,     0,   237,   425,     0,     0,     0,     0,   145,
       0,     0,   149,     0,     0,   153,     0,     0,   131,   299,
       0,   247,     0,   324,   326,   323,     0,   293,   306,     0,
     330,     0,   172,     0,   168,     0,   203,     0,   199,   225,
     424,   423,   120,   123,   148,   109,   147,   152,   109,   151,
     156,   109,   155,   126,   321,   109,     0,   327,     0,   308,
       0,     0,     0,     0,   320,   328,     0,     0,     0,     0,
     146,   150,   154,   319
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     1,     2,    59,    60,   611,    61,   508,   132,    63,
      64,    65,    66,    67,    68,    69,    70,    71,    72,    73,
      74,    75,    76,    77,    78,   225,    79,    80,    81,   230,
      82,    83,   511,   603,   512,   513,   604,   514,   391,    84,
      85,    86,   232,    87,    88,    89,   654,   655,   724,   725,
      90,    91,    92,   726,   805,   727,   808,   728,   811,    93,
     234,    94,   394,   520,   750,   751,   747,   748,   521,   616,
     522,   695,   612,   613,    95,   235,    96,   395,   527,   757,
     758,   755,   756,   621,   622,    97,    98,   236,    99,   529,
     530,   531,   532,   629,   630,   100,   101,   102,   637,   103,
     538,   104,   343,   344,   238,   401,   402,   239,   240,   105,
     106,   107,   108,   186,   109,   190,   110,   193,   194,   111,
     112,   113,   246,   247,   114,   333,   477,   478,   730,   481,
     582,   583,   671,   741,   742,   578,   665,   666,   781,   667,
     668,   737,   115,   335,   482,   585,   678,   116,   168,   339,
     340,   117,   118,   119,   120,   135,   122,   123,   124,   648,
     430,   558,   646,   125,   169,   345,   126,   127,   128,   155,
     201,   147,   426,   209
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -639
static const yytype_int16 yypact[] =
{
    -639,    51,   822,  -639,   111,  -639,  -639,  -639,  -639,  -639,
    -639,   143,   335,  3103,    59,    20,  3164,   481,  3225,    60,
    3286,  -639,  -639,  3347,   345,  3408,   409,   425,   115,  -639,
    -639,   406,  3469,   478,   346,  3530,   144,   499,   500,   508,
    3591,  5665,   186,  -639,  -639,  -639,  5967,  5541,  5967,   312,
     379,  5967,  5967,  5967,   491,  5967,  5967,  5967,  5967,  -639,
    -639,  -639,  -639,  -639,  -639,  -639,  -639,  -639,  -639,  -639,
    -639,  -639,  -639,  -639,  -639,  -639,  -639,  -639,  -639,  -639,
    3042,  -639,  -639,  3042,  -639,  -639,  -639,  -639,  3042,  -639,
    -639,  -639,  -639,  -639,  -639,  -639,  -639,  -639,  -639,  -639,
    -639,  -639,   156,  3042,    52,  -639,  -639,  -639,  -639,  -639,
    -639,  -639,   124,   306,  -639,  -639,  -639,  -639,  -639,  -639,
    -639,   823,  4132,  -639,   -31,  -639,  -639,  -639,  -639,  -639,
    -639,   327,    35,   227,    29,   301,  4192,   334,  -639,   337,
    -639,   370,   168,  4252,   194,   163,   193,   251,   385,  4422,
     408,   411,  4477,   421,  6467,    58,   424,  -639,  3042,   431,
    4514,   439,  -639,   459,  -639,   488,  -639,  4569,   422,    87,
     490,   510,   512,   513,  6467,    95,   534,   414,   210,  -639,
    -639,   537,  4606,   544,  -639,  -639,   125,   546,   548,   428,
     549,   550,   456,   126,  -639,   552,  -639,   553,  4661,  -639,
    6467,   610,  -639,  6195,  5773,  -639,   502,  6037,    38,    76,
    6578,   479,   561,  -639,   136,   351,   351,   146,   566,  -639,
     137,   146,   146,   146,   146,  -639,   571,   576,   577,  -639,
    -639,  -639,  -639,  -639,  -639,  -639,  -639,   350,  -639,  -639,
    -639,  -639,   579,   152,   588,  -639,   162,   226,   164,  5550,
     573,  5967,  5967,  5967,  5967,  5967,  5967,  5967,  5967,  5967,
    5967,   312,  5807,  -639,  -639,  5870,  5967,  3652,  5967,  5967,
    5967,  5967,  5967,  5967,  5967,  5967,  5967,  5967,   574,  5967,
    5967,  5967,  5967,  5967,  5967,  5967,  5967,  5967,  5967,  5967,
    5967,  5967,  5967,  5967,  -639,  -639,  5658,   586,  -639,  -639,
    -639,   587,  5967,  -639,  -639,  -639,  -639,  -639,  -639,  -639,
    -639,  -639,  -639,  -639,  -639,  -639,  5967,   587,  3713,  -639,
    -639,  -639,  -639,  -639,  -639,   591,  5967,  5967,  -639,  -639,
    -639,  -639,  -639,   201,  -639,   228,  -639,  -639,  -639,   189,
    -639,   592,  -639,   522,  -639,   527,  -639,  -639,  -639,  -639,
    -639,  -639,   324,  -639,  -639,  -639,  -639,  3774,  -639,  -639,
     604,  -639,   606,  -639,  -639,     8,  -639,   613,  -639,  -639,
    -639,  -639,  -639,  6467,  -639,  -639,  6232,  -639,  5904,  5967,
    -639,  -639,   562,  -639,  5967,  5967,  5967,  -639,  -639,  -639,
    -639,  1821,  1488,  1932,   580,   671,  1599,   213,  -639,  -639,
    2043,  -639,  3042,  -639,  -639,   172,  -639,  -639,   615,   621,
     239,  -639,  5967,  6097,  -639,  4698,  4753,  4790,  4845,  4882,
    4937,  4974,  5029,  5066,  5121,   301,   175,  -639,  6001,  5158,
     622,   241,  -639,   174,  5213,  -639,  3935,  6541,  6578,   266,
     266,   266,   266,   266,   266,   266,   266,  -639,   351,   351,
     351,   351,   366,   366,   420,   539,   539,   453,   453,   453,
     426,   146,   146,  5967,  6157,  -639,   542,  6467,  6269,  -639,
     629,  4312,  -639,  5250,  6467,   631,   632,  -639,   602,   636,
     640,   647,  -639,  -639,   540,  -639,   632,  5967,   648,   655,
     656,    80,  -639,   657,  -639,  -639,   658,  -639,  -639,  -639,
    -639,  6306,  6467,  -639,  6356,   266,   266,   662,  -639,   417,
    3835,   661,  -639,  -639,  -639,   663,   668,  -639,    12,   378,
    -639,   664,  -639,   672,  -639,    88,   667,  -639,    96,   669,
     649,  -639,  -639,  -639,  -639,   674,  2154,  -639,   625,  -639,
     220,  -639,  -639,  -639,  6393,  -639,  -639,  -639,  -639,  -639,
    -639,  -639,  -639,  -639,  -639,  -639,   312,  5967,    79,  4043,
    -639,  -639,  -639,  -639,  -639,  -639,  -639,  3896,  6430,  -639,
    5967,  5967,  -639,  -639,  -639,  -639,  -639,   187,   100,   681,
    -639,   628,   611,  -639,  -639,   158,  -639,  -639,  -639,  6467,
    -639,  -639,  -639,  -639,  -639,  -639,  5967,  -639,   684,  -639,
     685,  5305,   688,  -639,  -639,  -639,  -639,   233,  -639,  -639,
    -639,   619,    40,  -639,  -639,   694,   275,  -639,   396,  -639,
    -639,   123,  -639,   695,   696,  -639,  -639,  -639,   587,    33,
    -639,   698,  -639,  1710,  -639,   699,   659,   651,  -639,  -639,
    -639,   301,  5342,   242,   700,   632,   156,  -639,   652,  -639,
    6504,  -639,  6467,  4082,   933,  3042,  -639,  -639,  -639,   623,
     707,   706,   708,    78,  -639,  -639,  -639,  -639,   704,  -639,
     601,  -639,   640,  -639,  -639,  -639,  -639,   713,  -639,  6467,
    -639,  -639,  -639,  -639,  2265,  1488,  -639,  -639,    13,  -639,
    -639,    18,  -639,  -639,  3042,  -639,  -639,  -639,  -639,  -639,
     348,  -639,  -639,   712,  -639,   440,   587,  -639,  -639,  -639,
     721,  -639,  -639,  -639,   190,  -639,  -639,  -639,  -639,  5967,
     416,   434,   438,  -639,   716,   933,  -639,  -639,  -639,  -639,
     660,  5967,  -639,   638,   725,  -639,   723,   245,   727,  -639,
    -639,   -29,  -639,  -639,  -639,   728,  -639,  -639,  3042,  -639,
    -639,  3042,  -639,  2376,  -639,  -639,  3042,  -639,  3042,  -639,
    -639,  -639,   729,  -639,  -639,   730,  2487,  4372,   731,  -639,
    3042,   732,  -639,  3042,   734,  -639,  3042,   735,  -639,  -639,
    5397,   156,  5967,  -639,  -639,  -639,    25,  -639,  -639,   601,
    -639,  1044,  -639,  1155,  -639,  1266,  -639,  1377,  -639,  -639,
    -639,  -639,  -639,  -639,  -639,  -639,  -639,  -639,  -639,  -639,
    -639,  -639,  -639,  -639,  -639,  -639,  5434,  -639,   733,  -639,
    2598,  2709,  2820,  2931,  -639,  -639,   737,   738,   739,   740,
    -639,  -639,  -639,  -639
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -639,  -639,  -639,  -639,  -639,  -364,  -639,     2,  -639,  -639,
    -639,  -639,  -639,  -639,  -639,  -639,  -639,  -639,  -639,  -639,
    -639,  -639,  -639,  -639,  -639,  -639,  -639,  -639,  -639,  -639,
    -639,  -639,    62,  -639,  -639,  -639,  -639,  -639,  -182,  -639,
    -639,  -639,  -639,  -639,  -639,  -639,  -639,  -639,    19,  -639,
    -639,  -639,  -639,  -639,  -639,  -639,  -639,  -639,  -639,  -639,
    -639,  -639,  -639,  -639,  -639,  -639,  -639,  -639,   357,  -639,
    -639,  -639,  -639,    54,  -639,  -639,  -639,  -639,  -639,  -639,
    -639,  -639,  -639,  -639,    55,  -639,  -639,  -639,  -639,  -639,
    -639,   229,  -639,  -639,    56,  -639,  -251,  -639,  -639,  -639,
    -639,  -639,  -236,   271,  -638,  -639,  -639,  -639,  -639,  -639,
    -639,  -639,  -639,   726,  -639,  -639,  -639,  -639,   395,  -639,
    -639,  -639,  -103,  -639,  -639,  -639,  -639,  -639,  -639,   287,
    -639,    94,  -639,  -639,   -22,  -639,  -639,   184,  -639,   185,
     188,  -639,  -639,  -639,  -639,  -639,  -639,  -639,  -639,  -639,
     292,  -639,  -349,    -6,  -639,    -2,    17,   -80,   745,  -639,
    -639,  -639,  -639,  -639,  -639,  -639,  -639,  -639,  -639,   -45,
    -639,  -639,  -639,  -639
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -449
static const yytype_int16 yytable[] =
{
     121,   497,   208,   489,    62,   214,   133,   405,   715,   220,
     248,   145,   494,   607,   495,   146,   494,   494,   608,   609,
     610,   139,   494,   140,   608,   609,   610,   788,   296,   297,
     136,   817,   303,   143,   141,   149,   704,   152,   300,   380,
     154,   263,   160,   689,   298,   167,   789,   211,   392,   174,
     393,     3,   182,   242,   396,   263,   400,   198,   200,   325,
     137,   150,   138,   203,   207,   210,   263,   154,   215,   216,
     217,   154,   221,   222,   223,   224,   304,   382,   121,   734,
     644,   121,   229,   593,   735,   231,   121,   690,   341,   618,
     233,  -206,   619,   342,   620,   298,   381,   624,   350,   625,
     626,   121,   627,   658,   818,   241,   659,   496,   705,   243,
     301,   496,   496,   327,   129,   691,   165,   496,   166,     6,
       7,     8,     9,    10,   706,   244,   698,  -286,   359,   366,
     245,   263,   326,   327,   383,  -206,   645,   265,   660,   388,
     390,    21,    22,   815,   410,   183,   130,   184,   661,   662,
     185,   384,    30,   404,   296,   297,   121,   736,   342,  -243,
     329,   673,  -243,  -206,   659,   407,    41,   411,    43,    44,
     699,   310,    46,   539,    47,   563,   283,   284,   285,   286,
     287,   288,   289,   290,   291,   292,   293,   628,   656,   294,
     295,   764,   483,   663,    48,    49,   660,   314,   700,  -286,
     360,   367,   475,   265,  -296,   237,   661,   662,  -243,    51,
      52,   327,   327,   353,    53,   311,   533,   431,   373,   536,
     433,   376,    55,   638,    56,    57,    58,  -243,   540,   479,
     564,  -300,   245,   202,  -296,   409,   686,   408,  -448,   408,
     577,   315,   543,   657,   562,   712,   765,   486,   785,   327,
     556,   663,   249,   250,  -448,   294,   295,   354,   476,   425,
     534,   480,   486,   557,   484,   486,   413,   639,   415,   416,
     417,   418,   419,   420,   421,   422,   423,   424,   693,   429,
     687,   316,   154,   434,   436,   437,   438,   439,   440,   441,
     442,   443,   444,   445,   446,   466,   448,   449,   450,   451,
     452,   453,   454,   455,   456,   457,   458,   459,   460,   461,
     462,   469,   245,   464,   408,   302,   327,   327,     8,   467,
     786,   743,   694,   265,   749,   488,   317,   664,     6,     7,
     299,     9,    10,   468,   674,   471,   131,   307,    21,    22,
     308,     8,   318,   473,   474,   263,   156,   178,   157,   179,
     633,   397,   619,   398,   620,   278,   279,   280,   281,   282,
     249,   250,   283,   284,   285,   286,   287,   288,   289,   290,
     291,   292,   293,   309,   491,   294,   295,    43,    44,   614,
     212,  -177,   213,     6,     7,     8,     9,    10,   319,   121,
     121,   121,   158,   180,   121,   501,   502,   399,   121,   696,
     121,   504,   505,   506,   537,    21,    22,   170,   265,   714,
     161,   321,   171,   172,   322,   162,    30,   768,   598,   769,
     599,   684,   685,   265,   324,  -177,   163,   328,   338,   544,
      41,   164,    43,    44,   330,   771,    46,   772,    47,   774,
     743,   775,   332,   697,   626,   224,   627,   283,   284,   285,
     286,   287,   288,   289,   290,   291,   292,   293,    48,    49,
     294,   295,   334,   770,   285,   286,   287,   288,   289,   290,
     291,   292,   293,    51,    52,   294,   295,   265,    53,   176,
     568,   773,   144,   265,   177,   776,    55,     8,    56,    57,
      58,   336,   218,   346,   219,     6,     7,     8,     9,    10,
     187,   191,   352,   362,   589,   188,   192,    21,    22,   195,
     265,   753,   643,   347,   196,   348,   349,    21,    22,   286,
     287,   288,   289,   290,   291,   292,   293,   601,    30,   294,
     295,   292,   293,   766,   121,   294,   295,   351,   249,   250,
     355,   586,    41,   365,    43,    44,   338,   358,    46,   361,
      47,  -272,   363,   364,   641,   368,   369,   291,   292,   293,
     377,   263,   294,   295,   387,   791,   385,   386,   793,   389,
      48,    49,   161,   795,   642,   797,   224,   163,   195,   414,
     447,   516,   403,   517,   650,    51,    52,   652,   653,  -173,
      53,   406,   465,     8,   472,   485,   265,   486,    55,   487,
      56,    57,    58,   518,   519,     6,     7,   739,     9,    10,
     492,   371,   493,   679,     6,     7,     8,     9,    10,   192,
     503,   541,   703,   820,   542,   561,   821,  -176,   740,   822,
     570,   121,   572,   823,   576,   480,    21,    22,   342,   580,
     288,   289,   290,   291,   292,   293,   581,    30,   294,   295,
     584,   590,   121,   121,    43,    44,   723,   729,   591,   592,
    -273,    41,   594,    43,    44,   597,   605,    46,   372,    47,
     602,   606,   523,   615,   524,   617,   623,   634,   631,   528,
    -173,   636,   121,   121,   669,   670,   672,   680,   681,    48,
      49,   683,   121,   688,   525,   519,   754,   692,   701,   702,
     762,   707,   708,   713,    51,    52,   709,   710,   716,    53,
     732,   731,   196,   738,   733,   760,   779,    55,  -176,    56,
      57,    58,   745,   121,   763,   777,   782,   723,   783,   784,
     787,   790,   799,   800,   804,   807,   767,   810,   813,   825,
     830,   831,   832,   833,   778,   752,   121,   746,   780,   121,
     792,   121,   526,   794,   121,   759,   121,   588,   796,   632,
     798,   761,   498,   189,   121,   579,   744,   819,   121,   675,
     676,   121,   806,   677,   121,   809,   587,   175,   812,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   121,
       0,   121,     0,   121,     0,   121,     0,     0,     0,   816,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   121,   121,
     121,   121,    -2,     4,     0,     5,     6,     7,     8,     9,
      10,    11,    12,    13,    14,    15,     0,    16,     0,     0,
      17,     0,     0,     0,    18,    19,     0,    20,    21,    22,
      23,    24,     0,    25,    26,     0,    27,    28,    29,    30,
       0,    31,    32,    33,    34,    35,    36,    37,    38,     0,
      39,     0,    40,    41,    42,    43,    44,    45,     0,    46,
       0,    47,   249,   250,   251,   252,   253,   254,   255,   256,
     257,   258,   259,   260,     0,     0,     0,     0,   261,     0,
       0,    48,    49,     0,     0,     0,    50,     0,     0,     0,
       0,   262,     0,     0,     0,     0,    51,    52,     0,     0,
       0,    53,     0,     0,     0,     0,     0,    54,     0,    55,
       0,    56,    57,    58,     4,     0,     5,     6,     7,     8,
       9,    10,  -130,    12,    13,    14,    15,     0,    16,     0,
       0,    17,   720,   721,   722,    18,     0,     0,    20,    21,
      22,    23,    24,     0,    25,   226,     0,   227,    28,    29,
      30,     0,     0,    32,     0,     0,    35,     0,     0,     0,
       0,   228,     0,    40,    41,    42,    43,    44,    45,     0,
      46,     0,    47,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    48,    49,     0,     0,     0,    50,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    51,    52,     0,
       0,     0,    53,     0,     0,     0,     0,     0,    54,     0,
      55,     0,    56,    57,    58,     4,     0,     5,     6,     7,
       8,     9,    10,  -170,    12,    13,    14,    15,     0,    16,
       0,     0,    17,     0,     0,     0,    18,  -170,  -170,    20,
      21,    22,    23,    24,     0,    25,   226,     0,   227,    28,
      29,    30,     0,     0,    32,     0,     0,    35,     0,     0,
       0,  -170,   228,     0,    40,    41,    42,    43,    44,    45,
       0,    46,     0,    47,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    48,    49,     0,     0,     0,    50,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    51,    52,
       0,     0,     0,    53,     0,     0,     0,     0,     0,    54,
       0,    55,     0,    56,    57,    58,     4,     0,     5,     6,
       7,     8,     9,    10,  -166,    12,    13,    14,    15,     0,
      16,     0,     0,    17,     0,     0,     0,    18,  -166,  -166,
      20,    21,    22,    23,    24,     0,    25,   226,     0,   227,
      28,    29,    30,     0,     0,    32,     0,     0,    35,     0,
       0,     0,  -166,   228,     0,    40,    41,    42,    43,    44,
      45,     0,    46,     0,    47,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    48,    49,     0,     0,     0,    50,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    51,
      52,     0,     0,     0,    53,     0,     0,     0,     0,     0,
      54,     0,    55,     0,    56,    57,    58,     4,     0,     5,
       6,     7,     8,     9,    10,  -201,    12,    13,    14,    15,
       0,    16,     0,     0,    17,     0,     0,     0,    18,  -201,
    -201,    20,    21,    22,    23,    24,     0,    25,   226,     0,
     227,    28,    29,    30,     0,     0,    32,     0,     0,    35,
       0,     0,     0,  -201,   228,     0,    40,    41,    42,    43,
      44,    45,     0,    46,     0,    47,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    48,    49,     0,     0,     0,
      50,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      51,    52,     0,     0,     0,    53,     0,     0,     0,     0,
       0,    54,     0,    55,     0,    56,    57,    58,     4,     0,
       5,     6,     7,     8,     9,    10,  -197,    12,    13,    14,
      15,     0,    16,     0,     0,    17,     0,     0,     0,    18,
    -197,  -197,    20,    21,    22,    23,    24,     0,    25,   226,
       0,   227,    28,    29,    30,     0,     0,    32,     0,     0,
      35,     0,     0,     0,  -197,   228,     0,    40,    41,    42,
      43,    44,    45,     0,    46,     0,    47,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    48,    49,     0,     0,
       0,    50,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    51,    52,     0,     0,     0,    53,     0,     0,     0,
       0,     0,    54,     0,    55,     0,    56,    57,    58,     4,
       0,     5,     6,     7,     8,     9,    10,   -99,    12,    13,
      14,    15,     0,    16,   509,   510,    17,     0,     0,     0,
      18,     0,     0,    20,    21,    22,    23,    24,     0,    25,
     226,     0,   227,    28,    29,    30,     0,     0,    32,     0,
       0,    35,     0,     0,     0,     0,   228,     0,    40,    41,
      42,    43,    44,    45,     0,    46,     0,    47,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    48,    49,     0,
       0,     0,    50,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    51,    52,     0,     0,     0,    53,     0,     0,
       0,     0,     0,    54,     0,    55,     0,    56,    57,    58,
       4,     0,     5,     6,     7,     8,     9,    10,  -217,    12,
      13,    14,    15,     0,    16,     0,     0,    17,     0,     0,
       0,    18,     0,     0,    20,    21,    22,    23,    24,   528,
      25,   226,     0,   227,    28,    29,    30,     0,     0,    32,
       0,     0,    35,     0,     0,     0,     0,   228,     0,    40,
      41,    42,    43,    44,    45,     0,    46,     0,    47,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    48,    49,
       0,     0,     0,    50,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    51,    52,     0,     0,     0,    53,     0,
       0,     0,     0,     0,    54,     0,    55,     0,    56,    57,
      58,     4,     0,     5,     6,     7,     8,     9,    10,  -221,
      12,    13,    14,    15,     0,    16,     0,     0,    17,     0,
       0,     0,    18,     0,     0,    20,    21,    22,    23,    24,
    -221,    25,   226,     0,   227,    28,    29,    30,     0,     0,
      32,     0,     0,    35,     0,     0,     0,     0,   228,     0,
      40,    41,    42,    43,    44,    45,     0,    46,     0,    47,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    48,
      49,     0,     0,     0,    50,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    51,    52,     0,     0,     0,    53,
       0,     0,     0,     0,     0,    54,     0,    55,     0,    56,
      57,    58,     4,     0,     5,     6,     7,     8,     9,    10,
     507,    12,    13,    14,    15,     0,    16,     0,     0,    17,
       0,     0,     0,    18,     0,     0,    20,    21,    22,    23,
      24,     0,    25,   226,     0,   227,    28,    29,    30,     0,
       0,    32,     0,     0,    35,     0,     0,     0,     0,   228,
       0,    40,    41,    42,    43,    44,    45,     0,    46,     0,
      47,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      48,    49,     0,     0,     0,    50,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    51,    52,     0,     0,     0,
      53,     0,     0,     0,     0,     0,    54,     0,    55,     0,
      56,    57,    58,     4,     0,     5,     6,     7,     8,     9,
      10,   515,    12,    13,    14,    15,     0,    16,     0,     0,
      17,     0,     0,     0,    18,     0,     0,    20,    21,    22,
      23,    24,     0,    25,   226,     0,   227,    28,    29,    30,
       0,     0,    32,     0,     0,    35,     0,     0,     0,     0,
     228,     0,    40,    41,    42,    43,    44,    45,     0,    46,
       0,    47,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    48,    49,     0,     0,     0,    50,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    51,    52,     0,     0,
       0,    53,     0,     0,     0,     0,     0,    54,     0,    55,
       0,    56,    57,    58,     4,     0,     5,     6,     7,     8,
       9,    10,   535,    12,    13,    14,    15,     0,    16,     0,
       0,    17,     0,     0,     0,    18,     0,     0,    20,    21,
      22,    23,    24,     0,    25,   226,     0,   227,    28,    29,
      30,     0,     0,    32,     0,     0,    35,     0,     0,     0,
       0,   228,     0,    40,    41,    42,    43,    44,    45,     0,
      46,     0,    47,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    48,    49,     0,     0,     0,    50,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    51,    52,     0,
       0,     0,    53,     0,     0,     0,     0,     0,    54,     0,
      55,     0,    56,    57,    58,     4,     0,     5,     6,     7,
       8,     9,    10,   635,    12,    13,    14,    15,     0,    16,
       0,     0,    17,     0,     0,     0,    18,     0,     0,    20,
      21,    22,    23,    24,     0,    25,   226,     0,   227,    28,
      29,    30,     0,     0,    32,     0,     0,    35,     0,     0,
       0,     0,   228,     0,    40,    41,    42,    43,    44,    45,
       0,    46,     0,    47,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    48,    49,     0,     0,     0,    50,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    51,    52,
       0,     0,     0,    53,     0,     0,     0,     0,     0,    54,
       0,    55,     0,    56,    57,    58,     4,     0,     5,     6,
       7,     8,     9,    10,  -102,    12,    13,    14,    15,     0,
      16,     0,     0,    17,     0,     0,     0,    18,     0,     0,
      20,    21,    22,    23,    24,     0,    25,   226,     0,   227,
      28,    29,    30,     0,     0,    32,     0,     0,    35,     0,
       0,     0,     0,   228,     0,    40,    41,    42,    43,    44,
      45,     0,    46,     0,    47,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    48,    49,     0,     0,     0,    50,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    51,
      52,     0,     0,     0,    53,     0,     0,     0,     0,     0,
      54,     0,    55,     0,    56,    57,    58,     4,     0,     5,
       6,     7,     8,     9,    10,  -179,    12,    13,    14,    15,
       0,    16,     0,     0,    17,     0,     0,     0,    18,     0,
       0,    20,    21,    22,    23,    24,     0,    25,   226,     0,
     227,    28,    29,    30,     0,     0,    32,     0,     0,    35,
       0,     0,     0,     0,   228,     0,    40,    41,    42,    43,
      44,    45,     0,    46,     0,    47,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    48,    49,     0,     0,     0,
      50,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      51,    52,     0,     0,     0,    53,     0,     0,     0,     0,
       0,    54,     0,    55,     0,    56,    57,    58,     4,     0,
       5,     6,     7,     8,     9,    10,   801,    12,    13,    14,
      15,     0,    16,     0,     0,    17,     0,     0,     0,    18,
       0,     0,    20,    21,    22,    23,    24,     0,    25,   226,
       0,   227,    28,    29,    30,     0,     0,    32,     0,     0,
      35,     0,     0,     0,     0,   228,     0,    40,    41,    42,
      43,    44,    45,     0,    46,     0,    47,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    48,    49,     0,     0,
       0,    50,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    51,    52,     0,     0,     0,    53,     0,     0,     0,
       0,     0,    54,     0,    55,     0,    56,    57,    58,     4,
       0,     5,     6,     7,     8,     9,    10,   826,    12,    13,
      14,    15,     0,    16,     0,     0,    17,     0,     0,     0,
      18,     0,     0,    20,    21,    22,    23,    24,     0,    25,
     226,     0,   227,    28,    29,    30,     0,     0,    32,     0,
       0,    35,     0,     0,     0,     0,   228,     0,    40,    41,
      42,    43,    44,    45,     0,    46,     0,    47,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    48,    49,     0,
       0,     0,    50,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    51,    52,     0,     0,     0,    53,     0,     0,
       0,     0,     0,    54,     0,    55,     0,    56,    57,    58,
       4,     0,     5,     6,     7,     8,     9,    10,   827,    12,
      13,    14,    15,     0,    16,     0,     0,    17,     0,     0,
       0,    18,     0,     0,    20,    21,    22,    23,    24,     0,
      25,   226,     0,   227,    28,    29,    30,     0,     0,    32,
       0,     0,    35,     0,     0,     0,     0,   228,     0,    40,
      41,    42,    43,    44,    45,     0,    46,     0,    47,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    48,    49,
       0,     0,     0,    50,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    51,    52,     0,     0,     0,    53,     0,
       0,     0,     0,     0,    54,     0,    55,     0,    56,    57,
      58,     4,     0,     5,     6,     7,     8,     9,    10,   828,
      12,    13,    14,    15,     0,    16,     0,     0,    17,     0,
       0,     0,    18,     0,     0,    20,    21,    22,    23,    24,
       0,    25,   226,     0,   227,    28,    29,    30,     0,     0,
      32,     0,     0,    35,     0,     0,     0,     0,   228,     0,
      40,    41,    42,    43,    44,    45,     0,    46,     0,    47,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    48,
      49,     0,     0,     0,    50,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    51,    52,     0,     0,     0,    53,
       0,     0,     0,     0,     0,    54,     0,    55,     0,    56,
      57,    58,     4,     0,     5,     6,     7,     8,     9,    10,
     829,    12,    13,    14,    15,     0,    16,     0,     0,    17,
       0,     0,     0,    18,     0,     0,    20,    21,    22,    23,
      24,     0,    25,   226,     0,   227,    28,    29,    30,     0,
       0,    32,     0,     0,    35,     0,     0,     0,     0,   228,
       0,    40,    41,    42,    43,    44,    45,     0,    46,     0,
      47,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      48,    49,     0,     0,     0,    50,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    51,    52,     0,     0,     0,
      53,     0,     0,     0,     0,     0,    54,     0,    55,     0,
      56,    57,    58,     4,     0,     5,     6,     7,     8,     9,
      10,     0,    12,    13,    14,    15,     0,    16,     0,     0,
      17,     0,     0,     0,    18,     0,     0,    20,    21,    22,
      23,    24,     0,    25,   226,     0,   227,    28,    29,    30,
       0,     0,    32,     0,     0,    35,     0,     0,     0,     0,
     228,     0,    40,    41,    42,    43,    44,    45,     0,    46,
       0,    47,     0,     0,   134,     0,     0,     6,     7,     8,
       9,    10,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    48,    49,     0,     0,     0,    50,     0,     0,    21,
      22,     0,     0,     0,     0,     0,    51,    52,     0,     0,
      30,    53,     0,     0,     0,     0,     0,    54,     0,    55,
       0,    56,    57,    58,    41,     0,    43,    44,     0,     0,
      46,     0,    47,     0,     0,   142,     0,     0,     6,     7,
       8,     9,    10,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    48,    49,     0,     0,     0,     0,     0,     0,
      21,    22,     0,     0,     0,     0,     0,    51,    52,     0,
       0,    30,    53,     0,     0,     0,     0,     0,     0,     0,
      55,     0,    56,    57,    58,    41,     0,    43,    44,     0,
       0,    46,     0,    47,     0,     0,   148,     0,     0,     6,
       7,     8,     9,    10,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    48,    49,     0,     0,     0,     0,     0,
       0,    21,    22,     0,     0,     0,     0,     0,    51,    52,
       0,     0,    30,    53,     0,     0,     0,     0,     0,     0,
       0,    55,     0,    56,    57,    58,    41,     0,    43,    44,
       0,     0,    46,     0,    47,     0,     0,   151,     0,     0,
       6,     7,     8,     9,    10,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    48,    49,     0,     0,     0,     0,
       0,     0,    21,    22,     0,     0,     0,     0,     0,    51,
      52,     0,     0,    30,    53,     0,     0,     0,     0,     0,
       0,     0,    55,     0,    56,    57,    58,    41,     0,    43,
      44,     0,     0,    46,     0,    47,     0,     0,   153,     0,
       0,     6,     7,     8,     9,    10,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    48,    49,     0,     0,     0,
       0,     0,     0,    21,    22,     0,     0,     0,     0,     0,
      51,    52,     0,     0,    30,    53,     0,     0,     0,     0,
       0,     0,     0,    55,     0,    56,    57,    58,    41,     0,
      43,    44,     0,     0,    46,     0,    47,     0,     0,   159,
       0,     0,     6,     7,     8,     9,    10,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    48,    49,     0,     0,
       0,     0,     0,     0,    21,    22,     0,     0,     0,     0,
       0,    51,    52,     0,     0,    30,    53,     0,     0,     0,
       0,     0,     0,     0,    55,     0,    56,    57,    58,    41,
       0,    43,    44,     0,     0,    46,     0,    47,     0,     0,
     173,     0,     0,     6,     7,     8,     9,    10,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    48,    49,     0,
       0,     0,     0,     0,     0,    21,    22,     0,     0,     0,
       0,     0,    51,    52,     0,     0,    30,    53,     0,     0,
       0,     0,     0,     0,     0,    55,     0,    56,    57,    58,
      41,     0,    43,    44,     0,     0,    46,     0,    47,     0,
       0,   181,     0,     0,     6,     7,     8,     9,    10,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    48,    49,
       0,     0,     0,     0,     0,     0,    21,    22,     0,     0,
       0,     0,     0,    51,    52,     0,     0,    30,    53,     0,
       0,     0,     0,     0,     0,     0,    55,     0,    56,    57,
      58,    41,     0,    43,    44,     0,     0,    46,     0,    47,
       0,     0,   197,     0,     0,     6,     7,     8,     9,    10,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    48,
      49,     0,     0,     0,     0,     0,     0,    21,    22,     0,
       0,     0,     0,     0,    51,    52,     0,     0,    30,    53,
       0,     0,     0,     0,     0,     0,     0,    55,     0,    56,
      57,    58,    41,     0,    43,    44,     0,     0,    46,     0,
      47,     0,     0,   435,     0,     0,     6,     7,     8,     9,
      10,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      48,    49,     0,     0,     0,     0,     0,     0,    21,    22,
       0,     0,     0,     0,     0,    51,    52,     0,     0,    30,
      53,     0,     0,     0,     0,     0,     0,     0,    55,     0,
      56,    57,    58,    41,     0,    43,    44,     0,     0,    46,
       0,    47,     0,     0,   470,     0,     0,     6,     7,     8,
       9,    10,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    48,    49,     0,     0,     0,     0,     0,     0,    21,
      22,     0,     0,     0,     0,     0,    51,    52,     0,     0,
      30,    53,     0,     0,     0,     0,     0,     0,     0,    55,
       0,    56,    57,    58,    41,     0,    43,    44,     0,     0,
      46,     0,    47,     0,     0,   490,     0,     0,     6,     7,
       8,     9,    10,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    48,    49,     0,     0,     0,     0,     0,     0,
      21,    22,     0,     0,     0,     0,     0,    51,    52,     0,
       0,    30,    53,     0,     0,     0,     0,     0,     0,     0,
      55,     0,    56,    57,    58,    41,     0,    43,    44,     0,
       0,    46,     0,    47,     0,     0,   600,     0,     0,     6,
       7,     8,     9,    10,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    48,    49,     0,     0,     0,     0,     0,
       0,    21,    22,     0,     0,     0,     0,     0,    51,    52,
       0,     0,    30,    53,     0,     0,     0,     0,     0,     0,
       0,    55,     0,    56,    57,    58,    41,     0,    43,    44,
       0,     0,    46,     0,    47,     0,     0,   649,     0,     0,
       6,     7,     8,     9,    10,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    48,    49,     0,     0,     0,     0,
       0,     0,    21,    22,     0,     0,     0,     0,     0,    51,
      52,     0,     0,    30,    53,     0,   566,     0,     0,     0,
       0,     0,    55,     0,    56,    57,    58,    41,     0,    43,
      44,     0,     0,    46,     0,    47,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    48,    49,     0,     0,     0,
       0,     0,   567,     0,     0,     0,     0,     0,     0,     0,
      51,    52,   265,     0,     0,    53,     0,     0,     0,     0,
       0,     0,     0,    55,     0,    56,    57,    58,     0,     0,
       0,   267,   268,   269,     0,     0,   270,   271,   272,   273,
     274,   275,   276,   277,   278,   279,   280,   281,   282,     0,
       0,   283,   284,   285,   286,   287,   288,   289,   290,   291,
     292,   293,     0,     0,   294,   295,   647,     6,     7,     8,
       9,    10,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    21,
      22,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      30,     0,     0,     0,     0,   717,     0,     0,     0,     0,
       0,     0,     0,     0,    41,     0,    43,    44,     0,     0,
      46,     0,    47,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    48,    49,     0,     0,     0,     0,     0,   718,
       0,     0,     0,     0,     0,   264,     0,    51,    52,   265,
       0,     0,    53,     0,     0,     0,     0,     0,     0,     0,
      55,     0,    56,    57,    58,   719,     0,     0,   267,   268,
     269,     0,     0,   270,   271,   272,   273,   274,   275,   276,
     277,   278,   279,   280,   281,   282,     0,     0,   283,   284,
     285,   286,   287,   288,   289,   290,   291,   292,   293,   265,
       0,   294,   295,     0,     0,   305,     0,     0,     0,     0,
       0,     0,     0,   266,     0,     0,     0,     0,   267,   268,
     269,     0,     0,   270,   271,   272,   273,   274,   275,   276,
     277,   278,   279,   280,   281,   282,     0,     0,   283,   284,
     285,   286,   287,   288,   289,   290,   291,   292,   293,   306,
       0,   294,   295,     0,     0,     0,     0,     0,     0,   265,
       0,     0,     0,     0,     0,   312,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   267,   268,
     269,     0,     0,   270,   271,   272,   273,   274,   275,   276,
     277,   278,   279,   280,   281,   282,     0,     0,   283,   284,
     285,   286,   287,   288,   289,   290,   291,   292,   293,   313,
       0,   294,   295,     0,     0,     0,     0,     0,     0,   265,
       0,     0,     0,     0,     0,   573,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   267,   268,
     269,     0,     0,   270,   271,   272,   273,   274,   275,   276,
     277,   278,   279,   280,   281,   282,     0,     0,   283,   284,
     285,   286,   287,   288,   289,   290,   291,   292,   293,   574,
       0,   294,   295,     0,     0,     0,     0,     0,     0,   265,
       0,     0,     0,     0,     0,   802,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   267,   268,
     269,     0,     0,   270,   271,   272,   273,   274,   275,   276,
     277,   278,   279,   280,   281,   282,     0,     0,   283,   284,
     285,   286,   287,   288,   289,   290,   291,   292,   293,   803,
       0,   294,   295,     0,     0,   320,     0,     0,     0,   265,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   267,   268,
     269,     0,     0,   270,   271,   272,   273,   274,   275,   276,
     277,   278,   279,   280,   281,   282,     0,     0,   283,   284,
     285,   286,   287,   288,   289,   290,   291,   292,   293,   265,
     323,   294,   295,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   267,   268,
     269,     0,     0,   270,   271,   272,   273,   274,   275,   276,
     277,   278,   279,   280,   281,   282,     0,   331,   283,   284,
     285,   286,   287,   288,   289,   290,   291,   292,   293,     0,
       0,   294,   295,     0,   265,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   267,   268,   269,     0,     0,   270,   271,
     272,   273,   274,   275,   276,   277,   278,   279,   280,   281,
     282,   265,   337,   283,   284,   285,   286,   287,   288,   289,
     290,   291,   292,   293,     0,     0,   294,   295,     0,     0,
     267,   268,   269,     0,     0,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   279,   280,   281,   282,     0,   356,
     283,   284,   285,   286,   287,   288,   289,   290,   291,   292,
     293,     0,     0,   294,   295,     0,   265,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   267,   268,   269,     0,     0,
     270,   271,   272,   273,   274,   275,   276,   277,   278,   279,
     280,   281,   282,   265,   370,   283,   284,   285,   286,   287,
     288,   289,   290,   291,   292,   293,     0,     0,   294,   295,
       0,     0,   267,   268,   269,     0,     0,   270,   271,   272,
     273,   274,   275,   276,   277,   278,   279,   357,   281,   282,
       0,   546,   283,   284,   285,   286,   287,   288,   289,   290,
     291,   292,   293,     0,     0,   294,   295,     0,   265,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,   267,   268,   269,
       0,     0,   270,   271,   272,   273,   274,   275,   276,   277,
     278,   279,   280,   281,   282,   265,   547,   283,   284,   285,
     286,   287,   288,   289,   290,   291,   292,   293,     0,     0,
     294,   295,     0,     0,   267,   268,   269,     0,     0,   270,
     271,   272,   273,   274,   275,   276,   277,   278,   279,   280,
     281,   282,     0,   548,   283,   284,   285,   286,   287,   288,
     289,   290,   291,   292,   293,     0,     0,   294,   295,     0,
     265,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   267,
     268,   269,     0,     0,   270,   271,   272,   273,   274,   275,
     276,   277,   278,   279,   280,   281,   282,   265,   549,   283,
     284,   285,   286,   287,   288,   289,   290,   291,   292,   293,
       0,     0,   294,   295,     0,     0,   267,   268,   269,     0,
       0,   270,   271,   272,   273,   274,   275,   276,   277,   278,
     279,   280,   281,   282,     0,   550,   283,   284,   285,   286,
     287,   288,   289,   290,   291,   292,   293,     0,     0,   294,
     295,     0,   265,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   267,   268,   269,     0,     0,   270,   271,   272,   273,
     274,   275,   276,   277,   278,   279,   280,   281,   282,   265,
     551,   283,   284,   285,   286,   287,   288,   289,   290,   291,
     292,   293,     0,     0,   294,   295,     0,     0,   267,   268,
     269,     0,     0,   270,   271,   272,   273,   274,   275,   276,
     277,   278,   279,   280,   281,   282,     0,   552,   283,   284,
     285,   286,   287,   288,   289,   290,   291,   292,   293,     0,
       0,   294,   295,     0,   265,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   267,   268,   269,     0,     0,   270,   271,
     272,   273,   274,   275,   276,   277,   278,   279,   280,   281,
     282,   265,   553,   283,   284,   285,   286,   287,   288,   289,
     290,   291,   292,   293,     0,     0,   294,   295,     0,     0,
     267,   268,   269,     0,     0,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   279,   280,   281,   282,     0,   554,
     283,   284,   285,   286,   287,   288,   289,   290,   291,   292,
     293,     0,     0,   294,   295,     0,   265,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   267,   268,   269,     0,     0,
     270,   271,   272,   273,   274,   275,   276,   277,   278,   279,
     280,   281,   282,   265,   555,   283,   284,   285,   286,   287,
     288,   289,   290,   291,   292,   293,     0,     0,   294,   295,
       0,     0,   267,   268,   269,     0,     0,   270,   271,   272,
     273,   274,   275,   276,   277,   278,   279,   280,   281,   282,
       0,   560,   283,   284,   285,   286,   287,   288,   289,   290,
     291,   292,   293,     0,     0,   294,   295,     0,   265,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,   267,   268,   269,
       0,     0,   270,   271,   272,   273,   274,   275,   276,   277,
     278,   279,   280,   281,   282,   265,   565,   283,   284,   285,
     286,   287,   288,   289,   290,   291,   292,   293,     0,     0,
     294,   295,     0,     0,   267,   268,   269,     0,     0,   270,
     271,   272,   273,   274,   275,   276,   277,   278,   279,   280,
     281,   282,     0,   575,   283,   284,   285,   286,   287,   288,
     289,   290,   291,   292,   293,     0,     0,   294,   295,     0,
     265,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   267,
     268,   269,     0,     0,   270,   271,   272,   273,   274,   275,
     276,   277,   278,   279,   280,   281,   282,   265,   682,   283,
     284,   285,   286,   287,   288,   289,   290,   291,   292,   293,
       0,     0,   294,   295,     0,     0,   267,   268,   269,     0,
       0,   270,   271,   272,   273,   274,   275,   276,   277,   278,
     279,   280,   281,   282,     0,   711,   283,   284,   285,   286,
     287,   288,   289,   290,   291,   292,   293,     0,     0,   294,
     295,     0,   265,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   267,   268,   269,     0,     0,   270,   271,   272,   273,
     274,   275,   276,   277,   278,   279,   280,   281,   282,   265,
     814,   283,   284,   285,   286,   287,   288,   289,   290,   291,
     292,   293,     0,     0,   294,   295,     0,     0,   267,   268,
     269,     0,     0,   270,   271,   272,   273,   274,   275,   276,
     277,   278,   279,   280,   281,   282,     0,   824,   283,   284,
     285,   286,   287,   288,   289,   290,   291,   292,   293,     0,
       0,   294,   295,     0,   265,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   267,   268,   269,     0,     0,   270,   271,
     272,   273,   274,   275,   276,   277,   278,   279,   280,   281,
     282,   265,     0,   283,   284,   285,   286,   287,   288,   289,
     290,   291,   292,   293,     0,     0,   294,   295,     0,     0,
     267,   268,   269,     0,     0,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   279,   280,   281,   282,     0,     0,
     283,   284,   285,   286,   287,   288,   289,   290,   291,   292,
     293,     0,     0,   294,   295,     6,     7,     8,     9,    10,
       0,     0,     0,     0,     6,     7,     8,     9,    10,     0,
       0,     0,     0,     0,     0,     0,     0,    21,    22,     0,
       0,     0,     0,     0,     0,     0,    21,    22,    30,     0,
       0,     0,     0,     0,     0,     0,     0,    30,   204,     0,
       0,     0,    41,     0,    43,    44,     0,   204,    46,   205,
      47,    41,     0,    43,    44,     0,     0,    46,     0,    47,
       0,     0,     0,   206,     0,     0,     0,     0,     0,     0,
      48,    49,     0,     0,     0,     0,     0,     0,     0,    48,
      49,     0,     0,     0,     0,    51,    52,     0,     0,     0,
      53,     0,     0,     0,    51,    52,     0,     0,    55,    53,
      56,    57,    58,   412,     0,     0,     0,    55,     0,    56,
      57,    58,     6,     7,     8,     9,    10,     0,     0,     6,
       7,     8,     9,    10,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    21,    22,     0,     0,     0,     0,
       0,    21,    22,     0,     0,    30,     0,     0,     0,     0,
       0,     0,    30,     0,     0,   204,     0,     0,     0,    41,
       0,    43,    44,     0,     0,    46,    41,    47,    43,    44,
       0,     0,    46,   199,    47,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    48,    49,     0,
       0,     0,     0,     0,    48,    49,     0,     0,     0,     0,
       0,     0,    51,    52,     0,     0,     0,    53,     0,    51,
      52,   463,     0,     0,    53,    55,     0,    56,    57,    58,
       0,     0,    55,     0,    56,    57,    58,     6,     7,     8,
       9,    10,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    21,
      22,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      30,     6,     7,     8,     9,    10,     0,     0,     0,     0,
       0,     0,     0,     0,    41,     0,    43,    44,     0,     0,
      46,   375,    47,    21,    22,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    30,     0,     0,     0,     0,     0,
       0,     0,    48,    49,     0,   427,     0,     0,    41,     0,
      43,    44,     0,     0,    46,     0,    47,    51,    52,     0,
       0,     0,    53,     0,     6,     7,     8,     9,    10,     0,
      55,     0,    56,    57,    58,     0,    48,    49,     0,     0,
       0,     0,     0,     0,     0,     0,    21,    22,     0,     0,
       0,    51,    52,     0,     0,     0,    53,    30,     6,     7,
       8,     9,    10,     0,    55,     0,    56,    57,   428,     0,
       0,    41,     0,    43,    44,     0,   432,    46,     0,    47,
      21,    22,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    30,     0,     0,     0,     0,     0,     0,     0,    48,
      49,     0,     0,     0,     0,    41,     0,    43,    44,     0,
       0,    46,   500,    47,    51,    52,     0,     0,     0,    53,
       0,     6,     7,     8,     9,    10,     0,    55,     0,    56,
      57,    58,     0,    48,    49,     0,     0,     0,     0,     0,
       0,     0,     0,    21,    22,     0,     0,     0,    51,    52,
       0,     0,     0,    53,    30,     6,     7,     8,     9,    10,
       0,    55,     0,    56,    57,    58,     0,     0,    41,     0,
      43,    44,     0,     0,    46,     0,    47,    21,    22,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    30,     0,
       0,     0,     0,     0,     0,     0,    48,    49,     0,     0,
       0,     0,    41,     0,    43,    44,     0,     0,    46,     0,
      47,    51,    52,     0,     0,     0,    53,     0,     0,     0,
       0,     0,     0,     0,    55,     0,    56,    57,    58,     0,
      48,    49,     0,     0,   378,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   265,    51,    52,     0,     0,     0,
      53,     0,     0,     0,     0,     0,     0,     0,    55,   379,
      56,    57,   559,   267,   268,   269,     0,     0,   270,   271,
     272,   273,   274,   275,   276,   277,   278,   279,   280,   281,
     282,     0,     0,   283,   284,   285,   286,   287,   288,   289,
     290,   291,   292,   293,   378,     0,   294,   295,     0,     0,
       0,     0,     0,     0,   265,   545,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   267,   268,   269,     0,     0,   270,   271,
     272,   273,   274,   275,   276,   277,   278,   279,   280,   281,
     282,     0,     0,   283,   284,   285,   286,   287,   288,   289,
     290,   291,   292,   293,   378,     0,   294,   295,     0,     0,
       0,     0,     0,     0,   265,   569,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   267,   268,   269,     0,     0,   270,   271,
     272,   273,   274,   275,   276,   277,   278,   279,   280,   281,
     282,   374,   265,   283,   284,   285,   286,   287,   288,   289,
     290,   291,   292,   293,     0,     0,   294,   295,     0,     0,
       0,   267,   268,   269,     0,     0,   270,   271,   272,   273,
     274,   275,   276,   277,   278,   279,   280,   281,   282,   265,
     499,   283,   284,   285,   286,   287,   288,   289,   290,   291,
     292,   293,     0,     0,   294,   295,     0,     0,   267,   268,
     269,     0,     0,   270,   271,   272,   273,   274,   275,   276,
     277,   278,   279,   280,   281,   282,   265,     0,   283,   284,
     285,   286,   287,   288,   289,   290,   291,   292,   293,     0,
       0,   294,   295,   571,     0,   267,   268,   269,     0,     0,
     270,   271,   272,   273,   274,   275,   276,   277,   278,   279,
     280,   281,   282,   265,   595,   283,   284,   285,   286,   287,
     288,   289,   290,   291,   292,   293,     0,     0,   294,   295,
       0,     0,   267,   268,   269,     0,     0,   270,   271,   272,
     273,   274,   275,   276,   277,   278,   279,   280,   281,   282,
       0,     0,   283,   284,   285,   286,   287,   288,   289,   290,
     291,   292,   293,   265,     0,   294,   295,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   596,     0,
       0,     0,   267,   268,   269,     0,     0,   270,   271,   272,
     273,   274,   275,   276,   277,   278,   279,   280,   281,   282,
     265,   640,   283,   284,   285,   286,   287,   288,   289,   290,
     291,   292,   293,     0,     0,   294,   295,     0,     0,   267,
     268,   269,     0,     0,   270,   271,   272,   273,   274,   275,
     276,   277,   278,   279,   280,   281,   282,   265,   651,   283,
     284,   285,   286,   287,   288,   289,   290,   291,   292,   293,
       0,     0,   294,   295,     0,     0,   267,   268,   269,     0,
       0,   270,   271,   272,   273,   274,   275,   276,   277,   278,
     279,   280,   281,   282,   265,     0,   283,   284,   285,   286,
     287,   288,   289,   290,   291,   292,   293,     0,     0,   294,
     295,     0,     0,   267,   268,   269,     0,     0,   270,   271,
     272,   273,   274,   275,   276,   277,   278,   279,   280,   281,
     282,   265,     0,   283,   284,   285,   286,   287,   288,   289,
     290,   291,   292,   293,     0,     0,   294,   295,     0,     0,
       0,   268,   269,     0,     0,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   279,   280,   281,   282,   265,     0,
     283,   284,   285,   286,   287,   288,   289,   290,   291,   292,
     293,     0,     0,   294,   295,     0,     0,     0,     0,   269,
       0,     0,   270,   271,   272,   273,   274,   275,   276,   277,
     278,   279,   280,   281,   282,   265,     0,   283,   284,   285,
     286,   287,   288,   289,   290,   291,   292,   293,     0,     0,
     294,   295,     0,     0,     0,     0,     0,     0,     0,   270,
     271,   272,   273,   274,   275,   276,   277,   278,   279,   280,
     281,   282,     0,     0,   283,   284,   285,   286,   287,   288,
     289,   290,   291,   292,   293,     0,     0,   294,   295
};

static const yytype_int16 yycheck[] =
{
       2,   365,    47,   352,     2,    50,    12,   243,   646,    54,
     113,    17,     4,     1,     6,    17,     4,     4,     6,     7,
       8,     1,     4,     3,     6,     7,     8,    56,    59,    60,
      13,     6,     3,    16,    14,    18,     3,    20,     3,     1,
      23,   121,    25,     3,   124,    28,    75,    49,   230,    32,
     232,     0,    35,     1,   236,   135,   238,    40,    41,     1,
       1,     1,     3,    46,    47,    48,   146,    50,    51,    52,
      53,    54,    55,    56,    57,    58,    47,     1,    80,     1,
       1,    83,    80,     3,     6,    83,    88,    47,     1,     1,
      88,     3,     4,     6,     6,   175,    58,     1,     3,     3,
       4,   103,     6,     3,    79,   103,     6,    99,    75,    57,
      75,    99,    99,    75,     3,    75,     1,    99,     3,     4,
       5,     6,     7,     8,    91,     1,     3,     3,     3,     3,
       6,   211,    74,    75,    58,    47,    57,    57,    38,     3,
       3,    26,    27,   781,   247,     1,     3,     3,    48,    49,
       6,    75,    37,     1,    59,    60,   158,    79,     6,    72,
     158,     3,    75,    75,     6,     3,    51,     3,    53,    54,
      47,     3,    57,     1,    59,     1,    96,    97,    98,    99,
     100,   101,   102,   103,   104,   105,   106,    91,     1,   109,
     110,     1,     3,    93,    79,    80,    38,     3,    75,    75,
      75,    75,     1,    57,     3,    49,    48,    49,    56,    94,
      95,    75,    75,     3,    99,    47,     3,   262,   201,   401,
     265,   204,   107,     3,   109,   110,   111,    75,    56,     1,
      56,     3,     6,    47,    33,     9,     3,    75,    75,    75,
     476,    47,     3,    56,     3,     3,    56,    75,     3,    75,
      75,    93,    59,    60,    91,   109,   110,    47,    57,   261,
      47,    33,    75,    88,    75,    75,   249,    47,   251,   252,
     253,   254,   255,   256,   257,   258,   259,   260,     3,   262,
      47,    88,   265,   266,   267,   268,   269,   270,   271,   272,
     273,   274,   275,   276,   277,   301,   279,   280,   281,   282,
     283,   284,   285,   286,   287,   288,   289,   290,   291,   292,
     293,   317,     6,   296,    75,    88,    75,    75,     6,   302,
      75,   670,    47,    57,   688,     1,    75,   578,     4,     5,
       3,     7,     8,   316,   585,   318,     1,     3,    26,    27,
       3,     6,    91,   326,   327,   425,     1,     1,     3,     3,
     532,     1,     4,     3,     6,    89,    90,    91,    92,    93,
      59,    60,    96,    97,    98,    99,   100,   101,   102,   103,
     104,   105,   106,     3,   357,   109,   110,    53,    54,     1,
       1,     3,     3,     4,     5,     6,     7,     8,     3,   391,
     392,   393,    47,    47,   396,   378,   379,    47,   400,     3,
     402,   384,   385,   386,   402,    26,    27,     1,    57,   645,
       1,     3,     6,     7,     3,     6,    37,     1,     1,     3,
       3,   603,   604,    57,     3,    47,     1,     3,     6,   412,
      51,     6,    53,    54,     3,     1,    57,     3,    59,     1,
     789,     3,     3,    47,     4,   428,     6,    96,    97,    98,
      99,   100,   101,   102,   103,   104,   105,   106,    79,    80,
     109,   110,     3,    47,    98,    99,   100,   101,   102,   103,
     104,   105,   106,    94,    95,   109,   110,    57,    99,     1,
     463,    47,     1,    57,     6,    47,   107,     6,   109,   110,
     111,     3,     1,     3,     3,     4,     5,     6,     7,     8,
       1,     1,    88,    75,   487,     6,     6,    26,    27,     1,
      57,   693,   557,     3,     6,     3,     3,    26,    27,    99,
     100,   101,   102,   103,   104,   105,   106,   510,    37,   109,
     110,   105,   106,   715,   536,   109,   110,     3,    59,    60,
       3,     1,    51,    87,    53,    54,     6,     3,    57,     3,
      59,     3,     3,     3,   556,     3,     3,   104,   105,   106,
      58,   641,   109,   110,     3,   747,    87,    88,   750,     3,
      79,    80,     1,   755,   557,   757,   559,     1,     1,     6,
       6,     1,     3,     3,   567,    94,    95,   570,   571,     9,
      99,     3,     6,     6,     3,     3,    57,    75,   107,    72,
     109,   110,   111,    23,    24,     4,     5,     6,     7,     8,
       6,     1,     6,   596,     4,     5,     6,     7,     8,     6,
      58,     6,   628,   805,     3,     3,   808,    47,    27,   811,
      88,   633,     3,   815,     3,    33,    26,    27,     6,     3,
     101,   102,   103,   104,   105,   106,     6,    37,   109,   110,
       3,     3,   654,   655,    53,    54,   654,   655,     3,     3,
       3,    51,     4,    53,    54,     3,     3,    57,    58,    59,
       9,     3,     1,     9,     3,     3,     9,     3,     9,    30,
       9,    56,   684,   685,     3,    57,    75,     3,     3,    79,
      80,     3,   694,    74,    23,    24,   694,     3,     3,     3,
     706,     3,     3,     3,    94,    95,    47,    56,    56,    99,
       3,    88,     6,     9,     6,     3,    56,   107,    47,   109,
     110,   111,     9,   725,     3,     9,    88,   725,     3,     6,
       3,     3,     3,     3,     3,     3,   719,     3,     3,     6,
       3,     3,     3,     3,   725,   691,   748,   685,   731,   751,
     748,   753,   395,   751,   756,   700,   758,   486,   756,   530,
     758,   705,   367,    37,   766,   478,   672,   789,   770,   585,
     585,   773,   770,   585,   776,   773,   484,    32,   776,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   791,
      -1,   793,    -1,   795,    -1,   797,    -1,    -1,    -1,   782,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   820,   821,
     822,   823,     0,     1,    -1,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    13,    -1,    15,    -1,    -1,
      18,    -1,    -1,    -1,    22,    23,    -1,    25,    26,    27,
      28,    29,    -1,    31,    32,    -1,    34,    35,    36,    37,
      -1,    39,    40,    41,    42,    43,    44,    45,    46,    -1,
      48,    -1,    50,    51,    52,    53,    54,    55,    -1,    57,
      -1,    59,    59,    60,    61,    62,    63,    64,    65,    66,
      67,    68,    69,    70,    -1,    -1,    -1,    -1,    75,    -1,
      -1,    79,    80,    -1,    -1,    -1,    84,    -1,    -1,    -1,
      -1,    88,    -1,    -1,    -1,    -1,    94,    95,    -1,    -1,
      -1,    99,    -1,    -1,    -1,    -1,    -1,   105,    -1,   107,
      -1,   109,   110,   111,     1,    -1,     3,     4,     5,     6,
       7,     8,     9,    10,    11,    12,    13,    -1,    15,    -1,
      -1,    18,    19,    20,    21,    22,    -1,    -1,    25,    26,
      27,    28,    29,    -1,    31,    32,    -1,    34,    35,    36,
      37,    -1,    -1,    40,    -1,    -1,    43,    -1,    -1,    -1,
      -1,    48,    -1,    50,    51,    52,    53,    54,    55,    -1,
      57,    -1,    59,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    79,    80,    -1,    -1,    -1,    84,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    94,    95,    -1,
      -1,    -1,    99,    -1,    -1,    -1,    -1,    -1,   105,    -1,
     107,    -1,   109,   110,   111,     1,    -1,     3,     4,     5,
       6,     7,     8,     9,    10,    11,    12,    13,    -1,    15,
      -1,    -1,    18,    -1,    -1,    -1,    22,    23,    24,    25,
      26,    27,    28,    29,    -1,    31,    32,    -1,    34,    35,
      36,    37,    -1,    -1,    40,    -1,    -1,    43,    -1,    -1,
      -1,    47,    48,    -1,    50,    51,    52,    53,    54,    55,
      -1,    57,    -1,    59,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    79,    80,    -1,    -1,    -1,    84,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    94,    95,
      -1,    -1,    -1,    99,    -1,    -1,    -1,    -1,    -1,   105,
      -1,   107,    -1,   109,   110,   111,     1,    -1,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    -1,
      15,    -1,    -1,    18,    -1,    -1,    -1,    22,    23,    24,
      25,    26,    27,    28,    29,    -1,    31,    32,    -1,    34,
      35,    36,    37,    -1,    -1,    40,    -1,    -1,    43,    -1,
      -1,    -1,    47,    48,    -1,    50,    51,    52,    53,    54,
      55,    -1,    57,    -1,    59,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    79,    80,    -1,    -1,    -1,    84,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    94,
      95,    -1,    -1,    -1,    99,    -1,    -1,    -1,    -1,    -1,
     105,    -1,   107,    -1,   109,   110,   111,     1,    -1,     3,
       4,     5,     6,     7,     8,     9,    10,    11,    12,    13,
      -1,    15,    -1,    -1,    18,    -1,    -1,    -1,    22,    23,
      24,    25,    26,    27,    28,    29,    -1,    31,    32,    -1,
      34,    35,    36,    37,    -1,    -1,    40,    -1,    -1,    43,
      -1,    -1,    -1,    47,    48,    -1,    50,    51,    52,    53,
      54,    55,    -1,    57,    -1,    59,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    79,    80,    -1,    -1,    -1,
      84,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      94,    95,    -1,    -1,    -1,    99,    -1,    -1,    -1,    -1,
      -1,   105,    -1,   107,    -1,   109,   110,   111,     1,    -1,
       3,     4,     5,     6,     7,     8,     9,    10,    11,    12,
      13,    -1,    15,    -1,    -1,    18,    -1,    -1,    -1,    22,
      23,    24,    25,    26,    27,    28,    29,    -1,    31,    32,
      -1,    34,    35,    36,    37,    -1,    -1,    40,    -1,    -1,
      43,    -1,    -1,    -1,    47,    48,    -1,    50,    51,    52,
      53,    54,    55,    -1,    57,    -1,    59,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    79,    80,    -1,    -1,
      -1,    84,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    94,    95,    -1,    -1,    -1,    99,    -1,    -1,    -1,
      -1,    -1,   105,    -1,   107,    -1,   109,   110,   111,     1,
      -1,     3,     4,     5,     6,     7,     8,     9,    10,    11,
      12,    13,    -1,    15,    16,    17,    18,    -1,    -1,    -1,
      22,    -1,    -1,    25,    26,    27,    28,    29,    -1,    31,
      32,    -1,    34,    35,    36,    37,    -1,    -1,    40,    -1,
      -1,    43,    -1,    -1,    -1,    -1,    48,    -1,    50,    51,
      52,    53,    54,    55,    -1,    57,    -1,    59,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    79,    80,    -1,
      -1,    -1,    84,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    94,    95,    -1,    -1,    -1,    99,    -1,    -1,
      -1,    -1,    -1,   105,    -1,   107,    -1,   109,   110,   111,
       1,    -1,     3,     4,     5,     6,     7,     8,     9,    10,
      11,    12,    13,    -1,    15,    -1,    -1,    18,    -1,    -1,
      -1,    22,    -1,    -1,    25,    26,    27,    28,    29,    30,
      31,    32,    -1,    34,    35,    36,    37,    -1,    -1,    40,
      -1,    -1,    43,    -1,    -1,    -1,    -1,    48,    -1,    50,
      51,    52,    53,    54,    55,    -1,    57,    -1,    59,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    79,    80,
      -1,    -1,    -1,    84,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    94,    95,    -1,    -1,    -1,    99,    -1,
      -1,    -1,    -1,    -1,   105,    -1,   107,    -1,   109,   110,
     111,     1,    -1,     3,     4,     5,     6,     7,     8,     9,
      10,    11,    12,    13,    -1,    15,    -1,    -1,    18,    -1,
      -1,    -1,    22,    -1,    -1,    25,    26,    27,    28,    29,
      30,    31,    32,    -1,    34,    35,    36,    37,    -1,    -1,
      40,    -1,    -1,    43,    -1,    -1,    -1,    -1,    48,    -1,
      50,    51,    52,    53,    54,    55,    -1,    57,    -1,    59,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    79,
      80,    -1,    -1,    -1,    84,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    94,    95,    -1,    -1,    -1,    99,
      -1,    -1,    -1,    -1,    -1,   105,    -1,   107,    -1,   109,
     110,   111,     1,    -1,     3,     4,     5,     6,     7,     8,
       9,    10,    11,    12,    13,    -1,    15,    -1,    -1,    18,
      -1,    -1,    -1,    22,    -1,    -1,    25,    26,    27,    28,
      29,    -1,    31,    32,    -1,    34,    35,    36,    37,    -1,
      -1,    40,    -1,    -1,    43,    -1,    -1,    -1,    -1,    48,
      -1,    50,    51,    52,    53,    54,    55,    -1,    57,    -1,
      59,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      79,    80,    -1,    -1,    -1,    84,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    94,    95,    -1,    -1,    -1,
      99,    -1,    -1,    -1,    -1,    -1,   105,    -1,   107,    -1,
     109,   110,   111,     1,    -1,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    13,    -1,    15,    -1,    -1,
      18,    -1,    -1,    -1,    22,    -1,    -1,    25,    26,    27,
      28,    29,    -1,    31,    32,    -1,    34,    35,    36,    37,
      -1,    -1,    40,    -1,    -1,    43,    -1,    -1,    -1,    -1,
      48,    -1,    50,    51,    52,    53,    54,    55,    -1,    57,
      -1,    59,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    79,    80,    -1,    -1,    -1,    84,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    94,    95,    -1,    -1,
      -1,    99,    -1,    -1,    -1,    -1,    -1,   105,    -1,   107,
      -1,   109,   110,   111,     1,    -1,     3,     4,     5,     6,
       7,     8,     9,    10,    11,    12,    13,    -1,    15,    -1,
      -1,    18,    -1,    -1,    -1,    22,    -1,    -1,    25,    26,
      27,    28,    29,    -1,    31,    32,    -1,    34,    35,    36,
      37,    -1,    -1,    40,    -1,    -1,    43,    -1,    -1,    -1,
      -1,    48,    -1,    50,    51,    52,    53,    54,    55,    -1,
      57,    -1,    59,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    79,    80,    -1,    -1,    -1,    84,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    94,    95,    -1,
      -1,    -1,    99,    -1,    -1,    -1,    -1,    -1,   105,    -1,
     107,    -1,   109,   110,   111,     1,    -1,     3,     4,     5,
       6,     7,     8,     9,    10,    11,    12,    13,    -1,    15,
      -1,    -1,    18,    -1,    -1,    -1,    22,    -1,    -1,    25,
      26,    27,    28,    29,    -1,    31,    32,    -1,    34,    35,
      36,    37,    -1,    -1,    40,    -1,    -1,    43,    -1,    -1,
      -1,    -1,    48,    -1,    50,    51,    52,    53,    54,    55,
      -1,    57,    -1,    59,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    79,    80,    -1,    -1,    -1,    84,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    94,    95,
      -1,    -1,    -1,    99,    -1,    -1,    -1,    -1,    -1,   105,
      -1,   107,    -1,   109,   110,   111,     1,    -1,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    -1,
      15,    -1,    -1,    18,    -1,    -1,    -1,    22,    -1,    -1,
      25,    26,    27,    28,    29,    -1,    31,    32,    -1,    34,
      35,    36,    37,    -1,    -1,    40,    -1,    -1,    43,    -1,
      -1,    -1,    -1,    48,    -1,    50,    51,    52,    53,    54,
      55,    -1,    57,    -1,    59,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    79,    80,    -1,    -1,    -1,    84,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    94,
      95,    -1,    -1,    -1,    99,    -1,    -1,    -1,    -1,    -1,
     105,    -1,   107,    -1,   109,   110,   111,     1,    -1,     3,
       4,     5,     6,     7,     8,     9,    10,    11,    12,    13,
      -1,    15,    -1,    -1,    18,    -1,    -1,    -1,    22,    -1,
      -1,    25,    26,    27,    28,    29,    -1,    31,    32,    -1,
      34,    35,    36,    37,    -1,    -1,    40,    -1,    -1,    43,
      -1,    -1,    -1,    -1,    48,    -1,    50,    51,    52,    53,
      54,    55,    -1,    57,    -1,    59,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    79,    80,    -1,    -1,    -1,
      84,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      94,    95,    -1,    -1,    -1,    99,    -1,    -1,    -1,    -1,
      -1,   105,    -1,   107,    -1,   109,   110,   111,     1,    -1,
       3,     4,     5,     6,     7,     8,     9,    10,    11,    12,
      13,    -1,    15,    -1,    -1,    18,    -1,    -1,    -1,    22,
      -1,    -1,    25,    26,    27,    28,    29,    -1,    31,    32,
      -1,    34,    35,    36,    37,    -1,    -1,    40,    -1,    -1,
      43,    -1,    -1,    -1,    -1,    48,    -1,    50,    51,    52,
      53,    54,    55,    -1,    57,    -1,    59,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    79,    80,    -1,    -1,
      -1,    84,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    94,    95,    -1,    -1,    -1,    99,    -1,    -1,    -1,
      -1,    -1,   105,    -1,   107,    -1,   109,   110,   111,     1,
      -1,     3,     4,     5,     6,     7,     8,     9,    10,    11,
      12,    13,    -1,    15,    -1,    -1,    18,    -1,    -1,    -1,
      22,    -1,    -1,    25,    26,    27,    28,    29,    -1,    31,
      32,    -1,    34,    35,    36,    37,    -1,    -1,    40,    -1,
      -1,    43,    -1,    -1,    -1,    -1,    48,    -1,    50,    51,
      52,    53,    54,    55,    -1,    57,    -1,    59,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    79,    80,    -1,
      -1,    -1,    84,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    94,    95,    -1,    -1,    -1,    99,    -1,    -1,
      -1,    -1,    -1,   105,    -1,   107,    -1,   109,   110,   111,
       1,    -1,     3,     4,     5,     6,     7,     8,     9,    10,
      11,    12,    13,    -1,    15,    -1,    -1,    18,    -1,    -1,
      -1,    22,    -1,    -1,    25,    26,    27,    28,    29,    -1,
      31,    32,    -1,    34,    35,    36,    37,    -1,    -1,    40,
      -1,    -1,    43,    -1,    -1,    -1,    -1,    48,    -1,    50,
      51,    52,    53,    54,    55,    -1,    57,    -1,    59,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    79,    80,
      -1,    -1,    -1,    84,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    94,    95,    -1,    -1,    -1,    99,    -1,
      -1,    -1,    -1,    -1,   105,    -1,   107,    -1,   109,   110,
     111,     1,    -1,     3,     4,     5,     6,     7,     8,     9,
      10,    11,    12,    13,    -1,    15,    -1,    -1,    18,    -1,
      -1,    -1,    22,    -1,    -1,    25,    26,    27,    28,    29,
      -1,    31,    32,    -1,    34,    35,    36,    37,    -1,    -1,
      40,    -1,    -1,    43,    -1,    -1,    -1,    -1,    48,    -1,
      50,    51,    52,    53,    54,    55,    -1,    57,    -1,    59,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    79,
      80,    -1,    -1,    -1,    84,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    94,    95,    -1,    -1,    -1,    99,
      -1,    -1,    -1,    -1,    -1,   105,    -1,   107,    -1,   109,
     110,   111,     1,    -1,     3,     4,     5,     6,     7,     8,
       9,    10,    11,    12,    13,    -1,    15,    -1,    -1,    18,
      -1,    -1,    -1,    22,    -1,    -1,    25,    26,    27,    28,
      29,    -1,    31,    32,    -1,    34,    35,    36,    37,    -1,
      -1,    40,    -1,    -1,    43,    -1,    -1,    -1,    -1,    48,
      -1,    50,    51,    52,    53,    54,    55,    -1,    57,    -1,
      59,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      79,    80,    -1,    -1,    -1,    84,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    94,    95,    -1,    -1,    -1,
      99,    -1,    -1,    -1,    -1,    -1,   105,    -1,   107,    -1,
     109,   110,   111,     1,    -1,     3,     4,     5,     6,     7,
       8,    -1,    10,    11,    12,    13,    -1,    15,    -1,    -1,
      18,    -1,    -1,    -1,    22,    -1,    -1,    25,    26,    27,
      28,    29,    -1,    31,    32,    -1,    34,    35,    36,    37,
      -1,    -1,    40,    -1,    -1,    43,    -1,    -1,    -1,    -1,
      48,    -1,    50,    51,    52,    53,    54,    55,    -1,    57,
      -1,    59,    -1,    -1,     1,    -1,    -1,     4,     5,     6,
       7,     8,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    79,    80,    -1,    -1,    -1,    84,    -1,    -1,    26,
      27,    -1,    -1,    -1,    -1,    -1,    94,    95,    -1,    -1,
      37,    99,    -1,    -1,    -1,    -1,    -1,   105,    -1,   107,
      -1,   109,   110,   111,    51,    -1,    53,    54,    -1,    -1,
      57,    -1,    59,    -1,    -1,     1,    -1,    -1,     4,     5,
       6,     7,     8,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    79,    80,    -1,    -1,    -1,    -1,    -1,    -1,
      26,    27,    -1,    -1,    -1,    -1,    -1,    94,    95,    -1,
      -1,    37,    99,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
     107,    -1,   109,   110,   111,    51,    -1,    53,    54,    -1,
      -1,    57,    -1,    59,    -1,    -1,     1,    -1,    -1,     4,
       5,     6,     7,     8,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    79,    80,    -1,    -1,    -1,    -1,    -1,
      -1,    26,    27,    -1,    -1,    -1,    -1,    -1,    94,    95,
      -1,    -1,    37,    99,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,   107,    -1,   109,   110,   111,    51,    -1,    53,    54,
      -1,    -1,    57,    -1,    59,    -1,    -1,     1,    -1,    -1,
       4,     5,     6,     7,     8,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    79,    80,    -1,    -1,    -1,    -1,
      -1,    -1,    26,    27,    -1,    -1,    -1,    -1,    -1,    94,
      95,    -1,    -1,    37,    99,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,   107,    -1,   109,   110,   111,    51,    -1,    53,
      54,    -1,    -1,    57,    -1,    59,    -1,    -1,     1,    -1,
      -1,     4,     5,     6,     7,     8,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    79,    80,    -1,    -1,    -1,
      -1,    -1,    -1,    26,    27,    -1,    -1,    -1,    -1,    -1,
      94,    95,    -1,    -1,    37,    99,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,   107,    -1,   109,   110,   111,    51,    -1,
      53,    54,    -1,    -1,    57,    -1,    59,    -1,    -1,     1,
      -1,    -1,     4,     5,     6,     7,     8,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    79,    80,    -1,    -1,
      -1,    -1,    -1,    -1,    26,    27,    -1,    -1,    -1,    -1,
      -1,    94,    95,    -1,    -1,    37,    99,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,   107,    -1,   109,   110,   111,    51,
      -1,    53,    54,    -1,    -1,    57,    -1,    59,    -1,    -1,
       1,    -1,    -1,     4,     5,     6,     7,     8,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    79,    80,    -1,
      -1,    -1,    -1,    -1,    -1,    26,    27,    -1,    -1,    -1,
      -1,    -1,    94,    95,    -1,    -1,    37,    99,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,   107,    -1,   109,   110,   111,
      51,    -1,    53,    54,    -1,    -1,    57,    -1,    59,    -1,
      -1,     1,    -1,    -1,     4,     5,     6,     7,     8,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    79,    80,
      -1,    -1,    -1,    -1,    -1,    -1,    26,    27,    -1,    -1,
      -1,    -1,    -1,    94,    95,    -1,    -1,    37,    99,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,   107,    -1,   109,   110,
     111,    51,    -1,    53,    54,    -1,    -1,    57,    -1,    59,
      -1,    -1,     1,    -1,    -1,     4,     5,     6,     7,     8,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    79,
      80,    -1,    -1,    -1,    -1,    -1,    -1,    26,    27,    -1,
      -1,    -1,    -1,    -1,    94,    95,    -1,    -1,    37,    99,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,   107,    -1,   109,
     110,   111,    51,    -1,    53,    54,    -1,    -1,    57,    -1,
      59,    -1,    -1,     1,    -1,    -1,     4,     5,     6,     7,
       8,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      79,    80,    -1,    -1,    -1,    -1,    -1,    -1,    26,    27,
      -1,    -1,    -1,    -1,    -1,    94,    95,    -1,    -1,    37,
      99,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   107,    -1,
     109,   110,   111,    51,    -1,    53,    54,    -1,    -1,    57,
      -1,    59,    -1,    -1,     1,    -1,    -1,     4,     5,     6,
       7,     8,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    79,    80,    -1,    -1,    -1,    -1,    -1,    -1,    26,
      27,    -1,    -1,    -1,    -1,    -1,    94,    95,    -1,    -1,
      37,    99,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   107,
      -1,   109,   110,   111,    51,    -1,    53,    54,    -1,    -1,
      57,    -1,    59,    -1,    -1,     1,    -1,    -1,     4,     5,
       6,     7,     8,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    79,    80,    -1,    -1,    -1,    -1,    -1,    -1,
      26,    27,    -1,    -1,    -1,    -1,    -1,    94,    95,    -1,
      -1,    37,    99,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
     107,    -1,   109,   110,   111,    51,    -1,    53,    54,    -1,
      -1,    57,    -1,    59,    -1,    -1,     1,    -1,    -1,     4,
       5,     6,     7,     8,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    79,    80,    -1,    -1,    -1,    -1,    -1,
      -1,    26,    27,    -1,    -1,    -1,    -1,    -1,    94,    95,
      -1,    -1,    37,    99,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,   107,    -1,   109,   110,   111,    51,    -1,    53,    54,
      -1,    -1,    57,    -1,    59,    -1,    -1,     1,    -1,    -1,
       4,     5,     6,     7,     8,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    79,    80,    -1,    -1,    -1,    -1,
      -1,    -1,    26,    27,    -1,    -1,    -1,    -1,    -1,    94,
      95,    -1,    -1,    37,    99,    -1,     1,    -1,    -1,    -1,
      -1,    -1,   107,    -1,   109,   110,   111,    51,    -1,    53,
      54,    -1,    -1,    57,    -1,    59,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    79,    80,    -1,    -1,    -1,
      -1,    -1,    47,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      94,    95,    57,    -1,    -1,    99,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,   107,    -1,   109,   110,   111,    -1,    -1,
      -1,    76,    77,    78,    -1,    -1,    81,    82,    83,    84,
      85,    86,    87,    88,    89,    90,    91,    92,    93,    -1,
      -1,    96,    97,    98,    99,   100,   101,   102,   103,   104,
     105,   106,    -1,    -1,   109,   110,     3,     4,     5,     6,
       7,     8,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    26,
      27,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      37,    -1,    -1,    -1,    -1,     3,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    51,    -1,    53,    54,    -1,    -1,
      57,    -1,    59,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    79,    80,    -1,    -1,    -1,    -1,    -1,    47,
      -1,    -1,    -1,    -1,    -1,     3,    -1,    94,    95,    57,
      -1,    -1,    99,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
     107,    -1,   109,   110,   111,    73,    -1,    -1,    76,    77,
      78,    -1,    -1,    81,    82,    83,    84,    85,    86,    87,
      88,    89,    90,    91,    92,    93,    -1,    -1,    96,    97,
      98,    99,   100,   101,   102,   103,   104,   105,   106,    57,
      -1,   109,   110,    -1,    -1,     3,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    71,    -1,    -1,    -1,    -1,    76,    77,
      78,    -1,    -1,    81,    82,    83,    84,    85,    86,    87,
      88,    89,    90,    91,    92,    93,    -1,    -1,    96,    97,
      98,    99,   100,   101,   102,   103,   104,   105,   106,    47,
      -1,   109,   110,    -1,    -1,    -1,    -1,    -1,    -1,    57,
      -1,    -1,    -1,    -1,    -1,     3,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    76,    77,
      78,    -1,    -1,    81,    82,    83,    84,    85,    86,    87,
      88,    89,    90,    91,    92,    93,    -1,    -1,    96,    97,
      98,    99,   100,   101,   102,   103,   104,   105,   106,    47,
      -1,   109,   110,    -1,    -1,    -1,    -1,    -1,    -1,    57,
      -1,    -1,    -1,    -1,    -1,     3,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    76,    77,
      78,    -1,    -1,    81,    82,    83,    84,    85,    86,    87,
      88,    89,    90,    91,    92,    93,    -1,    -1,    96,    97,
      98,    99,   100,   101,   102,   103,   104,   105,   106,    47,
      -1,   109,   110,    -1,    -1,    -1,    -1,    -1,    -1,    57,
      -1,    -1,    -1,    -1,    -1,     3,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    76,    77,
      78,    -1,    -1,    81,    82,    83,    84,    85,    86,    87,
      88,    89,    90,    91,    92,    93,    -1,    -1,    96,    97,
      98,    99,   100,   101,   102,   103,   104,   105,   106,    47,
      -1,   109,   110,    -1,    -1,     3,    -1,    -1,    -1,    57,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    76,    77,
      78,    -1,    -1,    81,    82,    83,    84,    85,    86,    87,
      88,    89,    90,    91,    92,    93,    -1,    -1,    96,    97,
      98,    99,   100,   101,   102,   103,   104,   105,   106,    57,
       3,   109,   110,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    76,    77,
      78,    -1,    -1,    81,    82,    83,    84,    85,    86,    87,
      88,    89,    90,    91,    92,    93,    -1,     3,    96,    97,
      98,    99,   100,   101,   102,   103,   104,   105,   106,    -1,
      -1,   109,   110,    -1,    57,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    76,    77,    78,    -1,    -1,    81,    82,
      83,    84,    85,    86,    87,    88,    89,    90,    91,    92,
      93,    57,     3,    96,    97,    98,    99,   100,   101,   102,
     103,   104,   105,   106,    -1,    -1,   109,   110,    -1,    -1,
      76,    77,    78,    -1,    -1,    81,    82,    83,    84,    85,
      86,    87,    88,    89,    90,    91,    92,    93,    -1,     3,
      96,    97,    98,    99,   100,   101,   102,   103,   104,   105,
     106,    -1,    -1,   109,   110,    -1,    57,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    76,    77,    78,    -1,    -1,
      81,    82,    83,    84,    85,    86,    87,    88,    89,    90,
      91,    92,    93,    57,     3,    96,    97,    98,    99,   100,
     101,   102,   103,   104,   105,   106,    -1,    -1,   109,   110,
      -1,    -1,    76,    77,    78,    -1,    -1,    81,    82,    83,
      84,    85,    86,    87,    88,    89,    90,    91,    92,    93,
      -1,     3,    96,    97,    98,    99,   100,   101,   102,   103,
     104,   105,   106,    -1,    -1,   109,   110,    -1,    57,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    76,    77,    78,
      -1,    -1,    81,    82,    83,    84,    85,    86,    87,    88,
      89,    90,    91,    92,    93,    57,     3,    96,    97,    98,
      99,   100,   101,   102,   103,   104,   105,   106,    -1,    -1,
     109,   110,    -1,    -1,    76,    77,    78,    -1,    -1,    81,
      82,    83,    84,    85,    86,    87,    88,    89,    90,    91,
      92,    93,    -1,     3,    96,    97,    98,    99,   100,   101,
     102,   103,   104,   105,   106,    -1,    -1,   109,   110,    -1,
      57,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    76,
      77,    78,    -1,    -1,    81,    82,    83,    84,    85,    86,
      87,    88,    89,    90,    91,    92,    93,    57,     3,    96,
      97,    98,    99,   100,   101,   102,   103,   104,   105,   106,
      -1,    -1,   109,   110,    -1,    -1,    76,    77,    78,    -1,
      -1,    81,    82,    83,    84,    85,    86,    87,    88,    89,
      90,    91,    92,    93,    -1,     3,    96,    97,    98,    99,
     100,   101,   102,   103,   104,   105,   106,    -1,    -1,   109,
     110,    -1,    57,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    76,    77,    78,    -1,    -1,    81,    82,    83,    84,
      85,    86,    87,    88,    89,    90,    91,    92,    93,    57,
       3,    96,    97,    98,    99,   100,   101,   102,   103,   104,
     105,   106,    -1,    -1,   109,   110,    -1,    -1,    76,    77,
      78,    -1,    -1,    81,    82,    83,    84,    85,    86,    87,
      88,    89,    90,    91,    92,    93,    -1,     3,    96,    97,
      98,    99,   100,   101,   102,   103,   104,   105,   106,    -1,
      -1,   109,   110,    -1,    57,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    76,    77,    78,    -1,    -1,    81,    82,
      83,    84,    85,    86,    87,    88,    89,    90,    91,    92,
      93,    57,     3,    96,    97,    98,    99,   100,   101,   102,
     103,   104,   105,   106,    -1,    -1,   109,   110,    -1,    -1,
      76,    77,    78,    -1,    -1,    81,    82,    83,    84,    85,
      86,    87,    88,    89,    90,    91,    92,    93,    -1,     3,
      96,    97,    98,    99,   100,   101,   102,   103,   104,   105,
     106,    -1,    -1,   109,   110,    -1,    57,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    76,    77,    78,    -1,    -1,
      81,    82,    83,    84,    85,    86,    87,    88,    89,    90,
      91,    92,    93,    57,     3,    96,    97,    98,    99,   100,
     101,   102,   103,   104,   105,   106,    -1,    -1,   109,   110,
      -1,    -1,    76,    77,    78,    -1,    -1,    81,    82,    83,
      84,    85,    86,    87,    88,    89,    90,    91,    92,    93,
      -1,     3,    96,    97,    98,    99,   100,   101,   102,   103,
     104,   105,   106,    -1,    -1,   109,   110,    -1,    57,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    76,    77,    78,
      -1,    -1,    81,    82,    83,    84,    85,    86,    87,    88,
      89,    90,    91,    92,    93,    57,     3,    96,    97,    98,
      99,   100,   101,   102,   103,   104,   105,   106,    -1,    -1,
     109,   110,    -1,    -1,    76,    77,    78,    -1,    -1,    81,
      82,    83,    84,    85,    86,    87,    88,    89,    90,    91,
      92,    93,    -1,     3,    96,    97,    98,    99,   100,   101,
     102,   103,   104,   105,   106,    -1,    -1,   109,   110,    -1,
      57,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    76,
      77,    78,    -1,    -1,    81,    82,    83,    84,    85,    86,
      87,    88,    89,    90,    91,    92,    93,    57,     3,    96,
      97,    98,    99,   100,   101,   102,   103,   104,   105,   106,
      -1,    -1,   109,   110,    -1,    -1,    76,    77,    78,    -1,
      -1,    81,    82,    83,    84,    85,    86,    87,    88,    89,
      90,    91,    92,    93,    -1,     3,    96,    97,    98,    99,
     100,   101,   102,   103,   104,   105,   106,    -1,    -1,   109,
     110,    -1,    57,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    76,    77,    78,    -1,    -1,    81,    82,    83,    84,
      85,    86,    87,    88,    89,    90,    91,    92,    93,    57,
       3,    96,    97,    98,    99,   100,   101,   102,   103,   104,
     105,   106,    -1,    -1,   109,   110,    -1,    -1,    76,    77,
      78,    -1,    -1,    81,    82,    83,    84,    85,    86,    87,
      88,    89,    90,    91,    92,    93,    -1,     3,    96,    97,
      98,    99,   100,   101,   102,   103,   104,   105,   106,    -1,
      -1,   109,   110,    -1,    57,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    76,    77,    78,    -1,    -1,    81,    82,
      83,    84,    85,    86,    87,    88,    89,    90,    91,    92,
      93,    57,    -1,    96,    97,    98,    99,   100,   101,   102,
     103,   104,   105,   106,    -1,    -1,   109,   110,    -1,    -1,
      76,    77,    78,    -1,    -1,    81,    82,    83,    84,    85,
      86,    87,    88,    89,    90,    91,    92,    93,    -1,    -1,
      96,    97,    98,    99,   100,   101,   102,   103,   104,   105,
     106,    -1,    -1,   109,   110,     4,     5,     6,     7,     8,
      -1,    -1,    -1,    -1,     4,     5,     6,     7,     8,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    26,    27,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    26,    27,    37,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    37,    47,    -1,
      -1,    -1,    51,    -1,    53,    54,    -1,    47,    57,    58,
      59,    51,    -1,    53,    54,    -1,    -1,    57,    -1,    59,
      -1,    -1,    -1,    72,    -1,    -1,    -1,    -1,    -1,    -1,
      79,    80,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    79,
      80,    -1,    -1,    -1,    -1,    94,    95,    -1,    -1,    -1,
      99,    -1,    -1,    -1,    94,    95,    -1,    -1,   107,    99,
     109,   110,   111,   103,    -1,    -1,    -1,   107,    -1,   109,
     110,   111,     4,     5,     6,     7,     8,    -1,    -1,     4,
       5,     6,     7,     8,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    26,    27,    -1,    -1,    -1,    -1,
      -1,    26,    27,    -1,    -1,    37,    -1,    -1,    -1,    -1,
      -1,    -1,    37,    -1,    -1,    47,    -1,    -1,    -1,    51,
      -1,    53,    54,    -1,    -1,    57,    51,    59,    53,    54,
      -1,    -1,    57,    58,    59,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    79,    80,    -1,
      -1,    -1,    -1,    -1,    79,    80,    -1,    -1,    -1,    -1,
      -1,    -1,    94,    95,    -1,    -1,    -1,    99,    -1,    94,
      95,   103,    -1,    -1,    99,   107,    -1,   109,   110,   111,
      -1,    -1,   107,    -1,   109,   110,   111,     4,     5,     6,
       7,     8,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    26,
      27,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      37,     4,     5,     6,     7,     8,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    51,    -1,    53,    54,    -1,    -1,
      57,    58,    59,    26,    27,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    37,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    79,    80,    -1,    48,    -1,    -1,    51,    -1,
      53,    54,    -1,    -1,    57,    -1,    59,    94,    95,    -1,
      -1,    -1,    99,    -1,     4,     5,     6,     7,     8,    -1,
     107,    -1,   109,   110,   111,    -1,    79,    80,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    26,    27,    -1,    -1,
      -1,    94,    95,    -1,    -1,    -1,    99,    37,     4,     5,
       6,     7,     8,    -1,   107,    -1,   109,   110,   111,    -1,
      -1,    51,    -1,    53,    54,    -1,    56,    57,    -1,    59,
      26,    27,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    37,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    79,
      80,    -1,    -1,    -1,    -1,    51,    -1,    53,    54,    -1,
      -1,    57,    58,    59,    94,    95,    -1,    -1,    -1,    99,
      -1,     4,     5,     6,     7,     8,    -1,   107,    -1,   109,
     110,   111,    -1,    79,    80,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    26,    27,    -1,    -1,    -1,    94,    95,
      -1,    -1,    -1,    99,    37,     4,     5,     6,     7,     8,
      -1,   107,    -1,   109,   110,   111,    -1,    -1,    51,    -1,
      53,    54,    -1,    -1,    57,    -1,    59,    26,    27,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    37,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    79,    80,    -1,    -1,
      -1,    -1,    51,    -1,    53,    54,    -1,    -1,    57,    -1,
      59,    94,    95,    -1,    -1,    -1,    99,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,   107,    -1,   109,   110,   111,    -1,
      79,    80,    -1,    -1,    47,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    57,    94,    95,    -1,    -1,    -1,
      99,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   107,    72,
     109,   110,   111,    76,    77,    78,    -1,    -1,    81,    82,
      83,    84,    85,    86,    87,    88,    89,    90,    91,    92,
      93,    -1,    -1,    96,    97,    98,    99,   100,   101,   102,
     103,   104,   105,   106,    47,    -1,   109,   110,    -1,    -1,
      -1,    -1,    -1,    -1,    57,    58,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    76,    77,    78,    -1,    -1,    81,    82,
      83,    84,    85,    86,    87,    88,    89,    90,    91,    92,
      93,    -1,    -1,    96,    97,    98,    99,   100,   101,   102,
     103,   104,   105,   106,    47,    -1,   109,   110,    -1,    -1,
      -1,    -1,    -1,    -1,    57,    58,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    76,    77,    78,    -1,    -1,    81,    82,
      83,    84,    85,    86,    87,    88,    89,    90,    91,    92,
      93,    56,    57,    96,    97,    98,    99,   100,   101,   102,
     103,   104,   105,   106,    -1,    -1,   109,   110,    -1,    -1,
      -1,    76,    77,    78,    -1,    -1,    81,    82,    83,    84,
      85,    86,    87,    88,    89,    90,    91,    92,    93,    57,
      58,    96,    97,    98,    99,   100,   101,   102,   103,   104,
     105,   106,    -1,    -1,   109,   110,    -1,    -1,    76,    77,
      78,    -1,    -1,    81,    82,    83,    84,    85,    86,    87,
      88,    89,    90,    91,    92,    93,    57,    -1,    96,    97,
      98,    99,   100,   101,   102,   103,   104,   105,   106,    -1,
      -1,   109,   110,    74,    -1,    76,    77,    78,    -1,    -1,
      81,    82,    83,    84,    85,    86,    87,    88,    89,    90,
      91,    92,    93,    57,    58,    96,    97,    98,    99,   100,
     101,   102,   103,   104,   105,   106,    -1,    -1,   109,   110,
      -1,    -1,    76,    77,    78,    -1,    -1,    81,    82,    83,
      84,    85,    86,    87,    88,    89,    90,    91,    92,    93,
      -1,    -1,    96,    97,    98,    99,   100,   101,   102,   103,
     104,   105,   106,    57,    -1,   109,   110,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    72,    -1,
      -1,    -1,    76,    77,    78,    -1,    -1,    81,    82,    83,
      84,    85,    86,    87,    88,    89,    90,    91,    92,    93,
      57,    58,    96,    97,    98,    99,   100,   101,   102,   103,
     104,   105,   106,    -1,    -1,   109,   110,    -1,    -1,    76,
      77,    78,    -1,    -1,    81,    82,    83,    84,    85,    86,
      87,    88,    89,    90,    91,    92,    93,    57,    58,    96,
      97,    98,    99,   100,   101,   102,   103,   104,   105,   106,
      -1,    -1,   109,   110,    -1,    -1,    76,    77,    78,    -1,
      -1,    81,    82,    83,    84,    85,    86,    87,    88,    89,
      90,    91,    92,    93,    57,    -1,    96,    97,    98,    99,
     100,   101,   102,   103,   104,   105,   106,    -1,    -1,   109,
     110,    -1,    -1,    76,    77,    78,    -1,    -1,    81,    82,
      83,    84,    85,    86,    87,    88,    89,    90,    91,    92,
      93,    57,    -1,    96,    97,    98,    99,   100,   101,   102,
     103,   104,   105,   106,    -1,    -1,   109,   110,    -1,    -1,
      -1,    77,    78,    -1,    -1,    81,    82,    83,    84,    85,
      86,    87,    88,    89,    90,    91,    92,    93,    57,    -1,
      96,    97,    98,    99,   100,   101,   102,   103,   104,   105,
     106,    -1,    -1,   109,   110,    -1,    -1,    -1,    -1,    78,
      -1,    -1,    81,    82,    83,    84,    85,    86,    87,    88,
      89,    90,    91,    92,    93,    57,    -1,    96,    97,    98,
      99,   100,   101,   102,   103,   104,   105,   106,    -1,    -1,
     109,   110,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    81,
      82,    83,    84,    85,    86,    87,    88,    89,    90,    91,
      92,    93,    -1,    -1,    96,    97,    98,    99,   100,   101,
     102,   103,   104,   105,   106,    -1,    -1,   109,   110
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint16 yystos[] =
{
       0,   113,   114,     0,     1,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    13,    15,    18,    22,    23,
      25,    26,    27,    28,    29,    31,    32,    34,    35,    36,
      37,    39,    40,    41,    42,    43,    44,    45,    46,    48,
      50,    51,    52,    53,    54,    55,    57,    59,    79,    80,
      84,    94,    95,    99,   105,   107,   109,   110,   111,   115,
     116,   118,   119,   121,   122,   123,   124,   125,   126,   127,
     128,   129,   130,   131,   132,   133,   134,   135,   136,   138,
     139,   140,   142,   143,   151,   152,   153,   155,   156,   157,
     162,   163,   164,   171,   173,   186,   188,   197,   198,   200,
     207,   208,   209,   211,   213,   221,   222,   223,   224,   226,
     228,   231,   232,   233,   236,   254,   259,   263,   264,   265,
     266,   267,   268,   269,   270,   275,   278,   279,   280,     3,
       3,     1,   120,   265,     1,   267,   268,     1,     3,     1,
       3,    14,     1,   268,     1,   265,   267,   283,     1,   268,
       1,     1,   268,     1,   268,   281,     1,     3,    47,     1,
     268,     1,     6,     1,     6,     1,     3,   268,   260,   276,
       1,     6,     7,     1,   268,   270,     1,     6,     1,     3,
      47,     1,   268,     1,     3,     6,   225,     1,     6,   225,
     227,     1,     6,   229,   230,     1,     6,     1,   268,    58,
     268,   282,    47,   268,    47,    58,    72,   268,   281,   285,
     268,   267,     1,     3,   281,   268,   268,   268,     1,     3,
     281,   268,   268,   268,   268,   137,    32,    34,    48,   119,
     141,   119,   154,   119,   172,   187,   199,    49,   216,   219,
     220,   119,     1,    57,     1,     6,   234,   235,   234,    59,
      60,    61,    62,    63,    64,    65,    66,    67,    68,    69,
      70,    75,    88,   269,     3,    57,    71,    76,    77,    78,
      81,    82,    83,    84,    85,    86,    87,    88,    89,    90,
      91,    92,    93,    96,    97,    98,    99,   100,   101,   102,
     103,   104,   105,   106,   109,   110,    59,    60,   269,     3,
       3,    75,    88,     3,    47,     3,    47,     3,     3,     3,
       3,    47,     3,    47,     3,    47,    88,    75,    91,     3,
       3,     3,     3,     3,     3,     1,    74,    75,     3,   119,
       3,     3,     3,   237,     3,   255,     3,     3,     6,   261,
     262,     1,     6,   214,   215,   277,     3,     3,     3,     3,
       3,     3,    88,     3,    47,     3,     3,    91,     3,     3,
      75,     3,    75,     3,     3,    87,     3,    75,     3,     3,
       3,     1,    58,   268,    56,    58,   268,    58,    47,    72,
       1,    58,     1,    58,    75,    87,    88,     3,     3,     3,
       3,   150,   150,   150,   174,   189,   150,     1,     3,    47,
     150,   217,   218,     3,     1,   214,     3,     3,    75,     9,
     234,     3,   103,   268,     6,   268,   268,   268,   268,   268,
     268,   268,   268,   268,   268,   267,   284,    48,   111,   268,
     272,   281,    56,   281,   268,     1,   268,   268,   268,   268,
     268,   268,   268,   268,   268,   268,   268,     6,   268,   268,
     268,   268,   268,   268,   268,   268,   268,   268,   268,   268,
     268,   268,   268,   103,   268,     6,   265,   268,   268,   265,
       1,   268,     3,   268,   268,     1,    57,   238,   239,     1,
      33,   241,   256,     3,    75,     3,    75,    72,     1,   264,
       1,   268,     6,     6,     4,     6,    99,   117,   230,    58,
      58,   268,   268,    58,   268,   268,   268,     9,   119,    16,
      17,   144,   146,   147,   149,     9,     1,     3,    23,    24,
     175,   180,   182,     1,     3,    23,   180,   190,    30,   201,
     202,   203,   204,     3,    47,     9,   150,   119,   212,     1,
      56,     6,     3,     3,   268,    58,     3,     3,     3,     3,
       3,     3,     3,     3,     3,     3,    75,    88,   273,   111,
       3,     3,     3,     1,    56,     3,     1,    47,   268,    58,
      88,    74,     3,     3,    47,     3,     3,   214,   247,   241,
       3,     6,   242,   243,     3,   257,     1,   262,   215,   268,
       3,     3,     3,     3,     4,    58,    72,     3,     1,     3,
       1,   268,     9,   145,   148,     3,     3,     1,     6,     7,
       8,   117,   184,   185,     1,     9,   181,     3,     1,     4,
       6,   195,   196,     9,     1,     3,     4,     6,    91,   205,
     206,     9,   203,   150,     3,     9,    56,   210,     3,    47,
      58,   267,   268,   281,     1,    57,   274,     3,   271,     1,
     268,    58,   268,   268,   158,   159,     1,    56,     3,     6,
      38,    48,    49,    93,   208,   248,   249,   251,   252,     3,
      57,   244,    75,     3,   208,   249,   251,   252,   258,   268,
       3,     3,     3,     3,   150,   150,     3,    47,    74,     3,
      47,    75,     3,     3,    47,   183,     3,    47,     3,    47,
      75,     3,     3,   265,     3,    75,    91,     3,     3,    47,
      56,     3,     3,     3,   214,   216,    56,     3,    47,    73,
      19,    20,    21,   119,   160,   161,   165,   167,   169,   119,
     240,    88,     3,     6,     1,     6,    79,   253,     9,     6,
      27,   245,   246,   264,   243,     9,   144,   178,   179,   117,
     176,   177,   185,   150,   119,   193,   194,   191,   192,   196,
       3,   206,   265,     3,     1,    56,   150,   268,     1,     3,
      47,     1,     3,    47,     1,     3,    47,     9,   160,    56,
     268,   250,    88,     3,     6,     3,    75,     3,    56,    75,
       3,   150,   119,   150,   119,   150,   119,   150,   119,     3,
       3,     9,     3,    47,     3,   166,   119,     3,   168,   119,
       3,   170,   119,     3,     3,   216,   268,     6,    79,   246,
     150,   150,   150,   150,     3,     6,     9,     9,     9,     9,
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
#line 209 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_lone_end ); ;}
    break;

  case 7:
#line 210 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_case_outside ); ;}
    break;

  case 8:
#line 216 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(1) - (1)].stringp) != 0 )
            COMPILER->addLoad( *(yyvsp[(1) - (1)].stringp) );
      ;}
    break;

  case 10:
#line 222 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if( (yyvsp[(1) - (1)].fal_stat) != 0 )
            COMPILER->addFunction( (yyvsp[(1) - (1)].fal_stat) );
      ;}
    break;

  case 11:
#line 227 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(1) - (1)].fal_stat) != 0 )
            COMPILER->addClass( (yyvsp[(1) - (1)].fal_stat) );
      ;}
    break;

  case 12:
#line 232 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(1) - (1)].fal_stat) != 0 )
            COMPILER->addClass( (yyvsp[(1) - (1)].fal_stat) );
      ;}
    break;

  case 13:
#line 237 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if( (yyvsp[(1) - (1)].fal_stat) != 0 )
            COMPILER->addStatement( (yyvsp[(1) - (1)].fal_stat) );
      ;}
    break;

  case 19:
#line 249 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.integer) = - (yyvsp[(2) - (2)].integer); ;}
    break;

  case 20:
#line 254 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getContext() != 0 )
            COMPILER->raiseError(Falcon::e_toplevel_load );
         (yyval.stringp) = (yyvsp[(2) - (3)].stringp);
      ;}
    break;

  case 21:
#line 260 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getContext() != 0 )
            COMPILER->raiseError(Falcon::e_toplevel_load );
         (yyval.stringp) = (yyvsp[(2) - (3)].stringp);
      ;}
    break;

  case 22:
#line 266 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_load );
         (yyval.stringp) = 0;
      ;}
    break;

  case 23:
#line 273 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->checkLocalUndefined(); (yyval.fal_stat) = (yyvsp[(1) - (1)].fal_stat); ;}
    break;

  case 24:
#line 274 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = 0; ;}
    break;

  case 25:
#line 277 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_toplevel_func ); (yyval.fal_stat) = 0; ;}
    break;

  case 26:
#line 278 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_toplevel_obj ); (yyval.fal_stat) = 0; ;}
    break;

  case 27:
#line 279 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_toplevel_class ); (yyval.fal_stat) = 0; ;}
    break;

  case 28:
#line 280 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syntax ); (yyval.fal_stat) = 0;;}
    break;

  case 29:
#line 285 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defContext( true );
      COMPILER->defineVal( (yyvsp[(1) - (3)].fal_val) );
      COMPILER->addStatement( new Falcon::StmtAssignment( CURRENT_LINE, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) );
   ;}
    break;

  case 30:
#line 290 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(3) - (5)].fal_val) );
      COMPILER->addStatement( new Falcon::StmtAssignment( CURRENT_LINE, (yyvsp[(3) - (5)].fal_val), (yyvsp[(5) - (5)].fal_val) ) );
   ;}
    break;

  case 31:
#line 297 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = new Falcon::StmtAutoexpr( LINE, (yyvsp[(1) - (2)].fal_val) ); ;}
    break;

  case 53:
#line 323 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->defContext( false );  (yyval.fal_stat)=0; ;}
    break;

  case 54:
#line 325 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError( Falcon::e_syn_def ); ;}
    break;

  case 55:
#line 330 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAssignment( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 56:
#line 334 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (5)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtUnref( LINE, (yyvsp[(1) - (5)].fal_val) );
   ;}
    break;

  case 57:
#line 338 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
         (yyval.fal_stat) = new Falcon::StmtAssignment( LINE, (yyvsp[(1) - (4)].fal_val), new Falcon::Value( (yyvsp[(3) - (4)].fal_adecl) ) );
      ;}
    break;

  case 58:
#line 342 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
         (yyval.fal_stat) = new Falcon::StmtAssignment( LINE, (yyvsp[(1) - (4)].fal_val), new Falcon::Value( (yyvsp[(3) - (4)].fal_val) ) );
      ;}
    break;

  case 59:
#line 346 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(1) - (6)].fal_val) );
         (yyvsp[(3) - (6)].fal_adecl)->pushFront( (yyvsp[(1) - (6)].fal_val) );
         (yyval.fal_stat) = new Falcon::StmtAssignment( LINE, new Falcon::Value((yyvsp[(3) - (6)].fal_adecl)), (yyvsp[(5) - (6)].fal_val) );
      ;}
    break;

  case 60:
#line 351 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(1) - (6)].fal_val) );
         (yyvsp[(3) - (6)].fal_adecl)->pushFront( (yyvsp[(1) - (6)].fal_val) );
         (yyval.fal_stat) = new Falcon::StmtAssignment( LINE, new Falcon::Value((yyvsp[(3) - (6)].fal_adecl)), new Falcon::Value( (yyvsp[(5) - (6)].fal_adecl) ) );
      ;}
    break;

  case 72:
#line 375 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAutoAdd( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 73:
#line 382 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAutoSub( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 74:
#line 389 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAutoMul( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 75:
#line 396 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAutoDiv( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 76:
#line 403 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAutoMod( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 77:
#line 410 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAutoPow( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 78:
#line 417 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAutoBAND( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 79:
#line 424 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAutoBOR( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 80:
#line 431 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAutoBXOR( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 81:
#line 437 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAutoSHL( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 82:
#line 443 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAutoSHR( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 83:
#line 451 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtWhile *w = new Falcon::StmtWhile( LINE, (yyvsp[(1) - (1)].fal_val) );
         COMPILER->pushLoop( w );
         COMPILER->pushContext( w );
         COMPILER->pushContextSet( &w->children() );
      ;}
    break;

  case 84:
#line 458 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtWhile *w = static_cast<Falcon::StmtWhile *>(COMPILER->getContext());
         COMPILER->popLoop();
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = w;
      ;}
    break;

  case 85:
#line 465 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtWhile *w = new Falcon::StmtWhile( LINE, (yyvsp[(1) - (2)].fal_val) );
         if ( (yyvsp[(2) - (2)].fal_stat) != 0 )
            w->children().push_back( (yyvsp[(2) - (2)].fal_stat) );
         (yyval.fal_stat) = w;
      ;}
    break;

  case 86:
#line 473 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); ;}
    break;

  case 87:
#line 474 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = 0; ;}
    break;

  case 88:
#line 475 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_while ); (yyval.fal_val) = 0; ;}
    break;

  case 89:
#line 479 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); ;}
    break;

  case 90:
#line 480 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = 0; ;}
    break;

  case 91:
#line 481 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_while, "", CURRENT_LINE ); (yyval.fal_val) = 0; ;}
    break;

  case 92:
#line 485 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtIf *stmt = new Falcon::StmtIf( LINE, (yyvsp[(1) - (1)].fal_val) );
         COMPILER->pushContext( stmt );
         COMPILER->pushContextSet( &stmt->children() );
      ;}
    break;

  case 93:
#line 493 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtIf *stmt = static_cast<Falcon::StmtIf *>(COMPILER->getContext());
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = stmt;
      ;}
    break;

  case 94:
#line 500 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // use LINE as statement includes EOL
         Falcon::StmtIf *stmt = new Falcon::StmtIf( LINE, (yyvsp[(1) - (2)].fal_val) );
         if( (yyvsp[(2) - (2)].fal_stat) != 0 )
            stmt->children().push_back( (yyvsp[(2) - (2)].fal_stat) );
         (yyval.fal_stat) = stmt;
      ;}
    break;

  case 95:
#line 510 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); ;}
    break;

  case 96:
#line 511 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {  COMPILER->raiseError(Falcon::e_syn_if ); (yyval.fal_val) = 0; ;}
    break;

  case 97:
#line 515 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); ;}
    break;

  case 98:
#line 516 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {  COMPILER->raiseError(Falcon::e_syn_if, "", CURRENT_LINE ); (yyval.fal_val) = 0; ;}
    break;

  case 101:
#line 523 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtIf *stmt = static_cast<Falcon::StmtIf *>(COMPILER->getContext());
         COMPILER->popContextSet();
         COMPILER->pushContextSet( &stmt->elseChildren() );
      ;}
    break;

  case 104:
#line 533 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_else ); ;}
    break;

  case 105:
#line 538 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtIf *stmt = static_cast<Falcon::StmtIf *>(COMPILER->getContext());
         COMPILER->popContextSet();
         Falcon::StmtElif *elif = new Falcon::StmtElif( LINE, (yyvsp[(1) - (1)].fal_val) );
         stmt->elifChildren().push_back( elif );
         COMPILER->pushContextSet( &elif->children() );
      ;}
    break;

  case 107:
#line 550 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); ;}
    break;

  case 108:
#line 551 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_elif ); (yyval.fal_val) = 0; ;}
    break;

  case 110:
#line 556 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->addStatement( (yyvsp[(2) - (2)].fal_stat) );
   ;}
    break;

  case 111:
#line 563 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getLoop() == 0 ) {
            COMPILER->raiseError(Falcon::e_break_out );
            (yyval.fal_stat) = 0;
         }
         else
            (yyval.fal_stat) = new Falcon::StmtBreak( LINE );
      ;}
    break;

  case 112:
#line 572 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_break );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 113:
#line 580 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getLoop() == 0 ) {
            COMPILER->raiseError(Falcon::e_continue_out );
            (yyval.fal_stat) = 0;
         }
         else
            (yyval.fal_stat) = new Falcon::StmtContinue( LINE );
      ;}
    break;

  case 114:
#line 590 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getLoop() == 0 ) {
            COMPILER->raiseError(Falcon::e_continue_out );
            (yyval.fal_stat) = 0;
         }
         else
            (yyval.fal_stat) = new Falcon::StmtContinue( LINE, true );
      ;}
    break;

  case 115:
#line 599 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_continue );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 116:
#line 606 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtFor *f = static_cast<Falcon::StmtFor *>( (yyvsp[(1) - (1)].fal_stat) );
         COMPILER->pushLoop( f );
         COMPILER->pushContext( f );
         COMPILER->pushContextSet( &f->children() );
      ;}
    break;

  case 117:
#line 613 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtFor *f = static_cast<Falcon::StmtFor *>(COMPILER->getContext());
         COMPILER->popLoop();
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = f;
      ;}
    break;

  case 118:
#line 621 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 119:
#line 636 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(2) - (7)].fal_val) );
         (yyval.fal_stat) = new Falcon::StmtFor( LINE, (yyvsp[(2) - (7)].fal_val), (yyvsp[(4) - (7)].fal_val), (yyvsp[(6) - (7)].fal_val) );
      ;}
    break;

  case 120:
#line 640 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(2) - (9)].fal_val) );
         (yyval.fal_stat) = new Falcon::StmtFor( LINE, (yyvsp[(2) - (9)].fal_val), (yyvsp[(4) - (9)].fal_val), (yyvsp[(6) - (9)].fal_val), (yyvsp[(8) - (9)].fal_val) );
      ;}
    break;

  case 121:
#line 645 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_for );
         (yyval.fal_stat) = new Falcon::StmtFor( LINE, 0, 0, 0 );
      ;}
    break;

  case 122:
#line 652 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(2) - (7)].fal_val) );
         (yyval.fal_stat) = new Falcon::StmtFor( CURRENT_LINE, (yyvsp[(2) - (7)].fal_val), (yyvsp[(4) - (7)].fal_val), (yyvsp[(6) - (7)].fal_val) );
      ;}
    break;

  case 123:
#line 656 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(2) - (9)].fal_val) );
         (yyval.fal_stat) = new Falcon::StmtFor( CURRENT_LINE, (yyvsp[(2) - (9)].fal_val), (yyvsp[(4) - (9)].fal_val), (yyvsp[(6) - (9)].fal_val), (yyvsp[(8) - (9)].fal_val) );
      ;}
    break;

  case 124:
#line 661 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_for, "", CURRENT_LINE );
         (yyval.fal_stat) = new Falcon::StmtFor( CURRENT_LINE, 0, 0, 0 );
      ;}
    break;

  case 125:
#line 670 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 126:
#line 686 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         COMPILER->popLoop();
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = f;
      ;}
    break;

  case 127:
#line 694 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
          Falcon::StmtForin *f;
         Falcon::ArrayDecl *decl = (yyvsp[(2) - (5)].fal_adecl);
         if ( decl->front() == decl->back() ) {
            f = new Falcon::StmtForin( CURRENT_LINE, (Falcon::Value *) decl->front(), (yyvsp[(4) - (5)].fal_val) );
            decl->deletor(0);
            delete decl;
         }
         else
            f = new Falcon::StmtForin( CURRENT_LINE, new Falcon::Value(decl), (yyvsp[(4) - (5)].fal_val) );

         COMPILER->pushLoop( f );
         COMPILER->pushContext( f );
         COMPILER->pushContextSet( &f->children() );
      ;}
    break;

  case 128:
#line 710 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(7) - (7)].fal_stat) != 0 )
            COMPILER->addStatement( (yyvsp[(7) - (7)].fal_stat) );
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         COMPILER->popLoop();
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = f;
      ;}
    break;

  case 129:
#line 720 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError( Falcon::e_syn_forin ); ;}
    break;

  case 132:
#line 729 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(1) - (1)].fal_stat) != 0 )
         {
            Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
            f->children().push_back( (yyvsp[(1) - (1)].fal_stat) );
         }
      ;}
    break;

  case 136:
#line 743 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 137:
#line 756 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_fordot );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 138:
#line 764 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = new Falcon::StmtSelfPrint( LINE, (yyvsp[(2) - (3)].fal_adecl) );
      ;}
    break;

  case 139:
#line 768 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_self_print );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 140:
#line 774 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyvsp[(2) - (3)].fal_adecl)->pushBack( new Falcon::Value( COMPILER->addString( "\n" ) ) );
         (yyval.fal_stat) = new Falcon::StmtSelfPrint( LINE, (yyvsp[(2) - (3)].fal_adecl) );
      ;}
    break;

  case 141:
#line 780 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::ArrayDecl *adecl = new Falcon::ArrayDecl();
         adecl->pushBack( new Falcon::Value( COMPILER->addString( "\n" ) ) );
         (yyval.fal_stat) = new Falcon::StmtSelfPrint( LINE, adecl );
      ;}
    break;

  case 142:
#line 787 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_self_print );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 143:
#line 792 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_self_print );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 144:
#line 801 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::ArrayDecl *adecl = new Falcon::ArrayDecl();
      adecl->pushBack( new Falcon::Value( (yyvsp[(1) - (1)].stringp) ) );
      (yyval.fal_stat) = new Falcon::StmtSelfPrint( LINE, adecl );
   ;}
    break;

  case 145:
#line 810 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 146:
#line 822 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->popContextSet(); ;}
    break;

  case 147:
#line 824 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         if( ! f->firstBlock().empty() )
         {
            COMPILER->raiseError( Falcon::e_already_forfirst );
         }
         if ( (yyvsp[(3) - (3)].fal_stat) != 0 )
            f->firstBlock().push_back( (yyvsp[(3) - (3)].fal_stat) );
      ;}
    break;

  case 148:
#line 833 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_forfirst ); ;}
    break;

  case 149:
#line 837 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 150:
#line 849 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->popContextSet(); ;}
    break;

  case 151:
#line 850 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         if( ! f->lastBlock().empty() )
         {
            COMPILER->raiseError( Falcon::e_already_forlast );
         }
         if ( (yyvsp[(3) - (3)].fal_stat) != 0 )
            f->lastBlock().push_back( (yyvsp[(3) - (3)].fal_stat) );
      ;}
    break;

  case 152:
#line 859 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_forlast ); ;}
    break;

  case 153:
#line 863 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         if( ! f->middleBlock().empty() )
         {
            COMPILER->raiseError( Falcon::e_already_formiddle );
         }
		 // Push anyhow an empty item, that is needed for empty last blocks
		 // Apparently you get a segfault without it.
		 // (Note that the formiddle: version below does *not* need it
		 f->middleBlock().push_back( new Falcon::StmtNone( LINE ) );
         COMPILER->pushContextSet( &f->middleBlock() );
      ;}
    break;

  case 154:
#line 877 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->popContextSet(); ;}
    break;

  case 155:
#line 879 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         if( ! f->middleBlock().empty() )
         {
            COMPILER->raiseError( Falcon::e_already_formiddle );
         }
         if ( (yyvsp[(3) - (3)].fal_stat) != 0 )
            f->middleBlock().push_back( (yyvsp[(3) - (3)].fal_stat) );
      ;}
    break;

  case 156:
#line 888 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_formiddle ); ;}
    break;

  case 157:
#line 892 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::StmtSwitch *stmt = new Falcon::StmtSwitch( LINE, (yyvsp[(1) - (1)].fal_val) );
         COMPILER->pushContext( stmt );
         COMPILER->pushContextSet( &stmt->blocks() );
      ;}
    break;

  case 158:
#line 900 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = stmt;
      ;}
    break;

  case 159:
#line 909 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); ;}
    break;

  case 160:
#line 911 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_switch_decl );
         (yyval.fal_val) = 0;
      ;}
    break;

  case 163:
#line 920 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_switch_body ); ;}
    break;

  case 165:
#line 926 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      ;}
    break;

  case 167:
#line 936 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( CURRENT_LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      ;}
    break;

  case 168:
#line 944 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addStatement( (yyvsp[(5) - (5)].fal_stat) );
      ;}
    break;

  case 169:
#line 948 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_case_decl );

         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      ;}
    break;

  case 171:
#line 960 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_case_decl );

         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( CURRENT_LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      ;}
    break;

  case 172:
#line 970 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addStatement( (yyvsp[(5) - (5)].fal_stat) );
      ;}
    break;

  case 174:
#line 979 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 178:
#line 993 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_default_decl ); ;}
    break;

  case 180:
#line 997 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addStatement( (yyvsp[(2) - (2)].fal_stat) );
      ;}
    break;

  case 183:
#line 1009 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         // todo: correct error
         if ( stmt->nilBlock() != -1 )
            COMPILER->raiseError(Falcon::e_switch_clash, "nil entry", CURRENT_LINE );
         stmt->nilBlock( stmt->currentBlock() );
      ;}
    break;

  case 184:
#line 1018 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 185:
#line 1030 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 186:
#line 1041 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 187:
#line 1052 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 188:
#line 1072 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::StmtSelect *stmt = new Falcon::StmtSelect( LINE, (yyvsp[(1) - (1)].fal_val) );
         COMPILER->pushContext( stmt );
         COMPILER->pushContextSet( &stmt->blocks() );
      ;}
    break;

  case 189:
#line 1080 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = stmt;
      ;}
    break;

  case 190:
#line 1089 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); ;}
    break;

  case 191:
#line 1091 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_select_decl );
         (yyval.fal_val) = 0;
      ;}
    break;

  case 194:
#line 1100 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_select_body ); ;}
    break;

  case 196:
#line 1106 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      ;}
    break;

  case 198:
#line 1116 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( CURRENT_LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      ;}
    break;

  case 199:
#line 1125 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addStatement( (yyvsp[(5) - (5)].fal_stat) );
      ;}
    break;

  case 200:
#line 1129 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_case_decl );

         Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      ;}
    break;

  case 202:
#line 1141 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_case_decl );

        Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( CURRENT_LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      ;}
    break;

  case 203:
#line 1151 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addStatement( (yyvsp[(5) - (5)].fal_stat) );
      ;}
    break;

  case 207:
#line 1165 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 208:
#line 1177 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 209:
#line 1198 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = new Falcon::StmtGive( LINE, (yyvsp[(4) - (5)].fal_val), (yyvsp[(2) - (5)].fal_adecl) );
      ;}
    break;

  case 210:
#line 1202 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = new Falcon::StmtGive( LINE, 0, (yyvsp[(2) - (4)].fal_adecl) );
         COMPILER->raiseError(Falcon::e_syn_give );
      ;}
    break;

  case 211:
#line 1206 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_give ); (yyval.fal_stat) = 0; ;}
    break;

  case 212:
#line 1214 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::StmtTry *t = new Falcon::StmtTry( CURRENT_LINE );
      if ( (yyvsp[(3) - (3)].fal_stat) != 0 )
          t->children().push_back( (yyvsp[(3) - (3)].fal_stat) );
      (yyval.fal_stat) = t;
   ;}
    break;

  case 213:
#line 1221 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtTry *t = new Falcon::StmtTry( LINE );
         COMPILER->pushContext( t );
         COMPILER->pushContextSet( &t->children() );
      ;}
    break;

  case 214:
#line 1231 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = COMPILER->getContext();
         COMPILER->popContext();
         COMPILER->popContextSet();
      ;}
    break;

  case 216:
#line 1240 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_try ); ;}
    break;

  case 222:
#line 1260 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 223:
#line 1278 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 224:
#line 1298 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->popContextSet(); // popping previous catch

         Falcon::StmtTry *t = static_cast<Falcon::StmtTry *>( COMPILER->getContext() );
         Falcon::StmtCatchBlock *lst = new Falcon::StmtCatchBlock( LINE, 0 );
         COMPILER->pushContextSet( &lst->children() );
         t->addHandler( lst );
      ;}
    break;

  case 225:
#line 1308 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->popContextSet(); // popping previous catch

         Falcon::StmtTry *t = static_cast<Falcon::StmtTry *>( COMPILER->getContext() );
         COMPILER->defineVal( (yyvsp[(4) - (5)].fal_val) );
         Falcon::StmtCatchBlock *lst = new Falcon::StmtCatchBlock( LINE, (yyvsp[(4) - (5)].fal_val) );
         COMPILER->pushContextSet( &lst->children() );
         t->addHandler( lst );
      ;}
    break;

  case 226:
#line 1319 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->raiseError( Falcon::e_syn_catch );
   ;}
    break;

  case 229:
#line 1332 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 230:
#line 1344 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 231:
#line 1366 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = new Falcon::StmtRaise( LINE, (yyvsp[(2) - (3)].fal_val) ); ;}
    break;

  case 232:
#line 1367 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_raise ); (yyval.fal_stat) = 0; ;}
    break;

  case 233:
#line 1379 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = COMPILER->getContext();
         COMPILER->closeFunction();
      ;}
    break;

  case 234:
#line 1385 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addStatement( (yyvsp[(2) - (2)].fal_stat) );
         (yyval.fal_stat) = COMPILER->getContext();
         COMPILER->closeFunction();
      ;}
    break;

  case 236:
#line 1394 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); ;}
    break;

  case 237:
#line 1395 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError(Falcon::e_syn_funcdecl, COMPILER->tempLine(), CTX_LINE );
      ;}
    break;

  case 238:
#line 1398 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_funcdecl ); ;}
    break;

  case 240:
#line 1403 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); ;}
    break;

  case 241:
#line 1404 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError(Falcon::e_syn_funcdecl, COMPILER->tempLine(), CTX_LINE );
      ;}
    break;

  case 242:
#line 1411 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 246:
#line 1472 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 248:
#line 1489 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());
         COMPILER->pushContextSet( &func->staticBlock() );
         COMPILER->staticPrefix( &func->symbol()->name() );
      ;}
    break;

  case 249:
#line 1495 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->popContextSet();
         COMPILER->staticPrefix(0);
      ;}
    break;

  case 250:
#line 1500 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());
         COMPILER->pushContextSet( &func->staticBlock() );
         COMPILER->staticPrefix( &func->symbol()->name() );
      ;}
    break;

  case 251:
#line 1506 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addStatement( (yyvsp[(3) - (3)].fal_stat) );
         COMPILER->popContextSet();
         COMPILER->staticPrefix(0);
      ;}
    break;

  case 253:
#line 1515 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_static ); ;}
    break;

  case 255:
#line 1520 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_static, "", CURRENT_LINE ); ;}
    break;

  case 256:
#line 1530 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = new Falcon::StmtLaunch( LINE, (yyvsp[(2) - (3)].fal_val) );
      ;}
    break;

  case 257:
#line 1533 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_launch ); (yyval.fal_stat) = 0; ;}
    break;

  case 258:
#line 1542 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getFunction() == 0 )
            COMPILER->raiseError(Falcon::e_pass_outside );
         else
            (yyval.fal_stat) = new Falcon::StmtPass( LINE, (yyvsp[(2) - (3)].fal_val) );
      ;}
    break;

  case 259:
#line 1549 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // define the expression anyhow so we don't have fake errors below
         COMPILER->defineVal( (yyvsp[(4) - (5)].fal_val) );

         if ( COMPILER->getFunction() == 0 )
            COMPILER->raiseError(Falcon::e_pass_outside );
         else
            (yyval.fal_stat) = new Falcon::StmtPass( LINE, (yyvsp[(2) - (5)].fal_val), (yyvsp[(4) - (5)].fal_val) );
      ;}
    break;

  case 260:
#line 1559 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         delete (yyvsp[(2) - (5)].fal_val);
         COMPILER->raiseError(Falcon::e_syn_pass_in );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 261:
#line 1565 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_pass );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 262:
#line 1577 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 263:
#line 1587 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_inv_const_val );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 264:
#line 1592 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_const );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 265:
#line 1604 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->sourceTree()->isExportAll() )
            COMPILER->raiseError(Falcon::e_export_all );
         else
            COMPILER->sourceTree()->setExportAll();
         // no effect
         (yyval.fal_stat)=0;
      ;}
    break;

  case 266:
#line 1613 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->sourceTree()->isExportAll() )
            COMPILER->raiseError(Falcon::e_export_all );
         // no effect
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 267:
#line 1620 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_export );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 268:
#line 1628 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Symbol *sym = COMPILER->addGlobalSymbol( (yyvsp[(1) - (1)].stringp) );
         sym->exported(true);
      ;}
    break;

  case 269:
#line 1633 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Symbol *sym = COMPILER->addGlobalSymbol( (yyvsp[(3) - (3)].stringp) );
         sym->exported(true);
      ;}
    break;

  case 270:
#line 1641 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 271:
#line 1645 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_import );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 272:
#line 1653 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Symbol *sym = COMPILER->addGlobalSymbol( (yyvsp[(1) - (1)].stringp) );
         sym->imported(true);
      ;}
    break;

  case 273:
#line 1658 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Symbol *sym = COMPILER->addGlobalSymbol( (yyvsp[(3) - (3)].stringp) );
         sym->imported(true);
      ;}
    break;

  case 274:
#line 1670 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // no effect
         (yyval.fal_stat)=0;
      ;}
    break;

  case 275:
#line 1675 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_directive );
         (yyval.fal_stat)=0;
     ;}
    break;

  case 278:
#line 1688 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->setDirective( *(yyvsp[(1) - (3)].stringp), *(yyvsp[(3) - (3)].stringp) );
      ;}
    break;

  case 279:
#line 1692 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->setDirective( *(yyvsp[(1) - (3)].stringp), (yyvsp[(3) - (3)].integer) );
      ;}
    break;

  case 280:
#line 1706 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // no other action:
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 281:
#line 1713 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // no other action:
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 283:
#line 1721 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_attributes ); ;}
    break;

  case 285:
#line 1725 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_attributes, "", CURRENT_LINE ); ;}
    break;

  case 287:
#line 1731 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addAttribute( (yyvsp[(1) - (1)].stringp) );
         ;}
    break;

  case 288:
#line 1735 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addAttribute( (yyvsp[(3) - (3)].stringp) );
         ;}
    break;

  case 291:
#line 1744 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->raiseError(Falcon::e_inv_attrib );
   ;}
    break;

  case 292:
#line 1755 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 293:
#line 1789 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 295:
#line 1817 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_class );
      ;}
    break;

  case 298:
#line 1825 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); ;}
    break;

  case 299:
#line 1826 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError(Falcon::e_syn_class, COMPILER->tempLine(), CTX_LINE );
      ;}
    break;

  case 304:
#line 1843 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 305:
#line 1876 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_adecl) = 0; ;}
    break;

  case 306:
#line 1881 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      (yyval.fal_adecl) = (yyvsp[(2) - (3)].fal_adecl);
   ;}
    break;

  case 307:
#line 1887 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_adecl) = new Falcon::ArrayDecl(); (yyval.fal_adecl)->pushBack( (yyvsp[(1) - (1)].fal_val) ); ;}
    break;

  case 308:
#line 1888 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyvsp[(1) - (3)].fal_adecl)->pushBack( (yyvsp[(3) - (3)].fal_val) ); (yyval.fal_adecl) = (yyvsp[(1) - (3)].fal_adecl); ;}
    break;

  case 310:
#line 1894 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // the symbol must be a parameter, or we raise an error
         Falcon::Symbol *sym = COMPILER->searchLocalSymbol( (yyvsp[(1) - (1)].stringp) );
         if ( sym == 0 || sym->type() != Falcon::Symbol::tparam ) {
            sym = COMPILER->addGlobalSymbol( (yyvsp[(1) - (1)].stringp) );
         }
         (yyval.fal_val) = new Falcon::Value( sym );
      ;}
    break;

  case 311:
#line 1902 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setSelf(); ;}
    break;

  case 315:
#line 1912 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->addFunction( (yyvsp[(1) - (1)].fal_stat) );
   ;}
    break;

  case 316:
#line 1915 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
      if ( cls->initGiven() ) {
         COMPILER->raiseError(Falcon::e_prop_pinit );
      }
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

  case 318:
#line 1937 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 319:
#line 1961 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());

         COMPILER->popContext();
         COMPILER->popContextSet();
         COMPILER->popFunction();
      ;}
    break;

  case 320:
#line 1972 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 321:
#line 1994 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 324:
#line 2024 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->raiseError(Falcon::e_syn_hasdef );
   ;}
    break;

  case 325:
#line 2031 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
         Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();

         // The symbolmay be undefined or defined; it's not our task to define it here.
         clsdef->has().pushBack( COMPILER->addGlobalSymbol( (yyvsp[(1) - (1)].stringp) ) );
      ;}
    break;

  case 326:
#line 2039 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
         Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();
         clsdef->hasnt().pushBack( COMPILER->addGlobalSymbol( (yyvsp[(2) - (2)].stringp) ) );
      ;}
    break;

  case 327:
#line 2045 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
         Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();
         clsdef->has().pushBack( COMPILER->addGlobalSymbol( (yyvsp[(3) - (3)].stringp) ) );
      ;}
    break;

  case 328:
#line 2051 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
         Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();
         clsdef->hasnt().pushBack( COMPILER->addGlobalSymbol( (yyvsp[(4) - (4)].stringp) ) );
      ;}
    break;

  case 329:
#line 2064 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 330:
#line 2104 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 332:
#line 2129 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_object );
      ;}
    break;

  case 336:
#line 2141 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->addFunction( (yyvsp[(1) - (1)].fal_stat) );
   ;}
    break;

  case 337:
#line 2144 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 339:
#line 2172 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtGlobal *glob = new Falcon::StmtGlobal( CURRENT_LINE );
         COMPILER->pushContext( glob );
      ;}
    break;

  case 340:
#line 2177 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 343:
#line 2192 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_global );
      ;}
    break;

  case 344:
#line 2199 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // we create (or retrieve) a globalized symbol
         Falcon::Symbol *sym = COMPILER->globalize( (yyvsp[(1) - (1)].stringp) );

         // then we add the symbol to the global statement (it's just for symbolic asm generation).
         Falcon::StmtGlobal *glob = static_cast<Falcon::StmtGlobal *>( COMPILER->getContext() );
         glob->addSymbol( sym );
      ;}
    break;

  case 345:
#line 2214 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = new Falcon::StmtReturn(LINE, 0); ;}
    break;

  case 346:
#line 2215 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = new Falcon::StmtReturn( LINE, (yyvsp[(2) - (3)].fal_val) ); ;}
    break;

  case 347:
#line 2216 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_return ); (yyval.fal_stat) = 0; ;}
    break;

  case 348:
#line 2226 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); ;}
    break;

  case 349:
#line 2227 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( true ); ;}
    break;

  case 350:
#line 2228 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( false ); ;}
    break;

  case 351:
#line 2229 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].integer) ); ;}
    break;

  case 352:
#line 2230 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].numeric) ); ;}
    break;

  case 353:
#line 2231 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].stringp) ); ;}
    break;

  case 354:
#line 2236 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 356:
#line 2254 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setSelf(); ;}
    break;

  case 357:
#line 2255 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setSender(); ;}
    break;

  case 359:
#line 2267 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_access, (yyvsp[(1) - (2)].fal_val), (yyvsp[(2) - (2)].fal_val) );
         (yyval.fal_val) = new Falcon::Value( exp );
      ;}
    break;

  case 360:
#line 2272 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_access, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
         (yyval.fal_val) = new Falcon::Value( exp );
      ;}
    break;

  case 361:
#line 2277 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_byte_access, (yyvsp[(1) - (5)].fal_val), (yyvsp[(4) - (5)].fal_val) );
         (yyval.fal_val) = new Falcon::Value( exp );
      ;}
    break;

  case 362:
#line 2283 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_obj_access, (yyvsp[(1) - (3)].fal_val), new Falcon::Value( (yyvsp[(3) - (3)].stringp) ) );
         if ( (yyvsp[(3) - (3)].stringp)->getCharAt(0) == '_' && ! (yyvsp[(1) - (3)].fal_val)->isSelf() )
         {
            COMPILER->raiseError(Falcon::e_priv_access, COMPILER->tempLine() );
         }
         (yyval.fal_val) = new Falcon::Value( exp );
      ;}
    break;

  case 365:
#line 2298 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_plus, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 366:
#line 2299 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_neg, (yyvsp[(2) - (2)].fal_val) ) ); ;}
    break;

  case 367:
#line 2300 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_minus, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 368:
#line 2301 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_times, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 369:
#line 2302 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_divide, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 370:
#line 2303 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_modulo, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 371:
#line 2304 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_power, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 372:
#line 2305 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_and, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 373:
#line 2306 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_or, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 374:
#line 2307 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_xor, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 375:
#line 2308 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_shift_left, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 376:
#line 2309 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_shift_right, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 377:
#line 2310 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_not, (yyvsp[(2) - (2)].fal_val) ) ); ;}
    break;

  case 378:
#line 2311 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->defineVal( (yyvsp[(2) - (4)].fal_val) ); (yyval.fal_val) =
        new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_let, (yyvsp[(2) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) ); ;}
    break;

  case 379:
#line 2313 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->defineVal( (yyvsp[(2) - (4)].fal_val) ); (yyval.fal_val) =
        new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_let, (yyvsp[(2) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) ); ;}
    break;

  case 380:
#line 2315 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_neq, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 381:
#line 2316 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_post_inc, (yyvsp[(1) - (2)].fal_val) ) ); ;}
    break;

  case 382:
#line 2317 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_pre_inc, (yyvsp[(2) - (2)].fal_val) ) ); ;}
    break;

  case 383:
#line 2318 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_post_dec, (yyvsp[(1) - (2)].fal_val) ) ); ;}
    break;

  case 384:
#line 2319 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_pre_dec, (yyvsp[(2) - (2)].fal_val) ) ); ;}
    break;

  case 385:
#line 2320 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_eq, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 386:
#line 2321 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_eq, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 387:
#line 2322 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_eq, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 388:
#line 2323 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_gt, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 389:
#line 2324 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_lt, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 390:
#line 2325 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_ge, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 391:
#line 2326 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_le, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 392:
#line 2327 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_and, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 393:
#line 2328 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_or, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 394:
#line 2329 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_not, (yyvsp[(2) - (2)].fal_val) ) ); ;}
    break;

  case 395:
#line 2330 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_has, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 396:
#line 2331 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_hasnt, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 397:
#line 2332 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_in, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 398:
#line 2333 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_notin, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 399:
#line 2334 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_provides, (yyvsp[(1) - (3)].fal_val), new Falcon::Value( (yyvsp[(3) - (3)].stringp) ) ) ); ;}
    break;

  case 400:
#line 2335 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (2)].fal_val) ); ;}
    break;

  case 401:
#line 2336 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_strexpand, (yyvsp[(2) - (2)].fal_val) ) ); ;}
    break;

  case 402:
#line 2337 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_indirect, (yyvsp[(2) - (2)].fal_val) ) ); ;}
    break;

  case 405:
#line 2340 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_obj_access, (yyvsp[(1) - (3)].fal_val), new Falcon::Value( (yyvsp[(3) - (3)].stringp) ) );
         if ( (yyvsp[(3) - (3)].stringp)->getCharAt(0) == '_' )
         {
            COMPILER->raiseError(Falcon::e_priv_access, COMPILER->tempLine() );
         }
         (yyval.fal_val) = new Falcon::Value( exp );
      ;}
    break;

  case 406:
#line 2348 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_access, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
         (yyval.fal_val) = new Falcon::Value( exp );
      ;}
    break;

  case 407:
#line 2352 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_byte_access, (yyvsp[(1) - (5)].fal_val), (yyvsp[(4) - (5)].fal_val) );
         (yyval.fal_val) = new Falcon::Value( exp );
      ;}
    break;

  case 408:
#line 2356 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_access, (yyvsp[(1) - (2)].fal_val), (yyvsp[(2) - (2)].fal_val) );
         (yyval.fal_val) = new Falcon::Value( exp );
      ;}
    break;

  case 413:
#line 2364 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); ;}
    break;

  case 414:
#line 2369 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( new Falcon::Value( (Falcon::int64) 0 ) ) );
      ;}
    break;

  case 415:
#line 2372 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( (yyvsp[(2) - (4)].fal_val) ) );
      ;}
    break;

  case 416:
#line 2375 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( new Falcon::Value( (Falcon::int64) 0 ), (yyvsp[(3) - (4)].fal_val) ) );
      ;}
    break;

  case 417:
#line 2378 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( (yyvsp[(2) - (5)].fal_val), (yyvsp[(4) - (5)].fal_val) ) );
      ;}
    break;

  case 418:
#line 2385 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_funcall,
                                      (yyvsp[(1) - (4)].fal_val), new Falcon::Value( (yyvsp[(3) - (4)].fal_adecl) ) ) );
      ;}
    break;

  case 419:
#line 2391 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_funcall, (yyvsp[(1) - (3)].fal_val), 0 ) );
      ;}
    break;

  case 420:
#line 2395 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); ;}
    break;

  case 421:
#line 2396 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         delete (yyvsp[(3) - (6)].fal_adecl);
         COMPILER->raiseContextError(Falcon::e_syn_funcall, COMPILER->tempLine(), CTX_LINE );
         (yyval.fal_val) = new Falcon::Value;
      ;}
    break;

  case 422:
#line 2405 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 423:
#line 2438 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());
            (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_lambda ,
               new Falcon::Value( func->symbol() ) ) );
            COMPILER->closeFunction();
         ;}
    break;

  case 425:
#line 2449 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError(Falcon::e_syn_funcdecl, LINE, CTX_LINE );
      ;}
    break;

  case 426:
#line 2453 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_funcdecl );
      ;}
    break;

  case 427:
#line 2461 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 428:
#line 2492 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());
            COMPILER->addStatement( new Falcon::StmtReturn( LINE, (yyvsp[(5) - (5)].fal_val) ) );
            (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_lambda ,
               new Falcon::Value( func->symbol() ) ) );
            COMPILER->checkLocalUndefined();
            COMPILER->closeFunction();
         ;}
    break;

  case 430:
#line 2506 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_lambda );
      ;}
    break;

  case 431:
#line 2515 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      (yyval.fal_val) = new Falcon::Value( new
         Falcon::Expression( Falcon::Expression::t_iif, (yyvsp[(1) - (5)].fal_val), (yyvsp[(3) - (5)].fal_val), (yyvsp[(5) - (5)].fal_val) ) );
   ;}
    break;

  case 432:
#line 2520 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      delete (yyvsp[(1) - (5)].fal_val);
      delete (yyvsp[(3) - (5)].fal_val);
      COMPILER->raiseError(Falcon::e_syn_iif, CURRENT_LINE );
      (yyval.fal_val) = new Falcon::Value;
   ;}
    break;

  case 433:
#line 2527 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      delete (yyvsp[(1) - (4)].fal_val);
      delete (yyvsp[(3) - (4)].fal_val);
      COMPILER->raiseError(Falcon::e_syn_iif, CURRENT_LINE );
      (yyval.fal_val) = new Falcon::Value;
   ;}
    break;

  case 434:
#line 2534 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         delete (yyvsp[(1) - (3)].fal_val);
         COMPILER->raiseError(Falcon::e_syn_iif, CURRENT_LINE );
         (yyval.fal_val) = new Falcon::Value;
      ;}
    break;

  case 435:
#line 2543 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {  (yyval.fal_val) = new Falcon::Value( new Falcon::ArrayDecl() ); ;}
    break;

  case 436:
#line 2545 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (3)].fal_adecl) );
      ;}
    break;

  case 437:
#line 2549 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError( Falcon::e_syn_arraydecl, CURRENT_LINE, CTX_LINE );
         (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (3)].fal_adecl) );
      ;}
    break;

  case 438:
#line 2555 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {  (yyval.fal_val) = new Falcon::Value( new Falcon::ArrayDecl() ); ;}
    break;

  case 439:
#line 2557 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (3)].fal_adecl) );
      ;}
    break;

  case 440:
#line 2561 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError( Falcon::e_syn_arraydecl, CURRENT_LINE, CTX_LINE );
         (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (3)].fal_adecl) );
      ;}
    break;

  case 441:
#line 2569 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {  (yyval.fal_val) = new Falcon::Value( new Falcon::DictDecl() ); ;}
    break;

  case 442:
#line 2570 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (3)].fal_ddecl) ); ;}
    break;

  case 443:
#line 2572 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError( Falcon::e_syn_dictdecl, LINE, CTX_LINE );
         (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (4)].fal_ddecl) );
      ;}
    break;

  case 444:
#line 2579 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_adecl) = new Falcon::ArrayDecl(); (yyval.fal_adecl)->pushBack( (yyvsp[(1) - (1)].fal_val) ); ;}
    break;

  case 445:
#line 2580 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyvsp[(1) - (3)].fal_adecl)->pushBack( (yyvsp[(3) - (3)].fal_val) ); (yyval.fal_adecl) = (yyvsp[(1) - (3)].fal_adecl); ;}
    break;

  case 446:
#line 2584 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_adecl) = new Falcon::ArrayDecl(); (yyval.fal_adecl)->pushBack( (yyvsp[(1) - (1)].fal_val) ); ;}
    break;

  case 447:
#line 2585 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyvsp[(1) - (2)].fal_adecl)->pushBack( (yyvsp[(2) - (2)].fal_val) ); (yyval.fal_adecl) = (yyvsp[(1) - (2)].fal_adecl); ;}
    break;

  case 448:
#line 2589 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(1) - (1)].fal_val) );
         Falcon::ArrayDecl *ad = new Falcon::ArrayDecl();
         ad->pushBack( (yyvsp[(1) - (1)].fal_val) );
         (yyval.fal_adecl) = ad;
      ;}
    break;

  case 449:
#line 2595 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(3) - (3)].fal_val) );
         (yyvsp[(1) - (3)].fal_adecl)->pushBack( (yyvsp[(3) - (3)].fal_val) );
      ;}
    break;

  case 450:
#line 2602 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(1) - (1)].fal_val) );
         Falcon::ArrayDecl *ad = new Falcon::ArrayDecl();
         ad->pushBack( (yyvsp[(1) - (1)].fal_val) );
         (yyval.fal_adecl) = ad;
      ;}
    break;

  case 451:
#line 2608 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(3) - (3)].fal_val) );
         (yyvsp[(1) - (3)].fal_adecl)->pushBack( (yyvsp[(3) - (3)].fal_val) );
      ;}
    break;

  case 452:
#line 2615 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_ddecl) = new Falcon::DictDecl(); (yyval.fal_ddecl)->pushBack( (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ); ;}
    break;

  case 453:
#line 2616 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyvsp[(1) - (5)].fal_ddecl)->pushBack( (yyvsp[(3) - (5)].fal_val), (yyvsp[(5) - (5)].fal_val) ); (yyval.fal_ddecl) = (yyvsp[(1) - (5)].fal_ddecl); ;}
    break;


/* Line 1267 of yacc.c.  */
#line 6452 "/home/gian/Progetti/falcon/core/engine/src_parser.cpp"
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


#line 2620 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
 /* c code */


void flc_src_error (const char *s)  /* Called by yyparse on error */
{
   /* do nothing: manage it in the action */
}

/* end of src_parser.yy */


